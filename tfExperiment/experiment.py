import os
import inspect
import tensorflow as tf

from dotmap import DotMap as dm 

from stepTimer import Timer
from dataSaver import DataSaver

from .cacheManager import CacheManager

# keys to be defined in child
# , epochs, saveAfter, testAfter, loud = True

def initDatasaver(rootPath, sessionType):
    dataSavePath = os.path.join(rootPath, 'data', sessionType )
    dataSaver = DataSaver(dataSavePath)
    return dataSaver

def withEnv(env, cb, *args):
    if 'env' in list(inspect.signature(cb).parameters.keys()):
        return cb(*args, env = env)
    else:
        return cb(*args)

def initEnvironment(rootPath):
    print('graph location ======================================>')
    print('tensorboard --logdir ', os.path.join(rootPath, 'graph'))
    print('<====================================== graph location ')
    env = dm()

    env.modelSavePath = os.path.join(rootPath, 'trainedModels')
    env.modelSaveDir = os.path.join(env.modelSavePath, 'model.ckpt')
    
    env.graphSavePath = os.path.join(rootPath, 'graph')

    # init all directories
    training = env.training
    training.dataSaver = initDatasaver(rootPath, 'training')

    testing = env.testing
    testing.dataSaver = initDatasaver(rootPath, 'testing')

    os.makedirs(env.modelSavePath, exist_ok = True)

    return env

class Experiment():
    def __init__(self):      
        # this should be overwritten
        if not hasattr(self, 'config'):
            self.config = tf.ConfigProto()
            self.config.gpu_options.allow_growth = True

        if not hasattr(self, 'loud'):
            self.loud = None

        if not hasattr(self, 'rootPath'):
            self.rootPath = os.getcwd()

        if not hasattr(self, 'sourcePath'):
            self.sourcePath = os.path.join(os.getcwd(), '..', '..')

        if not hasattr(self, 'toCacheFiles'):
            self.toCacheFiles = ['network']

        if not hasattr(self, 'epochs'):
            self.epochs = None

        if not hasattr(self, 'validateAfter'):
            self.validateAfter = None

        if not hasattr(self, 'saveAfter'):
            self.saveAfter = None

        if not hasattr(self, 'trainAfter'):
            self.trainAfter = None

        if not hasattr(self, 'net'):
            self.net = None

        if not hasattr(self, 'max_to_keep'):
            self.max_to_keep = None

        if not hasattr(self, 'keep_checkpoint_every_n_hours'):
            self.keep_checkpoint_every_n_hours = None

        self.env = initEnvironment(self.rootPath)
        self.saver = None
        
    def cache(self, files = None):
        checkpoint, _ = self.getCurrentCheckpoint()
        if files != None:
            cacheManager = CacheManager(self.rootPath, self.sourcePath, files)

            if not checkpoint:
                cacheManager.createCache()
            return cacheManager.moduleLoader

    def networkLoader(self):
        return self.cache(self.toCacheFiles)

    def build(self):
        print('===> Building Graph...')
        loader = self.networkLoader()
        self.net = self.buildNetwork(loader)
        self.saver = tf.train.Saver(
            max_to_keep = self.max_to_keep,
            keep_checkpoint_every_n_hours = self.keep_checkpoint_every_n_hours
        )
        print('===> Graph Built')


    def printFun(self, *args):
        if self.loud:
            print(*args)

    def saveGraph(self, session):
        with tf.Session() as session:
            tf.summary.FileWriter(self.env.graphSavePath).add_graph(session.graph)

    def getCurrentCheckpoint(self):
        # resetore session is present
        checkpoint = tf.train.latest_checkpoint(self.env.modelSaveDir)
        epoch = 0
        if checkpoint:
            epoch = checkpoint.split('-')[-1]
            epoch = int(epoch)
        
        return checkpoint, epoch

    def saveSession(self, session, epoch):
        self.saver.save(
            session,
            self.env.modelSavePath,
            global_step = epoch
        )
        self.printFun('===> Session Saved @ epoch nr {epoch} ')

    def setUpSession(self, session):
        tf.global_variables_initializer().run()

        checkpoint, epoch = self.getCurrentCheckpoint()
        
        if checkpoint != None:
            print('loading checkpoint :', checkpoint)
            self.saver.restore(session, checkpoint)

        return epoch, checkpoint

    def runEpoch(self, session, type):
        if not hasattr(self, type):
            return

        fun = getattr(self, type)

        timer = Timer()
        self.printFun(f'===> Starting {type}')
        withEnv(self.env, fun, session)
        self.printFun(f'===> Ended {type} after: {timer.elapsedTot()}')

    def run(self, epochs, saveAfter, validateAfter):
        self.epochs = epochs
        self.saveAfter = saveAfter
        self.validateAfter = validateAfter
        
        env = self.env
        timer = Timer()

        with tf.Session(config = self.config) as session:
            epoch, _ = self.setUpSession(session)
            lastEpoch = epoch

            for j in range(lastEpoch, lastEpoch + self.epochs):
                env.training.currentEpoch = epoch + 1

                self.runEpoch(session, 'train')

                if (((j - lastEpoch + 1)  % self.saveAfter) == 0):
                    self.saveSession(session, env.training.currentEpoch)

                if ((j - lastEpoch + 1)  % self.validateAfter) == 0:
                    self.runEpoch(session, 'validate')
            print('===> Training Completed')
            print('Tot Time Elapsed ', timer.elapsedTot() )
    

    def runTesting(self, type = 'testing'):
        env = self.env
        with tf.Session(config = self.config) as session:
            epoch, checkpoint = self.setUpSession(session)
            env.training.currentEpoch = epoch
            if not checkpoint:
                raise Exception(f'No model saved @{self.env.modelSavePath}')

            timer = Timer()
            self.runEpoch(session, 'testing')
            print(f'===> Testing Completed after {timer.elapsedTot()}')

        




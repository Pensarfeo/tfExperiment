import os
import inspect
import tensorflow as tf
import json

from box import Box 

from stepTimer import Timer
from dataSaver import DataSaver

from .cacheManager import CacheManager

from .errors import Errors

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

def shortenHyperparamsString(hyperparams):
    pathString = []
    for k in list(hyperparams.keys()):
        if len(k) <= 2:
            pathString.append(f'{k}--{hyperparams[k]}')
        else:
            uppercases = ''.join([ l for l in k if l.isupper() ])
            neBlock = f'{k[0]}{uppercases}--{hyperparams[k]}'
            pathString.append(neBlock)

    return '__'.join(pathString)

def initEnvironment(rootPath, hyperparams, repetition):
    hyperString = shortenHyperparamsString(hyperparams)
    repString = str(repetition)
    rootOutputPath = os.path.join(rootPath, 'output', hyperString)
    outputPath = os.path.join(rootOutputPath, repString)
    

    env = Box(hyperparams = hyperparams, repetition = repetition)

    env.hyperparams = hyperparams
    env.repetition = repetition

    env.dataSavePath = os.path.join(outputPath, 'data')
    env.modelSavePath = os.path.join(outputPath, 'trainedModels')
    env.modelSaveDir = os.path.join(env.modelSavePath, 'model.ckpt')
    
    env.graphSavePath = os.path.join(rootOutputPath, 'graph')


    hyperparamsFilePath = os.path.join(rootOutputPath, 'hyperparams.json')

    print('graph location ======================================>')
    print('tensorboard --logdir ', os.path.join(rootOutputPath, 'graph'))
    print('<====================================== graph location ')


    os.makedirs(env.modelSavePath, exist_ok = True)
    os.makedirs(env.dataSavePath, exist_ok = True)
    print(f'===> Output dir created @ {hyperString}/{outputPath}')

    if os.path.isfile(hyperparamsFilePath):
        previousHyperparams = json.load(open(hyperparamsFilePath, 'r'))

        prevHString = str(previousHyperparams)
        newHString = str(hyperparams.to_dict())

        if (prevHString != newHString):
            Errors.nonMatchingHyperparams(newHString, prevHString)
    else:
        with open(hyperparamsFilePath, 'w', encoding='utf-8') as f:
            json.dump(hyperparams.to_dict(), f, ensure_ascii=False, indent=4)
            print(f'===> HyperParams File created.')
    
    # init all directories
    env.training = Box(
        dataSaver = initDatasaver(outputPath, 'training'),
    )

    env.testing = Box(
        dataSaver = initDatasaver(outputPath, 'testing'),
    )
    return env

class Experiment():
    def __init__(self, hyperparams, repetition = 0):

        try:
            int(repetition)
        except:
            ErrorsrepetitionMustBeInt(repetition)

        # this should be overwritten
        if not hasattr(self, 'config'):
            self.config = tf.ConfigProto()
            self.config.gpu_options.allow_growth = True

        if not hasattr(self, 'loud'):
            self.loud = True

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
            self.max_to_keep = 10

        if not hasattr(self, 'keep_checkpoint_every_n_hours'):
            self.keep_checkpoint_every_n_hours = 1

        self.env = initEnvironment(self.rootPath, hyperparams, repetition)
        self.saver = None
        
    def cache(self, files = None, rebuildCache = True):
        if files != None:
            cacheManager = CacheManager(self.rootPath, self.sourcePath, files)
            inOwnBranch, expName, branchName = cacheManager.inExperimentsOwnBranch()

            checkpoint, _ = self.getCurrentCheckpoint()

            if (not checkpoint) and inOwnBranch and rebuildCache:
                cacheManager.createCache()
            elif (not inOwnBranch) and cacheManager.cached:
                wrongBranchMessage = 'ATTENTION Not in experiment\'s branch. Loading from cache.'
                print('===>', wrongBranchMessage)                
            elif (not inOwnBranch) and not(cacheManager.cached):
                Errors.noCacheAndNotInBranch()
            else:
                print('===> Loading network module from cache.')

            return cacheManager.moduleLoader

    def print(self, *args):
        if self.loud:
            print(*args)

    def networkLoader(self, rebuildCache):
        return self.cache(self.toCacheFiles, rebuildCache = rebuildCache)

    def build(self, rebuildCache = True):
        print('===> Building Graph...')
        loader = self.networkLoader(rebuildCache)
        self.net = self.buildNetwork(loader, self.env.hyperparams)
        self.saver = tf.train.Saver(
            max_to_keep = self.max_to_keep,
            keep_checkpoint_every_n_hours = self.keep_checkpoint_every_n_hours
        )
        print('===> Graph Built')

    def saveGraph(self, session = None):
        if not session:
            session = tf.Session()
        
        tf.summary.FileWriter(self.env.graphSavePath).add_graph(session.graph)

    def getCurrentCheckpoint(self):
        # resetore session is present
        checkpoint = tf.train.latest_checkpoint(self.env.modelSavePath)
        epoch = 0
        if checkpoint:
            epoch = checkpoint.split('-')[-1]
            epoch = int(epoch)
        
        return checkpoint, epoch

    def saveSession(self, session, epoch):
        self.saver.save(
            session,
            self.env.modelSaveDir,
            global_step = epoch
        )
        self.print(f'===> Session Saved @ epoch nr {epoch} ')

    def setUpSession(self, session):
        tf.global_variables_initializer().run()

        checkpoint, epoch = self.getCurrentCheckpoint()
        if checkpoint != None:
            self.print('loading checkpoint :', checkpoint)
            self.saver.restore(session, checkpoint)

        return epoch, checkpoint

    def runTask(self, session, type):
        if not hasattr(self, type):
            return

        fun = getattr(self, type)
        timer = Timer()
        self.print(f'===> Starting {type}')
        runOutput = withEnv(self.env, fun, session)
        self.print(f'===> Ended {type} after: {timer.elapsedTot()}')
        return runOutput

    def stopRunLoop(self, *args):
        return False

    def run(self, epochs, saveAfter, validateAfter):
        self.epochs = epochs
        self.saveAfter = saveAfter
        self.validateAfter = validateAfter

        env = self.env
        timer = Timer()

        with tf.Session(config = self.config) as session:
            env.training.currentEpoch, _ = self.setUpSession(session)
            for j in range(0, self.epochs):
                env.training.currentEpoch += 1

                print('EPOCH ===>', env.training.currentEpoch)
                trainingOutput = self.runTask(session, 'train')
                
                if (((j + 1)  % self.saveAfter) == 0):
                    self.saveSession(session, env.training.currentEpoch)

                if ((j + 1)  % self.validateAfter) == 0:
                    validationOutput = self.runTask(session, 'validate')
                
                stopLoop = self.stopRunLoop(trainingOutput, validationOutput)

                if stopLoop == True:
                    print('breaking loop')
                    break

            print('===> Training Completed')
            print('Tot Time Elapsed ', timer.elapsedTot() )
    

    def runTesting(self, type = 'testing'):
        env = self.env
        with tf.Session(config = self.config) as session:
            epoch, checkpoint = self.setUpSession(session)
            env.training.currentEpoch = epoch
            if not checkpoint:
                Errors.noCheckpointFoundAt(self.env.modelSavePath)

            timer = Timer()
            self.runTask(session, 'testing')
            print(f'===> Testing Completed after {timer.elapsedTot()}')
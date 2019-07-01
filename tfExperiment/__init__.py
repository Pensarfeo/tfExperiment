import os
import tensorflow as tf
import inspect
from dotmap import DotMap as dm

from dataSaver import DataSaver
from stepTimer import Timer
from pygit2 import Repository

class DisposableSession():
    def __init__(self, epochs, saveAfter, testAfter, experiment = None):
        self.experiment = experiment
        self.epochs = epochs
        self.saveAfter = saveAfter
        self._saveGraph = False
        self.testAfter = testAfter
        self.trainCallback = None
        self.testCallback = None

    def saveGraph(self):
        self._saveGraph = True

    def __enter__(self):
        return self

    def __exit__(self, typeE, value, traceback):
        self.experiment.train(
            trainCallback = self.trainCallback,
            epochs = self.epochs,
            saveModelAfter = self.saveAfter,
            saveGraph = self._saveGraph,
            testCallback = self.testCallback,
            testAfter = self.testAfter
        )




class Experiment():
    def __init__(self, name = None, finalizeGraph = False):
        self.runName = Repository('.').head.shorthand if name == None else name
        self.modelSaveDir = os.path.join(os.getcwd(), 'output', self.runName, 'trainedModels')
        self.modelSavePath = os.path.join(self.modelSaveDir, 'model.ckpt')

        self.graphSavePath = os.path.join('output', self.runName, 'graph')
        
        print('graph location ======================================>')
        print('tensorboard --logdir ', self.graphSavePath)
        print('<====================================== graph location ')

        self.env = dm()
        self.env.training.currentEpoch = 0
        # setSaver
        self.finalizeGraph = finalizeGraph

        if not tf.get_default_graph().finalized:
            self.saver = tf.train.Saver()

        self.initDatasaver('training')
        self.initDatasaver('testing')

        self.config = None

    def initDatasaver(self, type):
        env = self.env[type]
        env.dataSavePath = os.path.join('output', self.runName, 'data', type )
        
        env.dataSaver = DataSaver(env.dataSavePath) 
   
    '''
        Run training function
        iterable: function that should run the network
        epochs: total number of epochs to run
        saveModelAfter: save model after N elapsed epochs
    '''
    def withEnv(self, env, cb, *args):
        if 'env' in list(inspect.signature(cb).parameters.keys()):
            return cb(*args, env = env)
        else:
            return cb(*args)

    def saveGraph(self):
        with tf.Session() as session:
            tf.summary.FileWriter(self.graphSavePath).add_graph(session.graph)

    def setUpTrainingSession(self, session, saveGraph):
        if saveGraph:
            tf.summary.FileWriter(self.graphSavePath).add_graph(session.graph)

        if not tf.get_default_graph().finalized:
            tf.global_variables_initializer().run()
        
        if self.finalizeGraph:
            session.graph.finalize()
            self.graphFinalized = True

        # resetore session is present
        checkpoint = tf.train.latest_checkpoint(self.modelSaveDir)
        
        if checkpoint != None:
            print('loading checkpoint :', checkpoint)
            self.saver.restore(session, checkpoint)

        # made directory to save model to
        if (not os.path.exists(self.modelSaveDir)):
            os.makedirs(self.modelSaveDir)

    def trainingSession(self, epochs, saveAfter, testAfter = 0):        
        disposableSession = DisposableSession(epochs, saveAfter, testAfter, self)
        return disposableSession

    def train(
        self,
        trainCallback,
        epochs = 1,
        saveModelAfter = 2,
        saveGraph = False,
        testCallback = None,
        testAfter = 0
            ):
        
        env = self.env
        
        # Allow memory to grow so not all memory is assigned at once
        # This is a real issue with multy gpu which can make everything crash
        if self.config:
            config = self.config
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

        with tf.Session(config = config) as session:
            # save gaph
            self.setUpTrainingSession(session, saveGraph)
            
            timer = Timer()
            lastEpoch = env.training.currentEpoch

            for j in range(lastEpoch, lastEpoch + epochs):
                env.training.currentEpoch = env.training.currentEpoch + 1
                self.withEnv(env, trainCallback, session)

                if (((j - lastEpoch + 1)  % saveModelAfter) == 0):
                    print('Tot Time Elapsed: ', timer.elapsedTot(), ' after ', j + 1, 'epochs elapsed')

                    print('------------------ Saving Session ------------------')
                    self.saver.save(
                        session,
                        self.modelSavePath,
                        global_step = self.env.training.currentEpoch
                    )

                if (testCallback!= None) and (((j - lastEpoch + 1)  % testAfter) == 0):
                    print('Tot Time Elapsed: ', timer.elapsedTot(), ' after ', j + 1, 'epochs elapsed')

                    print('------------------ Testing ------------------')
                    self.withEnv(env, testCallback, session)

            print('------------------ Training Completed ------------------')
            print('Tot Time Elapsed ', timer.elapsedTot() )

        return env
    
    def test(self, iterable):
        env = self.env

        with tf.Session() as session:
            # save gaph
            if not tf.get_default_graph().finalized:
                tf.global_variables_initializer().run()

            # resetore session is present
            checkpoint = tf.train.latest_checkpoint(self.modelSaveDir)
            if checkpoint != None:
                self.saver.restore(session, checkpoint)
            else:
                raise Exception(f'No model saved @{self.modelSavePath}')

            timer = Timer()
            
            # turn this into a while loops that is ended when iterable returns false?
            self.withEnv(env, iterable, session)       

            print('------------------ Training Completed ------------------')
            print('Tot Time Elapsed ', timer.elapsedTot() )
        return env
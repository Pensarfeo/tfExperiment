# TfExperiment

A simple library to manage Tensorflow experiments though git and reduce boilerplate.
Compatible with tf 1.x

# Usage

This library relies of git to manage experiments. Each experiment should be a unique git branch, and the name of the experiment, if not give, will be the current git branch.

```python
experiment = tfExperiment.Experiment(finalizeGraph = False)
experiment.saveGraph()

# output
# > graph location ======================================>
# > tensorboard --logdir  output\experimentName\graph

# with with
with experiment.trainingSession(epochs = 125, saveAfter = 2, testAfter = 2) as ts:
    ts.saveGraph() # function to save the graph
    ts.trainCallback = runTrainingCallback
    ts.testCallback = runTestCallback

# as function
experiment.train(runTrainingCallback)
experiment.test(runTestCallback)
```

## API

### ```__init__(name = None, finalizeGraph = False, location = os.path.join(os.getcwd(), 'output'))```

* ```name: string```: Name of the experiment, if no name is provided the name of the current git branch will be used.

* ```finalizeGraph: bool```: Finalizes the graph. *Attention* I have not tried this feature much.

* ```location: string```: absolute path where the experiment results where saved in a folder with same name as name


### ```train(trainCallback, epochs = 1, saveModelAfter = 2, saveGraph = False, testCallback = None, testAfter = 0)```

Runs the training and validates/test the model

* ```trainCallback: function```: Function to be run at each epoch. This should contain your loop with the training actions to execute for each batch. The training callback can take 2 parameters: session (current tf.session), and env (if env is used you should use the exact name) experiment environment with access to functionalities like timer and dataSaver. 

* ```epochs: integer```: Number of epochs to run, that is to say the number of times the traininCallbacks will be called. *Attention:* the experiment object keeps track of the number of epochs run so far, so if you call ```experiment.train``` again, the epoch number will continue to grow from the last epoch number.

* ```saveModelAfter: integer```: Save the model after n epochs have run. This only considers the current run.

* ```saveGraph: bool```: If we should save the graph at the current run.

* ```testCallback: function```: Function to call to test/validate the current network. Similar to trainCallback.

* ```testAfter: integer```: test the model after n epochs have run. This only considers the current run.

### ```test(testCallback)```

Runs the testing/validation of the model once

* ```testCallback: function```: Function to call to test/validate the current network. Similar to trainCallback.

### ```env: Box object```

The env object contains

* ```env.training.currentEpoch: integer```: number of epochs since the instance was initialized.
* ```env.training.currentEpoch: integer```: number of epochs since the instance was initialized.
* ```env.training.dataSavePath: path string```: path in which the data will be used if dataSaver is used during training.
* ```env.training.dataSaver: dataSaver Instance```: dataSaver instance (initialized with env.training.dataSavePath) for training to the training file.
* ```env.testing.dataSavePath: path string``` path in which the data will be used if dataSaver is used during testing.
* ```env.testing.dataSaver: dataSaver Instance```: dataSaver instance (initialized with env.testing.dataSavePath) for testing to the training file.

### Proposed New API

```python
def TrainExperiment(Experiment):
    def __init__(self, constructor, ...):
        #someconfig
        #self.nrTotEpochs
        #self.epochsToValidateAfter
        #...
    
    def beforeEpoch
    def afterEpoch
    
    def beforeSave
    def afterSave
    
    def beforeTest
    def afterTest

    def beforeIteration
    def afterIteration
    
    def train(session, data, dataProvider = None):
        return 0 #trainingLoopPerSession
    
    def validate(session, data, dataProvider = None):
        return 0 #trainingLoopPerSession
    

experiment(TrainExperiment)
```
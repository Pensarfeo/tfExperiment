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



import os

class WithinPath():
    def __init__(self, toDir, fromDir = None):
        self.fromDir = fromDir if fromDir != None else os.getcwd()
        self.toDir = toDir

    def __enter__(self):
        os.chdir(self.toDir)
        return self

    def __exit__(self, typeE, value, traceback):
        os.chdir(self.fromDir)

withinPath = WithinPath

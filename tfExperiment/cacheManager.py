import importlib
import os
import shutil
import traceback
import logging

from .withinPath import withinPath

from pygit2 import Repository
import json
import sys


def copyFiles(files, destinationDir):
    for file in files:
        if os.path.isdir(file):  
            dest = os.path.join(destinationDir, file)
            os.makedirs(destinationDir, exist_ok = True)
            shutil.copytree(file, dest)
        else:
            finalDir = os.path.join(destinationDir, os.path.dirname(file))
            dest = os.path.join(destinationDir, file)
            os.makedirs(finalDir, exist_ok = True)
            shutil.copy2(file, dest)

defaultRootPath = os.getcwd()

class CacheManager():
    def __init__(self, rootPath, sourcePath, toCacheFiles):
        self.rootPath = defaultRootPath
        self.sourcePath = sourcePath

        self.cachePath = os.path.join(self.rootPath, 'cache')
        self.cached = os.path.isdir(self.cachePath)
        self.files = toCacheFiles

    def inExperimentsOwnBranch(self):
        data = json.load(open('info.json', 'r'))
        experimentName = data['name']
        branchName = Repository(self.sourcePath).head.shorthand

        return branchName == experimentName, experimentName, branchName
    # for now on we are going to be working from the experiment forder in the outputs... may be it needs a new name.
    # Every experiment need to be fully contained within the cache folder and everythin in the network is loaded relative to cache folder as root so no chance should be needed from the original script
    # 
    def moduleLoader(self, reload = False):
        with withinPath(toDir = self.cachePath, fromDir = self.rootPath):
            originalPath = sys.path
            
            if '.' not in sys.path:
                sys.path = ['.'].extend(originalPath)

            module = importlib.import_module('network')
            if reload:
                module = importlib.reload(module)

            sys.path = originalPath
        return module

    def destroyCache(self):
        try:
            shutil.rmtree('cache')
            self.cached = False
        except Exception as e:
            logging.error(traceback.format_exc())

    def fillCache(self):
        with withinPath(toDir = self.sourcePath, fromDir =  self.rootPath):
            copyFiles(self.files, self.cachePath)
            self.cached = True

    def createCache(self):
        if self.cached == True:
            print('===> Deleting cache...')
            self.destroyCache()
            self.cached = False
        elif self.cached == None:
            raise ValueError('cache state undefined')
        print('===> Building cache...')
        self.fillCache()
        print('===> New cache created...')
import os
import time
import json

from pygit2 import Repository

from .cacheManager import copyFiles
from .withinPath import withinPath

# experiment data
# data creation

def initializer(*toCopyFiles):
    expName = Repository('.').head.shorthand
    date = time.strftime("%Y-%m-%d-%H:%M")
    expPath = os.path.join('experiments', expName)
    
    if os.path.isdir(expPath):
        raise Exception(f'{expPath} expriment already exists, please delete the expeirment to continue')

    # create info
    print("Enter Experiment description:")
    description = input()

    info = dict(
        name = expName,
        data = date,
        description = description 
    )

    try:
        os.makedirs(expPath)
    except:
        raise Exception(f'{expPath} expriment already exists, please delete the expeirment to continue')


    with withinPath(toDir = expPath, fromDir = os.getcwd()):
        with open('info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)

    # copy files
    endPath = os.path.join(os.getcwd(), 'experiments', expName)
    copyFiles(toCopyFiles, endPath)


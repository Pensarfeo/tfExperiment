
class Errors():

    @staticmethod
    def nonMatchingHyperparams(newHp, oldHp):
        raise Exception(''.join([
            f'===> ERROR: Given hyperparams do not match previous hyperparams for same experiment file.\n',
            f'                New hyperparams: {newHp}\n'
            f'                Old hyperparams: {oldHp}\n'
            f'                Please follow up proper rules for hyperparams naming and ordering!'
        ]))

    @staticmethod
    def repetitionMustBeInt(repetition):
        raise Exception(f'@@@> ERROR: repetition must be and int, we got {repetition}')

    @staticmethod
    def noCacheAndNotInBranch():
        raise Exception('ERROR: Cache not found and not in eperiment\'s branch')
    
    @staticmethod
    def noCheckpointFoundAt(path):
        raise Exception(f'No model saved @{path}')
    
    @staticmethod
    def experimentDirExists(expPath):
        raise Exception(f'{expPath} expriment already exists, please delete the expeirment to continue')
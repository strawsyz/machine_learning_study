import pickle

def storeTree(inputTree, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(inputTree, fp)

def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)
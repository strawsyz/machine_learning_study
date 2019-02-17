import trees
import treePlotter
import rwtrees

myDat, labels = trees.createDataSet()
# print(myDat)
# shannonEnt = trees.calcShannonEnt(myDat)
# print(shannonEnt)
#
# print(trees.splitDataSet(myDat, 0, 1))
# print(trees.splitDataSet(myDat, 0, 0))
#
# print(trees.chooseBestFeatureToSplit(myDat))
#
myTree = trees.createTree(myDat, labels)
# print(myTree)
#
rwtrees.storeTree(myTree, 'classifierStorage.txt')
temp = rwtrees.grabTree('classifierStorage.txt')
print(temp)
print(1)

# fr = open('lenses.txt')
# lense = [inst.strip().split('\t') for inst in fr.readlines()]
# lenseLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
# lensesTree = trees.createTree(lense, lenseLabels)
# print(lensesTree)
# treePlotter.createPlot(lensesTree)
















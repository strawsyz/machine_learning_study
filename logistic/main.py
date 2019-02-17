import logRegres
import plot
from numpy import *
dataMat, labelMat = logRegres.loadDataSet()
# weight = logRegres.gradAscent(dataMat, labelMat)
# print(weight)
# print(type(weight))

# plot.plotBestFit(weight.getA(),dataMat,labelMat)

# weight = logRegres.stocGradAscent0(array(dataMat), labelMat)
# print(weight)
# print(type(weight))
#
# plot.plotBestFit(weight,dataMat,labelMat)

# weights = logRegres.stocGradAscent1(array(dataMat), labelMat)
# plot.plotBestFit(weights,dataMat,labelMat)

logRegres.muliTest()
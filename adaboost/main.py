import adaboost
import boost
from numpy import *
dataMat, dataLabels = adaboost.loadSimpData()
print(dataMat)
print(dataLabels)

D = mat(ones((5, 1)) / 5)
boost.buildStump(dataMat, dataLabels, D)
from numpy import *

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    单层决策树
    是一个弱分类算法
    通过与阈值threshVal比较分为+1和-1'''
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, dataLables, D):
    dataMatrix = mat(dataArr); labelMat = mat(dataLables).T
    m, n = shape(dataMatrix)
    numSteps = 10.0  # 用于在特征所有可能的值上遍历
    bestStump = {}  # 用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    bestClassEst = mat(zeros((m, 1)))
    minError = inf  # 将最小错误率设为正无穷，用于寻找可能的最小错误率
    for i in range(n):  # 循环每个特征
        # 第i个特征的最小值
        rangeMin = dataMatrix[:, i].min()
        # 第i个特征的最大值
        rangeMax = dataMatrix[:, i].max()
        # 计算步长
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):  # 循环每个步长，可以把阈值设置在取值范围之外
            for inequal in ['lt', 'gt']:  # 循环 大于和小于
                threshVal = rangeMin + float(j) * stepSize  # 设置阈值
                # 预测结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr  # 计算加权错误率，D是权重向量
                print("split : dim %d, thresh %.2f, thresh inequal: %s, the weighted "
                      "error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
        return bestStump, minError, bestClassEst




from numpy import *
def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    '''i是alpha的下标，m是所有alpha的数目'''
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(alpha, H, L):
    '''调整使alpha的值不大于H，不小于L'''
    if alpha > H:
        alpha = H
    if L > alpha:
        alpha = L
    return alpha


def smoSimple(dataMat, dataLabels, C, toler, maxIter):
    dataMatrix = mat(dataMat)
    dataLabels = mat(dataLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    # 初始化alpha
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, dataLabels).T *
                        (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler and (alphas[i] < C)) or
            )

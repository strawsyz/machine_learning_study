from numpy import *
def loadDataSet():
    '''读取数据'''
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    '''sigmoid函数'''
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    '''梯度上升算法'''
    # 为了进行矩阵的运算，都转成numpy的mat类型
    dataMat = mat(dataMatIn)  # 100*3
    labelMat = mat(classLabels).transpose()  # 100*1
    m, n = shape(dataMat)  # 100， 3
    alpha = 0.001  # 步长
    maxCycles = 500  # 循环次数
    weights = ones((n, 1))  # 3*1
    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)  # 100*1
        error = (labelMat - h)  # 100*1
        weights = weights + alpha * dataMat.transpose() * error  # 3*1
    return weights  # numpy.matrixlib.defmatrix.matrix

def stocGradAscent0(dataMatrix, classLabels):
    '''随机梯度上升算法'''
    m, n = shape(dataMatrix)  # 100,3
    alpha = 0.01
    weights = ones(n)  # 1*3
    for i in range(m):  # 有多少条数据就循环多少次
        h = sigmoid(sum(dataMatrix[i]*weights))  # 一个数值
        error = classLabels[i] - h  # 一个数值
        weights = weights + alpha * error * dataMatrix[i]  # 1*3
    return weights  # numpy.ndarray

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    '''改进版 随机梯度上升算法'''
    m, n = shape(dataMatrix)  # 100*3
    weights = ones(n)  # 1*3
    for j in range(numIter):
        # 为了能使用del()，要转成list
        dataIndex = list(range(m))
        for i in range(m):
            # alpha每次迭代都会调整，数据的波动
            # alpha会不断减小但是不会减小到0
            # 如果处理的问题是动态变化的，可以适当加大上述常数项，
            # 来确保新的值获得更大的回归系数
            # 当j<<max(i)时，alpha就不是严格下降
            alpha = 4 / (1.0+j+i) + 0.01
            # 设置随机数，减小周期性波动
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))  # 一个数值
            error = classLabels[randIndex] - h  # 一个数值
            weights = weights + alpha * error * dataMatrix[randIndex]  # 1*3
            del(dataIndex[randIndex])
    return weights  # numpy.ndarray类型


def classifyVector(inX, weights):
    '''根据回归系数和特征向量判断属于哪个类'''
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
def colicTest():
    # 加载数据
    frTest = open('horseColicTest.txt')
    frTrain = open('horseColicTraining.txt')
    #准备训练数据
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 进行500次迭代
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != \
                int(currLine[21]):
            errorCount += 1
    errorRate = float(errorCount)/numTestVec
    print('the error rate of this test is : %f'%errorRate)
    return errorRate

def muliTest():
    # 运行10次
    numTests = 10; errorSum = 0.0
    # 计算总错误率
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is : %f"% (numTests, errorSum/float(numTests)))


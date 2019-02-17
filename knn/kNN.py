from numpy import *
import operator
def creatDataSet():
    group = array([[1.0,1.1], [1.0, 1.0], [0, 0]], [0, 0.1])
    labels = ['A', 'A', 'B', 'B']
    return group,labels


def classify0(inX, dataSet, labels, k):
    '''
    使用欧式距离，k-邻近算法
    :param inX: 用于分类的向量inX
    :param dataSet:  输入的训练样本集为dataSet
    :param labels:   标签向量为labels
    :param k:  k是选择的邻居的数目
    :return:
    '''
    dataSetSize = dataSet.shape[0]  # 总共的数据条数
    # 计算欧式距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    # argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)
    # x[3]=-1最小，所以y[0]=3,x[5]=9最大，所以y[5]=5。
    sortedDistIndicies = distances.argsort()
    # 计算每个类别对应的个数
    classCount={}
    for i in range(k):
        votelabel = labels[sortedDistIndicies[i]]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                             key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    # 得到文件行数
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    # index = 0
    for index, line in enumerate(arrayOfLines):
        line = line.strip()  # 去掉前后的空白字符
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        # index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    # normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    # hoRatio是测试数据所占的比例
    hoRatio = 0.10
    # 加载数据
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    # 归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 获得数据的数量
    m = normMat.shape[0]
    # 测试数据的数量
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d,the real answer is: %d"
              %(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print(errorCount)
    print('the total error rate is: %f' % (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    ffMiles = float(input("frequent fliter miles earned per year?"))
    percentTats = float(input("percentage of time spent playing video games?"))
    iceCream = float(input("liters of ice cream consumed per year？"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ",
          resultList[classifierResult - 1])
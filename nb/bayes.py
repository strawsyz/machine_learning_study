from numpy import *


def loadDataSet():
    '''
    用来加载数据
    postingList: 进行词条切分后的文档集合
    classVec:类别标签
    使用伯努利模型的朴素贝叶斯分类器不考虑单词的出现次数，
    只考虑单词出现与否（0，1）
    '''
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec

def createVocabList(dataSet):
    '''返回文档中所有文字组成的集合
    dataSet是个二维数组
    '''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    '''将输入的单词集转化成向量'
    使用词集模型'''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return returnVec

def bagOfWords2Vec(vocabList, inputSet):
    '''将输入的单词集转化成向量'
    使用词袋模型'''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return returnVec

# def trainNBO(trainMatrix, trainCategory):
#     # 获得文档的句子数量
#     numTrainDocs = len(trainMatrix)
#     # 获得每个句子向量的长度
#     numWords = len(trainMatrix[0])
#     # 计算p（1）,及类型为1的概率    abusive：侮辱的
#     pAbusive = sum(trainCategory) / float(numTrainDocs)
#     # 初始化分子
#     p0Num = zeros(numWords)
#     p1Num = zeros(numWords)
#     # 初始化分母
#     p0Denom = 0.0
#     p1Denom = 0.0
#     # 循环每个句子
#     for i in range(numTrainDocs):
#         if trainCategory[i] == 1:  # 如果该句子向量的类型是1
#             p1Num += trainMatrix[i]  # 增加词的出现次数
#             p1Denom += sum(trainMatrix[i])  # 出现的所有的词的个数
#         else:
#             p0Num += trainMatrix[i]
#             p0Denom += sum(trainMatrix[i])
#     p0Vect = p0Num / p0Denom  # 计算p（wi|c0）
#     p1Vect = p1Num / p1Denom  # 计算p（wi|c1）
#     return p0Vect, p1Vect, pAbusive


def trainNBO(trainMatrix, trainCategory):
    '''
    训练朴素贝叶斯的改进版
    为了防止其中出现概率值为0的情况，
    把词的出现次数初始化为1，
    将分母初始化为2
    因为概率的值很小，相乘最后可能会变成0
    又因为In(a*b)=In(a)+In(b)，
    所以p1Num/p1Denom 改成 log(p1Num/p1Denom)
    '''
    # 获得文档的句子数量
    numTrainDocs = len(trainMatrix)
    # 获得每个句子向量的长度
    numWords = len(trainMatrix[0])
    # 计算p（1）,及类型为1的概率    abusive：侮辱的
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # 初始化分子
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    # 初始化分母
    p0Denom = 2.0
    p1Denom = 2.0
    # 循环每个句子
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 如果该句子向量的类型是1
            p1Num += trainMatrix[i]  # 向量相加，把句子中出现的词对应的个数加1
            p1Denom += sum(trainMatrix[i])  # 总词数增加
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 避免下溢出
    p0Vect = log(p0Num / p0Denom)  # 计算p（wi|c0）
    p1Vect = log(p1Num / p1Denom)  # 计算p（wi|c1）
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    使用训练好的数据根据朴素贝叶斯算法来进行分类
    vec2Classify 是待分类的词向量
    '''
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # 因为使用了log函数，相加相当于内部数值相乘
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNBO(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    # 转成词向量
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as :', classifyNB(thisDoc, p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    # 转成词向量
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as :', classifyNB(thisDoc, p0V, p1V, pAb))

# testingNB()
# 运行结果：
# ['love', 'my', 'dalmation'] classified as : 0
# ['stupid', 'garbage'] classified as : 1

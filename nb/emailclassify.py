import bayes
from numpy import *
def textParse(Str, minLength = 3):
    '''以非单词数字下划线组成的作为分隔符切分Str
    去掉长度小于3的单词，所有单词都小写'''
    import re
    listOfTokens = re.split(r'\W*', Str)
    return [tok.lower() for tok in listOfTokens if len(tok) > minLength-1]

def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = bayes.createVocabList(docList)
    # range对象不支持del，所以要转成list
    trainingSet = list(range(50)); testSet = []
    # 随机取10组数据作为测试机
    for i in range(10):  # 循环10次
        # 在0到49之间（包括0,49）取随机整数
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  # 删除数组的引用
    # 把剩下的40组数据作为训练集
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:  # 由于上面删除了10个引用，还剩40个
        trainMat.append(bayes.setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = bayes.trainNBO(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        # 转成词向量
        wordVector = bayes.setOfWords2Vec(vocabList, docList[docIndex])
        if bayes.classifyNB(wordVector, p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount)/len(testSet))

spamTest()
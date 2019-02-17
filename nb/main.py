import bayes


listOPosts, listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
print(myVocabList)
# 构建词向量
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
print(trainMat)

p0V, p1V, pAb = bayes.trainNBO(trainMat, listClasses)
print(p0V)
print(p1V)
print(pAb)
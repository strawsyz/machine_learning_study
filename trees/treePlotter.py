import matplotlib.pyplot as plt

decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''画带箭头的线'''
    createPlot.axl.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.axl = plt.subplot(111, frameon=False)
    plotNode('(a decision node)', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('(a leaf node)', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

# createPlot()

def getNumLeafs(tree):
    '''获得树的叶子数量'''
    numLeafs = 0
    # python3要使用list转换
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs

def getTreeDepth(tree):
    '''获得树的深度'''
    maxDepth = 0
    # python3要使用list转换
    firstStr = list(tree.keys())[0]
    secondDict = tree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth=thisDepth
    return maxDepth

def retriveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTrees[i]

tree = retriveTree(1)
print(getNumLeafs(tree))
print(getTreeDepth(tree))

def plotMidText(cntrPt, parentPt, txtString):
    '''在连接线上添加文本信息'''
    xMid = (parentPt[0]-cntrPt[0])/2.0 +cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 +cntrPt[1]
    createPlot.axl.text(xMid, yMid, txtString)

def plotTree(tree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(tree)
    depth = getTreeDepth(tree)
    firstStr = list(tree.keys())[0]
    # 计算子节点坐标
    cntrPt = (plotTree.xOff + (1.0+float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    # 绘制线上的文字
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 绘制节点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = tree[firstStr]
    # 将yOff减少1.0/plottree.totald，为画下一层做准备
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(tree):
    # 设置背景为白色
    fig = plt.figure(1, facecolor='white')
    # 清空画布
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # 111表示1行1列，第1个子图
    createPlot.axl = plt.subplot(111, frameon=True, **axprops)
    plotTree.totalW = float(getNumLeafs(tree))
    plotTree.totalD = float(getTreeDepth(tree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    # 在 1 X 1 的图纸上画，0.5,1.0是最高的中间的点
    plotTree(tree, (0.5, 1.0), '')
    plt.show()



if __name__ == '__main__':
    myTree = retriveTree(0)
    myTree['no surfacing'][3] = 'maybe'
    # 绘制图片
    # createPlot(myTree)

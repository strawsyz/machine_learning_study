import kNN
from numpy import *
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet.txt')

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# 要设置字体，不然xlabel和ylabel无法显示中文
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)
fig = plt.figure()
ax = fig.add_subplot(111)
# 使用第二、第三列数据。分别表示玩视频所耗的时间百分比
# 每周所消费的冰淇淋公升数
# 第三个参数设置点的大小，第四个参数设置点的颜色
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2],
           15.0*array(datingLabels), 15.0*array(datingLabels))
ax.set_xlabel('玩视频所耗的时间百分比',  fontproperties=font_set)
ax.set_ylabel('每周所消费的冰淇淋公升数',  fontproperties=font_set)
# plt.show()

# normDataSet, ranges, minVals = kNN.autoNorm(datingDataMat)

# kNN.datingClassTest()

kNN.classifyPerson()
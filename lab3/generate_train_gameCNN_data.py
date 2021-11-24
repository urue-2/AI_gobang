
'''

说明：
此处用的数据来自 ： https://github.com/TommyGong08/Gobang_AI 中的项目,特此表示感谢！
共有 1840 个数据 ， 当前棋盘状况，下一步应该走的位置 (1840, 15, 15) (1840, 2) (1840, 1)

'''
# 以下为ANN 准备特征向量

import numpy as np
from lab2.next_move import *
features = np.empty((1,241))
labels = np.empty((1,15,15))
boardx = np.load("ANN_training_data//x_train.npy")
poss = np.load("ANN_training_data//y_train.npy")
roles = np.load("ANN_training_data/r_train.npy")

for i in range(1840):
    cf1 = []
    r1 = find_all_lines(boardx[i], 2)
    for a in r1.values():
        cf1.append(a)
    cf1 = np.asarray(cf1)

    cf2 = []
    r2 = find_all_lines(boardx[i], 1)
    for b in r2.values():
        cf2.append(b)
    cf2 = np.asarray(cf2)


    cf = np.concatenate((cf1, cf2))
    role = roles[i]
    tfearure = np.concatenate((cf,role))
    turns = 0
    for m in range(15):
        for n in range(15):
            if boardx[i][m][n] != 0:
                turns+=1

    turns = np.array(turns).reshape(1)
    tfearure = np.concatenate((tfearure,turns))

    tfearure = np.concatenate((tfearure,boardx[i].reshape(15 * 15)))

    pos = np.zeros((15,15))
    pos[poss[i][0]][poss[i][1]] = roles[i]

    if i == 0:
        features = tfearure.reshape(1,241)
        labels = pos.reshape((1,15,15))
    else:

        features = np.concatenate((features,tfearure.reshape((1,241))))
        labels = np.concatenate((labels,pos.reshape((1,15,15))))
        print(features.shape,labels.shape)

fx = "ANN_training_data//features"
fy = "ANN_training_data//labels"
print(features.shape,labels.shape)
np.save(fx,features)
np.save(fy,labels)

# (1840, 241) (1840, 15, 15)


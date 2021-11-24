from GUI import *
import random as rd
import numpy as np
from PIL import Image

import time
#  无棋子 0 人1 白   ai 2 黑

#putpiece(self,x,y,role):
dataamount = 500
labels = np.empty((1,15,15))
for j in range(dataamount):
    print(j)
    a = Chess()
    for i in range(20):
        tbx= rd.randint(0,14)
        tby= rd.randint(0, 14)
        a.matrix[tbx,tby]= 2
        a.putpiece(tbx,tby,-1)
    for i in range(20):
        twx= rd.randint(0,14)
        twy= rd.randint(0, 14)
        if a.matrix[twx,twy]==0:
            a.matrix[twx,twy]= 1
            a.putpiece(twx,twy,1)

    a.canvas.postscript(file="data/"+str(j)+".ps",width = 480,height = 480 , colormode='color')

    if j == 0:
        labels = a.matrix.reshape((1,15,15))
    else:
        labels = np.concatenate((labels,a.matrix.reshape((1,15,15))))

    im = Image.open("data/"+str(j)+".ps")
    im.save("data/"+str(j)+".jpg")

    a.root.destroy()
    a.root.mainloop()


fp = 'data/label.txt'
np.save(fp,labels)
print(labels.shape)
print(j)

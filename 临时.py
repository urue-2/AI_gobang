import numpy as np
x = np.load("lab3/ANN_training_data/features.npy")
y = np.load("lab3/ANN_training_data/labels.npy")
r= np.load("lab3/ANN_training_data/r_train.npy")
print(x[0],y[0],r[0])
print(x.shape,y.shape,r.shape)
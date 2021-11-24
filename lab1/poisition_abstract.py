#用训练好的模型预测
import torch
from lab1 import poistion_cnnmodel as my_cnn_model
from PIL import Image
from torchvision import transforms
import numpy as np


#加载已训练好的模型
model = my_cnn_model.trymodel()
optimizer = torch.optim.Adam(model.parameters())

checkpoint = torch.load("lab1/model.pth")
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epochs = checkpoint['epoch']

image_transforms = transforms.Compose([
        transforms.Grayscale(1)
        ])

imagepath = "lab1/data/1.jpg"
test_one_data = Image.open(imagepath)
test_one_data = image_transforms(test_one_data)
test_one_data = np.array(test_one_data).reshape(362,362)
test_one_data = torch.FloatTensor(test_one_data)

re = model(test_one_data)
print(re.argmax(dim = 1))



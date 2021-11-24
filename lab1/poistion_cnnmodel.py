import torch
from  torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
x=[]
y=[]


# prepare image data
class boarddataset(data.Dataset):
    def __init__(self):
        self.labels = np.load("data/label.txt.npy")
        self.fpath='data/'
        self.bpath='.jpg'
        self.len =200
        self.image_transforms = transforms.Compose([
        transforms.Grayscale(1)
        ])

    def __getitem__(self, index):
        label = torch.FloatTensor(self.labels[index])
        # 白 黑 无
        #print('label',label.shape)
        #label = label.softmax(dim=1)

        image = Image.open(self.fpath+str(index)+self.bpath)
        image = self.image_transforms(image)
        image = np.array(image).reshape(362,362)
        image = torch.FloatTensor(image)
        return  image , label

    def __len__(self):
        return  self.len

    def labelprocess(self,label):
        t = torch.zeros(15,15,3)
        for i in range(15):
            for j in range(15):
                if label[i,j]== 1: #白色层
                    t[i,j,0] = torch.tensor(1)
                elif label[i,j]==-1: #黑色层
                    t[i, j, 1] = torch.tensor(1)
                else:
                    t[i, j, 2] = torch.tensor(1)  # 无子层
        return t

#model
class trymodel(nn.Module):
    def __init__(self):
        super().__init__()
        #362 * 362 (边缘都是非棋子，不加填充)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels = 3,
                kernel_size= 6 ,
                stride=2,
            ),
            nn.ReLU(),
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size= 3,
                stride= 2,
            )
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size= 3 ,
                stride=1,
            ),
            nn.ReLU(),
        )
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,
                stride=1,
            )
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=10,
                kernel_size= 3 ,
                stride=1,
            ),

            nn.ReLU(),
        )
        self.pool3 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,
                stride=1,
            )
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=12,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
        )
        self.pool4 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,
                stride=1,
            )
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=12,
                out_channels=15,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
        )
        self.pool5 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,
                stride=1,
            )
        )

        self.FC1 = nn.Linear(in_features=15*77*77,out_features=1000)
        self.FC2 = nn.Linear(1000,15*15*3)


    def forward(self,x):
        x = x.view(-1,1,362,362)
        out = self.conv1(x)
        print("conv1", out.shape)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.pool3(out)
        out = self.conv4(out)
        out = self.pool4(out)
        out = self.conv5(out)
        out = self.pool5(out)
        print(out.shape)
        out = out.view(out.shape[0],15*77*77)
        out = self.FC1(out)
        print("FC1", out.shape)
        out = self.FC2(out)
        print("FC2", out.shape)
        out = out.view(out.size(0),3,15,15)
        print('bs',out.shape)
        return F.softmax(out,dim=1)



if __name__ == '__main__':

    count = 0
    # parameters for training
    epoch_num = 2
    batch_size = 40

    i = 0

    #prepare data
    train_data = boarddataset()
    train_loader = data.DataLoader(dataset= train_data , batch_size=batch_size,shuffle= False )
    print('len of train_data',train_data.__len__())

    #construct model
    cnn = trymodel()
    optimizer = torch.optim.Adam(cnn.parameters())
    loss_func = nn.CrossEntropyLoss()

    # image : 362*362*3  board 15*15
    for epoch in range(epoch_num):
            for batch_image , batch_label in train_loader:
                i+=1
                x.append(i)

                #Image.fromarray(np.array(batch_image[0])).show()
                #print(batch_image[0].shape,batch_label[0])

                #print('image shape',batch_image.shape)
                #print('label shape', batch_label.shape)
                print(1)
                out = cnn(batch_image)
                #print(out.shape)

                print(2)

                loss = loss_func(out, batch_label.long())
                print(3)
                optimizer.zero_grad()
                print(4)
                loss.backward()
                print(5)
                optimizer.step()

                ty = loss.data.numpy()
                y.append(ty)
                print(str(count),'Epoch: ', epoch, '| train loss: %.4f' % ty)
                print("\n")
                count += 1

    #save model
    state = {'model': cnn.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, "model.pth")

    plt.plot(x,y)
    plt.xlabel('batch')
    # y轴文本
    plt.ylabel('loss')
    # 标题
    plt.title('loss change')
    plt.savefig('./5loss.jpg')
    plt.show()








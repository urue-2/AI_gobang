import torch
from  torch import nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
x=[]
y=[]

class dataset(data.Dataset):
    def __init__(self):
        self.cur_board = np.load("ANN_training_data/features.npy")
        self.next_move = np.load("ANN_training_data/labels.npy")
        self.len = 1840

    def __getitem__(self, index):
        cur_board = torch.FloatTensor(self.cur_board[index])
        next_move = torch.FloatTensor(self.next_move[index])
        print(cur_board.shape)
        return cur_board , next_move

    def __len__(self):
        return  self.len

class gobangmodel(nn.Module):
    def __init__(self):
        super().__init__()
        # 241
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=3,kernel_size=5, stride=1),nn.ReLU())
        self.pool1 = nn.MaxPool1d( kernel_size= 4,stride= 1)

        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=3,out_channels=6,kernel_size= 4 ,stride=2),nn.ReLU())
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=1)

        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=6, out_channels=12, kernel_size=3, stride=2),nn.ReLU())
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv4 = nn.Sequential(nn.Conv1d(in_channels=12, out_channels=15, kernel_size=4, stride=2),nn.ReLU())
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=1)

        self.conv5 = nn.Sequential(nn.Conv1d(in_channels=15, out_channels=20, kernel_size=2, stride=1),nn.ReLU())
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=1)

        # 18 20 1


        self.FC1 = nn.Linear(in_features=20*23,out_features=3*15*15)


    def forward(self,x):
        x = x.view(-1,1,241)
        out = self.conv1(x)


        out = self.pool1(out)

        out = self.conv2(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.pool3(out)

        out = self.conv4(out)
        out = self.pool4(out)

        out = self.conv5(out)
        out = self.pool5(out)



        print(out.shape,out.shape[0])
        out = out.view(out.shape[0], 20*23)
        out = self.FC1(out)
        print("FC1", out.shape)


        out = out.view(out.size(0),3,15,15)
        print('bs',out.shape)
        return F.softmax(out,dim=1)



if __name__ == '__main__':

    count = 0
    # parameters for training
    epoch_num = 2
    batch_size = 40

    #prepare data
    train_data = dataset()
    train_loader = data.DataLoader(dataset= train_data , batch_size=batch_size,shuffle= False )
    print('len of train_data',train_data.__len__())

    #construct model
    cnn = gobangmodel()
    optimizer = torch.optim.Adam(cnn.parameters())
    loss_func = nn.CrossEntropyLoss()

    i=0
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
                print(str(count),'Epoch: ', epoch, '| train loss: %.4f' %ty)
                print("\n")
                count += 1

    #save model
    state = {'model': cnn.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    torch.save(state, "gobangmodel_1.0.pth")

    plt.plot(x,y)
    plt.xlabel('batch')
    # y轴文本
    plt.ylabel('loss')
    # 标题
    plt.title('loss change')
    plt.savefig('./loss.jpg')
    plt.show()


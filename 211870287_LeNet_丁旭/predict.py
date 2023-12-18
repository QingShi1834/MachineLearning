import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
# from main import LeNet

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        # 卷积层1：输入图像深度=3，输出图像深度=16，卷积核大小=5*5，卷积步长=1;16表示输出维度，也表示卷积核个数
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1)
        # 池化层1：采用最大池化，区域集大小=2*2.池化步长=2
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1)
        # 池化层2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层1：输入大小=32*5*5，输出大小=120
        self.fc1 = nn.Linear(32*5*5,120)
        # 全连接层2
        self.fc2 = nn.Linear(120,84)
        # 全连接层3
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.relu(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)  # output(16, 14, 14)
        x = F.relu(self.conv2(x))  # output(32, 10, 10)
        x = self.pool2(x)  # output(32, 5, 5)
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x


transform = transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = LeNet()
net.load_state_dict(torch.load('Lenet.pth'))

im = Image.open('1.jpg').convert('RGB')
# print(im.shape)
im = transform(im)  # [C, H, W]
# 输入pytorch网络中要求的格式是[batch，channel，height，width]，所以这里增加一个维度
im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

with torch.no_grad():
    outputs = net(im)
    predict = torch.max(outputs, dim=1)[1].data.numpy() # 索引即classed中的类别
print(classes[int(predict)])

# 直接打印张量的预测结果
with torch.no_grad():
    outputs = net(im)
    predict = torch.softmax(outputs,dim=1) # [batch，channel，height，width],这里因为对batch不需要处理
print(predict)
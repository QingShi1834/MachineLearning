import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # 优化器


'''1.构建神经网络'''
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


'''2.下载数据集并查看部分数据'''
# transforms.Compose()函数将两个函数拼接起来。
# （ToTensor()：把一个PIL.Image转换成Tensor，Normalize()：标准化，即减均值，除以标准差）
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 训练集：下载CIFAR10数据集，如果没有事先下载该数据集，则将download参数改为True
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=False, transform=transform)
# 用DataLoader得到生成器，其中shuffle：是否将数据打乱；
# num_workers表示使用多进程加载的进程数，0代表不使用多进程
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=0)

# 测试集数据下载
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 显示图像
def imshow(img):
    # 因为标准化normalize是：output = (input-0.5)/0.5
    # 则反标准化unnormalize是：input = output*0.5 + 0.5
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # transpose()会更改多维数组的轴的顺序
    # Pytorch中是[channel，height，width],这里改为图像的[height，width，channel]
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 随机获取部分训练数据
dataiter = iter(trainloader)
images, labels = next(dataiter)
# 显示图像
# torchvision.utils.make_grid()将多张图片拼接在一张图中
imshow(torchvision.utils.make_grid(images))
# 打印标签
# str.join(sequence)：用于将序列中的元素以指定的字符str连接成一个新的字符串。这里的str是' '，空格
# %5s：表示输出字符串至少5个字符，不够5个的话，左侧用空格补
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


'''3.训练模型'''
# 有GPU就用GPU跑，没有就用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = LeNet()
net=net.to(device)
# print("该网络共有 {} 个参数".format(sum(x.numel() for x in net.parameters())))
# # 该网络共有 121182 个参数

# 对于多分类问题，应该使用Softmax函数，这里CIFAR10数据集属于多分类，却使用了交叉熵损失函数，
# 是因为进入CrossEntropyLoss()函数内部就会发现其中包含了Softmax函数
loss_function = nn.CrossEntropyLoss() # 使用交叉熵损失函数
# 优化器选择Adam，学习率设为0.001
optimizer = optim.Adam(net.parameters(), lr=0.001)
# 打印查看神经网络的结构
# print(net)
# # LeNet(
# #   (conv1): Conv2d(3, 16, kernel_size=(5, 5), stride=(1, 1))
# #   (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# #   (conv2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
# #   (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# #   (fc1): Linear(in_features=800, out_features=120, bias=True)
# #   (fc2): Linear(in_features=120, out_features=84, bias=True)
# #   (fc3): Linear(in_features=84, out_features=10, bias=True)
# # )


# 初始化损失值列表
loss_list = []
acc_list = []
for epoch in range(10): # 整个迭代10轮
    running_loss = 0.0 # 初始化损失函数值loss=0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, start=0):
        # 获取训练数据
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) # 将数据及标签传入GPU/CPU

        # 权重参数梯度清零
        optimizer.zero_grad()

        # 正向及反向传播
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        # 显示损失值
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))

            # 记录损失值
            loss_list.append(running_loss / 2000)
            acc_list.append(100 * correct / total)

            running_loss = 0.0
            correct = 0
            total = 0

print('Finished Training')

# 绘制损失函数随时间的变化曲线
plt.subplot(2, 1, 1)
plt.plot(loss_list)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')

# 绘制准确率随时间的变化曲线
plt.subplot(2, 1, 2)
plt.plot(acc_list)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.show()

'''4.在测试集中随机预测四张图看看效果'''
dataiter = iter(testloader)
images, labels = next(dataiter)

imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

images, labels = images.to(device), labels.to(device)
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]for j in range(4)))


'''5.测试模型在测试集上的准确率及10个类别的准确率'''
correct = 0
total = 0
# with是一个上下文管理器
# with torch.no_grad()表示其包括的内容不需要计算梯度，也不会进行反向传播，节省内存
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        # torch.max(outputs.data, 1)返回outputs每一行中最大值的那个元素，且返回其索引
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# 打印10个分类的准确率
class_correct = list(0. for i in range(10)) #class_correct=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
class_total = list(0. for i in range(10)) #class_total=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images) # outputs的维度是：4*10
        # torch.max(outputs.data, 1)返回outputs每一行中最大值的那个元素，且返回其索引
        # 此时predicted的维度是：4*1
        _, predicted = torch.max(outputs, 1)
        # 此时c的维度：4将预测值与实际标签进行比较，且进行降维
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


'''6.存储训练好的权重文件'''
save_path = './Lenet.pth'
torch.save(net.state_dict(), save_path)
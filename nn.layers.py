import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

tensor_trans = torchvision.transforms.ToTensor()
test_set = torchvision.datasets.CIFAR10(root="./datasets", train=False,transform=tensor_trans, download=True)
test_loader_64 = DataLoader(dataset=test_set, batch_size=64, shuffle=False, num_workers=0, drop_last=False)


class Conv_Module(nn.Module):
    def __init__(self):
        super(Conv_Module, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        return self.conv1(x)

class MaxPool_Module(nn.Module):
    def __init__(self):
        super(MaxPool_Module, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        return self.maxpool1(x)

class ReLU_Module(nn.Module):
    def __init__(self):
        super(ReLU_Module, self).__init__()
        # inplace=True 的时候会直接替换原始数据，否则不替换，默认不替换
        self.relu1 = nn.ReLU()

    def forward(self, x):
        return self.relu1(x)

class Sigmoid_Module(nn.Module):
    def __init__(self):
        super(Sigmoid_Module, self).__init__()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid1(x)

class Linear_Module(nn.Module):
    def __init__(self):
        super(Linear_Module, self).__init__()
        self.liner1 = nn.Linear(in_features=196608, out_features=10)

    def forward(self, x):
        return self.liner1(x)

class Dropout_Module(nn.Module):
    def __init__(self) -> None:
        super(Dropout_Module, self).__init__()
        self.dropout1 = nn.Dropout(p=0.1)
    
    def forward(self, x):
        return self.dropout1(x)

writer = SummaryWriter("logs")

conv_module = Conv_Module()
maxpool_module = MaxPool_Module()
relu_module = ReLU_Module()
sigmoid_module = Sigmoid_Module()
linear_module = Linear_Module()
dropout_module = Dropout_Module()

flag = 0
for data in test_loader_64:
    imgs, targets = data
    writer.add_images("input_imgs", imgs, flag)

    conv_img = conv_module(imgs)
    writer.add_images("conv_img", conv_img, flag)

    maxpool_img = maxpool_module(imgs)
    writer.add_images("maxpool_img", maxpool_img, flag)

    relu_img = relu_module(imgs)
    writer.add_images("relu_img", relu_img, flag)

    sigmoid_img = sigmoid_module(imgs)
    writer.add_images("sigmoid_img", sigmoid_img, flag)

    dropout_img = dropout_module(imgs)
    writer.add_images("dropout_img", dropout_img, flag)

    # 展平输入数据
    imgs = torch.reshape(imgs, (1,1,1,-1))
    # imgs = torch.flatten(imgs)
    # print(imgs.shape)
    if (imgs.shape[-1] == 196608):
        linear_img = linear_module(imgs)
        writer.add_images("linear_img", linear_img, flag)
    
    flag += 1
    print(flag)

writer.close()
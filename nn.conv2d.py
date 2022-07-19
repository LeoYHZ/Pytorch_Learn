import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

tensor_trans = torchvision.transforms.ToTensor()
test_set = torchvision.datasets.CIFAR10(root="./datasets", train=False,transform=tensor_trans, download=True)
test_loader_64 = DataLoader(dataset=test_set, batch_size=64, shuffle=False, num_workers=0, drop_last=False)


class Test_Module(nn.Module):
    def __init__(self):
        super(Test_Module, self).__init__()
        # 用于图像处理的时候 in_channels 一般设置为3
        # out_channels 实际上等于 in_channels 乘以卷积核的个数（有几个卷积核）
        # kernel_size=3 时卷积核设置为 3*3 可以自行设置其他特殊值
        # stride默认为1 每次卷积操作时步进为1 padding=0 不进行数据填充
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        return self.conv1(x)

writer = SummaryWriter("logs")

test_module = Test_Module()
flag = 0
for data in test_loader_64:
    imgs, targets = data
    ouput_data = test_module(imgs)
    writer.add_images("input_imgs", imgs, flag)

    # 输出为6个chanels，无法显示数据
    ouput_data = torch.reshape(ouput_data, (-1, 3, 30, 30))
    writer.add_images("conv2d_imgs", ouput_data, flag)
    flag += 1
    print(flag)
    # print(imgs.shape)
    # print(ouput_data.shape)

writer.close()
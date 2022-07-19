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
        # ceil_mode=True 时会保留所有的不满足kernel_size的数据进行最大池化
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        return self.maxpool(x)

writer = SummaryWriter("logs")

test_module = Test_Module()
flag = 0
for data in test_loader_64:
    imgs, targets = data
    ouput_data = test_module(imgs)
    writer.add_images("input_imgs", imgs, flag)

    writer.add_images("maxpool_imgs", ouput_data, flag)
    flag += 1
    print(flag)
    # print(imgs.shape)
    # print(ouput_data.shape)

writer.close()
from __future__ import barry_as_FLUFL
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Linear_Module(nn.Module):
    def __init__(self) -> None:
        super(Linear_Module, self).__init__()
        self.linear1 = nn.Linear(in_features=64*3*32*32, out_features=10)
    
    def forward(self, x):
        return self.linear1(x)


dataset = torchvision.datasets.CIFAR10("./datasets", transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)

writer = SummaryWriter(log_dir="logs")

linear_layer = Linear_Module()

flag = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input_img", imgs, global_step=flag)
    print(imgs.shape)
    # 展平数据
    img_reshape = torch.reshape(imgs, (1, 1, 1, -1))
    # reshape转为1*1*1*196608
    # img_reshape = torch.flatten(imgs)
    # flatten转为196608
    print(img_reshape.shape)
    linear_img = linear_layer(img_reshape)
    writer.add_images("linear_img", linear_img, global_step=flag)
    flag += 1
    if (flag > 10):
        break

writer.close()

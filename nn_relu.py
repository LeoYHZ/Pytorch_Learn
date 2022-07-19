from turtle import forward
from matplotlib.transforms import Transform
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


input = torch.tensor([[1, -0.5],[-1, 3]])
output = torch.reshape(input, (-1, 1, 2, 2))

print(output)

dataset = torchvision.datasets.CIFAR10("./datasets", train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Test_ReLU(nn.Module):
    def __init__(self) -> None:
        super(Test_ReLU, self).__init__()
        self.relu1 = nn.ReLU()
    
    def forward(self, x):
        return self.relu1(x)

test_relu = Test_ReLU()

# output = test_relu(input)
# print(output)

writer = SummaryWriter(log_dir="./logs")
flag = 0
for data in dataloader:
    imgs, targets = data
    output_imgs = test_relu(imgs)
    writer.add_images("input", imgs, global_step=flag)
    writer.add_images("ReLU", output_imgs, global_step=flag)
    flag += 1

writer.close()

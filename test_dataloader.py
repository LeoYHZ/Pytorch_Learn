import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=True,transform=tensor_trans, download=True)

test_loader_4 = DataLoader(dataset=test_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
# img, target = test_set[0]
#
# print(img.shape)
# print(test_set.classes[target])
flag = 0
for data in test_loader_4:
    writer.add_images("test_loader_4", data[0], flag)
    flag += 1
    print(flag)
    # print(data[0])
    # print(data[1])
    # break

test_loader_32 = DataLoader(dataset=test_set, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
flag = 0
for data in test_loader_32:
    writer.add_images("test_loader_32", data[0], flag)
    flag += 1
    print(flag)

writer.close()
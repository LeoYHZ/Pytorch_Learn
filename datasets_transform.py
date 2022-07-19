import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()

train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True,transform=tensor_trans, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False,transform=tensor_trans, download=True)

# print(test_set[0])
#
# img, target = test_set[0]
# print(img)
# print(test_set.classes[target])

for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


# ToTensor 的使用，了解Tensor数据类型
img_path = "test_data/train/ants_image/6240329_72c01e663e.jpg"
img = Image.open(img_path)
print(img)
print(type(img))

writer = SummaryWriter("logs")


# 创建一个ToTensor的类
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)
print(img_tensor)
print(type(img_tensor))

writer.add_image("img_tensor", img_tensor)
writer.close()
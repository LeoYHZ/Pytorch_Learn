from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "test_data/train/ants_image/6240329_72c01e663e.jpg"
img = Image.open(img_path)
writer = SummaryWriter("logs")

# ToTensor
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)
# Normalize
norm_trans = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = norm_trans(img_tensor)
# Resize
resize_trans = transforms.Resize((512,512))
img_resize = resize_trans(img_tensor)
# Compose Resize+Normalize
compose_trans = transforms.Compose([resize_trans, norm_trans])
img_compose = compose_trans(img_tensor)

# RandomCrop
RandomC_trans = transforms.RandomCrop((200))
for i in range(10):
    img_RandomC = RandomC_trans(img_tensor)
    writer.add_image("img_RandomC", img_RandomC, i)

# print(img_tensor[0][0][0])
# print(img_norm[0][0][0])

writer.add_image("img_tensor", img_tensor)
writer.add_image("img_norm", img_norm)
writer.add_image("img_resize", img_resize)
writer.add_image("img_compose", img_compose)
writer.close()
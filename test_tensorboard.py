from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
img_path = "test_data/train/ants_image/0013035.jpg"
img_path_2 = "test_data/train/ants_image/6240338_93729615ec.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
# print(img_array.shape)
img_path_2 = "test_data/train/ants_image/6240338_93729615ec.jpg"
img_PIL_2 = Image.open(img_path_2)
img_array_2 = np.array(img_PIL_2)

writer.add_image("test", img_array, 1, dataformats='HWC')
writer.add_image("test", img_array_2, 2, dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)
for i in range(100):
    writer.add_scalar("y=x", i, i)
writer.close()
from torch.utils.data import Dataset
from PIL import Image
import os

# img_path="./hymenoptera_data/train/ants/0013035.jpg"

class mydata(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)

        # dir_path = "hymenoptera_data/train/ants"
        # img_path_list = os.listdir(dir_path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)

root_dir = "hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_sir = "bees"
ants_dataset = mydata(root_dir, ants_label_dir)
bees_dataset = mydata(root_dir, bees_label_sir)
# print(ants_dataset[0])
# print(ants_dataset.__len__())
img_ant_1, label = ants_dataset[1]
img_ant_1.show()
img_bee_1, label = bees_dataset[1]
img_bee_1.show()


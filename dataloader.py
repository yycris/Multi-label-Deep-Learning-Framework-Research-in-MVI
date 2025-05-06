import torch
from torch.utils import data
from PIL import Image
import numpy as np
import os
import pandas
from utils import *
import torchvision.transforms as t

class Dataset(data.Dataset):
    def __init__(self, data_path_x, data_path_y, label_path, transform=None):
        self.data_path_x = data_path_x
        self.data_path_y = data_path_y
        self.label_path = label_path
        self.transform = transform
        self.data_path_fat = "./data/fat"

    def __len__(self):
        return len(self.data_path_x)

    def __getitem__(self, index):
        img_x = Image.open(self.data_path_x[index]).convert("RGB")
        i = self.data_path_x[index][13:]
        i = i[:-4]
        img_y = Image.open(self.data_path_y + '/' + i + '.png').convert('RGB')

        resize_transform = T.Resize([224, 224])
        img_x, img_y = resize_transform(img_x, img_y)

        img_fat = Image.open(self.data_path_fat + '/' + i + '.png').convert('L')
        resize_transform_1 = t.Resize((224, 224))
        img_fat = resize_transform_1(img_fat)
        img_x_np = np.array(img_x)  # (H, W, 3)
        img_y_np = np.array(img_y)  # (H, W, 3)
        img_fat_np = np.array(img_fat)  # (H, W)
        img_fat_np = np.expand_dims(img_fat_np, axis=2)  # (H, W, 1)
        img_x_fat_np = np.concatenate((img_x_np, img_fat_np), axis=2)  # (H, W, 4)
        img_y_fat_np = np.concatenate((img_y_np, img_fat_np), axis=2)  # (H, W, 4)
        img_x_fat = Image.fromarray(img_x_fat_np)
        img_y_fat = Image.fromarray(img_y_fat_np)

        label_data = pandas.read_csv(os.path.join(self.label_path))

        # label = torch.tensor(label_data.loc[label_data['病历号'] == int(i), ['group1']].values.item())

        label = label_data.loc[label_data['病历号'] == int(i), ['group1', 'gender', 'satellite_nodules', 'tumor_size']].values.flatten()
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform is not None:

            # img_x, img_y = self.transform(img_x, img_y)
            img_x, img_y = self.transform(img_x_fat, img_y_fat)

        return img_x, img_y, label

if __name__ == '__main__':
    data_path_x = ['./data/train/619030.png']
    data_path_y = './data/vein'
    label_path = './data/3y.csv'
    train_transform = TrainTransforms()
    train_dataset = Dataset(data_path_x, data_path_y, label_path, transform=train_transform)
    print(train_dataset[0][0].shape)

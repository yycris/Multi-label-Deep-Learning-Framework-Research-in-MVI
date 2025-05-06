from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import torch
from torch.utils import data
import pandas

class Dataset(data.Dataset):
    def __init__(self, data_path, label_path, transform = None):
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.data_path_fat = "./data/fat"

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        img = Image.open(self.data_path[index]).convert('RGB')

        resize_transform = transforms.Resize((224, 224))
        img = resize_transform(img)

        i = self.data_path[index][13:]
        i = i[:-4]
        img_fat = Image.open(self.data_path_fat + '/' + i + '.png').convert('L')
        img_fat = resize_transform(img_fat)

        img_np = np.array(img)  # (H, W, 3)
        img_fat_np = np.array(img_fat)  # (H, W)
        img_fat_np = np.expand_dims(img_fat_np, axis=2)  # (H, W, 1)
        img_fat_np = np.concatenate((img_np, img_fat_np), axis=2)  # (H, W, 4)
        img_fat = Image.fromarray(img_fat_np)

        label_data = pandas.read_csv(os.path.join(self.label_path))
        label = torch.tensor(label_data.loc[label_data['病历号'] == int(i), ['group1']].values.item())
        if self.transform is not None:
            img = self.transform(img_fat)

        return img, label

data_path = './data'
label_path = './data/3y.csv'

other_transform = transforms.Compose([
        transforms.ToTensor()
])
def data_list(datapath:str):
    train_images_path = []
    val_images_path = []
    itest_images_path = []
    etest_images_path = []
    img_path = [datapath+'/train', datapath+'/valid', datapath+'/itest', datapath+'/etest']
    for path in img_path:
        for root, dirs, files in os.walk(path):
            for file in files:
                if path == datapath+'/train':
                    train_images_path.append(os.path.join(root, file))
                elif path == datapath+'/valid':
                    val_images_path.append(os.path.join(root, file))
                elif path == datapath+'/itest':
                    itest_images_path.append(os.path.join(root, file))
                else:
                    etest_images_path.append(os.path.join(root, file))
    return train_images_path, val_images_path, itest_images_path, etest_images_path

train_images_path,  val_images_path, itest_images_path, etest_images_path = data_list(data_path)

train_data = Dataset(train_images_path, label_path, transform=other_transform)
train_loader = DataLoader(dataset=train_data, batch_size=311, shuffle=True)
train = next(iter(train_loader))[0]
train_mean = np.mean(train.numpy(), axis=(0, 2, 3))
train_std = np.std(train.numpy(), axis=(0, 2, 3))


print("train_mean:", train_mean)
print("train_std:", train_std)


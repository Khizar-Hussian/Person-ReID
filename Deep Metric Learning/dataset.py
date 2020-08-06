import os
import time
import torch
import random
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import torchvision.datasets as dset
from numpy.random import choice as npc
from torch.utils.data import Dataset, DataLoader


class TrainLoader(Dataset):
    def __init__(self, dataPath, dataset_name, transform=None):
        super().__init__()
        np.random.seed(0)
        self.transform = transform
        self.dataset_name = dataset_name
        if self.dataset_name == 'PRID':
            self.data = self.load_PRID(dataPath)
        else:
            self.data = None

    def load_PRID(self, datapath):
        """
            loads PRID dataset and its labels in memory
            first 100 images from cam_a and cam_b are positive pairs
            negative pairs are balanced
            augmentation used: mirroring
        """
        print("[TrainLoader/load_PRID] loading dataset ...")
        data = []
        for folder in os.listdir(datapath):
            images = []
            for _file in os.listdir(os.path.join(datapath, folder)):
                filePath = os.path.join(datapath, folder, _file)
                try:
                    im = Image.open(filePath).resize((48,128))
                    images.append(im)
                    images.append(ImageOps.mirror(im))
                except:
                    print(f"[TrainLoader/load_PRID] error opening {filePath}")
            data.append(images)

        pos_pairs = [data[0][:200], data[1][:200]]
        neg_pairs = [data[0][200:], data[1][200:]]

        while len(neg_pairs[0]) < len(neg_pairs[1]):
            neg_pairs[0].append(neg_pairs[1].pop())

        pos_pairs[0].extend(neg_pairs[0])
        pos_pairs[1].extend(neg_pairs[1])
        all_pairs = [[i,j] for (i, j) in zip(pos_pairs[0],pos_pairs[1])]
        # first 200 images are positive pairs
        for i in range(len(all_pairs)):
            if i < 200:
                all_pairs[i].append(1)
            else:
                all_pairs[i].append(-1)
        print("[TrainLoader/load_PRID] finished loading dataset")
        return all_pairs
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, Y, L = self.data[index][0], self.data[index][1], self.data[index][2]
        if self.transform:
            x = self.transform(X)
            y = self.transform(Y) 
        return x, y, L



def main():
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    training_set = TrainLoader(os.path.normpath('D:\\Academia\\Semester 7\\FYP Papers\\Datasets\\prid_2011\\single_shot\\train'), dataset_name='PRID', transform=data_transforms)
    print(training_set.__len__())
    generator = DataLoader(training_set, 128, True)
    for X,Y,L in generator:
        print(X.shape, Y.shape, len(L))
        break


if __name__ == "__main__":
    main()
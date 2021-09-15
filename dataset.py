#from conf import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch 
import albumentations as A
import multiprocessing as mp 

import numpy as np
import cv2
import os
import numpy as np


class GLRDataset(Dataset):
    def __init__(self, df, normalization = 'simple', mode = 'train'):  
        self.normalization = normalization
        self.df = df
        self.img = list(df[df.columns[0]])
        self.eps = 1e-6
        self.root_path = os.path.join(os.path.abspath('../'),'train')


    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        #extract labels
        label = self.df.iloc[idx, 1]
        
        # extract of sample (in: normalization img, out: label)
        img_path = os.path.join(self.root_path, f"{self.img[idx]}.jpg")
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
            img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_AREA)
        except:
            print("FAIL READING IMG {}".format(img_path))
            img = np.zeros((512,512, 3), dtype = np.int8)

        img = img.astype(np.float32)
        if self.normalization:
            img = self.normalize_img(img)

        img = torch.from_numpy(img.transpose((2,0,1)))
        

        return img, label


    def normalize_img(self, img):
        if self.normalization == 'channel':
            pixel_mean = img.mean((0,1))
            pixel_std = img.std((0,1)) + self.eps

            img = (img - pixel_mean[None, None, :])/pixel_std[None, None, :]
            img = img.clip(-20, 20)


        elif self.normalization == 'channel_mean':
            pixel_mean = img.mean((0,1))
            img = (img - pixel_mean[None,None,:])
            img = img.clip(-20,20)

        elif self.normalization == 'image':
            img = (img - img.mean()) / img.std() + self.eps
            img = img.clip(-20,20)

        elif self.normalization == 'simple':
            img = img/255

        elif self.normalization == 'inception':

            mean = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            std = np.array([0.5, 0.5 , 0.5], dtype=np.float32)
            img = img.astype(np.float32)
            img = img/255.
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)

        elif self.normalization == 'imagenet':

            mean = np.array([123.675, 116.28 , 103.53 ], dtype=np.float32)
            std = np.array([58.395 , 57.120, 57.375   ], dtype=np.float32)
            img = img.astype(np.float32)
            img -= mean
            img *= np.reciprocal(std, dtype=np.float32)

        else:
            pass

        return img

if __name__ == '__main__':
    init_path = os.path.abspath('../')

    df = pd.read_csv(os.path.join(init_path,'train_new.csv'), sep = "\t")

    dataset = GLRDataset(df)
    idx, img, label = dataset.__getitem__(20000)
    print(idx)
    print(img)
    print(label)

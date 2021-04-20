#!/usr/bin/env python
# coding: utf-8



import os
import torch
import numpy 
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from torchvision import transforms

import matplotlib.pyplot as plt

"""
1) Each image should be preprocessed as follows:
- First, all values should be in a range of [0,1]
- Substract mean of 0.1307, and divide by std 0.3081
- These preprocessing can be implemented using torchvision.transforms
"""


class MNIST(Dataset):
    def __init__(self, data_dir, mean = 0,std = 0):
        list_files = os.listdir(data_dir)
        
        list_images = []
        list_labels = []


        ## ADD Gaussian noise for regularization
        if std != 0 or mean != 0:
            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.1307],
                std = [0.3081]),
            GaussianNoise(mean, std)
            ])
            
        else:
        ## Not Use Gaussian noise
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.1307],
                    std = [0.3081])
                ])
        
        ## Extract Image & label information => List
        for file_name in list_files:
            ## Make Border : 28x28 => 32x32  ==> list
            img = ImageOps.expand(Image.open(data_dir + file_name), border = 2, fill = 'black')
            list_images.append(transform(img))
            
            ## Extract label ==> list
            file_name_no_png = file_name.replace('.png', '')
            label = int(file_name_no_png.split('_')[-1]) 
            list_labels.append(label)
        

        self.list_images = torch.stack(list_images) 
        self.list_labels = torch.tensor(list_labels)

        print("number of pictures : {}".format(self.__len__()))
        
    def __len__(self):
        return len(self.list_labels)
        
    def __getitem__(self, idx):
        img = self.list_images[idx]
        label = self.list_labels[idx]
        
        return img, label

    
class GaussianNoise(object):
    def __init__(self, mean=0., std=0):
        self.std = std
        self.mean = mean
    def __call__(self, img):
        ## ADD Gaussian Noise to img
        return img + torch.randn(img.size())*self.std + self.mean



def test():
    print("Verify mnist dataset==============================")
    
    train_data_dir = '../data/mnist-classification/test/'
        
    print("dataset dir: {}".format(train_data_dir))
    
    train_datas = MNIST(train_data_dir, 0,0.3)
    
    print("first image shape in dataset: {}".format(numpy.array(train_datas[0][0]).shape))
    print("first label in dataset: {}".format(numpy.array(train_datas[0][1])))
    
    for img, label in train_datas:
        imgplot = plt.imshow(  torch.squeeze(img.permute(1, 2, 0)) , cmap='gray')
        plt.show()
        exit()



if __name__ == '__main__':
    test()
    

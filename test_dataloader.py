# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 23:05:04 2019

@author: Marco
"""
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

def touch_find_whole_image_name(protopath,num):
    array_of_img=[]
    filename=os.listdir(protopath)
    for i in range(0,num):
        p_string=(protopath,'/',filename[0],'/',str(i+1),'.jpg')
        p_string=''.join(p_string)
        array_of_img.append(p_string) 
    return array_of_img
    
class datasets_test(Dataset):
    
    def __init__(self,data_path,datanum,data_transform=None):     
        super().__init__()
        self.data_list = touch_find_whole_image_name(data_path,datanum)        
        self.data_transform = data_transform  

    def __getitem__(self, index):
        data_path = self.data_list[index]
        data_img = Image.open(data_path).convert("RGB")   
        label = np.loadtxt('./predict/predict_label.txt', delimiter=',')
        #label = np.loadtxt('./%s.txt' %str(index+1), delimiter=',')
        if self.data_transform is not None:
            data = self.data_transform(data_img)        
        return data, label

    def __len__(self):
        return len(self.data_list) 

# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:11:03 2019

@author: Marco
"""
import os
from torch.utils.data import Dataset
from PIL import Image

def touch_find_whole_image_name(protopath,num):
    array_of_img=[]
    filename=os.listdir(protopath)
    for i in range(0,num):
        p_string=(protopath,'/',filename[0],'/pokemon (',str(i+1),').png')
        p_string=''.join(p_string)
        array_of_img.append(p_string) 
    return array_of_img
    
class datasets(Dataset):
    
    def __init__(self,proto_path,data_path,proto_transform=None,data_transform=None):     
        super().__init__()
        self.proto_list = touch_find_whole_image_name(proto_path,925)
        self.data_list = touch_find_whole_image_name(data_path,925)        
        self.proto_transform = proto_transform
        self.data_transform = data_transform       

    def __getitem__(self, index):
        proto_path = self.proto_list[index]
        data_path = self.data_list[index]
        proto_img = Image.open(proto_path).convert("RGB")
        data_img = Image.open(data_path).convert("RGB")
        if self.data_transform is not None:
            proto = self.proto_transform(proto_img)
        if self.data_transform is not None:
            data = self.data_transform(data_img)        
        return proto, data

    def __len__(self):
        return len(self.data_list)     

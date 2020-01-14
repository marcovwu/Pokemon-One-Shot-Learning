# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 23:01:59 2019

@author: Marco
"""
import math
import os
import torch
import random
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from dataloader import datasets
from argparse import ArgumentParser
from test_dataloader import datasets_test
#### Set Hyperparameters
parser = ArgumentParser(description='AutoEncoder + Prototype (AE)')
parser.add_argument('--epoch',          type=int,   default=450,    help='Training epochs')
parser.add_argument('--imagesize',      type=int,   default=64 ,    help='esized image')
parser.add_argument('--batch_size',     type=int,   default=16 ,    help='Batch size')
parser.add_argument('--cnn_chn',        type=int,   default=16 ,    help='convolution first channel')
parser.add_argument('--learning_rate',  type=float,   default=1e-4,    help='Learning rate')
parser.add_argument('--latent_variable_size',   type=int,   default=2000,             help='Latent size')
parser.add_argument('--protopath',   type=str,   default='./data',             help='Prototype path')
parser.add_argument('--headpath',   type=str,   default='./head',             help='Head image path')
parser.add_argument('--resume_ename',   type=str,   default='./models/e_AE_F_finetune(0.5)+rrc.pkl',             help='Head image path')
parser.add_argument('--resume_dname',   type=str,   default='./models/d_AE_F_finetune(0.5)+rrc.pkl',             help='Head image path')
parser.add_argument('--encode_name',   type=str,   default='./models/e_AE_F.pkl',             help='Head image path')
parser.add_argument('--decode_name',   type=str,   default='./models/d_AE_F.pkl',             help='Head image path')
args = parser.parse_args()

cnn=[args.cnn_chn,args.cnn_chn*2,args.cnn_chn*4,args.cnn_chn*8,args.cnn_chn*16]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#train==0 go to train
train=input(" if (train==0，others==1): ")
resume=1
if train=='0':
    print('resume==0, otherwise no nor')
    resume=input('resume : ')
if resume=='0':
    resume_ename=args.resume_ename
    resume_dname=args.resume_dname
else:
    resume_ename=None
    resume_dname=None
transform_ori = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.imagesize, args.imagesize),Image.BILINEAR),
        torchvision.transforms.ToTensor(),
        ])
transform_aug = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.Resize((args.imagesize, args.imagesize),Image.BILINEAR),
        torchvision.transforms.ToTensor(),
        ])
transform_aug_RRC = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(args.imagesize, interpolation=2),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.Resize((args.imagesize, args.imagesize),Image.BILINEAR),
        torchvision.transforms.ToTensor(),
        ])
prototype_tuple = datasets(args.protopath,args.protopath,transform_ori,transform_aug)
prototype_tuple_RRC = datasets(args.protopath,args.protopath,transform_ori,transform_aug_RRC)
head_tuple = datasets(args.headpath,args.protopath,transform_ori,transform_aug)
train_loader = torch.utils.data.DataLoader(
        prototype_tuple,
        batch_size=args.batch_size,
        shuffle=True)
train_loader_RRC = torch.utils.data.DataLoader(
        prototype_tuple_RRC,
        batch_size=args.batch_size,
        shuffle=True)
train_loader_head = torch.utils.data.DataLoader(
        head_tuple,
        batch_size=args.batch_size,
        shuffle=True)
           
class Encoder(nn.Module):
    def __init__(self,batchsize):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,cnn[0],3,padding=1),  
                        nn.ReLU(),
                        nn.BatchNorm2d(cnn[0]),
                        nn.Conv2d(cnn[0],cnn[1],3,padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(cnn[1]),
                        nn.Conv2d(cnn[1],cnn[2],3,padding=1), 
                        nn.ReLU(),
                        nn.BatchNorm2d(cnn[2]),
                        nn.MaxPool2d(2,2)  
        )
        self.affine = nn.Sequential(
        )                       
        self.layer2 = nn.Sequential(
                        nn.Conv2d(cnn[2],cnn[3],3,padding=1), 
                        nn.ReLU(),
                        nn.BatchNorm2d(cnn[3]),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(cnn[3],cnn[4],3,padding=1), 
                        nn.ReLU()
        )

        self.batch_size=batchsize
        self.fc1=nn.Linear(cnn[4]*int(args.imagesize/4)*int(args.imagesize/4),args.latent_variable_size)
        self.fc2=nn.Linear(cnn[4]*int(args.imagesize/4)*int(args.imagesize/4),args.latent_variable_size)
                
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        num=self.batch_size
        if (int(out.numel())/cnn[4]/int(args.imagesize/4)/int(args.imagesize/4))!=self.batch_size:
            num=925%self.batch_size
        out = out.view(num, -1)
        
        return self.fc1(out),self.fc2(out)

def stn(x,num):    
    a=random.randint(-45,45)
    width=random.uniform(0.5,1.5)
    high=random.uniform(0.5,1.5)
    angle = a*math.pi/180
    theta_n = torch.tensor([
            [math.cos(angle)*width,math.sin(-angle),0],
            [math.sin(angle),math.cos(angle)*high ,0]
            ], dtype=torch.float)
    theta = torch.Tensor(num,2,3)
    theta[:]=theta_n
    grid = F.affine_grid(theta, x.size())
    output = F.grid_sample(x, grid)
    return output

def small_stn(x,num):   
    a=random.randint(-45,45)
    width=random.uniform(1.5,2.5)
    high=random.uniform(1.5,2.5)
    angle = a*math.pi/180
    theta_n = torch.tensor([
            [math.cos(angle)*width,math.sin(-angle),0],
            [math.sin(angle),math.cos(angle)*high ,0]
            ], dtype=torch.float)
    theta = torch.Tensor(num,2,3)
    theta[:]=theta_n
    grid = F.affine_grid(theta, x.size())
    output = F.grid_sample(x, grid)
    output = torch.where(output == 0, torch.Tensor([1]), output) 
    return output
     
class Decoder(nn.Module):
    def __init__(self,batchsize):
        super(Decoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.ConvTranspose2d(cnn[4],cnn[3],3,2,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(cnn[3]),
                        nn.ConvTranspose2d(cnn[3],cnn[2],3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(cnn[2])
        )
        self.layer2 = nn.Sequential(
                        nn.ConvTranspose2d(cnn[2],cnn[0],3,1,1),
                        nn.ReLU(),
                        nn.BatchNorm2d(cnn[0]),
                        nn.ConvTranspose2d(cnn[0],3,3,2,1,1),
                        nn.Sigmoid()
        )
        self.batch_size=batchsize
        self.dc=nn.Linear(args.latent_variable_size,cnn[4]*int(args.imagesize/4)*int(args.imagesize/4))
        
    def forward(self,x):
        x=self.dc(x)
        num=self.batch_size
        if (int(x.numel())/cnn[4]/int(args.imagesize/4)/int(args.imagesize/4))!=self.batch_size:
            num=925%self.batch_size
        out = x.view(num,cnn[4],int(args.imagesize/4),int(args.imagesize/4))
        out = self.layer1(out)
        out = self.layer2(out)
        return out
    
def takefirst(elem):
    return elem[0]    
    
if __name__ == "__main__":  
    if int(train)==0:
        loss_func = nn.BCELoss()
        loss_func.reduction = 'sum'
        encoder = Encoder(args.batch_size).cuda()
        decoder = Decoder(args.batch_size).cuda()
        if resume=='0':
            encoder.load_state_dict(torch.load(resume_ename))
            decoder.load_state_dict(torch.load(resume_dname))
        parameters = list(encoder.parameters())+ list(decoder.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
        best=100.0
        index=0
        average_loss=[]
        best_average=100.0
        avg_index=0
        best_score=0
        best_scoreindex=0
        best_scoreloss=0
        for i in range(args.epoch):
            print("Start Training in epoch:",i+1)
            sum_loss=0  
            
            #flip
            print('Training aug_data........')
            for j,(proto,image) in enumerate(train_loader):
                image= image.cuda()
                proto = proto.cuda()
                optimizer.zero_grad()                
                mu, logvar = encoder(image)
                res = decoder(mu)                 
                loss = loss_func(res,proto)                                                                                 
                loss.backward()
                optimizer.step()
                sum_loss+=loss/res.numel()
                print('epoch:%d aug_data--batch:%d/%d Loss:%.4f' %(i+1,j+1,int(925/args.batch_size)+1,float(loss/res.numel())))
            if loss/res.numel()<best:
                print('Saving modle...')
                best=loss/res.numel()
                index=i             
                torch.save(encoder.state_dict(), args.encode_name)  
                torch.save(decoder.state_dict(), args.decode_name) 
                excute=0
                print("Saved >>> model <<< Successful!!")

            #RRC
            print('Training RRC_data........')
            for j,(proto,image) in enumerate(train_loader_RRC):
                proto = proto.cuda()
                image = image.cuda()
                optimizer.zero_grad()
                mu, logvar = encoder(image)  
                res = decoder(mu) 
                loss = loss_func(res,proto)                                                                          
                loss.backward()
                optimizer.step()
                sum_loss+=loss/res.numel()
                print('epoch:%d rrc_data--batch:%d/%d Loss:%.4f' %(i+1,j+1,int(925/args.batch_size)+1,float(loss/res.numel())))
            if loss/res.numel()<best:
                print('Saving modle...')
                best=loss/res.numel()
                index=i                
                torch.save(encoder.state_dict(), args.encode_name)  
                torch.save(decoder.state_dict(), args.decode_name)
                excute=0
                print("Saved >>> model <<< Successful!!") 
            
            #smaller stn
            print('Training smaller stn_data........')
            for j,(proto,image) in enumerate(train_loader):
                num=args.batch_size
                if image.numel()!=args.batch_size*3*args.imagesize*args.imagesize:
                    num=len(prototype_tuple)%args.batch_size
                image=small_stn(image,num)
                proto = proto.cuda()
                image = image.cuda()
                optimizer.zero_grad()
                mu, logvar = encoder(image)  
                res = decoder(mu) 
                loss = loss_func(res,proto)                                                                          
                loss.backward()
                optimizer.step()
                sum_loss+=loss/res.numel()
                print('epoch:%d smaller stn_data--batch:%d/%d Loss:%.4f' %(i+1,j+1,int(925/args.batch_size)+1,float(loss/res.numel())))
            if loss/res.numel()<best:
                print('Saving modle...')
                best=loss/res.numel()
                index=i                
                torch.save(encoder.state_dict(), args.encode_name)  
                torch.save(decoder.state_dict(), args.decode_name)
                excute=0
                print("Saved >>> model <<< Successful!!")
                
            #stn
            print('Training stn_data........')
            for j,(proto,image) in enumerate(train_loader):
                num=args.batch_size
                if image.numel()!=args.batch_size*3*args.imagesize*args.imagesize:
                    num=len(prototype_tuple)%args.batch_size
                image=stn(image,num)
                proto = proto.cuda()
                image = image.cuda()
                optimizer.zero_grad()
                mu, logvar = encoder(image)  
                res = decoder(mu) 
                loss = loss_func(res,proto)                                                                          
                loss.backward()
                optimizer.step()
                sum_loss+=loss/res.numel()
                print('epoch:%d stn_data--batch:%d/%d Loss:%.4f' %(i+1,j+1,int(925/args.batch_size)+1,float(loss/res.numel())))
            if loss/res.numel()<best:
                print('Saving modle...')
                best=loss/res.numel()
                index=i                
                torch.save(encoder.state_dict(), args.encode_name)  
                torch.save(decoder.state_dict(), args.decode_name)
                excute=0
                print("Saved >>> model <<< Successful!!")

            average_loss.append(sum_loss/(j+1)/4)  
            print(' -Average_distance: ',round(float(sum_loss/(j+1)/4),4),"",'in epoch :',i+1)
            if sum_loss/(j+1)/4<best_average:
                print('Saving best_average modle...')
                best_average=sum_loss/(j+1)/4
                avg_index=i                
                torch.save(encoder.state_dict(), args.encode_name)  
                torch.save(decoder.state_dict(), args.decode_name)
                excute=0
                print("Saved >>> best_average model <<< Successful!!") 
            print(' -Best_avg_distance: ',round(float(best_average),4),"",'in epoch :',avg_index+1)
            if i%10==0:
                torchvision.utils.save_image(res,'./train_all/decode_image_epoch%d.jpg' %i)
                torchvision.utils.save_image(image,'./train_all/ori_image_epoch%d.jpg' %i)
                
            #################################################################################    
            #################################################################################
            ####test
            
            if excute==0:
                #encoder=encoder.eval()
                datapath ='./predict/predict_data'
                datanum=73
                print("Testing  data in epoch:%d" %(i+1))   
                data_transform = torchvision.transforms.Compose([
                        torchvision.transforms.Resize((args.imagesize, args.imagesize),Image.BILINEAR),
                        torchvision.transforms.ToTensor(),
                        ])                    
                data=datasets_test(datapath,datanum,data_transform)
                data_loader = torch.utils.data.DataLoader(
                        data,
                        batch_size=1,
                        shuffle=False)
                print("Loading  proto... 0%", end='\r')                  
                encoder.batch_size=1
                proto_tuple = datasets(args.protopath,args.protopath,transform_ori,transform_ori)
                latentarray=np.array((925,1,int(args.latent_variable_size)))
                latentarray = np.zeros(latentarray)
                for u in range(0,925):
                    string='Loading  proto... '+str(int((u+1)/925*100))+'%'
                    print(string, end='\r')
                    proto_tensor=proto_tuple[u][0].resize_(1,3,args.imagesize,args.imagesize) 
                    proto_tensor_cuda=proto_tensor.cuda(async=True)
                    use=encoder(proto_tensor_cuda)
                    use=use[0].cpu().detach().numpy()
                    latentarray[u][0]=use
                    vector_prot=latentarray 
                print("Loading  proto... 100%")
                print("Loaded >>> proto data <<< Successful!!") 
                print("\n--------test data loading--------\n")
                p_sumnew=[]
                p_list=[]
                Score=0
                top1=0
                top3=0
                top5=0
                top10=0
                encoder = encoder.cpu()
                for j,(image,label) in enumerate(data_loader):
                    data_latent,_ = encoder(image) 
                    p_sumnew=[]
                    a=data_latent[0].double()
                    for l in range(0,len(vector_prot[:])):            
                        b=torch.from_numpy(vector_prot[l])
                        use_p=F.cosine_similarity(a.unsqueeze(0), b, dim=1)
                        p_sumnew.append([use_p,l])
                    p_sumnew.sort(key=takefirst,reverse = True)
                    if p_sumnew[0][1]+1==label[0][j]:
                        top1+=1
                        top3+=1
                        top5+=1
                        top10+=1
                        print('epoch:%d predict number%d/%d RANK:1' %(i+1,j+1,datanum))
                        p_list.append(1)
                    elif p_sumnew[1][1]+1==label[0][j]:
                        top3+=1
                        top5+=1
                        top10+=1
                        print('epoch:%d predict number%d/%d RANK:2'  %(i+1,j+1,datanum))
                        p_list.append(2)
                    elif p_sumnew[2][1]+1==label[0][j]:
                        top3+=1
                        top5+=1
                        top10+=1
                        print('epoch:%d predict number%d/%d RANK:3' %(i+1,j+1,datanum))
                        p_list.append(3)
                    elif p_sumnew[3][1]+1==label[0][j]:
                        top5+=1
                        top10+=1
                        print('epoch:%d predict number%d/%d RANK:4'  %(i+1,j+1,datanum))
                        p_list.append(4)
                    elif p_sumnew[4][1]+1==label[0][j]:
                        top5+=1
                        top10+=1
                        print('epoch:%d predict number%d/%d RANK:5'  %(i+1,j+1,datanum))
                        p_list.append(5)
                    elif p_sumnew[5][1]+1==label[0][j]:
                        top10+=1
                        print('epoch:%d predict number%d/%d RANK:6'  %(i+1,j+1,datanum))
                        p_list.append(6)
                    elif p_sumnew[6][1]+1==label[0][j]:
                        top10+=1
                        print('epoch:%d predict number%d/%d RANK:7'  %(i+1,j+1,datanum))
                        p_list.append(7)
                    elif p_sumnew[7][1]+1==label[0][j]:
                        top10+=1
                        print('epoch:%d predict number%d/%d RANK:8'  %(i+1,j+1,datanum))
                        p_list.append(8)
                    elif p_sumnew[8][1]+1==label[0][j]:
                        top10+=1
                        print('epoch:%d predict number%d/%d RANK:9'  %(i+1,j+1,datanum))
                        p_list.append(9)
                    elif p_sumnew[9][1]+1==label[0][j]:
                        top10+=1
                        print('epoch:%d predict number%d/%d RANK:10'  %(i+1,j+1,datanum))
                        p_list.append(10)
                    else:
                        print('epoch:%d predict number%d/%d RANK:X'  %(i+1,j+1,datanum))
                        p_list.append(0)
                Score = (top5*0.6+top3*0.25+top1*0.1+top10*0.05)/datanum*100
                print('Top1:%d Top3:%d Top5:%d Top10:%d'  %(top1, top3, top5, top10))
                print('Top1:%.2f Top3:%.2f Top5:%.2f Top10:%.2f'  %(top1/datanum, top3/datanum, top5/datanum, top10/datanum))
                print('Score in epoch%d :%.2f[loss:%.4f] --Best_Score in epoch%d :%.2f[loss:%.4f]\n' %(i+1,Score,sum_loss/(int(925/args.batch_size)+1)/4,best_scoreindex+1,best_score,best_scoreloss))    
                string = open('./train_all/test_txt/epoch%d.txt' %(i+1), 'w')
                string.write('epoch:%d average_loss:%.4f\n' %(i+1,sum_loss/(int(925/args.batch_size)+1)/4))
                string.write('Top1:  %d Top3:  %d Top5: %d  Top10:  %d\n'  %(top1, top3, top5, top10))
                string.write('Top1:%.2f Top3:%.2f Top5:%.2f Top10:%.2f\n'  %(top1/datanum, top3/datanum, top5/datanum, top10/datanum))
                string.write('Score:%.2f\n' %Score)
                for l in range(0,datanum):
                    string.write('%2.d: %2.d\n' %(l+1,p_list[l]))
                string.close()
                if Score>best_score:
                    best_score=Score
                    best_scoreindex=i
                    best_scoreloss=sum_loss/(int(925/args.batch_size)+1)/4
                    string = open('./train_all/test_txt/Best_score.txt' , 'w')
                    string.write('epoch:%d average_loss:%.4f\n' %(i+1,sum_loss/(int(925/args.batch_size)+1)/4))
                    string.write('Top1:  %d Top3:  %d Top5: %d  Top10:  %d\n'  %(top1, top3, top5, top10))
                    string.write('Top1:%.2f Top3:%.2f Top5:%.2f Top10:%.2f\n'  %(top1/datanum, top3/datanum, top5/datanum, top10/datanum))
                    string.write('Best_score:%.2f\n' %Score)
                    for l in range(0,datanum):
                        string.write('%2.d: %2.d\n' %(l+1,p_list[l]))
                    string.close()
                    encoder.batch_size=args.batch_size
                    encoder.cuda()
                    print('Saving best score modle...')
                    torch.save(encoder.state_dict(), './models/e_AE_F_best_score.pkl')  
                    torch.save(decoder.state_dict(), './models/d_AE_F_best_score.pkl')  
                    print("Saved >>> best score modle <<< Successful!!")
                else:
                    encoder.batch_size=args.batch_size
                    encoder.cuda()
        y=np.arange(0,args.epoch,1)
        plt.title('Loss')
        plt.plot(y,average_loss)
        plt.savefig('./train_all/Loss.jpg')
        plt.show()      
        print(' -Min_distance: ',round(float(best),4),"",'in epoch :',index+1)
        print(' -Average_min_distance: ',round(float(best_average),4),"",'in epoch :',avg_index+1)
            
           
            
     
        
        
        
        
        
        
        
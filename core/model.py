#!/usr/bin/env python
# coding: utf-8



import torch.nn as nn
import torch
import numpy as np



class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

    - For a detailed architecture, refer to the lecture note
    - Freely choose activation functions as you want
    - For subsampling, use max pooling with kernel_size = (2,2)
    - Output should be a logit vector
    """


    def __init__(self, device):
        super(LeNet5,self).__init__()
        self.device = device
        
        self.c1 = nn.Conv2d(1,6,5)
        
        self.s2 = nn.Sequential(
                nn.MaxPool2d(2, stride = 2),
                nn.Sigmoid()
                )
        
        
        self.c3, self.c3_idx = self.c3_init()
        
        self.s4 = nn.Sequential(
                nn.MaxPool2d(2, stride = 2),
                nn.Sigmoid()
                )
        
        
        self.c5 = nn.Conv2d(16,120,5)
        
        self.fc6 = nn.Sequential(
                nn.Linear(120,84),
                nn.Tanh() 
                )
        self.A = 1.7159 #scaled hyperbolic tangent
        
        
        
        self.fc7 = nn.Sequential(
                nn.Linear(84,10),
                nn.Softmax(dim=1) #euclidean radial basis function unit XX => softmax
                )
        
    def c3_init(self):
        # Making Custom MLP Filter : C3
        c3 = nn.ModuleList()
        
        for _ in range(6):
            c3.append(nn.Conv2d(3,1,5))
        
        for _ in range(9):
            c3.append(nn.Conv2d(4,1,5))
        
        c3.append(nn.Conv2d(6,1,5))
        
        
        c3_idx = []
        
        temp = [0,1,2,3,4,5,0,1,2,3,4,5]
        
        for i in range(6):
            c3_idx.append(temp[i:i+3])
        
        for i in range(6):
            c3_idx.append(temp[i:i+4])  
        
        for i in range(3):
            c3_idx.append(temp[i:i+2]+temp[i+3:i+5])   
        
        c3_idx.append(temp[0:6])
        
        
        return c3, c3_idx        
        

        
        
    def forward(self,img): # orign: batchsize x 28 x 28 ==> input: batchsize x 32 x 32

        f1 = self.c1(img) # batchsize x 6 x 28 x 28
        f2 = self.s2(f1) # batchsize x 6 x 14 x 14
        
        f3 = torch.zeros(16,img.size()[0],10,10 , device = self.device)
        
        for i, idx_list in enumerate(self.c3_idx):
            f3[i] = torch.squeeze( self.c3[i](f2[:, idx_list,:,:])) # batchsize x 10 x 10
        
        f3 = f3.transpose(0,1) # 16 x batchsize x 10 x 10 => batchsize x 16 x 10 x 10
        f4 = self.s4(f3) # batchsize x 16 x 5 x 5 
        
        f5 = self.c5(f4) # batchsize x 120
        f5 = torch.squeeze(f5,2)
        f5 = torch.squeeze(f5,2)        
        f6 = self.fc6(f5)*self.A # batchsize x 84
        output = self.fc7(f6) #batchsize x 10

        return output

# In[ ]:


class CustomMLP(nn.Module):
    # Your custom MLP model

    #   - Note that the number of model parameters should be about the same
    #      with LeNet-5    
    
    def __init__(self, layer_option):
        # write your codes here
        super(CustomMLP,self).__init__()
        self.layers = nn.Sequential()

        
        prev = 0
        
        for i, num in enumerate(layer_option):
            if i == 0:
                self.layers.add_module('input_{}'.format(i+1), nn.Linear(1024, num))
                prev = num               
                
            else:
                self.layers.add_module('ReLU_{}'.format(i), nn.ReLU())
                self.layers.add_module('hidden_{}'.format(i+1) , nn.Linear(prev, num))
                prev = num
        
        i = i+1
        self.layers.add_module('ReLU_{}'.format(i), nn.ReLU())
        self.layers.add_module('hidden_{}'.format(i+1) , nn.Linear(num, 10) )
        self.layers.add_module('softmax',nn.Softmax(dim=1) )
        

                

    def forward(self, img):
        # write your codes here # img :  bathsize x 32 x 32
        batch_size = img.size()[0]
        img = img.view(batch_size, -1) # batch_size x 1024
        output = self.layers(img)# batch_size x 10

        return output
    



if __name__ == '__main__':
    
    print("Test LeNet5 model!")
    batch_size = 5
    input = torch.rand(batch_size, 1, 32,32).to(torch.device("cuda"))
    model = LeNet5().to(torch.device("cuda"))
    output = model(input)

    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

    print("Test CustomMLP model!")
    img = torch.rand(50, 32, 32)
    model = CustomMLP()
    output = model(img)
    print(output.size())
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)

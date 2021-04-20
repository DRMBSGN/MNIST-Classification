#!/usr/bin/env python
# coding: utf-8

# In[15]:


from core.model import LeNet5, CustomMLP
from core import dataset

from torch.optim import SGD
from torch.nn import CrossEntropyLoss

# import some packages you need here
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from glob import glob

from utils.CustomTimer import CheckTime
from utils.graph import draw_model_graph
import argparse


# In[10]:


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """
    loss_list = []
    correct = 0
    
    # write your codes here
    for i, (input_data, label) in enumerate( trn_loader) :
        input_data = input_data.to(device)
        label= label.to(device)
        
        pred = model(input_data)
        loss = criterion( pred , label )
        
        loss_list.append(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        correct += ( torch.argmax(pred.data.cpu(), dim=1) == label.data.cpu()).float().sum()
        
        if ( (i%1000) == 999 ):
            print(" progressing [ {:06d} / {:06d}] // train_loss: {:.6f} ".format(i+1, len(trn_loader), loss))
            
    
    loss_list = torch.stack(loss_list)
    trn_loss = torch.mean(loss_list)
    
    acc = 100 * correct / (len(trn_loader)*input_data.size()[0])
    
    
    return trn_loss, acc


# In[3]:


def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """

    # write your codes here
    loss_list = []
    correct = 0
    

    for i, (input_data, label) in enumerate( tst_loader) :
        input_data = input_data.to(device)
        label= label.to(device)
        
        pred = model(input_data)
        loss = criterion( pred , label )
        
        loss_list.append(loss)
        
        
        correct += ( torch.argmax(pred.data.cpu(), dim=1) == label.data.cpu()).float().sum()

            
    
    loss_list = torch.stack(loss_list)
    tst_loss = torch.mean(loss_list)
    
    acc = 100 * correct / (len(tst_loader)*input_data.size()[0])

    
    
    

    return tst_loss, acc


# In[22]:


def main(args):
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets 
        2) DataLoaders for training and testing 
        3) model  
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9 
        5) cost function: use torch.nn.CrossEntropyLoss 

    """

    # write your codes here    
    
    # Configuration
    mode = args.mode
    model_name = args.model
    options = args.o
    
    
    if mode == 'train':
        train_data_dir = args.d + '/train/'
    elif mode == 'test':
        test_data_dir = args.d + '/test/'
    elif mode == 'graph_compare':
        if model_name == 'LeNet5':
            models_name = ['LeNet5', 'LeNet5_insert_noise_s0.1_m0.0', 'LeNet5_insert_noise_s0.2_m0.0', 'LeNet5_insert_noise_s0.3_m0.0', 'LeNet5_weight_decay_0.0001', 'LeNet5_weight_decay_0.001', 'LeNet5_weight_decay_0.01']
        elif model_name == 'CustomMLP_6':
            models_name = ['CustomMLP_6','CustomMLP_6_weight_decay_1e-05' ,'CustomMLP_6_weight_decay_0.0001' ,'CustomMLP_6_weight_decay_0.001']
        else : 
            models_name = ['LeNet5','CustomMLP_1','CustomMLP_2', 'CustomMLP_3', 'CustomMLP_4', 'CustomMLP_5', 'CustomMLP_6']
        
    
    
    model_path = args.m
    device = torch.device("cuda:"+str(args.cuda))
    lr = 0.01
    momentum = 0.6
    
    batch_size = args.b
    epoch = args.e
    
    use_ckpt = args.c
    
    
    
    
    
    if model_name == "CustomMLP_1":
        layer_option = [54, 47, 35, 10, 39]
    elif model_name == "CustomMLP_2":
        layer_option = [55, 35, 30, 34]
    elif model_name == "CustomMLP_3":
        layer_option = [55, 34, 33, 31]
    elif model_name == "CustomMLP_4":
        layer_option = [55, 41, 41]
    elif model_name == "CustomMLP_5":
        layer_option = [56, 51]
    elif model_name == "CustomMLP_6":
        layer_option = [58]
    
    
    
        
    ##change models
    if mode != "graph_compare":
        if model_name.split('_')[0] == "LeNet5":
            model = LeNet5( device ).to(device)

        elif model_name.split('_')[0] == "CustomMLP":
            model = CustomMLP(layer_option).to(device)
    

    
    
    ##change model name
    if options:
        model_name = model_name + '_' + options
    
    if options == "weight_decay":
        weight_decay = args.w
        gausian_noise_mean = 0.
        gausian_noise_std = 0.
        model_name +=  '_'+ str(weight_decay)
    elif options == "insert_noise":
        weight_decay = 0.
        gausian_noise_mean = args.mean
        gausian_noise_std = args.std
        model_name +=  '_s'+ str(gausian_noise_std)+"_m"+str(gausian_noise_mean)
    else:
        weight_decay = 0.
   
    ##change criterion
    criterion = CrossEntropyLoss()
    
    
    #Custom TimeModule
    mytime = CheckTime()

    
    
    
    if mode == "train":
        
        # Load Dataset
        print("{} Start Loading Train Dataset ===================================".format(mytime.get_running_time_str()))

        train_dataset = dataset.MNIST(train_data_dir, gausian_noise_mean, gausian_noise_std)
        train_dataloader = DataLoader(train_dataset,batch_size, shuffle=True )
        
        # initiate optimizer
        optimizer = SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)
                
        
        # If use checkpoint ...
        
        if use_ckpt :
            ckpt_files = glob(model_path+'{}_model_*.pt'.format(model_name)) 
            ckpt_files.sort()
            
            ckpt_model_path = ckpt_files[-1]
            epoch_info = torch.load(ckpt_model_path, map_location = device)
            
            start_epoch = epoch_info['epoch'] - 1
            
            model.load_state_dict(epoch_info['model'])
            optimizer.load_state_dic(epoch_info['optimizer'])
            
            
            total_trn_loss = epoch_info['total_trn_loss']
            total_trn_acc = epoch_info['total_trn_acc']
            
        
        else:
            start_epoch = 0
            total_trn_loss = []
            total_trn_acc = []
            
        
        
        
        # Check Random Parameter Model Loss & Accuracy
        print("{} Check Random Parameter Model {}========================================= ".format(mytime.get_running_time_str() , model_name))
        
        with torch.no_grad():
            trn_loss, trn_acc = test(model , train_dataloader, device, criterion)
            total_trn_loss.append(trn_loss.item())
            total_trn_acc.append(trn_acc.item())
            i = 0
            
            torch.save({
                    'epoch' : i,
                    'model' : model.state_dict(),
                    'opimizer' : optimizer.state_dict(),
                    'total_trn_loss' : total_trn_loss,
                    'total_trn_acc' : total_trn_acc},
                    
                    model_path +'{}_model_{:04d}.pt'.format(model_name, i))
        
        
        print("{} train {} // epoch: {} // loss: {:.6f} // accuracy: {:.2f} ".format(mytime.get_running_time_str() , model_name, i, trn_loss ,trn_acc ))
        
        
        
        
        # Start traing model
        print("{} Start Training {}========================================= ".format(mytime.get_running_time_str() , model_name))
        for i in range(start_epoch, epoch):
            trn_loss, trn_acc = train(model , train_dataloader, device, criterion, optimizer)
            
            total_trn_loss.append(trn_loss.item())
            total_trn_acc.append(trn_acc.item())
            
            torch.save({
                'epoch' : i,
                'model' : model.state_dict(),
                'opimizer' : optimizer.state_dict(),
                'total_trn_loss' : total_trn_loss,
                'total_trn_acc' : total_trn_acc},
                
                model_path +'{}_model_{:04d}.pt'.format(model_name, i+1))
            
            print("{} train {} // epoch: {} // loss: {:.6f} // accuracy: {:.2f} ".format(mytime.get_running_time_str() , model_name, i+1, trn_loss ,trn_acc ))
            
    
            
            

            
            
    if mode == "test" :
        #Start Loading Test Dataset
        print("{} Start Loading Test Dataset ===================================".format(mytime.get_running_time_str()))
        test_dataset = dataset.MNIST(test_data_dir)
        test_dataloader = DataLoader(test_dataset,batch_size, shuffle=True )
        
        # Start Testing model 
        with torch.no_grad():

            ckpt_files = glob(model_path + '{}_model_*.pt'.format(model_name)) 
            ckpt_files.sort()
            
            total_tst_loss = []
            total_tst_acc = []
            
            
            for i, ckpt_model_path in enumerate(ckpt_files):
                
                epoch_info = torch.load(ckpt_model_path, map_location = device)
                
                model.load_state_dict(epoch_info['model'])
                
                tst_loss, tst_acc = test(model, test_dataloader, device, criterion)

                total_tst_loss.append( tst_loss.item() )
                total_tst_acc.append( tst_acc.item() )
                
                epoch_info['total_tst_loss'] = total_tst_loss
                epoch_info['total_tst_acc'] = total_tst_acc
                
                torch.save( epoch_info , ckpt_model_path)
                
                print("{} test {} // model_num: {} // loss: {:.6f} // accuracy: {:.2f} ".format(mytime.get_running_time_str() , model_name, i, tst_loss ,tst_acc ))
                    
            
    
    if mode == "graph" :
        
        #Load models to draw graph
        ckpt_files = glob(model_path+'{}_model_*.pt'.format(model_name)) 
        ckpt_files.sort()
        
        epoch_info = torch.load(ckpt_files[-1])
        
        #initiate loss and accuracy dictionary
        loss_dic = {}
        acc_dic = {}
        
        
        #add loss and accuracy list 
        loss_dic['train'] = epoch_info['total_trn_loss']
        loss_dic['test'] = epoch_info['total_tst_loss']
        
        acc_dic['train'] = epoch_info['total_trn_acc']
        acc_dic['test'] = epoch_info['total_tst_acc']
        
        num_epoch = len(loss_dic['train'])
        

        
        #Draw Graph per model: trn_loss + tst_loss 
        graph_name = "Loss (model - {}) ".format(model_name)
        draw_model_graph(graph_name, num_epoch, loss_dic, graph_mode  = "loss" , save = args.s , zoom_plot = args.z)
        
        #Draw Graph per model: trn_acc + tst_acc
        graph_name = "Accuracy (model - {}) ".format(model_name)
        draw_model_graph(graph_name, num_epoch, acc_dic, graph_mode  = "acc" , save = args.s , zoom_plot = args.z)
        

        
    
    if mode == "graph_compare" :        
        tst_loss_dic = {}
        tst_acc_dic = {}
        
        
        print(models_name)
        
        
        #Load pre-defined models
        for model_name in models_name:
            model_file_name = model_path+'{}_model_{:04d}.pt'.format(model_name,epoch) 
            epoch_info = torch.load(model_file_name)

            tst_loss_dic[model_name] = epoch_info['total_tst_loss']
            tst_acc_dic[model_name] = epoch_info['total_tst_acc']
            
            num_epoch = len(tst_loss_dic[model_name])
        
                   
        
        #Comparison models: tst_loss 
        graph_name = "Compare Loss"
        draw_model_graph(graph_name, num_epoch, tst_loss_dic, graph_mode  = "loss" , save = args.s, zoom_plot = args.z)
        
        
        #Comparison models: tst_acc
        graph_name = "Compare Accuracy"
        draw_model_graph(graph_name, num_epoch, tst_acc_dic, graph_mode  = "acc" , save = args.s, zoom_plot = args.z)
        
        
        


    

    

def configuration():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', required = True, help = "test / train / graph / graph_compare")
    parser.add_argument('--model', required = True, help = 'model name')
    
    parser.add_argument('-b', required = False, type = int,default = 10,  help = 'batch size [default: 10 ]')
    parser.add_argument('-d', required = False,  type = str, default = 'data/mnist-classification', help = 'dataset files dir [default: data/mnist-classification/]')
    parser.add_argument('-e', required = False, type = int, default = 10,  help = "epoch [default:10]")
    
    parser.add_argument('-m', required = False, type = str, default = 'model/',  help = "model dir [default: model/]")
    
    parser.add_argument('-o', required = False, type = str , default = None, help = "model option weight_decay / insert_noise [default:None]")
    parser.add_argument('-w', required = False, type = float, default = 0.9,  help = "weight decay lambda [default:0.9]")
    
    parser.add_argument('-s', required = False, type=str2bool, default = False, help = "save plot [default: False]")
    parser.add_argument('-z', required = False, type=str2bool, default = False, help = "zoom plot [default: False]")
    
    parser.add_argument('-c', required = False, type=str2bool, default = False, help = "use ckpt [default:False]")

    parser.add_argument('--std', required = False, type = float, default = 0., help="add gausian noise in training [default:0.]")
    parser.add_argument('--mean', required = False, type = float, default = 0., help="add gausian noise in training [default:0.]")
    
    parser.add_argument('--cuda', required = False, type = int, default = 0, help = "cuda num [default:0]")
    
    args = parser.parse_args()

    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    




if __name__ == '__main__':
   args = configuration()
   main(args)


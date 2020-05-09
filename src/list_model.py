import torch
import torchvision
import numpy as np
import torch.nn as nn
from data_utils import *
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from ram_utils import gen_paths
import sys
from model_builder import IRNet, build_ir_model
import pdb 
from list_loss import *

def generate_sim(x,y):  
    # 128 X 127 
    simi_mat= torch.matmul(x,x.T)  #Cosine similarity calculaiton . i,j 
    my_list = list()
    num_samples = x.shape[0]
    r = torch.arange(num_samples)
    for i,e in enumerate(simi_mat):
        my_list.append(e[r != i]) 
        #my_list.append(e) 
    simi_mat  = torch.stack(my_list)
    my_list = list()
    for i,e in enumerate(y):
        temp = e ==y
        my_list.append(temp[ r != i])
        #my_list.append(temp)
    rel_mat = torch.stack(my_list)
    return simi_mat,rel_mat

if __name__ == "__main__":
    repo_path = sys.argv[1]
    (data_path,model_path,embd_path) = gen_paths(repo_path)
    train_data_path = f"{data_path}train/"
    BATCHES=32
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('loading_data...')
    # create data loader
    TINY = list_wise_tiny(data_path)
    DATALOADER = DataLoader(TINY, batch_size=BATCHES,shuffle=True) 
    #reminder to set numworker = 10 later 

    print('define hyper parameters and import model')
    # Hyper-parameters
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.0001

    #create the modela
    weight_path = f"{model_path}resnet50"
    MODEL= build_ir_model(weight_path).to(DEVICE)

    # Loss and optimizer
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr = 0.0001)

    # For updating learning rate
    def update_lr(OPTIMIZER, lr):
        for param_group in OPTIMIZER.param_groups:
            param_group['lr'] = lr

    LOSS_TR = []
    BIG_L = []

    print('begin training')
    # Train the model
    TOTAL_STEP = len(DATALOADER)
    CURR_LR = LEARNING_RATE
    print(TOTAL_STEP)
    print('')
    print('')
    for epoch in range(NUM_EPOCHS):
        for i,(s1,s2,imclass) in enumerate(DATALOADER):
            P = torch.cat((s1,s2),0)
            imclass = torch.cat((imclass,imclass),0)
            OPTIMIZER.zero_grad()
            #forward pass
            P= P.to(DEVICE) 
            P = MODEL(P)  # batchsize , ( somethign)
            p_norm = torch.norm(P, p=2, dim=1).reshape((BATCHES*2,1))
            P= P.div(p_norm.expand_as(P))
            dist_mat,truth_mat= generate_sim(P,imclass)
            # compute loss
            my_loss = TAPLoss()
            loss = my_loss(dist_mat.cpu(),truth_mat.cpu())
            print(loss)
            # Backward and optimize
            loss.backward()
            LOSS_TR.append(loss.item())
            OPTIMIZER.step()
            if (i+1) % 100 == 0:
                temp = sum(LOSS_TR)/len(LOSS_TR)
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, NUM_EPOCHS, i+1, TOTAL_STEP, temp))
                BIG_L = BIG_L + LOSS_TR
                LOSS_TR = []
        # Decay learning rate
        if (epoch+1) % 3 == 0:
            CURR_LR /= 1.5
            update_lr(OPTIMIZER, CURR_LR)
        torch.save(MODEL.state_dict(), 'MRS_list'+str(epoch)+'.ckpt')
        try :
            np.save('loss_file',BIG_L)
        except :
            pass

import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sampling import *
from med_sampling import * 
from list_loss import * 
import pdb 
import sys 

def build_model(DEVICE,embd_size=124,MODEL_PATH=None):
    #create the model
    MODEL  = models.resnet50(pretrained=True)
    num_features = MODEL.fc.in_features
    MODEL.fc = nn.Linear(num_features,embd_size) 
    MODEL = MODEL.to(DEVICE)
    if MODEL_PATH: 
        MODEL.load_state_dict(torch.load(MODEL_PATH))
    return MODEL

def parse_inputs():
    data_path=sys.argv[1]
    loss_name=sys.argv[2]
    embdDim = int(sys.argv[3])
    checkpointName = sys.argv[4]
    if len(sys.argv)==6:
        prevState=sys.argv[5]
    else: 
        prevState=None    
    return (data_path,loss_name,embdDim,checkpointName,prevState)
def update_lr(optimizer,lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 

def triplet_train_loop(device,epochs,data,model,opti,checkpointName):  
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    LOSS_TR = []
    BIG_L = []
    CURR_LR = LEARNING_RATE
    total_step = len(data)
    for epoch in range(epochs):
        for i, D in enumerate(data):
            opti.zero_grad()
            print(i,end='\r')
            #forward pass
            P = model(D[0].to(device))
            Q = model(D[1].to(device))
            R = model(D[2].to(device))
            # compute loss
            loss = triplet_loss(P,Q,R)
            # Backward and optimize
            loss.backward()
            LOSS_TR.append(loss.item())
            OPTIMIZER.step()
            print(loss)
            if (i+1) % 100 == 0:
                temp = sum(LOSS_TR)/len(LOSS_TR)
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1, epochs, i+1,total_step, temp))
                BIG_L = BIG_L + LOSS_TR
                LOSS_TR = []
        # Decay learning rate
        if (epoch+1) % 3 == 0:
            CURR_LR /= 1.5
            update_lr(OPTIMIZER, CURR_LR)
        torch.save(MODEL.state_dict(),checkpointName+"_"+str(epoch)+'.ckpt')
    np.save('loss_file',BIG_L)
def generate_sim(x,y):  
    # 128 X 127 
    simi_mat= torch.matmul(x,x.T)  #Cosine similarity calculaiton . i,j 
    my_list = list()
    num_samples = x.shape[0]
    r = torch.arange(num_samples)
    for i,e in enumerate(simi_mat):
        my_list.append(e[r != i]) 
    simi_mat  = torch.stack(my_list)
    my_list = list()
    for i,e in enumerate(y):
        temp = e ==y
        my_list.append(temp[ r != i])
    rel_mat = torch.stack(my_list)
    return simi_mat,rel_mat

def list_train_loop(device,learn_rate,epochs,data,model,opti,checkpointName):
    my_loss = TAPLoss()
    BIG_L = []
    LOSS_TR=[]
    for epoch in range(epochs):
        for i, (s1,s2,imclass) in enumerate(data): # changed D,L,IDX to just be D 
            print(i,end='\r')
            P = torch.cat((s1,s2),0)
            P = P.to(device)
            imclass = torch.cat((imclass,imclass),0)
            opti.zero_grad()
            #forward pass
            P = MODEL(P)  # batchsize , ( somethign)
            p_norm = torch.norm(P, p=2, dim=1).reshape((P.shape[0],1))
            P= P.div(p_norm.expand_as(P))
            dist_mat,truth_mat= generate_sim(P,imclass)
            # compute loss
            loss = my_loss(dist_mat.cpu(),truth_mat.cpu())
            print(loss)
            # Backward and optimize
            loss.backward()
            LOSS_TR.append(loss.item())
            opti.step()
            if (i+1) % 100 == 0:
                temp = sum(LOSS_TR)/len(LOSS_TR)
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}".format(epoch+1,epochs, i+1,epochs, temp))
                BIG_L = BIG_L + LOSS_TR
                LOSS_TR = []
        # Decay learning rate
        if (epoch+1) % 3 == 0:
            learn_rate /= 1.5
            update_lr(opti,learn_rate)
        torch.save(MODEL.state_dict(), 'MRS_'+checkpointName+str(epoch)+'.ckpt')
        try :
            np.save('loss_file',BIG_L)
        except :
            pass

if __name__=="__main__":
    (data_path,loss_name,embdDim,checkpointName,prevState) = parse_inputs() 
    print("Training on data: {d_path}".format(d_path=data_path))
    print("Loss used: {l}".format(l=loss_name))
    print("Embedding size: {d}".format(d=embdDim))
    print("Checkpoints stored under {ck}".format(ck=checkpointName) ) 
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('loading_data...')
    # create data loader
    med = True 
    if med :
        TINY = med_factory(mode="train",sampling=loss_name,data_path=data_path)
    else: 
        TINY = tiny_factory(mode="train",sampling=loss_name,data_path=data_path)
    DATALOADER = DataLoader(TINY, batch_size=16,shuffle=True)
    #  Create model params 
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.0001 
    MODEL = build_model(DEVICE,embd_size=embdDim,MODEL_PATH=prevState) 
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr = 0.0001)
    #internal helper functions for the lolz 
    print('begin training')
    # Train the model
    list_train_loop(DEVICE,LEARNING_RATE,NUM_EPOCHS,DATALOADER,MODEL,OPTIMIZER,checkpointName)

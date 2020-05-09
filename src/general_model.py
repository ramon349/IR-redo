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
    """ Script to build the standard backboen of our project. 
        main point of variance in our project is embedding size and 
        loss fuction used. For consistency all models are built from the same funciton 
    """ 
    MODEL  = models.resnet101(pretrained=True)
    num_features = MODEL.fc.in_features
    MODEL.fc = nn.Linear(num_features,embd_size) 
    MODEL = MODEL.to(DEVICE)
    if MODEL_PATH: 
        MODEL.load_state_dict(torch.load(MODEL_PATH))
    return MODEL

def parse_inputs():
    """ Model training is detrmined by imputs provided  by a bash script 
    data_path: is the path to the data
    loss_name: loss function to be used either triplet or list loss are expected 
    embedDim: is the embedding dimensionality 
    checkPointName: This is the base name for our embedding name. Should be 
    descriptive 
    prevState: should be path to a checkpoint if we intend to restart learning 
    """
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
    """ Helper function to help update learning rate.
    It's possible our loss may get stuck at some point. we'll decrease 
    learning rate to be more fine grained over time 

    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 

def triplet_train_loop(device,epochs,data,model,opti,checkpointName):  
    """This is the training loop for the triplet loss function. 
    sampling of data is unique. 
    """ 

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

def list_train_loop(device,learn_rate,epochs,data,model,opti,checkpointName):
    my_loss = TAPLoss()
    BIG_L = []
    LOSS_TR=[]
    for epoch in range(epochs):
        for i, (s1,s2,imclass) in enumerate(data): # see samplign code for s1,s2
            print(i,end='\r')
            # stack samples vertically onto a single batch 
            P = torch.cat((s1,s2),0)
            P = P.to(device)
            imclass = torch.cat((imclass,imclass),0)
            opti.zero_grad()
            #forward pass
            P = MODEL(P)  # batchsize , ( somethign)
            #Normalize vectors to unit norm 
            p_norm = torch.norm(P, p=2, dim=1).reshape((P.shape[0],1))
            P= P.div(p_norm.expand_as(P))
            # calculate similarity matrix needed for loss calculation 
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
        np.save('loss_file',BIG_L)

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

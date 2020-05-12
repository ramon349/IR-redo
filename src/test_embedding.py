import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
from PIL import Image
import skimage.color as col
from skimage import io, transform
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pdb 
import sys 
import numpy as np 
from data_utils import * 
from sampling import * 
from med_sampling import * 
from general_model import *
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
def get_class_names(file2class,files):
    # labels are still in torch format need to be converted if
    label_list = list(labels.cpu().numpy()) 
    named_labels = [file2class[e] for e in label_list]
    return named_labels
if __name__ == "__main__":
    model_chk = sys.argv[1]
    out_name=sys.argv[2]
    data=sys.argv[3]  
    embd_dim = int(sys.argv[4])
    data_path = sys.argv[5] 
    #logging stuff 
    print("Wil use checkpoint: {}".format(model_chk))
    print("embedding saved as: {}".format(out_name))
    print("data mode to be used: {}".format(data))
    print("Embedding dimension: {}".format(embd_dim))
    print("data path: {} ".format(data_path))
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # create data loade
    print("loading tiny modules")
    if data =="tiny":
        other_name = "tiny"
        print("hi, we are working on tiny")
        Tiny = tiny_factory(mode='test',data_path=data_path)
    else:  
        other_name= "med" 
        Tiny=med_factory(mode='test',data_path=data_path)
        print("med")
    dataloader = DataLoader(Tiny, batch_size=100,num_workers=16)
    #create the model
    model = build_model(device,embd_size=embd_dim,MODEL_PATH=model_chk) 
    def forward(x):
        x = x.type('torch.FloatTensor').to(device)
        return(model(x))
    embd_list = list()
    model.eval()
    label_list = list()
    with torch.no_grad():
        for k,(i,j) in enumerate(dataloader):
            print(k,end='\r')
            temp = forward(i)
            embd_list.append(temp.cpu().numpy())
            label_list.extend(j)
    embd = np.vstack(embd_list)
    embd_label = np.vstack(label_list)
    np.save(out_name,embd)
    np.save('test_{}_{}_label.npy'.format(other_name,embd_dim),label_list)

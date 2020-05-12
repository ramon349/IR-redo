
import os
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import skimage.color as col
from skimage import io, transform
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sys 
import pdb 
from metrics import * 
import numpy as np 
from sampling import * 
from med_sampling import * 
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
def get_med_counts():
    from collections import Counter 
    path = '/home/rlcorre/IR-final-project/IR-redo/train_med_128_label.npy'
    labels = np.load(path)
    return Counter(labels)
if __name__ =="__main__":
    embd_model_name = sys.argv[1]
    embd_dim = int(sys.argv[2])
    #use the trian_list-embd 
    # thest_list_embd 
    print('libraries imoprted')
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('loading training data...')
    # create data loader
    train_path = "/labs/sharmalab/cbir/dataset2/"
    val_path = "/labs/sharmalab/cbir/dataset2/val/"
    test = med_factory(mode='test',data_path=val_path) 
    train = med_factory(mode='eval',data_path=train_path)
    train_em = np.load('./embeddings/train_{}.npy'.format(embd_model_name))
    test_em = np.load('./embeddings/test_{}.npy'.format(embd_model_name)) 
    # get med counts for recall calculation 
    counts = get_med_counts()
    print('------------------')
    mean_list = list()
    from collections import defaultdict
    x = defaultdict(list)
    prec_30 = list()
    prec_10 = list() 
    rec_30 = list() 
    rec_10 = list() 
    for i in range(130):
        idx = i 
        temp = test_em[idx].reshape(1,embd_dim)
        print('QUERY IMAGE :')
        print(test.sample(idx))
        query_class = test.sample(idx)[1]
        query_label = train.class_dict[query_class]
        temp = (train_em-temp)**2
        temp = temp.sum(axis=1)**0.5
        temp2 = temp.argsort()[:30]
        labels = list()
        for j in temp2:
            labels.append(train.sample(j)[1]==query_label)
        for k in range(0,4):
            print(train.sample(temp2[k])[0])
        prec_k = acc_all(labels) 
        rec_k = recall_all(labels,query_class,counts)
        mean_list.append(np.mean(prec_k))
        prec_30.append(prec_k[29])
        prec_10.append(prec_k[9])
        rec_30.append(rec_k[29])
        rec_10.append(rec_k[9])
        x[query_label].append(np.mean(acc_all(labels)))
    print("Rec @30 mean val {}".format(np.mean(rec_30)))
    print("Rec @10 mean val {}".format(np.mean(rec_10)))
    print("Prec @30 mean val {}".format(np.mean(prec_30)))
    print("Prec @10 mean val {}".format(np.mean(prec_10)))
    print("The overal MAP is {}".format(np.mean(mean_list)))
    for k in x:  
        print("For class {} we have mean MAP of {}".format(train.rev_dict[k], np.mean(x[k])) )

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

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


if __name__ =="__main__":
    embd_model_name = sys.argv[1]
    #use the trian_list-embd 
    # thest_list_embd 
    print('libraries imoprted')
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('loading training data...')
    # create data loader
    train_path = "/home/rlcorre/IR-final-project/deep-ranking-master/tiny-imagenet-200/"
    val_path = "/home/rlcorre/IR-final-project/deep-ranking-master/tiny-imagenet-200/val/"
    test = tiny_test(val_path)
    train = sing_tiny(train_path)

    train_em = np.load('./embeddings/train_{}_embd.npy'.format(embd_model_name))
    test_em = np.load('./embeddings/test_{}_embd.npy'.format(embd_model_name))
    print('------------------')
    mean_list = list()
    from collections import defaultdict
    x = defaultdict(list)
    for i in range(130):
        idx = i 
        temp = test_em[idx].reshape(1,124)
        print('QUERY IMAGE :')
        print(test.sample(idx))
        query_class = test.class_dic[test.sample(idx) ]
        query_label = train.class_dict[query_class]
        temp = (train_em-temp)**2
        temp = temp.sum(axis=1)**0.5
        temp2 = temp.argsort()[:30]
        labels = list()
        for j in temp2:
            labels.append(train.sample(j)[1]==query_label)
        for k in range(0,4):
            print(train.sample(temp2[k])[0])
        mean_list.append(np.mean(acc_10(labels)[-1]))
        x[query_label].append(np.mean(acc_10(labels)[-1]))

    for k in x:
        print("For class {} we have mean MAP of {}".format(k, np.mean(x[k])) )

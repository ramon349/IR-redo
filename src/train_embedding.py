

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

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Tiny(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, loader = pil_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if transform == None :
            transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224),
                                                        torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                        torchvision.transforms.RandomVerticalFlip(p=0.5),
                                                        torchvision.transforms.ToTensor()])
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
        # class_dict -> n01443537 : 0 etc
        self.class_dict = {}
        # rev_dict -> 0 : n01443537 etc
        self.rev_dict = {}
        # image dict -> n01443537 : np.array([n01443537_0.JPEG    n01443537_150.JPEG  
        #                               n01443537_200.JPEG  n01443537_251.JPEG etc]) 
        self.image_dict = {}
        # big_dict -> idx : [img_name, class]
        self.big_dict = {}

        L = []

        for i,j in enumerate(os.listdir(os.path.join(self.root_dir))):
            self.class_dict[j] = i
            self.rev_dict[i] = j
            self.image_dict[j] = np.array(os.listdir(os.path.join(self.root_dir,j,'images')))
            for k,l in enumerate(os.listdir(os.path.join(self.root_dir,j,'images'))):
                L.append((l,i))
        for i,j in enumerate(L):
            self.big_dict[i] = j


        self.num_classes = 200

    def _sample(self,idx):
        im, im_class = self.big_dict[idx]
        path = os.path.join(self.root_dir,self.rev_dict[im_class],'images',im)
        return path, im_class

    def __len__(self):
        return len(self.big_dict)

    def __getitem__(self, idx):
        paths,im_class = self._sample(idx)
        temp = self.loader(paths)
        if self.transform:
            temp = self.transform(temp)
        return temp, im_class

if __name__ == "__main__":
    model_path = sys.argv[1] 
    embd_name = sys.argv[2]
    process = sys.argv[3]
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # create data loader
    if process=="tiny":
        other_name = "tiny"
        print("hi")
        Tiny = Tiny('/home/rlcorre/IR-final-project/deep-ranking-master/tiny-imagenet-200/train/')
    else:  
        other_name= "med" 
        Tiny=one_LTiny("/labs/sharmalab/cbir/dataset2/train/")
        print("med")
    dataloader = DataLoader(Tiny, batch_size=100)
    #create the model
    model  = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features,124)
    model = model.to(device)

    model.load_state_dict(torch.load(model_path))
    def forward(x):
        x = x.type('torch.FloatTensor').to(device)
        return(model(x))
    embd_list = list()
    model.eval()
    with torch.no_grad():
        L = []
        for k,(i,j) in enumerate(dataloader):
            print(k,end='\r')
            temp = forward(i)
            embd_list.append(temp.cpu().numpy())
            L = L+list(j.numpy())

    embd = np.vstack(embd_list)
    np.save(embd_name,embd)

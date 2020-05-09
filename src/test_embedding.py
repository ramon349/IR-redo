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
import pdb 
import sys 
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

        self.images = os.listdir(os.path.join(self.root_dir,'images'))
        annot_path = os.path.join(self.root_dir,'val_annotations.txt')
        self.image_class = np.array(pd.read_csv(annot_path, sep='\t')[['mage','class']]).astype('str')
        self.class_dic = {}
        for i in self.image_class :
            self.class_dic[i[0]]=i[1]

    def _sample(self,idx):
        path = os.path.join(self.root_dir,'images',self.images[idx])
        return path,self.class_dic[self.images[idx]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        paths,lab = self._sample(idx)
        temp = self.loader(paths)
        if self.transform:
            temp = self.transform(temp)
        return temp,lab
class one_LTiny(Dataset): 
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
        self.images = os.listdir(os.path.join(self.root_dir,'images'))
        annot_path = os.path.join(self.root_dir,'val_annotations2.txt')
        self.image_class = np.array(pd.read_csv(annot_path,sep="\t")[['mage','class']]).astype('str')
        self.class_dic = {}
        for i in self.image_class :
            self.class_dic[i[0]]=i[1]
    
    def __len__(self):
       return 130 
    def _sample(self,idx):
        im, im_class = self.image_class[idx]
        p1 = os.path.join(self.root_dir,'images',im)
        return ([p1],im_class)

    def __getitem__(self, idx):
        paths,imclass= self._sample(idx)
        images = []
        for i in paths :
            temp = self.loader(i)
            if self.transform:
                temp = self.transform(temp)
            images.append(temp)
        return (images[0],imclass)

if __name__ == "__main__":
    model_path = sys.argv[1] 
    embd_name = sys.argv[2]
    process = sys.argv[3]    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('loading training data...')
    # create data loader
    if process =="tiny":
        Tiny = Tiny('/home/rlcorre/IR-final-project/deep-ranking-master/tiny-imagenet-200/val')
    else: 
        other_name = "med" 
        Tiny = one_LTiny("/labs/sharmalab/cbir/dataset2/val/")
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

    L = []
    embd_list = list()
    model.eval()
    with torch.no_grad():
        for k,(i,j) in enumerate(dataloader):
            print(k,end='\r')
            temp = forward(i)
            embd_list.append(temp.cpu().numpy())
            L = L+list(j)
    embd = np.vstack(embd_list)
    np.save(embd_name,embd)

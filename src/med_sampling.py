import os
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import skimage.color as col
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import pdb 

def pil_loader(path):
    with open(path, 'rb') as f:
        IMG = Image.open(f)
        return IMG.convert('RGB')

class med_samp(Dataset):
    """ Custom dataset with triplet sampling, for the tiny image net dataset"""

    def __init__(self, root_dir='tiny-imagenet-200',transform=None, loader = pil_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if transform == None :
            transform = torchvision.transforms.Compose([torchvision.transforms.Resize(224),torchvision.transforms.RandomHorizontalFlip(p=0.5),torchvision.transforms.RandomVerticalFlip(p=0.5),torchvision.transforms.ToTensor()])

        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
        self.class_dict = {}
        self.rev_dict = {}
        self.image_dict = {}
        self.big_dict = {}
        L = []
        dataset_paths = os.listdir(os.path.join(self.root_dir,'train'))
        dataset_paths = [e for e in dataset_paths if e != '.DS_Store']
        for i,j in enumerate(dataset_paths):
            self.class_dict[j] = i
            self.rev_dict[i] = j
            image_paths:list = os.listdir(os.path.join(self.root_dir,'train',j))
            self.image_dict[j] = np.array(image_paths )
            for k,l in enumerate(os.listdir(os.path.join(self.root_dir,'train',j))):
                L.append((l,i))
        for i,j in enumerate(L):
            self.big_dict[i] = j
        self.num_classes = 12

    def sample(self,idx):
            im, im_class = self.big_dict[idx]
            im2 = np.random.choice(self.image_dict[self.rev_dict[im_class]])
            numbers = list(range(0,im_class)) + list(range(im_class+1,200))
            class3 = np.random.choice(numbers)
            im3 = np.random.choice(self.image_dict[self.rev_dict[class3]])
            p1 = os.path.join(self.root_dir,'train',self.rev_dict[im_class],im)
            p2 = os.path.join(self.root_dir,'train',self.rev_dict[im_class],im2)
            p3 = os.path.join(self.root_dir,'train',self.rev_dict[class3],im3)
            return[p1,p2,p3]

    def __len__(self):
        return len(self.big_dict)

    def __getitem__(self, idx):
        paths = self.sample(idx)
        images = []
        for i in paths :
            temp = self.loader(i)
            if self.transform:
                temp = self.transform(temp)
            images.append(temp)
        return (images[0],images[1],images[2])
class med_list(med_samp):
    def sample(self,idx):
        im, im_class = self.big_dict[idx]
        im2 = np.random.choice(self.image_dict[self.rev_dict[im_class]])
        p1 = os.path.join(self.root_dir,'train',self.rev_dict[im_class],im)
        p2 = os.path.join(self.root_dir,'train',self.rev_dict[im_class],im2)
        return [p1,p2],im_class
    def __getitem__(self, idx):
        paths,im_class = self.sample(idx)
        images = []
        for i in paths :
            temp = self.loader(i)
            if self.transform:
                temp = self.transform(temp)
            images.append(temp)
        return (images[0],images[1],im_class)
class med_sing(med_samp): 
    def sample(self,idx):
        im, im_class = self.big_dict[idx]
        p1 = os.path.join(self.root_dir,'train',self.rev_dict[im_class],im)
        return [p1],im_class
    def __getitem__(self, idx):
        paths,im_class = self.sample(idx)
        images = []
        for i in paths :
            temp = self.loader(i)
            if self.transform:
                temp = self.transform(temp)
            images.append(temp)
        return (images[0],im_class)
    
class med_test(Dataset):
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
        annot_path = os.path.join(self.root_dir,'val_annotations2.txt')
        self.image_class = np.array(pd.read_csv(annot_path, sep='\t')[['mage','class']]).astype('str')
        self.class_dic = {}
        for i in self.image_class :
            self.class_dic[i[0]]=i[1]

    def sample(self,idx):
        im = self.images[idx]
        im_class = self.class_dic[im]
        p1 = os.path.join(self.root_dir,'images',im)
        return [p1],im_class
    def __getitem__(self, idx):
        paths,im_class = self.sample(idx)
        images = []
        for i in paths :
            temp = self.loader(i)
            if self.transform:
                temp = self.transform(temp)
            images.append(temp)
        return (images[0],im_class)

    def __len__(self):
        return len(self.images)
def med_factory(mode="train",sampling=None,data_path=None):
    if mode =="test": 
        return med_test(data_path)
    if mode =="train":
        if sampling=="triplet":
            return  med_samp(data_path)
        if sampling =="list": 
            return med_list(data_path)
    if mode == "eval": 
        return med_sing(data_path) 
    raise ValueError("parameter names are incorrect ")
 

if __name__ == "__main__":
    import sys 
    data = sys.argv[1]
    model ="triplet" 
    sampling="single"
    out = tiny_factory(model=model,sampling=sampling,data_path=data) 
    for i in range(0,10):
        print(out.sample(i))
    d = Tiny(data) 

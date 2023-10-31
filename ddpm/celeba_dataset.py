import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from ddpm import script_utils

from PIL import Image
import os
import joblib

# NAME_LIST = ['_skin.png', '_nose.png', '_eye_g.png', '_l_eye.png', '_r_eye.png', '_l_brow.png', '_r_brow.png', '_l_ear.png', '_r_ear.png', '_mouth.png', '_u_lip.png', '_l_lip.png', '_hair.png', '_hat.png', '_ear_r.png', '_neck_l.png', '_neck.png', '_cloth.png', '00']
NAME_LIST = ['00', '_skin.png', '_nose.png', '_eye_g.png', '_l_eye.png', '_r_eye.png', '_l_brow.png', '_r_brow.png', '_l_ear.png', '_r_ear.png', '_mouth.png', '_u_lip.png', '_l_lip.png', '_hair.png', '_hat.png', '_ear_r.png', '_neck_l.png', '_neck.png', '_cloth.png']

def img_loader(path, file, mask=False):
    if not mask:
        pil_img = Image.open(path+file)
        pil_img = pil_img.resize((256, 256), Image.LANCZOS)
        return pil_img
    else:
        pil_img = Image.open(path)
        pil_img = pil_img.resize((256, 256), Image.LANCZOS).convert('L')
        np_img = np.asarray(pil_img)
        return np_img

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor / 255)*2 - 1
    return processed_tensor

class CelabaDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.data_dir = 'celeba_img/'
        self.mask_dir = 'celeba_mask/'
        self.transform = transform
        self.loader = img_loader

    def __len__(self):
        return len(os.listdir(self.root + self.data_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = os.listdir(self.root + self.data_dir)[idx]
        img = self.loader(self.root + self.data_dir, file, mask=False)
        if self.transform:
            img = self.transform(img) 

        mask_array = np.zeros((19, 256, 256))
        for i, mask in enumerate(NAME_LIST):
            mask_name = self.root + self.mask_dir + file.split('.')[0].zfill(5) + mask
            
            if os.path.isfile(mask_name):
                
                mask_array[i] = self.loader(mask_name, file='', mask=True)
                
            else:
                mask_array[i] = np.zeros((256, 256))


        mask_array = fixed_image_standardization(np.float32(mask_array))
        return img, mask_array

class CelebaDataset2(Dataset):
    def __init__(self, transform=None):
        self.data_dir = './datasets/celeba_img/'
        self.map_list = joblib.load('./datasets/map_list.pkl')
        self.mask_list = joblib.load('./datasets/mask_list.pkl')
        self.transform = transform
        self.loader = img_loader

    def __len__(self):
        return len(os.listdir(self.data_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file = os.listdir(self.data_dir)[idx]  
        data_i = int(file.split('.')[0])
        img = self.loader(self.data_dir, file, mask=False)
        if self.transform:
            img = self.transform(img)

        feat_map = self.map_list[data_i]
        
        mask = self.mask_list[data_i]
        
        return img, feat_map, mask #.transpose(1,2,0)


class CelebaSketchDataset(Dataset):
    def __init__(self, transform=None):
        self.img_dir = './datasets/celeba_img_nobg'
        self.sketch_dir = './datasets/celeba_sketch_nobg'
        self.transform = transform
        self.img_loader = img_loader
        self.sketch_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5,
                                    std=1)
                ])

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = os.listdir(self.img_dir)[idx]
        img = Image.open(os.path.join(self.img_dir, img_file))
        img = img.resize((256, 256), Image.LANCZOS)

        sketch_file = img_file.split('.')[0] + '_edges.jpg'
        sketch = Image.open(os.path.join(self.sketch_dir, sketch_file)).convert('L')
        # sketch = sketch.resize((512, 512), Image.LANCZOS)
        if self.transform:
            img = self.transform(img)

        sketch = self.sketch_transform(sketch)

        return img, sketch
    
class CelebaSketchMaskedDataset(Dataset):
    def __init__(self, transform=None):
        self.img_dir = './datasets/celeba_img_nobg'
        self.sketch_dir = './datasets/celeba_sketch_nobg'
        self.transform = transform
        self.img_loader = img_loader
        self.sketch_transform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Normalize(mean=0.5,
                                    std=0.5)
                ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = os.listdir(self.img_dir)[idx]
        img = Image.open(os.path.join(self.img_dir, img_file))
        img = img.resize((256, 256), Image.LANCZOS)

        sketch_file = img_file.split('.')[0] + '_edges.jpg'
        sketch = Image.open(os.path.join(self.sketch_dir, sketch_file)).convert('L')

        
        img = self.to_tensor(img)
        sketch = self.to_tensor(sketch)

        
        mask, map_mask = script_utils.get_mask(1,256)
        
        mask = mask.squeeze(0)
        map_mask = map_mask.squeeze(0)
        masked_img = img * (1-mask)/2

        sketch = sketch * (1+mask)/2
        sketch = sketch + torch.ones((1,256,256))*(1-mask)/2
        # sketch = sketch + masked_img

        if self.transform:
            img = self.transform(img)

        sketch = self.sketch_transform(sketch)

        

        return img, masked_img, map_mask, sketch
    
class CelebaSketchMaskedDatasetVQ(Dataset):
    def __init__(self, transform=None):
        self.img_dir = './datasets/celeba_img_nobg'
        self.sketch_dir = './datasets/celeba_sketch_nobg'
        self.transform = transform
        self.img_loader = img_loader
        self.sketch_transform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[1, 1, 1])
                ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = os.listdir(self.img_dir)[idx]
        img = Image.open(os.path.join(self.img_dir, img_file))
        img = img.resize((256, 256), Image.LANCZOS)

        sketch_file = img_file.split('.')[0] + '_edges.jpg'
        sketch = Image.open(os.path.join(self.sketch_dir, sketch_file))#.convert('L')

        
        img = self.to_tensor(img)
        sketch = self.to_tensor(sketch)

        
        mask, map_mask = script_utils.get_mask(1,256)
        
        mask = mask.squeeze(0)
        map_mask = map_mask.squeeze(0)
        masked_img = img * (1-mask)/2

        sketch = sketch * (1+mask)/2
        # sketch = sketch + torch.ones((1,256,256))*(1-mask)/2
        sketch = sketch + masked_img

        if self.transform:
            img = self.transform(img)

        sketch = self.sketch_transform(sketch)

        

        return img, map_mask, sketch
    
class CelebaSketchDatasetVQ(Dataset):
    def __init__(self, transform=None):
        self.img_dir = './datasets/celeba_img_nobg'
        self.sketch_dir = './datasets/celeba_sketch_nobg'
        self.transform = transform
        self.img_loader = img_loader
        self.sketch_transform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[1, 1, 1])
                ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = os.listdir(self.img_dir)[idx]
        img = Image.open(os.path.join(self.img_dir, img_file))
        img = img.resize((256, 256), Image.LANCZOS)

        sketch_file = img_file.split('.')[0] + '_edges.jpg'
        sketch = Image.open(os.path.join(self.sketch_dir, sketch_file))#.convert('L')

        img = self.to_tensor(img)
        sketch = self.to_tensor(sketch)

        if self.transform:
            img = self.transform(img)

        sketch = self.sketch_transform(sketch)

        return img, sketch
    
class CelebaSketchDatasetVQ2(Dataset):
    def __init__(self, transform=None):
        self.img_dir = './datasets/celeba_img_nobg'
        self.sketch_dir = './datasets/celeba_sketch_nobg'
        self.transform = transform
        self.img_loader = img_loader
        self.sketch_transform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.5],
                                     std=[1])
                ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = os.listdir(self.img_dir)[idx]
        img = Image.open(os.path.join(self.img_dir, img_file))#.convert('L')
        img = img.resize((256, 256), Image.LANCZOS)

        sketch_file = img_file.split('.')[0] + '_edges.jpg'
        sketch = Image.open(os.path.join(self.sketch_dir, sketch_file)).convert('L')

        img = self.to_tensor(img)
        sketch = self.to_tensor(sketch)

        if self.transform:
            img = self.transform(img)

        sketch = self.sketch_transform(sketch)

        return img, sketch
    
class CelebaVQ(Dataset):
    def __init__(self, transform=None):
        self.img_dir = './datasets/celeba_img_nobg'
        self.sketch_dir = './datasets/celeba_sketch_nobg'
        self.transform = transform
        self.img_loader = img_loader
        self.sketch_transform = transforms.Compose([
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.5],
                                     std=[1])
                ])
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(os.listdir(self.img_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file = os.listdir(self.img_dir)[idx]
        # img = Image.open(os.path.join(self.img_dir, img_file))
        # img = img.resize((256, 256), Image.LANCZOS)

        sketch_file = img_file.split('.')[0] + '_edges.jpg'
        sketch = Image.open(os.path.join(self.sketch_dir, sketch_file)).convert('L')

        # img = self.to_tensor(img)
        sketch = self.to_tensor(sketch)

        # mask, map_mask = script_utils.get_mask(1,256)
        
        # mask = mask.squeeze(0)
        # map_mask = map_mask.squeeze(0)
        # masked_img = img * (1-mask)/2

        # sketch = sketch * (1+mask)/2
        # sketch = sketch + masked_img

        # if self.transform:
            # img = self.transform(img)

        sketch = self.sketch_transform(sketch)

        return sketch
    





        
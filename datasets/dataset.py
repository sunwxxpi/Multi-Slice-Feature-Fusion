import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as T
from scipy import ndimage
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

def random_rot_flip(image, label):
    # image: (H,W,3), label:(H,W)
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0,1))
    label = np.rot90(label, k, axes=(0,1))
    
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    
    return image, label

def random_rotate(image, label):
    # image: (H,W,3), label: (H,W)
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, axes=(0,1), order=0, reshape=False)
    label = ndimage.rotate(label, angle, axes=(0,1), order=0, reshape=False)
    
    return image, label

def ct_normalization(image, lower=1016, upper=1807, mean=1223.2043595897762, std=133.03651991499345):
    np.clip(image, lower, upper, out=image)
    image = (image - mean) / max(std, 1e-8)

    return image

def shuffle_within_batch(batch):
    random.shuffle(batch)

    return default_collate(batch)

class RandomAugmentation:
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if np.random.rand() > 0.5:
            image, label = random_rot_flip(image, label)
        if np.random.rand() > 0.5:
            image, label = random_rotate(image, label)
            
        sample['image'], sample['label'] = image, label

        return sample

class Resize:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        # image:(H,W,3), label:(H,W)
        image, label = sample['image'], sample['label']
        x, y = image.shape[0], image.shape[1]
        
        if (x, y) != (self.output_size[0], self.output_size[1]):
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            
        sample['image'], sample['label'] = image, label

        return sample

class ToTensor:
    def __call__(self, sample):
        # image: (H,W,3) -> (3,H,W)
        image, label = sample['image'], sample['label']
        
        image = torch.from_numpy(image.astype(np.float32))
        image = image.permute(2,0,1)  # (3,H,W)
        label = torch.from_numpy(label.astype(np.int64))
        
        sample['image'], sample['label'] = image, label

        return sample

class COCA_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, train_ratio=0.8):
        self.transform = transform
        self.split = split
        self.data_dir = base_dir

        if split in ["train", "val"]:
            with open(os.path.join(list_dir, "train.txt"), 'r') as f:
                full_sample_list = f.readlines()
            train_samples, val_samples = train_test_split(full_sample_list, train_size=train_ratio, shuffle=False, random_state=42)
            self.sample_list = train_samples if split == "train" else val_samples
        else:
            with open(os.path.join(list_dir, "test.txt"), 'r') as f:
                self.sample_list = f.readlines()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx].strip('\n')
        data_path = os.path.join(self.data_dir, sample_name + '.npz')
        data = np.load(data_path)
        
        # image: (H,W,3), label:(H,W)
        image, label = data['image'], data['label']
        image = ct_normalization(image)
        
        sample = {'image': image, 'label': label, 'case_name': sample_name}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
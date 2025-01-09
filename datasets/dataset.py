import os
import random
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as T
from scipy import ndimage
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

def random_rot_flip(image, label):
    """Randomly rotate and flip the image and label."""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    
    return image, label

def random_rotate(image, label):
    """Randomly rotate the image and label by a small angle."""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    
    return image, label

def ct_normalization(image, lower=1017, upper=1801, mean=1222.90087890625, std=132.62820434570312):
    """Normalize the CT image using fixed intensity range and standardization."""
    np.clip(image, lower, upper, out=image)
    image = (image - mean) / max(std, 1e-8)
    
    return image

def fixed_min_max_normalization(image, min_val=0, max_val=2500):
    """Normalize the image based on fixed min and max values of 0 and 2500."""
    normalized_img = (image - min_val) / (max_val - min_val)
    
    return np.clip(normalized_img, 0, 1)

def shuffle_within_batch(batch):
    random.shuffle(batch)
    
    return default_collate(batch)

class RandomAugmentation:
    """Apply random rotations and flips to the image and label."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        if np.random.rand() > 0.5:
            image, label = random_rot_flip(image, label)
        if np.random.rand() > 0.5:
            image, label = random_rotate(image, label)
            
        sample['image'], sample['label'] = image, label
        
        return sample

class Resize:
    """Resize the image and label to the desired output size."""
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y = image.shape
        
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            
        sample['image'], sample['label'] = image, label
        
        return sample

class ToTensor:
    """Convert numpy arrays to PyTorch tensors."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int64))
        
        sample['image'], sample['label'] = image, label
        
        return sample

class RandomGenerator:
    """Compose random augmentations and preprocessing for training."""
    def __init__(self, output_size):
        self.transform = T.Compose([
            RandomAugmentation(),
            Resize(output_size),
            ToTensor()
        ])

    def __call__(self, sample):
        return self.transform(sample)

class COCA_dataset(Dataset):
    """Custom dataset for COCA data."""
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
            with open(os.path.join(list_dir, "test_vol.txt"), 'r') as f:
                self.sample_list = f.readlines()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx].strip('\n')

        if self.split in ["train", "val"]:
            # Use .npz files for train and val
            data_path = os.path.join(self.data_dir, sample_name + '.npz')
            data = np.load(data_path)
            
            image, label = data['image'], data['label']
        else:
            # Use .npy.h5 files for test
            filepath = os.path.join(self.data_dir, f"{sample_name}.npy.h5")
            
            with h5py.File(filepath, 'r') as data:
                image, label = data['image'][:], data['label'][:]

        image = ct_normalization(image)

        sample = {'image': image, 'label': label, 'case_name': sample_name}

        if self.transform:
            sample = self.transform(sample)

        return sample
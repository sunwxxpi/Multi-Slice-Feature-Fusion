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

# def ct_normalization(image, lower=1017, upper=1801, mean=1222.90087890625, std=132.62820434570312):
# def ct_normalization(image, lower=-2.0, upper=1521.0, mean=355.3804931640625, std=282.9181213378906):
def ct_normalization(image, lower=-2.001523494720459, upper=1521.00390625, mean=355.3795166015625, std=282.9187316894531):
    """Normalize the CT image using fixed intensity range and standardization."""
    np.clip(image, lower, upper, out=image)
    image = (image - mean) / max(std, 1e-8)
    
    return image

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

def load_hu_stats(path):
    # hu_stats_433.json -> ct_normalization 인자 dict
    import json
    with open(path, 'r') as f:
        s = json.load(f)
    return {k: float(s[k]) for k in ('lower', 'upper', 'mean', 'std')}

class COCAVolumeDataset(Dataset):
    """5-fold CV 용 per-case 볼륨 데이터셋 (MSFFM 미적용 = 단일 슬라이스 baseline).

    image_dir / label_dir 에는 case 당 (D,H,W) .npy 가 있고 memmap 으로 읽는다.
    sample_list 의 각 항목은 `case{gidx:04d}_slice{n:03d}` (n = triplet 시작 인덱스).
    MSFFM 브랜치와 동일한 fold 자산을 공유하되, 인접 슬라이스 없이 center(n+1)
    한 장만 (H,W) 로 반환한다 → 기존 COCA_dataset 과 동일한 1채널 입력 형식.
    center 슬라이스 집합·라벨이 MSFFM 평가와 일치하므로 공정 비교가 보장된다.
    """
    def __init__(self, image_dir, label_dir, sample_list, transform=None, hu_stats=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.hu = hu_stats or {}
        self.sample_list = [s.strip() for s in sample_list if s.strip()]
        self._mm = {}  # cid -> (image_memmap, label_memmap), worker(fork) 별로 lazy 채움

    def __len__(self):
        return len(self.sample_list)

    def _get_volumes(self, cid):
        mm = self._mm.get(cid)
        if mm is None:
            img = np.load(os.path.join(self.image_dir, cid + '.npy'), mmap_mode='r')
            lab = np.load(os.path.join(self.label_dir, cid + '.npy'), mmap_mode='r')
            mm = (img, lab)
            self._mm[cid] = mm
        return mm

    def __getitem__(self, idx):
        sample_name = self.sample_list[idx]
        cid, n = sample_name.split('_slice', 1)
        n = int(n)

        img_vol, lab_vol = self._get_volumes(cid)
        # center(n+1) 한 장만 (H,W) 로 복사 (memmap 은 read-only → ct_normalization in-place 위해 사본 필요)
        image = np.array(img_vol[n + 1])
        label = np.array(lab_vol[n + 1])

        image = ct_normalization(image, **self.hu)

        sample = {'image': image, 'label': label, 'case_name': sample_name}

        if self.transform:
            sample = self.transform(sample)

        return sample
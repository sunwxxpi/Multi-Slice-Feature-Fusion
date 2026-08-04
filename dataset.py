import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
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

# def ct_normalization(image, lower=1017, upper=1801, mean=1222.90087890625, std=132.62820434570312):
def ct_normalization(image, lower=-2.0, upper=1521.0, mean=355.3804931640625, std=282.9181213378906):
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

def load_hu_stats(path):
    # hu_stats_433.json -> ct_normalization 인자 dict
    import json
    with open(path, 'r') as f:
        s = json.load(f)
    return {k: float(s[k]) for k in ('lower', 'upper', 'mean', 'std')}

class COCAVolumeDataset(Dataset):
    """5-fold CV 용 per-case 볼륨 데이터셋.

    image_dir / label_dir 에는 case 당 (D,H,W) .npy 가 있고 memmap 으로 읽는다.
    sample_list 의 각 항목은 `case{gidx:04d}_slice{n:03d}` (n = triplet 시작 인덱스).
    n 번째 sample 은 vol[n:n+3] 의 3채널(prev/center/next)과 center(n+1) 라벨을 반환한다.
    기존 COCA_dataset 과 동일한 (H,W,3) image / (H,W) label 텐서를 만들어
    ct_normalization·Resize·ToTensor 를 그대로 재사용한다.
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
        # (3,H,W) 연속 평면 부분읽기 -> 쓰기 가능 사본 -> (H,W,3) prev/center/next
        image = np.ascontiguousarray(np.transpose(np.array(img_vol[n:n + 3]), (1, 2, 0)))
        label = np.array(lab_vol[n + 1])  # center 슬라이스

        image = ct_normalization(image, **self.hu)

        sample = {'image': image, 'label': label, 'case_name': sample_name}

        if self.transform:
            sample = self.transform(sample)

        return sample
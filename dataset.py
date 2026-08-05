import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from scipy import ndimage
from scipy.ndimage import zoom

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
    num_slices=3 이면 vol[n:n+3] 을 (H,W,3) prev/center/next 로, 1 이면 center(n+1)
    한 장을 (H,W,1) 로 반환한다. 어느 쪽이든 (H,W,C) 규약이라 Resize·ToTensor 를 공유한다.
    """
    def __init__(self, image_dir, label_dir, sample_list, transform=None, hu_stats=None, num_slices=3):
        assert num_slices in (1, 3), f"num_slices 는 1 또는 3 이어야 함 (받은 값: {num_slices})"
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.hu = hu_stats or {}
        self.num_slices = num_slices
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
        # memmap 은 read-only 라 ct_normalization in-place 를 위해 사본이 필요하다.
        # num_slices=1 이어도 (H,W,1) 로 맞춰 transform 을 3채널과 공유한다.
        start = n if self.num_slices == 3 else n + 1
        image = np.ascontiguousarray(
            np.transpose(np.array(img_vol[start:start + self.num_slices]), (1, 2, 0)))
        label = np.array(lab_vol[n + 1])  # center 슬라이스

        image = ct_normalization(image, **self.hu)

        sample = {'image': image, 'label': label, 'case_name': sample_name}

        if self.transform:
            sample = self.transform(sample)

        return sample
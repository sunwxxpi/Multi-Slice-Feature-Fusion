import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Paths and configurations
DATASET_DIR = '/home/psw/nnUNet/data/nnUNet_raw/Dataset001_COCA'
OUTPUT_DIR = '/home/psw/SAU-Net/data/COCA_3frames'
LIST_DIR = os.path.join(OUTPUT_DIR, 'lists_COCA')
os.makedirs(LIST_DIR, exist_ok=True)

# File paths
SPLITS = {
    'train': {
        'ct_path': os.path.join(DATASET_DIR, 'imagesTr'),
        'seg_path': os.path.join(DATASET_DIR, 'labelsTr'),
        'save_path': os.path.join(OUTPUT_DIR, 'train_npz'),
        'list_file': os.path.join(LIST_DIR, 'train.txt')
    },
    'test': {
        'ct_path': os.path.join(DATASET_DIR, 'imagesVal'),
        'seg_path': os.path.join(DATASET_DIR, 'labelsVal'),
        'save_path': os.path.join(OUTPUT_DIR, 'test_npz'),
        'list_file': os.path.join(LIST_DIR, 'test.txt')
    }
}

def process_file(ct_path, seg_path, save_path, list_file, split):
    os.makedirs(save_path, exist_ok=True)
    with open(list_file, 'w') as list_f:
        for ct_file in tqdm(sorted(os.listdir(ct_path)), desc=f'Processing {split}', unit='files'):
            if not ct_file.endswith('.nii.gz'):
                continue
            base_name = ct_file.replace('_0000.nii.gz', '')  # Base name extraction
            case_number = base_name.split('_')[-1].zfill(4)  # Padded case number
            image_path = os.path.join(ct_path, ct_file)
            label_path = os.path.join(seg_path, base_name + '.nii.gz')

            if not os.path.exists(label_path):
                print(f"Label file not found: {label_path}")
                continue

            # Load image and label data
            ct_array = nib.load(image_path).get_fdata().astype(np.float32)
            seg_array = nib.load(label_path).get_fdata().astype(np.uint8)

            # For both train and test, we store 3-slice npz:
            # (H,W,Depth), extract every triplet of slices as (H,W,3) image
            # and central slice as label.
            for slice_idx in range(ct_array.shape[2] - 2):
                image_slices = ct_array[:, :, slice_idx:slice_idx + 3]  # (H,W,3)
                label_slice = seg_array[:, :, slice_idx + 1]  # Center slice label
                npz_filename = os.path.join(save_path, f'case{case_number}_slice{slice_idx:03d}.npz')
                np.savez(npz_filename, image=image_slices, label=label_slice)
                list_f.write(f'case{case_number}_slice{slice_idx:03d}\n')

# Process train and test splits
for split, paths in SPLITS.items():
    process_file(
        ct_path=paths['ct_path'],
        seg_path=paths['seg_path'],
        save_path=paths['save_path'],
        list_file=paths['list_file'],
        split=split
    )
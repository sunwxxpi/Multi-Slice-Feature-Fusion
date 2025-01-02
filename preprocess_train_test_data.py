import os
import numpy as np
import nibabel as nib
import h5py
from tqdm import tqdm

# Paths and configurations
DATASET_DIR = '/home/psw/nnUNet/data/nnUNet_raw/Dataset001_COCA'
OUTPUT_DIR = '/home/psw/SAU-Net/data/COCA'
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
        'save_path': os.path.join(OUTPUT_DIR, 'test_vol_h5'),
        'list_file': os.path.join(LIST_DIR, 'test_vol.txt')
    }
}

# Function to process a single file
def process_file(ct_path, seg_path, save_path, list_file, split):
    with open(list_file, 'w') as list_f:
        for ct_file in tqdm(sorted(os.listdir(ct_path)), desc=f'Processing {split}', unit='files'):
            if not ct_file.endswith('.nii.gz'):
                print(f"Skipping non-NIFTI file: {ct_file}")
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

            # Train split: save slices as NPZ
            if split == 'train':
                os.makedirs(save_path, exist_ok=True)
                for slice_idx in range(ct_array.shape[2]):
                    image_slice = ct_array[:, :, slice_idx]
                    label_slice = seg_array[:, :, slice_idx]
                    npz_filename = os.path.join(save_path, f'case{case_number}_slice{slice_idx:03d}.npz')
                    
                    np.savez(npz_filename, image=image_slice, label=label_slice)
                    list_f.write(f'case{case_number}_slice{slice_idx:03d}\n')
            # Test split: save as HDF5
            elif split == 'test':
                os.makedirs(save_path, exist_ok=True)
                h5_filename = os.path.join(save_path, f'case{case_number}.npy.h5')
                
                with h5py.File(h5_filename, 'w') as hf:
                    hf.create_dataset('image', data=ct_array.transpose(2, 0, 1), compression="gzip", dtype='float32')
                    hf.create_dataset('label', data=seg_array.transpose(2, 0, 1), compression="gzip", dtype='uint8')
                list_f.write(f'case{case_number}\n')

# Process each split
for split, paths in SPLITS.items():
    process_file(
        ct_path=paths['ct_path'],
        seg_path=paths['seg_path'],
        save_path=paths['save_path'],
        list_file=paths['list_file'],
        split=split
    )
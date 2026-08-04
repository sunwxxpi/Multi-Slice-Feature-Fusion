import os
import shutil
from natsort import natsorted

source_dir = "./COCA/COCA_final"
target_dir = "./Dataset001_COCA"

train_images_dir = os.path.join(target_dir, "imagesTr")
train_labels_dir = os.path.join(target_dir, "labelsTr")
val_images_dir = os.path.join(target_dir, "imagesVal")
val_labels_dir = os.path.join(target_dir, "labelsVal")

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# 환자 디렉토리 목록을 번호 순서대로 가져옵니다.
patient_dirs = natsorted([d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))])

# 300명은 Train, 133명은 Validation으로 나눔
train_patients = patient_dirs[:300]
val_patients = patient_dirs[300:433]

def copy_files(patient_dirs, images_dir, labels_dir, prefix):
    for idx, patient in enumerate(patient_dirs, start=1):
        nii_dir = os.path.join(source_dir, patient, 'nii')
        
        if os.path.exists(nii_dir):
            for file_name in os.listdir(nii_dir):
                if file_name.endswith('.nii.gz'):
                    source_file = os.path.join(nii_dir, file_name)
                    
                    if file_name.endswith(".nii.gz") and not file_name.startswith(f"{patient}_label"):
                        # images 파일의 새로운 이름을 생성
                        new_name = f"{prefix}_{patient}_{idx:04d}_0000.nii.gz"
                        dest_file = os.path.join(images_dir, new_name)
                    elif "_label_filled.nii.gz" in file_name:
                        # labels 파일의 새로운 이름을 생성
                        new_name = f"{prefix}_{patient}_{idx:04d}.nii.gz"
                        dest_file = os.path.join(labels_dir, new_name)
                    else:
                        continue
                    
                    shutil.copy(source_file, dest_file)

# Train 및 Validation 파일 복사 및 이름 변경 수행
copy_files(train_patients, train_images_dir, train_labels_dir, "COCA_Tr")
copy_files(val_patients, val_images_dir, val_labels_dir, "COCA_Val")

print("파일 복사가 완료되었습니다.")
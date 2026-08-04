import os
import shutil
import numpy as np
import pydicom
import nibabel as nib
from tqdm import tqdm

def dicom_to_nifti(dicom_dir, output_path):
    # DICOM 디렉토리에서 모든 .dcm 파일을 가져옵니다.
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    
    dicoms = []
    for f in dicom_files:
        dcm = pydicom.dcmread(f)  # DICOM 파일을 읽습니다.
        
        # DICOM 파일이 'ImagePositionPatient' 또는 'PixelSpacing' 속성을 가지고 있는지 확인
        if hasattr(dcm, 'ImagePositionPatient') or hasattr(dcm, 'PixelSpacing'):
            dicoms.append(dcm)
        else:
            print(f"Warning: {f} does not have 'ImagePositionPatient' or 'PixelSpacing' attribute and will be skipped.")
    
    # 유효한 DICOM 파일이 없으면 함수 종료
    if not dicoms:
        print(f"No valid DICOM files found in {dicom_dir}. Skipping NIfTI conversion.")
        return
    
    # 'ImagePositionPatient'의 z 좌표를 기준으로 DICOM 파일을 정렬
    dicoms.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # 볼륨 이미지를 생성할 크기 설정
    image_shape = list(dicoms[0].pixel_array.shape)
    image_shape.append(len(dicoms))  # z 축 크기를 추가
    
    # z 축 간격(z spacing) 계산
    if len(dicoms) > 1:
        z_spacing = dicoms[1].ImagePositionPatient[2] - dicoms[0].ImagePositionPatient[2]
        
        # z spacing이 3이 아니면 경고 출력
        if z_spacing != 3.0:
            print(f"Z spacing of {dicom_dir}: {z_spacing}")
    else:
        z_spacing = 3.0  # 한 장만 있으면 기본 z spacing을 3.0으로 설정
    
    # voxel 크기 설정 (x, y, z)
    voxel_spacing = (dicoms[0].PixelSpacing[0], dicoms[0].PixelSpacing[1], z_spacing)
    
    # DICOM 이미지를 3D 볼륨으로 변환할 배열을 초기화
    volume = np.zeros(image_shape, dtype=dicoms[0].pixel_array.dtype)
    
    # 각 DICOM 파일에서 이미지를 가져와 3D 볼륨에 삽입
    for i, dcm in enumerate(dicoms):
        axial_slice = np.flip(np.rot90(dcm.pixel_array, k=3), axis=0)  # 이미지 회전 및 반전
        volume[:, :, i] = axial_slice
    
    # Affine matrix 생성 (이미지 방향 및 위치 정의)
    orientation = np.array(dcm.ImageOrientationPatient).reshape(2, 3)
    row_cosine = orientation[0]
    col_cosine = orientation[1]
    slice_cosine = np.cross(row_cosine, col_cosine)
    
    affine = np.eye(4)
    affine[0:3, 0] = row_cosine * voxel_spacing[0]
    affine[0:3, 1] = col_cosine * voxel_spacing[1]
    affine[0:3, 2] = slice_cosine * voxel_spacing[2]
    affine[0:3, 3] = dicoms[0].ImagePositionPatient
    
    # NIfTI 파일로 저장
    nifti_img = nib.Nifti1Image(volume, affine)
    nib.save(nifti_img, output_path)

if __name__ == "__main__":
    base_dir = "./COCA/Gated_release_final"
    calcium_xml_dir = os.path.join(base_dir, "calcium_xml")
    patient_dir = os.path.join(base_dir, "patient")
    dataset_final_dir = os.path.join("COCA", "COCA_final")

    os.makedirs(dataset_final_dir, exist_ok=True)
    
    # 각 환자의 XML 파일을 순회하며 작업 처리
    with tqdm(os.listdir(calcium_xml_dir), desc="Processing patients") as pbar:
        for xml_file in pbar:
            if xml_file.endswith(".xml"):
                pbar.set_postfix({"Current File": xml_file})
                
                patient_number = os.path.splitext(xml_file)[0]
                
                patient_final_dir = os.path.join(dataset_final_dir, patient_number)
                os.makedirs(patient_final_dir, exist_ok=True)
                
                dcm_dir = os.path.join(patient_final_dir, "dcm")
                xml_dir = os.path.join(patient_final_dir, "xml")
                nii_dir = os.path.join(patient_final_dir, "nii")
                
                os.makedirs(dcm_dir, exist_ok=True)
                os.makedirs(xml_dir, exist_ok=True)
                os.makedirs(nii_dir, exist_ok=True)
                
                shutil.copy(os.path.join(calcium_xml_dir, xml_file), xml_dir)
                
                patient_dcm_dir = os.path.join(patient_dir, patient_number)
                if os.path.exists(patient_dcm_dir):
                    dicom_found = False  # DICOM 파일이 존재하는지 확인
                    
                    # 환자의 DICOM 파일들을 검색하고 복사
                    for root, dirs, files in os.walk(patient_dcm_dir):
                        for dcm_file in files:
                            if dcm_file.endswith(".dcm"):
                                dicom_found = True
                                shutil.copy(os.path.join(root, dcm_file), dcm_dir)
                    
                    # DICOM 파일이 있으면 NIfTI 파일로 변환
                    if dicom_found:
                        nifti_output_path = os.path.join(nii_dir, f"{patient_number}.nii.gz")
                        dicom_to_nifti(dcm_dir, nifti_output_path)
                    else:
                        print(f"No DICOM files found for patient {patient_number} in {patient_dcm_dir}")

    print("COCA Dataset 구조화 작업 완료!")
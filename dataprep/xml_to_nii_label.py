import os
import numpy as np
import cv2
import pydicom
import nibabel as nib
from xml.etree import ElementTree as ET
from tqdm import tqdm

# DICOM 파일 읽기
def load_dicom_images(dicom_dir):
    # DICOM 디렉토리에서 모든 .dcm 파일을 읽어서 slices 리스트에 저장
    slices = [pydicom.dcmread(os.path.join(dicom_dir, f)) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    # 이미지의 z 축 위치를 기준으로 정렬
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # 각 slice의 픽셀 데이터를 스택으로 쌓아 3D 이미지 데이터 생성
    image_data = np.stack([s.pixel_array for s in slices])
    
    return image_data, slices

# XML 파일 파싱
def parse_xml(xml_file):
    # XML 파일을 파싱하여 트리 구조 생성
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # XML 구조에서 이미지와 ROI 정보를 담고 있는 배열을 찾음
    images_dict = root.find('dict')
    images_array = images_dict.find('array')
    
    rois = []
    
    for image_dict in images_array.findall('dict'):
        image_index = None
        roi_name = None
        rois_for_image = []
        
        for i, elem in enumerate(list(image_dict)):
            # 이미지 인덱스를 가져옴
            if elem.tag == 'key' and elem.text == 'ImageIndex':
                image_index = int(list(image_dict)[i + 1].text)
                
            # 이미지에 포함된 ROI 정보를 가져옴
            if elem.tag == 'key' and elem.text == 'ROIs':
                rois_array = list(image_dict)[i + 1]
                
                for roi_dict in rois_array.findall('dict'):
                    points_mm = []
                    roi_name = None
                    
                    for j, roi_elem in enumerate(list(roi_dict)):
                        # ROI의 좌표를 mm 단위로 가져옴
                        if roi_elem.tag == 'key' and roi_elem.text == 'Point_mm':
                            points_array = list(roi_dict)[j + 1]
                            points_mm = [tuple(map(float, p.text.strip('()').split(','))) for p in points_array.findall('string')]
                        
                        # ROI의 이름을 가져옴
                        if roi_elem.tag == 'key' and roi_elem.text == 'Name':
                            roi_name = list(roi_dict)[j + 1].text

                    # ROI의 좌표와 이름이 존재하는 경우 리스트에 추가
                    if points_mm and roi_name:
                        rois_for_image.append((points_mm, roi_name))

        # 인덱스와 ROI를 리스트에 추가
        if image_index is not None:
            rois.append((image_index, rois_for_image))
    
    return rois

# ROI 좌표를 연결하여 다각형을 채우는 함수
def fill_roi(image_shape, roi_pixels):
    filled_image = np.zeros(image_shape, dtype=np.uint8)
    
    # 각 슬라이스 별로 처리
    for slice_index in range(image_shape[0]):
        # 해당 슬라이스에 속하는 픽셀 좌표를 가져옴
        slice_pixels = [(col, row) for idx, row, col in roi_pixels if idx == slice_index]
        
        # 최소 3개의 점이 있어야 다각형을 그릴 수 있음
        if len(slice_pixels) >= 3:
            # 좌표들을 NumPy 배열로 변환
            contour = np.array(slice_pixels, dtype=np.int32)
            
            # 다각형의 내부를 채움
            cv2.fillPoly(filled_image[slice_index], [contour], 255)
    
    return filled_image

# ROI 좌표를 DICOM 이미지의 픽셀 좌표로 변환
def convert_mm_to_pixel(points_mm, slices, image_data):
    pixel_coords = []
    
    for point in points_mm:
        point = np.array(point)
        # z 위치가 가장 가까운 slice를 찾음
        slice_index = min(range(len(slices)), key=lambda i: abs(slices[i].ImagePositionPatient[2] - point[2]))
        slice = slices[slice_index]
        
        # 이미지의 방향 및 위치, 픽셀 간격을 가져옴
        image_orientation = np.array(slice.ImageOrientationPatient, dtype=np.float64)
        image_position = np.array(slice.ImagePositionPatient, dtype=np.float64)
        pixel_spacing = np.array(slice.PixelSpacing, dtype=np.float64)
        
        # DICOM 이미지의 행과 열 방향 코사인 벡터
        row_cosine = image_orientation[0:3]
        col_cosine = image_orientation[3:6]
        
        # 물리 좌표를 픽셀 좌표로 변환
        row = np.dot(point - image_position, row_cosine) / pixel_spacing[1]
        col = np.dot(point - image_position, col_cosine) / pixel_spacing[0]
        
        # NIfTI 좌표계 변환 (좌표계 차이에 따른 변환 수행)
        row = image_data.shape[1] - 1 - row  # DICOM의 row 좌표를 NIfTI의 y축으로 변환
        col = image_data.shape[2] - 1 - col  # DICOM의 col 좌표를 NIfTI의 x축으로 변환

        # 변환된 픽셀 좌표를 리스트에 추가
        pixel_coords.append((slice_index, int(round(row)), int(round(col))))
    
    return pixel_coords

# NIfTI 파일 생성 및 저장 (개별 처리된 ROI들을 결합)
def create_label_nii(image_shape, all_roi_pixels, output_file, voxel_spacing, origin, image_orientation, fill=False):
    # 레이블 데이터를 위한 빈 3D 배열 생성
    label_data = np.zeros(image_shape, dtype=np.int16)
    
    for roi_pixels, roi_name in all_roi_pixels:
        roi_label = np.zeros(image_shape, dtype=np.int16)
        roi_value = label_value_from_name(roi_name)  # ROI 이름에 따라 라벨 값을 설정
        
        if fill:
            # ROI를 다각형으로 채움
            filled_image = fill_roi(image_shape, roi_pixels)
            # 채워진 ROI 이미지에 해당 라벨 값을 적용 (0이 아닌 경우에만)
            roi_label = (filled_image > 0).astype(np.int16) * roi_value  # 0이 아닌 값에 대해만 roi_value를 곱함
        else:
            # 각 슬라이스에서 ROI 위치에 라벨 값 할당 (fill=False일 때)
            for slice_index, row, col in roi_pixels:
                roi_label[slice_index, row, col] = roi_value

        # 최대값을 사용해 각 ROI의 라벨 데이터를 결합 (다른 ROI와 중복되는 경우를 방지)
        label_data = np.maximum(label_data, roi_label)

    # NIfTI 형식에 맞게 데이터 축 순서를 변경
    label_data = np.transpose(label_data, (1, 2, 0))  # (slice, row, col) -> (row, col, slice)
    
    # NIfTI 헤더를 위한 affine 행렬 생성
    row_cosine = np.array(image_orientation[:3], dtype=np.float64)
    col_cosine = np.array(image_orientation[3:], dtype=np.float64)
    slice_cosine = np.cross(row_cosine, col_cosine)
    
    affine = np.eye(4)
    affine[0:3, 0] = row_cosine * float(voxel_spacing[0])
    affine[0:3, 1] = col_cosine * float(voxel_spacing[1])
    affine[0:3, 2] = slice_cosine * float(voxel_spacing[2])
    affine[0:3, 3] = np.array(origin, dtype=np.float64)
    
    # NIfTI 파일로 저장
    nii_img = nib.Nifti1Image(label_data, affine)
    nii_img.header.set_zooms(voxel_spacing)
    nib.save(nii_img, output_file)

# ROI 이름에 따라 라벨 값을 반환
def label_value_from_name(roi_name):
    if roi_name == 'Left Coronary Artery':
        return 1
    elif roi_name == 'Left Anterior Descending Artery':
        return 2
    elif roi_name == 'Left Circumflex Artery':
        return 3
    elif roi_name == 'Right Coronary Artery':
        return 4
    else:
        return 0  # Unknown or unlabelled

if __name__ == '__main__':
    root_dir = './COCA/COCA_final'
    
    with tqdm(os.listdir(root_dir), desc="Processing patients") as pbar:
        for patient_id in pbar:
            pbar.set_postfix({"Current Patient": patient_id})
            
            patient_dir = os.path.join(root_dir, patient_id)
            
            if os.path.isdir(patient_dir):
                dicom_dir = os.path.join(patient_dir, 'dcm')
                xml_file = os.path.join(patient_dir, 'xml', f'{patient_id}.xml')
                nii_output_dir = os.path.join(patient_dir, 'nii')
                
                # DICOM 및 XML 파일이 존재하는지 확인
                if os.path.exists(dicom_dir) and os.path.exists(xml_file):
                    os.makedirs(nii_output_dir, exist_ok=True)
                    
                    # 1. DICOM 이미지 로드
                    image_data, slices = load_dicom_images(dicom_dir)

                    # 2. XML 파일 파싱
                    rois = parse_xml(xml_file)

                    # 3. ROI를 이미지 픽셀 좌표로 변환
                    all_roi_pixels = []
                    for image_index, rois_for_image in rois:
                        for roi_points_mm, roi_name in rois_for_image:
                            roi_pixels = convert_mm_to_pixel(roi_points_mm, slices, image_data)
                            all_roi_pixels.append((roi_pixels, roi_name))

                    # NIfTI 파일 생성을 위한 DICOM 파일의 voxel spacing 및 origin 정보 추출
                    voxel_spacing = np.array([
                                        float(slices[0].PixelSpacing[0]), 
                                        float(slices[0].PixelSpacing[1]), 
                                        abs(slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2])
                                        ])
                    origin = np.array(slices[0].ImagePositionPatient, dtype=np.float64)
                    image_orientation = np.array(slices[0].ImageOrientationPatient, dtype=np.float64)
                    
                    # 4. NIfTI 라벨 파일 생성 및 저장 (두 가지 버전: fill O, fill X)
                    nii_output_path_filled = os.path.join(nii_output_dir, f'{patient_id}_label_filled.nii.gz')
                    nii_output_path_unfilled = os.path.join(nii_output_dir, f'{patient_id}_label_unfilled.nii.gz')
                    
                    create_label_nii(image_data.shape, all_roi_pixels, nii_output_path_filled, voxel_spacing, origin, image_orientation, fill=True)
                    create_label_nii(image_data.shape, all_roi_pixels, nii_output_path_unfilled, voxel_spacing, origin, image_orientation, fill=False)
                else:
                    print(f"Skipping patient: {patient_id} (Missing DICOM or XML files)")
                    
    print("XML to NIfTI Label 작업 완료!")
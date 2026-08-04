import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pydicom
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
        rois_for_image = []
        
        for elem in image_dict:
            # 이미지 인덱스를 가져옴
            if elem.tag == 'key' and elem.text == 'ImageIndex':
                image_index = int(next(image_dict.iter('integer')).text)
                
            # 이미지에 포함된 ROI 정보를 가져옴
            if elem.tag == 'key' and elem.text == 'ROIs':
                rois_array = next(image_dict.iter('array'))
                
                for roi_dict in rois_array.findall('dict'):
                    points_mm = []
                    
                    for roi_elem in roi_dict:
                        # ROI의 좌표를 mm 단위로 가져옴
                        if roi_elem.tag == 'key' and roi_elem.text == 'Point_mm':
                            points_array = next(roi_dict.iter('array'))
                            points_mm = [tuple(map(float, p.text.strip('()').split(','))) for p in points_array.iter('string')]

                    rois_for_image.append(points_mm)
        
        # 인덱스와 ROI를 리스트에 추가
        if image_index is not None:
            rois.append((image_index, rois_for_image))
    
    return rois

# ROI 좌표를 DICOM 이미지의 픽셀 좌표로 변환
def convert_mm_to_pixel(points_mm, slices):
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
        pixel_coords.append((slice_index, int(round(row)), int(round(col))))
    
    return pixel_coords

# DICOM 이미지와 라벨을 겹쳐서 PNG 파일로 저장
def save_slices_with_labels(image_data, all_roi_pixels, dicom_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    total_slices = image_data.shape[0]  # 전체 슬라이스 개수
    
    # 사용할 색상 목록 (matplotlib의 Tableau 색상 팔레트 사용)
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for slice_index in range(total_slices):
        # 현재 슬라이스 이미지 가져오기
        slice_img = image_data[slice_index]
        
        # 이미지 좌우 반전
        slice_img = np.fliplr(slice_img)

        # 슬라이스 내의 여러 ROI를 다른 색상으로 시각화
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(slice_img, cmap='gray')
        
        roi_count = 0  # 슬라이스 내에서 ROI 색상을 순환시키기 위한 변수

        for roi_coords_list in all_roi_pixels:
            for roi_coords in roi_coords_list:
                if roi_coords and roi_coords[0][0] == slice_index:  # 현재 슬라이스에 해당하는 ROI만 시각화
                    label_overlay = np.zeros_like(slice_img, dtype=np.uint8)
                    
                    for slice, row, col in roi_coords:
                        label_overlay[row, col] = 255  # 라벨 부분을 흰색으로 표시

                    # ROI 시각화
                    label_overlay = np.transpose(np.fliplr(label_overlay)[::-1, ::-1])
                    ax.contour(label_overlay, levels=[0.5], colors=[colors[roi_count % len(colors)]], alpha=0.5)
                    
                    roi_count += 1  # 다음 ROI에 다른 색상 적용
        
        ax.axis('off')  # 축을 숨김

        # DICOM 파일 이름을 기반으로 저장 파일 이름 생성
        dicom_filename = os.path.basename(dicom_files[slice_index])
        dicom_basename, dicom_ext = os.path.splitext(dicom_filename)

        # 슬라이스를 PNG로 저장
        output_file = os.path.join(output_dir, f'{dicom_basename}-{slice_index:03d}.png')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == '__main__':
    root_dir = './COCA/COCA_final'
    
    with tqdm(os.listdir(root_dir), desc="Processing dirs") as pbar:
        for dir_name in pbar:
            pbar.set_postfix({"Current Dir": dir_name})
            
            dir_path = os.path.join(root_dir, dir_name)
            
            if os.path.isdir(dir_path):
                dicom_dir = os.path.join(dir_path, 'dcm')
                xml_file = os.path.join(dir_path, 'xml', f'{dir_name}.xml')
                output_dir = dicom_dir
                
                if os.path.exists(dicom_dir) and os.path.exists(xml_file):
                    # 1. DICOM 이미지 로드
                    image_data, slices = load_dicom_images(dicom_dir)

                    # 2. XML 파일 파싱
                    rois = parse_xml(xml_file)

                    # 3. ROI를 이미지 픽셀 좌표로 변환
                    all_roi_pixels = []
                    for image_index, rois_for_image in rois:
                        roi_pixels_for_image = []
                        
                        for roi_points_mm in rois_for_image:
                            roi_pixels = convert_mm_to_pixel(roi_points_mm, slices)
                            roi_pixels_for_image.append(roi_pixels)
                            
                        all_roi_pixels.append(roi_pixels_for_image)

                    # 4. 슬라이스별로 DICOM 이미지와 라벨을 겹쳐서 저장
                    save_slices_with_labels(image_data, all_roi_pixels, [s.filename for s in slices], output_dir)
                else:
                    print(f"Skipping directory: {dir_name} (Missing DICOM or XML files)")
                    
    print("XML to PNG Label 작업 완료!")
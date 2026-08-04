import pydicom

# DICOM 파일을 읽습니다.
file_path = 'COCA/Gated_release_final/patient/159/Pro_Gated_Calcium_Score_(CS)_3.0_Qr36_2_BestDiast_71_%/IM-3255-0021.dcm'
dicom_data = pydicom.dcmread(file_path)

# ImagePositionPatient 태그가 없다면 추가합니다.
if 'ImagePositionPatient' not in dicom_data:
    dicom_data.ImagePositionPatient = [-65.83203125, -256.33203125, -182]

# PixelSpacing 태그가 없다면 추가합니다.
if 'PixelSpacing' not in dicom_data:
    dicom_data.PixelSpacing = [0.3359375, 0.3359375]

# 수정된 DICOM 파일을 저장합니다.
output_file_path = 'COCA/Gated_release_final/patient/159/Pro_Gated_Calcium_Score_(CS)_3.0_Qr36_2_BestDiast_71_%/IM-3255-0021_modified.dcm'
dicom_data.save_as(output_file_path)

print(f"Modified DICOM file saved to {output_file_path}")
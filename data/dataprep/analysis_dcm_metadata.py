import pydicom

dicom_file_path = 'COCA/COCA_final/38/dcm/IM-5430-0025.dcm'
dicom_data = pydicom.dcmread(dicom_file_path)

# print(dicom_data)

if 'ImagePositionPatient' in dicom_data:
    print(f"Image Position (Patient): {dicom_data.ImagePositionPatient}") # Image Position (Patient): [-65.83203125, -256.33203125, -182]

if 'PixelSpacing' in dicom_data:
    print(f"Pixel Spacing: {dicom_data.PixelSpacing}")  # Pixel Spacing: [0.3359375, 0.3359375]
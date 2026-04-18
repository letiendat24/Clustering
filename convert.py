import os
import nibabel as nib
import tifffile as tiff
import numpy as np

def convert_mnc_to_tif(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".mnc.gz"):
            mnc_path = os.path.join(input_dir, filename)
            tif_filename = filename.replace(".mnc.gz", ".tif")
            tif_path = os.path.join(output_dir, tif_filename)

            try:
                img = nib.load(mnc_path)
                data = img.get_fdata()

                if data.ndim == 3:
                    data = np.transpose(data, (2, 0, 1))

                data = data.astype(np.float32)

                tiff.imwrite(tif_path, data, imagej=True)
                print(f"[THÀNH CÔNG] Đã xuất file: {tif_filename}")

            except Exception as e:
                print(f"[LỖI] Không thể xử lý {filename}: {e}")

# Các thành viên đổi đường dẫn này trỏ tới thư mục chứa data trên máy cá nhân
INPUT_FOLDER = './data/medical_mri/raw_mnc'       
OUTPUT_FOLDER = './data/medical_mri/processed_tif' 

if __name__ == "__main__":
    convert_mnc_to_tif(INPUT_FOLDER, OUTPUT_FOLDER)
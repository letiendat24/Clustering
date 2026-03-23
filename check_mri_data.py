import nibabel as nib
import numpy as np

# 1. Khai báo đường dẫn đến 2 file dữ liệu
# (Sử dụng đúng đường dẫn đã chạy thành công ở main_runner)
img_path = 'dataset/MRI/t1_icbm_normal_1mm_pn3_rf20.mnc'
label_path = 'dataset/MRI/label_t1_icbm_normal_1mm_pn3_rf20.mnc'

print("Đang đọc dữ liệu MRI...")
try:
    img_data = nib.load(img_path).get_fdata()
    label_data = nib.load(label_path).get_fdata()
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file. Hãy kiểm tra lại đường dẫn!")
    exit()

# 2. Phân tích Ảnh gốc (Input)
print("\n" + "="*50)
print("PHẦN 1: THÔNG TIN ẢNH GỐC (ẢNH NHIỄU CẦN PHÂN CỤM)")
print("="*50)
print(f"- Kích thước khối 3D (X, Y, Z) : {img_data.shape}")
print(f"- Tổng số điểm ảnh (Voxels)    : {img_data.size:,}")
print(f"- Cường độ sáng nhỏ nhất (Min) : {np.min(img_data)}")
print(f"- Cường độ sáng lớn nhất (Max) : {np.max(img_data)}")
print(f"- Cường độ sáng trung bình     : {np.mean(img_data):.2f}")
print(f"- Kiểu dữ liệu (Data type)     : {img_data.dtype}")

# 3. Phân tích file Nhãn (Ground Truth)
print("\n" + "="*50)
print("PHẦN 2: THÔNG TIN FILE NHÃN (GROUND TRUTH)")
print("="*50)
print(f"- Kích thước khối 3D (X, Y, Z) : {label_data.shape}")
print(f"- Các giá trị nhãn đang có     : {np.unique(label_data)}")
print(f"- Tổng số loại mô (Số cụm)     : {len(np.unique(label_data))}")

# 4. Phân tích Lát cắt thực tế đưa vào thuật toán (Lát số 90)
slice_idx = 90
slice_90 = img_data[:, :, slice_idx]

print("\n" + "="*50)
print(f"PHẦN 3: THÔNG TIN LÁT CẮT SỐ {slice_idx} (ĐƯA VÀO CHẠY THUẬT TOÁN)")
print("="*50)
print(f"- Kích thước ma trận 2D        : {slice_90.shape}")
print(f"- Kích thước sau khi Flatten   : {slice_90.reshape(-1, 1).shape}")
print("="*50 + "\n")
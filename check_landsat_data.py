import rasterio
import numpy as np
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

band_paths = [
    r"dataset/Landsat/hn-mini_8_SR_B2.tif", # Band 2: Blue
    r"dataset/Landsat/hn-mini_8_SR_B3.tif", # Band 3: Green
    r"dataset/Landsat/hn-mini_8_SR_B4.tif", # Band 4: Red
    r"dataset/Landsat/hn-mini_8_SR_B5.tif"  # Band 5: Near Infrared (NIR)
]
label_path = r"dataset/Landsat/pseudo_label_hn-mini_8.tif"

print("Đang đọc dữ liệu vệ tinh Landsat...")
try:
    bands_data = []
    meta = None
    # Đọc 4 band ảnh
    for i, p in enumerate(band_paths):
        with rasterio.open(p) as src:
            bands_data.append(src.read(1).astype('float32'))
            if i == 0:
                meta = src.meta.copy()
    
    # Gộp 4 band lại thành 1 khối không gian (H, W, C)
    img_data = np.stack(bands_data, axis=-1)
    
    # Đọc file nhãn
    with rasterio.open(label_path) as src:
        label_data = src.read(1)
        label_meta = src.meta
        
except rasterio.errors.RasterioIOError:
    print("Lỗi: Không tìm thấy file. Hãy kiểm tra lại đường dẫn .tif!")
    exit()

# 2. Phân tích Ảnh gốc (Input)
print("\n" + "="*50)
print("PHẦN 1: THÔNG TIN ẢNH ĐA PHỔ LANDSAT (INPUT)")
print("="*50)
print(f"- Kích thước khối ảnh (H, W, Bands) : {img_data.shape}")
h, w, c = img_data.shape
print(f"- Tổng số điểm ảnh (Pixels)         : {h * w:,}")
print(f"- Số lượng đặc trưng (Kênh màu)     : {c} (B2, B3, B4, B5)")
print(f"- Cường độ nhỏ nhất (Min toàn ảnh)  : {np.min(img_data)}")
print(f"- Cường độ lớn nhất (Max toàn ảnh)  : {np.max(img_data)}")
print(f"- Hệ tọa độ (CRS)                   : {meta['crs']}")
print(f"- Kiểu dữ liệu (Data type)          : {img_data.dtype}")

# 3. Phân tích file Nhãn (Ground Truth/Pseudo Label)
print("\n" + "="*50)
print("PHẦN 2: THÔNG TIN FILE NHÃN (PSEUDO LABEL)")
print("="*50)
print(f"- Kích thước ma trận nhãn (H, W)    : {label_data.shape}")
print(f"- Các giá trị nhãn đang có          : {np.unique(label_data)}")
print(f"- Tổng số loại lớp phủ (Số cụm)     : {len(np.unique(label_data))}")

# 4. Phân tích Ma trận đưa vào thuật toán
X = img_data.reshape(-1, c)

print("\n" + "="*50)
print("PHẦN 3: MA TRẬN ĐẦU VÀO THUẬT TOÁN FCM")
print("="*50)
print(f"- Kích thước ma trận 2D gốc         : ({h}, {w})")
print(f"- Kích thước sau khi Flatten (X)    : {X.shape}")
print(f"- Giải thích: Mỗi điểm ảnh giờ đây là một vector chứa {c} giá trị.")
print("="*50 + "\n")
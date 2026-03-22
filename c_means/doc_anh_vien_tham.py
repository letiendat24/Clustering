import os
import time
import numpy as np
import cv2
from PIL import Image

# -----------------------------------------
# 1️⃣ Đọc ảnh
# -----------------------------------------
def load_any_image(image_path, resize=None):
    """
    Đọc ảnh PNG/JPG hoặc GeoTIFF, luôn trả về (H, W, C)
    """
    t0 = time.time()
    ext = os.path.splitext(image_path)[1].lower()

    if ext in [".png", ".jpg", ".jpeg"]:
        # Dùng PIL cho JPG/PNG, ép RGB
        img = np.array(Image.open(image_path).convert("RGB"))
    else:
        # Dùng rasterio cho GeoTIFF
        import rasterio
        with rasterio.open(image_path) as src:
            img = src.read()  # (bands, H, W)
            img = np.transpose(img, (1, 2, 0))  # (H, W, bands)
            # Nếu nhiều hơn 3 kênh, lấy 3 kênh đầu
            if img.shape[2] > 3:
                img = img[:, :, :3]
            # Nếu chỉ 1 kênh → nhân 3 lần để thành RGB
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)

    # Resize nếu cần
    if resize:
        img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)

    dt = time.time() - t0
    print(f"Đọc dữ liệu: {img.shape[0]*img.shape[1]:,} pixel | {img.shape[-1]} kênh | {dt:.3f}s")
    return img, img.shape

# -----------------------------------------
# 2️⃣ Chuẩn hóa và reshape ảnh cho clustering
# -----------------------------------------
def normalize_and_reshape(img):
    """
    Chuẩn hóa ảnh [0,1] và reshape về (N_pixel, C)
    """
    if img.ndim == 2:
        img = img[:, :, None]
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    reshaped = img.reshape(-1, img.shape[-1])
    return reshaped, reshaped.shape[0]

# -----------------------------------------
# 3️⃣ Đọc nhãn
# -----------------------------------------

# Hàm này thì đọc được ảnh nhãn .png của landcoverai rồi nhưng đọc ảnh đen trắng lại trả về 180 lớp là không đúng, trong khi hàm ở dưới thì ngược lại
# def load_label_image(label_path, target_shape, color_map=None):
#     """
#     Đọc ảnh nhãn (.png, .jpg, .tif)
#     Nếu color_map: chuyển RGB -> class_id
#     """
#     ext = os.path.splitext(label_path)[1].lower()
#     label = None

#     if ext in [".png", ".jpg", ".jpeg"]:
#         label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
#         if label is None:
#             print(f"[LỖI] Không đọc được nhãn: {label_path}")
#             return None
#         if len(label.shape) == 3:
#             # Nếu có color_map, map RGB → class_id
#             if color_map is not None:
#                 label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
#                 label_id = np.zeros(label.shape[:2], dtype=np.uint8)
#                 for rgb, idx in color_map.items():
#                     mask = np.all(label == np.array(rgb), axis=-1)
#                     label_id[mask] = idx
#                 label = label_id
#             else:
#                 # không map → grayscale
#                 label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

#     elif ext in [".tif", ".tiff"]:
#         import rasterio
#         with rasterio.open(label_path) as src:
#             label = src.read(1)

#     # Resize về target_shape
#     if label.shape[0:2] != target_shape[0:2]:
#         label = cv2.resize(label, (target_shape[1], target_shape[0]),
#                            interpolation=cv2.INTER_NEAREST)

#     return label.reshape(-1).astype(np.int32)
# def load_label_image(label_path, target_shape):
#     import cv2
#     import numpy as np

#     ext = os.path.splitext(label_path)[1].lower()
    
#     # Đọc ảnh
#     if ext in [".png", ".jpg", ".jpeg"]:
#         label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
#         if label is None:
#             print(f"[LỖI] Không đọc được nhãn: {label_path}")
#             return None
#         # Nếu 3 kênh, so sánh với ngưỡng để tạo nhãn 0/1
#         if len(label.shape) == 3:
#             # Chỉ lấy kênh đầu hoặc chuyển về grayscale trung bình
#             gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
#             # So với ngưỡng 128 → 0/1
#             label = np.where(gray > 128, 1, 0).astype(np.uint8)
#         else:
#             # 1 kênh: chỉ cần >128 →1, <=128 →0
#             label = np.where(label > 128, 1, 0).astype(np.uint8)

#     elif ext in [".tif", ".tiff"]:
#         import rasterio
#         with rasterio.open(label_path) as src:
#             label = src.read(1)
#         label = label.astype(np.uint8)

#     # Resize về target_shape
#     if label.shape[0:2] != target_shape[0:2]:
#         label = cv2.resize(label, (target_shape[1], target_shape[0]),
#                            interpolation=cv2.INTER_NEAREST)

#     return label.reshape(-1).astype(np.int32)




def load_label_image(label_path, target_shape, color_map=None, binary_threshold=128):
    """
    Đọc ảnh nhãn (.png, .jpg, .tif) và trả về mảng 1 chiều (N_pixel,).
    
    - color_map: dict RGB -> class_id, dùng cho ảnh RGB nhiều lớp.
    - binary_threshold: nếu ảnh nhị phân (đen-trắng) thì ngưỡng để map 0/1.
    
    Tự động xử lý:
    - Nhãn nhị phân (2 lớp) -> [0,1]
    - Nhãn đa lớp -> map tất cả giá trị khác nhau về 0..K-1
    """
    import cv2
    import numpy as np
    import os

    ext = os.path.splitext(label_path)[1].lower()
    label = None

    # 1️⃣ Đọc ảnh
    if ext in [".png", ".jpg", ".jpeg"]:
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label is None:
            print(f"[LỖI] Không đọc được nhãn: {label_path}")
            return None

        # Nếu RGB
        if len(label.shape) == 3:
            if color_map is not None:
                # map RGB -> class_id
                label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
                label_id = np.zeros(label.shape[:2], dtype=np.uint8)
                for rgb, idx in color_map.items():
                    mask = np.all(label == np.array(rgb), axis=-1)
                    label_id[mask] = idx
                label = label_id
            else:
                # RGB -> grayscale
                gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
                # Nếu chỉ 2 giá trị khác nhau -> nhị phân
                unique_vals = np.unique(gray)
                if len(unique_vals) == 2:
                    min_val, max_val = unique_vals
                    label = np.where(gray == min_val, 0, 1)
                else:
                    label = gray

        else:
            # 1 kênh
            unique_vals = np.unique(label)
            if len(unique_vals) == 2:
                min_val, max_val = unique_vals
                label = np.where(label == min_val, 0, 1)
            else:
                label = label

    elif ext in [".tif", ".tiff"]:
        import rasterio
        with rasterio.open(label_path) as src:
            label = src.read(1)
        label = label.astype(np.uint8)

    # 2️⃣ Resize về target_shape
    if label.shape[0:2] != target_shape[0:2]:
        label = cv2.resize(label, (target_shape[1], target_shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    # 3️⃣ Chuẩn hóa nhãn về 0..K-1
    unique_vals = np.unique(label)
    if len(unique_vals) == 2:
        # nhị phân
        min_val, max_val = unique_vals
        label = np.where(label == min_val, 0, 1)
    else:
        # đa lớp
        label_map = {val: idx for idx, val in enumerate(unique_vals)}
        for val, idx in label_map.items():
            label[label == val] = idx

    return label.reshape(-1).astype(np.int32)


# -----------------------------------------
# 4️⃣ Gộp tất cả
# -----------------------------------------
def load_and_prepare_data(image_path, label_path, resize=(128, 128)):
    """
    Đọc ảnh, đọc nhãn, chuẩn hóa, reshape
    """
    t0 = time.time()
    img, shape = load_any_image(image_path, resize)
    if img is None:
        return None, None, None, 0

    data, n_samples = normalize_and_reshape(img)
    labels = load_label_image(label_path, shape)
    if labels is None:
        return None, None, None, 0

    dt = time.time() - t0
    print(f"Chuẩn bị dữ liệu xong: {n_samples:,} pixel | {img.shape[-1]} kênh | {dt:.3f}s")
    return data, labels, n_samples, dt

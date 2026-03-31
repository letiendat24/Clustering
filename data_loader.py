import numpy as np
import nibabel as nib
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter
from skimage import morphology
from sklearn.preprocessing import LabelEncoder
import warnings
from rasterio.errors import NotGeoreferencedWarning
from skimage.transform import resize

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# =====================================================================
# PHẦN 1: CÁC HÀM TIỀN XỬ LÝ ẢNH MRI (TEAM'S FUNCTIONS)
# =====================================================================
THRESHOLD = 0.05
SIGMA = 0.
DATA_SHAPE = None

def normalize_minmax(img, epsilon=1e-8):
    """Chuẩn hóa min max về [0;1]"""
    return (img - np.min(img)) / (np.max(img) - np.min(img) + epsilon)

def brain_mask_threshold(img, epsilon=1e-8, threshold=THRESHOLD):
    """Loại bỏ nền, giữ lại > 0,05"""
    norm = normalize_minmax(img, epsilon)
    return norm * (norm > threshold)

def denoise_image(img, sigma=SIGMA):
    """Lọc ảnh bằng gaussian """
    return gaussian_filter(img, sigma=sigma)

def skull_stripping(img, threshold=THRESHOLD):
    """Xóa nền, lọc nhiễu nhỏ, lấp lỗ trống"""
    binary = img > threshold
    cleaned = morphology.remove_small_objects(binary, min_size=50)
    cleaned = morphology.binary_closing(cleaned, morphology.disk(3))
    return img * cleaned

def normalize_zscore(img):
    """Chuẩn hóa Z-score"""
    mean = np.mean(img)
    std = np.std(img) + 1e-8 
    return (img - mean) / std

def read_mri_slice(path: str, slice_index: int, axis: int, is_label=False) -> np.ndarray:
    """Đọc ảnh theo mặt cắt """
    img = nib.load(path)
    data = img.get_fdata()
    global DATA_SHAPE
    DATA_SHAPE = data.shape
    
    if axis == 0:
        slice_img = data[slice_index, :, :]
    elif axis == 1:
        slice_img = data[:, slice_index, :]
    elif axis == 2:
        slice_img = data[:, :, slice_index]
    else:
        raise ValueError("Axis phải là 0, 1 hoặc 2")
    return np.rot90(slice_img.T)

def load_mri_pipeline(img_path, label_path, slice_index=90, axis=2, target_size=None):
    """
    HÀM ĐÓNG GÓI CHO MAIN_RUNNER:
    Áp dụng kỹ thuật Resize & Anti-aliasing giống Landsat để đạt điểm cực cao.
    """
    EPSILON = 1e-8
    
    # ==========================================
    # BƯỚC 1: ĐỌC VÀ TIỀN XỬ LÝ CƠ BẢN
    # ==========================================
    slice_img = read_mri_slice(img_path, slice_index, axis)
    normalized = normalize_minmax(slice_img, EPSILON)
    masked = brain_mask_threshold(normalized, EPSILON)
    denoised = denoise_image(masked, sigma=0.5)
    final_img = skull_stripping(denoised, threshold=0.05)

    # Đọc nhãn gốc (2D)
    true_labels_raw = read_mri_slice(label_path, slice_index, axis, is_label=True)

    # ==========================================
    # Nếu trong main_runner có truyền target_size vào thì mới resize
    if target_size is not None:
        # Resize ảnh (order=1: bilinear, có anti_aliasing để diệt nhiễu)
        final_img = resize(
            final_img, 
            output_shape=target_size, 
            order=1, 
            mode='reflect', 
            anti_aliasing=True
        )
        
        # Resize nhãn (order=0: nearest, KHÔNG anti_aliasing để giữ nguyên số nguyên 0,1,2,3...)
        true_labels_raw = resize(
            true_labels_raw, 
            output_shape=target_size, 
            order=0, 
            mode='edge', 
            anti_aliasing=False
        )
        # Ép kiểu nhãn về số nguyên cho chắc chắn
        true_labels_raw = np.round(true_labels_raw).astype(int)

    # ==========================================
    # BƯỚC 3: DUỖI ẢNH VÀ ENCODE NHÃN
    # ==========================================
    # Duỗi ảnh thành (N, 1)
    X = final_img.flatten().reshape(-1, 1)
    
    # Duỗi nhãn thành 1D
    true_labels_flat = true_labels_raw.flatten().astype(int)
    
    le = LabelEncoder()
    true_labels_mapped = le.fit_transform(true_labels_flat)
    n_clusters = len(np.unique(true_labels_mapped))
    
    return X, true_labels_mapped, n_clusters, final_img.shape
    """
    HÀM ĐÓNG GÓI CHO MAIN_RUNNER:
    Thực thi toàn bộ pipeline đọc ảnh, tiền xử lý và chuyển thành ma trận X cho thuật toán.
    """
    EPSILON = 1e-8
    
    # 1. Đọc ảnh và tiền xử lý
    slice_img = read_mri_slice(img_path, slice_index, axis)
    normalized = normalize_minmax(slice_img, EPSILON)
    masked = brain_mask_threshold(normalized, EPSILON)
    denoised = denoise_image(masked, sigma=0.5)
    # Nếu muốn dùng skull_stripping thì mở comment dòng dưới:
    final_img = skull_stripping(denoised, threshold=0.1)
    final_img = denoised

    # Duỗi ảnh thành đầu vào X
    X = final_img.flatten().reshape(-1, 1)

    # 2. Đọc nhãn tương ứng và remap
    true_labels_raw = read_mri_slice(label_path, slice_index, axis, is_label=True)
    true_labels_flat = true_labels_raw.flatten().astype(int)
    
    le = LabelEncoder()
    true_labels_mapped = le.fit_transform(true_labels_flat)
    
    n_clusters = len(np.unique(true_labels_mapped))
    
    return X, true_labels_mapped, n_clusters, final_img.shape

# =====================================================================
# PHẦN 2: CÁC HÀM TIỀN XỬ LÝ ẢNH VIỄN THÁM LANDSAT (TEAM'S FUNCTIONS)
# =====================================================================

def load_landsat_4bands(paths, normalize=True):
    """return X = (H,W,4) float32"""
    arr = []
    meta = None

    for i, p in enumerate(paths):
        with rasterio.open(p) as src:
            band = src.read(1).astype('float32')
            if normalize:
                bmin, bmax = np.percentile(band, (2,98))
                band = np.clip((band-bmin)/(bmax-bmin+1e-6), 0, 1)
            arr.append(band)
            if i == 0:
                meta = src.meta.copy()

    X_img = np.stack(arr, axis=-1)   # H W C
    return X_img, meta

def load_label_landsat(label_path, ref_meta):
    """Đọc nhãn và reproject đồng bộ với ảnh vệ tinh"""
    with rasterio.open(label_path) as src:
        label = src.read(1)
        label_meta = src.meta

    dst = np.zeros((ref_meta['height'], ref_meta['width']), dtype=label.dtype)

    reproject(
        source=label,
        destination=dst,
        src_transform=label_meta['transform'],
        src_crs=label_meta['crs'],
        dst_transform=ref_meta['transform'],
        dst_crs=ref_meta['crs'],
        resampling=Resampling.nearest
    )
    return dst

def load_landsat_pipeline(img_paths, label_path):
    """
    HÀM ĐÓNG GÓI CHO MAIN_RUNNER (Landsat):
    Chuyển đổi dữ liệu vệ tinh thành ma trận X để đưa vào thuật toán FCM.
    """
    # 1. Đọc và chuẩn hóa ảnh vệ tinh đa phổ
    X_img, meta = load_landsat_4bands(img_paths, normalize=True)
    h, w, c = X_img.shape
    
    # Duỗi khối HxWxC thành NxC (N = H*W pixel, mỗi pixel có C band đặc trưng)
    X = X_img.reshape(-1, c) 
    
    # 2. Đọc nhãn, đồng bộ tọa độ (reproject)
    Y_img = load_label_landsat(label_path, meta)
    true_labels_flat = Y_img.flatten().astype(int)
    
    le = LabelEncoder()
    true_labels_mapped = le.fit_transform(true_labels_flat)
    n_clusters = len(np.unique(true_labels_mapped))
    
    return X, true_labels_mapped, n_clusters, (h, w)
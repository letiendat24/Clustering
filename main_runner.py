import os
import sys
import time
import numpy as np
import nibabel as nib

# --- CÁC IMPORT TỪ DỰ ÁN CỦA BẠN ---
from c_means.fcm_np import FCM
from c_means.ssfcm2019 import SSFCM2
from c_means.s3fcm import S3FCM
from c_means.adsfcm import ADSFCM, FastADSFCM
from dataset.dataset import fetch_data_from_local, LabelEncoder
from c_means.utility import round_float, extract_labels, best_map
from c_means.validity import (dunn, davies_bouldin, partition_coefficient, 
                              Xie_Benie, classification_entropy, silhouette, 
                              accuracy_score, f1_score)

# Import thuật toán bạn tự viết từ file ads3fcm.py
from ads3fcm import ADS3FCM 

# --- CẤU HÌNH THAM SỐ CHUNG ---
ROUND_FLOAT = 3
EPSILON = 1e-5
MAX_ITER = 1000
M = 2
SEED = 42
LAMDA1 = 1
LAMDA2 = 1
LAMDA3 = 10
PERCENT_LABELED = 0.2
NOISE_RATIO = 0
BETA = 1.0
TAU = 0.5
SPLIT = '\t'

def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
    return str(round_float(val, n=n))

def write_report(alg, process_time, step, X, V, U, true_label):
    labels = extract_labels(U)
    mapped_labels = best_map(true_label, labels)
    kqdg = [
        alg, wdvl(process_time, n=3), str(step),
        wdvl(dunn(X, labels)), wdvl(davies_bouldin(X, labels)),
        wdvl(partition_coefficient(U)), wdvl(Xie_Benie(X, V, U)),
        wdvl(classification_entropy(U)), wdvl(silhouette(X, labels)),
        wdvl(f1_score(true_label, mapped_labels)),
        wdvl(accuracy_score(true_label, mapped_labels))
    ]
    # Format lại để bảng in ra không bị lệch khi tên thuật toán ngắn/dài
    return f"{kqdg[0]:<10}" + "".join([f"{x:>10}" for x in kqdg[1:]])


# --- CÁC HÀM ĐỌC DỮ LIỆU ĐỘC LẬP ---

def load_tabular_data(data_id=602):
    """Hàm đọc dữ liệu dạng bảng (VD: Dry Bean)"""
    print(f"Đang tải dữ liệu UCI id={data_id}...")
    X, Y = fetch_data_from_local(data_id)
    n_clusters = len(np.unique(Y))
    
    # Mã hóa nhãn chữ thành số
    le = LabelEncoder()
    true_labels = le.fit_transform(Y)
    
    return X, true_labels, n_clusters

def load_mri_data(img_path, label_path, slice_idx=90):
    """Hàm đọc dữ liệu ảnh não 3D MRI (.mnc)"""
    print(f"Đang tải dữ liệu MRI từ {img_path}...")
    img_obj = nib.load(img_path)
    label_obj = nib.load(label_path)

    img_2d = img_obj.get_fdata()[:, :, slice_idx]
    label_2d = label_obj.get_fdata()[:, :, slice_idx]

    # Chuẩn hóa
    img_2d = np.clip(img_2d, 0, None)
    img_2d = img_2d / np.max(img_2d)

    # Duỗi ảnh thành ma trận N x 1
    X = img_2d.reshape(-1, 1)
    true_labels = label_2d.reshape(-1).astype(int)
    n_clusters = 4 # Nền, CSF, GM, WM

    return X, true_labels, n_clusters

# (MAIN PROCESS)
if __name__ == '__main__':
    _start_time = time.time()

    # Chọn 'TABULAR' cho Dry Bean, hoặc 'MRI' cho ảnh não BrainWeb
    DATA_TYPE = 'TABULAR' 

    if DATA_TYPE == 'TABULAR':
        X, true_labels, n_clusters = load_tabular_data(data_id=602)
    elif DATA_TYPE == 'MRI':
        X, true_labels, n_clusters = load_mri_data(
            img_path='t1_icbm_normal_1mm_pn3_rf20.mnc',
            label_path='label_t1_icbm_normal_1mm_pn3_rf20.mnc'
        )
    else:
        raise ValueError("DATA_TYPE không hợp lệ!")

    print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
    print(f'size={X.shape[0]} x {X.shape[1]}')

    # Xử lý nhãn bán giám sát (Semi-supervised)
    n_labeled = int(PERCENT_LABELED * len(true_labels))
    np.random.seed(SEED)
    labeled_indices = np.random.choice(len(true_labels), n_labeled, replace=False)

    labels_all = np.full_like(true_labels, -1)
    labels_all[labeled_indices] = true_labels[labeled_indices]

    # Thêm nhiễu vào nhãn
    n_noisy = int(NOISE_RATIO * n_labeled)
    noisy_idx = np.random.choice(labeled_indices, n_noisy, replace=False)
    for i in noisy_idx:
        possible_labels = list(set(true_labels) - {true_labels[i]})
        labels_all[i] = np.random.choice(possible_labels)

    print(f"Đã làm sai {n_noisy} nhãn trong {n_labeled} điểm được gán nhãn.")
    
    # CHẠY THUẬT TOÁN
    fcm = FCM(X, n_clusters=n_clusters, m=M, max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED)
    fcm.fit()

    ssfcm2 = SSFCM2(X, n_clusters=n_clusters, labels=labels_all, m=M, max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED, ALPHA=LAMDA1)
    ssfcm2.fit()

    s3fcm = S3FCM(X, n_clusters, M, MAX_ITER, EPSILON, SEED, None, LAMDA1, LAMDA2, labels=labels_all)
    s3fcm.fit()
    
    adsfcm = ADSFCM(X, n_clusters=n_clusters, labels=labels_all, m=M, max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED, ALPHA=LAMDA1, beta=BETA)
    adsfcm.fit()

    fast_adsfcm = FastADSFCM(X, n_clusters=n_clusters, labels=labels_all, m=M, max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED, alpha=LAMDA1, beta=BETA, tau=TAU)
    fast_adsfcm.fit()

    ads3fcm = ADS3FCM(X, n_clusters, M, MAX_ITER, EPSILON, SEED, LAMDA1, LAMDA1, LAMDA2, LAMDA3, labels=labels_all)
    ads3fcm.fit()

    # IN BÁO CÁO (Sử dụng string formatting để các cột không bị lệch)
    titles = ['Alg', 'Time', 'Step', 'DI+', 'DB-', 'PC+', 'XB-', 'CE-', 'SI+', 'F1+', 'AC+']
    print(f"{titles[0]:<10}" + "".join([f"{t:>10}" for t in titles[1:]]))
    print("-" * 110)

    print(write_report('FCM', fcm.time, fcm.step, X, fcm.centroids, fcm.u, true_labels))
    print(write_report('SSFCM2', ssfcm2.time, ssfcm2.step, X, ssfcm2.centroids, ssfcm2.u, true_labels))
    print(write_report('S3FCM', s3fcm.time, s3fcm.step, X, s3fcm.centroids, s3fcm.u, true_labels))
    print(write_report('ADSFCM', adsfcm.time, adsfcm.step, X, adsfcm.centroids, adsfcm.u, true_labels))
    print(write_report('FADSFCM', fast_adsfcm.time, fast_adsfcm.step, X, fast_adsfcm.centroids, fast_adsfcm.u, true_labels))
    print(write_report('ADS3FCM', ads3fcm.time, ads3fcm.step, X, ads3fcm.centroids, ads3fcm.u, true_labels))
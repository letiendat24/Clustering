import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # chỉnh theo CPU thật

import time
import numpy as np
import matplotlib.pyplot as plt

# --- THUẬT TOÁN ---
from c_means.fcm_np import FCM
from c_means.ssfcm2019 import SSFCM2
from c_means.s3fcm import S3FCM
from c_means.adsfcm import ADSFCM
from c_means.fast_adsfcm import FastADSFCM
# Import data loader (Đảm bảo file data_loader.py nằm cùng thư mục)
from data_loader import load_mri_pipeline, load_landsat_pipeline

# --- UTILITY ---
from c_means.utility import round_float, extract_labels, best_map, division_by_zero, distance_cdist
from c_means.validity import (
    dunn, davies_bouldin, partition_coefficient,
    Xie_Benie, classification_entropy, silhouette,
    accuracy_score, f1_score
)

# ================= CLASS ADS3FCM ĐÃ ĐƯỢC FIX LỖI =================
class ADS3FCM(S3FCM):
    # ĐÃ THÊM THAM SỐ `shape` ĐỂ KHÔI PHỤC TÍNH KHÔNG GIAN
    def __init__(self, X, n_clusters, m, max_iter, epsilon, seed, shape, alpha, lambda1, lambda2, lambda3, labels=None):
        # Truyền shape xuống class cha S3FCM
        super().__init__(X, n_clusters, m, max_iter, epsilon, seed, shape, lambda1, lambda2, labels=labels)
        self.lambda3 = lambda3
        self.tau = 0.5

    def _capnhat_mttv(self):
        d2 = division_by_zero(distance_cdist(self.X, self.centroids))**2
        inv_d2 = 1.0 / division_by_zero(d2)  # (N x C)
        C_i = self.lambda1 + self.lambda2 + self.lambda3 * ((self.b[:, None] + 1) ** 2)
        
        K_ij = self.b[:, None] * (self.lambda2 * self.u_hat + 3.0 * self.lambda3 * (self.b[:, None] + 1) * self.f.T) 
        
        alpha_prime = (C_i - np.sum(K_ij, axis=1, keepdims=True)) / division_by_zero(np.sum(inv_d2, axis=1, keepdims=True))
        
        self.u = (alpha_prime * inv_d2 + K_ij) / division_by_zero(C_i)
        return self.u

    def _capnhat_tamcum(self):
        w = self.lambda1* self.u ** self.m + self.lambda2 * ((self.u - self.u_hat * self.b[:, None])**self.m) + self.lambda3 * (((1 + self.b[:, None]) * self.u - 3 * self.f.T * self.b[:, None]) ** self.m)
        w[w < 0] = 0
        numerator = w.T @ self.X
        denominator = np.sum(w, axis=0)[:, None]
        self.centroids = numerator / division_by_zero(denominator)
        return self.centroids

# ================= CONFIG =================
ROUND_FLOAT = 3
EPSILON = 1e-5
MAX_ITER = 1000
M = 2
SEED = 42

LAMDA1 = 0.1  # Nên để nhỏ (vd: 0.1) thay vì 1 để kết quả mềm mại hơn
LAMDA2 = 1.0
LAMDA3 = 10.0

PERCENT_LABELED = 0.2
NOISE_RATIO = 0.0

BETA = 1.0
TAU = 0.5

# ================= FORMAT & REPORT =================
def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
    return str(round_float(val, n=n))

def write_report(alg, process_time, step, X, V, U, true_label):
    labels = extract_labels(U)
    mapped_labels = best_map(true_label, labels)

    kqdg = [
        alg,
        wdvl(process_time, 3),
        str(step),
        wdvl(dunn(X, labels)),
        wdvl(davies_bouldin(X, labels)),
        wdvl(partition_coefficient(U)),
        wdvl(Xie_Benie(X, V, U)),
        wdvl(classification_entropy(U)),
        wdvl(silhouette(X, labels)),  # ĐÃ MỞ LẠI SI+
        wdvl(f1_score(true_label, mapped_labels)),
        wdvl(accuracy_score(true_label, mapped_labels))
    ]
    # Căn lề lại để hiển thị đủ cột SI+
    return f"{kqdg[0]:<10}" + "".join([f"{x:>9}" for x in kqdg[1:]])

# ================= MAIN =================
if __name__ == '__main__':
    start_time = time.time()

    DATA_TYPE = 'MRI'  # đổi sang 'LANDSAT' nếu cần

    # ===== LOAD DATA =====
    if DATA_TYPE == 'MRI':
        X, true_labels, n_clusters, shape_2d = load_mri_pipeline(
            img_path='dataset/MRI/t1_icbm_normal_1mm_pn3_rf20.mnc',
            label_path='dataset/MRI/label_t1_icbm_normal_1mm_pn3_rf20.mnc',
            slice_index=91,
            axis=2,
            target_size=(120, 140)
        )
    elif DATA_TYPE == 'LANDSAT':
        bands = ['band1.tif', 'band2.tif', 'band3.tif', 'band4.tif']
        X, true_labels, n_clusters, shape_2d = load_landsat_pipeline(
            img_paths=bands,
            label_path='label.tif'
        )
    else:
        raise ValueError("DATA_TYPE không hợp lệ!")

    print("Thời gian lấy dữ liệu:", round_float(time.time() - start_time))
    print(f'Kích thước X = {X.shape[0]} x {X.shape[1]}')

    # ===== SEMI-SUPERVISED LABEL =====
    n_labeled = int(PERCENT_LABELED * len(true_labels))
    np.random.seed(SEED)

    labeled_indices = np.random.choice(len(true_labels), n_labeled, replace=False)

    labels_all = np.full_like(true_labels, -1)
    labels_all[labeled_indices] = true_labels[labeled_indices]

    # thêm nhiễu nếu cần
    n_noisy = int(NOISE_RATIO * n_labeled)
    if n_noisy > 0:
        noisy_idx = np.random.choice(labeled_indices, n_noisy, replace=False)
        for i in noisy_idx:
            possible_labels = list(set(true_labels) - {true_labels[i]})
            labels_all[i] = np.random.choice(possible_labels)

    print(f"Đã làm sai {n_noisy} nhãn trong {n_labeled} điểm được gán nhãn mỏ neo.")
    print("=" * 110)

    # ===== RUN ALGORITHMS =====
    fcm = FCM(X, n_clusters=n_clusters, m=M, max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED)
    fcm.fit()

    ssfcm2 = SSFCM2(X, n_clusters=n_clusters, labels=labels_all,
                    m=M, max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED, ALPHA=LAMDA1)
    ssfcm2.fit()

    s3fcm = S3FCM(X, n_clusters, M, MAX_ITER, EPSILON, SEED,
                  shape_2d, LAMDA1, LAMDA2, labels=labels_all)
    s3fcm.fit()

    adsfcm = ADSFCM(X, n_clusters=n_clusters, labels=labels_all,
                    m=M, max_iter=MAX_ITER, epsilon=EPSILON,
                    seed=SEED, ALPHA=LAMDA1, beta=BETA)
    adsfcm.fit()

    fast_adsfcm = FastADSFCM(X, n_clusters=n_clusters, labels=labels_all,
                            m=M, max_iter=MAX_ITER, epsilon=EPSILON,
                            seed=SEED, alpha=LAMDA1, beta=BETA, tau=TAU)
    fast_adsfcm.fit()

    # KHỞI TẠO ADS3FCM ĐÃ TRUYỀN ĐÚNG shape_2d VÀ CÁC THAM SỐ
    ads3fcm = ADS3FCM(X, n_clusters, M, MAX_ITER, EPSILON, SEED, shape_2d, 
                      LAMDA1, LAMDA1, LAMDA2, LAMDA3, labels=labels_all)
    ads3fcm.fit()

    # ===== PRINT REPORT =====
    titles = ['Alg', 'Time', 'Step', 'DI+', 'DB-', 'PC+', 'XB-', 'CE-', 'SI+', 'F1+', 'AC+']
    print(f"{titles[0]:<10}" + "".join([f"{t:>9}" for t in titles[1:]]))
    print("-" * 105)

    print(write_report('FCM', fcm.time, fcm.step, X, fcm.centroids, fcm.u, true_labels))
    print(write_report('SSFCM2', ssfcm2.time, ssfcm2.step, X, ssfcm2.centroids, ssfcm2.u, true_labels))
    print(write_report('S3FCM', s3fcm.time, s3fcm.step, X, s3fcm.centroids, s3fcm.u, true_labels))
    print(write_report('ADSFCM', adsfcm.time, adsfcm.step, X, adsfcm.centroids, adsfcm.u, true_labels))
    print(write_report('FADSFCM', fast_adsfcm.time, fast_adsfcm.step, X, fast_adsfcm.centroids, fast_adsfcm.u, true_labels))
    print(write_report('ADS3FCM', ads3fcm.time, ads3fcm.step, X, ads3fcm.centroids, ads3fcm.u, true_labels))

    # ===== VISUALIZE =====
    print("\nĐang tạo ảnh trực quan hóa kết quả...")

    pred_labels = extract_labels(ads3fcm.u)
    mapped_pred_labels = best_map(true_labels, pred_labels)

    h, w = shape_2d

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(X[:, 0].reshape(h, w), cmap='gray')
    plt.title("Ảnh MRI đã xử lý")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(true_labels.reshape(h, w), cmap='tab10')
    plt.title("Ground Truth")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(mapped_pred_labels.reshape(h, w), cmap='tab10')
    plt.title("ADS3FCM Result")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
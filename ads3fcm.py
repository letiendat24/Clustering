import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # chỉnh theo CPU thật


import numpy as np
from c_means.fcm_np import FCM
from c_means.ssfcm2019 import SSFCM2
from c_means.s3fcm import S3FCM
from c_means.utility import *
import time
from c_means.utility import round_float, extract_labels
from dataset.dataset import fetch_data_from_local, TEST_CASES, LabelEncoder
from c_means.validity import dunn, davies_bouldin, partition_coefficient, Xie_Benie, classification_entropy, silhouette, hypervolume, accuracy_score,f1_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from c_means.adsfcm import ADSFCM, FastADSFCM


class ADS3FCM(S3FCM):
    def __init__(self, X,n_clusters,m,max_iter,epsilon,seed,alpha,lambda1,lambda2, lambda3,labels=None):
        super().__init__(X,n_clusters,m,max_iter,epsilon,seed,alpha,lambda1,lambda2,labels=labels)
        self.lambda3 = lambda3
        self.tau =0.5

    def _capnhat_mttv(self):
        # Tính khoảng cách bình phương giữa X và centroids
        d2 = division_by_zero(distance_cdist(self.X, self.centroids))**2
        inv_d2 = 1.0 / division_by_zero(d2)  # (N x C)
        C_i = self.lambda1 + self.lambda2 + self.lambda3 * ((self.b[:, None] + 1) ** 2)
        
        # Tính K_ij

        K_ij = self.b[:, None] * (self.lambda2 * self.u_hat + 3.0 * self.lambda3 * (self.b[:, None] + 1) * self.f.T) # dùng f thay u hat
        # K_ij = self.b[:, None] * self.u_hat * (self.lambda2 + 3*self.lambda3*(1 + self.b[:, None]))

        
        # Tính alpha_prime
        alpha_prime = (C_i - np.sum(K_ij, axis=1, keepdims=True)) / division_by_zero(np.sum(inv_d2, axis=1, keepdims=True))
        
        # Cập nhật u
        self.u = (alpha_prime * inv_d2 + K_ij) / division_by_zero(C_i)
        return self.u


    def _capnhat_tamcum(self):
        w =self.lambda1* self.u ** self.m + self.lambda2 * ((self.u - self.u_hat * self.b[:, None])**self.m) + self.lambda3 * (((1 + self.b[:, None]) * self.u - 3 * self.f.T * self.b[:, None]) ** self.m)
        w[w < 0] = 0
        numerator = w.T @ self.X
        denominator = np.sum(w, axis=0)[:, None]
        self.centroids = numerator / division_by_zero(denominator)
        return self.centroids



if __name__ == '__main__':

    import nibabel as nib

    ROUND_FLOAT = 3
    EPSILON = 1e-5
    MAX_ITER = 1000
    M = 2
    SEED = 42
    
    LAMDA1=1
    LAMDA2=1
    LAMDA3 = 10
    PERCENT_LABELED = 0.2
    noise_ratio = 0
    BETA =1.0 # ADSFCM
    TAU = 0.5
    # --------------------
    EPOCHS = 100
    # ALPHA =0.05
    LAM = 1e-3
    SPLIT = '\t'
    # =======================================

    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    def write_report(alg: str, index: int, process_time: float, step: int, X: np.ndarray, V: np.ndarray, U: np.ndarray, true_label: np.ndarray) -> str:
        labels = extract_labels(U)  # Giai mo
        mapped_labels = best_map(true_labels, labels)
        kqdg = [
            alg,
            wdvl(process_time, n=3),
            str(step),
            wdvl(dunn(X, labels)),  # DI
            wdvl(davies_bouldin(X, labels)),  # DB
            wdvl(partition_coefficient(U)),  # PC
            wdvl(Xie_Benie(X, V, U)),  # XB
            wdvl(classification_entropy(U)),  # CE
            wdvl(silhouette(X, labels)),  # SI
            # wdvl(hypervolume(U, m=2)),  # FHV
            wdvl(f1_score(true_labels,mapped_labels)),
            wdvl(accuracy_score(true_labels, mapped_labels))

        ]
        return SPLIT.join(kqdg)
    
    
 # =======================================
    
    clustering_report = []
    data_id =602
    if data_id in TEST_CASES:
        _start_time = time.time()
        _TEST = TEST_CASES[data_id]
        _dt = fetch_data_from_local(data_id)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        X, Y = _dt['X'], _dt['Y']
        _size = f"{_dt['data']['num_instances']} x {_dt['data']['num_features']}"
        print(f'size={_size}')
        n_clusters = _TEST['n_cluster']
    

    # Gán nhãn cho dữ liệu
    dlec = LabelEncoder()
    true_labels = dlec.fit_transform(Y.flatten())
    
    # Chọn ngẫu nhiên 20% dữ liệu đã được gắn nhãn
    n_labeled = int(PERCENT_LABELED * len(true_labels))

    np.random.seed(SEED)
    labeled_indices = np.random.choice(len(true_labels), n_labeled, replace=False)
    # tổng số lượng    lượng muốn chọn   chọn ko lặp lại

    # Gán nhãn là -1 cho tất cả các điểm
    labels_all = np.full_like(true_labels, -1)
    # Gán nhãn cho 20% dữ liệu
    labels_all[labeled_indices] = true_labels[labeled_indices]

    # =======================================================================
    # Thêm bước tạo nhãn sai 
    
    n_noisy = int(noise_ratio * n_labeled)
    noisy_idx = np.random.choice(labeled_indices, n_noisy, replace=False)

    for i in noisy_idx:
        possible_labels = list(set(true_labels) - {true_labels[i]})
        labels_all[i] = np.random.choice(possible_labels)

    print(f"Đã làm sai {n_noisy} nhãn ({noise_ratio*100}%) trong {n_labeled} ({PERCENT_LABELED*100}%)điểm được gán nhãn. ")
    print(f"Tham số: m={M}, max_iter={MAX_ITER}, epsilon={EPSILON}, seed={SEED}, lambda1={LAMDA1}, lambda2={LAMDA2},lamda3 ={LAMDA3},percent_labeled ={PERCENT_LABELED}")
    # =========================================================================
    fcm = FCM(X, n_clusters=n_clusters, m=M,max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED)
    fcm.fit()

    ssfcm2 = SSFCM2(X, n_clusters=n_clusters, labels=labels_all,m=M, max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED,ALPHA=LAMDA1)
    ssfcm2.fit()

    s3fcm = S3FCM(X, n_clusters, M, MAX_ITER, EPSILON, SEED,None, LAMDA1, LAMDA2,labels=labels_all)
    s3fcm.fit()
    adsfcm = ADSFCM(X, n_clusters=n_clusters, labels=labels_all, m=M, max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED, ALPHA=LAMDA1, beta=BETA)
    adsfcm.fit()

    fast_adsfcm = FastADSFCM(X, n_clusters=n_clusters, labels=labels_all, m=M, max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED, alpha=LAMDA1, beta=BETA, tau=TAU)
    fast_adsfcm.fit()

    ads3fcm = ADS3FCM(X,n_clusters,M,MAX_ITER,EPSILON,SEED,LAMDA1,LAMDA1,LAMDA2,LAMDA3,labels=labels_all)
    ads3fcm.fit()

    titles = ['Alg', 'Time', 'Step', 'DI+','DB-', 'PC+', 'XB-', 'CE-', 'SI+', 'F1+','AC+']
    print(SPLIT.join(titles))

    print(write_report(alg='FCM', index=0, process_time=fcm.time,step=fcm.step, X=X, V=fcm.centroids, U=fcm.u,true_label=true_labels))
    print(write_report(alg='SSFCM2', index=2, process_time=ssfcm2.time,step=ssfcm2.step, X=X, V=ssfcm2.centroids, U=ssfcm2.u,true_label=true_labels))
    print(write_report(alg='S3FCM', index=3, process_time=s3fcm.time,step=s3fcm.step, X=X, V=s3fcm.centroids, U=s3fcm.u,true_label=true_labels))
    print(write_report(alg='ADSFCM', index=4, process_time=adsfcm.time,step=adsfcm.step, X=X, V=adsfcm.centroids, U=adsfcm.u,true_label=true_labels))
    print(write_report(alg='FADSFCM', index=5, process_time=fast_adsfcm.time, step=fast_adsfcm.step, X=X, V=fast_adsfcm.centroids, U=fast_adsfcm.u, true_label=true_labels))
    print(write_report(alg='ADS3FCM', index=6, process_time=ads3fcm.time,step=ads3fcm.step, X=X, V=ads3fcm.centroids, U=ads3fcm.u,true_label=true_labels))


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # chỉnh theo CPU thật

from c_means.fcm import Dfcm as FCM
import numpy as np
from c_means.utility import *
from c_means.ssfcm2019 import SSFCM2
import time
from c_means.utility import round_float, extract_labels
from dataset.dataset import fetch_data_from_local, TEST_CASES, LabelEncoder
from c_means.validity import dunn, davies_bouldin, partition_coefficient, Xie_Benie, classification_entropy, silhouette, hypervolume, accuracy_score
from c_means.s3fcm import S3FCM
class ADSFCM(SSFCM2):
    def __init__(self, X, n_clusters, m, max_iter, epsilon, seed,ALPHA, beta , labels=None):
        super().__init__(X, n_clusters, m, max_iter, epsilon, seed,ALPHA, labels)
        self.beta =beta
    
    def _capnhat_mttv(self):
        """
        Cập nhật ma trận thành viên u_ij theo công thức (23) của ADSFCM.
        """
        d_squared = division_by_zero(distance_cdist(self.X, self.centroids)**2) # (n, c) - Công thức dùng d^2

        # Chuẩn bị các biến với shape phù hợp để tính toán
        b_broadcast = self.b[:, None]  # (n, 1)
        one_plus_b_broadcast = (1 + self.b)[:, None] # (n, 1)
        f_T = self.f.T # (n, c)
        mau =  (self.beta + self.alpha * one_plus_b_broadcast**2)
        norm_factor = 1.0 /division_by_zero(mau) # (n, 1)
        
        # thành phần thứ nhất
        mau_fcm_part = np.sum(d_squared[:, :, None] / division_by_zero(d_squared[:, None, :]), axis=2) # (n, c)
        tu_term1 = (self.beta + self.alpha * one_plus_b_broadcast**2 - 3 * self.alpha * b_broadcast * one_plus_b_broadcast * np.sum(self.f, axis=0)[:, None]) # (n, 1)
        term1 = tu_term1 / division_by_zero(mau_fcm_part) # (n, c)

        # Thành phần thứ hai 
        term2 = (3 * self.alpha * b_broadcast * one_plus_b_broadcast * f_T)/division_by_zero(mau) # (n, c)

        # 3. Kết hợp lại để ra ma trận membership cuối cùng
        self.u = norm_factor * (term1 + term2)
        return self.u

    def _capnhat_tamcum(self):
        """
        Cập nhật tâm cụm v_j theo công thức (24) của ADSFCM.
        """
      
        # Tính toán trọng số w_ij theo công thức ADSFCM
        asymmetric_term = (self.u * (1 + self.b[:, None] ) - 3 * self.b[:, None]  * self.f.T)**self.m # (n, c)
        w = self.beta * self.u**self.m + self.alpha * asymmetric_term # (n, c)

        # Cập nhật tâm cụm bằng trung bình có trọng số
        tu = w.T @ self.X  # (c, n) @ (n, d) = (c, d)
        mau = np.sum(w, axis=0)[:, None]  # (c, 1)

        self.centroids = tu / division_by_zero(mau)
        return self.centroids
    
class FastADSFCM(ADSFCM):
    def __init__(self, X, n_clusters, m, max_iter, epsilon, seed, alpha, beta,tau, labels=None ):
        super().__init__(X, n_clusters, m, max_iter, epsilon, seed, alpha, beta, labels)
        self.tau = tau  # Ngưỡng lọc cho Affinity Center Filtering

    def _capnhat_mttv(self):
        # Bước 1: tính membership gốc
        u_initial = super()._capnhat_mttv()
        u_scaled = u_initial.copy()

        # Bước 2: Tính khoảng cách bình phương và d_min
        d_squared = division_by_zero(distance_cdist(self.X, self.centroids)**2)  # (n, c)
        d_min_squared = np.min(d_squared, axis=1, keepdims=True)  # (n,1)

        # -------------------------------
        # Cơ chế lọc: dùng hiệu số (theo công thức trong hình)
        delta_i = np.full((self.X.shape[0], 1), self.tau)  # có thể thay self.tau bằng ngưỡng δ_i
        delta_j = np.full((1, self.centroids.shape[0]), self.tau)  # ngưỡng δ_j

        lhs = d_squared - delta_j        # (n, c)
        rhs = d_min_squared + delta_i    # (n, 1)
        affinity_mask = ~(lhs >= rhs)    # True nếu j được giữ lại (có affinity)
        non_affinity_mask = ~affinity_mask
        # -------------------------------

        # Bước 3: Membership scaling
        unlabeled_mask = (self.b == 0)[:, None]
        labeled_mask = (self.b == 1)[:, None]

        # Unlabeled: non-affinity = 0
        u_scaled[unlabeled_mask & non_affinity_mask] = 0.0

        # Labeled: non-affinity = giá trị f
        f_T = self.f.T  # (n, c)
        u_scaled[labeled_mask & non_affinity_mask] = f_T[labeled_mask & non_affinity_mask]

        # Bước 4: Renormalize
        row_sums = u_scaled.sum(axis=1, keepdims=True)
        zero_rows = (row_sums <= 1e-12).flatten()
        if zero_rows.any():
            u_scaled[zero_rows, :] = u_initial[zero_rows, :]
            row_sums = u_scaled.sum(axis=1, keepdims=True)

        u_final = division_by_zero(u_scaled / row_sums)

        self.u = u_final
        return self.u


if __name__ == '__main__':

    ROUND_FLOAT = 3
    EPSILON = 1e-5
    MAX_ITER = 1000
    M = 2
    SEED = 42
    
    LAMDA1=0.1
    LAMDA2=1.0
    PERCENT_LABELED = 0.2
    noise_ratio = 0.5
    BETA =1.0 # ADSFCM
    TAU = 0.5
    # --------------------
    
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
            wdvl(hypervolume(U, m=2)),  # FHV
            wdvl(accuracy_score(true_labels, mapped_labels))

        ]
        return SPLIT.join(kqdg)
   
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    def plot_clusters(X, results):
        """
        X: dữ liệu gốc (n_samples, n_features)
        results: dict { "Tên thuật toán": (labels, centers) }
        """
        # Fit PCA 1 lần trên dữ liệu gốc
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
        if n == 1: axes = [axes]
        
        for ax, (name, (labels, centers)) in zip(axes, results.items()):
            sc = ax.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap="tab10", s=20)
            if centers is not None:
                C_2d = pca.transform(centers)  # dùng cùng PCA
                ax.scatter(C_2d[:,0], C_2d[:,1], c="red", marker="X", s=100)
            ax.set_title(name)
        plt.show()

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    def plot_clusters(X, results):
        """
        X: dữ liệu gốc (n_samples, n_features)
        results: dict { "Tên thuật toán": (labels, centers) }
        """
        # Fit PCA 1 lần trên dữ liệu gốc
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
        if n == 1: axes = [axes]
        
        for ax, (name, (labels, centers)) in zip(axes, results.items()):
            sc = ax.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap="tab10", s=20)
            if centers is not None:
                C_2d = pca.transform(centers)  # dùng cùng PCA
                ax.scatter(C_2d[:,0], C_2d[:,1], c="red", marker="X", s=100)
            ax.set_title(name)
        plt.show()

    # =======================================

    clustering_report = []
    data_id = 109
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
        
        
        # ===============================================
       # Gán nhãn cho dữ liệu
        dlec = LabelEncoder()
        true_labels = dlec.fit_transform(_dt['Y'].flatten())
        
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
        print(f"Tham số: m={M}, max_iter={MAX_ITER}, epsilon={EPSILON}, seed={SEED}, lambda1={LAMDA1}, lambda2={LAMDA2},percent_labeled ={PERCENT_LABELED}")
        # ================================================================================================
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

        # --------------------------------
        # In kết quả
        titles = ['Alg', 'Time', 'Step', 'DI+','DB-', 'PC+', 'XB-', 'CE-', 'SI+', 'FHV-','AC+']
        print(SPLIT.join(titles))
        print(write_report(alg='FCM', index=0, process_time=fcm.time,step=fcm.step, X=X, V=fcm.centroids, U=fcm.u,true_label=true_labels))
        print(write_report(alg='SSFCM2', index=0, process_time=ssfcm2.time,step=ssfcm2.step, X=X, V=ssfcm2.centroids, U=ssfcm2.u,true_label=true_labels))
        print(write_report(alg='S3FCM', index=2, process_time=s3fcm.time,step=s3fcm.step, X=X, V=s3fcm.centroids, U=s3fcm.u,true_label=true_labels))
        print(write_report(alg='ADSFCM', index=0, process_time=adsfcm.time,step=adsfcm.step, X=X, V=adsfcm.centroids, U=adsfcm.u,true_label=true_labels))
        print(write_report(alg='FADSFCM', index=0, process_time=fast_adsfcm.time, step=fast_adsfcm.step, X=X, V=fast_adsfcm.centroids, U=fast_adsfcm.u, true_label=true_labels))

        results = {
            "Input":(labels_all,None),
            "FCM": (extract_labels(fcm.u),fcm.centroids),
            "SSFCM": (extract_labels(ssfcm2.u),ssfcm2.centroids),
            "S3FCM": (extract_labels(s3fcm.u),s3fcm.centroids ),
            "ADSFCM":(extract_labels(adsfcm.u),adsfcm.centroids),
            "FADSFCM":(extract_labels(fast_adsfcm.u),fast_adsfcm.centroids)
            }

        plot_clusters(X, results)
import numpy as np
from c_means.fcm import Dfcm as FCM
from c_means.ssfcm2019 import SSFCM2
from c_means.utility import *
import time
from c_means.utility import round_float, extract_labels
from dataset.dataset import fetch_data_from_local, TEST_CASES, LabelEncoder
from c_means.validity import dunn, davies_bouldin, partition_coefficient, Xie_Benie, classification_entropy, silhouette, hypervolume, accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# from k_means.kmpp import KMPP
class S3FCM(SSFCM2):
    def __init__(self, X, n_clusters, m, max_iter, epsilon, seed,ALPHA,lambda1, lambda2, labels=None):
        self.labels = labels  

        super().__init__(X, n_clusters, m, max_iter, epsilon, seed, ALPHA, labels)

        
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        fcm = FCM(self.X, n_clusters=self.n_clusters, m=self.m,max_iter=self.max_iter, epsilon=self.epsilon, seed=self.seed)
        fcm.fit()
        self.u_hat = fcm.u
        # self.u_hat = U_hat if U_hat is not None else np.zeros((self.n_data, n_clusters))
    

    def _khoitao_tamcum(self):
        """
        Khởi tạo tâm cụm V(0) từ các điểm đã gán nhãn.
        Nếu không có nhãn thì fallback về random như FCM.
        """
        if self.labels is not None and np.any(self.labels >= 0):
            labeled_idx = np.where(self.labels >= 0)[0]
            labeled_labels = self.labels[labeled_idx]

            one_hot = np.eye(self.n_clusters)[labeled_labels]
            sums = one_hot.T @ self.X[labeled_idx]     # (c, d)
            counts = one_hot.sum(axis=0)[:, None]      # (c, 1)
            self.centroids = sums / np.maximum(counts, 1)
        else:
            # fallback: random như FCM
            rand_idx = np.random.choice(self.n_data, self.n_clusters, replace=False)
            self.centroids = self.X[rand_idx]
        return self.centroids

    def _capnhat_mttv(self):
        
        d =division_by_zero( distance_cdist(self.X, self.centroids))
        delta = self.lambda1 * (self.f.T * self.b[:, None]) + self.lambda2 * (self.u_hat * self.b[:, None])
        mau = np.sum(d[:, :, None]**2 / d[:, None, :]**2, axis=2)   # (n, c)
        tu = (1 + self.lambda1 + self.lambda2) - np.sum(delta, axis=1)[:, None]  # (n, 1) broadcast -> (n, c)
        self.u = (1.0 / (1 + self.lambda1 + self.lambda2)) * (tu / mau + delta)
        self.u = self.u / np.sum(self.u, axis=1, keepdims=True)
        return self.u
   


    def _capnhat_tamcum(self):
        w = (self.u ** self.m) \
                + self.lambda1 * ((self.u - self.f.T * self.b[:, None]) ** self.m) \
                + self.lambda2 * ((self.u - self.u_hat * self.b[:, None]) ** self.m)  # (n, c)
        tu = w.T @ self.X       # (c, n) @ (n, d) -> (c, d)
        mau = np.sum(w, axis=0)[:, None]  # (c, 1)
        self.centroids = tu / division_by_zero(mau)
        
        return self.centroids
       
    def fit(self):
        time_start = time.time()
        fcm = FCM(self.X, n_clusters=self.n_clusters, m=self.m,max_iter=self.max_iter, epsilon=self.epsilon, seed=self.seed)
        fcm.fit()
        self.u_hat = fcm.u
        self.centroids = self._khoitao_tamcum()
        while(self.step < self.max_iter):
            self.step += 1
            old_u = self.u.copy()
            self.centroids = self._capnhat_tamcum()  # Bước 2 : cập nhật tâm cụm
            self.u = self._capnhat_mttv()     # Bước 3: cập nhật ma trận thành viên
            if self._check_exit(old_u=old_u):  # Bước 4: kiểm tra điều kiện hội tụ
                break
        time_end = time.time()
        self.time = time_end - time_start
        return self.u, self.centroids, self.step
        

if __name__ == '__main__':

    ROUND_FLOAT = 3
    EPSILON = 1e-5
    MAX_ITER = 1000
    M = 2
    SEED = 42
    
    LAMDA1=0.1
    LAMDA2=1
    PERCENT_LABELED = 0.2
    noise_ratio = 0.5
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

    # =======================================

    clustering_report = []
    data_id = 53
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
        
        # =========================================================================
        from deep_learning.ae import AutoEncoder

        
        
        fcm = FCM(X, n_clusters=n_clusters, m=M,max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED)
        # fcm.centroids = kmpp.centroids
        fcm.fit()
        
        from c_means.ssfcm import SSFCM
        ssfcm = SSFCM(X, n_clusters=n_clusters, labels=labels_all,m=M, max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED)
        u, centroids, u_ngang, step = ssfcm.fit()

        ssfcm2 = SSFCM2(X, n_clusters=n_clusters, labels=labels_all,m=M, max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED,ALPHA=LAMDA1)
        ssfcm2.fit()

        

        s3fcm = S3FCM(X, n_clusters, M, MAX_ITER, EPSILON, SEED,None, LAMDA1, LAMDA2,labels=labels_all)
        s3fcm.fit()
        # -----------------------------------------------------------------------------------------

        #  # Chuẩn hóa X trước khi AE
        # # X_AE = StandardScaler().fit_transform(X)
        # X_AE = MinMaxScaler().fit_transform(X)
        # d = X_AE.shape[1]
        # layer_sizes = AutoEncoder.build_layer_sizes(d, hidden_dims=[d, int(d/2)])
        # # 6;3 với wine, iris
        # # [d,d,d/2] --> iris >90%
        # # [16,16,12]
        # ae = AutoEncoder(layer_sizes, alpha=ALPHA, lam=LAM, seed=SEED)
        # ae.fit(X_AE, epochs=EPOCHS,)
        # Z = ae.encode(X_AE)
        # lambda_max = AutoEncoder.compute_lambda_max(X)
        # M = AutoEncoder.suggest_m(lambda_max)
        # # Z = StandardScaler().fit_transform(Z)
        # Z = MinMaxScaler().fit_transform(Z)

        # fcm_ae = FCM(Z, n_clusters=n_clusters, m=M,max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED)
        # fcm_ae.fit()


        # ssfcm2_ae = SSFCM2(Z, n_clusters=n_clusters, labels=labels_all,m=M, max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED,ALPHA=LAMDA1)
        # ssfcm2_ae.fit()


        # s3fcm_ae = S3FCM(Z, n_clusters, M, MAX_ITER, EPSILON, SEED,ALPHA, LAMDA1, LAMDA2,labels=labels_all)
        # s3fcm_ae.fit()

        # print("Layer-sizes:",layer_sizes)
        # --------------------------------
        titles = ['Alg', 'Time', 'Step', 'DI+','DB-', 'PC+', 'XB-', 'CE-', 'SI+', 'FHV-','AC+']
        print(SPLIT.join(titles))
        # print("-----------------------------------DỮ LIỆU THÔ----------------------------------------------")
        print(write_report(alg='FCM', index=0, process_time=fcm.time,step=fcm.step, X=X, V=fcm.centroids, U=fcm.u,true_label=true_labels))
        print(write_report(alg='SSFCM', index=4, process_time=ssfcm.time,step=ssfcm.step, X=X, V=ssfcm.centroids, U=ssfcm.u,true_label=true_labels))
        print(write_report(alg='SSFCM2', index=1, process_time=ssfcm2.time,step=ssfcm2.step, X=X, V=ssfcm2.centroids, U=ssfcm2.u,true_label=true_labels))
        print(write_report(alg='S3FCM', index=2, process_time=s3fcm.time,step=s3fcm.step, X=X, V=s3fcm.centroids, U=s3fcm.u,true_label=true_labels))
        

        # print("-----------------------------------MÃ HÓA ĐẶC TRƯNG VỚI AUTOENCODER------------------------------------------")
        # print(write_report(alg='FCM', index=3, process_time=fcm_ae.time,step=fcm_ae.step, X=Z, V=fcm_ae.centroids, U=fcm_ae.u,true_label=true_labels))
        # print(write_report(alg='SSFCM2', index=4, process_time=ssfcm2_ae.time,step=ssfcm2_ae.step, X=Z, V=ssfcm2_ae.centroids, U=ssfcm2_ae.u,true_label=true_labels))
        # print(write_report(alg='S3FCM', index=5, process_time=s3fcm_ae.time,step=s3fcm_ae.step, X=Z, V=s3fcm_ae.centroids, U=s3fcm_ae.u,true_label=true_labels))

        results = {
            "Input":(labels_all,None),
            "FCM": (extract_labels(fcm.u),fcm.centroids),
            "SSFCM1":(extract_labels(ssfcm.u),ssfcm.centroids),
            "SSFCM2": (extract_labels(ssfcm2.u),ssfcm2.centroids),
            "S3FCM": (extract_labels(s3fcm.u),s3fcm.centroids )
            
            # "FCM_AE": (extract_labels(fcm_ae.u),fcm_ae.centroids),
            # "SSFCM2_AE": (extract_labels(ssfcm2_ae.u),ssfcm2_ae.centroids),
            # "S3FCM_AE": (extract_labels(s3fcm_ae.u),s3fcm_ae.centroids )
        }

        plot_clusters(X, results)


        # ================================
        # OK 19/09/2025

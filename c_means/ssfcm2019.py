from c_means.fcm_np import FCM
import numpy as np
from c_means.utility import *
# from c_means.ssfcm import SSFCM
import time
from c_means.utility import round_float, extract_labels
from dataset.dataset import fetch_data_from_local, TEST_CASES, LabelEncoder
from c_means.validity import dunn, davies_bouldin, partition_coefficient, Xie_Benie, classification_entropy, silhouette, hypervolume, accuracy_score

class SSFCM2(FCM):
    def __init__(self, X, n_clusters, m, max_iter, epsilon, seed,ALPHA,labels=None):
        super().__init__(X, n_clusters, m, max_iter, epsilon, seed)
        self.labels = labels
        self.b= self._init_b()
        self.f= self._init_f()
        
        self.alpha = ALPHA

    def _init_b(self):
        y = np.array(self.labels)
        b = (y != -1).astype(int)
        return b
    
    def _init_f(self):
        f = np.zeros((self.n_clusters, self.n_data))  # (c, n)
        if self.labels is not None:
            for i in range(self.n_data):
                if self.labels[i] != -1:
                    f[self.labels[i], i] = 1
        return f

    def _capnhat_mttv(self):
        d = division_by_zero(distance_cdist(self.X, self.centroids))  # (n, c)
        # phần tử chia khoảng cách
        mau_bt1 = np.sum((d[:, :, None] / d[:, None, :]) ** self.m, axis=2)  # (n, c)
        # tu_bt1: (n, c), vì f bây giờ là (c, n) => f.sum(axis=0) -> (n,)
        tu_bt1 = 1 + self.alpha * (1 - self.b * np.sum(self.f, axis=0))  # (n,)
        tu_bt1 = tu_bt1[:, None]  # (n, 1)
        self.u = (1.0 / (1 + self.alpha)) * (tu_bt1 / mau_bt1 + self.alpha * (self.f.T * self.b[:, None]))  # (n, c)
        return self.u

    def _capnhat_tamcum(self):
    
        w = (self.u**self.m) + self.alpha * ((self.u - self.f.T * self.b[:, None])**self.m)  # (n, c)
        tu = w.T @ self.X      # (c, n) @ (n, d) = (c, d)
        mau = np.sum(w, axis=0)[:, None]  # (c,1)

        self.centroids = tu / division_by_zero(mau)
        return self.centroids
    
if __name__ == '__main__':

    ROUND_FLOAT = 3
    EPSILON = 1e-5
    MAX_ITER = 1000
    M = 2
    SEED = 42
    ALPHA =0.1
    SPLIT = '\t'
    # =======================================

    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    def write_report(alg: str, index: int, process_time: float, step: int, X: np.ndarray, V: np.ndarray, U: np.ndarray, true_label: np.ndarray) -> str:
        labels = extract_labels(U)  # Giai mo
        mapped_labels = best_map(true_labels, labels)
        kqdg = [
            alg,
            wdvl(process_time, n=2),
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
    # =======================================

    clustering_report = []
    data_id = 602
    if data_id in TEST_CASES:
        _start_time = time.time()
        _TEST = TEST_CASES[data_id]
        _dt = fetch_data_from_local(data_id)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Data getting time:", round_float(time.time() - _start_time))
        X, Y = _dt['X'], _dt['Y']
        _size = f"{_dt['data']['num_instances']} x {_dt['data']['num_features']}"
        print(f'size={_size}')
        n_clusters = _TEST['n_cluster']

    # from c_means.doc_anh_vien_tham import *
    # image_path = r"C:\Users\khact\Desktop\NCKH\dataset\Image\0.jpg"       # ảnh gốc (RGB)
    # label_path = r"C:\Users\khact\Desktop\NCKH\dataset\Mask\0.png"        # ảnh nhãn (mask)
    # water_label_id = 255 
    # resize = (893,551)
    # target_size = resize
    # X, Y, n_samples, load_time = load_and_prepare_data(
    #     image_path, label_path, water_label_id, resize
    # )

    # if X is None:
    #     exit()
    # n_clusters = 2  # nước / không nước
    # print(f"Ảnh có {n_samples} điểm ảnh, phân {n_clusters} cụm")
        # ===============================================
    # Gán nhãn cho dữ liệu
    dlec = LabelEncoder()
    true_labels = dlec.fit_transform(Y.flatten())
    # Chọn ngẫu nhiên 20% dữ liệu đã được gắn nhãn
    n_labeled = int(0.2 * len(true_labels)+1)

    np.random.seed(SEED)
    labeled_indices = np.random.choice(len(true_labels), n_labeled, replace=False)
    # tổng số lượng    lượng muốn chọn   chọn ko lặp lại

    # Gán nhãn là -1 cho tất cả các điểm
    labels_all = np.full_like(true_labels, -1)
    # Gán nhãn cho 20% dữ liệu
    labels_all[labeled_indices] = true_labels[labeled_indices]
    # =======================================================================

    

    # ssfcm = SSFCM(X, n_clusters=n_clusters, labels=labels_all,m=M, max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED)
    # ssfcm.fit()
    fcm = FCM(X, n_clusters=n_clusters, m=M,max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED)
    fcm.fit()

    ssfcm2 = SSFCM2(X, n_clusters=n_clusters, labels=labels_all,m=M, max_iter=MAX_ITER, epsilon=EPSILON, seed=SEED,ALPHA=ALPHA)
    ssfcm2.fit()
    # --------------------------------
    titles = ['Alg', 'Time', 'Step', 'DI+','DB-', 'PC+', 'XB-', 'CE-', 'SI+', 'FHV-','AC+']
    print(SPLIT.join(titles))
    print(write_report(alg='FCM', index=0, process_time=fcm.time,step=fcm.step, X=X, V=fcm.centroids, U=fcm.u,true_label=true_labels))
    # print(write_report(alg='SSFCM', index=0, process_time=ssfcm.time,step=ssfcm.step, X=X, V=ssfcm.centroids, U=ssfcm.u,true_label=true_labels))
    print(write_report(alg='SSFCM', index=0, process_time=ssfcm2.time,step=ssfcm2.step, X=X, V=ssfcm2.centroids, U=ssfcm2.u,true_label=true_labels))

    from c_means.visualization import plot_clusters
    results = {
        "Input":(labels_all,None),
        "FCM": (extract_labels(fcm.u),fcm.centroids),
        "SSFCM": (extract_labels(ssfcm2.u),ssfcm2.centroids),

    }

    plot_clusters(X, results)
    from c_means.visualization import plot_clusters
    results = {
        "Input":(labels_all,None),
        "FCM": (extract_labels(fcm.u),fcm.centroids),
        "SSFCM2": (extract_labels(ssfcm2.u),ssfcm2.centroids),
        # "S3FCM": (extract_labels(s3fcm.u),s3fcm.centroids ),
        # "ADSFCM":(extract_labels(adsfcm.u),adsfcm.centroids),
        # "ADS3FCM": (extract_labels(sadsfcm.u),sadsfcm.centroids )

    }

    plot_clusters(X, results)
    # print('f:\n',sadsfcm.f)
    # print('u_hat:\n',sadsfcm.u_hat)


# =====================================================
# Trực quan hóa ảnh gốc, nhãn thật, và kết quả phân cụm
# =====================================================

# import matplotlib.pyplot as plt
# from c_means.visualization import visualize_segmentation_bw
# from c_means.doc_anh_vien_tham import load_rgb_image_only

# original_img = load_rgb_image_only(image_path, target_size)

# # Tạo dict chứa nhãn kết quả của từng thuật toán
# cluster_results = {
#     "FCM": extract_labels(fcm.u),
#     "SSFCM2": extract_labels(ssfcm2.u),
#     # "S3FCM": extract_labels(s3fcm.u),
#     # "ADSFCM": extract_labels(adsfcm.u),
#     # "ADS3FCM": extract_labels(sadsfcm.u),
# }

# visualize_segmentation_bw(original_img, true_labels, cluster_results, resize)

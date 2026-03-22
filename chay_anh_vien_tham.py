import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # chỉnh theo CPU thật


import numpy as np
from c_means.fcm import Dfcm as FCM
from c_means.ssfcm2019 import SSFCM2
from c_means.s3fcm import S3FCM
from c_means.utility import *
import time
from c_means.utility import round_float, extract_labels
from dataset.dataset import fetch_data_from_local, TEST_CASES, LabelEncoder
from c_means.validity import dunn, davies_bouldin, partition_coefficient, Xie_Benie, classification_entropy, silhouette, hypervolume, accuracy_score,f1_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from c_means.adsfcm import ADSFCM, FastADSFCM
from c_means.doc_anh_vien_tham import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


if __name__ == '__main__':

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
    image_path = r"D:\NCKH\code\M-33-20-D-c-4-2_241.jpg"      
    label_path = r"D:\NCKH\code\M-33-20-D-c-4-2_241_m.png"  
    resize = (64,64)
    target_size = resize
    X, Y, n_samples, load_time = load_and_prepare_data( image_path, label_path, resize)
    Y = load_label_image(label_path, (128,128))
# # ==================DOC ANH 4 BAND=====================================================================
    # from c_means.landsat_read import load_landsat_4bands, load_label

    # # Danh sách đường dẫn 4 band
    # band_paths = [
    #     r"C:\Users\khact\Desktop\NCKH\dataset\hanoi\hanoi_SR_B2.tif",
    #     r"C:\Users\khact\Desktop\NCKH\dataset\hanoi\hanoi_SR_B3.tif",
    #     r"C:\Users\khact\Desktop\NCKH\dataset\hanoi\hanoi_SR_B4.tif",
    #     r"C:\Users\khact\Desktop\NCKH\dataset\hanoi\hanoi_SR_B5.tif"
    # ]

    # # Đọc ảnh và nhãn
    # X, meta = load_landsat_4bands(band_paths, normalize=True)
    # Y = load_label(r"C:\Users\khact\Desktop\NCKH\dataset\hanoi\hanoilabel2.tif")
    # =====================================================================================================
    unique_labels = np.unique(Y)
    print("Các nhãn có trong ảnh:", unique_labels)

    if X is None:
        exit()
    n_clusters =4
   

    # Nếu X có 3 chiều (H, W, C) thì chuyển về (N, C)
    if X.ndim == 3:
        X = X.reshape(-1, X.shape[-1])

    # print(X.shape)
    # print(f"Ảnh có {n_samples} điểm ảnh, phân {n_clusters} cụm")
    # ===============================================

    
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
    # Ánh xạ nhãn về dạng liên tục 0..K-1
    # Ánh xạ nhãn thật về dải 0..K-1, bỏ qua -1
    unique_labels = np.unique(labels_all[labels_all != -1])
    label_map = {val: idx for idx, val in enumerate(unique_labels)}

    labels_mapped = np.full_like(labels_all, -1)
    for val, idx in label_map.items():
        labels_mapped[labels_all == val] = idx

    labels_all = labels_mapped
    labels_all = np.where(labels_all >= n_clusters, labels_all % n_clusters, labels_all)


    
    
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
    from c_means.ads3fcm import ADS3FCM
    ads3fcm = ADS3FCM(X,n_clusters,M,MAX_ITER,EPSILON,SEED,LAMDA1,LAMDA1,LAMDA2,LAMDA3,labels=labels_all)
    ads3fcm.fit()

    titles = ['Alg', 'Time', 'Step', 'DI+','DB-', 'PC+', 'XB-', 'CE-', 'SI+', 'F1+','AC+']
    print(SPLIT.join(titles))

    print(write_report(alg='FCM', index=0, process_time=fcm.time,step=fcm.step, X=X, V=fcm.centroids, U=fcm.u,true_label=true_labels))
    print(write_report(alg='SSFCM2', index=2, process_time=ssfcm2.time,step=ssfcm2.step, X=X, V=ssfcm2.centroids, U=ssfcm2.u,true_label=true_labels))
    print(write_report(alg='S3FCM', index=3, process_time=s3fcm.time,step=s3fcm.step, X=X, V=s3fcm.centroids, U=s3fcm.u,true_label=true_labels))
    print(write_report(alg='ADSFCM', index=4, process_time=adsfcm.time,step=adsfcm.step, X=X, V=adsfcm.centroids, U=adsfcm.u,true_label=true_labels))
    # print(write_report(alg='FADSFCM', index=5, process_time=fast_adsfcm.time, step=fast_adsfcm.step, X=X, V=fast_adsfcm.centroids, U=fast_adsfcm.u, true_label=true_labels))
    print(write_report(alg='ADS3FCM', index=6, process_time=ads3fcm.time,step=ads3fcm.step, X=X, V=ads3fcm.centroids, U=ads3fcm.u,true_label=true_labels))

from c_means.visualization import visualize_segmentation_auto

# Đọc ảnh gốc
original_img, _ = load_any_image(image_path, resize)

# Lấy nhãn phân cụm từ từng thuật toán
cluster_results = {
    "FCM": extract_labels(fcm.u),
    "SSFCM2": extract_labels(ssfcm2.u),
    "S3FCM": extract_labels(s3fcm.u),
    "ADSFCM": extract_labels(adsfcm.u),
    "ads3fcm": extract_labels(ads3fcm.u),
}

visualize_segmentation_auto(
    original_image=original_img,
    true_labels=true_labels,
    cluster_labels_dict=cluster_results,
    resize=(128, 128),
    save_dir="results/segmentation",
    save_name="slice91_compare.png"
)

import numpy as np
from c_means.utility import distance_cdist, division_by_zero
from c_means.adsfcm import ADSFCM

class FastADSFCM(ADSFCM):
    def __init__(self, X, n_clusters, m, max_iter, epsilon, seed, alpha, beta, tau, labels=None):
        # Gọi hàm khởi tạo của class mẹ (ADSFCM)
        super().__init__(X, n_clusters, m, max_iter, epsilon, seed, alpha, beta, labels)
        self.tau = tau  # Ngưỡng lọc (Affinity Threshold)

    def _capnhat_mttv(self):
        # BƯỚC 1: Tính khoảng cách bình phương (Nhanh, cơ bản)
        d_squared = distance_cdist(self.X, self.centroids)**2  # (n, c)

        # BƯỚC 2: Tạo bộ lọc (Affinity Mask) ngay từ đầu
        d_min_squared = np.min(d_squared, axis=1, keepdims=True)  # (n, 1)

        delta_i = np.full((self.X.shape[0], 1), self.tau)
        delta_j = np.full((1, self.centroids.shape[0]), self.tau)

        lhs = d_squared - delta_j        
        rhs = d_min_squared + delta_i    
        affinity_mask = ~(lhs >= rhs)    # True nếu cụm đủ gần để giữ lại
        non_affinity_mask = ~affinity_mask

        # BƯỚC 3: Tính toán độ thuộc (Chỉ tính cho vùng được giữ lại)
        # Các biến này kế thừa từ cấu trúc của ADSFCM gốc
        b_broadcast = self.b[:, None]  # (n, 1)
        one_plus_b_broadcast = (1 + self.b)[:, None] # (n, 1)
        f_T = self.f.T # (n, c)
        
        mau = (self.beta + self.alpha * one_plus_b_broadcast**2)
        norm_factor = 1.0 / division_by_zero(mau) # (n, 1)

        # KỸ THUẬT TĂNG TỐC: Gán khoảng cách của cụm bị loại = vô cùng (inf)
        # Khi chia cho inf, kết quả sẽ = 0, hàm sum sẽ lờ nó đi, tiết kiệm phép tính.
        d_squared_masked = d_squared.copy()
        d_squared_masked[non_affinity_mask] = np.inf 

        # Tính toán các thành phần công thức giống ADSFCM nhưng trên ma trận đã mask
        mau_fcm_part = np.sum(d_squared[:, :, None] / division_by_zero(d_squared_masked[:, None, :]), axis=2) # (n, c)

        tu_term1 = (self.beta + self.alpha * one_plus_b_broadcast**2 - 3 * self.alpha * b_broadcast * one_plus_b_broadcast * np.sum(self.f, axis=0)[:, None]) # (n, 1)
        term1 = tu_term1 / division_by_zero(mau_fcm_part) # (n, c)

        term2 = (3 * self.alpha * b_broadcast * one_plus_b_broadcast * f_T) / division_by_zero(mau) # (n, c)

        # Ma trận độ thuộc sơ bộ
        u_new = norm_factor * (term1 + term2)

        # BƯỚC 4: Xử lý nhãn (Semi-supervised) và ép các cụm xa về 0
        u_scaled = u_new.copy()

        unlabeled_mask = (self.b == 0)[:, None]
        labeled_mask = (self.b == 1)[:, None]

        # Điểm chưa có nhãn: Cụm bị loại ép chặt về 0
        u_scaled[unlabeled_mask & non_affinity_mask] = 0.0

        # Điểm có nhãn: Lấy giá trị cố định từ Ground Truth (f_T)
        u_scaled[labeled_mask & non_affinity_mask] = f_T[labeled_mask & non_affinity_mask]

        # BƯỚC 5: Chuẩn hóa lại an toàn (Renormalize)
        row_sums = u_scaled.sum(axis=1, keepdims=True)
        
        # Phòng hờ: Nếu điểm ảnh nào bị lọc mất tất cả các cụm (tổng = 0)
        zero_rows = (row_sums <= 1e-12).flatten()
        if zero_rows.any():
            u_scaled[zero_rows, :] = u_new[zero_rows, :]
            row_sums[zero_rows] = u_scaled[zero_rows, :].sum(axis=1, keepdims=True)

        # Chia lấy tỷ lệ, dùng np.maximum để tuyệt đối không lỗi ZeroDivision
        u_final = u_scaled / np.maximum(row_sums, 1e-12)

        # CHỐT CHẶN: Ép kết quả về đoạn [0, 1] để sửa dứt điểm lỗi PC+ và XB-
        self.u = np.clip(u_final, 0.0, 1.0)
        
        return self.u
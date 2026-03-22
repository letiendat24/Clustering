# # import os
# # import matplotlib.pyplot as plt
# # import numpy as np

# # def visualize_segmentation_auto(original_image, true_labels, cluster_labels_dict,
# #                                 resize=(128, 128), save_dir=None, save_name="segmentation_result.png"):
# #     """
# #     Hiển thị ảnh gốc, nhãn gốc và kết quả phân cụm.
# #     - Tự động chọn hiển thị đen trắng nếu chỉ có 2 nhãn.
# #     - Có thể lưu hình ra thư mục nếu save_dir != None.
# #     """
# #     h, w = resize
# #     gt_mask = true_labels.reshape(h, w)
# #     unique_gt = np.unique(gt_mask)
    
# #     n = len(cluster_labels_dict) + 2
# #     fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    
# #     # --- Ảnh gốc ---
# #     axes[0].imshow(original_image)
# #     axes[0].set_title("Ảnh gốc (RGB)")
# #     axes[0].axis("off")
    
# #     # --- Ảnh nhãn gốc ---
# #     cmap_gt = 'gray' if len(unique_gt) <= 2 else 'viridis'
# #     axes[1].imshow(gt_mask, cmap=cmap_gt)
# #     axes[1].set_title("Ảnh nhãn gốc")
# #     axes[1].axis("off")
    
# #     # --- Các kết quả thuật toán ---
# #     for i, (name, labels) in enumerate(cluster_labels_dict.items(), start=2):
# #         seg = labels.reshape(h, w)
# #         unique_vals = np.unique(seg)
        
# #         if len(unique_vals) == 2 and len(unique_gt) == 2:
# #             overlap0 = np.sum((seg == unique_vals[0]) & (gt_mask == unique_gt[-1]))
# #             overlap1 = np.sum((seg == unique_vals[1]) & (gt_mask == unique_gt[-1]))
# #             seg_disp = (seg == unique_vals[0]).astype(np.uint8) if overlap0 > overlap1 else (seg == unique_vals[1]).astype(np.uint8)
# #             cmap_seg = 'gray'
# #         else:
# #             seg_disp = seg
# #             cmap_seg = 'viridis' if len(unique_vals) <= 10 else 'nipy_spectral'

# #         axes[i].imshow(seg_disp, cmap=cmap_seg)
# #         axes[i].set_title(f"KQ {name}")
# #         axes[i].axis("off")
    
# #     plt.tight_layout()
    
# #     # --- Lưu nếu có yêu cầu ---
# #     if save_dir is not None:
# #         os.makedirs(save_dir, exist_ok=True)
# #         save_path = os.path.join(save_dir, save_name)
# #         plt.savefig(save_path, bbox_inches='tight', dpi=300)
# #         print(f"✅ Đã lưu hình vào: {save_path}")
# #         plt.close(fig)
# #     else:
# #         plt.show()


# import os
# import matplotlib.pyplot as plt
# import numpy as np

# def visualize_segmentation_auto(original_image, true_labels, cluster_labels_dict,
#                                 resize=(128, 128), save_dir=None, save_name="segmentation_result.png"):
#     """
#     Hiển thị ảnh gốc, nhãn gốc và kết quả phân cụm.
#     - Tự động chọn hiển thị đen trắng nếu chỉ có 2 nhãn.
#     - Tự nhận biết ảnh đa phổ và tạo RGB tổng hợp nếu cần.
#     - Có thể lưu hình ra thư mục nếu save_dir != None.
#     """
#     h, w = resize
#     gt_mask = true_labels.reshape(h, w)
#     unique_gt = np.unique(gt_mask)
    
#     n = len(cluster_labels_dict) + 2
#     fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    
#     # --- Ảnh gốc ---
#     if original_image.ndim == 2 or original_image.shape[2] == 1:
#         # Ảnh 1 kênh → grayscale
#         axes[0].imshow(original_image.squeeze(), cmap='gray')
#         axes[0].set_title("Ảnh gốc (Grayscale)")
#     elif original_image.shape[2] == 3:
#         # Ảnh RGB → hiển thị bình thường
#         axes[0].imshow(original_image)
#         axes[0].set_title("Ảnh gốc (RGB)")
#     else:
#         # Ảnh đa phổ → tạo RGB tổng hợp từ 3 kênh đầu
#         rgb_img = original_image[:, :, :3]
#         rgb_img = rgb_img / np.max(rgb_img)  # chuẩn hóa 0-1
#         axes[0].imshow(rgb_img)
#         axes[0].set_title("Ảnh gốc (RGB tổng hợp)")
    
#     axes[0].axis("off")
    
#     # --- Ảnh nhãn gốc ---
#     cmap_gt = 'gray' if len(unique_gt) <= 2 else 'viridis'
#     axes[1].imshow(gt_mask, cmap=cmap_gt)
#     axes[1].set_title("Ảnh nhãn gốc")
#     axes[1].axis("off")
    
#     # --- Các kết quả thuật toán ---
#     for i, (name, labels) in enumerate(cluster_labels_dict.items(), start=2):
#         seg = labels.reshape(h, w)
#         unique_vals = np.unique(seg)
        
#         if len(unique_vals) == 2 and len(unique_gt) == 2:
#             overlap0 = np.sum((seg == unique_vals[0]) & (gt_mask == unique_gt[-1]))
#             overlap1 = np.sum((seg == unique_vals[1]) & (gt_mask == unique_gt[-1]))
#             seg_disp = (seg == unique_vals[0]).astype(np.uint8) if overlap0 > overlap1 else (seg == unique_vals[1]).astype(np.uint8)
#             cmap_seg = 'gray'
#         else:
#             seg_disp = seg
#             cmap_seg = 'viridis' if len(unique_vals) <= 10 else 'nipy_spectral'

#         axes[i].imshow(seg_disp, cmap=cmap_seg)
#         axes[i].set_title(f"KQ {name}")
#         axes[i].axis("off")
    
#     plt.tight_layout()
    
#     # --- Lưu nếu có yêu cầu ---
#     if save_dir is not None:
#         os.makedirs(save_dir, exist_ok=True)
#         save_path = os.path.join(save_dir, save_name)
#         plt.savefig(save_path, bbox_inches='tight', dpi=300)
#         print(f"✅ Đã lưu hình vào: {save_path}")
#         plt.close(fig)
#     else:
#         plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA





def create_image_from_X(X, h, w):
    """
    Hàm tự tạo ảnh từ ma trận X.
    - 1 band  → grayscale
    - >=3 band → dùng 3 band đầu làm RGB
    - 2 band  → PCA để tạo RGB
    """
    k = X.shape[1]

    if k == 1:
        img = X.reshape(h, w)
        return img, "gray"

    if k >= 3:
        # Lấy 3 band đầu để tạo RGB
        img = X[:, :3].reshape(h, w, 3)
        img = img.astype(float)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img, None

    if k == 2:
        # 2 band → PCA ra 3 kênh
        pca = PCA(n_components=3)
        rgb = pca.fit_transform(X)
        rgb = rgb.reshape(h, w, 3)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        return rgb, None


def visualize_segmentation_auto(original_image, true_labels, cluster_labels_dict,
                                X=None, resize=(128, 128), save_dir=None, save_name="segmentation_result.png"):
    """
    Hiển thị:
    - Ảnh gốc hoặc ảnh tổng hợp từ X
    - Nhãn gốc
    - Các kết quả phân cụm

    Nếu original_image = None → tự động tạo ảnh từ X.
    """

    h, w = resize
    gt_mask = true_labels.reshape(h, w)
    unique_gt = np.unique(gt_mask)

    n = len(cluster_labels_dict) + 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    # --------------------------------------------------
    # 1. Xử lý ảnh gốc hoặc tạo ảnh từ X
    # --------------------------------------------------
    if original_image is None:
        if X is None:
            raise ValueError("Bạn không cung cấp original_image hoặc X để tạo ảnh!")

        img0, cmap0 = create_image_from_X(X, h, w)
        axes[0].imshow(img0, cmap=cmap0)
        axes[0].set_title("Ảnh tổng hợp từ X")
    else:
        if original_image.ndim == 2:
            rgb_img = np.stack([original_image]*3, axis=-1)
            axes[0].imshow(rgb_img)
            axes[0].set_title("Ảnh gốc (grayscale → RGB)")
        elif original_image.ndim == 3:
            if original_image.shape[2] == 1:
                rgb_img = np.concatenate([original_image]*3, axis=-1)
                axes[0].imshow(rgb_img)
                axes[0].set_title("Ảnh gốc (1 kênh → RGB)")
            elif original_image.shape[2] == 2:
                h, w, _ = original_image.shape
                reshaped = original_image.reshape(-1, 2)
                pca = PCA(n_components=3)
                rgb_img = pca.fit_transform(reshaped).reshape(h, w, 3)
                rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
                axes[0].imshow(rgb_img)
                axes[0].set_title("Ảnh gốc (2 kênh → RGB)")
            elif original_image.shape[2] >= 3:
                rgb_img = original_image[:, :, :3]
                rgb_img = rgb_img / np.max(rgb_img)
                axes[0].imshow(rgb_img)
                axes[0].set_title("Ảnh gốc (RGB)")


    axes[0].axis("off")

    # --------------------------------------------------
    # 2. Ảnh nhãn gốc
    # --------------------------------------------------
    cmap_gt = 'gray' if len(unique_gt) <= 2 else 'viridis'
    axes[1].imshow(gt_mask, cmap=cmap_gt)
    axes[1].set_title("Ảnh nhãn gốc")
    axes[1].axis("off")

    # --------------------------------------------------
    # 3. Ảnh phân cụm
    # --------------------------------------------------
    for i, (name, labels) in enumerate(cluster_labels_dict.items(), start=2):
        seg = labels.reshape(h, w)
        unique_vals = np.unique(seg)

        # Nếu nhị phân → auto align foreground
        if len(unique_vals) == 2 and len(unique_gt) == 2:
            overlap0 = np.sum((seg == unique_vals[0]) & (gt_mask == unique_gt[-1]))
            overlap1 = np.sum((seg == unique_vals[1]) & (gt_mask == unique_gt[-1]))
            seg_disp = (seg == unique_vals[0]).astype(np.uint8) if overlap0 > overlap1 else (seg == unique_vals[1]).astype(np.uint8)
            cmap_seg = 'gray'
        else:
            seg_disp = seg
            cmap_seg = 'viridis' if len(unique_vals) <= 10 else 'nipy_spectral'

        axes[i].imshow(seg_disp, cmap=cmap_seg)
        axes[i].set_title(f"KQ {name}")
        axes[i].axis("off")

    plt.tight_layout()

    # --------------------------------------------------
    # 4. Lưu hoặc show
    # --------------------------------------------------
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f" Đã lưu hình vào: {save_path}")
        plt.close(fig)
    else:
        plt.show()

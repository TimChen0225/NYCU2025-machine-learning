import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import argparse
import os
import time
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D


def load_data(image_path):
    """
    讀取圖片並轉換為作業要求的格式。
    回傳:
        spatial_data: (10000, 2) 每一列是 [row, col] 座標
        color_data:   (10000, 3) 每一列是 [R, G, B] 顏色值
        img_shape:    圖片原始尺寸 (100, 100)
    """
    try:
        # 使用 matplotlib 讀取圖片
        img = plt.imread(image_path)

        # 如果圖片是 PNG 且有 Alpha channel (RGBA)，只取 RGB
        if img.shape[-1] == 4:
            img = img[:, :, :3]

        # 如果讀進來是 0-1 (float)，可以視情況轉回 0-255，或是直接用 float 計算
        # 這裡保留原始讀取格式，通常 matplotlib png 會是 0-1
        img_data = img.reshape(-1, 3)
        img_data = img_data

        rows, cols, _ = img.shape

        # 建立空間座標 (Spatial Information)
        # S(x): coordinate of the pixel
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        spatial_data = np.stack([y.flatten(), x.flatten()], axis=1)  # (row, col)

        print(f"Loaded {image_path}: Shape {img.shape}, Data points: {len(img_data)}")
        return spatial_data, img_data, (rows, cols)

    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None


def save_gif(image_list, output_filename, duration=0.5):
    """
    將一系列的圖片矩陣存成 GIF 動畫
    """
    if not image_list:
        print("No images to save for GIF.")
        return

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_filename) if os.path.dirname(output_filename) else ".", exist_ok=True)

    # 轉換數值範圍以正確存圖 (假設輸入是 0-1 或 0-255)
    formatted_images = []
    for img in image_list:
        if img.dtype != np.uint8:
            formatted_images.append((img * 255).astype(np.uint8))
        else:
            formatted_images.append(img)

    imageio.mimsave(output_filename, formatted_images, duration=duration)
    print(f"Saved GIF to {output_filename}")


def visualize_cluster(labels, img_shape, original_color_data=None):
    """
    將分群標籤 (Labels) 轉換為視覺化圖片
    策略：計算該群所有像素的「平均顏色」來作為該群的代表色
    """
    # 建立結果圖片 (先用 Flatten 的格式)
    result_img_flat = np.zeros((img_shape[0] * img_shape[1], 3))

    unique_labels = np.unique(labels)

    if original_color_data is not None:
        # === 策略 A: 使用平均顏色 (您的想法) ===
        for label in unique_labels:
            # 找出屬於該群的所有像素索引
            mask = labels == label

            # 計算這些像素的 RGB 平均值
            if np.sum(mask) > 0:  # 避免除以零
                mean_color = np.mean(original_color_data[mask], axis=0)
                result_img_flat[mask] = mean_color
    else:
        # === 策略 B: 如果沒有提供原始顏色，退回隨機顏色 ===
        colors = np.random.rand(len(unique_labels), 3)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            result_img_flat[mask] = colors[i]

    # Reshape 回原本圖片的長寬 (H, W, 3)
    return result_img_flat.reshape(img_shape[0], img_shape[1], 3)


def save_eigenspace_plots(eigen_vectors, labels, save_dir, file_prefix):
    """
    同時儲存 2D 與 3D (如果維度足夠) 的 Eigenspace 分佈圖

    Args:
        eigen_vectors: (N, k) 特徵向量矩陣
        labels: (N,) 分群結果
        save_dir: 儲存的基礎目錄 (例如: results/image1/Spectral_RatioCut)
        file_prefix: 檔名前綴 (例如: Eigenspace_k3_random)
    """
    # 確保基礎目錄存在
    os.makedirs(save_dir, exist_ok=True)

    # 取得特徵向量的維度 (k)
    dim = eigen_vectors.shape[1]

    # ==========================
    # 1. 繪製 2D 圖 (取前兩個維度)
    # ==========================
    dir_2d = os.path.join(save_dir, "2D")
    os.makedirs(dir_2d, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(eigen_vectors[:, 0], eigen_vectors[:, 1], c=labels, cmap="viridis", s=2, alpha=0.6)
    plt.title(f"2D Projection: {file_prefix}")
    plt.xlabel("Eigenvector 1")
    plt.ylabel("Eigenvector 2")

    path_2d = os.path.join(dir_2d, f"{file_prefix}_2D.png")
    plt.savefig(path_2d)
    plt.close()
    print(f"Saved 2D plot: {path_2d}")

    # ==========================
    # 2. 繪製 3D 圖 (如果 k >= 3)
    # ==========================
    if dim >= 3:
        dir_3d = os.path.join(save_dir, "3D")
        os.makedirs(dir_3d, exist_ok=True)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 取前三個維度
        ax.scatter(
            eigen_vectors[:, 0], eigen_vectors[:, 1], eigen_vectors[:, 2], c=labels, cmap="viridis", s=2, alpha=0.6
        )

        ax.set_title(f"3D Projection: {file_prefix}")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")

        # 調整視角 (Optional: 可以調整 elev 和 azim 來改變觀看角度)
        ax.view_init(elev=30, azim=45)

        path_3d = os.path.join(dir_3d, f"{file_prefix}_3D.png")
        plt.savefig(path_3d)
        plt.close()
        print(f"Saved 3D plot: {path_3d}")
    else:
        # 如果 k=2，就不用畫 3D 了，因為第三維不存在
        pass


def log_execution_info(log_path, method, k, gamma_c, init_method, frames, duration):
    """
    將執行資訊寫入 Log 檔案
    """
    # 如果檔案不存在，先寫入 Header
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("Method,K,Gamma_C,Init,Iterations,Duration_sec\n")

    with open(log_path, "a") as f:
        f.write(f"{method},{k},{gamma_c},{init_method},{len(frames)},{duration:.4f}\n")
    print(f"Logged execution info to {log_path}")


def compute_gram_matrix(spatial_data, color_data, gamma_s, gamma_c):
    print("Computing Gram Matrix... (This might take a while)")
    spatial_dist_sq = cdist(spatial_data, spatial_data, "sqeuclidean")
    color_dist_sq = cdist(color_data, color_data, "sqeuclidean")
    kernel_matrix = np.exp(-(gamma_s * spatial_dist_sq + gamma_c * color_dist_sq))
    return kernel_matrix


class KernelKMeans:
    def __init__(self, k, max_iters=100, init_method="random"):
        self.k = k
        self.max_iters = max_iters
        self.init_method = init_method

    def kmeans_plus_plus_init(self, kernel_matrix):
        n = kernel_matrix.shape[0]
        diag_vals = np.diag(kernel_matrix)

        centers_idx = [np.random.choice(n)]
        min_dist_sq = np.full(n, np.inf)

        for _ in range(self.k - 1):
            current_center_idx = centers_idx[-1]

            dist_sq = diag_vals + diag_vals[current_center_idx] - 2 * kernel_matrix[:, current_center_idx]
            dist_sq = np.maximum(dist_sq, 0)
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)

            probs = min_dist_sq / np.sum(min_dist_sq)
            next_center = np.random.choice(n, p=probs)
            centers_idx.append(next_center)

        center_diag_vals = diag_vals[centers_idx]  # shape (k,)

        dists_to_centers = (
            diag_vals[:, np.newaxis] + center_diag_vals[np.newaxis, :] - 2 * kernel_matrix[:, centers_idx]
        )

        labels = np.argmin(dists_to_centers, axis=1)

        return labels

    def fit(self, kernel_matrix, img_shape, color_data):
        n = kernel_matrix.shape[0]
        history_images = []

        if self.init_method == "k-means++":
            labels = self.kmeans_plus_plus_init(kernel_matrix)
        else:
            # Default to random initialization
            labels = np.random.randint(0, self.k, n)

        for i in range(self.max_iters):
            current_img = visualize_cluster(labels, img_shape, color_data)
            history_images.append(current_img)

            label_encode = np.zeros((n, self.k))
            label_encode[np.arange(n), labels] = 1

            counts = label_encode.sum(axis=0)
            counts[counts == 0] = 1  # ensure no division by zero

            # x: target node, x_c: nodes in cluster c, C: size of cluster c
            # dist = k(x,x) - (2/C) * sum[k(x, x_c)] + (1/C^2) * sum[k(x_c, x_c')]
            dist1 = 2 * kernel_matrix @ label_encode / counts
            cluster_internal_sum = label_encode * (kernel_matrix @ label_encode)
            dist2 = cluster_internal_sum / (counts**2)

            dist_sum = -dist1 + dist2
            new_labels = dist_sum.argmin(axis=1)

            if np.all(labels == new_labels):
                break
            labels = new_labels

        # add last visualization
        history_images.append(visualize_cluster(labels, img_shape, color_data))

        return labels, history_images


class SpectralClustering:
    def __init__(self, k, mode="normalized", max_iters=100, init_method="random"):
        self.k = k
        self.mode = mode
        self.max_iters = max_iters
        self.init_method = init_method

    def kmeans_plus_plus_init(self, data):
        n_samples, n_features = data.shape
        centers = np.zeros((self.k, n_features))
        centers[0] = data[np.random.randint(n_samples)]

        for i in range(1, self.k):
            dist_sq = cdist(data, centers[:i], "sqeuclidean").min(axis=1)
            probs = dist_sq / dist_sq.sum()
            next_center = np.random.choice(n_samples, p=probs)
            centers[i] = data[next_center]

        return centers

    def fit(self, kernel_matrix, img_shape, color_data):
        print(f"Running Spectral Clustering ({self.mode} cut)...")
        n = kernel_matrix.shape[0]

        d = np.sum(kernel_matrix, axis=1)

        if self.mode == "ratio":
            L = np.diag(d) - kernel_matrix
        else:
            # Normalized Laplacian: L_sym = I - D^-1/2 * W * D^-1/2
            d_sqrt_inv = 1.0 / np.sqrt(d)
            # D_inv @ W @ D_inv
            L = np.eye(n) - (d_sqrt_inv[:, np.newaxis] * kernel_matrix * d_sqrt_inv[np.newaxis, :])

        print("Solving Eigenvalue problem...")
        eigenvals, eigenvecs = np.linalg.eigh(L)

        U = eigenvecs[:, : self.k]

        if self.mode == "normalized":
            row_sums = np.linalg.norm(U, axis=1)
            row_sums[row_sums == 0] = 1
            U = U / row_sums[:, np.newaxis]

        print("Running K-means on Eigenspace...")

        if self.init_method == "k-means++":
            centers = self.kmeans_plus_plus_init(U)
        else:
            centers = U[np.random.choice(n, self.k, replace=False)]

        labels = np.zeros(n, dtype=int)
        history_images = []

        for i in range(self.max_iters):
            dists = cdist(U, centers, "sqeuclidean")

            new_labels = np.argmin(dists, axis=1)

            current_img = visualize_cluster(new_labels, img_shape, color_data)
            history_images.append(current_img)

            if np.all(labels == new_labels):
                break

            labels = new_labels

            for c in range(self.k):
                mask = labels == c
                if np.any(mask):
                    centers[c] = np.mean(U[mask], axis=0)
                else:
                    centers[c] = U[np.random.choice(n)]

        return labels, history_images, U


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML HW6: Kernel K-means & Spectral Clustering")

    parser.add_argument("--image", type=str, default="image1.png", help="Path to input image")
    parser.add_argument("--k", type=int, default=2, help="Number of clusters")
    parser.add_argument("--gamma_s", type=float, default=0.001, help="Spatial kernel parameter")
    parser.add_argument("--gamma_c", type=float, default=10, help="Color kernel parameter")
    parser.add_argument("--init", type=str, default="random", help="Initialization method(random or k-means++)")

    args = parser.parse_args()

    IMAGE_PATH = args.image
    K_CLUSTERS = args.k
    GAMMA_S = args.gamma_s
    GAMMA_C = args.gamma_c
    INIT_METHOD = args.init

    img_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]

    base_dir = os.path.join("results", img_name)
    LOG_FILE = os.path.join(base_dir, "execution_log.csv")

    dir_kkm = os.path.join(base_dir, "KernelKMeans")
    dir_ratio = os.path.join(base_dir, "Spectral_RatioCut")
    dir_norm = os.path.join(base_dir, "Spectral_NormalizedCut")

    print(f"========================================")
    print(f"Running with settings:")
    print(f"Image: {IMAGE_PATH} (Output ID: {img_name})")
    print(f"Clusters (k): {K_CLUSTERS}")
    print(f"Gamma_s: {GAMMA_S}, Gamma_c: {GAMMA_C}")
    print(f"Initialization: {INIT_METHOD}")
    print(f"Output Directory: {base_dir}")
    print(f"Log File: {LOG_FILE}")
    print(f"========================================")

    print(">>> Loading Data...")
    spatial, color, shape = load_data(IMAGE_PATH)

    if spatial is not None:
        print(">>> Computing Kernel...")
        gram_mat = compute_gram_matrix(spatial, color, GAMMA_S, GAMMA_C)

        # Kernel K-means
        print(f">>> Running Kernel K-means (k={K_CLUSTERS})...")
        kkm = KernelKMeans(k=K_CLUSTERS, max_iters=200, init_method=INIT_METHOD)

        start_time = time.time()
        kkm_labels, kkm_gif_imgs = kkm.fit(gram_mat, shape, color)
        end_time = time.time()

        kkm_duration = end_time - start_time

        kkm_path = os.path.join(dir_kkm, f"k{K_CLUSTERS}_C{GAMMA_C}_{INIT_METHOD}.gif")
        save_gif(kkm_gif_imgs, kkm_path)

        log_execution_info(LOG_FILE, "KernelKMeans", K_CLUSTERS, GAMMA_C, INIT_METHOD, kkm_gif_imgs, kkm_duration)

        # Spectral Clustering (Ratio Cut)
        print(f">>> Running Spectral Clustering (Ratio Cut, k={K_CLUSTERS})...")
        sc_ratio = SpectralClustering(k=K_CLUSTERS, mode="ratio", init_method=INIT_METHOD)

        start_time = time.time()
        ratio_labels, ratio_gif_imgs, ratio_vecs = sc_ratio.fit(gram_mat, shape, color)
        end_time = time.time()

        ratio_duration = end_time - start_time

        ratio_gif_path = os.path.join(dir_ratio, f"k{K_CLUSTERS}_C{GAMMA_C}_{INIT_METHOD}.gif")
        save_gif(ratio_gif_imgs, ratio_gif_path)

        log_execution_info(LOG_FILE, "Spectral_Ratio", K_CLUSTERS, GAMMA_C, INIT_METHOD, ratio_gif_imgs, ratio_duration)

        ratio_plot_path = os.path.join(dir_ratio, f"Eigenspace_k{K_CLUSTERS}_C{GAMMA_C}_{INIT_METHOD}")
        save_eigenspace_plots(
            ratio_vecs,
            ratio_labels,
            save_dir=dir_ratio,
            file_prefix=f"Eigenspace_k{K_CLUSTERS}_C{GAMMA_C}_{INIT_METHOD}",
        )

        # Spectral Clustering (Normalized Cut)
        print(f">>> Running Spectral Clustering (Normalized Cut, k={K_CLUSTERS})...")
        sc_norm = SpectralClustering(k=K_CLUSTERS, mode="normalized", init_method=INIT_METHOD)

        start_time = time.time()
        norm_labels, norm_gif_imgs, norm_vecs = sc_norm.fit(gram_mat, shape, color)
        end_time = time.time()

        norm_duration = end_time - start_time

        norm_gif_path = os.path.join(dir_norm, f"k{K_CLUSTERS}_C{GAMMA_C}_{INIT_METHOD}.gif")
        save_gif(norm_gif_imgs, norm_gif_path)

        log_execution_info(LOG_FILE, "Spectral_Norm", K_CLUSTERS, GAMMA_C, INIT_METHOD, norm_gif_imgs, norm_duration)

        norm_plot_path = os.path.join(dir_norm, f"Eigenspace_k{K_CLUSTERS}_C{GAMMA_C}_{INIT_METHOD}")
        save_eigenspace_plots(
            norm_vecs, norm_labels, save_dir=dir_norm, file_prefix=f"Eigenspace_k{K_CLUSTERS}_C{GAMMA_C}_{INIT_METHOD}"
        )

        print(f"\nDone! Log saved to '{LOG_FILE}'")

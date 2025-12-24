import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import cdist

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================


def load_data(data_dir, resize_shape=(50, 50)):
    """
    讀取 Yale Face Database。
    假設資料夾內檔名格式包含 subject 資訊 (例如 subject01.happy.gif)
    或是依照資料夾結構排列。這裡示範通用讀取法。
    """
    print(f"Loading data from {data_dir}...")
    images = []
    labels = []

    # 這裡需要根據你解壓縮後的實際檔名格式微調
    # 作業說明：15 subjects, 11 images each.
    # 檔名通常類似: subject01.centerlight, subject01.glasses ...

    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return np.array([]), np.array([])

    file_list = [f for f in os.listdir(data_dir) if not f.startswith(".")]
    file_list.sort()  # 確保順序固定

    for filename in file_list:
        # 簡單的 label 解析，假設檔名以 subjectXX 開頭
        # 你可能需要根據實際解壓縮後的檔名修改這裡
        if not filename.startswith("subject"):
            continue

        subject_id = int(filename[7:9])  # subject01 -> 1

        filepath = os.path.join(data_dir, filename)
        # 讀取圖片 (Grayscale)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # 有些 Yale 資料集是 .gif，cv2 可能預設不讀，需注意
            # 如果是 gif，可用 plt.imread 或 PIL
            import matplotlib.image as mpimg

            try:
                img = mpimg.imread(filepath)
                if len(img.shape) > 2:
                    img = img.mean(axis=2)  # 轉灰階
            except:
                print(f"Skipping {filename}")
                continue

        # Resize for easier implementation [cite: 9]
        img_resized = cv2.resize(img, resize_shape)

        # Flatten image to vector
        img_vector = img_resized.flatten()

        images.append(img_vector)
        labels.append(subject_id)

    X = np.array(images)
    y = np.array(labels)

    print(f"Data loaded: {X.shape[0]} images, {X.shape[1]} features (pixels).")
    return X, y


def split_train_test(X, y):
    """
    作業說明[cite: 8]:
    Total 165 images. Train: 135, Test: 30.
    這通常意味著前 9 張訓練，後 2 張測試，或是隨機分配。
    這裡示範依照作業常見的固定切分 (每個人前 9 張 train, 後 2 張 test)。
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    unique_subjects = np.unique(y)

    for subject in unique_subjects:
        indices = np.where(y == subject)[0]
        # 假設每個 subject 有 11 張
        # 分配前 9 張給 Train，後 2 張給 Test (或是根據作業具體要求的檔名分配)
        train_idx = indices[:9]
        test_idx = indices[9:]

        X_train.extend(X[train_idx])
        y_train.extend(y[train_idx])
        X_test.extend(X[test_idx])
        y_test.extend(y[test_idx])

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


# ==========================================
# 2. Visualization Functions
# ==========================================


def plot_faces(faces, n_rows, n_cols, title, image_shape=(50, 50)):
    """
    畫出 Eigenfaces 或 Fisherfaces [cite: 11]
    faces: (n_faces, n_pixels)
    """
    plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5))
    for i in range(n_rows * n_cols):
        if i >= len(faces):
            break
        plt.subplot(n_rows, n_cols, i + 1)
        # Reshape vector back to image
        face_img = faces[i].reshape(image_shape)
        plt.imshow(face_img, cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.show()


def plot_reconstruction(originals, reconstructed, n_images=10, image_shape=(50, 50)):
    """
    畫出原始圖片與重建圖片的對比 [cite: 11]
    """
    plt.figure(figsize=(20, 4))
    for i in range(n_images):
        # Original
        plt.subplot(2, n_images, i + 1)
        plt.imshow(originals[i].reshape(image_shape), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # Reconstructed
        plt.subplot(2, n_images, i + 1 + n_images)
        plt.imshow(reconstructed[i].reshape(image_shape), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")
    plt.show()


# ==========================================
# 3. Models (Framework)
# ==========================================


class PCA:
    def __init__(self, n_components=25):
        self.n_components = n_components
        self.mean_face = None
        self.eigenvectors = None  # Component matrix (n_components, n_features)

    def fit(self, X):
        """
        TODO: 實作 PCA 訓練過程 [cite: 11]
        1. Compute mean face
        2. Center the data (X - mean)
        3. Compute Covariance Matrix (hint: use SVD or X^T X trick if d >> n)
        4. Compute Eigenvalues and Eigenvectors
        5. Sort and select top n_components
        """
        pass

    def transform(self, X):
        """
        TODO: 投影資料到低維空間
        Return: Projected data (n_samples, n_components)
        """
        pass

    def inverse_transform(self, X_projected):
        """
        TODO: 重建人臉
        Return: Reconstructed data (n_samples, n_features)
        """
        pass


class LDA:
    def __init__(self, n_components=25):
        self.n_components = n_components
        self.eigenvectors = None  # Fisherfaces

    def fit(self, X, y):
        """
        TODO: 實作 LDA 訓練過程 [cite: 11]
        1. Compute global mean
        2. Compute mean vector for each class
        3. Compute Within-class scatter matrix (S_W)
        4. Compute Between-class scatter matrix (S_B)
        5. Solve generalized eigenvalue problem: S_B * w = lambda * S_W * w
           (Note: S_W might be singular, check lecture on PCA+LDA approach)
        """
        pass

    def transform(self, X):
        """
        TODO: Project data
        """
        pass

    def reconstruct(self, X_projected):
        """
        LDA 通常不用於重建，但若作業要求看 Fisherfaces，
        其實就是看 self.eigenvectors (W矩陣)
        """
        pass


# ==========================================
# 4. Kernel Models (Framework)
# ==========================================


def rbf_kernel(X1, X2, gamma=0.01):
    """
    TODO: 實作 RBF Kernel
    K(x, y) = exp(-gamma * ||x - y||^2)
    可以使用 scipy.spatial.distance.cdist 加速
    """
    pass


def polynomial_kernel(X1, X2, degree=2, coef0=1):
    """
    TODO: 實作 Polynomial Kernel
    """
    pass


class KernelPCA:
    def __init__(self, n_components=25, kernel_func=rbf_kernel):
        self.n_components = n_components
        self.kernel_func = kernel_func
        self.alphas = None  # Eigenvectors of the kernel matrix
        self.lambdas = None  # Eigenvalues
        self.X_fit = None  # Store training data for projection

    def fit(self, X):
        """
        TODO: 實作 Kernel PCA [cite: 15]
        1. Compute Kernel Matrix K
        2. Center the Kernel Matrix (Important!)
           K_centered = K - 1_n K - K 1_n + 1_n K 1_n
        3. Solve eigenvalue problem for K_centered
        4. Normalize eigenvectors (alphas)
        """
        pass

    def transform(self, X):
        """
        TODO: Project new data
        Need to compute kernel between X (new) and self.X_fit (train)
        Then project using self.alphas
        """
        pass


class KernelLDA:
    def __init__(self, n_components=14, kernel_func=rbf_kernel):
        # LDA components usually <= C-1
        self.n_components = n_components
        self.kernel_func = kernel_func
        self.alphas = None
        self.X_fit = None
        self.y_fit = None

    def fit(self, X, y):
        """
        TODO: 實作 Kernel LDA [cite: 15]
        1. Compute Kernel Matrix K
        2. Compute Kernel Scatter Matrices (M and N) within feature space
        3. Solve generalized eigenvalue problem
        """
        pass

    def transform(self, X):
        """
        TODO: Project data using alphas and kernel values
        """
        pass


# ==========================================
# 5. Classifier
# ==========================================


def knn_predict(X_train, y_train, X_test, k=3):
    """
    TODO: 實作 k-Nearest Neighbors [cite: 14]
    1. Compute distances between X_test and X_train (use cdist)
    2. Find indices of k smallest distances
    3. Vote for the most common label
    """
    pass


# ==========================================
# 6. Main Execution Block
# ==========================================

if __name__ == "__main__":
    # 填入你的資料集路徑
    DATA_DIR = "./Yale_Face_Database"

    # 1. Load Data
    X, y = load_data(DATA_DIR)

    if len(X) > 0:
        X_train, y_train, X_test, y_test = split_train_test(X, y)
        print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

        # Example Pipeline for PCA (You need to fill in the class methods first)
        # ---------------------------------------------------------
        # print("Running PCA...")
        # pca = PCA(n_components=25)
        # pca.fit(X_train)

        # # Show Eigenfaces
        # plot_faces(pca.eigenvectors, 5, 5, "Top 25 Eigenfaces")

        # # Reconstruction
        # chosen_idx = np.random.choice(len(X_train), 10, replace=False)
        # X_subset = X_train[chosen_idx]
        # X_proj = pca.transform(X_subset)
        # X_rec = pca.inverse_transform(X_proj)
        # plot_reconstruction(X_subset, X_rec)

        # # Recognition (KNN)
        # X_train_proj = pca.transform(X_train)
        # X_test_proj = pca.transform(X_test)
        # y_pred = knn_predict(X_train_proj, y_train, X_test_proj, k=3)
        # acc = np.mean(y_pred == y_test)
        # print(f"PCA Recognition Accuracy: {acc:.2f}")

        # 重複上述步驟於 LDA, Kernel PCA, Kernel LDA...

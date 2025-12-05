#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import struct  # 用於解析二進位檔案
import os  # 用於檢查檔案是否存在
from scipy.special import logsumexp


class EMClusterer:

    def __init__(self, n_clusters=10, n_features=784):
        self.n_clusters = n_clusters
        self.n_features = n_features

        self.prior = np.ones(n_clusters) / n_clusters
        self.P_ = np.random.uniform(0.25, 0.75, size=(n_features, n_clusters))
        self.cluster_labels_ = np.zeros(n_clusters, dtype=int)

    def fit(self, X, y_true, max_iter=10, tolerance=1e-4):
        log_likelihood_old = -np.inf

        for i in range(max_iter):
            # --- E-Step ---
            W, log_likelihood_new = self._e_step(X)

            # --- M-Step ---
            self._m_step(X, W)

            difference = log_likelihood_new - log_likelihood_old
            print(f"No. of Iteration: {i+1}, Difference: {difference:.10f}")

            # 印出第一次和最後一次的 "imagination"
            if i == 0 or (i == max_iter - 1):
                self.print_imaginations(labeled=False)

            if np.abs(difference) < tolerance and i > 0:
                print("Convergence reached.")
                break
            log_likelihood_old = log_likelihood_new

        print("\nTraining complete. Assigning labels...")

        self._assign_labels(X, y_true)

        self.print_imaginations(labeled=True)

        self.calculate_all_metrics(X, y_true)

        print(f"\nTotal iteration to converge: {i+1}")  #

    def _e_step(self, X):
        log_P = np.log(self.P_)
        log_one_minus_P = np.log(1 - self.P_)
        log_prob_ik = X @ log_P + (1 - X) @ log_one_minus_P

        log_prob_unnormalized = log_prob_ik + np.log(self.prior)
        log_likelihood = np.sum(logsumexp(log_prob_unnormalized, axis=1))

        log_W = log_prob_unnormalized - logsumexp(log_prob_unnormalized, axis=1, keepdims=True)
        W = np.exp(log_W)

        return W, log_likelihood

    def _m_step(self, X, W):
        N = X.shape[0]
        Nk = np.sum(W, axis=0)
        self.prior = Nk / N
        Nk_safe = np.maximum(Nk, 1e-10)
        self.P_ = (X.T @ W) / Nk_safe
        self.P_ = np.clip(self.P_, 1e-6, 1 - 1e-6)

    def _assign_labels(self, X, y_true):
        W, _ = self._e_step(X)
        cluster_preds = np.argmax(W, axis=1)

        print("Label assignment (Cluster index -> Digit label):")

        for k in range(self.n_clusters):
            labels_in_cluster = y_true[cluster_preds == k]
            most_common_label = 0

            if labels_in_cluster.size > 0:
                counts = np.bincount(labels_in_cluster, minlength=10)
                most_common_label = np.argmax(counts)
            self.cluster_labels_[k] = most_common_label
        print(self.cluster_labels_)

    def calculate_all_metrics(self, X, y_true):
        W, _ = self._e_step(X)
        cluster_preds = np.argmax(W, axis=1)
        digit_preds = self.cluster_labels_[cluster_preds]

        total_error = 0
        for digit in range(10):
            print(f"\nConfusion Matrix {digit}:")  #

            # 2. 建立 "is number X" vs "isn't number X" 的
            is_digit_true = y_true == digit
            is_digit_pred = digit_preds == digit

            # 3. 計算 TP, TN, FP, FN
            tp = np.sum(is_digit_true & is_digit_pred)
            tn = np.sum(~is_digit_true & ~is_digit_pred)
            fp = np.sum(~is_digit_true & is_digit_pred)
            fn = np.sum(is_digit_true & ~is_digit_pred)
            total_error += fp + fn

            # 4. 印出 CM
            # 格式參考 [cite: 341-348]
            print(f"                   Predict number {digit} | Predict not number {digit}")
            print(f"Is number {digit}      | {tp:^19} | {fn:^19}")
            print(f"Isn't number {digit}   | {fp:^19} | {tn:^19}")

            # 5. 計算 Sensitivity, Specificity
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            print(f"\nSensitivity (Successfully predict number {digit}): {sensitivity:.5f}")  #
            print(f"Specificity (Successfully predict not number {digit}): {specificity:.5f}")  #

            error_rate = np.mean(y_true != digit_preds)
            print(f"\nTotal error rate: {error_rate}")

    def print_imaginations(self, labeled=False):
        print("\n" + "-" * 30)
        print("Printing Imaginations...")
        print("-" * 30)

        for k in range(self.n_clusters):
            # 決定標籤
            if labeled:
                label_name = f"labeled class {self.cluster_labels_[k]}:"  #
            else:
                label_name = f"class {k}:"  #

            print(f"\n{label_name}")

            # 取得 P 參數並重塑
            cluster_p = self.P_[:, k]
            # Binarize P (e.g., > 0.5) to match the 0/1 output format
            image = (cluster_p > 0.5).astype(int).reshape(28, 28)

            # 印出 28x28 圖像 [cite: 229-253]
            for row in image:
                print("".join(map(str, row)))


def load_mnist_data(image_file, label_file):
    # 檢查檔案是否存在
    if not os.path.exists(image_file):
        print(f"Error: Image file not found at {image_file}")
        print("Please download 'train-images-idx3-ubyte' from the E3 system.")[cite:387]
        return None, None
    if not os.path.exists(label_file):
        print(f"Error: Label file not found at {label_file}")
        print("Please download 'train-labels-idx1-ubyte' from the E3 system.")[cite:387]
        return None, None

    # --- 1. 讀取標籤檔案 (train-labels-idx1-ubyte) --- [cite: 400]
    with open(label_file, "rb") as f_label:
        # 讀取 header (Magic number, Number of labels)
        # '>II' 表示 2 個 32-bit unsigned int, big-endian [cite: 401, 402]
        magic, n_labels = struct.unpack(">II", f_label.read(8))
        if magic != 2049:  # 0x00000801 [cite: 401]
            raise ValueError(f"Label file magic number error: {magic}")

        # 讀取所有標籤
        y_train = np.fromfile(f_label, dtype=np.uint8)  # [cite: 402]
        if len(y_train) != n_labels:
            raise ValueError("Mismatch in number of labels.")

    # --- 2. 讀取影像檔案 (train-images-idx3-ubyte) --- [cite: 398]
    with open(image_file, "rb") as f_image:
        # 讀取 header (Magic, N_images, N_rows, N_cols)
        # '>IIII' 表示 4 個 32-bit unsigned int, big-endian [cite: 399]
        magic, n_images, n_rows, n_cols = struct.unpack(">IIII", f_image.read(16))
        if magic != 2051:  # 0x00000803 [cite: 399]
            raise ValueError(f"Image file magic number error: {magic}")
        if n_images != n_labels:
            raise ValueError("Image and label count mismatch.")
        if n_rows != 28 or n_cols != 28:  # [cite: 399]
            raise ValueError("Image dimensions are not 28x28.")

        # 讀取所有像素
        n_features = n_rows * n_cols
        X_train_raw = np.fromfile(f_image, dtype=np.uint8)  # [cite: 399]
        X_train_flat = X_train_raw.reshape(n_images, n_features)

    # --- 3. 根據 EM (HW4) 的要求進行二元化 ---
    # "Binning the gray level value into two bins"
    X_train_bin = (X_train_flat > 120).astype(int)

    print(f"Loaded {n_images} training samples from ubyte files.")
    print(f"Data shape (binarized): {X_train_bin.shape}")

    return X_train_bin, y_train


def main():
    IMAGE_FILE = "/Users/timchen/program/machine_learning/HW4/data/train-images.idx3-ubyte_."
    LABEL_FILE = "/Users/timchen/program/machine_learning/HW4/data/train-labels.idx1-ubyte_."

    X_train, y_train = load_mnist_data(IMAGE_FILE, LABEL_FILE)

    if X_train is None:
        print("Failed to load data. Exiting.")
        return

    em_model = EMClusterer(n_clusters=10, n_features=784)

    em_model.fit(X_train, y_train)


if __name__ == "__main__":
    # 為了讓隨機初始化可重現 (方便 debug)
    np.random.seed(42)
    main()

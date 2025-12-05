import sys
import struct
import math
import argparse
import numpy as np


def load_mnist_images(filepath):
    """
    讀取 MNIST 影像檔 (idx3-ubyte) [cite: 6, 9, 17, 18]
    """
    with open(filepath, "rb") as f:
        # 讀取標頭 (Magic number, number of images, number of rows, number of columns) [cite: 18]
        # >IIII 表示用 big-endian 讀取 4 個 unsigned integer [cite: 9]
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"錯誤的 Magic number！應為 2051，但讀到 {magic}")

        # 讀取所有像素資料
        # np.fromfile 是高效讀取二進位資料到 numpy array 的方法
        # dtype=np.uint8 代表讀取 unsigned byte (0-255) [cite: 18]
        image_data = np.fromfile(f, dtype=np.uint8)

        # 將一維陣列重塑為 (影像數量, 每個影像的像素數)
        images = image_data.reshape(num_images, num_rows * num_cols)
        print(f"成功讀取 {num_images} 張影像，每張大小為 {num_rows}x{num_cols}。")
        return images


def load_mnist_labels(filepath):
    """
    讀取 MNIST 標籤檔 (idx1-ubyte) [cite: 11, 19, 21]
    """
    with open(filepath, "rb") as f:
        # 讀取標頭 (Magic number, number of items) [cite: 20, 21]
        magic, num_items = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"錯誤的 Magic number！應為 2049，但讀到 {magic}")

        # 讀取所有標籤資料 [cite: 21]
        labels = np.fromfile(f, dtype=np.uint8)

        print(f"成功讀取 {num_items} 個標籤。")
        return labels


class NaiveBayesClassifier:
    def __init__(self, mode):
        """
        初始化分類器
        mode: 'discrete' 或 'continuous' [cite: 15, 16]
        """
        if mode not in ["discrete", "continuous"]:
            raise ValueError("模式必須是 'discrete' 或 'continuous'")
        self.mode = mode

        self.priors = np.zeros(10)  # P(C=c)，每個類別的先驗機率
        self.log_likelihoods = None  # P(X_i|C=c)，概似度

    def train(self, train_images, train_labels):
        """
        訓練 Naive Bayes 分類器
        """
        print(f"開始以 {self.mode} 模式進行訓練...")
        num_samples, num_features = train_images.shape
        num_classes = 10
        for c in range(num_classes):
            self.priors[c] = np.sum(train_labels == c) / num_samples

        if self.mode == "discrete":
            # (類別, 特徵, bin)
            self.log_likelihoods = np.full((10, num_features, 32), 1e-5)  # 加上 pseudocount
            for image in range(num_samples):
                label = train_labels[image]
                for pixel_idx in range(num_features):
                    pixel_value = train_images[image][pixel_idx]
                    bin_index = pixel_value // 8
                    self.log_likelihoods[label, pixel_idx, bin_index] += 1
            # 將次數轉為機率
            for c in range(10):
                class_count = np.sum(train_labels == c)
                self.log_likelihoods[c, :, :] /= class_count
                self.priors[c] = class_count / num_samples
            self.log_likelihoods = np.log(self.log_likelihoods)

        else:  # continuous
            self.parameters = np.zeros((num_classes, num_features, 2))

            for c in range(num_classes):
                images_in_class = train_images[train_labels == c]
                mean = np.mean(images_in_class, axis=0)
                var = np.var(images_in_class, axis=0)

                self.parameters[c, :, 0] = mean
                self.parameters[c, :, 1] = var

            self.parameters[:, :, 1] = np.maximum(self.parameters[:, :, 1], 1000)
            self.log_likelihoods = self.parameters

        print("訓練完成。")

    def predict(self, image):
        """
        對單一影像進行預測，並回傳後驗機率的對數值
        """
        log_posteriors = np.log(self.priors + 1e-10)  # 加上極小值避免 log(0)
        for c in range(10):
            log_likelihood_sum = 0
            for i in range(image.shape[0]):
                pixel_value = image[i]
                if self.mode == "discrete":
                    bin_index = pixel_value // 8
                    log_likelihood = self.log_likelihoods[c, i, bin_index] + 1e-10  # 避免 log(0)
                    log_likelihood_sum += log_likelihood

                else:  # continuous
                    mean = self.log_likelihoods[c, i, 0]
                    variance = self.log_likelihoods[c, i, 1]
                    log_likelihood = -0.5 * np.log(2 * np.pi * variance) - ((pixel_value - mean) ** 2) / (2 * variance)
                    log_likelihood_sum += log_likelihood

            log_posteriors[c] += log_likelihood_sum

        prediction = np.argmax(log_posteriors)

        M = np.max(log_posteriors)
        shifted_logs = log_posteriors - M
        exps = np.exp(shifted_logs)
        sum_exps = np.sum(exps)
        normalized_posteriors = exps / sum_exps

        return normalized_posteriors, prediction

    def test(self, test_images, test_labels):
        """
        測試分類器並回報錯誤率
        """
        print("\n開始測試...")
        correct_predictions = 0
        num_test_samples = test_images.shape[0]

        for i in range(num_test_samples):
            image = test_images[i]
            label = test_labels[i]

            posteriors, prediction = self.predict(image)

            if i < 5:  # 只印出前幾筆的詳細資料作為範例
                print("Posterior (in log scale):")  # 題目要求 log scale，但範例輸出不是 [cite: 23, 37]
                # 這裡我們印出正規化後的機率
                for c, p in enumerate(posteriors):
                    print(f"{c}: {p}")
                print(f"Prediction: {prediction}, Ans: {label}\n")

            if prediction == label:
                correct_predictions += 1
            if (i + 1) % 1000 == 0:
                print(f"已測試 {i + 1}/{num_test_samples} 筆資料...")
                print(f"目前正確率: {correct_predictions / (i + 1):.4f}")

        error_rate = 1.0 - (correct_predictions / num_test_samples)
        return error_rate

    def visualize_imagination(self):
        """
        視覺化分類器對每個數字的「想像」 [cite: 25]
        """
        print("Imagination of numbers in Bayesian classifier:")
        for c in range(10):
            print(f"{c}:")
            imagination_image = np.zeros(784)
            if self.mode == "continuous":
                # 取出所有像素的平均值
                means = self.log_likelihoods[c, :, 0]
                imagination_image[means >= 128] = 1
            else:  # discrete
                for i in range(784):
                    white_pixel_score = 0
                    black_pixel_score = 0
                    for bin_idx in range(32):
                        if bin_idx < 16:
                            white_pixel_score += self.log_likelihoods[c, i, bin_idx]
                        else:
                            black_pixel_score += self.log_likelihoods[c, i, bin_idx]
                    if white_pixel_score > black_pixel_score:
                        imagination_image[i] = 0
                    else:
                        imagination_image[i] = 1

            # 將 784 的一維陣列轉為 28x28 並印出
            for i in range(28):
                line = "".join(map(str, map(int, imagination_image[i * 28 : (i + 1) * 28])))
                print(line)
            print()


def main():
    parser = argparse.ArgumentParser(description="實作 Naive Bayes 分類器來辨識 MNIST 手寫數字")
    parser.add_argument("train_images_path", help="訓練影像檔路徑 (train-images-idx3-ubyte)")
    parser.add_argument("train_labels_path", help="訓練標籤檔路徑 (train-labels-idx1-ubyte)")
    parser.add_argument("test_images_path", help="測試影像檔路徑 (t10k-images-idx3-ubyte)")
    parser.add_argument("test_labels_path", help="測試標籤檔路徑 (t10k-labels-idx1-ubyte)")
    parser.add_argument("mode", type=int, choices=[0, 1], help="切換模式 (0: discrete, 1: continuous)")
    args = parser.parse_args()

    # 載入資料
    print("正在載入 MNIST 資料...")
    train_images = load_mnist_images(args.train_images_path)
    train_labels = load_mnist_labels(args.train_labels_path)
    test_images = load_mnist_images(args.test_images_path)
    test_labels = load_mnist_labels(args.test_labels_path)

    mode_str = "discrete" if args.mode == 0 else "continuous"

    # 建立、訓練與測試模型
    classifier = NaiveBayesClassifier(mode=mode_str)
    classifier.train(train_images, train_labels)
    error_rate = classifier.test(test_images, test_labels)

    # 視覺化結果
    classifier.visualize_imagination()

    # 回報最終錯誤率
    print(f"Final error rate: {error_rate:.4f}")


if __name__ == "__main__":
    main()

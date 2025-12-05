import argparse
import math
from scipy.special import comb


def calculate_binomial_likelihood(num_trials, num_successes):
    if num_trials == 0:
        return 1.0

    p_hat = num_successes / num_trials

    if p_hat == 0.0 or p_hat == 1.0:
        return 1.0

    log_likelihood = (
        math.log(comb(num_trials, num_successes, exact=True))
        + num_successes * math.log(p_hat)
        + (num_trials - num_successes) * math.log(1 - p_hat)
    )

    return math.exp(log_likelihood)


def online_learning(filepath, initial_a, initial_b):
    # 初始化 Beta prior 參數
    a = initial_a
    b = initial_b

    case_num = 1

    with open(filepath, "r") as f:
        for line in f:
            # 去除每行結尾的換行符
            sequence = line.strip()
            if not sequence:
                continue

            print(f"case {case_num}: {sequence}")

            # 從序列中計算 trial 和 success 的次數
            num_ones = sequence.count("1")  # 成功的次數 (k)
            num_zeros = sequence.count("0")  # 失敗的次數
            num_trials = len(sequence)  # 總試驗次數 (n)

            likelihood = calculate_binomial_likelihood(num_trials, num_ones)
            print(f"Likelihood: {likelihood}")

            print(f"Beta prior:     a={a} b={b}")

            a += num_ones
            b += num_zeros

            print(f"Beta posterior: a={a} b={b}\n")

            case_num += 1


def main():
    parser = argparse.ArgumentParser(description="使用 Beta-Binomial 共軛進行線上學習")
    parser.add_argument("filepath", help="包含二元序列結果的輸入檔案路徑")
    parser.add_argument("a", type=int, help="Beta prior 的初始 a 參數")
    parser.add_argument("b", type=int, help="Beta prior 的初始 b 參數")
    args = parser.parse_args()

    # 執行線上學習主程式
    online_learning(args.filepath, args.a, args.b)


if __name__ == "__main__":
    main()

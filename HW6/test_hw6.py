import itertools
import subprocess
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_experiment(params):
    """
    Worker 函數：負責執行單次實驗
    """
    # 組合指令
    cmd = [sys.executable, "hw6.py"]

    # 用來在主程式顯示現在跑到哪組參數 (僅供識別用)
    # 例如: "image:image1.png, k:2..."
    param_str = ", ".join([f"{k}:{v}" for k, v in params.items()])

    for key, value in params.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    try:
        # --- 關鍵修改 ---
        # stdout=subprocess.DEVNULL: 把標準輸出丟掉
        # stderr=subprocess.DEVNULL: 把錯誤訊息丟掉 (如果要保留錯誤訊息在螢幕上，這行可以拿掉)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        return f"成功: [{param_str}]"

    except subprocess.CalledProcessError:
        # 即使 output 丟掉了，如果程式 exit code 非 0，還是會被這裡抓到
        return f"!!! 失敗 !!!: [{param_str}]"
    except Exception as e:
        return f"!!! 未知錯誤 !!!: [{param_str}] - {e}"


if __name__ == "__main__":

    # 1. 參數設定
    param_grid = {
        "image": ["image1.png", "image2.png"],
        "k": [2, 3, 4, 5],
        "init": ["random", "k_means++"],
        "gamma_c": [5, 10, 15],
    }

    # 2. 產生組合
    keys = param_grid.keys()
    values = param_grid.values()
    all_experiments = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    total_jobs = len(all_experiments)

    # 3. 設定核心數
    # 這個作業的測試要把我的mac和我的耐心其中一個搞炸啦
    max_workers = 3

    print(f"預計執行 {total_jobs} 次實驗，使用 {max_workers} 個核心平行運算...\n")

    # 4. 開始執行
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {executor.submit(run_experiment, params): params for params in all_experiments}

        for i, future in enumerate(as_completed(future_to_params)):
            result = future.result()
            # 這裡的 print 是唯一會出現在螢幕上的東西
            print(f"[{i+1}/{total_jobs}] {result}")

    print("\n所有排列組合測試完畢。")

"""計測用の共通関数（src/bench.py）

使い方:
    from bench import bench, save_result

    x = torch.randn(1, 3, 224, 224, device="cuda")
    def run():
        with torch.inference_mode():
            model(x)

    r = bench(run)
    save_result(r, model_name="vit_base_patch16_224", device="cuda",
                precision="fp32", batch_size=1, note="baseline")
"""
import csv
import os
import statistics
import time
from datetime import datetime

import torch

# CSVの列の順番（ここを変えると既存のCSVと列がずれるので注意）
CSV_COLUMNS = [
    "timestamp",      # 計測した日時
    "model",          # モデル名
    "device",         # cpu / cuda
    "device_name",    # GPU名やCPUのスレッド数など
    "precision",      # fp32 / fp16 / int8 など
    "batch_size",
    "median_ms",      # レイテンシの中央値（ミリ秒） 推論全体時間がどれくらいかかったかの測定
    "p90_ms",         # レイテンシのp90（ミリ秒）
    "throughput",     # 1秒あたりの処理枚数（batch_size / 中央値）
    "n_iter",         # 計測した回数
    "torch_version",
    "note",           # 手法の名前など自由メモ（例：baseline, sdpa, compile）
]


def bench(fn, n_warmup=10, n_iter=50, cuda=True):
    """引数なしの関数 fn を n_iter 回実行して、レイテンシの中央値とp90を返す"""
    for _ in range(n_warmup):
        fn()

    times = []
    for _ in range(n_iter):
        if cuda:
            torch.cuda.synchronize() # 「GPU上の、これまでの処理が全部終わるまで、CPUはここで待つ」という命令
        t = time.perf_counter() # CPU側のタイマーで計測している
        fn()
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)

    times.sort()
    return {
        "median_ms": statistics.median(times),
        "p90_ms": times[int(len(times) * 0.9)],
        "n_iter": n_iter,
    }


def _device_name(device):
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return f"cpu ({torch.get_num_threads()} threads)"


def save_result(result, model_name, device, precision, batch_size,
                note="", path="results/benchmarks.csv"):
    """bench() の結果を CSV に1行追記する。ファイルがなければヘッダー付きで作る"""
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "device": device,
        "device_name": _device_name(device),
        "precision": precision,
        "batch_size": batch_size,
        "median_ms": round(result["median_ms"], 4),
        "p90_ms": round(result["p90_ms"], 4),
        "throughput": round(batch_size / (result["median_ms"] / 1000), 2),
        "n_iter": result.get("n_iter", ""),
        "torch_version": torch.__version__,
        "note": note,
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    print(f"[saved] {model_name} | {device} | {precision} | bs={batch_size} | "
          f"median {row['median_ms']:.3f} ms | p90 {row['p90_ms']:.3f} ms | "
          f"{row['throughput']:.1f} /s | {note}")
    return row

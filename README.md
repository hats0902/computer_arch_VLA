# 7日間 学習計画：コンピュータアーキテクチャ × 高速化 × VLA
## 〜 ViT → VLAの推論高速化レポートを作りながら学ぶ 〜

## ゴール

1. コンピュータアーキテクチャの概要を理解している
2. 高速化に必要な要素を理解している（CPU / GPU）
3. VLAの基本アーキテクチャ（Vision / LLM / Planner）を理解している
4. **【成果物】** 推論を計測・分析・高速化し、「なぜ速くなったか」を説明したレポートをGitHubで公開している
   - ViT-B/16で、PyTorchでの高速化と **ONNX → TensorRT（FP16 / INT8）へのエッジ向け移植** を一通り行う
   - 同じ手順を小型VLA（SmolVLA）に当てはめ、VLAのどこが重いかを分析する

## 毎日の構成

各Dayは次の5つで構成している。上から順に進める。

| 項目 | 内容 |
|---|---|
| **① 読む範囲** | どの本の、どの章・節を、どの深さで読むか（精読／流し読み／飛ばす） |
| **② 本の内容と実装の対応** | 読んだ内容のうち、どれを実装で確かめ、どれは確かめないか |
| **③ 実装に必要な理解のレベル** | 「ここまで分かっていれば実装に進んでよい」という目安 |
| **④ 実装ステップ** | 具体的な手順と、その日に残す成果物 |
| **⑤ 理解チェックポイント** | 1日の最後に、何も見ずに口頭で説明できるか確認する |

**読み方の記号**
- **精読**：図を自分で描き直せるまで読む
- **流し読み**：見出しと図と太字を追い、「こういう話がある」と分かれば十分
- **飛ばす**：今回のゴールには不要

## 前提

- 1日あたり3〜4時間（読む1.5時間＋実装2時間程度）。7日で終わらなかった作業のために **予備日を2日** 見ておく
- 実装は **Google Colab（T4 GPU）** で行う。手元のM1 Mac（メモリ8GB）は、本を読む、READMEやコードを編集する、CPUの軽い実験に使う
- 教材はできるだけ日本語を使う。日本語の教材がないもの（Roofline、TensorRT、VLA）だけ英語を使い、🇬🇧 マークを付けている

## 教材

| 略称 | 教材 | 形式 | 使う日 |
|---|---|---|---|
| **キタミ式** | 『キタミ式イラストIT塾 基本情報技術者』（技術評論社） | 紙／電子版 | Day 1〜2 |
| **CPU本** | 『プログラマーのためのCPU入門』（ラムダノート） | PDF版あり | Day 1〜2, 4 |
| **GPU本** | 『[増補改訂]GPUを支える技術』（技術評論社） | 電子版あり | Day 3〜6 |
| **ViT本** | 『Vision Transformer入門』（技術評論社） | 紙／電子版 | Day 6〜7 |
| 🇬🇧 Horace He | "Making Deep Learning Go Brrrr From First Principles"（horace.io） | Web | Day 4 |
| 🇬🇧 TensorRT | NVIDIA TensorRT Developer Guide / Support Matrix | Web | Day 6 |
| 🇬🇧 VLA論文 | RT-2 / OpenVLA / π0 / SmolVLA（arXiv:2506.01844） | arXiv | Day 7 |
| Zenn / Qiita | 各日に検索キーワードを記載 | Web | 全日 |

※ 新装版のフルカラーのコンピュータアーキテクチャ入門書が手元にあれば、キタミ式の代わりに第7・8章（Day 1）、第10章（Day 2）を使ってよい

## 題材

| Day | 題材 | 理由 |
|---|---|---|
| 1〜6 | **ViT-B/16**（`timm` の `vit_base_patch16_224`、約8,600万パラメータ） | 経験のあるモデルで中身に迷わない。行列積（compute-bound）とLayerNorm・GELU・softmax（memory-bound）が混ざっていてRooflineの題材に向く。ImageNetの学習済みモデルなので **Imagenetteで精度を評価できる** |
| 6後半〜7 | **SmolVLA**（`lerobot/smolvla_base`、約4.5億パラメータ） | Vision（ViT系）＋ LLM ＋ Planner（flow matchingのaction expert）がそろった小型VLA |

## 使うGPUとツール

| ツール | T4での扱い | 使う日 |
|---|---|---|
| PyTorch（FP16、SDPA、`torch.compile`） | ◎ | Day 2〜5 |
| ONNX Runtime | ◎ | Day 6（ONNXの出力確認） |
| TensorRT（通常版） | ◎ 正式対応（FP16 / INT8のTensor Core） | Day 6 |
| TensorRT-LLM | △ T4はコミュニティサポートのみ | 使わない |
| TensorRT-Edge-LLM、ReCogDrive | × T4では動かない／載らない | 「その先」でAWSのg5 / g6を使う |

- T4は **BF16、FP8、FlashAttention-2に対応していない**。FP16とINT8を使う

## 全体像

| Day | テーマ | 主に読む本 | 実装の成果物 |
|---|---|---|---|
| 1 | メモリ階層とキャッシュ | キタミ式、CPU本 第5章 | ViTの重み・中間データのサイズとキャッシュの比較表、推論時間の下限 |
| 2 | CPUの高速化 | CPU本 第2〜3章、第9章 | `bench.py`、CPUのスレッド数ごとのレイテンシ、アムダールの法則での分析 |
| 3 | GPUアーキテクチャ | GPU本 第1・4・6章 | T4の仕様と実測バンド幅、CPU vs GPU、FP32 vs FP16、バッチサイズの表 |
| 4 | ボトルネック分析（Roofline） | Horace He、CPU本 第13章 | Roofline図、演算子ごとの時間の内訳 |
| 5 | 深層学習の高速化 | GPU本 第5章、解説記事 | 手法ごとのレイテンシの表とグラフ |
| 6 | TensorRT ＋ ViT → VLM | TensorRT、ViT本 第2章 | TensorRT FP16 / INT8 の比較表（速度・精度） |
| 7 | VLA | ViT本 第5章、VLA論文 | SmolVLAの部品ごとの分析、レポート |

## リポジトリ構成（最終形）

```
computer_arch_VLA/
├── README.md                  # レポート本体
├── .gitignore
├── notebooks/
│   ├── 01_vit_size_vs_cache.ipynb     # Day 1
│   ├── 02_vit_cpu_baseline.ipynb      # Day 2
│   ├── 03_vit_gpu_baseline.ipynb      # Day 3
│   ├── 04_vit_profile_roofline.ipynb  # Day 4
│   ├── 05_vit_optimizations.ipynb     # Day 5
│   ├── 06_vit_tensorrt.ipynb          # Day 6
│   └── 07_smolvla_analysis.ipynb      # Day 7
├── src/
│   ├── bench.py               # 計測の共通関数（bench, save_result）
│   ├── gpu_info.py            # GPUの仕様の取得（L2、SM数、バンド幅など）
│   ├── eval_imagenette.py     # Imagenetteでの精度評価
│   └── export_trt.py          # ONNX出力・TensorRTエンジン作成
├── results/                   # 計測結果のCSV
└── figures/                   # グラフ、Roofline図
```

- GitHubを正本にし、M1 MacとColabの両方からclone / pushする。ColabではHTTPS＋Personal Access Token（Colabのシークレットに保存）でcloneする。**トークンはセルに直接書かない**
- Colabの環境はセッションが切れると消える。**計測結果のCSVとグラフは、終了前に必ずpushする**
- モデルの重み、ONNX、TensorRTのエンジン、データセットは `.gitignore` で除外する

---

## Day 1：メモリ階層とキャッシュ

**この日のゴール**：「データはどこにあり、どれくらいの速さで届くか」を説明でき、ViTのデータがキャッシュに収まるかどうかを数字で示せる

### ① 読む範囲（約1.5時間）

| 本 | 章・節 | 読み方 | 目的 |
|---|---|---|---|
| キタミ式 | CPU・メモリ・キャッシュのページ（試験分野の「コンピュータ構成要素」） | 流し読み（30分） | 図で全体像をつかむ |
| CPU本 | 第5章 5.1 メモリアクセスに伴うパイプラインの停滞 | 精読 | メモリが遅いと何が困るか |
| CPU本 | 第5章 5.2 キャッシュメモリによる緩和 | 精読 | キャッシュの仕組み、キャッシュライン、局所性 |
| CPU本 | 第5章 5.3 初期参照ミス ／ 5.4 容量性ミス | 精読 | キャッシュが効かない2つのケース |
| CPU本 | 第5章 5.5 競合性ミス | 流し読み | 今回の実装では扱わない |
| CPU本 | 第5章 5.6 キャッシュミスの測定 | 流し読み | 測り方の考え方だけ |
| CPU本 | 第6章 仮想記憶 | 飛ばす | 今回のゴールには不要 |
| GPU本 | 第1章 1.4 CPUとGPUの違い | 流し読み | 明日以降のGPUの話の予告として |

### ② 本の内容と実装の対応

| 本で学ぶこと | 実装で確かめること |
|---|---|
| <input type="checkbox" checked>メモリ階層（レジスタ → L1 → L2 → DRAM） | <input type="checkbox" checked>T4のL1・L2・VRAMの容量を取得して並べる |
| <input type="checkbox" checked>キャッシュライン、空間的局所性 | <input type="checkbox" checked>NumPyで行方向と列方向の合計の速度を比べる |
| <input type="checkbox" checked>容量性ミス（5.4） | <input type="checkbox" checked>ViTの重みがL2に収まらない → 推論のたびにVRAMから読み直すことを、サイズの計算で示す |
| <input type="checkbox" checked>初期参照ミス（5.3） | <input type="checkbox">Day 2の計測でウォームアップを入れる理由として使う（今日は実装しない） |
| <input type="checkbox" checked>競合性ミス（5.5） | <input type="checkbox">確かめない |

### ③ 実装に必要な理解のレベル

- 「近い階層ほど小さくて速い」「DRAM（VRAM）は計算に比べて圧倒的に遅い」と言える
- キャッシュが効くのは「同じデータを再利用する」「近くのデータを続けて使う」ときだと言える
- 「キャッシュに収まらないデータは、使うたびに遠くから取ってくるしかない（容量性ミス）」と言える
- GPUではL2が全SMで共有の最後のキャッシュで、L1はSMごとに分かれている、と知っている（詳しくはDay 3）

### ④ 実装ステップ（約2時間）

1. **リポジトリを準備する**（30分）
   - GitHubでprivateリポジトリを作る（`.gitignore` はPythonのテンプレート＋ `*.onnx` `*.engine` `*.pt` `*.safetensors` `data/` などを追記）
   - M1 Macにclone、ColabからHTTPS＋トークンでcloneできることを確認する
2. **空間的局所性を体感する**（10分、M1でOK）
   ```python
   import numpy as np, time
   a = np.random.rand(8000, 8000)
   t = time.perf_counter(); a.sum(axis=1); print("行方向:", time.perf_counter() - t)
   t = time.perf_counter(); a.sum(axis=0); print("列方向:", time.perf_counter() - t)
   ```
3. **ViTを読み込んで推論する**（10分）
   ```python
   import torch, timm
   model = timm.create_model("vit_base_patch16_224", pretrained=True).to("cuda").eval()
   x = torch.randn(1, 3, 224, 224, device="cuda")
   with torch.inference_mode():
       print(model(x).shape)   # (1, 1000)
   ```
4. **GPUの仕様を取得する**（15分）
   - L2・SMの数・メモリのクロックとバス幅：NVIDIAドライバ（`libcuda.so`）の `cuDeviceGetAttribute` を `ctypes` で呼んで取得する
   - L1：APIでは取得できないので、`torch.cuda.get_device_capability()` で世代を取得し、Turing Tuning Guideの値（L1は最大64KB）を当てはめる
   - 取得するコードは `src/gpu_info.py` にまとめ、以降の日でも使い回す
5. **重みのサイズを部品ごとに出す**（20分）
   - モデル直下の部品（patch_embed / blocks / norm / head）と、`cls_token`・`pos_embed`
   - ブロック1個の中の部品（norm1 / attn.qkv / attn.proj / norm2 / mlp.fc1 / mlp.fc2）
   - FP32 / FP16 / INT8 のサイズと、L1・L2に収まるか（○✕）←これはやっていない
6. **中間データのサイズを出す**（20分）
   - forward hookで各層の出力のshapeを取り、bs=1とbs=64でサイズを計算する
   - ブロックの出力（197×768）、MLPの途中（197×3072）、attention行列（12×197×197）
7. **推論時間の下限を計算する**（5分）
   - 下限 ＝ 重みのサイズ ÷ メモリバンド幅（FP16なら約172MB ÷ 320GB/s ≒ 0.54ms）
8. **READMEに残す**（10分）：比較表、下限の値、結果から分かったこと（2〜3行）

**この日の成果物**：`notebooks/01_vit_size_vs_cache.ipynb`、`src/gpu_info.py`、`results/day1_*.csv`、READMEの比較表

### ⑤ 理解チェックポイント
- メモリ階層の各段について、容量と速度の大まかな関係を言える
- キャッシュが効く理由を、2種類の局所性を使って説明できる
- ViTの重みと中間データが、それぞれL1・L2と比べてどのくらいの大きさか言える
- 重みがL2に収まらないことから、「推論時間には下限がある」と説明できる

---

## Day 2：CPUの高速化の仕組み

**この日のゴール**：CPUが速く動く工夫と、それを邪魔するものを説明でき、スレッド数を増やしても速さが比例しない理由を自分の計測結果で説明できる

### ① 読む範囲（約1.5時間）

| 本 | 章・節 | 読み方 | 目的 |
|---|---|---|---|
| キタミ式 | パイプライン、マルチコアのページ | 流し読み（20分） | 図でイメージをつかむ |
| CPU本 | 第2章 2.1 逐次処理 〜 2.3 スーパースカラ化 | 精読 | 命令を重ねて実行する工夫 |
| CPU本 | 第2章 2.4 〜 2.5 スーパーパイプライン化 | 流し読み | 名前と概要だけ |
| CPU本 | 第3章 3.1 データ依存関係とは ／ 3.3 スーパースカラ化への影響 ／ 3.4 アウトオブオーダー実行 | 精読 | 何が並列化を妨げるか |
| CPU本 | 第3章 3.5 〜 3.7 | 流し読み | 対策の存在だけ知る |
| CPU本 | 第3章 3.8 命令レイテンシの計測実験 | 飛ばす | アセンブリの実験なので今回は不要 |
| CPU本 | 第4章 4.1 分岐命令とその課題 ／ 4.3 分岐予測による緩和 | 流し読み | 分岐予測ミスで遅くなることだけ |
| CPU本 | 第9章 9.1 マルチプロセッサの多様な方式 ／ 9.5 マルチプロセッサを活かすためには ／ 9.7 遅くする要因 | 流し読み | スレッド数の実験の背景 |
| CPU本 | 第7〜8章、第10〜12章 | 飛ばす | 今回のゴールには不要 |
| Zenn / Qiita | 「アムダールの法則」「SIMD NumPy なぜ速い」 | 1本ずつ | 実験の分析に使う |

### ② 本の内容と実装の対応

| 本で学ぶこと | 実装で確かめること |
|---|---|
| <input type="checkbox" checked>パイプライン、スーパースカラ、アウトオブオーダー（第2〜3章） | **直接は測らない**。PyTorchの内部でCPUが自動でやっていること。結果の考察で言葉として使う |
| <input type="checkbox" checked>データ依存（第3章） | ViTは「1層目の結果がないと2層目を計算できない」→ 層をまたいだ並列化ができない部分があることを、アムダールの法則の分析で使う |
| 分岐予測（第4章） | 確かめない |
| SIMD（Zenn記事） | （任意）Pythonのforループ vs NumPy の速度差 |
| マルチプロセッサ（第9章）、アムダールの法則 | **スレッド数を1, 2, 4と変えてレイテンシを測り、並列化できている割合 p を逆算する** |
| 初期参照ミス（Day 1の第5章 5.3） | 計測の前にウォームアップを入れる |

### ③ 実装に必要な理解のレベル

- アムダールの法則 S(n) = 1 / ((1 − p) + p / n) を書けて、「並列化できない部分が上限を決める」と言える
- データ依存があると、命令（や層）を同時に実行できないと言える
- 計測では「最初の数回は遅い（初期参照ミスや初期化）ので除く」「ばらつくので中央値とp90を見る」と言える

### ④ 実装ステップ（約2時間）

1. **`src/bench.py` を置く**（15分）：`bench()`（ウォームアップ＋計測、中央値とp90）と `save_result()`（CSVに追記）
2. **CPUの準備**（10分）
   - `os.cpu_count()` でColabのCPUのコア数を確認する（たいてい2コア）
   - ViTを **CPU上に** 読み込む（`.to("cuda")` しない）。`next(model.parameters()).device` で `cpu` を確認する
3. **スレッド数ごとに測る**（30分）
   ```python
   x = torch.randn(1, 3, 224, 224)
   def run():
       with torch.inference_mode():
           model(x)
   for n in [1, 2, 4]:
       torch.set_num_threads(n)
       r = bench(run, n_warmup=3, n_iter=20, cuda=False)
       save_result(r, "vit_base_patch16_224", "cpu", "fp32", 1, note=f"threads={n}")
   ```
4. **アムダールの法則で分析する**（20分）
   - 倍率 S2 ＝ 1スレッドの時間 ÷ 2スレッドの時間
   - 並列化できている割合 p ＝ (1 − 1/S2) ÷ (1 − 1/2)
   - コアを無限に増やしたときの最大倍率 ＝ 1 / (1 − p)
   - コア数より多いスレッド（2コアで4スレッド）で速くならない理由も考える
5. **（任意）M1 Macでも測る**（20分）：スレッド数1, 2, 4, 8。高性能コア4つ＋高効率コア4つなので、4を超えたところで伸び方が変わるかを見る
6. **（任意）SIMDを体感する**（10分）：同じ足し算をPythonのforループとNumPyで比べる
7. **READMEに残す**（15分）：スレッド数ごとの表、p と最大倍率、速くならない理由の考察（3〜5行）

**この日の成果物**：`src/bench.py`、`notebooks/02_vit_cpu_baseline.ipynb`、`results/benchmarks.csv`、READMEの表と考察

### ⑤ 理解チェックポイント
- パイプラインで「1命令の時間は変わらないのにスループットが上がる」理由を説明できる
- スーパースカラで同時に実行できるのは、依存関係がない命令だけだと説明できる
- アムダールの法則を式で書け、自分の計測結果から p を求めて説明できる
- 計測でウォームアップと中央値／p90を使う理由を説明できる

---

## Day 3：GPUアーキテクチャ

**この日のゴール**：GPUの構造を図で描け、CPUとの設計思想の違いと、FP16やバッチサイズで速くなる理由を自分の計測結果で説明できる

### ① 読む範囲（約1.5時間）

| 本 | 章・節 | 読み方 | 目的 |
|---|---|---|---|
| GPU本 | 第1章 1.4 CPUとGPUの違い | 精読 | 設計思想の違い |
| GPU本 | 第1章 1.6 GPUとおもな処理方式 | 流し読み | SIMD / SIMTの用語 |
| GPU本 | 第4章 4.1 GPUの並列処理方式 ／ 4.2 GPUの構造 | 精読 | SIMT、SM、warp、メモリ階層 |
| GPU本 | 第4章 4.3 〜 4.5 | 飛ばす | AMD・Armの話、エラー訂正は不要 |
| GPU本 | 第6章 6.1 GPUのデバイスメモリ ／ 6.2 CPUとGPU間のデータ伝送 | 精読 | VRAMのバンド幅、PCIe |
| GPU本 | 第5章 5.2 CUDA | 流し読み | スレッド・ブロック・グリッドの考え方だけ |
| GPU本 | 第2〜3章 | 飛ばす | グラフィックスの歴史は不要 |
| CPU本 | 付録E E.1 GPUにおける高密度化のアプローチ | 流し読み | Day 2のパイプラインの話とGPUをつなぐ |
| Zenn / Qiita | 「Tensor Core 仕組み」 | 1本 | FP16の比較の背景 |

### ② 本の内容と実装の対応

| 本で学ぶこと | 実装で確かめること |
|---|---|
| CPUとGPUの違い（1.4） | 同じViTの推論をCPU（Day 2）とGPUで比べる |
| SIMT、warp、大量のスレッドで待ち時間を隠す（4.1〜4.2） | バッチサイズを上げるとスループットが上がり、どこかで頭打ちになることを確かめる |
| デバイスメモリのバンド幅（6.1） | 大きなテンソルのコピーで、T4の **実測バンド幅** を測る。Day 1の下限を実測値で計算し直す |
| CPUとGPU間のデータ伝送（6.2） | 入力をCPUからGPUへ転送する時間を測る |
| Tensor Core（記事） | FP32 vs FP16 で速度を比べる |
| CUDAのプログラミング（5.2） | 確かめない（カーネルは書かない） |

### ③ 実装に必要な理解のレベル

- GPU → SM（T4は40個）→ warp（32スレッド）という階層を言える
- GPUは「大量のスレッドを切り替えて待ち時間を隠す」設計で、仕事が少ない（bs=1）と使い切れないと言える
- Tensor CoreはFP16などで行列積を高速に計算する専用ユニットだと言える
- GPUは非同期に動くので、時間を測るときは `torch.cuda.synchronize()` が必要だと言える

### ④ 実装ステップ（約2時間）

1. **T4の仕様をまとめる**（15分）
   - `src/gpu_info.py` でSM数・L2・バンド幅（理論値）を取得する
   - FP32とFP16（Tensor Core）のピーク性能は、NVIDIAの仕様表で調べる
   - BF16とFlashAttention-2に対応していない理由（Turing世代）も書く
2. **実測バンド幅を測る**（15分）
   ```python
   x = torch.empty(512 * 1024**2, dtype=torch.uint8, device="cuda"); y = torch.empty_like(x)
   r = bench(lambda: y.copy_(x))
   print("実測バンド幅:", 2 * x.numel() / (r["median_ms"] / 1000) / 1e9, "GB/s")  # 読み＋書きで2倍
   ```
   - 理論値（約320GB/s）と比べ、Day 1の下限を実測値で計算し直す
3. **CPU vs GPU**（15分）：bs=1、FP32で、GPUのレイテンシを測ってDay 2のCPUと比べる
4. **転送時間を測る**（10分）：CPU上の入力を `.to("cuda")` する時間だけを別に測る
5. **FP32 vs FP16**（25分）
   - `model.half()` と `x.half()` でFP16にして測る
   - レイテンシ、ピークメモリ（`torch.cuda.max_memory_allocated()`）
   - FP32の出力とのずれ：ロジットの最大誤差、top-1が一致するか
6. **バッチサイズを変える**（25分）
   - bs=1, 8, 32, 64 で、FP16のレイテンシとスループット（枚/秒）を測る
   - スループットのグラフを `figures/` に保存する
7. **Day 1の下限と比べる**（5分）：bs=1のFP16のレイテンシは、下限（約0.5ms）の何倍か
8. **READMEに残す**（10分）：T4の仕様表、比較表、グラフ、考察

**この日の成果物**：`notebooks/03_vit_gpu_baseline.ipynb`、`results/benchmarks.csv`（追記）、スループットのグラフ、READMEの表と考察

### ⑤ 理解チェックポイント
- SM / warp / スレッドの関係を図に描ける
- GPUは「大量のスレッドで待ち時間を隠す」設計だと、CPUとの対比で説明できる
- FP16で速くなった理由を「Tensor Core」と「データ量が半分」の2つの観点で説明できる
- bs=1ではGPUを使い切れず、bsを上げるとスループットが上がる理由を説明できる
- 実測のレイテンシが下限より遅い理由の仮説を言える（Day 4で確かめる）

---

## Day 4：ボトルネック分析（Roofline）★最重要日

**この日のゴール**：ある処理が「計算が律速」か「メモリが律速」かを、自分で描いたRoofline図で判断できる

### ① 読む範囲（約1.5時間）

| 教材 | 範囲 | 読み方 | 目的 |
|---|---|---|---|
| Zenn / Qiita | 「Roofline モデル」「演算強度 memory bound」 | 1〜2本 | 日本語で全体像をつかむ |
| 🇬🇧 Horace He | 全体（Compute / Memory Bandwidth / Overheadの3節） | 精読 | 3つのボトルネックの考え方 |
| GPU本 | 第6章 6.1 GPUのデバイスメモリ | 読み直し | 演算強度の観点で読み直す |
| CPU本 | 第13章 13.1 〜 13.3 高速なソフトウェアを書く際には何に注目すべきか | 流し読み | CPU側の視点での「何を優先すべきか」 |

### ② 本の内容と実装の対応

| 学ぶこと | 実装で確かめること |
|---|---|
| compute / memory / overhead の3分類 | ViTの演算子ごとの時間を、3つのどれに当たるか分類する |
| 演算強度 ＝ FLOPs ÷ バイト数 | Linear、LayerNorm、GELU、softmaxの演算強度を手計算する |
| Roofline図 | 行列積と要素演算を実測して、T4のRoofline図を自分で描く |
| バッチサイズで演算強度が変わる | bs=1とbs=64で、図の上の点の位置がどう動くかを見る |

### ③ 実装に必要な理解のレベル

- 演算強度の定義を言え、行列積は高く、要素ごとの演算は低いと言える
- Roofline図の横軸・縦軸・屋根の折れ点（ピークFLOPS ÷ バンド幅）の意味を言える
- 折れ点より左はmemory-bound、右はcompute-boundと言える

### ④ 実装ステップ（約2時間）

1. **Roofline図の土台を作る**（40分）
   - 行列積（FP16、N=128, 256, …, 8192）の時間を測り、達成FLOPS（2N³ ÷ 時間）と演算強度（2N³ ÷ 3N²×2バイト）を計算する
   - 要素ごとの演算（`x + y`、GELU）も同様に測る
   - 屋根の線（ピークFLOPSと実測バンド幅）を引き、点をプロットする
2. **ViTの演算子ごとの時間を測る**（30分）
   - `torch.profiler` でFP16、bs=1の推論を記録する
   - 上位10個の演算子と、種類ごと（行列積／LayerNorm／GELU／softmax／その他）の時間の割合を表にする
3. **ViTの演算の演算強度を手計算する**（20分）
   - Linear（768→3072）、LayerNorm、GELU、softmaxのFLOPsとバイト数
   - 手順1の図にプロットし、memory-boundかcompute-boundか判定する
4. **bs=64で同じことをする**（15分）：内訳と図の上の位置の変化を見る
5. **振り返り**（5分）：以前GPUのFLOPSから学習時間を見積もったとき、実測とずれた原因を、この図を使って書き直す
6. **READMEに残す**（10分）：Roofline図、演算子の内訳の表、判定結果、考察
7. （任意）MobileNetV2でも同じプロファイルをして、CNNとの違いを見る

**この日の成果物**：`notebooks/04_vit_profile_roofline.ipynb`、`figures/roofline.png`、READMEの図と表

### ⑤ 理解チェックポイント
- 3つのボトルネックを、それぞれViTの例を挙げて説明できる
- Roofline図を何も見ずに描き、折れ点の意味を説明できる
- ViTのどの演算がmemory-boundで、どれがcompute-boundかを、自分の結果で言える
- bsを上げると図の上の点がどう動くか説明できる
- FLOPSだけで処理時間を見積もると外れる理由を説明できる

---

## Day 5：深層学習の高速化手法

**この日のゴール**：主な高速化手法が「Day 4のどのボトルネックを解消しているか」を、自分の計測結果で説明できる

### ① 読む範囲（約1.5時間）

| 教材 | 範囲 | 読み方 | 目的 |
|---|---|---|---|
| GPU本 | 第5章 5.4 GPUプログラムの最適化 | 精読 | coalesced access、shared memoryの使い方（tilingの考え方） |
| GPU本 | 第7章 7.1 ディープラーニングとGPU | 流し読み | DLでGPUが使われる理由 |
| Zenn / Qiita | 「FlashAttention 解説」「kernel fusion」「torch.compile 仕組み」「CUDA Graphs」 | 各1本 | 今日試す手法の仕組み |
| 🇬🇧 FlashAttention論文 | IntroductionとFigure 1だけ | 流し読み | 日本語記事の後なら図だけで分かる |

### ② 本の内容と実装の対応

| 手法 | 何を減らすか | 効くボトルネック | 実装で確かめること |
|---|---|---|---|
| shared memoryの活用・tiling（5.4） | VRAMへのアクセス回数 | memory | 直接は書かない。SDPA・FlashAttentionの中身として理解する |
| SDPA / FlashAttention | attention行列をVRAMに書かない | memory | SDPAあり／なしの比較。解像度を上げてトークン数を増やしたときの差 |
| kernel fusion（`torch.compile`） | VRAMへの書き戻し・カーネル起動回数 | memory / overhead | `torch.compile` の前後の比較、プロファイルでLayerNormやGELUが融合されたか |
| CUDA Graphs | カーネル起動の待ち時間 | overhead | `mode="reduce-overhead"` の比較（特にbs=1） |
| batching | 重み1回の読み込みあたりの計算を増やす | memory | 各手法の効果がbsで変わるか |
| 量子化（INT8） | 転送バイト数（INT8で計算できれば計算も） | memory / compute | 今日は扱わない（Day 6でTensorRTを使う） |
| KV cache | 同じ部分の再計算 | compute | 今日は扱わない（Day 7で登場） |

### ③ 実装に必要な理解のレベル

- 表の各手法について「何を減らしているか」を1行で言える
- FlashAttentionは「計算量は同じでも、VRAMとの往復を減らすから速い」と言える
- bs=1ではoverhead（カーネル起動）の割合が大きくなりやすいと言える

### ④ 実装ステップ（約2時間）

Day 3のFP16（bs=1）を基準に、**手法を1つずつ足しては計測**する。毎回「レイテンシ（中央値/p90）・ピークメモリ・FP32の出力とのずれ」を記録し、`save_result()` の `note` に手法名を書く。

1. **SDPAあり／なし**（30分）
   - timmのViTには、SDPAを使うかどうかの設定がある（バージョンによって方法が違うので、timmのドキュメントで確認する）
   - 224×224（197トークン）と384×384（577トークン、`vit_base_patch16_384`）で比べ、トークン数が増えると差がどう変わるか見る
2. **`torch.compile`**（25分）
   - 初回のコンパイル時間と、2回目以降の速度を分けて記録する
   - プロファイルを取り、Day 4と比べて演算子の数がどう変わったか見る
3. **CUDA Graphs**（15分）：`torch.compile(mode="reduce-overhead")` をbs=1で試す
4. **bsを変えて効果を比べる**（25分）：bs=1, 8, 32, 64 で、各手法の効果がどう変わるか
5. **まとめる**（15分）：手法ごとの表とグラフ、**最速の構成** を記録する（Day 6でTensorRTと比べる）
6. **READMEに残す**（10分）：表、グラフ、「効いた手法と効かなかった手法」をRooflineの言葉で説明

**この日の成果物**：`notebooks/05_vit_optimizations.ipynb`、手法ごとのグラフ、READMEの表と考察

### ⑤ 理解チェックポイント
- 表の各手法について「何を減らしているか」を自分の言葉で言える
- FlashAttentionが「計算量は同じなのに速い」理由を、メモリ階層を使って説明できる
- `torch.compile` やCUDA Graphsで速くなった（ならなかった）理由を説明できる
- bs=1で効く手法と、bsが大きいときに効く手法の違いを説明できる

---

## Day 6：TensorRT ＋ ViT → VLM ★エッジ移植の練習

**この日のゴール**：「ONNX変換 → 量子化 → エンジン化 → 精度と速度の検証」というエッジ移植の流れを一通り経験し、VLMの構成を説明できる

### ① 読む範囲（約1.5時間）

| 教材 | 範囲 | 読み方 | 目的 |
|---|---|---|---|
| Zenn / Qiita | 「TensorRT 入門 ONNX」「TensorRT INT8 量子化 キャリブレーション」 | 1〜2本 | TensorRTの流れを日本語でつかむ |
| 🇬🇧 TensorRT Developer Guide | 量子化（explicit quantization / Q/DQ）の節、Support MatrixのT4の行 | 必要な部分だけ | 使うバージョンでの推奨手順を確認 |
| GPU本 | 第8章 8.1 ディープラーニング用のハードウェア | 流し読み | INT8で計算する専用回路の背景 |
| ViT本 | 第2章 2-2 ViTの全体像 〜 2-5 Encoder | 流し読み（30分） | ViTの構造の復習（経験済みなので短く） |
| ViT本 | 第1章 1-2 Vision and languageへの拡張 | 流し読み | 画像と言語を組み合わせる流れ |
| Zenn / Qiita | 「SmolVLM 解説」「CLIP SigLIP 違い」 | 1本 | VLMの構成（Vision Encoder → Projector → LLM） |

### ② 本の内容と実装の対応

| 学ぶこと | 実装で確かめること |
|---|---|
| TensorRTの最適化（fusion、カーネル選択、精度変更） | PyTorchの最速構成（Day 5）とTensorRT FP16を比べる |
| INT8量子化、キャリブレーション、Q/DQ | TensorRT INT8でPTQし、Imagenetteで精度を測る（PTQ練習と同じ考え方） |
| INT8専用の回路（GPU本 8.1、T4のINT8 Tensor Core） | INT8で速くなるかを確かめる |
| ViTの構造（ViT本 第2章） | ONNXに変換するときの入出力のshapeとして確認する |
| VLMの構成 | 後半：SmolVLAのVision Encoderを特定し、同じ手順を当てはめる |

### ③ 実装に必要な理解のレベル

- TensorRTは「ONNXを読み込んで、そのGPU専用に最適化したエンジンを作る」ものだと言える
- INT8化には、値の範囲からスケールを決める処理（キャリブレーション）が必要だと言える
- 精度が落ちやすい層（最初・最後の層、LayerNorm、softmax）をFP16のまま残す選択肢があると知っている

### ④ 実装ステップ（約2.5時間）

**前半：ViTをTensorRT化する**
1. **精度評価の準備**（20分）：`src/eval_imagenette.py`。Imagenetteのval全データで、ImageNetの1000クラスのうちImagenetteの10クラスのロジットだけを使ってtop-1精度を測る（PTQ練習のImagenetteの準備手順を流用）
2. **ONNXに変換する**（20分）：`torch.onnx.export`（入出力のshapeは固定）。ONNX Runtime（CPU）で推論し、PyTorchの出力と一致するか確認する
3. **TensorRT FP16**（30分）：`pip install tensorrt`、Python APIでエンジンを作って推論・計測する
4. **TensorRT INT8（PTQ）**（40分）
   - Imagenetteのtrainから、キャリブレーション用の画像を100〜500枚取り出す
   - 使うバージョンで推奨される方法（ModelOptでQ/DQ入りのONNXを作る、など）でINT8エンジンを作る
   - 一部の層をFP16のまま残した場合も試す
5. **比較表を作る**（15分）

   | | PyTorch FP32 | PyTorch FP16 | PyTorch 最速（Day 5） | TensorRT FP16 | TensorRT INT8 |
   |---|---|---|---|---|---|
   | レイテンシ bs=1 | | | | | |
   | スループット bs=64 | | | | | |
   | ピークメモリ | | | | | |
   | Imagenette top-1精度 | | | | | |

6. **つまずいた点を記録する**（5分）：変換できなかった演算子、精度が落ちた層、その対処

**後半：SmolVLAのVision Encoderに当てはめる**（時間が足りなければ予備日へ）
7. SmolVLAを読み込み（Day 7の手順1）、`modeling_smolvla.py` でVision Encoderの部分と、画像1枚が何トークンになるかを確認する
8. Vision Encoderだけを切り出し、TensorRT FP16にしてViT-B/16と比べる

**この日の成果物**：`notebooks/06_vit_tensorrt.ipynb`、`src/eval_imagenette.py`、`src/export_trt.py`、READMEの比較表とつまずいた点

### ⑤ 理解チェックポイント
- TensorRTがPyTorchより速くなる理由を、fusion・カーネル選択・精度の3つで説明できる
- INT8で速くなる条件と、精度が落ちやすい層を、自分の結果で言える
- キャリブレーションが何を決めているか説明できる
- VLMの3つの構成要素（Vision Encoder / Projector / LLM）の役割を説明できる

---

## Day 7：VLA（Vision / LLM / Planner）

**この日のゴール**：VLAの3つのブロックと、行動の出し方の2つの流派を説明でき、SmolVLAのどこが重いかを自分の計測で示せる

### ① 読む範囲（約1.5時間）

| 教材 | 範囲 | 読み方 | 目的 |
|---|---|---|---|
| ViT本 | 第5章 5-1 Vision and Languageのサブタスク ／ 5-4 Embodied AIへの応用 | 精読 | ロボットなど「体を持つAI」への応用の流れ |
| Zenn / Qiita / 技術ブログ | 「VLA Vision-Language-Action 解説」「RT-2 解説」「OpenVLA 解説」「π0 flow matching」「SmolVLA 解説」 | 2〜3本 | VLAの全体像を日本語でつかむ |
| 🇬🇧 VLA論文 | RT-2、OpenVLA、π0、SmolVLA：アーキテクチャの図と、その説明の段落だけ | 流し読み | 構成を図で比べる |
| （任意）ReCogDrive論文 | アーキテクチャの図 | 流し読み | 自動運転でも同じ構図であることの確認 |

### ② 本の内容と実装の対応

| 学ぶこと | 実装で確かめること |
|---|---|
| VLAの3ブロック（Vision / LLM / Planner） | SmolVLAの部品ごとのパラメータ数と時間の内訳 |
| 離散トークン型と連続値生成型 | SmolVLAは連続値生成型（flow matching）であることをコードで確認する |
| flow matchingのステップ | ステップ数を変えて、レイテンシと出力のずれを測る |
| action chunking | `select_action` がchunkをキューにためる仕組みをコードで確認し、制御周波数を計算する |
| KV cache | flow matchingのステップでVLMの計算を使い回しているかを、コードとプロファイルで確認する |

### ③ 実装に必要な理解のレベル

- VLAは「画像と指示を理解して（VLM）、行動を出す（Planner）」構成だと言える
- SmolVLAは模倣学習のモデルで、`policy` / `select_action` はLeRobotの共通の名前だと知っている
- `select_action` は2回目以降キューから取り出すだけなので、計測の前に `policy.reset()` が必要だと知っている

### ④ 実装ステップ（約2.5時間）

1. **SmolVLAを動かす**（30分）
   ```python
   import torch
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   from lerobot.policies.factory import make_pre_post_processors
   from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

   model_id = "lerobot/smolvla_base"
   policy = SmolVLAPolicy.from_pretrained(model_id).to("cuda").eval()
   preprocess, postprocess = make_pre_post_processors(
       policy.config, model_id, preprocessor_overrides={"device_processor": {"device": "cuda"}})

   dataset = LeRobotDataset("lerobot/libero", episodes=[0])   # 必要なエピソードだけ取得
   batch = preprocess(dict(dataset[0]))
   with torch.inference_mode():
       policy.reset()                          # キューを空にして必ず推論させる
       action = postprocess(policy.select_action(batch))   # frame ではなく batch を渡す
   print(action.shape)
   ```
   - バージョンによってimportのパスや引数名が違う。エラーが出たら公式のREADMEを確認する
   - 入力名のエラーは `policy.config.input_features` と `dataset.features.keys()` を見比べる
   - flow matchingの乱数を固定し、FP32の出力を「基準の出力」として保存する
2. **部品ごとに分解する**（30分）：Vision Encoder / LLM部分 / action expert のパラメータ数と、`torch.profiler` での時間の内訳。Vision EncoderをViT-B/16と比べる
3. **アムダールの法則で見積もる**（10分）：Day 6のTensorRT化の結果から、Vision部分だけ速くしたらVLA全体は最大何倍速くなるか
4. **flow matchingのステップ数を変える**（25分）：1, 2, 5, 10, 20 でレイテンシと基準出力からのずれを測り、グラフにする
5. **制御周波数を計算する**（10分）：レイテンシとchunk sizeから、何Hzで制御できるか
6. **レポートを書く**（45分〜、予備日も使う）
   1. 概要（主な結果）
   2. 環境と計測方法
   3. ViTのベースライン（CPU vs GPU、FP32 vs FP16、bs）
   4. ボトルネック分析（演算子の内訳、Roofline図）
   5. 高速化の結果（PyTorch）
   6. エッジ移植（TensorRT FP16 / INT8、精度、つまずいた点）
   7. VLAへの展開（SmolVLAの構成図、部品ごとの時間、ステップ数、制御周波数）
   8. 考察（効いた手法・効かなかった手法をRooflineの言葉で）
   9. 制約と今後

**この日の成果物**：`notebooks/07_smolvla_analysis.ipynb`、グラフ、レポートとしてのREADME

### ⑤ 理解チェックポイント
- VLAのVision / LLM / Plannerの役割を、SmolVLAを例に説明できる
- 離散トークン型と連続値生成型の違いを説明でき、SmolVLAがどちらか言える
- SmolVLAのどの部品が一番時間を使っているか、自分の計測結果で説明できる
- action chunkingが推論速度の問題をどう解決しているか説明できる
- 自分のレポートの結論を、1分で口頭説明できる

---

## 最終チェック（ゴール達成の確認）

- [ ] **ゴール1**：メモリ階層・キャッシュ・CPU/GPUの構造を、図を描きながら10分で説明できる
- [ ] **ゴール2**：「この処理を速くしたい」と言われたら、① ボトルネックを判定（Roofline／Profiler）→ ② 対応する手法を選ぶ、という手順で考えられる
- [ ] **ゴール3**：VLAのアーキテクチャ図を何も見ずに描き、Plannerの2つの流派を比較できる
- [ ] **ゴール4**：レポートをGitHubで公開し、「なぜ速くなったか／ならなかったか」を面談で説明できる

## その先に進むなら

- **SmolVLA全体のTensorRT化**：VLM部分とaction expertもエンジン化する。flow matchingのループとKV cacheの扱いが難所
- **AWSのg5 / g6でTensorRT-Edge-LLM**：g5（A10G）やg6（L4）は開発用GPUの条件を満たす。Isaac Lab用の環境と共用できる
- **ReCogDriveを動かす**：g6（L4, 24GB）で、INT4 / FP8に量子化したVLM部分を動かす（難易度高）
- **Arm向けの展開**：Arm公式の "SmolVLA ExecuTorch conversion" の学習パス
- **シミュレータで性能評価**：LIBEROで、高速化前後のタスク成功率を比べる
- **本の残りの章**：CPU本 第9〜12章（マルチプロセッサ、キャッシュコヒーレンス）、GPU本 第5章（CUDA・OpenCL）、『CUDA Cプロフェッショナルプログラミング』
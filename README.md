# 7日間 学習計画：コンピュータアーキテクチャ × 高速化 × VLA
## 〜 SmolVLAの推論高速化レポートを作りながら学ぶ 〜

## ゴール

1. コンピュータアーキテクチャの概要を理解している
2. 高速化に必要な要素を理解している（CPU / GPU）
3. VLAの基本アーキテクチャ（Vision / LLM / Planner）を理解している
4. **【成果物】** 小型VLA（SmolVLA）の推論を計測・分析・高速化し、「なぜ速くなったか」を説明したレポートをGitHubで公開している
   - PyTorchレベルの高速化に加え、**Vision EncoderをONNX → TensorRT（FP16 / INT8）に変換するエッジ向けの移植作業**まで含める

## 進め方

毎日 **「理論（午前の部）」→「成果物の作業（午後の部）」** の2本立てで進める。その日に学んだ理論を、その日のうちにSmolVLAで確かめる。

- 1日あたり3〜4時間（理論1.5時間＋成果物2時間程度）
- 成果物の作業は **Google Colab（T4 GPU）** で行う。手元のM1 Mac（メモリ8GB）は、教材を読むこととCPU側の軽い実験に使う
- 毎日の最後に「理解チェックポイント」を **何も見ずに口頭で説明できるか** 確認する
- 7日で終わらなかった作業のために **予備日を2日** 見ておく（主にレポートの仕上げ用）
- 教材はできるだけ日本語を使う。日本語の教材がないもの（Roofline、VLA）だけ英語を使い、🇬🇧 マークを付けている

## 題材：SmolVLA を選ぶ理由

- Hugging Faceが公開している **約4.5億パラメータの小型VLA**（`lerobot/smolvla_base`）。T4（16GB）で余裕を持って動く
- **Vision（SigLIP系のViT）＋ LLM（SmolVLM2の前半の層）＋ Planner（flow matchingのaction expert、約1億パラメータ）** という構成で、ゴール3の3つのブロックがそのまま揃っている
- 性質の違う部品が1つのモデルに入っているため、**部品ごとに「memory-bound か compute-bound か」が変わる**。高速化の効果の違いが見えやすい
- ロボットの実機やシミュレータがなくても、LeRobotの公開データセットのサンプルを入力にすれば推論できる

## 使うGPUとツールの整理

| ツール | T4での扱い | この計画での使い方 |
|---|---|---|
| PyTorch（FP16、SDPA、`torch.compile`） | ◎ | Day 3〜5 の高速化 |
| **TensorRT（通常版）** | ◎ 正式対応（FP16 / INT8のTensor Coreが使える） | **Day 6：Vision EncoderをONNX経由でエンジン化** |
| bitsandbytes（INT8 / 4bit） | ○ | Day 5（任意）：「メモリは減るが速くならない」例として |
| TensorRT-LLM | △ T4はコミュニティサポートのみ | 使わない |
| TensorRT-Edge-LLM | × T4は対象外（Ampere以降・Jetson/DRIVE向け） | 「その先」でAWSのg5 / g6を使って試す |
| ReCogDrive | × FP16では重みだけで約16GB、データも大きい | 「その先」でAWSのg6（L4, 24GB）を使って試す |

- T4は **BF16、FP8、FlashAttention-2に対応していない**。FP16とINT8を使う

## 全体像

| Day | 理論 | 成果物の作業 |
|---|---|---|
| 1 | メモリ階層とキャッシュ | 環境構築、SmolVLAを動かす、モデルサイズを把握 |
| 2 | CPUの高速化の仕組み | 計測の仕組みを作る、CPUでのベースライン |
| 3 | GPUアーキテクチャ | GPUでのベースライン、FP32 vs FP16 |
| 4 | ボトルネック分析（Roofline） | 部品ごとのプロファイル、Roofline図 |
| 5 | 深層学習の高速化手法 | PyTorchで高速化手法を1つずつ適用して計測 |
| 6 | ViT → VLM ＋ TensorRTの基礎 | **Vision EncoderをTensorRT化（FP16 / INT8 PTQ）** |
| 7 | VLA（Vision / LLM / Planner） | Planner側の解析（flow matchingのステップ数）、レポート作成 |

## 教材一覧

| 教材 | 形式 | 使う日 | ひとこと |
|---|---|---|---|
| 『キタミ式イラストIT塾 基本情報技術者』（技術評論社） | 紙／電子版 | Day 1〜2 | ほぼ全ページがイラスト。コンピュータ構成の章だけ使う |
| 『プログラマーのためのCPU入門』（ラムダノート） | **PDF版（購入後すぐDL）** | Day 1〜2 | 高速化の視点。キャッシュ・パイプライン・データ依存の章 |
| 『[増補改訂]GPUを支える技術』（技術評論社） | **電子版あり** | Day 3〜4 | GPUの構造をコードなしで理解できる |
| 『Vision Transformer入門』（技術評論社） | 紙／電子版 | Day 6 | ViT〜マルチモーダルまで日本語で整理されている |
| Zenn / Qiita の解説記事 | Web | 全日 | 各日に検索キーワードを記載 |
| 🇬🇧 Horace He "Making Deep Learning Go Brrrr From First Principles" | Web | Day 4 | 日本語で代わりになる教材がないため必読。図が多い |
| Zenn / Qiita の TensorRT 解説記事 | Web | Day 6 | 「TensorRT 入門 ONNX」「TensorRT INT8 量子化」で検索 |
| 🇬🇧 NVIDIA TensorRT Developer Guide / Support Matrix | Web | Day 6 | 量子化の章と、T4で使える精度の表 |
| 🇬🇧 Hugging Face Blog "SmolVLA: Efficient Vision-Language-Action Model…" | Web | Day 1, 6, 7 | 成果物の題材の公式解説 |
| 🇬🇧 SmolVLA論文（arXiv:2506.01844） | arXiv | Day 6〜7 | アーキテクチャの図と説明の段落だけ |
| 🇬🇧 RT-2 / OpenVLA / π0 の論文 | arXiv | Day 7 | SmolVLAと比較するため。図だけで可 |

※ 手元に新装版のフルカラーのコンピュータアーキテクチャ入門書があれば、Day 1は第7・8章、Day 2は第10章を図解教材として使う（キタミ式の代わりにしてよい）

## 成果物のリポジトリ構成（最終形）

```
smolvla-inference-optimization/
├── README.md              # レポート本体（結果表・グラフ・考察）
├── notebooks/
│   ├── 01_setup_and_run.ipynb       # Day 1
│   ├── 02_benchmark_baseline.ipynb  # Day 2〜3
│   ├── 03_profile_roofline.ipynb    # Day 4
│   ├── 04_optimizations.ipynb       # Day 5
│   ├── 05_vision_tensorrt.ipynb     # Day 6
│   └── 06_planner_analysis.ipynb    # Day 7
├── src/
│   ├── bench.py           # 計測用の共通関数
│   └── export_vision.py   # Vision EncoderのONNX出力・TensorRTエンジン作成
└── figures/               # Roofline図、レイテンシのグラフなど
```

---

## Day 1：メモリ階層とキャッシュ ／ SmolVLAを動かす

### 理論
**教材**
1. 『キタミ式』：コンピュータ構成要素（CPU・メモリ・キャッシュ）の章で、図で全体像をつかむ
2. 『プログラマーのためのCPU入門』（PDF版）：キャッシュ・メモリ関連の章
- （補助）Zenn / Qiita で「キャッシュ 局所性」「レイテンシ 一覧 プログラマ」を検索

**学習内容** 
- <input type="checkbox" checked> メモリ階層：レジスタ → L1 → L2 → L3 → DRAM → SSD
- キャッシュライン（64バイト）、ヒット／ミス
- 時間的局所性（同じデータを再利用）と空間的局所性（近くのデータを連続で使う）
- レイテンシとバンド幅の違い、メモリウォール

**小演習（M1でOK）**：NumPyで行方向と列方向の合計を比較する（`a.sum(axis=1)` と `a.sum(axis=0)`）

### 成果物の作業
1. GitHubにリポジトリを作る
2. Colabで LeRobot をインストールし、SmolVLAを読み込んで1回推論する
   ```python
   # pip install "lerobot[smolvla]"
   from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
   policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to("cuda").eval()
   ```
   - LeRobotはバージョンによってimportのパスが変わる（例：`lerobot.common.policies...`）。動かなければモデルカードとGitHubのREADMEで現行の書き方を確認する
   - 入力には、LeRobotの公開データセット（例：`lerobot/svla_so100_stacking`）から1サンプル取り出して使う
3. 部品ごとのパラメータ数を数え、FP32 / FP16 / INT8 のときのサイズ（MB）を計算する
   ```python
   for name, m in policy.named_children():
       print(name, sum(p.numel() for p in m.parameters()) / 1e6, "M")
   ```
4. 計算したサイズを、T4のL2キャッシュ（4MB）やVRAM（16GB）と比べ、メモに残す

> この日の時点では、モデルの中身はブラックボックスのままでよい。「動いた」「何MBあるか」まで分かれば十分

### ✅ 理解チェックポイント
- メモリ階層の各段について、容量と速度の大まかな関係を言える
- キャッシュが効く理由を、2種類の局所性を使って説明できる
- 「DRAMの待ち時間が計算より圧倒的に長い」ことがキャッシュの存在理由だと説明できる
- SmolVLAの重みがFP32で何MBあり、それがキャッシュやVRAMと比べてどのくらいの大きさか言える

---

## Day 2：CPUの高速化の仕組み ／ 計測の仕組みを作る

### 理論
**教材**
1. 『キタミ式』：パイプライン、マルチコアの部分を図で確認
2. 『プログラマーのためのCPU入門』（PDF版）：パイプライン、スーパースカラ、データ依存、分岐予測、SIMDの章
- （補助）Zenn / Qiita で「アムダールの法則」「SIMD NumPy なぜ速い」を検索

**学習内容**
- パイプライン、スーパースカラ、アウトオブオーダー実行
- データ依存と分岐予測ミスによるストール
- SIMD（1命令で複数データを処理）
- マルチコアとアムダールの法則
- CPUは「少数の強いコアで待ち時間を短くする」設計

### 成果物の作業
1. `src/bench.py` に計測用の関数を作る。以降のすべての計測で使い回す
   ```python
   import time, statistics, torch

   def bench(fn, n_warmup=10, n_iter=50, cuda=True):
       for _ in range(n_warmup):
           fn()
       times = []
       for _ in range(n_iter):
           if cuda: torch.cuda.synchronize()
           t = time.perf_counter()
           fn()
           if cuda: torch.cuda.synchronize()
           times.append((time.perf_counter() - t) * 1000)
       times.sort()
       return {"median_ms": statistics.median(times),
               "p90_ms": times[int(len(times) * 0.9)]}
   ```
2. **flow matchingのノイズを固定する**。SmolVLAは推論ごとにランダムなノイズから行動を生成するので、seedを固定しないと手法ごとの出力を比較できない
3. 「基準の出力」としてFP32の推論結果（行動のchunk）を保存しておく。Day 5以降、高速化で出力がどれだけずれたかをこれと比べる
4. **CPUでのベースライン**：ColabのCPUで、スレッド数（`torch.set_num_threads`）を1, 2, 4と変えてレイテンシを測る。スレッド数を倍にしても速度が倍にならない理由を、アムダールの法則で考える

### ✅ 理解チェックポイント
- パイプラインで「1命令の時間は変わらないのにスループットが上がる」理由を説明できる
- データ依存と分岐予測ミスが高速化を妨げる理由を説明できる
- アムダールの法則を式で書け、スレッド数を増やした実験結果と結びつけて説明できる
- 計測でウォームアップと中央値／p90を使う理由を説明できる

---

## Day 3：GPUアーキテクチャ ／ GPUでのベースライン

### 理論
**教材**
1. 『[増補改訂]GPUを支える技術』（電子版）：CPUとGPUの違い、SIMT、超並列処理、GPUのメモリとデータ転送の章
- （補助）Zenn / Qiita で「GPU アーキテクチャ SM warp」「Tensor Core 仕組み」を検索
- （補助）🇬🇧 Modal "GPU Glossary"：用語を1つずつ図つきで説明しているサイト

**学習内容**
- GPU → SM → warp（32スレッド）→ スレッド、SIMT
- GPUのメモリ階層：レジスタ → shared memory / L1 → L2 → HBM（VRAM）
- CUDAコアとTensor Core、対応する精度
- coalesced access、occupancyとレイテンシ隠蔽
- CPU⇔GPU間の転送（PCIe）
- GPUは非同期に実行されるので、計測には `torch.cuda.synchronize()` が必要

### 成果物の作業
1. T4の仕様を調べてREADMEに表で載せる：FP32とFP16（Tensor Core）のピーク性能、メモリバンド幅、L2キャッシュ容量、VRAM容量
   - **T4はBF16とFlashAttention-2に対応していない**（FP16とINT8は使える）。なぜ対応していないか（GPUの世代）も調べて書く
2. GPUでFP32のベースラインを測る（bs=1のレイテンシ、ピークメモリ `torch.cuda.max_memory_allocated()`）
3. **CPU vs GPU** でレイテンシを比べる。入力をCPUからGPUへ転送する時間も別に測る
4. **FP32 vs FP16**（`policy.half()` または `torch.autocast`）で比べる。速度、メモリ、基準出力からのずれ（MSE、最大誤差）を記録する

### ✅ 理解チェックポイント
- SM / warp / スレッドの関係を図に描ける
- GPUは「大量のスレッドで待ち時間を隠す」設計だと、CPUとの対比で説明できる
- Tensor Coreが何のためのユニットか、T4ではどの精度で使えるか言える
- FP16で速くなった理由を「Tensor Core」と「データ量が半分になる」の2つの観点で説明できる
- GPUの計測に `synchronize` が必要な理由を説明できる

---

## Day 4：ボトルネック分析（Roofline）／ 部品ごとのプロファイル ★最重要日

### 理論
**教材**
1. Zenn / Qiita で「Roofline モデル」「演算強度 memory bound」を検索し、図つきの解説記事を1〜2本読む
2. 🇬🇧 **Horace He "Making Deep Learning Go Brrrr From First Principles"** ← 必読
3. 『GPUを支える技術』：メモリバンド幅とデータ転送の章を、演算強度の観点で読み直す

**学習内容**
- ボトルネックの3分類：compute-bound / memory-bound / overhead-bound
- 演算強度 ＝ FLOPs ÷ 転送バイト数
- Roofline図：屋根の折れ点 ＝ ピークFLOPS ÷ メモリバンド幅
- 行列積は演算強度が高く、要素ごとの演算（加算、GELU、LayerNorm）は低い
- バッチサイズを上げると、重みの再利用が増えて演算強度が上がる

### 成果物の作業
1. **Roofline図の土台を作る**：T4で、サイズを変えた行列積（N=128〜8192）と要素ごとの演算を測り、達成FLOPSと演算強度をプロットする。T4のピーク性能とバンド幅から屋根の線を引く
2. **部品ごとに時間を分解する**：`torch.profiler` でSmolVLAの1回の推論を記録し、Vision Encoder / VLM（LLM部分）/ action expert のどこに時間を使っているかを集計する
   - `torch.profiler.record_function("vision")` などで区間に名前を付けると集計しやすい
3. 時間を多く使っている演算子の上位10個を表にする
4. 各部品について、bs=1での演算強度を概算し、手順1のRoofline図にプロットする。memory-boundか compute-boundか判定する
5. **振り返り**：以前GPUのFLOPSから学習時間を見積もったとき、実測とずれた原因の仮説を、この図を使って書き直す

### ✅ 理解チェックポイント
- 3つのボトルネックを、それぞれ例を挙げて説明できる
- 演算強度を定義でき、行列積と要素演算で値が大きく違う理由を説明できる
- Roofline図を何も見ずに描き、折れ点の意味を説明できる
- SmolVLAのどの部品がmemory-boundで、どれがcompute-boundかを、自分の計測結果を使って言える
- FLOPSだけで処理時間を見積もると外れる理由を説明できる

---

## Day 5：深層学習の高速化手法 ／ 手法を1つずつ適用する

### 理論
**教材**
1. Zenn / Qiita で以下を検索し、各テーマ1本ずつ解説記事を読む
   - 「FlashAttention 解説」「kernel fusion」「torch.compile 仕組み」
   - 「混合精度 AMP」「量子化 推論 高速化 INT8」「LLM.int8 遅い」
   - 「KV cache 仕組み」「LLM 推論 memory bound decode」
2. 🇬🇧 FlashAttention論文（arXiv:2205.14135）：IntroductionとFigure 1だけ

**学習内容：各手法が解消するボトルネック**

| 手法 | 何を減らすか | 主に効くボトルネック |
|---|---|---|
| kernel fusion / `torch.compile` | HBMへの書き戻し・カーネル起動回数 | memory / overhead |
| FlashAttention / SDPA | N×Nのattention行列をHBMに書かない | memory |
| FP16 | 転送バイト数を減らし、Tensor Coreを使えるようにする | memory / compute |
| 量子化（INT8/4bit） | 重みの転送バイト数 | memory（ただし逆量子化の処理が増える） |
| batching | 重み1回の読み込みあたりの計算を増やす | memory |
| KV cache | 同じ部分の再計算 | compute（代わりにメモリを消費） |
| CUDA Graphs | カーネル起動のオーバーヘッド | overhead |

### 成果物の作業
Day 3のFP16版を基準にして、**手法を1つずつ足しては計測**する。毎回「レイテンシ（中央値/p90）・ピークメモリ・基準出力からのずれ」の3つを記録する。

1. **attentionの実装**：SDPA（`F.scaled_dot_product_attention`）が使われているか確認し、素朴な実装と比べる
2. **`torch.compile`**：初回のコンパイル時間と、2回目以降の速度を分けて記録する
3. **batching**：bs=1, 4, 16, 32 でレイテンシとスループット（サンプル/秒）を測る。Day 4のRoofline図で、bsを上げると点がどう動くか確認する
4. **（任意）bitsandbytesでの量子化**：INT8 / 4bitで、LLM部分の重みを量子化する
   - **速くならない、むしろ遅くなる可能性が高い**。メモリは減るのに遅くなる理由（逆量子化の処理、外れ値の分離処理、bs=1では計算が少ない）をプロファイル結果から説明する
   - 翌日のTensorRTのINT8（Tensor CoreでINT8のまま計算する）と比べると、「量子化すれば速くなるわけではなく、ハードでINT8のまま計算できるかが鍵」という対比になる
5. 結果を1つの表とグラフ（手法ごとのレイテンシの積み上げ）にまとめる
6. Day 4のプロファイルから、**Vision Encoderが全体の何%の時間を使っているか**を確認しておく（翌日TensorRT化したとき、全体がどれだけ速くなるかをアムダールの法則で予想するため）

### ✅ 理解チェックポイント
- 表の各手法について「何を減らしているか」を自分の言葉で言える
- FlashAttentionが「計算量は同じなのに速い」理由を、メモリ階層を使って説明できる
- batchingでスループットが上がる理由を、演算強度で説明できる
- 量子化が速度向上につながる条件と、つながらない条件を、自分の計測結果を使って言える
- `torch.compile` で速くなった（ならなかった）理由を、overheadやfusionの観点で説明できる

---

## Day 6：ViT → VLM ＋ TensorRT ／ Vision EncoderをTensorRT化 ★エッジ移植の練習

**目標**：VLMのVision側の構造を理解し、その部分を「ONNX変換 → 量子化 → エンジン化 → 精度と速度の検証」というエッジ移植の流れで高速化できるようになる

### 理論（ViT・VLMは経験済みなので短めに、TensorRTに時間を使う）
**教材**
1. 『Vision Transformer入門』：ViTの基本構造の章を復習（30分）、マルチモーダル（CLIPなど）の章を拾い読み
2. Zenn / Qiita で「SmolVLM 解説」「CLIP SigLIP 違い」を検索（VLMの構成を確認）
3. Zenn / Qiita で「TensorRT 入門 ONNX」「TensorRT INT8 量子化 キャリブレーション」を検索し、解説記事を1〜2本読む
4. 🇬🇧 NVIDIA TensorRT Developer Guide：量子化（explicit quantization / Q/DQ）の節と、Support MatrixのT4の行

**学習内容**
- **ViT・VLMの復習**
  - パッチ分割 → 線形射影 → 位置埋め込み → Transformer Encoder（224×224、パッチ16 → 196トークン）
  - VLMの基本構成：Vision Encoder（SigLIPなど）→ Projector → LLM
  - SmolVLAの工夫：画像トークンの圧縮、LLMの前半の層だけを使う
- **TensorRTの基本**
  - 何をしているか：層の融合（fusion）、各層で最速のカーネルを選ぶ（tactic選択）、精度の変更（FP16 / INT8）
  - 流れ：PyTorch → ONNX → TensorRTエンジン（GPUの種類ごとに作り直しが必要）
  - INT8の2つの方式：キャリブレーションでスケールを決める方式（implicit）と、ONNXにQ/DQノードを入れる方式（explicit）。TensorRT 10ではexplicitが推奨で、NVIDIA ModelOptなどでQ/DQ入りのONNXを作る。**進めているPTQ練習（ONNX RuntimeのQDQ形式）と同じ考え方**
  - **T4のINT8 Tensor Core**で、INT8のまま行列積を計算できるから速くなる（bitsandbytesとの違い）

### 成果物の作業
1. **構造を確認する**：`modeling_smolvla.py` を読み、Vision Encoder部分のモジュール名、画像1枚が何トークンになるか、LLMの何層目までを使っているかを確認する。ViT → Projector → LLM の図にshapeを書き込む（READMEに載せる）
2. **ONNXに変換する**：Vision Encoder（＋可能ならProjector）だけを切り出し、`torch.onnx.export` でONNXにする
   - 入出力のshapeを固定する（エッジ向けでは固定shapeが基本）
   - ONNX Runtime（CPU）で推論し、PyTorchの出力と一致するか確認する（コサイン類似度、最大誤差）
3. **TensorRTのFP16エンジンを作る**：Colabで `pip install tensorrt` し、Python APIでエンジンを作って推論する
4. **TensorRTのINT8エンジンを作る（PTQ）**
   - キャリブレーション用に、LeRobotのデータセットから画像を100〜500枚取り出す
   - ModelOptなどでQ/DQ入りのONNXを作るか、TensorRTのキャリブレーション機能を使ってINT8エンジンを作る（使うバージョンで推奨される方法に従う）
   - 一部の層（最初のパッチ埋め込みなど）をFP16のまま残すと、精度と速度がどう変わるかも試す
5. **比較表を作る**（Vision Encoder単体）

   | | PyTorch FP32 | PyTorch FP16 | TensorRT FP16 | TensorRT INT8 |
   |---|---|---|---|---|
   | レイテンシ（中央値 / p90） | | | | |
   | ピークメモリ | | | | |
   | FP32出力とのコサイン類似度 | | | | |

6. **全体への影響を見積もる**：Day 5で測った「Vision Encoderが全体に占める割合」から、アムダールの法則で「VLA全体は最大何倍速くなるか」を計算する。可能なら、TensorRTの出力をSmolVLAの残りの部分に戻して推論し、最終的な行動のずれも測る
7. つまずいた点（ONNX変換できなかった演算子、精度が落ちた層など）をメモしておく。**エッジ移植では、このつまずきとその対処こそが一番価値のある記録になる**

> **時間が足りない場合**：手順4（INT8）は予備日に回してよい。FP16のTensorRT化までで、移植の流れは一通り経験できる
> **（任意・予備日）**：カメラの台数を1〜3枚に変え、系列長とVision・LLM部分のレイテンシの関係をグラフにする

### ✅ 理解チェックポイント
- ViTの処理の流れと、VLMの3つの構成要素の役割をshapeつきで説明できる
- SmolVLAで画像1枚が何トークンになり、それがレイテンシにどう効くか説明できる
- TensorRTがPyTorchより速くなる理由を、fusion・カーネル選択・精度の3つの観点で説明できる
- INT8で速くなる条件（INT8のTensor Coreで計算できるか）を、bitsandbytesの結果と対比して説明できる
- PTQのキャリブレーションが何を決めているか、Q/DQノードが何をしているか説明できる
- Vision Encoderだけ速くしたとき、全体がどれだけ速くなるかをアムダールの法則で説明できる

---

## Day 7：VLA（Vision / LLM / Planner）／ Planner側の解析とレポート作成

### 理論
**教材**
1. Zenn / Qiita / 企業の技術ブログで以下を検索し、日本語の解説を読む
   - 「VLA Vision-Language-Action 解説」「RT-2 解説」「OpenVLA 解説」「π0 解説 flow matching」
   - 「SmolVLA 解説」「action chunking」
2. 🇬🇧 論文はアーキテクチャの図とその説明の段落だけ読む
   - RT-2（2023）、OpenVLA（arXiv:2406.09246）、π0（arXiv:2410.24164）
- （任意）自動運転VLA：ReCogDrive（VLM＋拡散モデルのplanner）

**学習内容**
```
カメラ画像 ─▶ [Vision Encoder (ViT)] ─┐
                                      ├─▶ [LLM / VLM backbone] ─▶ [Planner / Action Head] ─▶ 行動
言語指示・ロボットの状態 ─────────────┘        (状況理解)            (具体的な行動を出力)
```

**Plannerの2つの流派**

| | 離散トークン型 | 連続値生成型 |
|---|---|---|
| 代表例 | RT-2, OpenVLA | π0, **SmolVLA**, GR00T N1 |
| 行動の表し方 | 各次元を256段階に量子化し、トークンとして出力 | flow matching / 拡散モデルで連続値を生成 |
| Planner | LLM自身が行動トークンを出す | LLMとは別の小さな **action expert** |
| 推論の特徴 | 行動の次元数だけ自己回帰で生成するので遅い | LLM部分は1回、action expertを数ステップ回す |

- **action chunking**：1回の推論で複数ステップ分の行動をまとめて出す
- **非同期推論**（SmolVLAの特徴）：行動を実行しながら次の推論を進め、待ち時間を隠す
- flow matchingのステップを回すとき、VLM部分の出力（KV cache）を使い回せるので、ステップごとにVLM全体を計算し直さなくてよい

### 成果物の作業
1. **flow matchingのステップ数を変える実験**：既定値（10ステップ）から 1, 2, 5, 10, 20 と変え、レイテンシと基準出力からのずれを測る。「速さと精度のトレードオフ」のグラフにする
2. コードを読み、ステップを回すときに **VLM部分の計算（KV cache）が使い回されているか** を確認する。使い回されていれば、action expertだけが繰り返し計算されていることを、Day 4のプロファイルと照らし合わせて確認する
3. **制御周波数を計算する**：レイテンシと1回で出す行動の数（chunk size）から、何Hzでロボットを制御できるかを計算する
4. **レポート（README）を書く**（予備日も使う）
   1. 概要：何をしたか、主な結果（一番速い構成で何倍速くなったか）
   2. 環境：T4の仕様、ライブラリのバージョン、計測方法
   3. SmolVLAの構成図（Vision / LLM / Planner、shapeつき）
   4. ベースライン：CPU vs GPU、FP32 vs FP16
   5. ボトルネック分析：部品ごとの時間の内訳、Roofline図
   6. 高速化の結果（PyTorch）：手法ごとの表とグラフ
   7. **エッジ移植：Vision EncoderのTensorRT化**（FP16 / INT8の比較表、つまずいた点と対処）
   8. Plannerの解析：flow matchingのステップ数の実験、制御周波数
   9. 考察：効いた手法と効かなかった手法を、それぞれRooflineの言葉で説明する（bitsandbytesとTensorRTのINT8の違いなど）
   10. 制約と今後：T4の制約、VLA全体のTensorRT化、Jetson向けの展開

### ✅ 理解チェックポイント
- VLAのVision / LLM / Plannerの役割を、SmolVLAを例に説明できる
- 離散トークン型と連続値生成型の違いを説明でき、SmolVLAがどちらか言える
- flow matchingのステップ数がレイテンシと精度にどう効くか、自分の計測結果で説明できる
- action chunkingと非同期推論が、推論速度の問題をどう解決しているか説明できる
- 自分のレポートの結論を、1分で口頭説明できる

---

## 最終チェック（ゴール達成の確認）

- [ ] **ゴール1**：メモリ階層・キャッシュ・CPU/GPUの構造を、図を描きながら10分で説明できる
- [ ] **ゴール2**：「この処理を速くしたい」と言われたら、① ボトルネックを判定（Roofline／Profiler）→ ② 対応する手法を選ぶ、という手順で考えられる
- [ ] **ゴール3**：VLAのアーキテクチャ図を何も見ずに描き、Plannerの2つの流派を比較できる
- [ ] **ゴール4**：レポートをGitHubで公開し、「なぜ速くなったか／ならなかったか」を面談で説明できる

## その先に進むなら

- **SmolVLA全体のTensorRT化**：VLM部分とaction expertもONNX → TensorRTに変換し、VLA全体をエンジン化する。flow matchingのループやKV cacheの扱いが難所
- **AWSのg5 / g6でTensorRT-Edge-LLM**：g5（A10G, Ampere）やg6（L4, Ada）は開発用GPUの条件を満たす。Isaac Lab用の環境と共用できる
- **ReCogDriveを動かす**：g6（L4, 24GB）で、INT4 / FP8に量子化したVLM部分を動かす。TensorRT-Edge-LLMはInternVL系のVLMに対応しているが、diffusion plannerは自分でつなぐ必要がある（難易度高）
- **Arm向けの展開**：Arm公式の "SmolVLA ExecuTorch conversion" の学習パス
- **シミュレータで性能評価**：LIBEROなどのシミュレータで、高速化前後のタスク成功率を比べる
- Triton で fused kernel を1本書いてみる
- 本格的な教科書：パタヘネ『コンピュータの構成と設計』、『CUDA Cプロフェッショナルプログラミング』
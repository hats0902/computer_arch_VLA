# ONNX・TFLite・PTQ 入門課題

## 目的

本課題では、学習済みの画像分類モデルを題材に、モデル形式の変換、推論、量子化までの一連の流れを体験します。

最終的に、以下の処理を実行してください。

```
PyTorch
  ↓
ONNX FP32
  ↓
TFLite FP32
  ↓
TFLite INT8（PTQ）
```

学習そのものは本課題の対象外とします。

使用するモデルは `torchvision.models` の ImageNet 学習済み `MobileNetV2` とします。

---

## Day 1：ONNX・TFLiteへの変換

### 問1：MobileNetV2を取得し、ONNX形式へ変換する

`torchvision.models` から、ImageNetで学習済みの `MobileNetV2` を取得してください。

取得したモデルを推論可能な状態にし、任意の画像1枚に対してPyTorch上で推論してください。

その後、モデルをONNX形式へ変換してください。

```
torchvision.models
      ↓
MobileNetV2
      ↓
PyTorchで推論
      ↓
ONNXへ変換
      ↓
model.onnx
```

確認事項：

1. モデルへの入力Tensorのshapeは何ですか。 →torch.Size(\[32, 3, 224, 224\]): pytorch  
2. モデルへの入力Tensorのdtypeは何ですか。 →torch.Size(\[32, 3, 224, 224\]): tensor  
3. モデルの出力Tensorのshapeは何ですか。→  
4. 出力値は何を表していますか。→losgits各クラスごとに確率を表したもの  
5. 推論時に `model.eval()` を実行する理由は何ですか。→評価用にdropout層を無効化する、学習中に蓄積した移動平均・分散を使用する  
6. ONNXへの変換時に、なぜサンプル入力Tensorが必要なのでしょうか。→このサンプルを使ってONNX保存時にモデルの計算の流れを把握する  
7. 生成されたONNXファイルのサイズを確認してください。→torch: 14,258,573 bytesからONNX: 14,210,500 bytes  
8. ONNXモデルの入力名・出力名を確認してください。→input: modelInput, output: linear（おそらくfoward層の出力そのまんま）  
9. ONNXモデルの入出力shapeがPyTorchモデルと一致していることを確認してください。→確認済

---

### 問2：ONNX Runtimeで推論する

作成したONNXモデルを、ONNX Runtimeを使用して実行してください。

PyTorchモデルとONNXモデルに同じ入力データを与え、推論結果を比較してください。

確認事項：

1. ONNX Runtimeへ渡す入力データのshapeとdtypeを確認してください。→numpy  
2. PyTorchとONNX Runtimeで予測されたクラスは一致しましたか。→した  
3. 両者の出力値は完全に一致しましたか。→しなかった  
4. 一致しない場合、どの程度の差がありましたか。→平均絶対誤差: 1.35e-06  
5. 推論時に使用するモデル形式が変わっても、同じモデルとして利用できるのはなぜでしょうか。→ONNXは演算のグラフと重みでモデルの構造まで保存されるため言語が関係ない。それを複数の言語で呼び出すことができる=複数の言語でモデルを書いても対応できる

---

### 問3：ONNXモデルをTFLite形式へ変換する

作成したONNXモデルをTFLite形式へ変換してください。

```
model.onnx
    ↓
ONNX → TFLite変換
    ↓
model.tflite
```

変換後、TFLiteのInterpreterを使用して、同じ画像に対する推論を実行してください。

確認事項：

1. TFLiteモデルの入力Tensorのshapeとdtypeを確認してください。  
2. TFLiteモデルの出力Tensorのshapeとdtypeを確認してください。  
3. ONNX RuntimeとTFLiteで予測されたクラスは一致しましたか。  
4. 出力値にはどの程度の差がありましたか。  
5. ONNXモデルとTFLiteモデルのファイルサイズを比較してください。  
6. ONNXからTFLiteへ変換する際、入出力のshapeやデータ形式に変化がないか確認してください。

---

## Day 2：Post-Training Quantization

### 問4：量子化前のモデルを測定する

量子化前のTFLiteモデルについて、以下を測定してください。

- モデルサイズ  
- テストデータに対するAccuracy  
- 複数回推論した場合の平均推論時間

以降、この結果を量子化後のモデルと比較します。

---

### 問5：PTQについて調べる

以下の用語について、それぞれ簡潔に説明してください。

1. Quantization: 量子化、モデルの重みとアクチベーション値（＝各層の出力）をFLOATからINTに置き換えることで推論の高速化や消費電力の削減に繋げること  
2. Post-Training Quantization（PTQ）: 学習後に量子化が行われる。学習中に行うQATと区別される。再学習が不要なため低コストでできるが精度が低下する可能性がある。  
3. FP32: 1つの数を32ビット（4バイト）で表す浮動小数点数の形式で、小数点以下7桁程度まで正確に表せる、ニューラルネットの学習と推論で標準的に使われる形式です。  
4. INT8: 1つの数を8ビット（1バイト）の整数で表す形式で、256通りの値しか取れないため、量子化ではスケールと組み合わせて小数を近似的に表し、メモリをFP32の1/4に減らして高速化します。  
5. Calibration: アクチベーション値を丸める際に使うscaleを決める  
6. Representative Dataset: Calibrationのために使われるデータセット  
7. scale: アクチベーション値を整数に丸めるために使う値。各層に1つあることが多く、チャネルごとに異なる値を持つこともある。  
8. zero\_point: 活性化にマイナスの値が含まれて最小値が負になる場合に、小数の0.0が整数のどの値に対応するかを表す値で、ものさしの開始位置をずらす役割を持つ。

また、次の問いに答えてください。

「学習済みモデルの値を単純にFP32からINT8へ変換するだけではなく、Calibrationが必要になるのはなぜでしょうか。」

→モデルの重みをINT8に変換しても、テンソルの出力もINTに変換しないとINT×FLOATでモデルの出力がFLOATになってしまう。そのため各層の出力もINTに変換する必要があり、scaleを決定するために代表的なデータを入力してどんな出力があるか確認する必要がある。

---

### 問6：INT8のPTQを実施する

TFLiteモデルに対して、Representative Datasetを使用したINT8のPost-Training Quantizationを実施してください。

```
TFLite FP32
     ↓
Representative Dataset
     ↓
Calibration
     ↓
TFLite INT8
```

量子化後、以下を確認してください。

&nbsp;

TFLiteは使えなかったので、ONNXRuntimeでそのまま実装

1. 入力Tensorのdtypeは何ですか。→numpy  
2. 出力Tensorのdtypeは何ですか。→numpy  
3. 入力Tensorの `scale` と `zero_point` を確認してください。→scale      \= 0.01865844801068306`,` zero\_point \= 114  (dtype: uint8)  
4. 出力Tensorの `scale` と `zero_point` を確認してください。`→scale      = 0.05548189580440521, zero_point = 51  (dtype: uint8)`  
5. FP32の値とINT8の値を対応付ける際、`scale` と `zero_point` がどのような役割を持つか説明してください。→scaleは元の重みやテンソルを割る係数のことでINTに変換するために用いられる。zero\_pointはscaleを計算する際にはもとの重みやテンソルの値のrangeを基準にするが小数の0.0が整数のどの値に対応するかを表す値で、ものさしの開始位置をずらす役割を持つ。

---

### 問7：FP32モデルとINT8モデルを比較する

以下の表を完成させてください。

| モデル | Accuracy | モデルサイズ | 平均推論時間 |
| :---- | ----: | ----: | ----: |
| ONNX FP32 |  |  |  |
| TFLite FP32 |  |  |  |
| TFLite INT8 |  |  |  |

比較結果をもとに、次の問いに答えてください。

1. INT8化によってモデルサイズはどの程度変化しましたか。  
2. Accuracyはどの程度変化しましたか。  
3. 平均推論時間はどの程度変化しましたか。  
4. 量子化によってAccuracyが低下する可能性があるのはなぜでしょうか。  
5. 今回の結果だけを見て、「INT8モデルは必ずFP32モデルより高速」と言えるでしょうか。その理由も説明してください。

---

## 最終問題

今回実施した処理について、以下の用語をすべて使用して説明してください。

- PyTorch  
- ONNX  
- ONNX Runtime  
- TFLite  
- PTQ  
- Calibration  
- INT8

あわせて、以下の3点について説明してください。

1. なぜPyTorchモデルをONNX形式へ変換するのか。  
2. なぜONNXモデルをTFLite形式へ変換するのか。  
3. なぜTFLiteモデルに対してPTQを行うのか。

---

## 完了条件

以下をすべて満たしたら課題完了とします。

```
☑︎ torchvisionからMobileNetV2を取得できる
☑︎ PyTorch上で推論できる
☑︎ PyTorch → ONNXの変換ができる
☑︎ ONNXモデルの入出力情報を確認できる
☑︎ ONNX Runtimeで推論できる
- ONNX → TFLiteの変換ができる
- TFLiteで推論できる
☑︎ Representative Datasetを用意できる
☑︎ Calibrationを伴うINT8 PTQを実施できる
□ scale / zero_pointを確認できる
□ FP32とINT8の精度・サイズ・推論時間を比較できる
□ 一連の処理を自分の言葉で説明できる
```

&nbsp;
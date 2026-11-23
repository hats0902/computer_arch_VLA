"""SmolVLM-256M の部品（Vision Encoder / Projector / LLM）を、1つずつ単独でプロファイルする

使い方（Colab T4）:
    !python src/profile_smolvlm.py

やること:
  1. モデルをFP16で読み込む（T4はbf16のTensor Coreを持たないため）
  2. 部品ごとのパラメータ数を数える
  3. 1回だけ通常の順伝播（prefill）を実行して、各部品への入力をフックで記録する
  4. 記録した入力を使って、部品を単独で torch.profiler にかけ、部品ごとに別の表を出す
     → 表の softmax や addmm が、どの部品のものか分かる
  5. results/profile_<部品名>.json にタイムライン（Perfettoで開ける）を保存する

注意:
  - 未実行のコードです。モデルの属性名や入力の渡し方は transformers のバージョンで
    変わることがあります。エラーが出たら、メッセージを確認して直してください。
  - 対象は prefill（入力全体を1回で処理する段階）のみ。decode（1トークンずつ生成）は含みません。
"""
import os

import torch
from PIL import Image
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
N_WARMUP = 5
N_ITER = 10
OUT_DIR = "results"

# 部品の名前 → model の中での属性パス（print(model) で実際の名前を確認すること）
COMPONENTS = {
    "vision_encoder": "model.vision_model",
    "projector": "model.connector",
    "llm": "model.text_model",
}


def get_submodule(model, path):
    obj = model
    for name in path.split("."):
        if not hasattr(obj, name):
            raise AttributeError(
                f"'{path}' が見つかりません（'{name}' がありません）。"
                f" print(model) で構造を確認して COMPONENTS を直してください。"
            )
        obj = getattr(obj, name)
    return obj


def count_params(module):
    n = sum(p.numel() for p in module.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in module.parameters()) / 1e6
    return n, size_mb


def to_detached(x):
    """記録した入力を、プロファイル中に再利用できる形にする"""
    if torch.is_tensor(x):
        return x.detach()
    if isinstance(x, (list, tuple)):
        return type(x)(to_detached(v) for v in x)
    if isinstance(x, dict):
        return {k: to_detached(v) for k, v in x.items()}
    return x


def self_cuda_time_ms(prof):
    """カーネルが実際に動いた時間の合計（ms）。PyTorchのバージョン差を吸収する"""
    total_us = 0.0
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", None)
        if t is None:
            t = getattr(e, "self_cuda_time_total", 0)
        total_us += t
    return total_us / 1000


def main():
    assert torch.cuda.is_available(), "GPUが見つかりません"
    os.makedirs(OUT_DIR, exist_ok=True)
    print("GPU:", torch.cuda.get_device_name(0))

    # 1. モデルの読み込み（FP16を明示）
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).cuda().eval()
    print("model dtype:", model.dtype)

    # 2. パラメータ数
    print("\n== パラメータ数 ==")
    modules = {name: get_submodule(model, path) for name, path in COMPONENTS.items()}
    for name, m in modules.items():
        n, mb = count_params(m)
        print(f"{name:15s} {n/1e6:8.1f} M params  {mb:8.1f} MB")

    # ダミー入力（画像1枚＋短い質問）
    image = Image.new("RGB", (512, 512))
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe the image."}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")
    inputs = {k: v.to(torch.float16) if v.is_floating_point() else v
              for k, v in inputs.items()}

    # 3. 各部品への入力を記録する（1回だけ通常の順伝播を実行）
    captured = {}
    handles = []

    def make_capture(name):
        def hook(module, args, kwargs):
            if name not in captured:      # 最初の1回だけ記録
                captured[name] = (to_detached(args), to_detached(kwargs))
        return hook

    for name, m in modules.items():
        handles.append(m.register_forward_pre_hook(make_capture(name), with_kwargs=True))

    with torch.inference_mode():
        model(**inputs)
    for h in handles:                     # フックは外しておく（再実行で重複させない）
        h.remove()

    missing = [n for n in modules if n not in captured]
    if missing:
        raise RuntimeError(f"入力を記録できなかった部品: {missing}")

    # LLMはKVキャッシュを使い回すと実行のたびに伸びるので、キャッシュを使わない設定にする
    if "llm" in captured:
        kw = captured["llm"][1]
        if "past_key_values" in kw:
            kw["past_key_values"] = None
        kw["use_cache"] = False

    # 4. 部品を単独でプロファイル
    summary = []
    for name, m in modules.items():
        args, kwargs = captured[name]

        def run(m=m, args=args, kwargs=kwargs):
            with torch.inference_mode():
                m(*args, **kwargs)

        for _ in range(N_WARMUP):
            run()
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(N_ITER):
                run()
            torch.cuda.synchronize()

        print(f"\n{'=' * 20} {name} {'=' * 20}")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

        gpu_ms = self_cuda_time_ms(prof) / N_ITER
        summary.append((name, gpu_ms))
        prof.export_chrome_trace(os.path.join(OUT_DIR, f"profile_{name}.json"))

    # 5. まとめ（1回あたりのGPU稼働時間）
    print("\n== 部品ごとのGPU稼働時間（Self CUDA time の合計 ÷ 反復回数） ==")
    total = sum(t for _, t in summary)
    for name, t in summary:
        print(f"{name:15s} {t:8.3f} ms  ({t / total * 100:5.1f}%)")
    print(f"{'合計':15s} {total:8.3f} ms")
    print("\n※ 全体の実測（wall）は bench.py で別に測り、合計との差を『GPUの遊び』として見る")


if __name__ == "__main__":
    main()

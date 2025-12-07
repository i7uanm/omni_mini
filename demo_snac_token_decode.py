import os
import torch
import soundfile as sf

# 复用 inference.py 里的组件和常量
from inference import (
    load_model,
    get_input_ids_TA,
    text_vocabsize,
    _eot,
    _pad_t,
    _eoa,
    padded_text_vocabsize,
)
from utils.snac_utils import reconscruct_snac, reconstruct_tensors
from litgpt.generate.base import generate_TA


def decode_snac_to_wav(snac_input,
                       snac_model,
                       out_path: str,
                       sample_rate: int = 24000) -> None:
    """
    snac_input: reconstruct_tensors(audiolist) 的返回值
                类型可能是 tensor，也可能是 list，直接按原项目用法喂给 snac_model.decode 即可
    snac_model: SNAC.from_pretrained(...) 得到的模型
    out_path:   输出 wav 路径
    """
    with torch.no_grad():
        # 关键点：完全模仿 inference.py 里的 A1_A2/T1_A2 写法
        audio_hat = snac_model.decode(snac_input)

    audio = audio_hat.squeeze().cpu().numpy().astype("float32")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, audio, sample_rate)
    print(f"[SNAC 解码] 已保存音频到: {out_path}")


def text_to_audio_via_tokens(
    text: str,
    ckpt_dir: str = "./checkpoint",
    device: str = "cuda:0",
) -> None:
    """
    整个流程：
    1) 文本 -> 输入 token（get_input_ids_TA）
    2) 语言模型 generate_TA 输出一整串 token（包含 audio + text）
    3) 从中提取 audio token，重构为 SNAC 需要的输入结构
    4) 用 snac_model.decode 解码为 wav
    """

    # 1. 加载模型（直接复用 inference.py 里的 load_model）
    fabric, model, text_tokenizer, snac_model, _ = load_model(ckpt_dir, device)
    model.eval()

    # 2. 文本编码成输入 token
    input_ids = get_input_ids_TA(text, text_tokenizer)

    # 3. 设置 KV cache，调用 generate_TA 让模型生成「文本->音频」
    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)

    tokenlist = generate_TA(
        model,
        None,               # 没有 audio_feature，纯文本输入
        input_ids,
        None,
        ["T1A2"],           # 文本 -> 音频 任务
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=padded_text_vocabsize,
        include_prompt=True,
        generate_text=True,
    )

    # tokenlist 结构说明：
    # - 长度为 8 的 list
    # - 前 7 个元素是 7 个 codebook 的 audio token 序列
    # - 最后一个元素是文本 token 序列
    print("=== 原始 tokenlist 信息 ===")
    print("tokenlist 长度（应为 8）:", len(tokenlist))
    for i in range(7):
        print(f"  第 {i} 个 audio codebook token 数量:", len(tokenlist[i]))
    print("  文本 token 数量:", len(tokenlist[-1]))

    # 4. 把 tokenlist 还原成 SNAC 需要的输入
    #    完全模仿 A1_A2 / T1_A2：
    #    audiolist = reconscruct_snac(tokenlist)
    #    audio    = reconstruct_tensors(audiolist)
    audiolist = reconscruct_snac(tokenlist)
    snac_input = reconstruct_tensors(audiolist)

    print("\n=== SNAC 输入结构信息（用于 debug） ===")
    print("snac_input 类型:", type(snac_input))
    try:
        # 如果本身是 tensor
        if hasattr(snac_input, "shape"):
            print("snac_input.shape:", snac_input.shape)
        # 如果是 list 且里面是 tensor
        elif isinstance(snac_input, list) and len(snac_input) > 0 and hasattr(snac_input[0], "shape"):
            print("snac_input[0].shape:", snac_input[0].shape)
    except Exception as e:
        print("打印 shape 时出错，仅作参考，不影响解码:", e)

    # 5. 用 SNAC 解码器把 token -> wav
    out_dir = "./output/demo_snac"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "text_to_audio_from_tokens.wav")
    decode_snac_to_wav(snac_input, snac_model, out_path)

    # 6. 顺便把模型生成的文本也解一下，方便对照理解
    text_tokens = tokenlist[-1]
    if text_vocabsize in text_tokens:
        text_tokens = text_tokens[: text_tokens.index(text_vocabsize)]
    decoded_text = text_tokenizer.decode(torch.tensor(text_tokens)).strip()
    print("\n=== 文本侧信息 ===")
    print("输入文本:", text)
    print("模型生成的文本（T2）:", decoded_text)

    # 清理 KV cache
    model.clear_kv_cache()


if __name__ == "__main__":
    # 这里可以随便改测试文本
    test_text = "What is your name?"
    text_to_audio_via_tokens(test_text)

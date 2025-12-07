import os
from typing import Optional, Union, Iterable

import numpy as np

from inference import OmniInference


class OmniEngine:
    """Small inference engine wrapper that exposes simple audio generation APIs.

    提供两个接口：
    1) generate_stream(...)：流式生成，逐 chunk 输出 16-bit PCM 字节流
    2) generate(...)：一次性返回整段 16-bit PCM 字节流或 numpy 数组
    """

    def __init__(self, ckpt_dir: str = "./checkpoint", device: str = "cuda:0"):
        self.ckpt_dir = ckpt_dir
        self.device = device
        if not os.path.exists(ckpt_dir):
            # Let OmniInference handle download if checkpoint missing
            print(f"checkpoint directory {ckpt_dir} not found, will download when loading model")
        # instantiate underlying inference helper which loads models
        self._infer = OmniInference(ckpt_dir=ckpt_dir, device=device)

    def generate_stream(
        self,
        audio_path: str,
        image_path: Optional[str] = None,
        stream_stride: int = 5,
    ) -> Iterable[bytes]:
        """流式生成回复音频。

        每当底层多生成 stream_stride 个音频 token，
        就会通过 SNAC 解码出一小段 PCM 数据并 yield 一次。

        Args:
            audio_path: 本地输入 .wav 路径（用户问题音频）
            image_path: 预留参数，目前未使用
            stream_stride: 底层每多少个音频 token 解码一次并输出

        Yields:
            bytes: 一小段连续的 16-bit 单声道 PCM 数据
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"audio file {audio_path} not found")

        # 直接转发 OmniInference 的流式输出
        for chunk in self._infer.run_AT_batch_stream(
            audio_path,
            stream_stride=stream_stride,
        ):
            if chunk:
                yield chunk

    def generate(
        self,
        audio_path: str,
        image_path: Optional[str] = None,
        return_numpy: bool = False,
        stream_stride: int = 5,
    ) -> Union[bytes, np.ndarray]:
        """一次性生成完整回复音频。

        内部复用 generate_stream，将所有流式 chunk 拼接成一整段 PCM。

        Args:
            audio_path: 本地输入 .wav 路径（用户问题音频）
            image_path: 预留参数，目前未使用
            return_numpy: 为 True 时返回 np.ndarray(dtype=int16)，否则返回 bytes
            stream_stride: 底层每多少个音频 token 解码一次并输出

        Returns:
            bytes 或 numpy.ndarray(int16)：完整的 16-bit PCM 音频数据
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"audio file {audio_path} not found")

        out = bytearray()
        for chunk in self.generate_stream(
            audio_path=audio_path,
            image_path=image_path,
            stream_stride=stream_stride,
        ):
            out.extend(chunk)

        data = bytes(out)
        if return_numpy:
            arr = np.frombuffer(data, dtype=np.int16)
            return arr
        return data


__all__ = ["OmniEngine"]
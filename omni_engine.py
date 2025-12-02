import os
from typing import Optional, Union

import numpy as np

from inference import OmniInference


class OmniEngine:
    """Small inference engine wrapper that exposes a simple generate(...) API.

    This class wraps the existing `OmniInference` defined in `inference.py` and
    provides a single `generate(audio_path, image_path)` method which returns
    the decoded response audio as raw bytes (int16 PCM) or as a numpy array.
    """

    def __init__(self, ckpt_dir: str = "./checkpoint", device: str = "cuda:0"):
        self.ckpt_dir = ckpt_dir
        self.device = device
        if not os.path.exists(ckpt_dir):
            # Let OmniInference handle download if checkpoint missing
            print(f"checkpoint directory {ckpt_dir} not found, will download when loading model")
        # instantiate underlying inference helper which loads models
        self._infer = OmniInference(ckpt_dir=ckpt_dir, device=device)

    def generate(
        self,
        audio_path: str,
        image_path: Optional[str] = None,
        return_numpy: bool = False,
    ) -> Union[bytes, np.ndarray]:
        """Generate a response audio for the given inputs.

        Args:
            audio_path: local path to input .wav (user question audio)
            image_path: local path to an image (currently accepted but unused)
            return_numpy: if True, return a `numpy.ndarray(dtype=np.int16)`; else return raw bytes

        Returns:
            bytes or numpy.ndarray: decoded response audio in 16-bit PCM format.
        """

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"audio file {audio_path} not found")

        # The existing `OmniInference.run_AT_batch_stream` yields audio chunks
        # (raw int16 bytes) produced by the Snac decoder. Iterate the stream
        # and concatenate the result into a single bytes object.
        out = bytearray()
        for chunk in self._infer.run_AT_batch_stream(audio_path):
            if chunk:
                out.extend(chunk)

        data = bytes(out)
        if return_numpy:
            arr = np.frombuffer(data, dtype=np.int16)
            return arr
        return data


__all__ = ["OmniEngine"]

"""A simple web interactive chat demo based on gradio."""

import os
import sys
import time
import gradio as gr
import base64
import numpy as np
import requests

# 添加项目根目录到 Python 路径，以便导入 inference 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_URL = os.getenv("API_URL", None)
client = None

if API_URL is None:
    from inference import OmniInference
    omni_client = OmniInference('./checkpoint', 'cuda:0')
    omni_client.warm_up()


OUT_CHUNK = 4096
OUT_RATE = 24000
OUT_CHANNELS = 1


def process_audio(audio):
    filepath = audio
    print(f"filepath: {filepath}")
    if filepath is None:
        return

    cnt = 0
    if API_URL is not None:
        with open(filepath, "rb") as f:
            data = f.read()
            base64_encoded = str(base64.b64encode(data), encoding="utf-8")
            files = {"audio": base64_encoded}
            tik = time.time()
            with requests.post(API_URL, json=files, stream=True) as response:
                try:
                    for chunk in response.iter_content(chunk_size=OUT_CHUNK):
                        if chunk:
                            # Convert chunk to numpy array
                            if cnt == 0:
                                print(f"first chunk time cost: {time.time() - tik:.3f}")
                            cnt += 1
                            audio_data = np.frombuffer(chunk, dtype=np.int16)
                            audio_data = audio_data.reshape(-1, OUT_CHANNELS)
                            yield OUT_RATE, audio_data.astype(np.int16)

                except Exception as e:
                    print(f"error: {e}")
    else:
        tik = time.time()
        for chunk in omni_client.run_AT_batch_stream(filepath):
            # Convert chunk to numpy array
            if cnt == 0:
                print(f"first chunk time cost: {time.time() - tik:.3f}")
            cnt += 1
            audio_data = np.frombuffer(chunk, dtype=np.int16)
            audio_data = audio_data.reshape(-1, OUT_CHANNELS)
            yield OUT_RATE, audio_data.astype(np.int16)


def main(port=None, share=False):
    # 设置环境变量以避免代理问题
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")

    demo = gr.Interface(
        process_audio,
        inputs=gr.Audio(type="filepath", label="Microphone"),
        outputs=[gr.Audio(label="Response", streaming=True, autoplay=True)],
        title="Chat Mini-Omni Demo",
        live=True,
    )
    
    launch_kwargs = {"share": share}
    if port is not None:
        launch_kwargs["server_name"] = "0.0.0.0"
        launch_kwargs["server_port"] = port
    else:
        launch_kwargs["server_name"] = "127.0.0.1"
    
    demo.queue().launch(**launch_kwargs)


if __name__ == "__main__":
    import fire

    fire.Fire(main)

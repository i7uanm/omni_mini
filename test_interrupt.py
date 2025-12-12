import os
import time
import threading
import numpy as np
import soundfile as sf
from omni_engine import OmniEngine
from realtime_vad_player import AudioPlayer


class MockAudioPlayer(AudioPlayer):
    """模拟播放器，不实际播放，只打印。"""
    def __init__(self):
        self._queue = []
        self._stop_flag = threading.Event()

    def play_chunk(self, chunk: bytes):
        print(f"Mock play chunk: {len(chunk)} bytes")

    def clear(self):
        self._queue.clear()
        print("Mock clear queue")

    def stop(self):
        self._stop_flag.set()
        print("Mock stop")


def test_interrupt_mechanism():
    """
    测试打断机制：使用两个音频文件模拟用户输入。
    - 第一个音频：触发生成和播放回复。
    - 第二个音频：在播放中途模拟新输入，触发打断。
    简化版：不使用 VAD，定时模拟打断。
    """
    # 初始化引擎
    engine = OmniEngine(ckpt_dir="./checkpoint", device="cuda:0")

    # 初始化播放器
    player = MockAudioPlayer()

    # 加载测试音频
    input1_path = "data/samples/output1.wav"
    input2_path = "data/samples/output1.wav"  # 用同一个模拟

    # 模拟缓冲和状态
    buffer_frames = []
    is_playing = False
    interrupt_event = threading.Event()

    # 模拟输入第一个音频（直接使用文件）
    print("使用第一个音频文件...")
    tmp_path = "tmp_test.wav"
    # 复制文件作为临时
    import shutil
    shutil.copy(input1_path, tmp_path)

    # 开始生成和播放
    interrupt_event.clear()
    is_playing = True
    print("开始播放回复...")

    # 启动播放线程
    def play_response():
        nonlocal is_playing
        for i, chunk in enumerate(engine.generate_stream(tmp_path, stream_stride=5), start=1):
            if interrupt_event.is_set():
                print("播放被打断。")
                break
            print(f"播放 Chunk {i}")
            player.play_chunk(chunk)
        is_playing = False
        print("播放完毕。")

    play_thread = threading.Thread(target=play_response)
    play_thread.start()

    # 等待播放开始，然后模拟打断（定时）
    time.sleep(2)  # 假设播放开始后2秒触发打断

    print("模拟打断...")
    interrupt_event.set()
    player.clear()
    is_playing = False

    # 等待播放线程结束
    play_thread.join()
    player.stop()

    print("测试完成。")


if __name__ == "__main__":
    test_interrupt_mechanism()
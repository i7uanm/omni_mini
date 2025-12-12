import os
import queue
import threading
import time
from typing import Optional, List

import numpy as np
import soundfile as sf
import torch
import pyaudio
import cv2

from omni_engine import OmniEngine


# =============== 配置参数 ===============

MIC_RATE = 16000       # 麦克风采样率（给 Whisper 用）
MIC_CHANNELS = 1
MIC_FORMAT = pyaudio.paInt16
MIC_CHUNK_MS = 100     # 每次从麦克风读 100ms
MIC_CHUNK = MIC_RATE * MIC_CHUNK_MS // 1000  # 每块样本数

# VAD 静音阈值：说话结束后，至少 0.5 秒静音才认为一句话结束
SILENCE_MS = 500
SILENCE_SAMPLES = MIC_RATE * SILENCE_MS // 1000

# 模型输出采样率（snac 是 24kHz）
ENGINE_SR = 24000


# =============== 播放线程 ===============

class AudioPlayer(threading.Thread):
    """单独的播放线程，负责播放从 OmniEngine 流式出来的 PCM chunk。"""

    def __init__(self, sample_rate: int = ENGINE_SR):
        super().__init__(daemon=True)
        self.sample_rate = sample_rate
        self._queue: "queue.Queue[bytes]" = queue.Queue()
        self._stop_flag = threading.Event()
        self._p = pyaudio.PyAudio()
        self._stream = self._p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
        )

    def run(self):
        while not self._stop_flag.is_set():
            try:
                chunk = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                # 约定 None 作为“清空播放”的标记
                continue
            self._stream.write(chunk)

    def play_chunk(self, chunk: bytes):
        """外部调用：提交一块 PCM 数据给播放器。"""
        self._queue.put(chunk)

    def clear(self):
        """清空当前播放队列（为下一阶段全双工打断做准备）。"""
        with self._queue.mutex:
            self._queue.queue.clear()

    def stop(self):
        self._stop_flag.set()
        time.sleep(0.2)
        self._stream.stop_stream()
        self._stream.close()
        self._p.terminate()


# =============== Silero VAD 加载 ===============

def load_silero_vad(device: str = "cpu"):
    """
    从 torch.hub 加载 Silero VAD 模型和工具函数。
    需要能访问 GitHub。
    """
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    model.to(device)
    model.eval()
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    return model, get_speech_timestamps


# =============== 实时 VAD + 缓冲逻辑 ===============

class VADMicLoop:
    """
    主体逻辑：
    - 不断从麦克风读数据，接入 Silero VAD；
    - 当检测到一句话结束时，把缓冲区里的音频写成临时 wav 文件；
    - 调用 OmniEngine.generate_stream，流式获得回复音频；
    - 把回复音频交给 AudioPlayer 播放。
    - 支持打断：如果在播放时检测到用户新语音，停止播放并清空队列。
    """

    def __init__(self, engine: OmniEngine, device: str = "cpu"):
        self.engine = engine
        self.device = device

        # 加载 Silero VAD
        print("Loading Silero VAD...")
        self.vad_model, self.get_speech_timestamps = load_silero_vad(device)
        print("Silero VAD loaded.")

        # 麦克风
        self._pa = pyaudio.PyAudio()
        self._mic_stream = self._pa.open(
            format=MIC_FORMAT,
            channels=MIC_CHANNELS,
            rate=MIC_RATE,
            input=True,
            frames_per_buffer=MIC_CHUNK,
        )

        # 播放器
        self.player = AudioPlayer(sample_rate=ENGINE_SR)
        self.player.start()

        # 视觉
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Warning: Could not open camera.")
        self.current_frame = None
        self.visual_thread = threading.Thread(target=self._visual_loop, daemon=True)
        self.visual_thread.start()

        # 缓冲区
        self._buffer_frames: List[bytes] = []
        self._running = False
        self.is_playing = False
        self.interrupt_event = threading.Event()
        self.last_was_speech = False
        self.current_image_path = None

    def _visual_loop(self):
        """视觉线程：每秒更新 current_frame。"""
        while self._running or not self._running:  # 运行直到程序结束
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
            time.sleep(1)  # 每秒更新

    def _capture_image(self):
        """抓取当前帧，resize 并保存为临时文件。"""
        if self.current_frame is None:
            return None
        # Resize 到 224x224 (假设模型需要)
        resized = cv2.resize(self.current_frame, (224, 224))
        # 保存为临时文件
        os.makedirs("tmp_images", exist_ok=True)
        path = os.path.join("tmp_images", f"frame_{int(time.time())}.jpg")
        cv2.imwrite(path, resized)
        return path

    def _buffer_to_wav_file(self, path: str):
        """把当前缓冲的麦克风数据写成一个 16kHz 单声道 wav 文件。"""
        if not self._buffer_frames:
            return False

        raw = b"".join(self._buffer_frames)
        audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        # 写成 float32 wav，whisper.load_audio 会自动处理
        sf.write(path, audio_np, MIC_RATE)
        return True

    def _detect_utterance_end(self, audio_np: np.ndarray) -> bool:
        """
        使用 Silero VAD 判断是否“说话结束”。

        简单策略：
        - 用 get_speech_timestamps 分析整段 buffer；
        - 如果存在语音段，取最后一个 end；
        - 如果 buffer 尾部静音超过 SILENCE_SAMPLES，则认为说话结束。
        """
        with torch.no_grad():
            tensor = torch.from_numpy(audio_np).to(self.device)
            timestamps = self.get_speech_timestamps(
                tensor,
                self.vad_model,
                sampling_rate=MIC_RATE,
            )

        if not timestamps:
            # 完全没检测到语音，当 buffer 很短时不认为结束，只认为还没开始说话
            return False

        last_end = timestamps[-1]["end"]
        silence = len(audio_np) - last_end
        # 尾部静音超过阈值，认为一句话结束
        return silence >= SILENCE_SAMPLES

    def _detect_new_speech(self, audio_np: np.ndarray) -> bool:
        """
        检测缓冲区中是否有任何语音段，用于打断机制。
        如果有语音段，认为用户又开始说话。
        """
        with torch.no_grad():
            tensor = torch.from_numpy(audio_np).to(self.device)
            timestamps = self.get_speech_timestamps(
                tensor,
                self.vad_model,
                sampling_rate=MIC_RATE,
            )
        return len(timestamps) > 0

    def loop(self):
        """
        主循环：
        - 从麦克风不断读 chunk，写入缓冲；
        - 每次更新 buffer 后用 VAD 检测是否说话结束；
        - 支持打断：如果在播放时检测到新语音，停止播放。
        - 支持视觉：检测说话开始时抓取图像。
        """
        print("开始实时 VAD 监听，按 Ctrl+C 退出。")
        self._running = True
        try:
            while self._running:
                data = self._mic_stream.read(MIC_CHUNK, exception_on_overflow=False)
                self._buffer_frames.append(data)

                # 转成 float32 波形做 VAD 检测
                raw = b"".join(self._buffer_frames)
                audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                if len(audio_np) < MIC_RATE * 0.5:
                    # 少于 0.5 秒，不做 VAD 判断
                    continue

                # 检测说话开始
                has_speech = self._detect_new_speech(audio_np)
                if has_speech and not self.last_was_speech:
                    print("[VAD] 检测到说话开始，抓取图像...")
                    self.current_image_path = self._capture_image()
                    self.last_was_speech = True

                # 检查打断：如果正在播放且检测到新语音，打断
                if self.is_playing and self._detect_new_speech(audio_np):
                    print("[Interrupt] 检测到用户新语音，打断当前播放。")
                    self.interrupt_event.set()
                    self.player.clear()
                    self.is_playing = False
                    # 清空缓冲区，准备新输入
                    self._buffer_frames.clear()
                    self.last_was_speech = False  # 重置
                    continue

                if self._detect_utterance_end(audio_np):
                    print("[VAD] 检测到一句话结束，开始调用 OmniEngine...")
                    # 把缓冲区写成临时 wav 文件
                    os.makedirs("tmp_inputs", exist_ok=True)
                    tmp_path = os.path.join("tmp_inputs", f"utt_{int(time.time())}.wav")
                    ok = self._buffer_to_wav_file(tmp_path)
                    # 清空缓冲区，为下一句做准备
                    self._buffer_frames.clear()
                    self.last_was_speech = False

                    if not ok:
                        print("[VAD] 缓冲为空，跳过。")
                        continue

                    # 清除打断事件，开始生成
                    self.interrupt_event.clear()
                    self.is_playing = True
                    # 调用流式引擎，并把 chunk 扔给播放器
                    for i, chunk in enumerate(self.engine.generate_stream(tmp_path, image_path=self.current_image_path, stream_stride=5), start=1):
                        if self.interrupt_event.is_set():
                            print("[Interrupt] 生成被打断。")
                            break
                        print(f"[Engine] Chunk {i} received, len={len(chunk)}")
                        self.player.play_chunk(chunk)

                    self.is_playing = False
                    self.current_image_path = None  # 重置
                    if not self.interrupt_event.is_set():
                        print("[Engine] 一句回复播放完毕，等待下一句语音输入。")

        except KeyboardInterrupt:
            print("收到 Ctrl+C，退出实时循环。")
        finally:
            self._running = False
            self._mic_stream.stop_stream()
            self._mic_stream.close()
            self._pa.terminate()
            self.player.stop()
            if self.cap.isOpened():
                self.cap.release()


def main():
    # 先检查当前系统是否有音频设备
    import pyaudio
    pa = pyaudio.PyAudio()
    dev_count = pa.get_device_count()
    if dev_count == 0:
        print("当前环境没有可用的音频设备（麦克风/扬声器），无法运行实时 VAD 录音播放。")
        print("建议在本地有声卡的机器上运行 realtime_vad_player.py，")
        print("在服务器上可以改用离线 wav + VAD 的脚本进行测试。")
        pa.terminate()
        return

    print(f"检测到 {dev_count} 个音频设备，继续初始化引擎与 VAD。")
    pa.terminate()

    # 初始化引擎
    engine = OmniEngine(ckpt_dir="./checkpoint", device="cuda:0")
    loop = VADMicLoop(engine, device="cpu")  # VAD 用 CPU 即可
    loop.loop()






if __name__ == "__main__":
    main()

from omni_engine import OmniEngine


def main():
    # 初始化引擎
    engine = OmniEngine(ckpt_dir="./checkpoint", device="cuda:0")

    # 输入测试音频路径，按你本地实际路径调整
    input_audio = "data/samples/output1.wav"

    print("开始流式生成与消费音频...")
    chunk_index = 0

    # 调用流式接口，逐 chunk 消费
    for chunk in engine.generate_stream(audio_path=input_audio, stream_stride=5):
        chunk_index += 1
        print(f"Chunk {chunk_index} received, length = {len(chunk)} bytes")

    print("流式生成结束。总共收到", chunk_index, "个 chunk。")


if __name__ == "__main__":
    main()

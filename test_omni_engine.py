from omni_engine import OmniEngine
import wave
import numpy as np

# Initialize the engine
engine = OmniEngine(ckpt_dir="./checkpoint", device="cuda:0")

# Path to input audio
input_audio = "data/samples/output1.wav"

# Save response audio as a valid .wav file
def save_as_wav(audio_data, output_path, sample_rate=24000):
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

# Generate response audio
try:
    response_audio = engine.generate(audio_path=input_audio)
    print(f"Response audio length: {len(response_audio)} bytes")

    # Save the response audio to a file
    save_as_wav(response_audio, "response.wav")

    print("Response audio saved as 'response.wav'")
except FileNotFoundError as e:
    print(f"Error: {e}")




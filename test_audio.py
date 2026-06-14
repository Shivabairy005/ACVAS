import sys
import time
import numpy as np
import pyaudio

def main():
    p = pyaudio.PyAudio()
    
    # Print system audio input devices to help debug
    print("=== Sound Input Devices ===")
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    default_input_idx = None
    
    for i in range(0, numdevices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get('maxInputChannels') > 0:
            is_default = " (DEFAULT)" if device_info.get('index') == p.get_default_input_device_info().get('index') else ""
            print(f"Device {device_info.get('index')}: {device_info.get('name')}{is_default}")
            if is_default:
                default_input_idx = device_info.get('index')

    if default_input_idx is None:
        try:
            default_device = p.get_default_input_device_info()
            default_input_idx = default_device.get('index')
            print(f"\nUsing default input device: {default_device.get('name')} (Index {default_input_idx})")
        except IOError:
            print("\nError: No default input device found! Please plug in a microphone.")
            p.terminate()
            return

    # Configuration
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024  # 1024 samples (about 64ms at 16kHz) for low latency/high reactivity

    print(f"\nOpening stream (Rate: {RATE}Hz, Chunk size: {CHUNK} samples)...")
    
    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=default_input_idx,
            frames_per_buffer=CHUNK
        )
    except Exception as e:
        print(f"Failed to open audio input stream: {e}")
        p.terminate()
        return

    print("Listening... Press Ctrl+C to stop.\n")
    print("RMS Volume Level Meter:")
    print("-" * 60)

    try:
        while True:
            # Read raw audio data from stream
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except IOError as e:
                # Sometimes overflow occurs, we can just skip or print warning
                continue
            
            # Convert binary data to numpy array of int16
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            
            # Normalize to [-1.0, 1.0] range
            audio_data /= 32768.0
            
            # Calculate RMS (Root Mean Square) volume level
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # Map RMS (typically 0.0 to 1.0, though normal speech is around 0.01 - 0.2) to a visual bar
            # We use a logarithmic or scaled multiplier to make it visually sensitive to normal talking
            bar_length = int(rms * 150)
            bar_length = min(bar_length, 50)  # Cap at 50 chars
            bar = "#" * bar_length
            spaces = " " * (50 - bar_length)
            
            # Print update in-place using carriage return
            sys.stdout.write(f"\rRMS: {rms:.4f} | [{bar}{spaces}]")
            sys.stdout.flush()
            
            # Small sleep to prevent CPU hogging (though stream.read blocks)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nStopping test...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Done.")

if __name__ == "__main__":
    main()

import asyncio
import os
import yaml
import numpy as np
import sys
import pyaudio

import capture

async def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=== Diagnostic: Testing capture.py ===")
    print(f"Sample Rate: {config['sample_rate']}")
    print(f"Chunk Duration: {config['chunk_duration_sec']} second(s)")
    
    audio_queue = asyncio.Queue(maxsize=5)
    
    # Run the capture task in the background
    capture_task = asyncio.create_task(capture.run_capture(audio_queue, config))
    
    print("Listening to audio queue... Press Ctrl+C to exit.")
    print("-" * 60)
    
    try:
        chunk_count = 0
        while True:
            # Get next chunk from the queue (blocks until a chunk is available)
            waveform = await audio_queue.get()
            chunk_count += 1
            
            # Calculate metrics
            rms = np.sqrt(np.mean(waveform ** 2))
            max_val = np.max(np.abs(waveform))
            
            # Print metrics
            bar_len = min(int(rms * 150), 30)
            bar = "#" * bar_len
            spaces = " " * (30 - bar_len)
            print(f"Chunk #{chunk_count:03d} | Length: {len(waveform)} samples | RMS: {rms:.5f} | Max: {max_val:.5f} | [{bar}{spaces}]")
            
    except KeyboardInterrupt:
        print("\nStopping diagnostic...")
    finally:
        capture_task.cancel()
        try:
            await capture_task
        except asyncio.CancelledError:
            pass
        print("Done.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

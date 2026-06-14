"""
capture.py — Audio capture for ACVAS.

Continuously reads 1-second chunks from the laptop microphone via PyAudio,
normalizes them to float32 in [-1.0, 1.0], and places them on an async queue.
"""

import asyncio
from functools import partial

import numpy as np
import pyaudio


async def run_capture(audio_queue: asyncio.Queue, config: dict) -> None:
    """Capture audio from the default microphone in an infinite loop.

    Parameters
    ----------
    audio_queue : asyncio.Queue
        Destination queue for normalised float32 waveforms.
        If the queue is full (maxsize=5), the chunk is silently dropped.
    config : dict
        Parsed config.yaml — needs ``sample_rate`` and
        ``chunk_duration_sec``.
    """
    rate = config["sample_rate"]
    chunk_size = int(rate * config["chunk_duration_sec"])  # Ensure integer size

    loop = asyncio.get_event_loop()

    while True:
        pa = None
        stream = None
        try:
            pa = pyaudio.PyAudio()
            
            # Retrieve default input device explicitly for Windows compatibility
            try:
                default_device = pa.get_default_input_device_info()
                default_input_idx = default_device.get('index')
                print(f"[capture] Using default input device: {default_device.get('name')} (Index {default_input_idx})")
            except IOError:
                print("[capture] Error: No default input device found! Retrying in 2s...")
                await asyncio.sleep(2.0)
                continue

            stream = pa.open(
                rate=rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                input_device_index=default_input_idx,
                frames_per_buffer=chunk_size,
            )
            print("[capture] Microphone stream opened - listening ...")

            while True:
                # Read raw bytes from microphone (blocking — run in executor)
                raw = await loop.run_in_executor(
                    None,
                    partial(stream.read, chunk_size, exception_on_overflow=False),
                )

                # Validate buffer length to prevent np.frombuffer crashes
                expected_bytes = chunk_size * 2
                if len(raw) < expected_bytes:
                    print(f"[capture] Warning: Short read ({len(raw)} bytes, expected {expected_bytes}). Skipping chunk.")
                    continue

                # Convert to normalised float32 waveform in [-1.0, 1.0]
                waveform = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                waveform /= 32768.0

                rms = np.sqrt(np.mean(waveform ** 2))
                print(f"[capture] Captured chunk: size={len(waveform)}, RMS={rms:.6f}")

                # Non-blocking put — drop chunk if queue is full
                try:
                    audio_queue.put_nowait(waveform)
                except asyncio.QueueFull:
                    pass  # Silently drop the chunk
        except Exception as e:
            print(f"[capture] Error in audio capture stream: {e}. Reconnecting in 2s...")
            await asyncio.sleep(2.0)
        finally:
            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception:
                    pass
                try:
                    stream.close()
                except Exception:
                    pass
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass

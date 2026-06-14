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
    chunk_size = rate * config["chunk_duration_sec"]  # 16000 samples = 1 s

    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=chunk_size,
    )

    loop = asyncio.get_event_loop()
    print("[capture] Microphone stream opened — listening …")

    try:
        while True:
            try:
                # Read raw bytes from microphone (blocking — run in executor)
                raw = await loop.run_in_executor(
                    None,
                    partial(stream.read, chunk_size, exception_on_overflow=False),
                )
            except Exception as e:
                print(f"[capture] Error reading audio stream: {e}")
                await asyncio.sleep(0.1)
                continue

            # Convert to normalised float32 waveform in [-1.0, 1.0]
            waveform = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            waveform /= 32768.0

            # Non-blocking put — drop chunk if queue is full
            try:
                audio_queue.put_nowait(waveform)
            except asyncio.QueueFull:
                pass  # Silently drop the chunk
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

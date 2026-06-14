"""
main.py — ACVAS orchestrator.

Wires together all modules and runs five concurrent async coroutines:
1. capture.run_capture  — microphone → audio_queue
2. inference_loop       — audio_queue → event_queue (via run_in_executor)
3. pipeline_loop        — event_queue → broadcast_queue (classifier + hysteresis + actuator + logger)
4. broadcast_loop       — broadcast_queue → WebSocket clients
5. server.start_server  — WebSocket + HTTP servers
"""

import asyncio
import os
import sys

import numpy as np
import yaml

import capture
import inference
import classifier
import hysteresis as hyst_mod
import actuator
import server
import logger


async def inference_loop(
    audio_queue: asyncio.Queue,
    event_queue: asyncio.Queue,
    config: dict,
) -> None:
    """Read waveforms from *audio_queue*, run YAMNet in a thread, and push
    ``(top_idx, score, rms)`` results into *event_queue*.
    """
    loop = asyncio.get_event_loop()
    print("[main] Inference loop started")

    while True:
        waveform = await audio_queue.get()
        rms = float(np.sqrt(np.mean(waveform ** 2)))
        top_idx, score = await loop.run_in_executor(
            None, inference.run_yamnet, waveform,
        )
        try:
            event_queue.put_nowait((top_idx, score, rms))
        except asyncio.QueueFull:
            pass  # Drop if downstream is slow


async def pipeline_loop(
    event_queue: asyncio.Queue,
    broadcast_queue: asyncio.Queue,
    config: dict,
) -> None:
    """Consume inference results, run classifier → hysteresis → actuator →
    logger, and enqueue broadcast payloads.
    """
    hyst = hyst_mod.HysteresisFilter(config)
    loop = asyncio.get_event_loop()

    # Retrieve current volume as the baseline last known volume
    try:
        last_known_volume = await loop.run_in_executor(None, actuator.get_current_volume)
    except Exception:
        last_known_volume = 0.5

    print("[main] Pipeline loop started")

    while True:
        top_idx, score, rms = await event_queue.get()

        # Classify
        env_label, conf = classifier.classify(top_idx, score, rms, config)

        # Hysteresis filter — returns new label only on confirmed transition
        confirmed = hyst.update(env_label)

        if confirmed is not None:
            if confirmed == "unknown":
                # Reset to the default (middle) volume instead of sticking to last known
                target_vol = config.get("default_volume", 0.50)
            else:
                # Look up target volume for the confirmed environment
                env_cfg = config["environments"].get(confirmed)
                target_vol = env_cfg["volume"] if env_cfg else config.get("default_volume", 0.50)
                last_known_volume = target_vol

            # Ramp volume (blocking — dispatch to executor, passing config parameters)
            await loop.run_in_executor(None, actuator.set_volume, target_vol, config)

            # Log the transition
            logger.log_event(confirmed, conf, target_vol)
            print(f"[main] Environment -> {confirmed} (conf={conf:.2f}, vol={target_vol:.0%})")

            # Enqueue broadcast payload
            payload = {
                "env": confirmed,
                "confidence": round(conf, 4),
                "volume": round(target_vol, 2),
            }
            try:
                broadcast_queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass


async def broadcast_loop(broadcast_queue: asyncio.Queue) -> None:
    """Read payloads from *broadcast_queue* and send to all WebSocket
    clients via ``server.broadcast()``.
    """
    print("[main] Broadcast loop started")

    while True:
        payload = await broadcast_queue.get()
        await server.broadcast(payload)


def main() -> None:
    """Entry point — load config, load model, run the async event loop."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print("  ACVAS - Ambient Context Volume Adaptation System")
    print("=" * 50)
    print()

    # Load YAMNet model (blocking, before event loop)
    inference.load_model()
    
    # Query current volume to initialize server state prior to loop start
    try:
        initial_vol = actuator.get_current_volume()
        server.STATE["volume"] = initial_vol
    except Exception as e:
        print(f"[main] Warning: Could not retrieve initial system volume: {e}")
        
    print()

    # Create async queues
    audio_queue     = asyncio.Queue(maxsize=5)
    event_queue     = asyncio.Queue(maxsize=5)
    broadcast_queue = asyncio.Queue(maxsize=10)

    # Run all five coroutines concurrently
    async def run_all():
        await asyncio.gather(
            capture.run_capture(audio_queue, config),
            inference_loop(audio_queue, event_queue, config),
            pipeline_loop(event_queue, broadcast_queue, config),
            broadcast_loop(broadcast_queue),
            server.start_server(config),
        )

    print("[main] Starting ACVAS ...")
    print(f"[main] Dashboard: http://0.0.0.0:{config['http_port']}")
    print(f"[main] WebSocket: ws://0.0.0.0:{config['ws_port']}")
    print()

    try:
        asyncio.run(run_all())
    except KeyboardInterrupt:
        print("\n[main] ACVAS stopped.")


if __name__ == "__main__":
    main()

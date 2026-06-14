"""
inference.py — YAMNet audio classification for ACVAS.

Loads the YAMNet model from TensorFlow Hub and runs inference on 1-second
audio waveforms, returning the top class index and its confidence score.
"""

import sys
import numpy as np
import tensorflow_hub as hub


# Module-level global — populated by load_model() before the event loop starts
model = None


def load_model() -> None:
    """Download and load the YAMNet model from TensorFlow Hub.

    This is a blocking call (~20 MB download on first run, cached afterwards).
    Must be called once from main.py before ``asyncio.run()`` starts.
    """
    global model
    print("[inference] Loading YAMNet model from TensorFlow Hub …")
    try:
        model = hub.load("https://tfhub.dev/google/yamnet/1")
        print("[inference] YAMNet model loaded successfully.")
    except Exception as e:
        print(f"\n[ERROR] Failed to load YAMNet model from TF Hub: {e}")
        print("Please check your internet connection and verify tensorflow-hub installation.")
        sys.exit(1)


def run_yamnet(waveform: np.ndarray) -> tuple[int, float]:
    """Run YAMNet inference on a 1-second waveform.

    Parameters
    ----------
    waveform : np.ndarray
        1-D float32 array of audio samples in [-1.0, 1.0], length 16 000.

    Returns
    -------
    tuple[int, float]
        ``(top_class_index, confidence)`` where ``top_class_index`` is the
        YAMNet class with the highest mean score (0–520) and ``confidence``
        is the corresponding score (0.0–1.0).
    """
    if model is None:
        raise RuntimeError("YAMNet model not loaded. Call load_model() first.")

    scores, embeddings, spectrogram = model(waveform)
    mean_scores = np.mean(scores.numpy(), axis=0)  # shape (521,)
    top_idx = int(np.argmax(mean_scores))
    confidence = float(np.max(mean_scores))
    return top_idx, confidence

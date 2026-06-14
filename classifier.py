"""
classifier.py — Environment classifier for ACVAS.

Maps a YAMNet top-class index to one of the predefined environment labels
(silent, office, home, crowded) using the indices defined in config.yaml.
"""


# Hardcoded mirror of config.yaml environments block for quick lookup.
# Keys = environment labels, values = set of YAMNet class indices.
YAMNET_ENV_MAP: dict[str, set[int]] = {
    "silent":  {494},
    "office":  {1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 63, 65, 66, 378, 380, 383, 385},
    "home":    {0, 132, 518, 519},
    "crowded": {6, 7, 8, 9, 10, 11, 64, 321, 390},
}


def classify(top_idx: int, score: float, rms: float, config: dict) -> tuple[str, float]:
    """Classify a YAMNet result into an environment label using physical energy (RMS)
    and semantic class.

    Parameters
    ----------
    top_idx : int
        The YAMNet class index with the highest mean score.
    score : float
        The confidence score for that class.
    rms : float
        The Root Mean Square physical energy of the audio chunk.
    config : dict
        Parsed config.yaml contents.

    Returns
    -------
    tuple[str, float]
        ``(env_label, score)`` — the matched environment or ``"unknown"``.
    """
    rms_silent = config.get("rms_silent_threshold", 0.001)
    rms_crowded = config.get("rms_crowded_threshold", 0.03)

    # 1. Absolute Silent Zone (low energy)
    if rms < rms_silent:
        return ("silent", 1.0)

    # 2. Loud/Crowded Zone (high energy)
    if rms > rms_crowded:
        return ("crowded", 1.0)

    # 3. Normal Zone (medium energy) - semantic classification
    if score < config["confidence_threshold"]:
        return ("unknown", score)

    for env_name, env_cfg in config["environments"].items():
        if top_idx in env_cfg["yamnet_indices"]:
            return (env_name, score)

    return ("unknown", score)

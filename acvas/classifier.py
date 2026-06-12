"""
classifier.py — Environment classifier for ACVAS.

Maps a YAMNet top-class index to one of the predefined environment labels
(silent, office, home, crowded) using the indices defined in config.yaml.
"""


# Hardcoded mirror of config.yaml environments block for quick lookup.
# Keys = environment labels, values = set of YAMNet class indices.
YAMNET_ENV_MAP: dict[str, set[int]] = {
    "silent":  {0, 9},
    "office":  {1, 2, 3, 132},
    "home":    {137, 138, 139, 140},
    "crowded": {5, 6, 40, 396},
}


def classify(top_idx: int, score: float, config: dict) -> tuple[str, float]:
    """Classify a YAMNet result into an environment label.

    Parameters
    ----------
    top_idx : int
        The YAMNet class index with the highest mean score.
    score : float
        The confidence score for that class.
    config : dict
        Parsed config.yaml contents (needs ``confidence_threshold`` and
        ``environments`` keys).

    Returns
    -------
    tuple[str, float]
        ``(env_label, score)`` — the matched environment or ``"unknown"`` if
        the score is below threshold or the index doesn't match any env.
    """
    if score < config["confidence_threshold"]:
        return ("unknown", score)

    for env_name, env_cfg in config["environments"].items():
        if top_idx in env_cfg["yamnet_indices"]:
            return (env_name, score)

    return ("unknown", score)

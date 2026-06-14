"""
logger.py — Event logger for ACVAS.

Appends environment transition events to acvas_log.csv with columns:
timestamp (ISO format), env, confidence (2dp), volume (2dp).
"""

import csv
import os
from datetime import datetime, timezone


_LOG_FILE = "acvas_log.csv"
_HEADER = ["timestamp", "env", "confidence", "volume"]


def log_event(env: str, confidence: float, volume: float) -> None:
    """Append one environment-transition row to the CSV log file.

    Creates the file with a header row on first call if it does not exist.
    """
    file_exists = os.path.isfile(_LOG_FILE)

    with open(_LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(_HEADER)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            env,
            f"{confidence:.2f}",
            f"{volume:.2f}",
        ])

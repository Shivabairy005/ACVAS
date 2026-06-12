"""
hysteresis.py — Hysteresis filter for ACVAS.

Prevents rapid environment switching by requiring a majority vote over a
sliding window and a minimum hold time between transitions.
"""

import time
from collections import Counter, deque


class HysteresisFilter:
    """Sliding-window majority-vote filter with cooldown.

    A new environment label is only confirmed when:
    1. It appears ≥ ``majority`` times in the last ``frames`` observations.
    2. At least ``min_hold_seconds`` have elapsed since the last transition.
    3. The label differs from the currently confirmed environment.
    """

    def __init__(self, config: dict) -> None:
        self._window: deque[str] = deque(maxlen=config["hysteresis_frames"])
        self._majority: int = config["hysteresis_majority"]
        self._min_hold: float = config["min_hold_seconds"]
        self.last_change_time: float = 0.0
        self.last_confirmed: str | None = None

    def update(self, env_label: str) -> str | None:
        """Feed a new observation and return the confirmed label, or None.

        Parameters
        ----------
        env_label : str
            The environment label produced by the classifier for the
            most recent audio chunk.

        Returns
        -------
        str | None
            The newly confirmed environment label if a transition is
            warranted, otherwise ``None``.
        """
        self._window.append(env_label)

        # Find the most common label in the sliding window
        counter = Counter(self._window)
        most_common_label, count = counter.most_common(1)[0]

        if (
            count >= self._majority
            and time.time() - self.last_change_time >= self._min_hold
            and most_common_label != self.last_confirmed
        ):
            self.last_change_time = time.time()
            self.last_confirmed = most_common_label
            return most_common_label

        return None

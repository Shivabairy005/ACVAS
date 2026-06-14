"""
actuator.py — System volume control for ACVAS.

Smoothly ramps the OS master volume to a target level.
Windows: uses pycaw (COM-based IAudioEndpointVolume).
Linux:   uses amixer subprocess.
"""

import sys
import time


def _get_windows_volume_interface():
    """Return the IAudioEndpointVolume COM interface (Windows only)."""
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from comtypes import CLSCTX_ALL

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return interface.QueryInterface(IAudioEndpointVolume)


def get_current_volume() -> float:
    """Return the current system master volume as a float in [0.0, 1.0]."""
    if sys.platform == "win32":
        import comtypes
        co_init = False
        try:
            comtypes.CoInitialize()
            co_init = True
        except OSError:
            pass
        try:
            volume = _get_windows_volume_interface()
            return volume.GetMasterVolumeLevelScalar()
        finally:
            if co_init:
                comtypes.CoUninitialize()
    else:
        # Linux fallback — parse amixer output
        import subprocess
        result = subprocess.run(
            ["amixer", "get", "Master"],
            capture_output=True, text=True, check=True,
        )
        # Extract percentage from output like "[75%]"
        for line in result.stdout.splitlines():
            if "%" in line:
                start = line.index("[") + 1
                end = line.index("%")
                return int(line[start:end]) / 100.0
        return 0.5  # Fallback default


def set_volume(target: float, config: dict = None) -> None:
    """Ramp the system volume from current level to *target* (0.0–1.0).

    Steps and interval can be customized via the config dictionary.
    This is a blocking call — dispatch via ``run_in_executor`` when
    called from an async context.
    """
    target = max(0.0, min(1.0, target))

    if sys.platform == "win32":
        import comtypes
        co_init = False
        try:
            comtypes.CoInitialize()
            co_init = True
        except OSError:
            pass
        try:
            volume_ctl = _get_windows_volume_interface()
            current = volume_ctl.GetMasterVolumeLevelScalar()
            
            step = config.get("volume_ramp_step", 0.05) if config else 0.05
            interval = (config.get("ramp_interval_ms", 50) / 1000.0) if config else 0.05

            if abs(current - target) < step:
                volume_ctl.SetMasterVolumeLevelScalar(target, None)
                return

            direction = 1 if target > current else -1
            level = current

            while (direction == 1 and level < target) or \
                  (direction == -1 and level > target):
                level += direction * step
                # Clamp to [0.0, 1.0] and don't overshoot target
                if direction == 1:
                    level = min(level, target)
                else:
                    level = max(level, target)
                volume_ctl.SetMasterVolumeLevelScalar(level, None)
                time.sleep(interval)
        finally:
            if co_init:
                comtypes.CoUninitialize()
    else:
        # Linux — single amixer call (no ramping granularity via CLI)
        import subprocess
        subprocess.run(
            ["amixer", "set", "Master", f"{int(target * 100)}%"],
            check=True,
        )

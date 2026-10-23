"""
Simple simulated fire/smoke alarm.

The reference paper's title promises an "early fire alarm system," but the
paper itself only ever performs detection — it never actually implements an
alarm/notification step. This module adds that missing piece: when smoke is
detected, it raises a clear visual alert in the terminal and plays a beep
sound (Windows-native, no extra install needed; silently skipped on other
operating systems where winsound isn't available).
"""
from __future__ import annotations

try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    _HAS_WINSOUND = False


def raise_alert(image_name: str, num_smoke_regions: int) -> None:
    """Print a red alert banner and beep, simulating an early fire alarm trigger."""
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"\n{RED}{BOLD}{'=' * 50}")
    print(f"  \U0001F6A8 ALERT: SMOKE DETECTED \U0001F6A8")
    print(f"  Image: {image_name}")
    print(f"  Smoke regions found: {num_smoke_regions}")
    print(f"{'=' * 50}{RESET}\n")

    if _HAS_WINSOUND:
        # three short beeps: frequency (Hz), duration (ms)
        for _ in range(3):
            winsound.Beep(1000, 200)

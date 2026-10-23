
from __future__ import annotations
import urllib.request

TOPIC = "smoke-alert-hp-84213"  # Must match the topic in your ntfy app


def send_phone_alert(image_name: str, num_smoke_regions: int) -> None:
    message = ( 
        f"SMOKE DETECTED in {image_name} "
        f"({num_smoke_regions} region(s) flagged). "
        "Student prototype alert - verify before acting."
    )

    req = urllib.request.Request(
        f"https://ntfy.sh/{TOPIC}",
        data=message.encode("utf-8"),
        headers={
            "Title": "FIRE / SMOKE EMERGENCY",
            "Priority": "5",
            "Tags": "rotating_light,fire",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(
                f"  Phone alert sent (HTTP {resp.status}) "
                f"to topic '{TOPIC}'"
            )
            print("  Maximum-priority notification requested.")
            print("  Check phone sound and vibration settings.")

    except Exception as e:
        print(f"  FAILED to send phone alert: {e}")
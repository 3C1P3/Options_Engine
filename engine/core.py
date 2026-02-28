"""
Options Engine core orchestrator.

Responsibilities (current version):
- Define symbol rotation (NVDA, IBIT, etc.).
- Manage paper/live mode flag.
- Run the live scalper strategy loop.

Later:
- Add scheduling for spread-bot scans.
- Centralize config loading and risk controls.
"""

import time
from datetime import datetime, time as dtime

# For now we import the scalper directly.
# live_scalper_clean.py currently has its own asyncio main() and event loop.
# We'll treat it as the "engine" for intraday scalping.
from strategies import live_scalper_clean as scalper


# Symbols the engine is responsible for.
SYMBOLS = ["NVDA", "IBIT"]  # extend later

# Mode: "paper" or "live"
MODE = "paper"


def is_market_open(now: datetime | None = None) -> bool:
    """
    Very simple local-time gate: 09:30–16:00.
    Replace with a proper calendar later.
    """
    now = now or datetime.now()
    return dtime(9, 30) <= now.time() <= dtime(16, 0)


def run_engine():
    """
    Top-level engine entry point.

    Current behavior:
    - Checks market hours.
    - When open, hands control to the live scalper.
    - When closed, sleeps and re-checks.
    """
    print("=== Options Engine starting ===")
    print(f"Mode: {MODE}")
    print(f"Symbols (engine-level): {', '.join(SYMBOLS)}")
    print("NOTE: live_scalper_clean uses its own WATCHLIST internally.\n")

    while True:
        if not is_market_open():
            print(f"[ENGINE] Market closed at {datetime.now().isoformat()}. Sleeping 60s...")
            time.sleep(60)
            continue

        print(f"[ENGINE] Market open at {datetime.now().isoformat()}.")
        print("[ENGINE] Handing off to live scalper (continuous loop)...")

        # For now, we simply run the scalper's own main.
        # This will block here until the scalper exits (Ctrl+C).
        try:
            scalper.main()  # assuming live_scalper_clean exposes main()
        except KeyboardInterrupt:
            print("[ENGINE] Received KeyboardInterrupt. Stopping engine.")
            break
        except Exception as e:
            print(f"[ENGINE] Scalper error: {e}. Restarting in 10 seconds...")
            time.sleep(10)
        else:
            # If scalper.main() returns cleanly, break out unless you want to restart.
            print("[ENGINE] Scalper exited normally. Stopping engine.")
            break

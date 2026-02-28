import time
from datetime import datetime, time as dtime

SYMBOLS = ["NVDA", "IBIT"]  # extend later

def is_market_open(now=None):
    now = now or datetime.now()
    # crude: 9:30–16:00 local; replace with proper calendar later
    return dtime(9, 30) <= now.time() <= dtime(16, 0)

def run_engine():
    print("Options Engine starting...")
    mode = "paper"  # or "live"
    print(f"Mode: {mode}")
    print(f"Symbols: {', '.join(SYMBOLS)}")

    while True:
        if not is_market_open():
            print("Market closed. Sleeping 60s...")
            time.sleep(60)
            continue

        loop_start = datetime.now()
        for symbol in SYMBOLS:
            process_symbol(symbol, mode)

        # simple pacing: one full rotation per second
        elapsed = (datetime.now() - loop_start).total_seconds()
        sleep_for = max(0.0, 1.0 - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)

def process_symbol(symbol, mode):
    # TODO: hook in Schwab quotes/chains and strategies
    print(f"[{datetime.now().isoformat()}] Processing {symbol} in {mode} mode")

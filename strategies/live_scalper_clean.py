import asyncio
import json
from datetime import datetime, timezone
import csv
import os

from schwab.auth import easy_client
from schwab.client import Client
from schwab.streaming import StreamClient

# ====== CONFIG – EDIT THESE ======
API_KEY = "hVlTy40IKJ5YFxjuIdpXw5W4FeBFAq4KJ262rTrF2lP2p5mX"
APP_SECRET = "TidSTWaPXo1Z0uaE7ijQEK0v3qJlZ3ot7XhbQuL00IF47rxnWCAjZjC5Mw37Qmft"
CALLBACK_URL = "https://127.0.0.1:8000/callback"
TOKEN_PATH = "tokens.json"           # same as in auth_refresh.py
ACCOUNT_ID = "64482506"              # numeric account ID, no dashes, as string

WATCHLIST = ["INTC", "NVDA", "QQQ", "IBIT", "SPY"]
BAR_MINUTES = 5
MAX_BARS_STORED = 300
# =================================

# ====== RANGE FILTER CONFIG ======
RANGE_WINDOW = 3          # use last 3 closed bars
MIN_RANGE_AVG = 0.30      # your 0.30 rule for ALL tickers
# =================================

# ====== TREND FILTER CONFIG (EMA200 + persisted state) ======
EMA_LENGTH = 200              # 200‑period EMA on closes
EMA_STATE_FILE = "ema_state.json"
# ========================================

# ====== RISK / CAPITAL STATE ======
capital = 500.0
daily_realized_pnl = 0.0
DAILY_LOSS_LIMIT = -500.0
RISK_PER_TRADE_FRACTION = 0.5  # 50% per-trade risk
# ==================================

# ====== EXECUTION / OPEN TRADES (PAPER) ======
open_trades = []  # list of dicts, one per paper “option trade”
# ============================================

LOG_FILE = "paper_trades.csv"

def init_log():
    """Create the CSV header once if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "symbol",
                "direction",
                "capital",
                "max_loss_per_trade",
                "max_premium_per_contract",
                "bar_open",
                "bar_high",
                "bar_low",
                "bar_close",
                "bar_volume",
            ])

# ====== EMA STATE PERSISTENCE HELPERS ======
def load_ema_state():
    """Load persisted EMA state per symbol from disk."""
    if not os.path.exists(EMA_STATE_FILE):
        return {}
    try:
        with open(EMA_STATE_FILE, "r") as f:
            raw = json.load(f)
        # Normalize structure: {symbol: {"ema": float, "bars_seen": int}}
        out = {}
        for sym, val in raw.items():
            if isinstance(val, dict) and "ema" in val and "bars_seen" in val:
                out[sym] = {
                    "ema": float(val["ema"]),
                    "bars_seen": int(val["bars_seen"]),
                }
        return out
    except Exception as e:
        print(f"[EMA_STATE] Error loading {EMA_STATE_FILE}: {e}")
        return {}

def save_ema_state(ema_state):
    """Persist EMA state dict to disk."""
    try:
        serializable = {}
        for sym, val in ema_state.items():
            serializable[sym] = {
                "ema": float(val.get("ema", 0.0)),
                "bars_seen": int(val.get("bars_seen", 0)),
            }
        with open(EMA_STATE_FILE, "w") as f:
            json.dump(serializable, f)
    except Exception as e:
        print(f"[EMA_STATE] Error saving {EMA_STATE_FILE}: {e}")

def update_ema_for_symbol(ema_state, symbol, close_price):
    """
    Incremental EMA200 update per symbol using persisted state.
    Returns current EMA (or None if not enough history yet).
    """
    alpha = 2 / (EMA_LENGTH + 1)

    state = ema_state.get(symbol)
    if state is None:
        # First bar ever for this symbol in this file
        ema = close_price
        bars_seen = 1
    else:
        ema_prev = state["ema"]
        bars_seen_prev = state["bars_seen"]
        # Standard recursive EMA update
        ema = close_price * alpha + ema_prev * (1 - alpha)
        bars_seen = bars_seen_prev + 1

    ema_state[symbol] = {"ema": ema, "bars_seen": bars_seen}

    # Only treat EMA as "valid" once we've effectively seen at least EMA_LENGTH bars.
    # Prior sessions contribute via bars_seen, so you don't restart from zero each day.
    if bars_seen < EMA_LENGTH:
        return None
    return ema
# =============================================

# ====== OPTION CHAIN HELPER ======
def fetch_clean_chain(client: Client, symbol: str):
    """
    Call Schwab option chain endpoint and return a flat list of options
    with a uniform structure.
    """
    resp = client.get_option_chain(symbol)
    data = resp.json()

    clean = []

    call_map = data.get("callExpDateMap", {}) or {}
    put_map = data.get("putExpDateMap", {}) or {}

    # Calls
    for exp_key, strikes in call_map.items():
        for strike_key, options in strikes.items():
            for opt in options:
                bid = float(opt.get("bid", 0.0) or 0.0)
                ask = float(opt.get("ask", 0.0) or 0.0)
                mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask, 0.0)

                clean.append({
                    "underlying": symbol,
                    "expiry": exp_key,
                    "strike": float(opt.get("strikePrice", strike_key)),
                    "side": "CALL",
                    "delta": float(opt.get("delta", 0.0) or 0.0),
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "raw": opt,
                })

    # Puts
    for exp_key, strikes in put_map.items():
        for strike_key, options in strikes.items():
            for opt in options:
                bid = float(opt.get("bid", 0.0) or 0.0)
                ask = float(opt.get("ask", 0.0) or 0.0)
                mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask, 0.0)

                clean.append({
                    "underlying": symbol,
                    "expiry": exp_key,
                    "strike": float(opt.get("strikePrice", strike_key)),
                    "side": "PUT",
                    "delta": float(opt.get("delta", 0.0) or 0.0),
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "raw": opt,
                })

    return clean

# ====== REAL DEEP ITM OPTION PICKER ======
def pick_deep_itm_contract(client: Client,
                           symbol: str,
                           direction: str,
                           max_premium_per_contract: float):
    """
    Deep ITM option selection:
    - Filter by side, |delta| >= 0.80, mid <= max_premium_per_contract
    - Prefer farthest expiry first, then highest |delta| in that expiry.
    """
    print(
        f"[PICKER] Searching chain for {direction} on {symbol} "
        f"max premium {max_premium_per_contract:.2f} "
        f"(deep ITM, LEAPS -> 0DTE)."
    )

    options = fetch_clean_chain(client, symbol)

    side = "CALL" if direction == "BUY_CALL" else "PUT"
    side_opts = [o for o in options if o["side"] == side]

    if not side_opts:
        print(f"[PICKER] No {side} options found for {symbol}.")
        return None

    expiries = sorted({o["expiry"] for o in side_opts}, reverse=True)

    best_candidate = None
    for exp in expiries:
        opts_this_exp = [o for o in side_opts if o["expiry"] == exp]
        candidates = []
        for o in opts_this_exp:
            delta_abs = abs(o["delta"])
            mid = o["mid"]
            if delta_abs >= 0.80 and mid <= max_premium_per_contract:
                candidates.append(o)
        if candidates:
            best_here = max(candidates, key=lambda x: abs(x["delta"]))
            best_candidate = best_here
            break

    if not best_candidate:
        print(
            f"[PICKER] No deep ITM {side} contract for {symbol} "
            f"under premium {max_premium_per_contract:.2f}."
        )
        return None

    opt_raw = best_candidate["raw"]

    contract = {
        "underlying": symbol,
        "option_symbol": opt_raw.get("symbol"),
        "direction": direction,
        "side": best_candidate["side"],
        "expiry": opt_raw.get("expirationDate", best_candidate["expiry"]),
        "strike": best_candidate["strike"],
        "premium": best_candidate["mid"],
        "delta": best_candidate["delta"],
    }

    print(
        f"[PICKER] Selected {contract['option_symbol']} "
        f"exp={contract['expiry']} strike={contract['strike']} "
        f"mid={contract['premium']:.2f} delta={contract['delta']:.3f}"
    )

    return contract
# =========================================

# ====== PAPER EXECUTION BOT (STYLE 1) ======
def execution_on_signal(symbol: str,
                        direction: str,
                        max_premium_per_contract: float,
                        bar_info: dict,
                        contract: dict | None):
    global open_trades

    entry_price = max_premium_per_contract
    tp_price = entry_price * 1.10   # +10% TP
    sl_price = entry_price * 0.95   # -5% SL

    trade_id = len(open_trades) + 1
    option_symbol = contract["option_symbol"] if contract else None

    open_trade = {
        "trade_id": trade_id,
        "underlying": symbol,
        "option_symbol": option_symbol,
        "direction": direction,
        "entry_price": entry_price,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "bar_datetime": bar_info["datetime"],
        "bar_open": bar_info["open"],
        "bar_high": bar_info["high"],
        "bar_low": bar_info["low"],
        "bar_close": bar_info["close"],
        "bar_volume": bar_info["volume"],
    }

    open_trades.append(open_trade)

    print(
        f"[EXEC] Created paper trade #{trade_id} for {symbol} {direction} "
        f"opt={option_symbol} entry={entry_price:.2f} TP={tp_price:.2f} SL={sl_price:.2f}"
    )

    exec_log = "paper_option_trades.csv"
    file_exists = os.path.exists(exec_log)
    with open(exec_log, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "trade_id", "underlying", "option_symbol", "direction",
                "entry_price", "tp_price", "sl_price",
                "bar_datetime", "bar_open", "bar_high",
                "bar_low", "bar_close", "bar_volume",
            ])
        writer.writerow([
            trade_id, symbol, option_symbol, direction,
            f"{entry_price:.2f}", f"{tp_price:.2f}", f"{sl_price:.2f}",
            bar_info["datetime"],
            f"{bar_info['open']:.2f}",
            f"{bar_info['high']:.2f}",
            f"{bar_info['low']:.2f}",
            f"{bar_info['close']:.2f}",
            bar_info["volume"],
        ])
# ==========================================

def floor_to_interval(ts_ms: int, minutes: int) -> datetime:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    floored_minute = (dt.minute // minutes) * minutes
    return dt.replace(minute=floored_minute, second=0, microsecond=0)

def is_trading_time(dt: datetime) -> bool:
    local_dt = dt.astimezone()
    hour = local_dt.hour
    minute = local_dt.minute

    in_open = (hour == 9 and minute >= 30) or (10 <= hour < 11)
    in_afternoon = (14 <= hour < 16)

    return in_open or in_afternoon

async def main():
    global capital, daily_realized_pnl

    init_log()

    client: Client = easy_client(
        api_key=API_KEY,
        app_secret=APP_SECRET,
        callback_url=CALLBACK_URL,
        token_path=TOKEN_PATH,
    )

    stream_client = StreamClient(client, account_id=int(ACCOUNT_ID))

    bar_state = {}
    history = {sym: [] for sym in WATCHLIST}

    # Load EMA state from previous sessions
    ema_state = load_ema_state()
    print(f"[EMA_STATE] Loaded EMA state: {ema_state}")

    def handle_quotes(message):
        nonlocal ema_state

        print(json.dumps(message, indent=2))

        for entry in message.get("content", []):
            symbol = entry.get("key")
            if symbol not in WATCHLIST:
                continue

            last_price = entry.get("LAST_PRICE")
            last_size = entry.get("LAST_SIZE", 0)
            ts_ms = entry.get("TRADE_TIME_MILLIS") or entry.get("QUOTE_TIME_MILLIS")
            if last_price is None or ts_ms is None:
                continue

            interval_start = floor_to_interval(ts_ms, BAR_MINUTES)
            state = bar_state.get(symbol)

            if state is None or state["interval_start"] != interval_start:
                # Close previous bar if exists
                if state is not None:
                    closed_bar = {
                        "datetime": int(state["interval_start"].timestamp() * 1000),
                        "open": state["open"],
                        "high": state["high"],
                        "low": state["low"],
                        "close": state["close"],
                        "volume": state["volume"],
                    }
                    hist = history[symbol]
                    hist.append(closed_bar)
                    if len(hist) > MAX_BARS_STORED:
                        del hist[0]

                    print(f"[INFO] {symbol} closed bars in history: {len(hist)}")

                    # Update EMA200 state with this bar's close and persist
                    ema_val = update_ema_for_symbol(ema_state, symbol, closed_bar["close"])
                    save_ema_state(ema_state)

                    interval_dt = datetime.fromtimestamp(
                        closed_bar["datetime"] / 1000, tz=timezone.utc
                    )
                    if not is_trading_time(interval_dt):
                        print("[SKIP] Outside trading window, no signal for this bar.")
                    else:
                        # ===== SIMPLE SCALPER LOGIC ON CLOSED BAR (PAPER OPTIONS) =====

                        if daily_realized_pnl <= DAILY_LOSS_LIMIT:
                            print(
                                f"[RISK] Daily loss limit hit (PnL={daily_realized_pnl:.2f}), "
                                f"no new signals for {symbol} this bar."
                            )
                        else:
                            curr = hist[-1]
                            bar_range = curr["high"] - curr["low"]
                            if bar_range <= 0:
                                bar_range = 0.01

                            # 3-bar average range filter
                            if len(hist) >= RANGE_WINDOW:
                                last_ranges = [
                                    b["high"] - b["low"] for b in hist[-RANGE_WINDOW:]
                                ]
                                avg_range = sum(last_ranges) / RANGE_WINDOW
                            else:
                                avg_range = 0.0

                            if avg_range < MIN_RANGE_AVG:
                                print(
                                    f"[SKIP] {symbol} avg_range={avg_range:.2f} "
                                    f"< {MIN_RANGE_AVG:.2f}, volatility too low."
                                )
                            else:
                                # EMA200 trend filter (state‑based)
                                if ema_val is None:
                                    print(
                                        f"[SKIP] {symbol} not enough data for EMA{EMA_LENGTH} "
                                        f"(bars_seen="
                                        f"{ema_state.get(symbol, {}).get('bars_seen', 0)}), "
                                        f"no trend-based signal yet."
                                    )
                                else:
                                    # Volume thresholds
                                    if symbol in ("IBIT", "INTC"):
                                        vol_threshold = 70_000
                                    else:
                                        vol_threshold = 100_000

                                    if curr["volume"] < vol_threshold:
                                        print(
                                            f"[SKIP] {symbol} volume too low "
                                            f"({curr['volume']} < {vol_threshold}), no signal."
                                        )
                                    else:
                                        # Direction logic
                                        if curr["close"] > curr["open"]:
                                            direction = "BUY_CALL"
                                        elif curr["close"] < curr["open"]:
                                            direction = "BUY_PUT"
                                        else:
                                            direction = None

                                        if direction is None:
                                            print(f"[SKIP] {symbol} bar is flat, no signal.")
                                        else:
                                            # Trend alignment vs EMA200
                                            if direction == "BUY_CALL" and curr["close"] <= ema_val:
                                                print(
                                                    f"[SKIP] {symbol} CALL blocked by EMA{EMA_LENGTH} "
                                                    f"(close {curr['close']:.2f} <= ema {ema_val:.2f})."
                                                )
                                            elif direction == "BUY_PUT" and curr["close"] >= ema_val:
                                                print(
                                                    f"[SKIP] {symbol} PUT blocked by EMA{EMA_LENGTH} "
                                                    f"(close {curr['close']:.2f} >= ema {ema_val:.2f})."
                                                )
                                            else:
                                                max_loss_per_trade = capital * RISK_PER_TRADE_FRACTION
                                                max_premium_per_contract = max_loss_per_trade / 100.0

                                                contract = pick_deep_itm_contract(
                                                    client=client,
                                                    symbol=symbol,
                                                    direction=direction,
                                                    max_premium_per_contract=max_premium_per_contract,
                                                )

                                                with open(LOG_FILE, "a", newline="") as f:
                                                    writer = csv.writer(f)
                                                    writer.writerow([
                                                        curr["datetime"],
                                                        symbol,
                                                        direction,
                                                        f"{capital:.2f}",
                                                        f"{max_loss_per_trade:.2f}",
                                                        f"{max_premium_per_contract:.2f}",
                                                        f"{curr['open']:.2f}",
                                                        f"{curr['high']:.2f}",
                                                        f"{curr['low']:.2f}",
                                                        f"{curr['close']:.2f}",
                                                        curr["volume"],
                                                    ])

                                                print(
                                                    f"[SIGNAL] {symbol} {direction} "
                                                    f"capital={capital:.2f} "
                                                    f"max_loss_per_trade={max_loss_per_trade:.2f} "
                                                    f"max_premium_per_contract={max_premium_per_contract:.2f} "
                                                    f"(bar_range={bar_range:.2f}, "
                                                    f"avg_range={avg_range:.2f}, "
                                                    f"ema{EMA_LENGTH}={ema_val:.2f}) "
                                                    f"contract={contract}"
                                                )

                                                execution_on_signal(
                                                    symbol=symbol,
                                                    direction=direction,
                                                    max_premium_per_contract=max_premium_per_contract,
                                                    bar_info=curr,
                                                    contract=contract,
                                                )
                        # ===== END SIMPLE SCALPER LOGIC =====

                    interval_dt_local = interval_dt.astimezone()
                    t_str = interval_dt_local.strftime("%Y-%m-%d %H:%M")
                    print(
                        f"[BAR CLOSED] {symbol} {t_str} "
                        f"O:{closed_bar['open']:.2f} H:{closed_bar['high']:.2f} "
                        f"L:{closed_bar['low']:.2f} C:{closed_bar['close']:.2f} "
                        f"V:{closed_bar['volume']}"
                    )

                # Start new bar
                bar_state[symbol] = {
                    "interval_start": interval_start,
                    "open": float(last_price),
                    "high": float(last_price),
                    "low": float(last_price),
                    "close": float(last_price),
                    "volume": int(last_size),
                }
            else:
                # Update existing bar
                state["high"] = max(state["high"], float(last_price))
                state["low"] = min(state["low"], float(last_price))
                state["close"] = float(last_price)
                state["volume"] += int(last_size)

    stream_client.add_level_one_equity_handler(handle_quotes)

    await stream_client.login()
    await stream_client.level_one_equity_subs(symbols=WATCHLIST)

    print(f"Starting CLEAN LIVE scalper for: {', '.join(WATCHLIST)} (bar = {BAR_MINUTES} min, EMA={EMA_LENGTH})")

    while True:
        await stream_client.handle_message()

if __name__ == "__main__":
    asyncio.run(main())

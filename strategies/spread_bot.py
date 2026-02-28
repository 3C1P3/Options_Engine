"""
Spread Bot - Core + Lotto Call Vertical Skeleton using schwab-py.

Flow:
  - Auth to Schwab via spread-bot token file
  - Ask you for a ticker (default from config) or AUTO whitelist mode
  - Pull call option chain
  - Compute 30-day SMA and apply directional filter
  - Pick a simple bull call vertical ("core" spread)
  - Optionally pick a further OTM bull call vertical ("lotto" spread)
  - Show you the structures, estimated debits, and exit plan
  - Ask for confirmation, then place multi-leg orders
  - Log fills to fills_log.jsonl for exit scripts

Supports PAPER_MODE in config.py:
  - When PAPER_MODE = True, orders are not sent to Schwab;
    payloads are printed and logged as simulated trades.
"""

import datetime as dt
import json
from pathlib import Path
from strategies.quant_signal import quant_score

from schwab.orders import options as option_orders

from auth import create_spread_bot_client
from config import (
    ACCOUNT_ID,
    DEFAULT_TICKER,
    ACCOUNT_SIZE,
    MAX_RISK_PER_TICKER,
    CORE_MAX_DEBIT,
    CORE_GAP,
    CORE_DTE_TARGET,
    CORE_QTY,
    LOTTO_ENABLE,
    LOTTO_MAX_DEBIT,
    LOTTO_GAP,
    LOTTO_STRIKE_OFFSET,
    LOTTO_QTY,
    PAPER_MODE,
    CORE_TP_MULTIPLIER,
    CORE_SL_MULTIPLIER,
    AUTO_WHITELIST,
)

FILLS_LOG = Path("fills_log.jsonl")

# DTE window around CORE_DTE_TARGET
DTE_MIN = 2
DTE_MAX = 7

# Whitelist for AUTO rotation
AUTO_WHITELIST = [
    "NVDA",
    "IONQ",
    "AMD",
    "SMCI",
    "CPNG",
    "GOOGL",
    "MU",
    "INTC",
    "PLTR",
    "IWM",
    "IBIT",
    "SNOW",
    "DELL",
    "EXK",
    "BE",
    "CDE",
]

# -------- data: price history / SMA --------

def get_sma30(client, symbol: str) -> float | None:
    """
    Fetch daily price history and compute 30-day simple moving average of closes.
    Uses schwab-py Client.get_price_history_every_day.
    """
    try:
        resp = client.get_price_history_every_day(
            symbol=symbol,
            need_extended_hours_data=False,
            need_previous_close=False,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"DEBUG: get_sma30 -> error fetching price history for {symbol}: {e}")
        return None

    data = resp.json()
    candles = data.get("candles") or []
    if len(candles) < 30:
        print(f"DEBUG: get_sma30 -> not enough candles for {symbol}, got {len(candles)}")
        return None

    # Take the last 30 closes
    last_30 = candles[-30:]
    closes = [c.get("close") for c in last_30 if c.get("close") is not None]

    if len(closes) < 30:
        print(f"DEBUG: get_sma30 -> missing close values, have {len(closes)}")
        return None

    sma30 = sum(closes) / 30.0
    print(f"DEBUG: get_sma30 -> {symbol} 30d SMA = {sma30:.4f}")
    return sma30

def build_quant_features_from_candles(candles: list, sma30: float) -> list[float]:
    """
    Build [ret_1d, norm_price_vs_sma] from daily candles and SMA30.

    candles: list of daily OHLCV dicts, oldest -> newest.
    """
    if sma30 is None or len(candles) < 2:
        return [0.0, 0.0]

    close_t = candles[-1].get("close")
    close_tm1 = candles[-2].get("close")
    if close_t is None or close_tm1 is None:
        return [0.0, 0.0]

    # 1-day return
    ret_1d = (close_t / close_tm1) - 1.0

    # Normalized distance vs SMA30 (use close_t like in training)
    if close_t > 0:
        norm_vs_sma = (close_t - sma30) / close_t
    else:
        norm_vs_sma = 0.0

    # Clip to [-0.5, 0.5]
    norm_vs_sma = max(-0.5, min(0.5, norm_vs_sma))

    return [ret_1d, norm_vs_sma]

# -------- data: option chain --------

def get_option_chain(client, symbol: str) -> dict:
    """
    Pull a basic call option chain for a symbol.
    """
    client.set_enforce_enums(False)

    resp = client.get_option_chain(
        symbol=symbol,
        contract_type="CALL",
        strike_range="ALL",
        include_underlying_quote=True,
    )
    resp.raise_for_status()
    chain = resp.json()

    # DEBUG: high-level info about chain
    print("\n==== DEBUG: OPTION CHAIN SUMMARY ====")
    print(f"Symbol: {symbol}")
    print(f"Chain keys: {list(chain.keys())}")
    print(f"underlyingPrice: {chain.get('underlyingPrice')}")
    print("callExpDateMap present:", "callExpDateMap" in chain)
    call_map = chain.get("callExpDateMap") or {}
    print(f"Number of expirations in callExpDateMap: {len(call_map)}")
    for i, (date_str, strikes_map) in enumerate(call_map.items()):
        if i >= 3:
            break
        strikes = list(strikes_map.keys())
        print(f"  Exp {i}: {date_str}, strikes sample: {strikes[:5]}")
    print("==== END DEBUG CHAIN SUMMARY ====\n")

    return chain


# -------- strategy: pick core spread --------

def select_core_vertical(chain: dict,
                         dte_target: int,
                         max_debit: float,
                         gap: float) -> dict | None:
    """
    Pick a simple bull call vertical:
      - expiration closest to dte_target within [DTE_MIN, DTE_MAX]
      - long strike just above underlying
      - short strike = long + gap
      - estimated debit <= max_debit (in dollars per spread)
    """
    underlying = chain.get("underlyingPrice")
    if underlying is None:
        print("DEBUG: select_core_vertical -> missing underlyingPrice")
        return None

    print(f"DEBUG: select_core_vertical -> underlyingPrice = {underlying}")

    today = dt.datetime.now().date()
    call_map = chain.get("callExpDateMap", {})
    if not call_map:
        print("DEBUG: select_core_vertical -> empty callExpDateMap")
        return None

    best_exp = None
    best_dte_diff = None

    # Find expiration closest to DTE_TARGET but constrained by DTE_MIN / DTE_MAX
    print(f"DEBUG: scanning expirations for best DTE match in range "
          f"[{DTE_MIN}, {DTE_MAX}] around target {dte_target}")
    for date_str, series in call_map.items():
        date_part = date_str.split(":")[0]
        try:
            exp_date = dt.datetime.strptime(date_part, "%Y-%m-%d").date()
        except ValueError:
            print(f"DEBUG: skipping unparsable date key {date_str}")
            continue

        dte = (exp_date - today).days
        print(f"  DEBUG: exp {date_str} -> exp_date={exp_date}, dte={dte}")
        if dte < DTE_MIN or dte > DTE_MAX:
            continue

        diff = abs(dte - dte_target)
        if best_dte_diff is None or diff < best_dte_diff:
            best_dte_diff = diff
            best_exp = (date_str, series)

    if best_exp is None:
        print("DEBUG: no valid expiration in DTE window for core")
        return None

    exp_key, strikes_map = best_exp
    print(f"DEBUG: chosen expiration for core: {exp_key}, "
          f"num strikes={len(strikes_map)}")

    # Long strike just above underlying
    long_candidate = None
    print("DEBUG: searching for long strike >= underlying")
    for strike_str, contracts in sorted(
        strikes_map.items(), key=lambda kv: float(kv[0])
    ):
        strike = float(strike_str)
        if strike >= underlying:
            long_candidate = contracts[0]
            print(f"  DEBUG: found long strike {strike}")
            break

    if long_candidate is None:
        print("DEBUG: no long strike >= underlying", underlying)
        return None

    long_strike = long_candidate["strikePrice"]
    short_strike = long_strike + gap
    print(f"DEBUG: core long_strike={long_strike}, "
          f"target short_strike={short_strike}, gap={gap}")

    short_candidate = None
    for strike_str, contracts in strikes_map.items():
        strike = float(strike_str)
        if abs(strike - short_strike) < 1e-6:
            short_candidate = contracts[0]
            print(f"  DEBUG: found short strike {strike}")
            break

    if short_candidate is None:
        print("DEBUG: no short strike at", short_strike)
        return None

    # Estimate mid prices
    def mid(c: dict) -> float:
        bid = c.get("bid", 0.0) or 0.0
        ask = c.get("ask", 0.0) or 0.0
        if bid == 0.0 and ask == 0.0:
            return 0.0
        return (bid + ask) / 2.0

    long_mid = mid(long_candidate)
    short_mid = mid(short_candidate)
    est_debit = (long_mid - short_mid) * 100.0  # 100x multiplier

    print(f"DEBUG: pricing core spread -> "
          f"long_mid={long_mid}, short_mid={short_mid}, est_debit={est_debit}, "
          f"max_debit={max_debit}")

    if est_debit <= 0 or est_debit > max_debit:
        print(f"DEBUG: est_debit {est_debit} rejected "
              f"(<=0 or > max_debit {max_debit})")
        return None

    print("DEBUG: core spread accepted.")
    return {
        "long": long_candidate,
        "short": short_candidate,
        "est_debit": est_debit,
    }


# -------- strategy: pick lotto spread --------

def select_lotto_vertical(chain: dict,
                          core_long_strike: float,
                          core_short_strike: float,
                          max_debit: float,
                          gap: float,
                          strike_offset: int) -> dict | None:
    """
    Pick a lotto bull call vertical:
      - Same expiration as the core spread (bucket containing core short)
      - Long leg is 'strike_offset' strikes above the core short strike
      - Short leg = lotto_long + gap
      - Estimated debit <= max_debit
    """
    call_map = chain.get("callExpDateMap", {})
    if not call_map:
        print("DEBUG: select_lotto_vertical -> empty callExpDateMap")
        return None

    # Find which expiration bucket contains the core short strike
    target_date_key = None

    print("DEBUG: select_lotto_vertical -> locating expiration for core short")
    for date_str, strikes_map in call_map.items():
        for strike_str in strikes_map.keys():
            if abs(float(strike_str) - core_short_strike) < 1e-6:
                target_date_key = date_str
                break
        if target_date_key is not None:
            break

    if target_date_key is None:
        print("DEBUG: lotto -> could not find expiration containing core short strike")
        return None

    strikes_map = call_map[target_date_key]
    print(f"DEBUG: lotto -> using expiration {target_date_key}, "
          f"num strikes={len(strikes_map)}")

    # Build a sorted list of available strikes for that expiration
    strikes_sorted = sorted(float(s) for s in strikes_map.keys())

    # Find index of the core short strike in that list
    try:
        core_index = strikes_sorted.index(core_short_strike)
    except ValueError:
        print("DEBUG: lotto -> core_short_strike not found in strikes_sorted")
        return None

    lotto_long_index = core_index + strike_offset
    if lotto_long_index >= len(strikes_sorted):
        print("DEBUG: lotto -> lotto_long_index out of range")
        return None

    lotto_long_strike = strikes_sorted[lotto_long_index]
    lotto_short_strike = lotto_long_strike + gap
    print(f"DEBUG: lotto -> lotto_long_strike={lotto_long_strike}, "
          f"target lotto_short_strike={lotto_short_strike}")

    # Fetch lotto contracts
    def get_contract_at_strike(strike_val: float):
        for strike_str, contracts in strikes_map.items():
            if abs(float(strike_str) - strike_val) < 1e-6:
                return contracts[0]
        return None

    long_contract = get_contract_at_strike(lotto_long_strike)
    short_contract = get_contract_at_strike(lotto_short_strike)

    if long_contract is None or short_contract is None:
        print("DEBUG: lotto -> missing long or short contract")
        return None

    def mid(c: dict) -> float:
        bid = c.get("bid", 0.0) or 0.0
        ask = c.get("ask", 0.0) or 0.0
        if bid == 0.0 and ask == 0.0:
            return 0.0
        return (bid + ask) / 2.0

    long_mid = mid(long_contract)
    short_mid = mid(short_contract)
    est_debit = (long_mid - short_mid) * 100.0

    print(f"DEBUG: pricing lotto spread -> "
          f"long_mid={long_mid}, short_mid={short_mid}, est_debit={est_debit}, "
          f"max_debit={max_debit}")

    if est_debit <= 0 or est_debit > max_debit:
        print(f"DEBUG: lotto est_debit {est_debit} rejected "
              f"(<=0 or > max_debit {max_debit})")
        return None

    print("DEBUG: lotto spread accepted.")
    return {
        "long": long_contract,
        "short": short_contract,
        "est_debit": est_debit,
    }


# -------- logging helpers --------

def log_fill(trade_type: str, symbol: str, spread: dict, quantity: int):
    """
    Append a core/lotto fill record to fills_log.jsonl.
    Now also logs option symbols, so exit_watcher can manage exits.
    """
    long_leg = spread["long"]
    short_leg = spread["short"]

    record = {
        "type": trade_type,
        "symbol": symbol,
        "exp": long_leg["expirationDate"].split("T")[0],
        "long_strike": float(long_leg["strikePrice"]),
        "short_strike": float(short_leg["strikePrice"]),
        "qty": int(quantity),
        "entry_debit": float(spread["est_debit"]),
        "long_symbol": long_leg["symbol"],
        "short_symbol": short_leg["symbol"],
        "timestamp": dt.datetime.now().isoformat(),
    }

    print(f"DEBUG: writing fill to {FILLS_LOG.resolve()}")
    with FILLS_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")

# -------- orders: build & place --------

def build_quant_features(symbol: str, candles: list, sma30: float, price: float):
    """
    Build [ret_1d, norm_price_vs_sma] for the quantum filter.

    candles: list of OHLCV dicts sorted oldest -> newest.
    price: current underlying price (e.g. from option chain underlyingPrice).
    """
    if len(candles) < 2 or sma30 is None:
        # Fallback: neutral features if we don't have enough history
        return [0.0, 0.0]

    # Assume last two candles are daily closes
    close_t = candles[-1]["close"]
    close_tm1 = candles[-2]["close"]
    ret_1d = (close_t / close_tm1) - 1.0

    norm_vs_sma = (price - sma30) / price if price > 0 else 0.0
    # Clip to keep angles in a sane range
    norm_vs_sma = max(-0.5, min(0.5, norm_vs_sma))

    return [ret_1d, norm_vs_sma]

def build_core_order(spread: dict, quantity: int):
    """
    Build a pre-filled bull call vertical open order for the core.
    """
    long_leg = spread["long"]
    short_leg = spread["short"]

    long_symbol = long_leg["symbol"]
    short_symbol = short_leg["symbol"]

    limit_debit = round(spread["est_debit"] / 100.0, 2)
    price_str = f"{limit_debit:.2f}"

    order_builder = option_orders.bull_call_vertical_open(
        long_call_symbol=long_symbol,
        short_call_symbol=short_symbol,
        quantity=quantity,
        net_debit=price_str,
    )

    return order_builder.build()

def build_lotto_order(spread: dict, quantity: int):
    """
    Build a pre-filled lotto bull call vertical open order.
    """
    long_leg = spread["long"]
    short_leg = spread["short"]

    long_symbol = long_leg["symbol"]
    short_symbol = short_leg["symbol"]

    limit_debit = round(spread["est_debit"] / 100.0, 2)
    price_str = f"{limit_debit:.2f}"

    order_builder = option_orders.bull_call_vertical_open(
        long_call_symbol=long_symbol,
        short_call_symbol=short_symbol,
        quantity=quantity,
        net_debit=price_str,
    )

    return order_builder.build()


def place_order(client, order) -> dict:
    """
    Place an order, or simulate it if PAPER_MODE is True.
    """
    if PAPER_MODE:
        print("\n[PAPER_MODE] Would place order payload:")
        print(order)
        return {"status": "PAPER_SIMULATED", "order": order}

    resp = client.place_order(
        account_id=ACCOUNT_ID,
        order=order,
    )
    resp.raise_for_status()
    return resp.json()

# -------- main helpers --------

def run_for_symbol(client, symbol: str) -> bool:
    """
    Run the full pipeline for a single symbol.
    Returns True if a valid core was found (and risk check passed), else False.
    """
    print(f"\n==== DEBUG: START RUN FOR {symbol} ====")

    current_price = None

    # 0) Pull a preview option chain so we have underlyingPrice and a chain handy
    chain_preview = get_option_chain(client, symbol)
    underlying_price = chain_preview.get("underlyingPrice")
    if underlying_price is None:
        print("DEBUG: missing underlyingPrice in preview chain, aborting symbol.")
        print("==== DEBUG: END RUN (NO UNDERLYING) ====\n")
        return False

    current_price = float(underlying_price)

    # 1) Directional filter: 30-day SMA
    sma30 = get_sma30(client, symbol)

    if sma30 is not None:
        print(f"DEBUG: directional filter -> price={current_price:.2f}, SMA30={sma30:.2f}")
        if current_price <= sma30:
            print("DEBUG: price <= SMA30, skipping bullish spread for this symbol.")
            print("==== DEBUG: END RUN (NO CORE) ====\n")
            return False
        else:
            print("DEBUG: directional filter passed (price > SMA30).")
    else:
        print(f"DEBUG: get_sma30 -> not enough candles for {symbol}")
        print("DEBUG: SMA30 unavailable, skipping this symbol.")
        print("==== DEBUG: END RUN (NO CORE) ====\n")
        return False
    # Quantum directional filter (extra gate)
    try:
        candles_resp = client.get_price_history_every_day(
            symbol=symbol,
            need_extended_hours_data=False,
            need_previous_close=False,
        )
        candles_resp.raise_for_status()
        candles_data = candles_resp.json()
        candles = candles_data.get("candles") or []
    except Exception as e:
        print(f"DEBUG: error fetching candles for quantum filter on {symbol}: {e}")
        candles = []

    features = build_quant_features_from_candles(candles, sma30)
    q_score = quant_score(features)
    print(f"DEBUG: quantum filter -> features={features}, score={q_score:.3f}")

    QUANT_THRESHOLD = 0.6
    if q_score < QUANT_THRESHOLD:
        print(f"DEBUG: quantum score {q_score:.3f} < {QUANT_THRESHOLD}, skipping {symbol}.")
        print("==== DEBUG: END RUN (NO CORE) ====\n")
        return False

    # 1b) Quantum directional filter
    try:
        candles_resp = client.get_price_history_every_day(
            symbol=symbol,
            need_extended_hours_data=False,
            need_previous_close=False,
        )
        candles_resp.raise_for_status()
        candles_data = candles_resp.json()
        candles = candles_data.get("candles") or []
    except Exception as e:
        print(f"DEBUG: error fetching candles for quantum filter on {symbol}: {e}")
        candles = []

    features = build_quant_features(symbol, candles, sma30, current_price)
    q_score = quant_score(features)
    print(f"DEBUG: quantum filter -> features={features}, score={q_score:.3f}")

    QUANT_THRESHOLD = 0.5
    if q_score < QUANT_THRESHOLD:
        print(f"DEBUG: quantum score {q_score:.3f} < {QUANT_THRESHOLD}, skipping {symbol}.")
        print("==== DEBUG: END RUN (NO CORE) ====\n")
        return False

    # 2) Select core vertical using the preview chain
    chain = chain_preview

    spread = select_core_vertical(
        chain,
        dte_target=CORE_DTE_TARGET,
        max_debit=CORE_MAX_DEBIT,
        gap=CORE_GAP,
    )

    if spread is None:
        print("No acceptable core vertical found under current constraints.")
        print("==== DEBUG: END RUN (NO CORE) ====\n")
        return False

    print("Selected CORE spread:")
    print(f"  Long:  {spread['long']['symbol']} @ {spread['long']['strikePrice']}")
    print(f"  Short: {spread['short']['symbol']} @ {spread['short']['strikePrice']}")
    print(f"  Est. debit per spread: ${spread['est_debit']:.2f}")

    # 2b) Optionally select lotto vertical
    lotto_spread = None
    if LOTTO_ENABLE:
        core_long = spread["long"]["strikePrice"]
        core_short = spread["short"]["strikePrice"]

        lotto_spread = select_lotto_vertical(
            chain,
            core_long_strike=core_long,
            core_short_strike=core_short,
            max_debit=LOTTO_MAX_DEBIT,
            gap=LOTTO_GAP,
            strike_offset=LOTTO_STRIKE_OFFSET,
        )

        if lotto_spread is not None:
            print("\nSelected LOTTO spread:")
            print(f"  Long:  {lotto_spread['long']['symbol']} @ {lotto_spread['long']['strikePrice']}")
            print(f"  Short: {lotto_spread['short']['symbol']} @ {lotto_spread['short']['strikePrice']}")
            print(f"  Est. debit per spread: ${lotto_spread['est_debit']:.2f}")
        else:
            print("\nNo acceptable lotto spread found under lotto constraints.")

    # 2c) Check per-ticker risk cap
    planned_core_risk = spread["est_debit"]
    planned_lotto_risk = lotto_spread["est_debit"] if lotto_spread is not None else 0.0
    total_planned_risk = planned_core_risk + planned_lotto_risk

    print(f"\nPlanned risk for {symbol}: "
          f"CORE ${planned_core_risk:.2f} + LOTTO ${planned_lotto_risk:.2f} "
          f"= ${total_planned_risk:.2f} (cap ${MAX_RISK_PER_TICKER:.2f})")

    # Directional context in final summary
    if sma30 is not None and current_price is not None:
        print(f"Directional context: price ${current_price:.2f} vs 30d SMA ${sma30:.2f}")

    # Exit-plan cheat sheet, based on debit
    core_debit_per_contract = planned_core_risk / 100.0
    tp_price = core_debit_per_contract * CORE_TP_MULTIPLIER
    sl_price = core_debit_per_contract * CORE_SL_MULTIPLIER

    print(f"\nEXIT PLAN for {symbol} CORE:")
    print(f"  Debit paid per spread: ${core_debit_per_contract:.2f}")
    print(f"  Take-profit trigger:   spread price ≈ ${tp_price:.2f}")
    print(f"  Stop-loss trigger:     spread price ≈ ${sl_price:.2f}")

    if total_planned_risk > MAX_RISK_PER_TICKER:
        print("Planned risk exceeds per-ticker cap. No orders will be sent.")
        print("==== DEBUG: END RUN (RISK TOO HIGH) ====\n")
        return False

    # 3) Confirm and send orders
    send_core = input("\nSend CORE order to Schwab? (y/n): ").strip().lower()
    if send_core == "y":
        core_order = build_core_order(spread, CORE_QTY)
        core_resp = place_order(client, core_order)
        print("Core order placed. Schwab response:")
        print(core_resp)
        log_fill("core", symbol, spread, CORE_QTY)
    else:
        print("Core order canceled.")

    if lotto_spread is not None:
        send_lotto = input("Send LOTTO order to Schwab? (y/n): ").strip().lower()
        if send_lotto == "y":
            lotto_order = build_lotto_order(lotto_spread, LOTTO_QTY)
            lotto_resp = place_order(client, lotto_order)
            print("Lotto order placed. Schwab response:")
            print(lotto_resp)
            log_fill("lotto", symbol, lotto_spread, LOTTO_QTY)
        else:
            print("Lotto order canceled.")

    print("==== DEBUG: END RUN (DONE) ====\n")
    return True

# -------- main flow --------

def main():
    client = create_spread_bot_client()

    user_input = input(
        f"Ticker to trade (default {DEFAULT_TICKER}, or type AUTO for whitelist): "
    ).strip().upper()

    if not user_input:
        symbol = DEFAULT_TICKER
        run_for_symbol(client, symbol)
        return

    if user_input == "AUTO":
        print(f"\nAUTO mode: rotating through whitelist {AUTO_WHITELIST}")
        for sym in AUTO_WHITELIST:
            print(f"\n=== Trying {sym} ===")
            ok = run_for_symbol(client, sym)
            if ok:
                print(f"\nAUTO mode: selected {sym} as first acceptable setup.")
                return
        print("\nAUTO mode: no acceptable core vertical found for any whitelist symbol.")
        return

    # Otherwise treat input as explicit symbol
    symbol = user_input
    run_for_symbol(client, symbol)


if __name__ == "__main__":
    main()

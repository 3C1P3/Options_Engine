# config.py

"""
Configuration for the Spread Bot.

Fill in:
  - APP_KEY, APP_SECRET, REDIRECT_URI from your Schwab developer app
  - ACCOUNT_ID as your Schwab trading account (hash or id per docs)
  - TOKEN_PATH is this bot's OWN token file, separate from your scalper bot
"""

# ---------- Schwab API credentials ----------

# Your Schwab application's app key (a.k.a. client_id)
APP_KEY = "hVlTy40IKJ5YFxjuIdpXw5W4FeBFAq4KJ262rTrF2lP2p5mX"

# Your Schwab application's secret
APP_SECRET = "TidSTWaPXo1Z0uaE7ijQEK0v3qJlZ3ot7XhbQuL00IF47rxnWCAjZjC5Mw37Qmft"

# Must EXACTLY match the callback / redirect URL configured
# for your Schwab app (including scheme, host, port, and trailing slash).
# Example: "https://127.0.0.1:8000/callback"
REDIRECT_URI = "https://127.0.0.1:8000/callback"


# ---------- Account / token settings ----------

# Your Schwab trading account identifier (raw or hashed, per your setup)
ACCOUNT_ID = "64482506"

# This bot's token file (SEPARATE from your scalper bot's token file)
TOKEN_PATH = "spreads_token.json"


# ---------- Mode ----------

# When True, do not send real orders; just print and log as paper trades.
PAPER_MODE = True


# ---------- Core spread defaults ----------

# Default ticker to trade if you just press Enter at the prompt
DEFAULT_TICKER = "NVDA"

# Account size (for risk-based checks)
ACCOUNT_SIZE = 500.0

# Max total risk per ticker (core + lotto), 30% of account
MAX_RISK_PER_TICKER = 150.0   # 150.0

# Core spread maximum debit per spread (15–20% of account)
# For 500 account, use about $75–100. Let's pick 90 as a cap.
CORE_MAX_DEBIT = 120.0

# Strike gap between long and short call (e.g., 5.0 = 195/200, 10.0 = 195/205)
CORE_GAP = 2.0

# Target days to expiration for the core spread
CORE_DTE_TARGET = 3

# How many core spreads to buy per trade
CORE_QTY = 1


# ---------- Lotto spread defaults ----------

# Turn lotto spread logic on/off
LOTTO_ENABLE = True

# Lotto max debit per spread (5–10% of account; pick 35 as middle)
LOTTO_MAX_DEBIT = 40.0

# Lotto spread width (usually same as core gap)
LOTTO_GAP = CORE_GAP

# How many strikes beyond the core short to place the lotto long
LOTTO_STRIKE_OFFSET = 2

# How many lotto spreads to buy per trade
LOTTO_QTY = 1

# Exit planning (as fraction of debit paid)
CORE_TP_MULTIPLIER = 1.6   # +60% profit target (sell at 1.6 * debit)
CORE_SL_MULTIPLIER = 0.3   # -30% max loss (sell at 0.3 * debit)

# Existing config values above...
# APP_KEY, APP_SECRET, ACCOUNT_ID, CORE_MAX_DEBIT, etc.

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
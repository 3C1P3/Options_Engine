# auth.py

"""
Authentication helper for the Spread Bot using schwab-py.

Make sure you have:
  pip install schwab-py

And that config.py contains:
  APP_KEY, APP_SECRET, REDIRECT_URI, TOKEN_PATH
"""

from schwab.auth import easy_client
from config import APP_KEY, APP_SECRET, REDIRECT_URI, TOKEN_PATH


def create_spread_bot_client():
    """
    Create and return a schwab-py Client for this bot.

    First run:
      - If TOKEN_PATH does not exist, opens a browser to complete Schwab login,
        then writes a new token file.

    Later runs:
      - Reuses and refreshes the token in TOKEN_PATH automatically.
    """
    client = easy_client(
        api_key=APP_KEY,
        app_secret=APP_SECRET,
        callback_url=REDIRECT_URI,
        token_path=TOKEN_PATH,
    )
    return client

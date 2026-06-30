"""
ICT Model 12 — Sunday Prep Analyzer
=====================================
Bread & Butter Scalp — Daily Bias Engine

Powered by mt5-connector — pip install mt5-connector
https://github.com/aulekator/mt5-connector

Based on ICT Charter Membership Content (2018–2024)
For educational purposes only. Not financial advice.

Steps performed:
  1. Connect to MetaTrader 5 via mt5-connector
  2. Fetch daily OHLC bars from your live MT5 terminal
  3. Calculate 20-Day IPDA Range (Premium vs Discount)
  4. Extract Previous Day High/Low (Draw on Liquidity)
  5. Assess Weekly Bias (Higher Highs/Lows structure)
  6. Score confidence and print full trade setup summary

Usage:
  python model12_analyzer.py
  python model12_analyzer.py --symbol XAUUSDm --days 20
  python model12_analyzer.py --symbol EURUSDm --days 40
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")

# ─────────────────────────────────────────────────────────────────────────────
# Check dependencies
# ─────────────────────────────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌  MetaTrader5 not installed.")
    print("    Run: pip install mt5-connector")
    print("    Note: Requires Windows + MT5 terminal open and logged in.")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("❌  pandas/numpy not installed. Run: pip install pandas numpy")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # .env is optional — credentials can be passed as args


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"
ORANGE = "\033[38;5;208m"


# ─────────────────────────────────────────────────────────────────────────────
# MT5 CONNECTION
# ─────────────────────────────────────────────────────────────────────────────
def connect_mt5(account: int, password: str, server: str) -> bool:
    """
    Initialize and log in to MetaTrader 5.
    Uses mt5-connector's MetaTrader5 dependency under the hood.
    """
    if not mt5.initialize():
        print(f"{RED}❌  MT5 initialize() failed — is the terminal open?{RESET}")
        print(f"{DIM}    Error: {mt5.last_error()}{RESET}")
        return False

    if not mt5.login(account, password=password, server=server):
        print(f"{RED}❌  MT5 login() failed — check account/password/server{RESET}")
        print(f"{DIM}    Error: {mt5.last_error()}{RESET}")
        mt5.shutdown()
        return False

    info = mt5.account_info()
    print(f"{GREEN}  ✔  Connected to MT5{RESET}")
    print(f"{DIM}     Account : {info.login}{RESET}")
    print(f"{DIM}     Name    : {info.name}{RESET}")
    print(f"{DIM}     Server  : {info.server}{RESET}")
    print(f"{DIM}     Balance : {info.balance} {info.currency}{RESET}")
    return True


def disconnect_mt5():
    mt5.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
def fetch_daily_bars(symbol: str, count: int = 120) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars directly from MT5 terminal.
    Uses TIMEFRAME_D1 — no third-party data provider needed.
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, count)

    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        raise ValueError(
            f"No daily data returned for '{symbol}'. "
            f"MT5 error: {error}. "
            f"Check symbol name matches your broker's Market Watch exactly."
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(columns={
        "open":      "Open",
        "high":      "High",
        "low":       "Low",
        "close":     "Close",
        "tick_volume": "Volume",
    })
    # Drop today's incomplete bar — use only completed days
    df = df.iloc[:-1]
    return df


def fetch_current_price(symbol: str) -> float:
    """Get the current live bid price from MT5."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise ValueError(f"Cannot get tick for '{symbol}'. Is the symbol in Market Watch?")
    return float(tick.bid)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — 20-DAY IPDA RANGE
# ─────────────────────────────────────────────────────────────────────────────
def calculate_ipda_range(df: pd.DataFrame, current_price: float, ipda_days: int = 20) -> dict:
    """
    Count back N completed trading days.
    Find highest high and lowest low = dealing range.
    Determine Premium (above 50%) or Discount (below 50%).
    """
    window     = df.tail(ipda_days)
    ipda_high  = float(window["High"].max())
    ipda_low   = float(window["Low"].min())
    ipda_mid   = (ipda_high + ipda_low) / 2.0
    ipda_range = ipda_high - ipda_low

    price_position = ((current_price - ipda_low) / ipda_range * 100) if ipda_range > 0 else 50.0
    is_premium     = current_price > ipda_mid

    return {
        "ipda_high":      round(ipda_high,      3),
        "ipda_low":       round(ipda_low,       3),
        "ipda_mid":       round(ipda_mid,       3),
        "ipda_range":     round(ipda_range,     3),
        "price_position": round(price_position, 1),
        "is_premium":     is_premium,
        "zone":           "PREMIUM" if is_premium else "DISCOUNT",
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — PREVIOUS DAY HIGH / LOW
# ─────────────────────────────────────────────────────────────────────────────
def get_pdh_pdl(df: pd.DataFrame, current_price: float) -> dict:
    """
    Last completed trading day = PDH / PDL.
    Sunday prep: last completed day = Friday.
    These are the Draw on Liquidity targets for the coming session.
    """
    prev_day   = df.iloc[-1]
    pdh        = round(float(prev_day["High"]),  3)
    pdl        = round(float(prev_day["Low"]),   3)
    prev_date  = df.index[-1].strftime("%A %d %b %Y")

    dist_to_pdh = round(abs(pdh - current_price), 3)
    dist_to_pdl = round(abs(current_price - pdl),  3)

    return {
        "pdh":         pdh,
        "pdl":         pdl,
        "prev_date":   prev_date,
        "dist_to_pdh": dist_to_pdh,
        "dist_to_pdl": dist_to_pdl,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — WEEKLY BIAS
# ─────────────────────────────────────────────────────────────────────────────
def get_weekly_bias(df: pd.DataFrame) -> dict:
    """
    Resample completed daily bars into weekly OHLC.
    Compare last two COMPLETED weeks:
      Higher Highs + Higher Lows = Bullish institutional order flow
      Lower Highs  + Lower Lows  = Bearish institutional order flow
    """
    weekly_high = df["High"].resample("W").max().dropna()
    weekly_low  = df["Low"].resample("W").min().dropna()

    if len(weekly_high) < 2:
        return {
            "bias":       "NEUTRAL",
            "reason":     "Insufficient weekly data",
            "curr_high":  0.0,
            "curr_low":   0.0,
            "prev_high":  0.0,
            "prev_low":   0.0,
        }

    curr_high = float(weekly_high.iloc[-1])
    curr_low  = float(weekly_low.iloc[-1])
    prev_high = float(weekly_high.iloc[-2])
    prev_low  = float(weekly_low.iloc[-2])

    higher_high = curr_high > prev_high
    higher_low  = curr_low  > prev_low
    lower_high  = curr_high < prev_high
    lower_low   = curr_low  < prev_low

    if higher_high and higher_low:
        bias, reason = "BULLISH", "Higher High + Higher Low ✔"
    elif lower_high and lower_low:
        bias, reason = "BEARISH", "Lower High + Lower Low ✔"
    elif higher_high and lower_low:
        bias, reason = "MIXED",   "Higher High but Lower Low (expanding range)"
    else:
        bias, reason = "MIXED",   "Lower High but Higher Low (contracting range)"

    return {
        "bias":      bias,
        "reason":    reason,
        "curr_high": round(curr_high, 3),
        "curr_low":  round(curr_low,  3),
        "prev_high": round(prev_high, 3),
        "prev_low":  round(prev_low,  3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — EQUAL HIGHS / EQUAL LOWS (Untouched Liquidity)
# ─────────────────────────────────────────────────────────────────────────────
def find_equal_levels(df: pd.DataFrame, current_price: float,
                      tolerance_pct: float = 0.001, lookback: int = 20) -> dict:
    """
    Scan the last N days for equal highs (buy stops above) and
    equal lows (sell stops below).
    Equal = within tolerance_pct of each other.
    Levels above current price = untouched buy side liquidity.
    Levels below current price = untouched sell side liquidity.
    """
    window = df.tail(lookback)
    highs  = window["High"].values
    lows   = window["Low"].values

    # Find clusters of equal highs above price
    equal_highs = []
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            if abs(highs[i] - highs[j]) / highs[i] < tolerance_pct:
                level = round((highs[i] + highs[j]) / 2, 3)
                if level > current_price:
                    equal_highs.append(level)

    # Find clusters of equal lows below price
    equal_lows = []
    for i in range(len(lows)):
        for j in range(i + 1, len(lows)):
            if abs(lows[i] - lows[j]) / lows[i] < tolerance_pct:
                level = round((lows[i] + lows[j]) / 2, 3)
                if level < current_price:
                    equal_lows.append(level)

    # Nearest untouched pools
    nearest_eqh = min(equal_highs, key=lambda x: abs(x - current_price)) if equal_highs else None
    nearest_eql = min(equal_lows,  key=lambda x: abs(x - current_price)) if equal_lows  else None

    return {
        "equal_highs":  sorted(set(equal_highs)),
        "equal_lows":   sorted(set(equal_lows), reverse=True),
        "nearest_eqh":  nearest_eqh,
        "nearest_eql":  nearest_eql,
        "has_bsl":      len(equal_highs) > 0,
        "has_ssl":      len(equal_lows)  > 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — FINAL BIAS + CONFIDENCE SCORE
# ─────────────────────────────────────────────────────────────────────────────
def calculate_bias(ipda: dict, pdh_pdl: dict,
                   weekly: dict, liquidity: dict) -> dict:
    """
    Score 4 conditions for LONG and SHORT.
    2/4 minimum to produce a directional bias.
    Confidence = conditions met / 4.

    LONG conditions:
      1. Price in DISCOUNT (below IPDA 50%)
      2. Weekly structure BULLISH (HH + HL)
      3. PDH is the closer untouched pool
      4. Equal highs (BSL) exist above price

    SHORT conditions:
      1. Price in PREMIUM (above IPDA 50%)
      2. Weekly structure BEARISH (LH + LL)
      3. PDL is the closer untouched pool
      4. Equal lows (SSL) exist below price
    """
    long_conditions  = []
    short_conditions = []

    # Condition 1 — IPDA Position
    if not ipda["is_premium"]:
        long_conditions.append(f"Price in DISCOUNT ({ipda['price_position']}% of IPDA range)")
    else:
        short_conditions.append(f"Price in PREMIUM ({ipda['price_position']}% of IPDA range)")

    # Condition 2 — Weekly Bias
    if weekly["bias"] == "BULLISH":
        long_conditions.append(f"Weekly BULLISH — {weekly['reason']}")
    elif weekly["bias"] == "BEARISH":
        short_conditions.append(f"Weekly BEARISH — {weekly['reason']}")

    # Condition 3 — Nearest liquidity pool
    if pdh_pdl["dist_to_pdh"] < pdh_pdl["dist_to_pdl"]:
        long_conditions.append(
            f"PDH closer ({pdh_pdl['dist_to_pdh']:.3f}) — buy side pool above"
        )
    else:
        short_conditions.append(
            f"PDL closer ({pdh_pdl['dist_to_pdl']:.3f}) — sell side pool below"
        )

    # Condition 4 — Equal highs/lows (untouched liquidity clusters)
    if liquidity["has_bsl"]:
        long_conditions.append(
            f"Equal highs (BSL) detected above @ {liquidity['nearest_eqh']}"
        )
    if liquidity["has_ssl"]:
        short_conditions.append(
            f"Equal lows (SSL) detected below @ {liquidity['nearest_eql']}"
        )

    total       = 4
    long_score  = len(long_conditions)
    short_score = len(short_conditions)

    if long_score > short_score and long_score >= 2:
        direction    = "LONG"
        target       = pdh_pdl["pdh"]
        target_label = "PDH"
        confidence   = long_score / total
        conditions   = long_conditions
    elif short_score > long_score and short_score >= 2:
        direction    = "SHORT"
        target       = pdh_pdl["pdl"]
        target_label = "PDL"
        confidence   = short_score / total
        conditions   = short_conditions
    else:
        direction    = "NEUTRAL"
        target       = None
        target_label = "N/A"
        confidence   = 0.0
        conditions   = ["Mixed signals — wait for Kill Zone confirmation"]

    return {
        "direction":    direction,
        "target":       target,
        "target_label": target_label,
        "confidence":   round(confidence, 2),
        "conditions":   conditions,
        "long_score":   long_score,
        "short_score":  short_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# KILL ZONE STATUS
# ─────────────────────────────────────────────────────────────────────────────
def kill_zone_status() -> dict:
    """Check if the current time falls inside a Model 12 Kill Zone.

    Kill zones are defined in New York local time (ICT convention).
    Using zoneinfo here means EST/EDT (daylight saving) is handled
    automatically — no manual UTC offset math, and no drift twice a year.
    """
    now_utc = datetime.now(timezone.utc)
    now_ny  = now_utc.astimezone(NY_TZ)
    hour    = now_ny.hour

    in_london = 2 <= hour < 5
    in_ny     = 7 <= hour < 10

    if in_london:
        name, status = "LONDON",   "ACTIVE 🟢"
    elif in_ny:
        name, status = "NEW YORK", "ACTIVE 🟢"
    else:
        name, status = "CLOSED",   "CLOSED 🔴"

    # Next Kill Zone (all boundaries in NY local time)
    if hour < 2:
        next_kz = f"London opens in {2 - hour}h (02:00 NY time)"
    elif hour < 5:
        next_kz = "London Kill Zone ACTIVE now"
    elif hour < 7:
        next_kz = f"New York opens in {7 - hour}h (07:00 NY time)"
    elif hour < 10:
        next_kz = "New York Kill Zone ACTIVE now"
    else:
        next_kz = f"London opens tomorrow at 02:00 NY time (in ~{26 - hour}h)"

    return {
        "active":    in_london or in_ny,
        "name":      name,
        "status":    status,
        "time_utc":  now_utc.strftime("%H:%M UTC"),
        "time_ny":   now_ny.strftime("%H:%M %Z"),
        "next_kz":   next_kz,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
def sep(char="─", width=56):
    print(DIM + char * width + RESET)


def print_report(symbol: str, current_price: float,
                 ipda: dict, pdh_pdl: dict, weekly: dict,
                 liquidity: dict, bias: dict, kz: dict,
                 ipda_days: int):

    direction = bias["direction"]
    dir_color = GREEN if direction == "LONG" else RED if direction == "SHORT" else YELLOW
    conf_pct  = int(bias["confidence"] * 100)
    conf_color = GREEN if conf_pct >= 75 else YELLOW if conf_pct >= 50 else RED

    now_str = datetime.now(timezone.utc).strftime("%A %d %b %Y  %H:%M UTC")

    print()
    sep("═")
    print(f"{BOLD}{CYAN}  ICT MODEL 12 — BREAD & BUTTER SCALP{RESET}")
    print(f"{BOLD}{CYAN}  Sunday Preparation Report{RESET}")
    sep("═")
    print(f"{DIM}  Symbol  : {symbol}{RESET}")
    print(f"{DIM}  Price   : {current_price}{RESET}")
    print(f"{DIM}  Time    : {now_str}{RESET}")
    print(f"{DIM}  IPDA    : {ipda_days}-day lookback{RESET}")
    print(f"{DIM}  Powered by mt5-connector — pip install mt5-connector{RESET}")
    sep("═")

    # ── STEP 1
    print(f"\n{BOLD}  STEP 1 — {ipda_days}-DAY IPDA RANGE{RESET}")
    sep()
    print(f"  High (Top of Range)    : {YELLOW}{ipda['ipda_high']}{RESET}")
    print(f"  Midpoint (Equilibrium) : {YELLOW}{ipda['ipda_mid']}{RESET}")
    print(f"  Low  (Bottom of Range) : {YELLOW}{ipda['ipda_low']}{RESET}")
    print(f"  Total Range            : {ipda['ipda_range']}")
    zone_color = RED if ipda["is_premium"] else GREEN
    print(f"  Price Position         : {zone_color}{ipda['price_position']}%  →  {ipda['zone']}{RESET}")
    if ipda["is_premium"]:
        print(f"\n  {RED}▶  Price ABOVE 50% — Algorithm likely targets PDL (sell side){RESET}")
    else:
        print(f"\n  {GREEN}▶  Price BELOW 50% — Algorithm likely targets PDH (buy side){RESET}")

    # ── STEP 2
    print(f"\n{BOLD}  STEP 2 — PREVIOUS DAY HIGH / LOW{RESET}")
    sep()
    print(f"  Reference Day : {pdh_pdl['prev_date']}")
    print(f"  PDH           : {GREEN}{pdh_pdl['pdh']}{RESET}  "
          f"{DIM}(dist: {pdh_pdl['dist_to_pdh']}){RESET}")
    print(f"  PDL           : {RED}{pdh_pdl['pdl']}{RESET}  "
          f"{DIM}(dist: {pdh_pdl['dist_to_pdl']}){RESET}")

    # ── STEP 3
    print(f"\n{BOLD}  STEP 3 — WEEKLY BIAS{RESET}")
    sep()
    wb_color = GREEN if weekly["bias"] == "BULLISH" else RED if weekly["bias"] == "BEARISH" else YELLOW
    print(f"  This Week  High : {weekly['curr_high']}")
    print(f"  This Week  Low  : {weekly['curr_low']}")
    print(f"  Last Week  High : {weekly['prev_high']}")
    print(f"  Last Week  Low  : {weekly['prev_low']}")
    print(f"  Weekly Bias     : {wb_color}{BOLD}{weekly['bias']}{RESET}  —  {weekly['reason']}")

    # ── STEP 4 — Liquidity
    print(f"\n{BOLD}  STEP 4 — UNTOUCHED LIQUIDITY POOLS{RESET}")
    sep()
    if liquidity["has_bsl"]:
        print(f"  Buy Side  (BSL) : {GREEN}Equal highs detected above price{RESET}")
        print(f"  Nearest BSL     : {GREEN}{liquidity['nearest_eqh']}{RESET}")
    else:
        print(f"  Buy Side  (BSL) : {DIM}No equal highs detected{RESET}")

    if liquidity["has_ssl"]:
        print(f"  Sell Side (SSL) : {RED}Equal lows detected below price{RESET}")
        print(f"  Nearest SSL     : {RED}{liquidity['nearest_eql']}{RESET}")
    else:
        print(f"  Sell Side (SSL) : {DIM}No equal lows detected{RESET}")

    # ── STEP 5 — Kill Zone
    print(f"\n{BOLD}  STEP 5 — KILL ZONE STATUS{RESET}")
    sep()
    kz_color = GREEN if kz["active"] else RED
    print(f"  Current UTC     : {kz['time_utc']}")
    print(f"  Current NY Time : {kz['time_ny']}")
    print(f"  Kill Zone       : {kz_color}{kz['name']}  {kz['status']}{RESET}")
    print(f"  Next            : {DIM}{kz['next_kz']}{RESET}")
    print(f"  {DIM}London: 02:00–05:00 NY time   |   New York: 07:00–10:00 NY time{RESET}")

    # ── FINAL SETUP
    print()
    sep("═")
    print(f"{BOLD}  FINAL MODEL 12 SETUP{RESET}")
    sep("═")
    print(f"  Direction    :  {dir_color}{BOLD}{direction}  "
          f"{'📈' if direction == 'LONG' else '📉' if direction == 'SHORT' else '⚖️'}{RESET}")
    print(f"  Entry Type   :  LIMIT at FVG + Order Block  🔥")
    if bias["target"]:
        print(f"  Target       :  {dir_color}{bias['target_label']} @ {bias['target']}{RESET}")
    print(f"  Model 12 TP  :  20 pips / $20 per trade")
    print(f"  Model 12 SL  :  20 pips / $20 stop loss")
    print(f"  Confidence   :  {conf_color}{BOLD}{conf_pct}%"
          f"  ({bias['long_score']}/4 LONG  |  {bias['short_score']}/4 SHORT){RESET}")

    print(f"\n  {BOLD}Conditions Met:{RESET}")
    for c in bias["conditions"]:
        print(f"  {dir_color}  ✔  {c}{RESET}")

    # ── CHECKLIST
    sep("═")
    print(f"\n{BOLD}  CONFIRMATION CHECKLIST{RESET}")
    sep()
    target_str = bias["target_label"] if bias["target"] else "target"
    print(f"  □  Check economic calendar for high impact news")
    print(f"  □  Open 15-min chart at London open (02:00 NY time)")
    print(f"  □  Confirm 15-min expansion heading toward {target_str}")
    print(f"  □  Wait for 5-min FVG + Order Block on retracement")
    print(f"  □  Enter ONLY inside Kill Zone (London or NY)")
    print(f"  □  Set 20-pip TP and 20-pip SL simultaneously")
    print(f"  □  Walk away — let the trade resolve itself")
    print()
    print(f"{DIM}  For educational purposes only. Not financial advice.{RESET}")
    sep("═")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ICT Model 12 — Sunday Prep Analyzer (powered by mt5-connector)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model12_analyzer.py
  python model12_analyzer.py --symbol XAUUSDm --days 20
  python model12_analyzer.py --symbol EURUSDm --days 40
  python model12_analyzer.py --symbol GBPUSDm --account 12345678 --password mypass --server Exness-MT5Trial9

Credentials (in order of priority):
  1. Command-line args  (--account, --password, --server)
  2. .env file          (MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER)
  3. Environment vars   (MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER)

Common symbols by broker:
  Exness standard : XAUUSDm, EURUSDm, GBPUSDm, BTCUSDm
  Exness raw/zero : XAUUSD,  EURUSD,  GBPUSD,  BTCUSD
  IC Markets      : XAUUSD,  EURUSD,  GBPUSD
  Pepperstone     : XAUUSD,  EURUSD,  GBPUSD
        """
    )
    parser.add_argument("--symbol",   type=str, default=None,
                        help="MT5 symbol (default: from .env MT5_SYMBOLS or XAUUSDm)")
    parser.add_argument("--days",     type=int, default=20,
                        help="IPDA lookback days (default: 20)")
    parser.add_argument("--account",  type=int, default=None,
                        help="MT5 account number (or set MT5_ACCOUNT in .env)")
    parser.add_argument("--password", type=str, default=None,
                        help="MT5 password (or set MT5_PASSWORD in .env)")
    parser.add_argument("--server",   type=str, default=None,
                        help="MT5 server name (or set MT5_SERVER in .env)")
    args = parser.parse_args()

    # Resolve credentials
    account  = args.account  or int(os.environ.get("MT5_ACCOUNT",  0))
    password = args.password or os.environ.get("MT5_PASSWORD", "")
    server   = args.server   or os.environ.get("MT5_SERVER",   "")
    symbols_env = os.environ.get("MT5_SYMBOLS", "XAUUSDm")
    symbol   = args.symbol   or symbols_env.split(",")[0]

    if not account or not password or not server:
        print(f"{RED}❌  MT5 credentials missing.{RESET}")
        print(f"    Add a .env file with:")
        print(f"      MT5_ACCOUNT=12345678")
        print(f"      MT5_PASSWORD=your_password")
        print(f"      MT5_SERVER=Exness-MT5Trial9")
        print(f"    Or pass --account --password --server as arguments.")
        sys.exit(1)

    print(f"\n{CYAN}  ICT Model 12 — Sunday Prep Analyzer{RESET}")
    print(f"{DIM}  Connecting to MT5 via mt5-connector...{RESET}\n")

    # Connect
    if not connect_mt5(account, password, server):
        sys.exit(1)

    try:
        # Fetch data
        print(f"\n{DIM}  Fetching daily bars for {symbol}...{RESET}")
        df            = fetch_daily_bars(symbol, count=120)
        current_price = fetch_current_price(symbol)
        print(f"{GREEN}  ✔  {len(df)} completed daily bars loaded{RESET}")
        print(f"{GREEN}  ✔  Current price: {current_price}{RESET}")

        # Run analysis
        ipda      = calculate_ipda_range(df, current_price, args.days)
        pdh_pdl   = get_pdh_pdl(df, current_price)
        weekly    = get_weekly_bias(df)
        liquidity = find_equal_levels(df, current_price)
        bias      = calculate_bias(ipda, pdh_pdl, weekly, liquidity)
        kz        = kill_zone_status()

        # Print report
        print_report(symbol, current_price, ipda, pdh_pdl,
                     weekly, liquidity, bias, kz, args.days)

    except ValueError as e:
        print(f"\n{RED}❌  {e}{RESET}")
        sys.exit(1)
    finally:
        disconnect_mt5()


if __name__ == "__main__":
    main()
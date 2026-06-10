# =============================================================================
# CRYPTO FUTURES TRADING BOT EVOLUTION PIPELINE
# =============================================================================
# DESCRIPTION:
#   Self-evolving trading bot pipeline that uses local LLMs (via Ollama) to
#   automatically improve trading strategies. Three bots compete and evolve
#   across cycles using DeepSeek (reasoning) + Qwen (code generation).
#
# HOW IT WORKS:
#   1. Downloads OHLCV data from Binance via CCXT (only if not already present)
#   2. Validates and arranges data into parquet files (only if not already done)
#   3. Runs baseline backtest on all 3 bots
#   4. DeepSeek-R1 generates improvement strategies (reasoning model)
#   5. Qwen2.5-Coder implements strategies as Python code (code model)
#   6. VectorBT backtests all variants on 15 crypto pairs
#   7. Best variant is saved; regression protection restores backups if needed
#   8. Loop repeats indefinitely (Ctrl+C to stop)
#
# REQUIREMENTS:
#   See README.md for full setup instructions
# =============================================================================

import requests, re, time, json, os, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import importlib.util
import tempfile

# =============================================================================
# ── DIRECTORY SETUP
# =============================================================================

BASE_DIR    = Path.home() / "trading_bot"       # Root folder for all bot files
DATA_DIR    = BASE_DIR / "data"                 # Parquet data files go here
BACKUP_DIR  = BASE_DIR / "best_backups"         # Best-ever bot backups
ARCHIVE_DIR = BASE_DIR / "top3_archive"         # Top-3 historical variants

BASE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(exist_ok=True)
ARCHIVE_DIR.mkdir(exist_ok=True)

# =============================================================================
# ── BOT DEFINITIONS
# Each bot has a name, file path, and strategy description.
# CHANGE: Add or remove bots here. Path = where the evolved code is saved.
# =============================================================================

BOTS = [
    {
        "name":     "Bot1_Trend",
        "path":     BASE_DIR / "bot1_trend" / "initial.py",
        "strategy": "EMA crossover trend following with ADX filter - FUTURES long and short"
    },
    {
        "name":     "Bot2_Reversion",
        "path":     BASE_DIR / "bot2_reversion" / "initial.py",
        "strategy": "Bollinger Band mean reversion with RSI - FUTURES long and short"
    },
    {
        "name":     "Bot3_Momentum",
        "path":     BASE_DIR / "bot3_momentum" / "initial.py",
        "strategy": "Volume breakout momentum with MACD - FUTURES long and short"
    },
]

# =============================================================================
# ── OLLAMA / MODEL SETTINGS
# CHANGE: Set REASON_MODEL and CODE_MODEL to whichever Ollama models you have.
#
# Recommended combos by RAM:
#   8 GB  RAM → REASON_MODEL = "deepseek-r1:1.5b"  CODE_MODEL = "qwen2.5-coder:3b"
#   16 GB RAM → REASON_MODEL = "deepseek-r1:7b"    CODE_MODEL = "qwen2.5-coder:7b"   ← DEFAULT
#   32 GB RAM → REASON_MODEL = "deepseek-r1:14b"   CODE_MODEL = "qwen2.5-coder:14b"
#   64 GB RAM → REASON_MODEL = "deepseek-r1:32b"   CODE_MODEL = "qwen2.5-coder:32b"
#
# Pull models before running:
#   ollama pull deepseek-r1:7b
#   ollama pull qwen2.5-coder:7b
# =============================================================================

OLLAMA_URL   = "http://localhost:11434/api/generate"   # CHANGE: if Ollama runs on different host/port
REASON_MODEL = "deepseek-r1:7b"                        # CHANGE: reasoning model (strategy ideas)
CODE_MODEL   = "qwen2.5-coder:7b"                      # CHANGE: code model (implements strategies)

# =============================================================================
# ── BACKTEST SETTINGS
# CHANGE: Adjust date split for your own walk-forward window.
# CHANGE: INIT_CASH is paper money only — no real funds are used.
# =============================================================================

TRAIN_END  = "2024-01-01"   # CHANGE: data before this date = training set
TEST_START = "2024-01-01"   # CHANGE: data from this date = out-of-sample test
INIT_CASH  = 1000.0         # CHANGE: starting paper capital in USDT
LEVERAGE   = 3              # CHANGE: futures leverage (1 = spot, 3 = 3x futures)
FEES       = 0.0004         # CHANGE: trading fee per side (Binance futures taker = 0.0004)

# =============================================================================
# ── RISK MANAGEMENT SETTINGS
# CHANGE: BASE_RISK / MAX_RISK control position sizing as % of equity per trade.
# CHANGE: STOP_ATR controls how many ATR units away the stop loss is placed.
# =============================================================================

BASE_RISK  = 0.006   # CHANGE: minimum risk per trade (0.6% of equity)
MAX_RISK   = 0.01    # CHANGE: maximum risk per trade (1.0% of equity)
STOP_ATR   = 2.0     # CHANGE: stop loss distance = STOP_ATR × ATR(14)

# =============================================================================
# ── TRADING PAIRS
# CHANGE: FAST_PAIRS used for quick screening (fewer pairs = faster).
# CHANGE: ALL_PAIRS used for full evaluation.
# All pairs must be available on Binance spot/futures.
# =============================================================================

FAST_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "XRP/USDT"
]

ALL_PAIRS = [
    "BTC/USDT",  "ETH/USDT",  "BNB/USDT",  "SOL/USDT",  "XRP/USDT",
    "ADA/USDT",  "DOGE/USDT", "MATIC/USDT","DOT/USDT",  "LTC/USDT",
    "AVAX/USDT", "LINK/USDT", "UNI/USDT",  "ATOM/USDT", "TRX/USDT"
]

# =============================================================================
# ── DATA DOWNLOAD SETTINGS
# CHANGE: TIMEFRAME to use different candle intervals.
# CHANGE: SINCE_DATE to download more or less history.
# CHANGE: EXCHANGE_ID to use a different CCXT-supported exchange.
# =============================================================================

TIMEFRAME   = "15m"             # CHANGE: candle interval ("1m","5m","15m","1h","4h","1d")
SINCE_DATE  = "2020-01-01"      # CHANGE: start date for historical data download
EXCHANGE_ID = "binance"         # CHANGE: any CCXT exchange (e.g. "bybit", "okx", "kraken")
TOP3_SIZE   = 3                 # number of top variants to archive per bot

# =============================================================================
# ── DATA INTEGRITY TRACKING
# Stores checksums so we only re-process changed/missing files.
# =============================================================================

DATA_MANIFEST = DATA_DIR / ".manifest.json"

def load_manifest():
    if DATA_MANIFEST.exists():
        try:
            return json.loads(DATA_MANIFEST.read_text())
        except:
            pass
    return {}

def save_manifest(manifest):
    DATA_MANIFEST.write_text(json.dumps(manifest, indent=2))

def file_checksum(path):
    h = hashlib.md5()
    h.update(Path(path).read_bytes())
    return h.hexdigest()

# =============================================================================
# ── DATA DOWNLOAD
# Downloads OHLCV candles from exchange via CCXT.
# Skips pairs that already have a valid parquet file (checksum match).
# =============================================================================

def download_data(pairs=None, force=False):
    """
    Download OHLCV data for all pairs and save as parquet.
    Only downloads a pair if its parquet file is missing or force=True.
    """
    try:
        import ccxt
    except ImportError:
        print("  [!] ccxt not installed. Run: pip install ccxt")
        return

    if pairs is None:
        pairs = ALL_PAIRS

    manifest = load_manifest()
    exchange_cls = getattr(ccxt, EXCHANGE_ID)
    exchange = exchange_cls({"enableRateLimit": True})

    print(f"\n{'='*60}")
    print(f"DATA DOWNLOAD  [{EXCHANGE_ID.upper()}  {TIMEFRAME}  since {SINCE_DATE}]")
    print(f"{'='*60}")

    since_ms = exchange.parse8601(f"{SINCE_DATE}T00:00:00Z")

    for pair in pairs:
        fname   = pair.replace("/", "_") + ".parquet"
        fpath   = DATA_DIR / fname

        # Skip if already downloaded and recorded in manifest
        if not force and fpath.exists() and manifest.get(fname, {}).get("downloaded"):
            print(f"  ✓ {pair} — already downloaded, skipping")
            continue

        print(f"  ↓ {pair} …", end=" ", flush=True)
        all_ohlcv = []
        cursor    = since_ms

        try:
            while True:
                ohlcv = exchange.fetch_ohlcv(pair, TIMEFRAME, since=cursor, limit=1000)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                cursor = ohlcv[-1][0] + 1
                if len(ohlcv) < 1000:
                    break
                time.sleep(exchange.rateLimit / 1000)

            df = pd.DataFrame(all_ohlcv, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df = df[~df.index.duplicated(keep="last")]
            df.sort_index(inplace=True)
            df.to_parquet(fpath)

            chk = file_checksum(fpath)
            manifest[fname] = {"downloaded": True, "checksum": chk, "rows": len(df), "arranged": False}
            save_manifest(manifest)
            print(f"{len(df):,} candles saved ✓")

        except Exception as e:
            print(f"FAILED — {e}")

    print(f"\nDownload complete. Files in: {DATA_DIR}\n")

# =============================================================================
# ── DATA ARRANGEMENT / VALIDATION
# Verifies each parquet has correct columns, no all-NaN rows, proper index.
# Marks files as "arranged" in manifest so they aren't re-processed.
# =============================================================================

def arrange_data(pairs=None, force=False):
    """
    Validate and clean downloaded parquet files.
    Only re-arranges if the file has changed or was never arranged.
    """
    if pairs is None:
        pairs = ALL_PAIRS

    manifest  = load_manifest()
    required  = {"open", "high", "low", "close", "volume"}
    fixed     = 0
    skipped   = 0

    print(f"\n{'='*60}")
    print("DATA ARRANGEMENT / VALIDATION")
    print(f"{'='*60}")

    for pair in pairs:
        fname = pair.replace("/", "_") + ".parquet"
        fpath = DATA_DIR / fname

        if not fpath.exists():
            print(f"  ✗ {pair} — parquet not found, run download first")
            continue

        chk = file_checksum(fpath)
        entry = manifest.get(fname, {})

        # Skip if already arranged and file hasn't changed
        if (not force
                and entry.get("arranged")
                and entry.get("checksum") == chk):
            skipped += 1
            print(f"  ✓ {pair} — already arranged, skipping")
            continue

        print(f"  ⚙ {pair} …", end=" ", flush=True)
        try:
            df = pd.read_parquet(fpath)

            # Ensure required columns exist
            missing = required - set(df.columns)
            if missing:
                print(f"MISSING COLUMNS {missing} — skipping")
                continue

            # Drop rows where all OHLCV are NaN
            df.dropna(subset=list(required), how="all", inplace=True)

            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)

            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep="last")]

            # Cast to float
            for col in required:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df.to_parquet(fpath)
            chk = file_checksum(fpath)
            manifest[fname]["arranged"] = True
            manifest[fname]["checksum"] = chk
            manifest[fname]["rows"]     = len(df)
            save_manifest(manifest)
            fixed += 1
            print(f"{len(df):,} rows OK ✓")

        except Exception as e:
            print(f"ERROR — {e}")

    print(f"\nArrangement complete. Fixed={fixed} Skipped={skipped}\n")

# =============================================================================
# ── OLLAMA HELPERS
# =============================================================================

def check_ollama():
    """Block until Ollama server is reachable."""
    while True:
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            if r.status_code == 200:
                return
        except:
            pass
        print("  [!] Ollama not running. Waiting 30s …")
        print("  [!] In another terminal:  ollama serve")
        time.sleep(30)

def ollama(model, prompt, idle_timeout=300):
    """Send a prompt to a local Ollama model and return the full response."""
    check_ollama()
    print(f"  [{model}] generating …", flush=True)
    response_text = ""
    try:
        with requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=idle_timeout
        ) as resp:
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode())
                        response_text += data.get("response", "")
                        if data.get("done"):
                            break
                    except:
                        pass
    except requests.exceptions.Timeout:
        print(f"  [{model}] idle timeout.")
    except Exception as e:
        print(f"  [{model}] error: {e}")
    return response_text

def clean_think(text):
    """Strip DeepSeek <think>…</think> chain-of-thought blocks."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    if "...done thinking." in text:
        return text.split("...done thinking.")[-1].strip()
    if "<think>" in text:
        parts = text.split("<think>")
        return parts[0].strip() or parts[-1].strip()
    return text.strip()

# =============================================================================
# ── DATA LOADING
# =============================================================================

def load_parquet(pair, phase="test"):
    """Load a pair's parquet and slice to train or test window."""
    fname = pair.replace("/", "_") + ".parquet"
    path  = DATA_DIR / fname
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if phase == "train":
        df = df[df.index < TRAIN_END]
    else:
        df = df[df.index >= TEST_START]
    return df if len(df) > 200 else None

# =============================================================================
# ── RISK / POSITION SIZING
# =============================================================================

def compute_dynamic_risk(adx_val, atr_val, atr_median_val, equity, equity_ma, risk_multiplier=1.0):
    """
    Scale risk per trade dynamically based on:
      - ADX trend strength  → higher ADX = more risk allowed
      - ATR vs its median   → high volatility = reduce risk
      - Equity vs MA        → drawdown regime = halve risk
    """
    if atr_median_val <= 0:
        return BASE_RISK
    regime    = max(0.0, min((adx_val - 20.0) / 20.0, 1.0))
    dr        = BASE_RISK + regime * (MAX_RISK - BASE_RISK)
    vol_ratio = max(0.75, min(atr_val / atr_median_val, 1.25))
    dr       *= (1.0 / vol_ratio)
    if equity < equity_ma:
        dr *= 0.5
    dr *= risk_multiplier
    return min(dr, MAX_RISK)

# =============================================================================
# ── TRAILING STOPS
# Tightens stop loss automatically as a trade moves into profit.
# Applied to the signal DataFrame before VectorBT runs.
# =============================================================================

def apply_trailing_stops(signals_df):
    df = signals_df.copy()
    if "atr" not in df.columns or "stop_dist" not in df.columns:
        return df

    in_long = False; in_short = False
    entry_price = 0.0; initial_risk = 0.0
    extreme_high = 0.0; extreme_low = float("inf")
    trailing_stop_long = 0.0; trailing_stop_short = float("inf")

    long_exits  = df["long_exit"].copy()
    short_exits = df["short_exit"].copy()

    for i in range(len(df)):
        row   = df.iloc[i]
        price = float(row["close"])
        atr   = float(row["atr"]) if not np.isnan(row["atr"]) else 0.001

        if in_long:
            extreme_high = max(extreme_high, price)
            current_r    = (price - entry_price) / initial_risk if initial_risk > 0 else 0
            if current_r >= 1.0:
                trailing_stop_long = max(trailing_stop_long, entry_price + 0.5 * atr)
            if current_r >= 1.5:
                trailing_stop_long = max(trailing_stop_long, extreme_high - 2.5 * atr)
            if current_r >= 2.0:
                trailing_stop_long = max(trailing_stop_long, extreme_high - 1.5 * atr)
            if price <= trailing_stop_long and trailing_stop_long > 0:
                long_exits.iloc[i] = True; in_long = False

        if in_short:
            extreme_low = min(extreme_low, price)
            current_r   = (entry_price - price) / initial_risk if initial_risk > 0 else 0
            if current_r >= 1.0:
                trailing_stop_short = min(trailing_stop_short, entry_price - 0.5 * atr)
            if current_r >= 1.5:
                trailing_stop_short = min(trailing_stop_short, extreme_low + 2.5 * atr)
            if current_r >= 2.0:
                trailing_stop_short = min(trailing_stop_short, extreme_low + 1.5 * atr)
            if price >= trailing_stop_short and trailing_stop_short < float("inf"):
                short_exits.iloc[i] = True; in_short = False

        if not in_long and not in_short:
            if row["long_entry"] and not row["long_exit"]:
                in_long = True; entry_price = price
                initial_risk       = float(row["stop_dist"]) if not np.isnan(row["stop_dist"]) else atr * STOP_ATR
                extreme_high       = price
                trailing_stop_long = price - initial_risk
            elif row["short_entry"] and not row["short_exit"]:
                in_short = True; entry_price = price
                initial_risk        = float(row["stop_dist"]) if not np.isnan(row["stop_dist"]) else atr * STOP_ATR
                extreme_low         = price
                trailing_stop_short = price + initial_risk

        if in_long  and (row["long_exit"]  or long_exits.iloc[i]):  in_long  = False
        if in_short and (row["short_exit"] or short_exits.iloc[i]): in_short = False

    df["long_exit"]  = long_exits
    df["short_exit"] = short_exits
    return df

# =============================================================================
# ── PYRAMIDING (add to winning trades)
# CHANGE: Pyramiding adds one unit at 1R profit. Set adds < N to limit additions.
# =============================================================================

def apply_pyramiding(signals_df, init_cash=INIT_CASH):
    df = signals_df.copy()
    if "atr" not in df.columns or "stop_dist" not in df.columns:
        return df

    in_long = False; entry_price = 0.0; initial_risk = 0.0; adds = 0
    equity  = init_cash; equity_curve = [init_cash]; equity_ma = init_cash; risk_mult = 1.0

    for i in range(len(df)):
        row   = df.iloc[i]
        price = float(row["close"])
        atr   = float(row["atr"]) if not np.isnan(row["atr"]) else 0.001
        atr_med = float(row.get("atr_median", atr))
        adx   = float(row.get("adx", 25))

        if len(equity_curve) >= 50:
            equity_ma = float(np.mean(equity_curve[-50:]))

        peak = max(equity_curve)
        dd   = (equity - peak) / peak if peak > 0 else 0

        # CHANGE: Drawdown thresholds for reducing risk
        if   dd < -0.20: risk_mult = 0.50
        elif dd < -0.10: risk_mult = 0.75
        else:            risk_mult = 1.0

        if in_long and adds < 1:   # CHANGE: adds < N controls max pyramid layers
            current_r = (price - entry_price) / initial_risk if initial_risk > 0 else 0
            if current_r >= 1.0:
                compute_dynamic_risk(adx, atr, atr_med, equity, equity_ma, risk_mult)
                adds += 1

        if in_long and row["long_exit"]:
            in_long = False; adds = 0

        if not in_long and row["long_entry"] and not row["long_exit"]:
            in_long = True; entry_price = price
            initial_risk = float(row["stop_dist"]) if not np.isnan(row["stop_dist"]) else atr * STOP_ATR
            adds = 0

        equity_curve.append(equity)
    return df

# =============================================================================
# ── VECTORBT BACKTEST ENGINE
# =============================================================================

def run_vbt_backtest(df, init_cash=INIT_CASH, fees=FEES):
    """Run a full VectorBT portfolio simulation and return key stats."""
    import vectorbt as vbt

    pf = vbt.Portfolio.from_signals(
        df["close"],
        entries       = df["long_entry"],
        exits         = df["long_exit"],
        short_entries = df["short_entry"],
        short_exits   = df["short_exit"],
        init_cash     = init_cash,
        fees          = fees,
        size          = 0.95,           # CHANGE: fraction of equity used per trade
        size_type     = "percent",
        freq          = "15min",        # CHANGE: match your TIMEFRAME
        accumulate    = False,
        upon_opposite_entry = "close",
    )

    total_return = float(pf.total_return()) * 100
    try:
        sharpe = float(pf.sharpe_ratio())
        if np.isnan(sharpe): sharpe = 0.0
    except: sharpe = 0.0
    try:
        max_dd = float(abs(pf.max_drawdown())) * 100
        if np.isnan(max_dd): max_dd = 100.0
    except: max_dd = 100.0
    try:
        n_trades = int(pf.trades.count())
        win_rate = float(pf.trades.win_rate()) * 100 if n_trades > 0 else 0.0
        if np.isnan(win_rate): win_rate = 0.0
    except: n_trades = 0; win_rate = 0.0

    return {
        "total_return": round(total_return, 2),
        "sharpe":       round(sharpe, 3),
        "max_dd":       round(max_dd, 2),
        "win_rate":     round(win_rate, 2),
        "num_trades":   n_trades
    }

# =============================================================================
# ── SCORING FORMULA
# CHANGE: Adjust weights (0.40 / 0.30 / 0.20 / 0.10) to reprioritise metrics.
#   Current priority: Sharpe > Return/DD ratio > Win-rate > penalise drawdown
# =============================================================================

def calculate_score(ar, ash, add, awr, atr):
    """
    ar  = avg return (%)
    ash = avg Sharpe
    add = avg max drawdown (%)
    awr = avg win rate (%)
    atr = avg trade count
    """
    if atr < 3:   # CHANGE: minimum trades required to get a non-zero score
        return 0.0
    s  = 0.0
    s += max(0.0, min((ash + 2.0) / 5.0, 1.0)) * 0.40   # Sharpe component
    if add > 0:
        s += max(0.0, min((ar / add + 1.0) / 3.0, 1.0)) * 0.30  # Return/DD component
    s += max(0.0, min(awr / 100.0, 1.0)) * 0.20           # Win-rate component
    s -= min(add / 50.0, 1.0) * 0.10                       # Drawdown penalty
    return max(0.0001, round(s, 4))

# =============================================================================
# ── RUN BACKTEST ON MULTIPLE PAIRS
# =============================================================================

def backtest_module_on_pairs(module, pairs, phase="test"):
    results = []
    for pair in pairs:
        df = load_parquet(pair, phase=phase)
        if df is None:
            continue
        try:
            df = module.compute_indicators(df.copy())
            df = module.get_signals(df)
            if "stop_dist" in df.columns:
                df = apply_trailing_stops(df)
                df = apply_pyramiding(df)
            df.dropna(inplace=True)
            if len(df) < 50:
                continue
            for col in ["long_entry", "long_exit", "short_entry", "short_exit"]:
                if col not in df.columns:
                    df[col] = False
                df[col] = df[col].fillna(False).astype(bool)
            stats = run_vbt_backtest(df)
            stats["pair"] = pair
            results.append(stats)
            print(f"  {pair}: return={stats['total_return']}% dd={stats['max_dd']}% "
                  f"wr={stats['win_rate']}% trades={stats['num_trades']}")
        except Exception as e:
            print(f"  {pair}: {e}")
    return results

# =============================================================================
# ── CODE AUTOFIX
# Repairs common mistakes the LLM makes before the code is even tested.
# =============================================================================

def _autofix_code(code_str):
    code_str = code_str.replace("\t", "    ")
    code_str = re.sub(r"```python\s*", "", code_str)
    code_str = re.sub(r"```\s*",       "", code_str)
    # Fix literal word 'value' used instead of a number
    code_str = re.sub(r'("\w+")\s*:\s*value\b', r'\1: 2.0', code_str)
    code_str = code_str.replace("\\ ", " ")

    # Inject c,h,l,v unpacking at top of compute_indicators
    if "def compute_indicators" in code_str:
        already = ("c, h, l, v" in code_str or "c,h,l,v" in code_str or
                   "c = df[" in code_str or "h = df[" in code_str)
        if not already:
            lines = code_str.split("\n")
            new_lines = []; waiting = False; done = False
            for line in lines:
                stripped = line.strip()
                if not done and "def compute_indicators" in line and stripped.endswith(":"):
                    waiting = True; new_lines.append(line); continue
                if waiting and not done:
                    if stripped and not stripped.startswith("#"):
                        new_lines.append('    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]')
                        done = True; waiting = False
                new_lines.append(line)
            code_str = "\n".join(new_lines)

    # Inject c,h,l,v in get_signals if used but not defined
    if "def get_signals" in code_str:
        gs = re.search(r"def get_signals\(df\):(.*?)(?=\ndef |\n# EVOLVE|\Z)", code_str, re.DOTALL)
        if gs:
            body = gs.group(1)
            uses = any(f" {v} " in body or f"({v}" in body for v in ["c","h","l","v"])
            has  = any(x in body for x in ["c, h, l, v","c,h,l,v","c = df"])
            if uses and not has:
                code_str = re.sub(
                    r"(def get_signals\(df\):)",
                    r'\1\n    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]',
                    code_str, count=1
                )

    # Ensure ATR column exists
    if "def compute_indicators" in code_str and 'df["atr"]' not in code_str:
        code_str = re.sub(
            r"(def compute_indicators\(df\):.*?)(\n    return df)",
            r'\1\n    df["atr"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()\2',
            code_str, flags=re.DOTALL, count=1
        )

    # Ensure atr_median column exists
    if "def compute_indicators" in code_str and "atr_median" not in code_str:
        code_str = re.sub(
            r'(df\["atr"\]\s*=\s*ta\.volatility\.AverageTrueRange[^\n]+)',
            r'\1\n    df["atr_median"] = df["atr"].rolling(100).median()',
            code_str, count=1
        )

    # Ensure ADX column exists if adx is referenced
    if "adx" in code_str and 'df["adx"]' not in code_str and "def compute_indicators" in code_str:
        code_str = re.sub(
            r"(def compute_indicators\(df\):.*?)(\n    return df)",
            r'\1\n    _adx_i = ta.trend.ADXIndicator(h, l, c, window=14)\n    df["adx"] = _adx_i.adx()\2',
            code_str, flags=re.DOTALL, count=1
        )

    # Ensure stop_dist exists in get_signals
    if "def get_signals" in code_str and "stop_dist" not in code_str:
        code_str = re.sub(
            r"(def get_signals\(df\):.*?)(\n    return df)",
            r'\1\n    df["stop_dist"] = CONFIG.get("stop_atr", 2.0) * df["atr"]\2',
            code_str, flags=re.DOTALL, count=1
        )

    # Ensure all 4 signal columns exist
    for sig in ["long_entry", "long_exit", "short_entry", "short_exit"]:
        if sig not in code_str and "def get_signals" in code_str:
            code_str = re.sub(
                r"(def get_signals\(df\):.*?)(\n    return df)",
                rf'\1\n    df["{sig}"] = pd.Series(False, index=df.index)\2',
                code_str, flags=re.DOTALL, count=1
            )

    return code_str

# =============================================================================
# ── CODE VALIDATION
# =============================================================================

def _validate_code(code_str):
    import ast
    try:
        ast.parse(code_str)
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"
    if "def compute_indicators" not in code_str: return False, "Missing compute_indicators"
    if "def get_signals"        not in code_str: return False, "Missing get_signals"
    if "long_entry"             not in code_str: return False, "Missing long_entry signal"
    if "stop_dist"              not in code_str: return False, "Missing stop_dist"
    return True, ""

# =============================================================================
# ── SELF-HEALING  (asks Qwen to fix its own broken code)
# =============================================================================

def self_heal_code(broken_code, error_msg):
    print(f"    [SELF-HEAL] Error: {error_msg[:80]}")
    print(f"    [SELF-HEAL] Asking {CODE_MODEL} to fix …")
    prompt = f"""Fix this broken Python trading strategy code.

ERROR: {error_msg}

BROKEN CODE:
{broken_code[:2500]}

FIX THESE COMMON MISTAKES:
1. CONFIG values must be real numbers, NEVER the word 'value'
   WRONG:  "stop_atr": value
   CORRECT: "stop_atr": 2.0
2. Variable 'c', 'h', 'l', 'v' must be defined at top of compute_indicators:
   c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
3. No syntax errors - check all parentheses are closed
4. Both functions need: return df at the end
5. All 4 signals needed: long_entry, long_exit, short_entry, short_exit
6. stop_dist = CONFIG["stop_atr"] * df["atr"] must exist in get_signals
7. Only use ta functions - no custom indicators
8. .astype(bool) on SAME line as condition

Return ONLY the complete fixed Python code. No explanation. No markdown."""

    response = ollama(CODE_MODEL, prompt, idle_timeout=120)
    m = re.search(r'```python\s*(.*?)```', response, re.DOTALL)
    if m:
        fixed = m.group(1).strip()
    elif "def compute_indicators" in response:
        lines = response.split("\n")
        start = 0
        for i, line in enumerate(lines):
            if (line.startswith("import ") or line.startswith("# ")
                    or "DATA_DIR" in line or "CONFIG" in line):
                start = i; break
        fixed = "\n".join(lines[start:]).strip()
    else:
        print("    [SELF-HEAL] Could not extract code from response")
        return None

    fixed = fixed.replace("\t", "    ")
    print(f"    [SELF-HEAL] Got fixed code ({len(fixed)} chars)")
    return fixed

# =============================================================================
# ── BACKTEST A CODE STRING (auto-fix → validate → load → run)
# =============================================================================

def backtest_code(code_str, label="v", phase="test", fast=False):
    tmp_path = None
    code_str = _autofix_code(code_str)
    ok, err  = _validate_code(code_str)

    if not ok:
        print(f"  [{label}] pre-validation failed: {err}")
        if "_healed" not in label:
            healed = self_heal_code(code_str, err)
            if healed:
                return backtest_code(healed, label + "_healed", phase, fast)
        return 0.0, f"error: {err}", []

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=BASE_DIR) as f:
            f.write(code_str); tmp_path = Path(f.name)

        spec   = importlib.util.spec_from_file_location("tmp_bot", tmp_path)
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except Exception as load_err:
            if tmp_path: tmp_path.unlink(missing_ok=True)
            err_str = str(load_err)
            if "_healed" not in label:
                print(f"  [{label}] error: {err_str[:80]} → self-healing …")
                healed = self_heal_code(code_str, err_str)
                if healed:
                    return backtest_code(healed, label + "_healed", phase, fast)
            else:
                print(f"  [{label}] heal failed: {err_str[:60]}")
            return 0.0, f"error: {err_str}", []

        if not hasattr(module, "compute_indicators"): raise RuntimeError("Missing compute_indicators")
        if not hasattr(module, "get_signals"):        raise RuntimeError("Missing get_signals")

        pairs   = FAST_PAIRS if fast else ALL_PAIRS
        results = backtest_module_on_pairs(module, pairs, phase=phase)
        if tmp_path: tmp_path.unlink(missing_ok=True)

        if not results:
            return 0.0, f"[{phase}] no results", []

        ar  = np.mean([r["total_return"] for r in results])
        ash = np.mean([r["sharpe"]       for r in results])
        add = np.mean([r["max_dd"]       for r in results])
        awr = np.mean([r["win_rate"]     for r in results])
        atr = np.mean([r["num_trades"]   for r in results])

        print(f"Avg [{phase}{'_fast' if fast else ''}]: "
              f"return={ar:.2f}% sharpe={ash:.3f} dd={add:.2f}% wr={awr:.2f}%")

        score   = calculate_score(ar, ash, add, awr, atr)
        summary = (f"[{phase}{'_fast' if fast else ''}] score={score} "
                   f"return={ar:.1f}% sharpe={ash:.2f} dd={add:.1f}% "
                   f"wr={awr:.1f}% trades={atr:.0f}")
        print(f"  [{label}] {summary}")
        return score, summary, results

    except Exception as e:
        print(f"  [{label}] error: {e}")
        if tmp_path: tmp_path.unlink(missing_ok=True)
        return 0.0, f"error: {e}", []

# =============================================================================
# ── FILE HELPERS
# =============================================================================

def read_bot(path):  return path.read_text() if path.exists() else ""
def write_bot(path, code): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(code)

def extract_config(code):
    m = re.search(r"CONFIG\s*=\s*\{.*?\}", code, re.DOTALL)
    return m.group(0)[:600] if m else "CONFIG not found"

def format_pairs(results):
    if not results: return "No results."
    lines = []
    for r in sorted(results, key=lambda x: x["total_return"]):
        lines.append(f"  {r['pair']}: return={r['total_return']}% dd={r['max_dd']}% "
                     f"wr={r['win_rate']}% trades={r['num_trades']}")
    return "\n".join(lines)

# =============================================================================
# ── BACKUP & ARCHIVE HELPERS
# =============================================================================

def backup_best(bot_name, code, score):
    (BACKUP_DIR / f"{bot_name}_best.py").write_text(code)
    (BACKUP_DIR / f"{bot_name}_score.txt").write_text(str(score))
    print(f"  ★ Backup saved: {bot_name} score={score}")

def load_backup(bot_name):
    bp = BACKUP_DIR / f"{bot_name}_best.py"
    sp = BACKUP_DIR / f"{bot_name}_score.txt"
    if bp.exists() and sp.exists():
        try: return bp.read_text(), float(sp.read_text().strip())
        except: pass
    return None, 0.0

def load_backup_score(bot_name):
    sp = BACKUP_DIR / f"{bot_name}_score.txt"
    try: return float(sp.read_text().strip()) if sp.exists() else 0.0
    except: return 0.0

def update_top3(bot_name, code, score):
    entries = []
    for i in range(1, TOP3_SIZE + 1):
        cp = ARCHIVE_DIR / f"{bot_name}_top{i}.py"
        sp = ARCHIVE_DIR / f"{bot_name}_top{i}_score.txt"
        if cp.exists() and sp.exists():
            try: entries.append((float(sp.read_text().strip()), cp.read_text()))
            except: pass
    entries.append((score, code))
    entries.sort(key=lambda x: x[0], reverse=True)
    entries = entries[:TOP3_SIZE]
    for rank, (sc, cd) in enumerate(entries, 1):
        (ARCHIVE_DIR / f"{bot_name}_top{rank}.py").write_text(cd)
        (ARCHIVE_DIR / f"{bot_name}_top{rank}_score.txt").write_text(str(sc))
    print(f"  Top3 updated: {[round(e[0], 4) for e in entries]}")

def update_history(path, cycle, summary):
    content  = read_bot(path)
    new_line = f"# Cycle {cycle}: {summary}"
    if "# RESULTS_HISTORY:" in content:
        lines = content.split("\n")
        out = []; history = []; in_hist = False
        for line in lines:
            if line.strip() == "# RESULTS_HISTORY:":
                in_hist = True; out.append(line); continue
            if in_hist and line.startswith("# Cycle"):
                history.append(line)
            elif in_hist and line.startswith("# [No"):
                pass
            elif in_hist and not line.startswith("#"):
                in_hist = False
                history = history[-9:]          # CHANGE: keep last N cycles in history
                history.insert(0, new_line)
                for h in history: out.append(h)
                out.append(line)
            else:
                out.append(line)
        content = "\n".join(out)
    else:
        content = f"# RESULTS_HISTORY:\n{new_line}\n" + content
    write_bot(path, content)

# =============================================================================
# ── LLM PIPELINE: GENERATE STRATEGIES (DeepSeek-R1)
# =============================================================================

def generate_strategies(bot, current_code, last_result, pair_results, cycle):
    prompt = f"""You are a quantitative trading strategy analyst for FUTURES markets.
Bot: {bot['name']}
Strategy: {bot['strategy']}
Cycle: {cycle}
Last result: {last_result}
Per-pair performance (worst first):
{format_pairs(pair_results)}
Current CONFIG:
{extract_config(current_code)}
Generate as many DIFFERENT improvement strategies as needed (min 3, max 7).
Format EXACTLY:
STRATEGY 1: [name]
- change [param] to [exact value]
- change [param] to [exact value]
- add filter: [exact condition with number]
- reason: [one line]
STRATEGY 2: [name]
- ...
Rules: specific numbers, different aspects each, goals: dd<30% wr>40% sharpe>1, no code, no questions"""

    response  = ollama(REASON_MODEL, prompt, idle_timeout=300)
    response  = clean_think(response)
    strategies = []
    parts = re.split(r"STRATEGY\s+\d+[:\.]", response)
    for part in parts[1:]:
        s = part.strip()
        if s and len(s) > 20:
            strategies.append(s)

    # Fallback if model returns too few
    while len(strategies) < 3:
        strategies.append(
            "Change ADX threshold to 35, EMA fast=21 slow=55, "
            "add RSI filter 45-70, volume_factor=2.0, stop_atr=2.5"
        )

    print(f"\n  DeepSeek: {len(strategies)} strategies")
    for i, s in enumerate(strategies):
        print(f"  S{i+1}: {s[:80]} …")
    return strategies

# =============================================================================
# ── LLM PIPELINE: IMPLEMENT STRATEGY AS CODE (Qwen2.5-Coder)
# =============================================================================

def implement_strategy(bot, current_code, strategy, variant_num):
    history_match = re.search(r"# RESULTS_HISTORY:(.*?)# EVOLVE-BLOCK", current_code, re.DOTALL)
    history = history_match.group(1).strip() if history_match else "# [No runs yet]"

    prompt = f"""You are an expert Python trading developer for FUTURES markets.
Implement this improvement: {strategy}
Bot type: {bot['strategy']}
AVAILABLE ta functions (use ONLY these):
  ta.trend.EMAIndicator(close, window=N).ema_indicator()
  ta.trend.ADXIndicator(high, low, close, window=N).adx() / .adx_pos() / .adx_neg()
  ta.momentum.RSIIndicator(close, window=N).rsi()
  ta.volatility.BollingerBands(close, window=N, window_dev=N).bollinger_hband() / .bollinger_lband() / .bollinger_mavg()
  ta.volatility.AverageTrueRange(high, low, close, window=N).average_true_range()
  ta.trend.MACD(close, window_fast=N, window_slow=N, window_sign=N).macd() / .macd_signal()
Output EXACTLY these 3 sections only:
===CONFIG===
{{
    "param1": value,
    "stop_atr": 2.0,
    "init_cash": 1000,
    "fees": 0.0004,
    "leverage": 3,
}}
===INDICATORS===
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["atr"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df["atr_median"] = df["atr"].rolling(100).median()
    return df
===SIGNALS===
    df["long_entry"]  = (condition).astype(bool)
    df["long_exit"]   = (condition).astype(bool)
    df["short_entry"] = (condition).astype(bool)
    df["short_exit"]  = (condition).astype(bool)
    df["stop_dist"]   = CONFIG["stop_atr"] * df["atr"]
    return df
CRITICAL RULES:
1. Output ONLY the 3 sections. No extra text or markdown.
2. CONFIG values must be real numbers. NEVER use the word 'value'.
3. INDICATORS first line MUST be: c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
4. All 4 signals must end with .astype(bool)
5. SIGNALS must have: df["stop_dist"] = CONFIG["stop_atr"] * df["atr"]
6. Use ONLY the ta functions listed above."""

    response = ollama(CODE_MODEL, prompt, idle_timeout=300)

    cm = re.search(r"===CONFIG===\s*(\{.*?\})",                  response, re.DOTALL)
    im = re.search(r"===INDICATORS===\s*(.*?)(?:===SIGNALS===|$)", response, re.DOTALL)
    sm = re.search(r"===SIGNALS===\s*(.*?)$",                     response, re.DOTALL)

    if not cm or not im or not sm:
        print(f"  V{variant_num}: bad format, using current"); return current_code

    config_block     = cm.group(1).strip()
    indicators_block = im.group(1).strip()
    signals_block    = sm.group(1).strip()

    if "long_entry" not in signals_block:
        print(f"  V{variant_num}: missing signals, using current"); return current_code

    if "return df" not in indicators_block: indicators_block += "\n    return df"
    if "return df" not in signals_block:    signals_block    += "\n    return df"

    for key, val in [("stop_atr","2.0"),("init_cash","1000"),("fees","0.0004"),("leverage","3")]:
        if f'"{key}"' not in config_block:
            config_block = config_block.rstrip("} \n") + f',\n    "{key}": {val},\n}}'

    def reindent(text):
        return "\n".join(
            "    " + line.lstrip() if line.strip() else line
            for line in text.strip().split("\n")
        )

    new_code = f"""# STRATEGY: {bot['strategy']}
# BOT: {bot['name']}
# RESULTS_HISTORY:
{history}

# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
import ta
from pathlib import Path

DATA_DIR = Path.home() / "trading_bot" / "data"

PAIRS = {ALL_PAIRS}

CONFIG = {config_block}

def compute_indicators(df):
{reindent(indicators_block)}

def get_signals(df):
{reindent(signals_block)}
# EVOLVE-BLOCK-END
"""
    print(f"  V{variant_num}: ready ✓")
    return new_code

# =============================================================================
# ── TWO-STAGE VARIANT TESTING
# Stage 1: fast screen on FAST_PAIRS (5 pairs)
# Stage 2: full test on top-3 from stage 1 using ALL_PAIRS (15 pairs)
# =============================================================================

def test_all_variants(variant_codes, phase="test"):
    total = len(variant_codes)
    print(f"\n  Stage 1: Fast test {total} variants (5 pairs) …")
    fast_results = []
    for i, code in enumerate(variant_codes):
        print(f"\n  Fast V{i+1}/{total} …")
        score, summary, pairs = backtest_code(code, f"FV{i+1}", phase=phase, fast=True)
        fast_results.append((score, summary, code, pairs, i))

    fast_results.sort(key=lambda x: x[0], reverse=True)
    top3 = fast_results[:3]

    print(f"\n  Stage 2: Full test top 3 (15 pairs) …")
    full_results = []
    for score_f, summary_f, code, _, orig_idx in top3:
        print(f"\n  Full V{orig_idx+1} (fast={score_f}) …")
        score_full, summary_full, pairs_full = backtest_code(
            code, f"FUV{orig_idx+1}", phase=phase, fast=False
        )
        full_results.append((score_full, summary_full, code, pairs_full))

    return full_results

# =============================================================================
# ── TOURNAMENT: PICK BEST VARIANT
# Tie-break on lower drawdown when scores differ by < 2%
# =============================================================================

def pick_best(variant_results):
    valid = [(s, summ, code, pairs) for s, summ, code, pairs in variant_results
             if code and "error" not in summ]
    if not valid:
        return 0.0, "no valid variants", "", []

    print(f"\n  Tournament ({len(valid)} variants):")
    for i, (s, summ, _, _) in enumerate(valid):
        print(f"  V{i+1}: {summ}")

    sorted_r = sorted(enumerate(valid), key=lambda x: x[1][0], reverse=True)
    best_idx, (best_score, best_summary, best_code, best_pairs) = sorted_r[0]

    # Tie-break: prefer lower drawdown when scores are within 2%
    if len(sorted_r) > 1:
        si, (ss, ssumm, scode, spairs) = sorted_r[1]
        if abs(best_score - ss) < 0.02 and best_score > 0:
            def get_dd(s):
                m = re.search(r"dd=([\d.]+)%", s)
                return float(m.group(1)) if m else 999
            if get_dd(ssumm) < get_dd(best_summary):
                best_idx = si; best_score = ss
                best_summary = ssumm; best_code = scode; best_pairs = spairs

    print(f"\n  Winner: V{best_idx+1} score={best_score}")
    return best_score, best_summary, best_code, best_pairs

# =============================================================================
# ── INITIAL BOT TEMPLATES
# These are the seed strategies that evolve over time.
# They are written to disk only once (if the file doesn't exist yet).
# CHANGE: Modify CONFIG values below to change the starting parameters.
# =============================================================================

BOT1 = '''# STRATEGY: EMA Crossover + ADX - FUTURES
# BOT: Bot1_Trend
# RESULTS_HISTORY:
# [No runs yet]
# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
import ta
from pathlib import Path

DATA_DIR = Path.home() / "trading_bot" / "data"
PAIRS = ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT","DOGE/USDT",
         "MATIC/USDT","DOT/USDT","LTC/USDT","AVAX/USDT","LINK/USDT","UNI/USDT","ATOM/USDT","TRX/USDT"]

# CHANGE: Tune these starting parameters. LLM will evolve them automatically.
CONFIG = {
    "ema_fast":        21,    # fast EMA period
    "ema_slow":        55,    # slow EMA period
    "ema_trend":      200,    # trend filter EMA period
    "adx_period":      14,    # ADX lookback
    "adx_threshold":   30,    # min ADX for trend confirmation
    "rsi_period":      14,
    "rsi_long_min":    45,    # RSI must be above this for longs
    "rsi_long_max":    75,    # RSI must be below this for longs
    "rsi_short_min":   25,
    "rsi_short_max":   55,
    "volume_factor":  1.5,    # volume must be N× its 20-period average
    "stop_atr":       2.0,    # stop loss = 2× ATR
    "init_cash":    1000,
    "fees":         0.0004,
    "leverage":        3,
}

def compute_indicators(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["ema_fast"]   = ta.trend.EMAIndicator(c, window=CONFIG["ema_fast"]).ema_indicator()
    df["ema_slow"]   = ta.trend.EMAIndicator(c, window=CONFIG["ema_slow"]).ema_indicator()
    df["ema_trend"]  = ta.trend.EMAIndicator(c, window=CONFIG["ema_trend"]).ema_indicator()
    adx              = ta.trend.ADXIndicator(h, l, c, window=CONFIG["adx_period"])
    df["adx"]        = adx.adx()
    df["adx_pos"]    = adx.adx_pos()
    df["adx_neg"]    = adx.adx_neg()
    df["rsi"]        = ta.momentum.RSIIndicator(c, window=CONFIG["rsi_period"]).rsi()
    df["atr"]        = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df["atr_median"] = df["atr"].rolling(100).median()
    df["vol_ma"]     = v.rolling(20).mean()
    df["vol_ok"]     = v > (df["vol_ma"] * CONFIG["volume_factor"])
    return df

def get_signals(df):
    cup    = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
    cdn    = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
    tup    = df["close"] > df["ema_trend"]
    tdn    = df["close"] < df["ema_trend"]
    strong = df["adx"] > CONFIG["adx_threshold"]

    df["long_entry"]  = (cup & tup & strong
                         & (df["adx_pos"] > df["adx_neg"])
                         & (df["rsi"] > CONFIG["rsi_long_min"])
                         & (df["rsi"] < CONFIG["rsi_long_max"])
                         & df["vol_ok"]).astype(bool)
    df["long_exit"]   = (cdn | (df["rsi"] > 80)).astype(bool)
    df["short_entry"] = (cdn & tdn & strong
                         & (df["adx_neg"] > df["adx_pos"])
                         & (df["rsi"] > CONFIG["rsi_short_min"])
                         & (df["rsi"] < CONFIG["rsi_short_max"])
                         & df["vol_ok"]).astype(bool)
    df["short_exit"]  = (cup | (df["rsi"] < 20)).astype(bool)
    df["stop_dist"]   = CONFIG["stop_atr"] * df["atr"]
    return df
# EVOLVE-BLOCK-END'''

BOT2 = '''# STRATEGY: Bollinger Band Mean Reversion - FUTURES
# BOT: Bot2_Reversion
# RESULTS_HISTORY:
# [No runs yet]
# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
import ta
from pathlib import Path

DATA_DIR = Path.home() / "trading_bot" / "data"
PAIRS = ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT","DOGE/USDT",
         "MATIC/USDT","DOT/USDT","LTC/USDT","AVAX/USDT","LINK/USDT","UNI/USDT","ATOM/USDT","TRX/USDT"]

# CHANGE: Tune these starting parameters. LLM will evolve them automatically.
CONFIG = {
    "bb_period":       20,    # Bollinger Band lookback
    "bb_std":         2.0,    # standard deviation multiplier
    "rsi_period":      14,
    "rsi_oversold":    30,    # buy signal when RSI < this
    "rsi_overbought":  70,    # sell signal when RSI > this
    "volume_factor":  1.2,
    "stop_atr":       2.0,
    "init_cash":    1000,
    "fees":         0.0004,
    "leverage":        3,
}

def compute_indicators(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    bb               = ta.volatility.BollingerBands(c, window=CONFIG["bb_period"], window_dev=CONFIG["bb_std"])
    df["bb_upper"]   = bb.bollinger_hband()
    df["bb_lower"]   = bb.bollinger_lband()
    df["bb_mid"]     = bb.bollinger_mavg()
    df["rsi"]        = ta.momentum.RSIIndicator(c, window=CONFIG["rsi_period"]).rsi()
    df["atr"]        = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df["atr_median"] = df["atr"].rolling(100).median()
    df["vol_ma"]     = v.rolling(20).mean()
    df["vol_ok"]     = v > (df["vol_ma"] * CONFIG["volume_factor"])
    return df

def get_signals(df):
    df["long_entry"]  = ((df["close"] <= df["bb_lower"])
                         & (df["rsi"] < CONFIG["rsi_oversold"])
                         & df["vol_ok"]).astype(bool)
    df["long_exit"]   = ((df["close"] >= df["bb_mid"])
                         | (df["rsi"] > CONFIG["rsi_overbought"])).astype(bool)
    df["short_entry"] = ((df["close"] >= df["bb_upper"])
                         & (df["rsi"] > CONFIG["rsi_overbought"])
                         & df["vol_ok"]).astype(bool)
    df["short_exit"]  = ((df["close"] <= df["bb_mid"])
                         | (df["rsi"] < CONFIG["rsi_oversold"])).astype(bool)
    df["stop_dist"]   = CONFIG["stop_atr"] * df["atr"]
    return df
# EVOLVE-BLOCK-END'''

BOT3 = '''# STRATEGY: Volume Breakout + MACD - FUTURES
# BOT: Bot3_Momentum
# RESULTS_HISTORY:
# [No runs yet]
# EVOLVE-BLOCK-START
import pandas as pd
import numpy as np
import ta
from pathlib import Path

DATA_DIR = Path.home() / "trading_bot" / "data"
PAIRS = ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT","DOGE/USDT",
         "MATIC/USDT","DOT/USDT","LTC/USDT","AVAX/USDT","LINK/USDT","UNI/USDT","ATOM/USDT","TRX/USDT"]

# CHANGE: Tune these starting parameters. LLM will evolve them automatically.
CONFIG = {
    "macd_fast":       12,
    "macd_slow":       26,
    "macd_signal":      9,
    "volume_factor":  2.0,    # high volume surge required for breakout
    "breakout_period": 20,    # lookback for high/low breakout
    "rsi_period":      14,
    "rsi_long_min":    50,    # momentum filter: RSI must be above midpoint for longs
    "rsi_short_max":   50,
    "stop_atr":       2.0,
    "init_cash":    1000,
    "fees":         0.0004,
    "leverage":        3,
}

def compute_indicators(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    macd             = ta.trend.MACD(c, window_fast=CONFIG["macd_fast"],
                                     window_slow=CONFIG["macd_slow"],
                                     window_sign=CONFIG["macd_signal"])
    df["macd"]       = macd.macd()
    df["macd_signal"]= macd.macd_signal()
    df["rsi"]        = ta.momentum.RSIIndicator(c, window=CONFIG["rsi_period"]).rsi()
    df["atr"]        = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df["atr_median"] = df["atr"].rolling(100).median()
    df["vol_ma"]     = v.rolling(20).mean()
    df["vol_surge"]  = v > (df["vol_ma"] * CONFIG["volume_factor"])
    df["high_break"] = c > h.rolling(CONFIG["breakout_period"]).max().shift(1)
    df["low_break"]  = c < l.rolling(CONFIG["breakout_period"]).min().shift(1)
    return df

def get_signals(df):
    mup = (df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))
    mdn = (df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))
    df["long_entry"]  = (mup & df["vol_surge"] & df["high_break"]
                         & (df["rsi"] > CONFIG["rsi_long_min"])).astype(bool)
    df["long_exit"]   = (mdn | (df["rsi"] > 80)).astype(bool)
    df["short_entry"] = (mdn & df["vol_surge"] & df["low_break"]
                         & (df["rsi"] < CONFIG["rsi_short_max"])).astype(bool)
    df["short_exit"]  = (mup | (df["rsi"] < 20)).astype(bool)
    df["stop_dist"]   = CONFIG["stop_atr"] * df["atr"]
    return df
# EVOLVE-BLOCK-END'''

BOT_TEMPLATES = {
    "Bot1_Trend":     BOT1,
    "Bot2_Reversion": BOT2,
    "Bot3_Momentum":  BOT3,
}

# =============================================================================
# ── BOT INITIALISATION (creates files only if they don't exist)
# =============================================================================

def initialize_bots():
    print("\nInitialising bots …")
    for bot in BOTS:
        if not bot["path"].exists():
            t = BOT_TEMPLATES.get(bot["name"], "")
            if t:
                bot["path"].parent.mkdir(parents=True, exist_ok=True)
                bot["path"].write_text(t)
                print(f"  Created {bot['name']} at {bot['path']}")
        else:
            print(f"  Found  {bot['name']} — continuing from last state")

# =============================================================================
# ── DATA READINESS CHECK
# Runs download + arrange only for pairs that need it.
# =============================================================================

def ensure_data_ready():
    """Check which pairs need downloading or arranging, then handle them."""
    manifest = load_manifest()
    need_download = []
    need_arrange  = []

    for pair in ALL_PAIRS:
        fname = pair.replace("/", "_") + ".parquet"
        fpath = DATA_DIR / fname
        entry = manifest.get(fname, {})

        if not fpath.exists() or not entry.get("downloaded"):
            need_download.append(pair)
        elif not entry.get("arranged"):
            need_arrange.append(pair)
        else:
            # File exists and was arranged — verify checksum
            try:
                chk = file_checksum(fpath)
                if chk != entry.get("checksum"):
                    need_arrange.append(pair)   # File changed externally
            except:
                need_arrange.append(pair)

    if need_download:
        print(f"\n  {len(need_download)} pairs need downloading: {need_download}")
        download_data(pairs=need_download)

    # After download, refresh for arrange check
    manifest = load_manifest()
    for pair in ALL_PAIRS:
        fname = pair.replace("/", "_") + ".parquet"
        entry = manifest.get(fname, {})
        if not entry.get("arranged") and pair not in need_arrange:
            need_arrange.append(pair)

    if need_arrange:
        print(f"\n  {len(need_arrange)} pairs need arranging: {need_arrange}")
        arrange_data(pairs=need_arrange)

    print("\n  ✓ All data ready.\n")

# =============================================================================
# ── MAIN EVOLUTION LOOP
# =============================================================================

def main():
    print("=" * 60)
    print("CRYPTO FUTURES TRADING BOT EVOLUTION PIPELINE")
    print(f"Capital: ${INIT_CASH}  |  Leverage: {LEVERAGE}x  |  Fees: {FEES}")
    print(f"Walk-forward: train < {TRAIN_END}  |  test >= {TEST_START}")
    print(f"Fast pairs: {len(FAST_PAIRS)}  |  Full pairs: {len(ALL_PAIRS)}")
    print(f"Reason model: {REASON_MODEL}  |  Code model: {CODE_MODEL}")
    print("Press Ctrl+C to stop at any time.")
    print("=" * 60)

    # ── Step 0: Ollama check
    print("\nChecking Ollama …"); check_ollama(); print("Ollama OK ✓\n")

    # ── Step 1: Data readiness (download + arrange only what's missing)
    ensure_data_ready()

    # ── Step 2: Bot file initialisation
    initialize_bots()

    # ── Evolution loop
    cycle       = 1
    best_scores = {bot["name"]: 0.0 for bot in BOTS}

    while True:
        print(f"\n{'=' * 60}\nCYCLE {cycle}\n{'=' * 60}")

        for bot in BOTS:
            print(f"\n{'─' * 50}\n  {bot['name']}\n{'─' * 50}")
            current_code  = read_bot(bot["path"])
            backup_score  = load_backup_score(bot["name"])

            # Baseline (fast)
            print("\n  [Step 0] Baseline (5 pairs fast) …")
            base_score, base_summary, base_pairs = backtest_code(
                current_code, "Baseline", phase="test", fast=True
            )
            update_history(bot["path"], cycle, base_summary)

            # Strategy generation
            print("\n  [Step 1] Generating strategies …")
            strategies = generate_strategies(bot, current_code, base_summary, base_pairs, cycle)
            n = len(strategies)
            print(f"  → {n} strategies")

            # Implementation
            print(f"\n  [Step 2] Implementing {n} variants …")
            variant_codes = []
            for i, strategy in enumerate(strategies):
                print(f"\n  V{i+1}/{n} …")
                code = implement_strategy(bot, current_code, strategy, i + 1)
                variant_codes.append(code)

            # Testing
            print("\n  [Step 3] Two-stage testing …")
            variant_results = test_all_variants(variant_codes, phase="test")

            # Selection
            print("\n  [Step 4] Selecting winner …")
            best_score, best_summary, best_code, best_pairs = pick_best(variant_results)

            # Save if improved
            if best_code and best_code != current_code:
                write_bot(bot["path"], best_code)
                print(f"  ✓ New code saved (score={best_score})")
            else:
                print("  = No code change this cycle")

            # Archive top-3
            if best_code and best_score > 0:
                update_top3(bot["name"], best_code, best_score)

            # Backup best-ever
            if best_score > max(best_scores[bot["name"]], backup_score):
                best_scores[bot["name"]] = best_score
                backup_best(bot["name"], best_code if best_code else current_code, best_score)

            # Regression protection: restore backup if score dropped >15%
            elif (backup_score > 0
                  and best_score < backup_score * 0.85
                  and best_score < best_scores[bot["name"]] * 0.85):
                backup_code, bs = load_backup(bot["name"])
                if backup_code:
                    write_bot(bot["path"], backup_code)
                    print(f"  ↩ Regression detected! Restored backup (score={bs})")

            print(f"\n  All-time best scores: {best_scores}")
            time.sleep(3)

        cycle += 1
        print(f"\n{'=' * 60}\nCycle {cycle - 1} complete! Starting cycle {cycle} …\n{'=' * 60}")
        time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline stopped by user.\nTo resume: python pipeline.py")

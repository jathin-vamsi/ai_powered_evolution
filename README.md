# Crypto Futures Trading Bot Evolution Pipeline

A self-evolving algorithmic trading pipeline that uses two local LLMs (via [Ollama](https://ollama.com)) to automatically improve three competing trading strategies across 15 crypto pairs — all on your own machine, no cloud, no API keys.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Ollama Setup](#ollama-setup)
5. [Running the Pipeline](#running-the-pipeline)
6. [Project Structure](#project-structure)
7. [Configuration Reference](#configuration-reference)
8. [The Three Bots](#the-three-bots)
9. [Understanding the Output](#understanding-the-output)
10. [Customisation Guide](#customisation-guide)
11. [Troubleshooting](#troubleshooting)
12. [Disclaimer](#disclaimer)

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE CYCLE                           │
│                                                             │
│  1. DATA CHECK ──► Download missing pairs (CCXT/Binance)   │
│                    Validate & clean parquet files           │
│                                                             │
│  2. BASELINE   ──► Backtest current bot on 5 fast pairs    │
│                                                             │
│  3. STRATEGISE ──► DeepSeek-R1 reads results & proposes    │
│                    3–7 concrete improvement strategies      │
│                                                             │
│  4. IMPLEMENT  ──► Qwen2.5-Coder writes Python code        │
│                    for each strategy variant                │
│                                                             │
│  5. TEST       ──► Stage 1: screen all variants (5 pairs)  │
│                    Stage 2: full test top 3 (15 pairs)      │
│                                                             │
│  6. SELECT     ──► Tournament picks the best score         │
│                    Regression protection restores backup    │
│                    if performance drops >15%               │
│                                                             │
│  7. REPEAT     ──► Loop forever, bots improve each cycle   │
└─────────────────────────────────────────────────────────────┘
```

Each cycle takes roughly **15–60 minutes** depending on your hardware and chosen models.

---

## System Requirements

### Minimum (will work but slowly)

| Component | Minimum |
|-----------|---------|
| CPU | 4-core (x86-64 or ARM64) |
| RAM | **8 GB** |
| Disk | 20 GB free |
| OS | Ubuntu 20.04+ / Debian 11+ / macOS 12+ |
| Python | 3.10+ |

### Recommended by model size

| RAM Available | Reason Model | Code Model | Speed |
|---------------|--------------|------------|-------|
| 8 GB | `deepseek-r1:1.5b` | `qwen2.5-coder:3b` | Slow |
| 16 GB | `deepseek-r1:7b` | `qwen2.5-coder:7b` | Good ← **default** |
| 32 GB | `deepseek-r1:14b` | `qwen2.5-coder:14b` | Fast |
| 64 GB | `deepseek-r1:32b` | `qwen2.5-coder:32b` | Very fast |

### GPU (optional but highly recommended)

A GPU drastically speeds up LLM inference. Any NVIDIA card with 6 GB+ VRAM will work. AMD and Apple Silicon are also supported by Ollama.

| GPU VRAM | Best model pair |
|----------|----------------|
| 6 GB | 7b models (partial offload) |
| 8 GB | 7b models (full offload) |
| 16 GB | 14b models |
| 24 GB+ | 32b models |

---

## Installation

### 1. Install system packages (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git curl
```

### 2. Clone or download this project

```bash
git clone https://github.com/YOUR_USERNAME/trading-bot-pipeline.git
cd trading-bot-pipeline
```

### 3. Create a Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Python dependencies

```bash
pip install --upgrade pip
pip install pandas numpy vectorbt ta ccxt requests pyarrow
```

Full dependency list:

| Package | Purpose |
|---------|---------|
| `pandas` | DataFrame processing |
| `numpy` | Numerical operations |
| `vectorbt` | Backtesting engine |
| `ta` | Technical indicators (EMA, RSI, MACD, etc.) |
| `ccxt` | Exchange API to download OHLCV data |
| `requests` | HTTP calls to Ollama |
| `pyarrow` | Read/write parquet files |

---

## Ollama Setup

[Ollama](https://ollama.com) runs LLMs locally. It must be running in a separate terminal while the pipeline runs.

### Install Ollama

```bash
# Linux / WSL
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama
```

### Pull the required models

Run these once before starting the pipeline:

```bash
# Default (16 GB RAM)
ollama pull deepseek-r1:7b
ollama pull qwen2.5-coder:7b

# OR for 8 GB RAM
ollama pull deepseek-r1:1.5b
ollama pull qwen2.5-coder:3b

# OR for 32 GB RAM
ollama pull deepseek-r1:14b
ollama pull qwen2.5-coder:14b
```

### Start Ollama server

In a **separate terminal**, keep this running while the pipeline runs:

```bash
ollama serve
```

Verify it works:

```bash
curl http://localhost:11434/api/tags
# Should return a JSON list of installed models
```

---

## Running the Pipeline

### Terminal 1 — Ollama server

```bash
ollama serve
```

### Terminal 2 — Pipeline

```bash
cd trading-bot-pipeline
source venv/bin/activate
python pipeline.py
```

On the **first run**, the pipeline will automatically:
1. Download 15 crypto pairs from Binance (15m candles since 2020)
2. Validate and arrange the parquet files
3. Create the three bot files with starting strategies
4. Begin the evolution loop

**Stop at any time with `Ctrl+C`.**  
**Resume by running `python pipeline.py` again** — all progress is saved.

---

## Project Structure

After the first run, this is what gets created in `~/trading_bot/`:

```
~/trading_bot/
│
├── data/                        ← OHLCV parquet files (auto-downloaded)
│   ├── BTC_USDT.parquet
│   ├── ETH_USDT.parquet
│   ├── ...
│   └── .manifest.json           ← tracks which files are downloaded/arranged
│
├── bot1_trend/
│   └── initial.py               ← Bot1's currently active strategy (auto-evolved)
│
├── bot2_reversion/
│   └── initial.py               ← Bot2's currently active strategy
│
├── bot3_momentum/
│   └── initial.py               ← Bot3's currently active strategy
│
├── best_backups/                ← Best-ever version of each bot (regression safety)
│   ├── Bot1_Trend_best.py
│   ├── Bot1_Trend_score.txt
│   └── ...
│
└── top3_archive/                ← Top 3 historical variants per bot
    ├── Bot1_Trend_top1.py
    ├── Bot1_Trend_top1_score.txt
    └── ...
```

---

## Configuration Reference

All key settings are at the top of `pipeline.py` with inline comments. The most important ones:

### Models — change to match your RAM

```python
REASON_MODEL = "deepseek-r1:7b"       # strategy ideas
CODE_MODEL   = "qwen2.5-coder:7b"     # code generation
```

### Date window — walk-forward split

```python
TRAIN_END  = "2024-01-01"   # data before this = training (not used in default flow)
TEST_START = "2024-01-01"   # data from this date = out-of-sample evaluation
```

### Capital and leverage

```python
INIT_CASH = 1000.0   # paper capital — no real money is used
LEVERAGE  = 3        # futures leverage multiplier
FEES      = 0.0004   # Binance futures taker fee (0.04%)
```

### Risk sizing

```python
BASE_RISK = 0.006   # 0.6% of equity risked per trade minimum
MAX_RISK  = 0.01    # 1.0% of equity risked per trade maximum
STOP_ATR  = 2.0     # stop = 2× ATR(14) away from entry
```

### Data source

```python
TIMEFRAME   = "15m"          # candle interval
SINCE_DATE  = "2020-01-01"   # download history start
EXCHANGE_ID = "binance"      # any CCXT exchange
```

### Scoring weights (inside `calculate_score`)

The score used to rank variants is:

| Component | Weight | Goal |
|-----------|--------|------|
| Sharpe ratio | 40% | > 1.0 |
| Return / Drawdown ratio | 30% | high |
| Win rate | 20% | > 40% |
| Drawdown penalty | −10% | < 30% |

---

## The Three Bots

### Bot 1 — Trend Follower (EMA + ADX)

Follows established trends using dual EMA crossover. Filtered by ADX (trend strength), RSI (momentum), volume confirmation, and a 200-period trend EMA.

Best in: trending markets, breakout periods.
Worst in: choppy, sideways price action.

### Bot 2 — Mean Reverter (Bollinger Bands + RSI)

Buys oversold dips below the lower Bollinger Band and sells overbought spikes above the upper band. RSI and volume confirm the signal.

Best in: ranging, sideways markets.
Worst in: strong trending conditions.

### Bot 3 — Momentum Breakout (MACD + Volume)

Enters on MACD crossover coinciding with a volume surge and a price breakout above/below the 20-period high/low.

Best in: volatile, high-volume breakout events.
Worst in: low-volume, quiet markets.

---

## Understanding the Output

A typical cycle looks like this:

```
============================================================
CYCLE 3
============================================================
──────────────────────────────────────────────────────
  Bot1_Trend
──────────────────────────────────────────────────────

  [Step 0] Baseline (5 pairs fast) …
  BTC/USDT: return=12.4% dd=18.2% wr=44.1% trades=38
  ETH/USDT: return=9.8%  dd=21.5% wr=41.2% trades=32
  ...
  [test_fast] score=0.3821 return=10.1% sharpe=0.91 dd=20.3% wr=43.1%

  [Step 1] Generating strategies …
  DeepSeek: 5 strategies
  S1: Raise ADX threshold to 35, tighten RSI band to 48-72 ...
  S2: Reduce EMA fast to 13, add RSI momentum confirmation ...
  ...

  [Step 2] Implementing 5 variants …
  V1: ready ✓
  V2: ready ✓
  ...

  [Step 3] Two-stage testing …
  Stage 1: Fast test 5 variants (5 pairs) …
  Stage 2: Full test top 3 (15 pairs) …

  [Step 4] Selecting winner …
  Tournament (3 variants):
  V1: score=0.4102 return=13.5% sharpe=1.12 dd=17.8%
  V2: score=0.3944 return=11.2% sharpe=0.98 dd=19.1%
  Winner: V1 score=0.4102

  ✓ New code saved (score=0.4102)
  ★ Backup saved: Bot1_Trend score=0.4102
  Top3 updated: [0.4102, 0.3821, 0.3654]
```

### Score interpretation

| Score | Quality |
|-------|---------|
| < 0.20 | Poor — avoid |
| 0.20–0.35 | Acceptable |
| 0.35–0.50 | Good |
| 0.50–0.65 | Very good |
| > 0.65 | Excellent |

---

## Customisation Guide

### Change models (most common)

In `pipeline.py`, find the model settings section and update:

```python
REASON_MODEL = "deepseek-r1:14b"       # use a bigger reasoning model
CODE_MODEL   = "qwen2.5-coder:14b"     # use a bigger code model
```

Then pull the new models in Ollama:
```bash
ollama pull deepseek-r1:14b
ollama pull qwen2.5-coder:14b
```

### Add a new trading pair

In `pipeline.py`:

```python
ALL_PAIRS = [
    ...
    "NEAR/USDT",    # ← add here
]
```

The pipeline will automatically download data for new pairs on the next run.

### Add a new bot

```python
BOTS = [
    ...
    {
        "name":     "Bot4_MyStrategy",
        "path":     BASE_DIR / "bot4_mystrategy" / "initial.py",
        "strategy": "Your strategy description here"
    },
]
```

Then add a template to `BOT_TEMPLATES`:

```python
BOT_TEMPLATES["Bot4_MyStrategy"] = '''
# ... your initial bot code ...
'''
```

### Change the backtest timeframe

```python
TIMEFRAME = "1h"    # change candle size

# Also update in run_vbt_backtest:
freq = "1h"         # must match
```

### Run on a different exchange

```python
EXCHANGE_ID = "bybit"    # or "okx", "kraken", etc.
```

Note: pair names may differ by exchange (e.g. `BTC/USDT:USDT` on Bybit futures).

---

## Troubleshooting

### "Ollama not running. Waiting 30s…"

Start Ollama in a separate terminal:
```bash
ollama serve
```

### "ccxt not installed"

```bash
pip install ccxt
```

### "parquet not found, run download first"

The data download failed for that pair. Check your internet connection, then re-run. The pipeline will only re-download missing pairs.

### "vectorbt" import error

```bash
pip install vectorbt
```

If VectorBT fails to install due to Numba issues:
```bash
pip install numba==0.57.1
pip install vectorbt
```

### LLM returns garbage / bad format

This is normal occasionally. The pipeline auto-heals bad code and falls back to the previous working version. If it happens every cycle, try a larger model.

### Running out of RAM

Switch to smaller models:
```python
REASON_MODEL = "deepseek-r1:1.5b"
CODE_MODEL   = "qwen2.5-coder:3b"
```

Or reduce the number of pairs tested:
```python
FAST_PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]   # 3 instead of 5
ALL_PAIRS  = FAST_PAIRS                               # use same for full test
```

---

## Disclaimer

This project is for **educational and research purposes only**.

- It does **not** execute any real trades.
- Past backtest performance does **not** guarantee future results.
- Cryptocurrency trading carries significant financial risk.
- Never trade with money you cannot afford to lose.
- Always do your own research before making any financial decisions.

The authors are not responsible for any financial losses incurred from using or adapting this code.

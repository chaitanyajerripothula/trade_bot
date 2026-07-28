import datetime
import time
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

# 1. Fetch Current F&O Universe to ensure zero unmapped symbols
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

home_url = "https://www.nseindia.com/"
pre_open_fo_url = "https://www.nseindia.com/api/market-data-pre-open?key=FO"

# archives.nseindia.com now serves a PDF; nsearchives still returns CSV.
FO_LOT_CSV_URLS = [
    "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv",
    "https://archives.nseindia.com/content/fo/fo_mktlots.csv",
]

INDEX_UNDERLYINGS = {
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "NIFTYNXT50",
    "SYMBOL",
    "UNDERLYING",
}

print("📡 Step 1: Requesting active F&O symbol matrix from NSE...")
session = requests.Session()
session.headers.update(headers)


def _clean_symbols(raw_symbols):
    cleaned = []
    for sym in raw_symbols:
        s = str(sym).strip().upper()
        if not s or s in INDEX_UNDERLYINGS or s.startswith("-") or s.lower() == "nan":
            continue
        cleaned.append(s)
    return sorted(set(cleaned))


def fetch_symbols_from_preopen(sess):
    last_err = None
    for attempt in range(1, 4):
        try:
            sess.get(home_url, timeout=15)
            response = sess.get(pre_open_fo_url, timeout=15)
            if response.status_code != 200:
                raise ConnectionError(f"NSE pre-open HTTP {response.status_code}")
            if "application/json" not in response.headers.get("content-type", "") and not response.text.lstrip().startswith(("{", "[")):
                raise ConnectionError("NSE pre-open returned non-JSON body")
            raw_rows = response.json().get("data", [])
            symbols = _clean_symbols(
                row.get("metadata", {}).get("symbol", "")
                for row in raw_rows
                if row.get("metadata", {}).get("symbol")
            )
            if symbols:
                print(f"✅ NSE pre-open returned {len(symbols)} symbols (attempt {attempt})")
                return symbols
            raise ConnectionError("NSE pre-open returned empty symbol list")
        except Exception as err:
            last_err = err
            print(f"⚠️ NSE pre-open attempt {attempt}/3 failed: {err}")
            time.sleep(2 * attempt)
    raise ConnectionError(last_err)


def fetch_symbols_from_lot_csv(sess):
    last_err = None
    for url in FO_LOT_CSV_URLS:
        try:
            print(f"↩️ Falling back to FO lot CSV: {url}")
            csv_resp = sess.get(url, timeout=20)
            csv_resp.raise_for_status()
            body = csv_resp.content
            if body[:4] == b"%PDF":
                raise ConnectionError("endpoint returned PDF, not CSV")
            text = csv_resp.text
            if "SYMBOL" not in text.upper():
                raise ConnectionError("response missing SYMBOL column")
            lot_df = pd.read_csv(StringIO(text))
            lot_df.columns = [str(c).strip() for c in lot_df.columns]
            col = next((c for c in lot_df.columns if c.upper() == "SYMBOL"), None)
            if col is None:
                raise ConnectionError(f"no SYMBOL column in {list(lot_df.columns)}")
            symbols = _clean_symbols(lot_df[col].tolist())
            if symbols:
                print(f"✅ Lot-size CSV returned {len(symbols)} symbols")
                return symbols
            raise ConnectionError("lot CSV produced zero equity symbols")
        except Exception as err:
            last_err = err
            print(f"⚠️ Lot-size CSV fallback failed for {url}: {err}")
    raise ConnectionError(last_err)


def fetch_fo_symbols(sess):
    """Prefer live pre-open FO API; fall back to NSE FO market-lot CSV."""
    try:
        return fetch_symbols_from_preopen(sess)
    except Exception as preopen_err:
        print(f"⚠️ Pre-open universe unavailable ({preopen_err}); trying lot CSV...")
    try:
        return fetch_symbols_from_lot_csv(sess)
    except Exception as lot_err:
        raise SystemExit(f"🚨 Could not obtain F&O symbol universe: {lot_err}") from lot_err


symbols = fetch_fo_symbols(session)

# 2. Reformat Symbols for Yahoo Finance Mapping (Requires ".NS" suffix)
yf_tickers = [f"{sym}.NS" for sym in symbols]

print(f"Box 📦 Step 2: Collected {len(yf_tickers)} active symbols. Querying historical liquidity vectors...")

# Define a 45-day lookback window to securely capture 20 active trading sessions 
# (Accounting for weekends and exchange holidays)
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=45)

# Batch download historical daily bars in a single optimized network request
historical_data = yf.download(
    tickers=yf_tickers,
    start=start_date,
    end=end_date,
    interval="1d",
    group_by="ticker",
    progress=False
)

# 3. Compute Pure 20-Day Volume Moving Averages
adv_records = []

print("⚙️ Step 3: Engineering mathematical ADV records...")
for sym in symbols:
    ticker_suffix = f"{sym}.NS"
    try:
        # Extract the volume series for the given ticker safely
        if ticker_suffix in historical_data.columns.levels[0]:
            volume_series = historical_data[ticker_suffix]['Volume'].dropna()
            
            # Slice strictly the last 20 completed sessions
            last_20_sessions = volume_series.tail(20)
            
            # Structural safety margin to ensure we have enough data points to build a clean mean
            if len(last_20_sessions) >= 15: 
                calculated_adv = int(last_20_sessions.mean())
                adv_records.append({
                    'Symbol': sym,
                    '20Day_ADV': calculated_adv
                })
    except Exception as err:
        # Prevent individual data parsing anomalies from crashing the entire pipeline
        continue

# 4. Export Cleaned Dataset to Local File System
output_df = pd.DataFrame(adv_records)
output_df.to_csv("fno_adv.csv", index=False)

print(f"\n✅ SUCCESS: 'fno_adv.csv' successfully rebuilt with {len(output_df)} active counters.")
print("Top 5 volume anchors generated:")
print(output_df.head(5).to_string(index=False))
import datetime
import requests
import pandas as pd
import yfinance as yf

# 1. Fetch Current F&O Universe to ensure zero unmapped symbols
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive"
}

home_url = "https://www.nseindia.com/"
pre_open_fo_url = "https://www.nseindia.com/api/market-data-pre-open?key=FO"

print("📡 Step 1: Requesting active F&O symbol matrix from NSE...")
session = requests.Session()
session.headers.update(headers)

FO_LOT_CSV = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"


def fetch_fo_symbols(sess):
    """Try pre-open FO API with retries, then archives lot-size CSV."""
    last_err = None
    for attempt in range(1, 4):
        try:
            sess.get(home_url, timeout=15)
            response = sess.get(pre_open_fo_url, timeout=15)
            if response.status_code != 200:
                raise ConnectionError(f"NSE pre-open HTTP {response.status_code}")
            raw_rows = response.json().get("data", [])
            symbols = [
                row.get("metadata", {}).get("symbol", "").strip()
                for row in raw_rows
                if row.get("metadata", {}).get("symbol")
            ]
            symbols = list(set(filter(None, symbols)))
            if symbols:
                print(f"✅ NSE pre-open returned {len(symbols)} symbols (attempt {attempt})")
                return symbols
            raise ConnectionError("NSE pre-open returned empty symbol list")
        except Exception as err:
            last_err = err
            print(f"⚠️ NSE pre-open attempt {attempt}/3 failed: {err}")
            import time
            time.sleep(2 * attempt)

    # Fallback: official FO market-lot CSV (often reachable when API is blocked)
    try:
        print("↩️ Falling back to archives fo_mktlots.csv ...")
        sess.get(home_url, timeout=15)
        csv_resp = sess.get(FO_LOT_CSV, timeout=20)
        csv_resp.raise_for_status()
        from io import StringIO
        lot_df = pd.read_csv(StringIO(csv_resp.text), skiprows=4)
        # Typical columns include SYMBOL under varying names
        col = next((c for c in lot_df.columns if str(c).strip().upper() == "SYMBOL"), None)
        if col is None:
            col = lot_df.columns[1] if len(lot_df.columns) > 1 else lot_df.columns[0]
        symbols = (
            lot_df[col]
            .astype(str)
            .str.strip()
            .replace({"": None, "nan": None, "Symbol": None, "SYMBOL": None})
            .dropna()
            .unique()
            .tolist()
        )
        # Drop index underlyings that are not equity tickers on Yahoo .NS
        skip = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
        symbols = [s for s in symbols if s and s.upper() not in skip and not s.startswith("-")]
        if symbols:
            print(f"✅ Lot-size CSV returned {len(symbols)} symbols")
            return symbols
    except Exception as err:
        last_err = err
        print(f"⚠️ Lot-size CSV fallback failed: {err}")

    raise SystemExit(f"🚨 Could not obtain F&O symbol universe: {last_err}")


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
"""Shared F&O ADV helpers used by evening harvest and morning monitor."""

from __future__ import annotations

import time
from io import StringIO

import pandas as pd
import requests
import yfinance as yf

NSE_HOME = "https://www.nseindia.com/"
PRE_OPEN_FO_URL = "https://www.nseindia.com/api/market-data-pre-open?key=FO"
FO_LOT_CSV_URLS = (
    "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv",
    "https://archives.nseindia.com/content/fo/fo_mktlots.csv",
)
INDEX_UNDERLYINGS = {
    "NIFTY",
    "BANKNIFTY",
    "FINNIFTY",
    "MIDCPNIFTY",
    "NIFTYNXT50",
    "SYMBOL",
    "UNDERLYING",
}

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


def make_session(headers: dict | None = None) -> requests.Session:
    session = requests.Session()
    session.headers.update(headers or DEFAULT_HEADERS)
    return session


def clean_symbols(raw_symbols) -> list[str]:
    cleaned = []
    for sym in raw_symbols:
        s = str(sym).strip().upper()
        if not s or s in INDEX_UNDERLYINGS or s.startswith("-") or s.lower() == "nan":
            continue
        cleaned.append(s)
    return sorted(set(cleaned))


def fetch_symbols_from_preopen(
    sess: requests.Session,
    *,
    attempts: int = 2,
    timeout: float = 8,
) -> list[str]:
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            sess.get(NSE_HOME, timeout=timeout)
            response = sess.get(PRE_OPEN_FO_URL, timeout=timeout)
            if response.status_code != 200:
                raise ConnectionError(f"NSE pre-open HTTP {response.status_code}")
            text = response.text.lstrip()
            ctype = response.headers.get("content-type", "")
            if "json" not in ctype and not text.startswith(("{", "[")):
                raise ConnectionError("NSE pre-open returned non-JSON body")
            raw_rows = response.json().get("data", [])
            symbols = clean_symbols(
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
            print(f"⚠️ NSE pre-open attempt {attempt}/{attempts} failed: {err}")
            if attempt < attempts:
                time.sleep(attempt)
    raise ConnectionError(last_err)


def fetch_symbols_from_lot_csv(sess: requests.Session, *, timeout: float = 20) -> list[str]:
    last_err: Exception | None = None
    for url in FO_LOT_CSV_URLS:
        try:
            print(f"↩️ FO lot CSV: {url}")
            csv_resp = sess.get(url, timeout=timeout)
            csv_resp.raise_for_status()
            if csv_resp.content[:4] == b"%PDF":
                raise ConnectionError("endpoint returned PDF, not CSV")
            text = csv_resp.text
            if "SYMBOL" not in text.upper():
                raise ConnectionError("response missing SYMBOL column")
            lot_df = pd.read_csv(StringIO(text))
            lot_df.columns = [str(c).strip() for c in lot_df.columns]
            col = next((c for c in lot_df.columns if c.upper() == "SYMBOL"), None)
            if col is None:
                raise ConnectionError(f"no SYMBOL column in {list(lot_df.columns)}")
            symbols = clean_symbols(lot_df[col].tolist())
            if symbols:
                print(f"✅ Lot-size CSV returned {len(symbols)} symbols")
                return symbols
            raise ConnectionError("lot CSV produced zero equity symbols")
        except Exception as err:
            last_err = err
            print(f"⚠️ Lot-size CSV fallback failed for {url}: {err}")
    raise ConnectionError(last_err)


def fetch_fo_symbols(sess: requests.Session, *, prefer_lot_csv: bool = False) -> list[str]:
    """
    Resolve active F&O equity symbols.

    prefer_lot_csv=True skips the blocked NSE pre-open path (saves ~10s on Actions).
    """
    if prefer_lot_csv:
        return fetch_symbols_from_lot_csv(sess)

    try:
        return fetch_symbols_from_preopen(sess)
    except Exception as preopen_err:
        print(f"⚠️ Pre-open universe unavailable ({preopen_err}); trying lot CSV...")
    return fetch_symbols_from_lot_csv(sess)


def _volume_frame(historical_data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Return Date x Symbol volume frame from yfinance multi-ticker download."""
    if historical_data is None or historical_data.empty:
        return pd.DataFrame()

    if isinstance(historical_data.columns, pd.MultiIndex):
        # Prefer xs('Volume') when present; fall back to column filtering.
        try:
            volumes = historical_data.xs("Volume", axis=1, level=-1)
        except KeyError:
            volumes = historical_data.loc[:, historical_data.columns.get_level_values(-1) == "Volume"]
            volumes.columns = volumes.columns.droplevel(-1)
    else:
        # Single-ticker download collapses to flat columns.
        if "Volume" not in historical_data.columns:
            return pd.DataFrame()
        volumes = historical_data[["Volume"]].copy()
        volumes.columns = [tickers[0]]

    # Normalize column labels to bare symbols (strip .NS)
    rename = {}
    for col in volumes.columns:
        name = str(col)
        rename[col] = name[:-3] if name.endswith(".NS") else name
    volumes = volumes.rename(columns=rename)
    return volumes


def compute_20day_adv(symbols: list[str], *, period: str = "1mo") -> pd.DataFrame:
    """
    Batch-download Yahoo bars and vectorize 20-session ADV.

    Uses threads=True and a 1-month window (~20 trading days) instead of a
    slower per-ticker loop over a 45-calendar-day span.
    """
    if not symbols:
        return pd.DataFrame(columns=["Symbol", "20Day_ADV"])

    tickers = [f"{sym}.NS" for sym in symbols]
    print(f"📦 Downloading {len(tickers)} Yahoo tickers (period={period}, threads=True)...")
    historical_data = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )

    volumes = _volume_frame(historical_data, tickers)
    if volumes.empty:
        return pd.DataFrame(columns=["Symbol", "20Day_ADV"])

    last_20 = volumes.tail(20)
    valid = last_20.count() >= 15
    adv = last_20.mean().where(valid).dropna().astype(int)
    adv.index = adv.index.astype(str).str.upper()
    out = (
        adv.rename("20Day_ADV")
        .rename_axis("Symbol")
        .reset_index()
        .sort_values("Symbol")
        .reset_index(drop=True)
    )
    return out

"""Shared F&O ADV helpers used by evening harvest and morning monitor."""

from __future__ import annotations

import random
import time
from io import StringIO

import pandas as pd
import yfinance as yf
from curl_cffi import requests as crequests

NSE_HOME = "https://www.nseindia.com/"
PRE_OPEN_FO_URL = "https://www.nseindia.com/api/market-data-pre-open?key=FO"
PRE_OPEN_PAGE = "https://www.nseindia.com/market-data/pre-open-market-cm-and-emerge-market"
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

# Browser-like headers. TLS fingerprint comes from curl_cffi impersonate="chrome".
DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": PRE_OPEN_PAGE,
    "Origin": "https://www.nseindia.com",
    "Connection": "keep-alive",
}


def make_session(headers: dict | None = None):
    """Chrome-impersonated session — plain requests is often blocked by NSE/Akamai."""
    session = crequests.Session(impersonate="chrome")
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


def _backoff_sleep(attempt: int, *, base: float = 1.0, cap: float = 8.0) -> None:
    # Exponential backoff with jitter: 1s, 2s, 4s... capped.
    delay = min(cap, base * (2 ** (attempt - 1))) + random.uniform(0, 0.4)
    time.sleep(delay)


def _is_rate_limited(status_code: int) -> bool:
    return status_code in {403, 429, 503}


def warm_nse_session(sess, *, timeout: float = 15) -> None:
    """
    Hit NSE home + pre-open page so Akamai cookies (_abck, bm_sz, etc.) are set
    before calling JSON APIs.
    """
    home = sess.get(NSE_HOME, timeout=timeout)
    if home.status_code >= 400:
        raise ConnectionError(f"NSE home warmup HTTP {home.status_code}")
    page = sess.get(PRE_OPEN_PAGE, timeout=timeout)
    if page.status_code >= 400:
        raise ConnectionError(f"NSE pre-open page warmup HTTP {page.status_code}")
    if not sess.cookies:
        raise ConnectionError("NSE warmup returned no cookies — likely blocked")


def fetch_preopen_payload(
    sess,
    *,
    attempts: int = 4,
    timeout: float = 15,
) -> list[dict]:
    """
    Fetch live pre-open FO rows with browser TLS impersonation, cookie warmup,
    and retries on 403/429/503 / non-JSON bodies.
    """
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            warm_nse_session(sess, timeout=timeout)
            # Small pause so cookie jar / bot checks settle
            time.sleep(0.4 + random.uniform(0, 0.3))
            response = sess.get(PRE_OPEN_FO_URL, timeout=timeout)

            if _is_rate_limited(response.status_code):
                raise ConnectionError(
                    f"NSE rate-limit/block HTTP {response.status_code}"
                )
            if response.status_code != 200:
                raise ConnectionError(f"NSE pre-open HTTP {response.status_code}")

            text = (response.text or "").lstrip()
            ctype = response.headers.get("content-type", "")
            if "json" not in ctype and not text.startswith(("{", "[")):
                raise ConnectionError("NSE pre-open returned non-JSON body")

            payload = response.json()
            raw_rows = payload.get("data", [])
            if not raw_rows:
                raise ConnectionError("NSE pre-open returned empty data")
            print(f"✅ NSE pre-open OK ({len(raw_rows)} rows, attempt {attempt})")
            return raw_rows
        except Exception as err:
            last_err = err
            print(f"⚠️ NSE pre-open attempt {attempt}/{attempts} failed: {err}")
            if attempt < attempts:
                _backoff_sleep(attempt)
                # Fresh cookies after hard blocks — stale jars often stick.
                if "403" in str(err) or "429" in str(err) or "cookie" in str(err).lower():
                    try:
                        sess.cookies.clear()
                    except Exception:
                        pass
    raise ConnectionError(last_err)


def fetch_symbols_from_preopen(
    sess,
    *,
    attempts: int = 4,
    timeout: float = 15,
) -> list[str]:
    raw_rows = fetch_preopen_payload(sess, attempts=attempts, timeout=timeout)
    symbols = clean_symbols(
        row.get("metadata", {}).get("symbol", "")
        for row in raw_rows
        if row.get("metadata", {}).get("symbol")
    )
    if not symbols:
        raise ConnectionError("NSE pre-open returned empty symbol list")
    print(f"✅ NSE pre-open returned {len(symbols)} symbols")
    return symbols


def fetch_symbols_from_lot_csv(sess, *, timeout: float = 20) -> list[str]:
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


def fetch_fo_symbols(sess, *, prefer_lot_csv: bool = False) -> list[str]:
    """
    Resolve active F&O equity symbols.

    prefer_lot_csv=True skips the blocked NSE pre-open path (saves time on Actions
    when only a symbol universe is needed for ADV).
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
        try:
            volumes = historical_data.xs("Volume", axis=1, level=-1)
        except KeyError:
            volumes = historical_data.loc[:, historical_data.columns.get_level_values(-1) == "Volume"]
            volumes.columns = volumes.columns.droplevel(-1)
    else:
        if "Volume" not in historical_data.columns:
            return pd.DataFrame()
        volumes = historical_data[["Volume"]].copy()
        volumes.columns = [tickers[0]]

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

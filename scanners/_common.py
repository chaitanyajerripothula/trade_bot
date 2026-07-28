"""Shared helpers for independent pre-open scanners."""

from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

from fo_adv import compute_20day_adv, fetch_preopen_payload, make_session

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587


def require_email_secrets() -> tuple[str, str, str]:
    sender = os.environ.get("SENDER_EMAIL")
    password = os.environ.get("SENDER_PASSWORD")
    receiver = os.environ.get("RECEIVER_EMAIL")
    if not all([sender, password, receiver]):
        raise SystemExit(
            "Missing SENDER_EMAIL, SENDER_PASSWORD, or RECEIVER_EMAIL environment variables."
        )
    return sender, password, receiver


def load_adv_lookup(symbols: list[str]) -> dict:
    if os.path.exists("fno_adv.csv"):
        print("📁 Found existing 'fno_adv.csv'. Loading lookup...")
        adv_database = pd.read_csv("fno_adv.csv")
        keys = adv_database["Symbol"].astype(str).str.strip().str.upper()
        return dict(zip(keys, adv_database["20Day_ADV"]))

    print("⚠️ 'fno_adv.csv' missing — building ADV (slow path)...")
    adv_df = compute_20day_adv(sorted(set(symbols)), period="1mo")
    adv_df.to_csv("fno_adv.csv", index=False)
    print("✅ Fresh 'fno_adv.csv' built for this run.")
    return dict(zip(adv_df["Symbol"].astype(str).str.upper(), adv_df["20Day_ADV"]))


def build_preopen_frame(raw_rows=None) -> pd.DataFrame:
    """Normalize NSE pre-open FO payload into one scanner-ready DataFrame."""
    session = make_session()
    if raw_rows is None:
        raw_rows = fetch_preopen_payload(session)

    records = []
    for row in raw_rows:
        metadata = row.get("metadata", {}) or {}
        detail = (row.get("detail", {}) or {}).get("preOpenMarket", {}) or {}
        symbol = str(metadata.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        buy_qty = pd.to_numeric(detail.get("totalBuyQuantity", 0), errors="coerce")
        sell_qty = pd.to_numeric(detail.get("totalSellQuantity", 0), errors="coerce")
        buy_qty = 0 if pd.isna(buy_qty) else float(buy_qty)
        sell_qty = 0 if pd.isna(sell_qty) else float(sell_qty)
        records.append(
            {
                "Symbol": symbol,
                "IEP_Open": metadata.get("iep", detail.get("IEP", 0)),
                "Prev_Close": metadata.get("previousClose", detail.get("prevClose", 0)),
                "Pct_Chg": metadata.get("pChange", detail.get("perChange", 0)),
                "Matched_Vol": detail.get("totalTradedVolume", metadata.get("finalQuantity", 0)),
                "Buy_Qty": buy_qty,
                "Sell_Qty": sell_qty,
                "Year_High": metadata.get("yearHigh", 0),
                "Year_Low": metadata.get("yearLow", 0),
                "Turnover": metadata.get("totalTurnover", 0),
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise SystemExit("No pre-open rows to evaluate.")

    for col in (
        "IEP_Open",
        "Prev_Close",
        "Pct_Chg",
        "Matched_Vol",
        "Buy_Qty",
        "Sell_Qty",
        "Year_High",
        "Year_Low",
        "Turnover",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    adv_lookup = load_adv_lookup(df["Symbol"].tolist())
    df["20Day_ADV"] = df["Symbol"].map(lambda s: adv_lookup.get(s, 999_999_999))
    df["20Day_ADV"] = pd.to_numeric(df["20Day_ADV"], errors="coerce").fillna(999_999_999)
    df["Footprint_Pct"] = (df["Matched_Vol"] / df["20Day_ADV"].replace(0, pd.NA)) * 100
    total_bs = (df["Buy_Qty"] + df["Sell_Qty"]).replace(0, pd.NA)
    df["Imbalance_Pct"] = ((df["Buy_Qty"] - df["Sell_Qty"]) / total_bs) * 100
    df["Gap_Pct"] = df["Pct_Chg"]
    near_high_den = df["Year_High"].replace(0, pd.NA)
    df["Pct_From_52W_High"] = ((df["IEP_Open"] - df["Year_High"]) / near_high_den) * 100
    near_low_den = df["Year_Low"].replace(0, pd.NA)
    df["Pct_From_52W_Low"] = ((df["IEP_Open"] - df["Year_Low"]) / near_low_den) * 100
    return df


def send_scanner_email(subject: str, title: str, dataframe: pd.DataFrame) -> None:
    sender, password, receiver = require_email_secrets()
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver
    html = f"""
    <html>
      <head>
        <style>
          table {{ border-collapse: collapse; width: 100%; font-family: monospace; }}
          th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
          th {{ background-color: #1a1a1a; color: white; }}
        </style>
      </head>
      <body>
        <h3>{title}</h3>
        {dataframe.to_html(index=False)}
        <br>
        <p><i>trade_bot pre-open scanner</i></p>
      </body>
    </html>
    """
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
    print(f"📨 Sent: {subject} ({len(dataframe)} rows)")


def emit_hits(scanner_name: str, subject: str, hits: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if hits.empty:
        print(f"[{scanner_name}] no hits — suppressing email.")
        return hits
    out = hits.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(2)
    send_scanner_email(subject, scanner_name, out[columns])
    print(f"[{scanner_name}] {len(out)} hits")
    return out

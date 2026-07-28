import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

from fo_adv import compute_20day_adv, fetch_preopen_payload, make_session

# ==========================================
# Secure configuration via env variables
# ==========================================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")


def send_alert_email(dataframe_html: str) -> None:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "SYSTEM OVERRIDE: HIGH CONVICTION PRE-OPEN TARGETS"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    html_content = f"""
    <html>
      <head>
        <style>
          table {{ border-collapse: collapse; width: 100%; font-family: monospace; }}
          th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
          th {{ background-color: #1a1a1a; color: white; }}
        </style>
      </head>
      <body>
        <h3>Live Pre-Market Institutional Crossings Filtered Matrix</h3>
        {dataframe_html}
        <br>
        <p><i>Executed securely via automated GitHub Cloud Architecture.</i></p>
      </body>
    </html>
    """
    msg.attach(MIMEText(html_content, "html"))
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
    print("📨 Transactional Mail dispatched successfully.")


def load_or_build_adv_lookup(raw_rows) -> dict:
    if os.path.exists("fno_adv.csv"):
        print("📁 Found existing 'fno_adv.csv'. Loading lookup...")
        adv_database = pd.read_csv("fno_adv.csv")
        symbols = adv_database["Symbol"].astype(str).str.strip().str.upper()
        return dict(zip(symbols, adv_database["20Day_ADV"]))

    print("⚠️ 'fno_adv.csv' missing — building ADV from pre-open symbols (slow path)...")
    symbols = [
        str(row.get("metadata", {}).get("symbol", "")).strip().upper()
        for row in raw_rows
        if row.get("metadata", {}).get("symbol")
    ]
    symbols = sorted({s for s in symbols if s})
    adv_df = compute_20day_adv(symbols, period="1mo")
    adv_df.to_csv("fno_adv.csv", index=False)
    print("✅ Fresh 'fno_adv.csv' built for this run.")
    return dict(zip(adv_df["Symbol"].astype(str).str.upper(), adv_df["20Day_ADV"]))


if __name__ == "__main__":
    # Fail fast before any network work — secrets empty previously wasted ~15s+.
    if not all([SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL]):
        raise SystemExit(
            "Missing SENDER_EMAIL, SENDER_PASSWORD, or RECEIVER_EMAIL environment variables."
        )

    session = make_session()
    raw_rows = fetch_preopen_payload(session)
    adv_lookup = load_or_build_adv_lookup(raw_rows)

    records = []
    for row in raw_rows:
        metadata = row.get("metadata", {})
        detail = row.get("detail", {}).get("preOpenMarket", {})
        symbol = str(metadata.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        records.append(
            {
                "Symbol": symbol,
                "IEP_Open": metadata.get("iep", 0),
                "Pct_Chg": metadata.get("pChange", 0),
                "Matched_Vol": detail.get("totalTradedVolume", 0),
                "20Day_ADV": adv_lookup.get(symbol, 999_999_999),
            }
        )

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise SystemExit("No pre-open rows to evaluate.")

    for col in ("IEP_Open", "Pct_Chg", "Matched_Vol", "20Day_ADV"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["Footprint_Pct"] = (df["Matched_Vol"] / df["20Day_ADV"].replace(0, pd.NA)) * 100
    high_conviction = (
        df[df["Footprint_Pct"] >= 5.0]
        .sort_values(by="Footprint_Pct", ascending=False)
        .copy()
    )

    if not high_conviction.empty:
        high_conviction["Footprint_Pct"] = high_conviction["Footprint_Pct"].round(2)
        send_alert_email(
            high_conviction[
                ["Symbol", "IEP_Open", "Pct_Chg", "Matched_Vol", "Footprint_Pct"]
            ].to_html(index=False)
        )
    else:
        print("No structural crossovers exceeded the 5% threshold today. Suppressing email transmission.")

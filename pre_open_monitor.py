import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import pandas as pd
import yfinance as yf

# ==========================================
# 🔐 SECURE CONFIGURATION VIA ENV VARIABLES
# ==========================================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
RECEIVER_EMAIL = os.environ.get("RECEIVER_EMAIL")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

def send_alert_email(dataframe_html):
    if not all([SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL]):
        raise SystemExit(
            "Missing SENDER_EMAIL, SENDER_PASSWORD, or RECEIVER_EMAIL environment variables."
        )

    msg = MIMEMultipart('alternative')
    msg['Subject'] = "🔥 SYSTEM OVERRIDE: HIGH CONVICTION PRE-OPEN TARGETS"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL

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
    msg.attach(MIMEText(html_content, 'html'))
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
    print("📨 Transactional Mail dispatched successfully.")

if __name__ == "__main__":
    session = requests.Session()
    session.headers.update(headers)

    # 1. Pull Live Active F&O Tickers from Exchange Match Engine
    try:
        session.get("https://www.nseindia.com/", timeout=10)
        response = session.get("https://www.nseindia.com/api/market-data-pre-open?key=FO", timeout=10)
        if response.status_code != 200:
            raise ConnectionError(f"NSE API rejected connection with code: {response.status_code}")
        
        raw_rows = response.json().get('data', [])
    except Exception as e:
        raise SystemExit(f"🚨 Failed to establish exchange connection handshake: {e}")

    # 2. Structural Guardrail: Validate or Dynamic-Build the ADV Database
    if os.path.exists("fno_adv.csv"):
        print("📁 Found existing 'fno_adv.csv' file. Loading dictionary index...")
        adv_database = pd.read_csv("fno_adv.csv")
        adv_lookup = dict(zip(adv_database['Symbol'].str.strip(), adv_database['20Day_ADV']))
    else:
        print("⚠️ 'fno_adv.csv' is missing from root workspace! Launching dynamic calculation engine...")
        symbols = [row.get('metadata', {}).get('symbol','').strip() for row in raw_rows if row.get('metadata', {}).get('symbol')]
        yf_tickers = [f"{sym}.NS" for sym in symbols if sym]
        
        # Pull daily historical bars to build the missing database frame instantly
        historical_data = yf.download(tickers=yf_tickers, period="40d", interval="1d", group_by="ticker", progress=False)
        
        adv_lookup = {}
        for sym in symbols:
            ticker_suffix = f"{sym}.NS"
            try:
                if ticker_suffix in historical_data.columns.levels[0]:
                    vols = historical_data[ticker_suffix]['Volume'].dropna().tail(20)
                    if len(vols) >= 15:
                        adv_lookup[sym] = int(vols.mean())
            except:
                continue
                
        # Save a local cache backup file to assist consecutive workflow blocks
        pd.DataFrame(list(adv_lookup.items()), columns=['Symbol', '20Day_ADV']).to_csv("fno_adv.csv", index=False)
        print("✅ Fresh 'fno_adv.csv' dynamically built and stored in workspace memory.")

    # 3. Compute Pure Institutional Footprint Ratios
    processed_list = []
    for row in raw_rows:
        metadata = row.get('metadata', {})
        detail = row.get('detail', {}).get('preOpenMarket', {})
        symbol = metadata.get('symbol', '').strip()
        
        # Default to an extreme volume roof if a ticker escapes mapping parameters
        historical_adv = adv_lookup.get(symbol, 999999999)
        
        processed_list.append({
            'Symbol': symbol,
            'IEP_Open': pd.to_numeric(metadata.get('iep', 0)),
            'Pct_Chg': pd.to_numeric(metadata.get('pChange', 0)),
            'Matched_Vol': pd.to_numeric(detail.get('totalTradedVolume', 0)),
            '20Day_ADV': historical_adv
        })
        
    df = pd.DataFrame(processed_list)
    df['Footprint_Pct'] = (df['Matched_Vol'] / df['20Day_ADV']) * 100
    high_conviction = df[df['Footprint_Pct'] >= 5.0].sort_values(by='Footprint_Pct', ascending=False)
    
    # 4. Filter and Transmit Reports
    if not high_conviction.empty:
        high_conviction['Footprint_Pct'] = high_conviction['Footprint_Pct'].round(2)
        send_alert_email(high_conviction[['Symbol', 'IEP_Open', 'Pct_Chg', 'Matched_Vol', 'Footprint_Pct']].to_html(index=False))
    else:
        print("No structural crossovers exceeded the 5% threshold today. Suppressing email transmission.")
import os

from fo_adv import compute_20day_adv, fetch_fo_symbols, make_session

# Evening ADV harvest prefers the FO lot CSV on CI — NSE pre-open is usually
# blocked from GitHub Actions and the retry/backoff wastes ~10s for no gain.
prefer_lot_csv = os.environ.get("PREFER_LOT_CSV", "1") == "1"

print("📡 Step 1: Resolving active F&O symbol universe...")
session = make_session()
symbols = fetch_fo_symbols(session, prefer_lot_csv=prefer_lot_csv)

print(f"⚙️ Step 2–3: Computing 20-day ADV for {len(symbols)} symbols...")
output_df = compute_20day_adv(symbols, period="1mo")
output_df.to_csv("fno_adv.csv", index=False)

print(f"\n✅ SUCCESS: 'fno_adv.csv' rebuilt with {len(output_df)} active counters.")
print("Top 5 volume anchors generated:")
print(output_df.head(5).to_string(index=False))

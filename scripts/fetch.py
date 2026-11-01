"""Tải nến BTC perpetual (ưu tiên OKX) và lưu CSV vào data/.

Thứ tự nguồn: OKX BTC-USDT-SWAP -> Kraken XBT/USD -> Coinbase BTC-USD.
Mỗi file: ts (ms UTC), open, high, low, close, volume, confirm (1 = nến đã đóng).
"""
import csv
import json
import os
import time
import urllib.request
from datetime import datetime, timezone

OUT = os.path.join(os.path.dirname(__file__), "..", "data")
TFS = ["1d", "4h", "1h", "15m", "5m"]
UA = {"User-Agent": "btc-data-fetcher/1.0"}


def get(url, timeout=20):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ---------- OKX ----------
OKX_BAR = {"1d": "1Dutc", "4h": "4H", "1h": "1H", "15m": "15m", "5m": "5m"}


def okx(tf, host="https://www.okx.com"):
    url = f"{host}/api/v5/market/candles?instId=BTC-USDT-SWAP&bar={OKX_BAR[tf]}&limit=300"
    j = get(url)
    if j.get("code") != "0" or not j.get("data"):
        raise RuntimeError(f"OKX lỗi: {j.get('msg')}")
    rows = []
    for c in reversed(j["data"]):
        rows.append([int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[6]), int(c[8])])
    return rows


def okx_extra(host="https://www.okx.com"):
    out = {}
    try:
        f = get(f"{host}/api/v5/public/funding-rate?instId=BTC-USDT-SWAP")["data"][0]
        out["funding_rate"] = float(f["fundingRate"])
        out["next_funding_time"] = int(f["fundingTime"])
    except Exception as e:  # noqa: BLE001
        out["funding_error"] = str(e)
    try:
        o = get(f"{host}/api/v5/public/open-interest?instType=SWAP&instId=BTC-USDT-SWAP")["data"][0]
        out["open_interest_btc"] = float(o["oiCcy"])
    except Exception as e:  # noqa: BLE001
        out["oi_error"] = str(e)
    try:
        h = get(f"{host}/api/v5/rubik/stat/contracts/open-interest-volume?ccy=BTC&period=1H")["data"]
        out["oi_history_1h"] = [[int(x[0]), float(x[1]), float(x[2])] for x in reversed(h[:48])]
    except Exception as e:  # noqa: BLE001
        out["oi_hist_error"] = str(e)
    return out


# ---------- Kraken ----------
KR_INT = {"1d": 1440, "4h": 240, "1h": 60, "15m": 15, "5m": 5}


def kraken(tf):
    j = get(f"https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval={KR_INT[tf]}")
    if j.get("error"):
        raise RuntimeError(f"Kraken lỗi: {j['error']}")
    key = [k for k in j["result"] if k != "last"][0]
    data = j["result"][key][-300:]
    rows = [[int(c[0]) * 1000, float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[6]), 1] for c in data]
    if rows:
        rows[-1][6] = 0  # nến cuối đang chạy
    return rows


# ---------- Coinbase ----------
CB_GRAN = {"1d": 86400, "1h": 3600, "15m": 900, "5m": 300}


def coinbase(tf):
    if tf == "4h":
        h = coinbase("1h")
        rows, bucket = [], None
        for r in h:
            b = r[0] // (4 * 3600_000) * (4 * 3600_000)
            if bucket is None or b != bucket[0]:
                if bucket:
                    rows.append(bucket)
                bucket = [b, r[1], r[2], r[3], r[4], r[5], r[6]]
            else:
                bucket[2] = max(bucket[2], r[2]); bucket[3] = min(bucket[3], r[3])
                bucket[4] = r[4]; bucket[5] += r[5]; bucket[6] = r[6]
        if bucket:
            rows.append(bucket)
        return rows
    data = get(f"https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity={CB_GRAN[tf]}")
    rows = [[int(c[0]) * 1000, float(c[3]), float(c[2]), float(c[1]), float(c[4]), float(c[5]), 1] for c in reversed(data)]
    if rows:
        rows[-1][6] = 0
    return rows


SOURCES = [
    ("OKX BTC-USDT-SWAP", lambda tf: okx(tf)),
    ("OKX BTC-USDT-SWAP (aws)", lambda tf: okx(tf, "https://aws.okx.com")),
    ("Kraken XBT/USD spot", kraken),
    ("Coinbase BTC-USD spot", coinbase),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    errors = []
    for name, fn in SOURCES:
        try:
            got = {tf: fn(tf) for tf in TFS}
            for tf in TFS:
                if len(got[tf]) < 50:
                    raise RuntimeError(f"{tf}: quá ít nến ({len(got[tf])})")
                time.sleep(0.2)
            break
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
            got = None
    if not got:
        raise SystemExit("Không nguồn nào trả dữ liệu:\n" + "\n".join(errors))

    for tf in TFS:
        with open(os.path.join(OUT, f"btc_{tf}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "open", "high", "low", "close", "volume", "confirm"])
            w.writerows(got[tf])

    meta = {
        "source": name,
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_close": got["5m"][-1][4],
        "errors_before_success": errors,
    }
    if name.startswith("OKX"):
        meta.update(okx_extra("https://aws.okx.com" if "aws" in name else "https://www.okx.com"))
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"OK nguồn={name} giá={meta['last_close']}")


if __name__ == "__main__":
    main()

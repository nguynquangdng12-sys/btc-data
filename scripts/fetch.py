"""Tải dữ liệu BTC cho trợ lý phân tích (chạy trên GitHub Actions).

- Nến BTC perpetual (ưu tiên OKX BTC-USDT-SWAP -> Kraken -> Coinbase): data/btc_{tf}.csv
  Mỗi file: ts (ms UTC), open, high, low, close, volume (BTC), confirm (1 = nến đã đóng).
- Nến ETH-USDT-SWAP (để dò SMT BTC/ETH): data/eth_1h.csv, data/eth_15m.csv
- Dấu chân dòng tiền lớn (OKX + Coinbase): data/flow.json
  OI, taker mua/bán (CVD), funding, tỷ lệ long/short top trader, thanh lý, tường lệnh, Coinbase premium.
- Lịch tin kinh tế (ForexFactory, làm mới mỗi 3 giờ): data/calendar.json
Mỗi phần lỗi riêng thì ghi vào "errors", không làm hỏng cả lần chạy.
"""
import csv
import json
import os
import time
import urllib.request
from datetime import datetime, timezone

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
TFS = ["1d", "4h", "1h", "15m", "5m"]
UA = {"User-Agent": "btc-data-fetcher/2.0"}
CT_VAL = 0.01  # 1 hợp đồng BTC-USDT-SWAP = 0,01 BTC


def get(url, timeout=20, tries=2):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1 + k)
    raise last


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- nến
OKX_BAR = {"1d": "1Dutc", "4h": "4H", "1h": "1H", "15m": "15m", "5m": "5m"}


def okx_candles(inst, bar, host="https://www.okx.com", limit=300):
    j = get(f"{host}/api/v5/market/candles?instId={inst}&bar={bar}&limit={limit}")
    if j.get("code") != "0" or not j.get("data"):
        raise RuntimeError(f"OKX lỗi: {j.get('msg')}")
    return [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[6]), int(c[8])]
            for c in reversed(j["data"])]


def okx(tf, host="https://www.okx.com"):
    return okx_candles("BTC-USDT-SWAP", OKX_BAR[tf], host)


KR_INT = {"1d": 1440, "4h": 240, "1h": 60, "15m": 15, "5m": 5}


def kraken(tf):
    j = get(f"https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval={KR_INT[tf]}")
    if j.get("error"):
        raise RuntimeError(f"Kraken lỗi: {j['error']}")
    key = [k for k in j["result"] if k != "last"][0]
    data = j["result"][key][-300:]
    rows = [[int(c[0]) * 1000, float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[6]), 1] for c in data]
    if rows:
        rows[-1][6] = 0
    return rows


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


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume", "confirm"])
        w.writerows(rows)


# ---------------------------------------------------------------- dấu chân dòng tiền (OKX)


def rubik_pages(host, path, params, pages=1, limit=100):
    """Gọi endpoint rubik (mới nhất trước), lật trang bằng 'end'. Trả danh sách hàng tăng dần theo ts."""
    rows, end = [], None
    for _ in range(pages):
        q = f"{host}{path}?{params}&limit={limit}" + (f"&end={end}" if end else "")
        j = get(q)
        if j.get("code") != "0":
            raise RuntimeError(f"{path}: {j.get('msg')}")
        d = j.get("data") or []
        if not d:
            break
        rows += d
        end = min(int(x[0]) for x in d)
        time.sleep(0.35)
    uniq = {int(x[0]): x for x in rows}
    return [uniq[k] for k in sorted(uniq)]


def okx_flow(host, errors):
    f = {}
    inst = "instId=BTC-USDT-SWAP"

    def part(name, fn):
        try:
            f[name] = fn()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
        time.sleep(0.25)

    def funding_now():
        x = get(f"{host}/api/v5/public/funding-rate?{inst}")["data"][0]
        return {"rate": float(x["fundingRate"]), "next_time": int(x["fundingTime"]),
                "next_rate": float(x["nextFundingRate"]) if x.get("nextFundingRate") else None}

    def funding_hist():
        d = get(f"{host}/api/v5/public/funding-rate-history?{inst}&limit=100")["data"]
        return sorted([[int(x["fundingTime"]), float(x.get("realizedRate") or x["fundingRate"])] for x in d])

    def oi_now():
        x = get(f"{host}/api/v5/public/open-interest?instType=SWAP&{inst}")["data"][0]
        return {"oi_btc": float(x["oiCcy"]), "ts": int(x["ts"])}

    def oi_hist(period, pages):
        # [ts, oi (hợp đồng), oiCcy (BTC), oiUsd]
        d = rubik_pages(host, "/api/v5/rubik/stat/contracts/open-interest-history", f"{inst}&period={period}", pages)
        return [[int(x[0]), float(x[2]), float(x[3])] for x in d]

    def taker(period, pages):
        # [ts, sellVol, buyVol] — unit=0: BTC
        d = rubik_pages(host, "/api/v5/rubik/stat/taker-volume-contract", f"{inst}&period={period}&unit=0", pages)
        return [[int(x[0]), float(x[1]), float(x[2])] for x in d]

    def ratio(path, period, pages=1):
        d = rubik_pages(host, f"/api/v5/rubik/stat/contracts/{path}", f"{inst}&period={period}", pages)
        return [[int(x[0]), float(x[1])] for x in d]

    def liquidations():
        out, after = [], None
        cutoff = (time.time() - 72 * 3600) * 1000
        for _ in range(12):
            q = f"{host}/api/v5/public/liquidation-orders?instType=SWAP&instFamily=BTC-USDT&state=filled&limit=100"
            if after:
                q += f"&after={after}"
            j = get(q)
            if j.get("code") != "0":
                raise RuntimeError(j.get("msg"))
            det = []
            for blk in j.get("data") or []:
                if blk.get("instId", "BTC-USDT-SWAP") != "BTC-USDT-SWAP":
                    continue
                det += blk.get("details") or []
            if not det:
                break
            for d in det:
                # posSide long bị thanh lý = lệnh bán cưỡng bức
                out.append([int(d["ts"]), float(d["bkPx"]), float(d["sz"]) * CT_VAL, d.get("posSide") or "", d.get("side") or ""])
            oldest = min(int(d["ts"]) for d in det)
            if oldest < cutoff or after == oldest:
                break
            after = oldest
            time.sleep(0.3)
        uniq = {(r[0], r[1], r[2], r[3]): r for r in out}
        return sorted(uniq.values())

    def book():
        x = get(f"{host}/api/v5/market/books-full?{inst}&sz=5000")["data"][0]
        conv = lambda lv: [[float(p[0]), float(p[1]) * CT_VAL] for p in lv]  # noqa: E731
        return {"ts": int(x["ts"]), "bids": conv(x["bids"]), "asks": conv(x["asks"])}

    part("funding", funding_now)
    part("funding_hist", funding_hist)
    part("oi_now", oi_now)
    part("oi_5m", lambda: oi_hist("5m", 1))
    part("oi_15m", lambda: oi_hist("15m", 2))
    part("oi_1h", lambda: oi_hist("1H", 2))
    part("taker_5m", lambda: taker("5m", 1))
    part("taker_15m", lambda: taker("15m", 2))
    part("taker_1h", lambda: taker("1H", 2))
    part("ls_top_pos_15m", lambda: ratio("long-short-position-ratio-contract-top-trader", "15m", 1))
    part("ls_top_pos_1h", lambda: ratio("long-short-position-ratio-contract-top-trader", "1H", 1))
    part("ls_top_acc_1h", lambda: ratio("long-short-account-ratio-contract-top-trader", "1H", 1))
    part("ls_all_acc_1h", lambda: ratio("long-short-account-ratio-contract", "1H", 1))
    part("liquidations", liquidations)
    part("book", book)
    return f


def coinbase_1h(errors):
    try:
        d = get("https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600")
        return [[int(c[0]) * 1000, float(c[4]), float(c[5])] for c in reversed(d)]  # ts, close, volume
    except Exception as e:  # noqa: BLE001
        errors.append(f"coinbase_1h: {e}")
        return None


# ---------------------------------------------------------------- lịch tin


def calendar(errors, max_age_h=3):
    path = os.path.join(OUT, "calendar.json")
    old = None
    if os.path.exists(path):
        try:
            old = json.load(open(path))
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(old["fetched_utc"].replace("Z", "+00:00"))).total_seconds()
            if age < max_age_h * 3600:
                return
        except Exception:  # noqa: BLE001
            old = None
    events = []
    for wk in ("thisweek", "nextweek"):
        try:
            d = get(f"https://nfs.faireconomy.media/ff_calendar_{wk}.json", tries=1)
            for e in d:
                if e.get("impact") not in ("High", "Medium"):
                    continue
                if e.get("country") not in ("USD", "EUR", "GBP", "JPY", "CNY", "ALL"):
                    continue
                if e.get("impact") == "Medium" and e.get("country") != "USD":
                    continue
                events.append({"title": e.get("title"), "country": e.get("country"), "impact": e.get("impact"),
                               "date": e.get("date"), "forecast": e.get("forecast"), "previous": e.get("previous"),
                               "actual": e.get("actual")})
        except Exception as e:  # noqa: BLE001
            if wk == "thisweek":
                errors.append(f"calendar: {e}")
        time.sleep(0.5)
    if not events and old:
        return
    events.sort(key=lambda e: e["date"] or "")
    json.dump({"source": "ForexFactory", "fetched_utc": now_iso(), "events": events}, open(path, "w"),
              ensure_ascii=False, indent=0)


# ---------------------------------------------------------------- main


def main():
    os.makedirs(OUT, exist_ok=True)
    errors = []
    got = None
    for name, fn in SOURCES:
        try:
            got = {}
            for tf in TFS:
                got[tf] = fn(tf)
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
        write_csv(os.path.join(OUT, f"btc_{tf}.csv"), got[tf])

    meta = {"source": name, "updated_utc": now_iso(), "last_close": got["5m"][-1][4], "errors_before_success": errors}
    flow_err = []
    flow = {"updated_utc": now_iso(), "ct_val_btc": CT_VAL}
    if name.startswith("OKX"):
        host = "https://aws.okx.com" if "aws" in name else "https://www.okx.com"
        for tf, bar in (("1h", "1H"), ("15m", "15m")):
            try:
                write_csv(os.path.join(OUT, f"eth_{tf}.csv"), okx_candles("ETH-USDT-SWAP", bar, host))
            except Exception as e:  # noqa: BLE001
                flow_err.append(f"eth_{tf}: {e}")
            time.sleep(0.2)
        flow.update(okx_flow(host, flow_err))
        # tương thích ngược với meta cũ
        if "funding" in flow:
            meta["funding_rate"] = flow["funding"]["rate"]; meta["next_funding_time"] = flow["funding"]["next_time"]
        if "oi_now" in flow:
            meta["open_interest_btc"] = flow["oi_now"]["oi_btc"]
    else:
        flow_err.append("nguồn nến không phải OKX: bỏ qua dữ liệu dòng tiền OKX")
    flow["coinbase_1h"] = coinbase_1h(flow_err)
    calendar(flow_err)
    flow["errors"] = flow_err
    json.dump(flow, open(os.path.join(OUT, "flow.json"), "w"), separators=(",", ":"))
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), ensure_ascii=False, indent=1)
    sizes = {k: (len(v) if isinstance(v, list) else ("ok" if v else "-")) for k, v in flow.items() if k not in ("errors",)}
    print(f"OK nguồn={name} giá={meta['last_close']}")
    print("flow:", json.dumps(sizes))
    for k in ("oi_5m", "taker_5m", "ls_top_pos_1h", "liquidations", "funding_hist"):
        v = flow.get(k)
        if isinstance(v, list) and v:
            print(f"  {k} mẫu cuối: {v[-1]}")
    if flow_err:
        print("Lỗi phụ:", *flow_err, sep="\n  ")


if __name__ == "__main__":
    main()

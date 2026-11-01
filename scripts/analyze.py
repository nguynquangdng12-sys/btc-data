"""Phân tích BTC theo Sổ tay (ICT/SMC + Wyckoff + Key Volume + Liquidity) và vẽ chart.

Cách dùng:  python scripts/analyze.py [--out out_dir] [--capital 1000]
Đầu ra:     out_dir/chart_h1.png, (nếu có tín hiệu) out_dir/chart_m5.png, out_dir/summary.json
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
VN = timezone(timedelta(hours=7))

# ---------------------------------------------------------------- dữ liệu


def load(tf):
    df = pd.read_csv(os.path.join(DATA, f"btc_{tf}.csv"))
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.reset_index(drop=True)


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ---------------------------------------------------------------- cấu trúc


def swings(df, n):
    """Trả danh sách swing đã xác nhận: dict(i, kind 'H'/'L', price, label)."""
    out = []
    h, l = df["high"].values, df["low"].values
    for i in range(n, len(df) - n):
        if h[i] == h[i - n:i + n + 1].max():
            out.append({"i": i, "kind": "H", "price": float(h[i])})
        if l[i] == l[i - n:i + n + 1].min():
            out.append({"i": i, "kind": "L", "price": float(l[i])})
    out.sort(key=lambda s: s["i"])
    last = {"H": None, "L": None}
    for s in out:
        p = last[s["kind"]]
        if s["kind"] == "H":
            s["label"] = "HH" if p is None or s["price"] > p else "LH"
        else:
            s["label"] = "HL" if p is not None and s["price"] > p else ("LL" if p is not None else "L")
        last[s["kind"]] = s["price"]
    return out


def structure(df, sw, n):
    """Tìm BOS / CHoCH theo giá đóng cửa. Trả (events, trend)."""
    events, trend = [], None
    c = df["close"].values
    pending_h = pending_l = None
    k = 0
    for i in range(len(df)):
        while k < len(sw) and sw[k]["i"] + n <= i:  # swing chỉ dùng được sau khi xác nhận
            if sw[k]["kind"] == "H":
                pending_h = sw[k]
            else:
                pending_l = sw[k]
            k += 1
        if pending_h and c[i] > pending_h["price"]:
            events.append({"type": "BOS" if trend == "up" else "CHoCH", "dir": "up", "level": pending_h["price"],
                           "from": pending_h["i"], "to": i})
            trend, pending_h = "up", None
        elif pending_l and c[i] < pending_l["price"]:
            events.append({"type": "BOS" if trend == "down" else "CHoCH", "dir": "down", "level": pending_l["price"],
                           "from": pending_l["i"], "to": i})
            trend, pending_l = "down", None
    return events, trend


def fvgs(df, min_pct=0.0005):
    out = []
    h, l = df["high"].values, df["low"].values
    for i in range(2, len(df)):
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) / h[i - 2] > min_pct:
            out.append({"dir": "up", "i": i - 1, "bot": float(h[i - 2]), "top": float(l[i])})
        if h[i] < l[i - 2] and (l[i - 2] - h[i]) / h[i] > min_pct:
            out.append({"dir": "down", "i": i - 1, "bot": float(h[i]), "top": float(l[i - 2])})
    for g in out:
        after = df.iloc[g["i"] + 2:]
        g["end"] = len(df) - 1
        g["filled"] = False
        if g["dir"] == "up":
            hit = after.index[after["low"] <= g["bot"]]
        else:
            hit = after.index[after["high"] >= g["top"]]
        if len(hit):
            g["filled"], g["end"] = True, int(hit[0])
        g["mid"] = (g["top"] + g["bot"]) / 2
    return out


def order_blocks(df, events):
    out = []
    o, c = df["open"].values, df["close"].values
    for e in events:
        rng = range(e["to"] - 1, max(e["from"] - 3, 0), -1)
        for j in rng:
            if (e["dir"] == "up" and c[j] < o[j]) or (e["dir"] == "down" and c[j] > o[j]):
                ob = {"dir": e["dir"], "i": j, "bot": float(df["low"][j]), "top": float(df["high"][j]), "event": e["type"]}
                after = df.iloc[e["to"] + 1:]
                broken = (after["close"] < ob["bot"]).any() if e["dir"] == "up" else (after["close"] > ob["top"]).any()
                tested = (after["low"] <= ob["top"]).any() if e["dir"] == "up" else (after["high"] >= ob["bot"]).any()
                ob["valid"], ob["tested"] = not broken, bool(tested)
                out.append(ob)
                break
    return out


def key_volume(df, lookback):
    w = df.iloc[-lookback:]
    i = int(w["volume"].idxmax())
    r = df.loc[i]
    body = abs(r["close"] - r["open"]); rng = r["high"] - r["low"]
    kind = "absorption (râu dài)" if rng > 0 and body / rng < 0.4 else ("đẩy giá tăng" if r["close"] > r["open"] else "đẩy giá giảm")
    after = df.iloc[i + 1:]
    broken = bool(((after["close"] > r["high"]) & (after["volume"] > r["volume"] * 0.8)).any() or
                  ((after["close"] < r["low"]) & (after["volume"] > r["volume"] * 0.8)).any())
    return {"i": i, "top": float(r["high"]), "bot": float(r["low"]), "vol": float(r["volume"]), "kind": kind,
            "broken": broken, "time_vn": r["dt"].astimezone(VN).strftime("%d/%m %H:%M")}


def spikes(df, mult=2.0):
    sma = df["volume"].rolling(20).mean()
    return [int(i) for i in df.index[df["volume"] >= mult * sma]]


def equal_levels(df, sw, kind, tol=0.001):
    pts = [s for s in sw if s["kind"] == kind]
    out = []
    for a in range(len(pts)):
        for b in range(a + 1, len(pts)):
            pa, pb = pts[a]["price"], pts[b]["price"]
            if abs(pa - pb) / pa <= tol and pts[b]["i"] - pts[a]["i"] >= 3:
                lvl = max(pa, pb) if kind == "H" else min(pa, pb)
                after = df.iloc[pts[b]["i"] + 1:]
                swept = bool((after["high"] > lvl).any()) if kind == "H" else bool((after["low"] < lvl).any())
                out.append({"price": lvl, "i": pts[a]["i"], "j": pts[b]["i"], "swept": swept})
    return out


# ---------------------------------------------------------------- phân tích chính


def analyze(capital):
    d1, h4, h1, m15, m5 = (load(t) for t in ["1d", "4h", "1h", "15m", "5m"])
    meta = json.load(open(os.path.join(DATA, "meta.json")))
    price = float(m5["close"].iloc[-1])
    now_utc = m5["dt"].iloc[-1].to_pydatetime()
    now_vn = now_utc.astimezone(VN)

    # D1
    sw_d1 = swings(d1, 2)
    ev_d1, trend_d1 = structure(d1, sw_d1, 2)
    hs = [s for s in sw_d1 if s["kind"] == "H"]; ls = [s for s in sw_d1 if s["kind"] == "L"]
    leg_hi, leg_lo = (hs[-1]["price"] if hs else d1["high"].max()), (ls[-1]["price"] if ls else d1["low"].min())
    eq = (leg_hi + leg_lo) / 2
    zone_pd = "premium" if price > eq else "discount"

    # Thanh khoản
    done_d1 = d1[d1["confirm"] == 1]
    pdh, pdl = float(done_d1["high"].iloc[-1]), float(done_d1["low"].iloc[-1])
    wk = d1.set_index("dt").resample("W-SUN", label="left", closed="left").agg({"high": "max", "low": "min"})
    pwh, pwl = (float(wk["high"].iloc[-2]), float(wk["low"].iloc[-2])) if len(wk) >= 2 else (pdh, pdl)
    today_vn = now_vn.date()
    asia_start = datetime(today_vn.year, today_vn.month, today_vn.day, 7, tzinfo=VN).astimezone(timezone.utc)
    asia_end = asia_start + timedelta(hours=5)
    asia = m5[(m5["dt"] >= asia_start) & (m5["dt"] < asia_end)]
    asia_hi = float(asia["high"].max()) if len(asia) else None
    asia_lo = float(asia["low"].min()) if len(asia) else None
    after_asia = m5[m5["dt"] >= asia_end]

    def swept(level, side, frame):
        if level is None or not len(frame):
            return False
        return bool((frame["high"] > level).any()) if side == "H" else bool((frame["low"] < level).any())

    today_m5 = m5[m5["dt"] >= datetime(today_vn.year, today_vn.month, today_vn.day, tzinfo=VN).astimezone(timezone.utc)]
    liq = [
        {"name": "PDH", "price": pdh, "side": "H", "swept": swept(pdh, "H", today_m5)},
        {"name": "PDL", "price": pdl, "side": "L", "swept": swept(pdl, "L", today_m5)},
        {"name": "PWH", "price": pwh, "side": "H", "swept": bool(d1["high"].iloc[-7:].max() > pwh)},
        {"name": "PWL", "price": pwl, "side": "L", "swept": bool(d1["low"].iloc[-7:].min() < pwl)},
    ]
    if asia_hi:
        liq += [{"name": "Asia H", "price": asia_hi, "side": "H", "swept": swept(asia_hi, "H", after_asia)},
                {"name": "Asia L", "price": asia_lo, "side": "L", "swept": swept(asia_lo, "L", after_asia)}]

    # H1
    sw_h1 = swings(h1, 3)
    ev_h1, trend_h1 = structure(h1, sw_h1, 3)
    fv_h1 = fvgs(h1)
    ob_h1 = order_blocks(h1, ev_h1)
    kv_h1 = key_volume(h1, 72)
    kv_h4 = key_volume(h4, 60)
    sp_h1 = spikes(h1)
    eqh = [e for e in equal_levels(h1, sw_h1, "H") if not e["swept"]]
    eql = [e for e in equal_levels(h1, sw_h1, "L") if not e["swept"]]
    atr_h1 = float(atr(h1).iloc[-1])

    # Fibonacci con sóng H1 gần nhất
    hh = [s for s in sw_h1 if s["kind"] == "H"]; ll = [s for s in sw_h1 if s["kind"] == "L"]
    fib = None
    pair = None
    for ih in range(len(hh) - 1, -1, -1):
        for il in range(len(ll) - 1, -1, -1):
            if abs(hh[ih]["price"] - ll[il]["price"]) >= 3 * float(atr(h1).iloc[-1]):
                pair = (hh[ih], ll[il]); break
        if pair:
            break
    if pair:
        hsw, lsw = pair
        a, b = (lsw, hsw) if lsw["i"] < hsw["i"] else (hsw, lsw)
        up = a["kind"] == "L"
        lo, hi = (a["price"], b["price"]) if up else (b["price"], a["price"])
        lv = {}
        for r in [0, 0.5, 0.618, 0.705, 0.79, 1]:
            lv[str(r)] = hi - (hi - lo) * r if up else lo + (hi - lo) * r
        fib = {"dir": "up" if up else "down", "from_i": a["i"], "to_i": b["i"], "levels": lv,
               "ote": sorted([lv["0.618"], lv["0.79"]])}

    # Chấm điểm vùng (tự động, tối đa 7; điểm vĩ mô +1 do người phân tích cộng thêm)
    bias = trend_d1
    zones = []
    cand = [dict(z, src="FVG") for z in fv_h1 if not z["filled"]] + [dict(z, src="OB") for z in ob_h1 if z["valid"]]
    for z in cand:
        if bias and z["dir"] != bias:
            continue
        s, why = 0, []
        mid = (z["top"] + z["bot"]) / 2
        if bias and ((bias == "up" and mid < eq) or (bias == "down" and mid > eq)):
            s += 2; why.append("đúng hướng D1 + " + ("discount" if bias == "up" else "premium"))
        s += 1; why.append(z["src"] + " H1")
        for kv in (kv_h1, kv_h4):
            if not kv["broken"] and kv["bot"] <= z["top"] and kv["top"] >= z["bot"]:
                s += 1; why.append("trùng Key Volume"); break
        side = "L" if z["dir"] == "up" else "H"
        near = [q for q in liq if q["side"] == side and q["swept"] and abs(q["price"] - mid) / mid < 0.006]
        if near:
            s += 2; why.append("sau quét " + ", ".join(q["name"] for q in near))
        if fib and fib["ote"][0] <= mid <= fib["ote"][1]:
            s += 1; why.append("trong OTE")
        dist = abs(price - mid) / price
        if dist > 0.04:
            continue
        zones.append({"src": z["src"], "dir": z["dir"], "bot": round(z["bot"], 1), "top": round(z["top"], 1),
                      "mid": round(mid, 1), "i": z["i"], "score_auto": s, "why": why, "dist_pct": round(dist * 100, 2)})
    zones.sort(key=lambda z: (-z["score_auto"], z["dist_pct"]))
    zones = zones[:3]

    # Kill zone & tín hiệu M5
    h = now_vn.hour + now_vn.minute / 60
    kz = "London" if 14 <= h < 17 else ("New York" if 19 <= h < 22 else None)
    signal = None
    sw_m5 = swings(m5, 2)
    ev_m5, _ = structure(m5, sw_m5, 2)
    fv_m5 = fvgs(m5, 0.0002)
    atr_m5 = float(atr(m5).iloc[-1])
    if bias:
        recent = m5.iloc[-48:]
        side = "L" if bias == "up" else "H"
        swept_now = [q for q in liq if q["side"] == side and
                     (bool((recent["low"] < q["price"]).any()) if side == "L" else bool((recent["high"] > q["price"]).any()))]
        mss = [e for e in ev_m5 if e["dir"] == bias and e["to"] >= len(m5) - 36]
        if swept_now and mss:
            e = mss[-1]
            body = (m5["close"] - m5["open"]).abs()
            disp = body.iloc[e["to"]] > 1.5 * body.iloc[-50:].mean()
            gaps = [g for g in fv_m5 if g["dir"] == bias and g["i"] >= e["from"] and not g["filled"]]
            if disp and gaps:
                g = gaps[-1]
                entry = g["mid"]
                if bias == "up":
                    sweep_ext = float(recent["low"].min()); sl = sweep_ext - 0.5 * atr_m5
                    targets = sorted([q["price"] for q in liq if q["side"] == "H" and q["price"] > entry and not q["swept"]] +
                                     [x["price"] for x in eqh if x["price"] > entry])
                else:
                    sweep_ext = float(recent["high"].max()); sl = sweep_ext + 0.5 * atr_m5
                    targets = sorted([q["price"] for q in liq if q["side"] == "L" and q["price"] < entry and not q["swept"]] +
                                     [x["price"] for x in eql if x["price"] < entry], reverse=True)
                risk = abs(entry - sl)
                tp1 = entry + 2 * risk if bias == "up" else entry - 2 * risk
                tp2 = targets[0] if targets else None
                if tp2 is not None and abs(tp2 - entry) < 2 * risk:
                    tp2 = targets[1] if len(targets) > 1 else None
                rr2 = abs(tp2 - entry) / risk if tp2 else None
                chased = (bias == "up" and price > entry + 0.5 * (tp1 - entry)) or (bias == "down" and price < entry - 0.5 * (entry - tp1))
                liq_ok = risk / entry < 0.012
                signal = {
                    "dir": "LONG" if bias == "up" else "SHORT", "entry": round(entry, 1), "sl": round(sl, 1),
                    "tp1": round(tp1, 1), "tp2": round(tp2, 1) if tp2 else None, "rr_tp2": round(rr2, 2) if rr2 else None,
                    "risk_pct_price": round(risk / entry * 100, 3), "sl_before_liq_60x": liq_ok, "chased": bool(chased),
                    "swept": [q["name"] for q in swept_now], "mss": e["type"], "mss_level": round(e["level"], 1),
                    "fvg": [round(g["bot"], 1), round(g["top"], 1)], "in_killzone": kz,
                    "size_btc_2pct": round(capital * 0.02 / risk, 4), "size_btc_5pct": round(capital * 0.05 / risk, 4),
                    "m5_from": int(min(e["from"], g["i"]) - 20),
                }
                signal["valid"] = bool(kz and liq_ok and not chased)

    summary = {
        "time_vn": now_vn.strftime("%Y-%m-%d %H:%M"), "source": meta.get("source"), "data_updated_utc": meta.get("updated_utc"),
        "price": round(price, 1), "d1_trend": trend_d1, "d1_last_event": ev_d1[-1] if ev_d1 else None,
        "d1_leg": [round(leg_lo, 1), round(leg_hi, 1)], "equilibrium": round(eq, 1), "premium_discount": zone_pd,
        "h1_trend": trend_h1, "h1_last_events": ev_h1[-3:], "killzone_now": kz,
        "liquidity": [{**q, "price": round(q["price"], 1)} for q in liq],
        "eqh": [round(e["price"], 1) for e in eqh[-3:]], "eql": [round(e["price"], 1) for e in eql[-3:]],
        "key_volume_h1": kv_h1, "key_volume_h4": kv_h4, "fib_h1": fib, "zones": zones,
        "atr_h1": round(atr_h1, 1), "atr_m5": round(atr_m5, 1),
        "funding_rate": meta.get("funding_rate"), "open_interest_btc": meta.get("open_interest_btc"),
        "signal": signal, "capital_assumed": capital,
    }
    ctx = dict(h4_dt=h4["dt"], d1=d1, h1=h1, m5=m5, sw_h1=sw_h1, ev_h1=ev_h1, fv_h1=fv_h1, ob_h1=ob_h1, kv_h1=kv_h1, kv_h4=kv_h4,
               sp_h1=sp_h1, liq=liq, eqh=eqh, eql=eql, fib=fib, zones=zones, sw_m5=sw_m5, ev_m5=ev_m5, fv_m5=fv_m5)
    return summary, ctx


# ---------------------------------------------------------------- vẽ

BG, FG, GRID = "#0f1117", "#d7dae0", "#232733"
UP, DN, BLUE, YEL, PURP, GRAY = "#26a69a", "#ef5350", "#42a5f5", "#ffca28", "#ab47bc", "#8a8f98"


def candles(ax, df, x0):
    for k, r in enumerate(df.itertuples()):
        x = k
        col = UP if r.close >= r.open else DN
        ax.vlines(x, r.low, r.high, color=col, linewidth=0.8)
        ax.add_patch(plt.Rectangle((x - 0.35, min(r.open, r.close)), 0.7, max(abs(r.close - r.open), 1e-6),
                                   color=col, linewidth=0))


def setup(fig_w=16, fig_h=10):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(5, 1, hspace=0.04)
    ax = fig.add_subplot(gs[:4, 0]); av = fig.add_subplot(gs[4, 0], sharex=ax)
    for a in (ax, av):
        a.set_facecolor(BG); a.tick_params(colors=FG); a.grid(color=GRID, linewidth=0.6)
        for s in a.spines.values():
            s.set_color(GRID)
    ax.yaxis.tick_right(); av.yaxis.tick_right()
    return fig, ax, av


def xticks(av, w):
    step = max(len(w) // 10, 1)
    pos = list(range(0, len(w), step))
    av.set_xticks(pos)
    av.set_xticklabels([w["dt"].iloc[p].astimezone(VN).strftime("%d/%m\n%H:%M") for p in pos], color=FG, fontsize=8)


def label(ax, x, y, text, color, ha="left", va="center", size=8.5):
    ax.text(x, y, text, color=color, fontsize=size, ha=ha, va=va,
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec=color, lw=0.6, alpha=0.85), zorder=6)


def place_right(ax, items, x, lo, hi, gap_frac=0.028):
    """items: (y, text, color). Dời nhãn để không chồng nhau, kẻ đường dẫn về mức giá thật."""
    gap = (hi - lo) * gap_frac
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + gap)
    over = ys[-1] - hi if ys else 0
    if over > 0:
        ys = [y - over for y in ys]
        for k in range(len(ys) - 2, -1, -1):
            ys[k] = min(ys[k], ys[k + 1] - gap)
    for (y0, txt, col), y in zip(items, ys):
        if abs(y - y0) > gap * 0.1:
            ax.plot([x - 0.3, x + 0.6], [y0, y], color=col, lw=0.5, alpha=0.7, clip_on=False)
        label(ax, x + 0.8, y, txt, col, size=7.5)


def draw_h1(s, c, path, n=120):
    h1 = c["h1"]; x0 = len(h1) - n; w = h1.iloc[x0:].reset_index(drop=True)
    X = lambda i: max(i - x0, 0)  # noqa: E731
    R = []  # nhãn cột phải
    fig, ax, av = setup()
    ymin, ymax = w["low"].min(), w["high"].max(); pad = (ymax - ymin) * 0.08
    ax.set_ylim(ymin - pad, ymax + pad); ax.set_xlim(-1, n + 20)

    # kill zone
    for k, dt in enumerate(w["dt"]):
        hv = dt.astimezone(VN).hour
        if 14 <= hv < 17:
            ax.axvspan(k - 0.5, k + 0.5, color=BLUE, alpha=0.06, lw=0)
        elif 19 <= hv < 22:
            ax.axvspan(k - 0.5, k + 0.5, color=PURP, alpha=0.06, lw=0)

    # fib
    f = c["fib"]
    if f and f["from_i"] >= x0:
        for r, v in f["levels"].items():
            ax.hlines(v, X(f["from_i"]), n, color=GRAY, lw=0.7, ls=":")
            if r in ("0.5", "0.618", "0.79"):
                R.append((v, f"Fib {r} {v:,.0f}", GRAY))
        ax.add_patch(plt.Rectangle((X(f["from_i"]), f["ote"][0]), n - X(f["from_i"]), f["ote"][1] - f["ote"][0],
                                   color=YEL, alpha=0.07, lw=0))
        eqv = f["levels"]["0.5"]
        ax.text(X(f["from_i"]), eqv, " Premium ↑ / Discount ↓ (OTE vàng)", color=GRAY, fontsize=7.5, va="bottom")

    # FVG
    for g in c["fv_h1"]:
        if g["end"] < x0 or g["filled"]:
            continue
        col = UP if g["dir"] == "up" else DN
        ax.add_patch(plt.Rectangle((X(g["i"]) - 0.5, g["bot"]), n - X(g["i"]) + 0.5, g["top"] - g["bot"],
                                   color=col, alpha=0.14, lw=0))
        ax.hlines(g["mid"], X(g["i"]), n, color=col, lw=0.6, ls="--", alpha=0.7)
        if g["i"] >= x0:
            ax.text(X(g["i"]) + 0.5, g["top"], "FVG", color=col, fontsize=7, va="bottom")

    # OB
    for o in c["ob_h1"]:
        if o["i"] < x0 or not o["valid"]:
            continue
        col = UP if o["dir"] == "up" else DN
        ax.add_patch(plt.Rectangle((X(o["i"]) - 0.5, o["bot"]), n - X(o["i"]) + 0.5, o["top"] - o["bot"],
                                   fill=False, ec=col, lw=1.4))
        ax.text(X(o["i"]), o["bot"], "OB", color=col, fontsize=7.5, va="top", fontweight="bold")

    # Key Volume
    for kv, name in ((c["kv_h1"], "KV H1"), (c["kv_h4"], "KV H4")):
        if kv is c["kv_h4"]:
            t4 = c["h4_dt"][kv["i"]]
            start = int((h1["dt"] >= t4).idxmax()) if (h1["dt"] >= t4).any() else x0
            for v, nm in ((kv["top"], "KV H4 đỉnh"), (kv["bot"], "KV H4 đáy")):
                ax.hlines(v, X(start), n, color=YEL, lw=1.0, ls="-.", alpha=0.8)
                R.append((v, f"{nm} {v:,.0f}" + (" (phá)" if kv["broken"] else ""), YEL))
        elif kv["i"] >= x0:
            ax.add_patch(plt.Rectangle((X(kv["i"]) - 0.5, kv["bot"]), n - X(kv["i"]) + 0.5, kv["top"] - kv["bot"],
                                       color=YEL, alpha=0.16, lw=0))
            label(ax, X(kv["i"]), kv["top"], name, YEL, va="bottom", size=7.5)

    # Liquidity
    for q in c["liq"] + [{"name": "EQH $$$", "price": e["price"], "side": "H", "swept": False} for e in c["eqh"][-2:]] + \
            [{"name": "EQL $$$", "price": e["price"], "side": "L", "swept": False} for e in c["eql"][-2:]]:
        if not (ymin - pad <= q["price"] <= ymax + pad):
            continue
        col = DN if q["side"] == "H" else UP
        ax.hlines(q["price"], 0, n, color=col, lw=0.9, ls=(0, (6, 3)), alpha=0.9 if not q["swept"] else 0.35)
        txt = f"{q['name']} {q['price']:,.0f}" + (" swept" if q["swept"] else "")
        R.append((q["price"], txt, col if not q["swept"] else GRAY))

    # BOS / CHoCH
    for e in c["ev_h1"]:
        if e["from"] < x0:
            continue
        col = UP if e["dir"] == "up" else DN
        ax.hlines(e["level"], X(e["from"]), X(e["to"]), color=col, lw=1.2, ls="-" if e["type"] == "BOS" else "--")
        ax.text((X(e["from"]) + X(e["to"])) / 2, e["level"], e["type"], color=col, fontsize=8, ha="center",
                va="bottom" if e["dir"] == "up" else "top", fontweight="bold")

    candles(ax, w, x0)

    # Swing point
    for sw in c["sw_h1"]:
        if sw["i"] < x0:
            continue
        top = sw["kind"] == "H"
        y = sw["price"] + (pad * 0.25 if top else -pad * 0.25)
        ax.text(X(sw["i"]), y, sw["label"], color=FG, fontsize=7, ha="center", va="bottom" if top else "top")

    # Zone scores
    for z in s["zones"]:
        col = UP if z["dir"] == "up" else DN
        if z["score_auto"] < 4:
            continue
        grade = "A+" if z["score_auto"] >= 6 else "B"
        label(ax, n - 1, z["mid"], f"Vùng {grade} {z['score_auto']}/7", col, ha="right", size=8)

    # giá hiện tại
    ax.axhline(s["price"], color=FG, lw=0.6, alpha=0.6)
    R.append((s["price"], f"Giá {s['price']:,.1f}", FG))
    place_right(ax, [r for r in R if ymin - pad <= r[0] <= ymax + pad], n, ymin - pad, ymax + pad)

    # volume
    sma = h1["volume"].rolling(20).mean().iloc[x0:].values
    cols = [UP if r.close >= r.open else DN for r in w.itertuples()]
    av.bar(range(n), w["volume"], color=cols, width=0.7, alpha=0.6)
    av.plot(range(n), sma, color=GRAY, lw=0.8)
    for i in c["sp_h1"]:
        if i >= x0:
            av.bar(X(i), h1["volume"][i], color=YEL, width=0.7)
    if c["kv_h1"]["i"] >= x0:
        av.text(X(c["kv_h1"]["i"]), h1["volume"][c["kv_h1"]["i"]], "KV", color=YEL, fontsize=7.5, ha="center", va="bottom")
    av.set_ylabel("Volume", color=FG); xticks(av, w)
    plt.setp(ax.get_xticklabels(), visible=False)

    concl = s.get("conclusion", "")
    ax.set_title(f"{s['source']} · H1 · {s['time_vn']} (giờ VN) · D1 {('TĂNG' if s['d1_trend']=='up' else 'GIẢM' if s['d1_trend']=='down' else '?')} · {s['premium_discount']}  {concl}",
                 color=FG, fontsize=12, loc="left")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=FG, lw=1.2, label="BOS (liền) / CHoCH (đứt)"),
               Patch(color=UP, alpha=0.3, label="FVG"), Patch(fill=False, ec=FG, label="Order Block"),
               Patch(color=YEL, alpha=0.4, label="Key Volume / spike"),
               Line2D([], [], color=DN, ls="--", label="Liquidity BSL"), Line2D([], [], color=UP, ls="--", label="Liquidity SSL"),
               Patch(color=BLUE, alpha=0.2, label="KZ London"), Patch(color=PURP, alpha=0.2, label="KZ New York")]
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, facecolor=BG, edgecolor=GRID, labelcolor=FG, ncol=4)
    fig.savefig(path, facecolor=BG, bbox_inches="tight"); plt.close(fig)


def draw_m5(s, c, path):
    sg = s["signal"]; m5 = c["m5"]
    x0 = max(min(sg["m5_from"], len(m5) - 150), 0); w = m5.iloc[x0:].reset_index(drop=True); n = len(w)
    X = lambda i: max(i - x0, 0)  # noqa: E731
    fig, ax, av = setup()
    lo = min(w["low"].min(), sg["sl"]); hi = max(w["high"].max(), sg["tp1"], sg["tp2"] or 0)
    pad = (hi - lo) * 0.06; ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(-1, n + 16)
    for g in c["fv_m5"]:
        if g["i"] >= x0 and not g["filled"]:
            col = UP if g["dir"] == "up" else DN
            ax.add_patch(plt.Rectangle((X(g["i"]) - 0.5, g["bot"]), n - X(g["i"]), g["top"] - g["bot"], color=col, alpha=0.14, lw=0))
    for e in c["ev_m5"]:
        if e["from"] >= x0:
            col = UP if e["dir"] == "up" else DN
            ax.hlines(e["level"], X(e["from"]), X(e["to"]), color=col, lw=1.1, ls="-" if e["type"] == "BOS" else "--")
            ax.text(X(e["to"]), e["level"], " " + e["type"], color=col, fontsize=7.5, va="center")
    for q in c["liq"]:
        if lo - pad <= q["price"] <= hi + pad:
            col = DN if q["side"] == "H" else UP
            ax.hlines(q["price"], 0, n, color=col, lw=0.8, ls=(0, (6, 3)), alpha=0.5)
            label(ax, n + 1, q["price"], q["name"] + (" swept" if q["swept"] else ""), GRAY if q["swept"] else col, size=7.5)
    candles(ax, w, x0)
    for name, v, col in (("Entry", sg["entry"], BLUE), ("SL", sg["sl"], DN), ("TP1 2R", sg["tp1"], UP), ("TP2", sg["tp2"], UP)):
        if v:
            ax.hlines(v, n * 0.55, n + 0.5, color=col, lw=1.6)
            label(ax, n + 1, v, f"{name} {v:,.1f}", col, size=8.5)
    ax.add_patch(plt.Rectangle((n * 0.55, min(sg["entry"], sg["sl"])), n * 0.45, abs(sg["entry"] - sg["sl"]), color=DN, alpha=0.08))
    ax.add_patch(plt.Rectangle((n * 0.55, min(sg["entry"], sg["tp1"])), n * 0.45, abs(sg["tp1"] - sg["entry"]), color=UP, alpha=0.08))
    cols = [UP if r.close >= r.open else DN for r in w.itertuples()]
    av.bar(range(n), w["volume"], color=cols, width=0.7, alpha=0.6); av.set_ylabel("Volume", color=FG); xticks(av, w)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_title(f"{s['source']} · M5 · {s['time_vn']} · {sg['dir']} · quét {', '.join(sg['swept'])} → {sg['mss']} · R:R TP2 {sg['rr_tp2']}",
                 color=FG, fontsize=12, loc="left")
    fig.savefig(path, facecolor=BG, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out"); ap.add_argument("--capital", type=float, default=1000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    s, c = analyze(a.capital)
    sg = s["signal"]
    s["conclusion"] = (f"· TÍN HIỆU {sg['dir']}" if sg and sg["valid"] else "· CHỜ")
    draw_h1(s, c, os.path.join(a.out, "chart_h1.png"))
    if sg:
        draw_m5(s, c, os.path.join(a.out, "chart_m5.png"))
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(s, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()

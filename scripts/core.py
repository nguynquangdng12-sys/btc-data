"""Hàm phân tích kỹ thuật dùng chung: dữ liệu, chỉ báo, cấu trúc ICT/SMC, Key Volume, Volume Profile."""
import json
import os
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
VN = timezone(timedelta(hours=7))


# ---------------------------------------------------------------- dữ liệu


def load(tf, prefix="btc"):
    p = os.path.join(DATA, f"{prefix}_{tf}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.reset_index(drop=True)


def load_json(name, default=None):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return default
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return default


# ---------------------------------------------------------------- chỉ báo


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def vwap_from(df, start_dt):
    """VWAP neo từ thời điểm start_dt (giá điển hình × volume). NaN trước điểm neo."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    m = df["dt"] >= start_dt
    pv = (tp * df["volume"]).where(m, 0).cumsum()
    vv = df["volume"].where(m, 0).cumsum()
    out = pv / vv.replace(0, np.nan)
    return out.where(m)


def vwap_session(df, hour_utc=0):
    """VWAP reset mỗi ngày tại hour_utc (mặc định 00:00 UTC = 07:00 VN, nến ngày OKX)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    key = (df["dt"] - pd.Timedelta(hours=hour_utc)).dt.floor("D")
    pv = (tp * df["volume"]).groupby(key).cumsum()
    vv = df["volume"].groupby(key).cumsum()
    return pv / vv.replace(0, np.nan)


def volume_profile(df, bins=60, va=0.70):
    """Volume Profile: rải volume mỗi nến đều theo high–low. Trả POC, VAH, VAL, HVN, LVN và histogram."""
    lo, hi = float(df["low"].min()), float(df["high"].max())
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    mids = (edges[:-1] + edges[1:]) / 2
    hist = np.zeros(bins)
    for h, l, v in zip(df["high"].values, df["low"].values, df["volume"].values):
        a = np.searchsorted(edges, l, side="right") - 1
        b = np.searchsorted(edges, h, side="left")
        a, b = max(a, 0), min(max(b, a + 1), bins)
        hist[a:b] += v / (b - a)
    poc_i = int(hist.argmax())
    tot, acc = hist.sum(), hist[poc_i]
    lo_i = hi_i = poc_i
    while acc < va * tot and (lo_i > 0 or hi_i < bins - 1):
        down = hist[lo_i - 1] if lo_i > 0 else -1
        up = hist[hi_i + 1] if hi_i < bins - 1 else -1
        if up >= down:
            hi_i += 1; acc += hist[hi_i]
        else:
            lo_i -= 1; acc += hist[lo_i]
    sm = np.convolve(hist, np.ones(3) / 3, mode="same")
    hvn = [float(mids[i]) for i in range(1, bins - 1) if sm[i] > sm[i - 1] and sm[i] >= sm[i + 1] and sm[i] > 1.2 * sm.mean()]
    lvn = [float(mids[i]) for i in range(1, bins - 1) if sm[i] < sm[i - 1] and sm[i] <= sm[i + 1] and sm[i] < 0.6 * sm.mean()]
    return {"poc": round(float(mids[poc_i]), 1), "vah": round(float(edges[hi_i + 1]), 1), "val": round(float(edges[lo_i]), 1),
            "hvn": [round(x, 1) for x in hvn], "lvn": [round(x, 1) for x in lvn],
            "mids": mids.tolist(), "hist": hist.tolist(), "step": float(edges[1] - edges[0])}


# ---------------------------------------------------------------- cấu trúc


def swings(df, n):
    """Swing đã xác nhận: dict(i, kind 'H'/'L', price, label HH/LH/HL/LL)."""
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
    """BOS / CHoCH theo giá đóng cửa. Trả (events, trend)."""
    events, trend = [], None
    c = df["close"].values
    pending_h = pending_l = None
    k = 0
    for i in range(len(df)):
        while k < len(sw) and sw[k]["i"] + n <= i:
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
        hit = after.index[after["low"] <= g["bot"]] if g["dir"] == "up" else after.index[after["high"] >= g["top"]]
        if len(hit):
            g["filled"], g["end"] = True, int(hit[0])
        g["mid"] = (g["top"] + g["bot"]) / 2
    return out


def order_blocks(df, events):
    out = []
    o, c = df["open"].values, df["close"].values
    for e in events:
        for j in range(e["to"] - 1, max(e["from"] - 3, 0), -1):
            if (e["dir"] == "up" and c[j] < o[j]) or (e["dir"] == "down" and c[j] > o[j]):
                ob = {"dir": e["dir"], "i": j, "bot": float(df["low"][j]), "top": float(df["high"][j]), "event": e["type"]}
                after = df.iloc[e["to"] + 1:]
                broken = (after["close"] < ob["bot"]).any() if e["dir"] == "up" else (after["close"] > ob["top"]).any()
                tested = (after["low"] <= ob["top"]).any() if e["dir"] == "up" else (after["high"] >= ob["bot"]).any()
                ob["valid"], ob["tested"] = not bool(broken), bool(tested)
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


def vsa_bars(df, lookback=96, mult=2.0, sw=None):
    """Nến volume đột biến kiểu VSA: hấp thụ (râu dài thân nhỏ) / đẩy giá (thân dài) / cao trào tại đỉnh-đáy."""
    sma = df["volume"].rolling(20).mean()
    out = []
    start = max(len(df) - lookback, 20)
    swing_idx = {s["i"]: s["kind"] for s in (sw or [])}
    for i in range(start, len(df)):
        v = df["volume"].iloc[i]
        if not sma.iloc[i] or v < mult * sma.iloc[i]:
            continue
        o, h, l, c = (float(df[k].iloc[i]) for k in ("open", "high", "low", "close"))
        rng = max(h - l, 1e-9); body = abs(c - o)
        upper, lower = h - max(o, c), min(o, c) - l
        if body / rng < 0.4:
            if lower > upper:
                kind, bias = "hấp thụ bán (râu dưới dài)", 1
            else:
                kind, bias = "hấp thụ mua (râu trên dài)", -1
        else:
            kind, bias = ("đẩy giá tăng" if c > o else "đẩy giá giảm"), (1 if c > o else -1)
        near = [k for k in range(i - 2, i + 3) if k in swing_idx]
        tag = ""
        if near:
            sk = swing_idx[near[0]]
            if sk == "L" and bias > 0:
                tag = "stopping volume tại đáy"
            elif sk == "H" and bias < 0:
                tag = "buying climax/upthrust tại đỉnh"
        out.append({"i": i, "time_vn": df["dt"].iloc[i].astimezone(VN).strftime("%d/%m %H:%M"), "price": round(c, 1),
                    "x_sma": round(float(v / sma.iloc[i]), 1), "kind": kind, "bias": bias, "tag": tag})
    return out


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


def dealing_range(df, sw, k=4):
    """Range giao dịch: đỉnh cao nhất / đáy thấp nhất trong k swing gần nhất mỗi loại."""
    hs = [x["price"] for x in sw if x["kind"] == "H"][-k:]
    ls = [x["price"] for x in sw if x["kind"] == "L"][-k:]
    hi = max(hs) if hs else float(df["high"].iloc[-60:].max())
    lo = min(ls) if ls else float(df["low"].iloc[-60:].min())
    return lo, hi, (lo + hi) / 2


def rsi_divergence(df, sw, r, max_age=40):
    """Phân kỳ RSI giữa 2 swing gần nhất cùng loại (chỉ xét swing trong max_age nến gần nhất)."""
    out = []
    n = len(df)
    for kind in ("H", "L"):
        pts = [s for s in sw if s["kind"] == kind and s["i"] >= n - max_age * 2]
        if len(pts) < 2:
            continue
        a, b = pts[-2], pts[-1]
        if b["i"] < n - max_age:
            continue
        ra, rb = float(r.iloc[a["i"]]), float(r.iloc[b["i"]])
        if kind == "H" and b["price"] > a["price"] and rb < ra - 2:
            out.append({"type": "bearish", "text": "giá HH nhưng RSI LH (phân kỳ giảm)", "i": [a["i"], b["i"]],
                        "price": [a["price"], b["price"]]})
        if kind == "L" and b["price"] < a["price"] and rb > ra + 2:
            out.append({"type": "bullish", "text": "giá LL nhưng RSI HL (phân kỳ tăng)", "i": [a["i"], b["i"]],
                        "price": [a["price"], b["price"]]})
    return out


def smt(btc, eth, sw, window=3, max_age=40):
    """SMT BTC/ETH: so 2 swing gần nhất của BTC với cực trị ETH cùng thời điểm."""
    if eth is None or len(eth) < 50:
        return []
    e = eth.set_index("dt")
    out = []
    n = len(btc)
    for kind in ("H", "L"):
        pts = [s for s in sw if s["kind"] == kind and s["i"] >= n - max_age * 2]
        if len(pts) < 2:
            continue
        a, b = pts[-2], pts[-1]
        if b["i"] < n - max_age:
            continue
        vals = []
        for p in (a, b):
            t0 = btc["dt"].iloc[max(p["i"] - window, 0)]; t1 = btc["dt"].iloc[min(p["i"] + window, n - 1)]
            seg = e.loc[t0:t1]
            if not len(seg):
                vals = None; break
            vals.append(float(seg["high"].max()) if kind == "H" else float(seg["low"].min()))
        if not vals:
            continue
        if kind == "H":
            if b["price"] > a["price"] and vals[1] <= vals[0]:
                out.append({"type": "bearish", "text": "BTC đỉnh cao hơn, ETH không xác nhận (SMT giảm)", "i": [a["i"], b["i"]]})
            elif b["price"] <= a["price"] and vals[1] > vals[0]:
                out.append({"type": "bearish", "text": "ETH đỉnh cao hơn, BTC không theo (SMT giảm)", "i": [a["i"], b["i"]]})
        else:
            if b["price"] < a["price"] and vals[1] >= vals[0]:
                out.append({"type": "bullish", "text": "BTC đáy thấp hơn, ETH giữ đáy (SMT tăng)", "i": [a["i"], b["i"]]})
            elif b["price"] >= a["price"] and vals[1] < vals[0]:
                out.append({"type": "bullish", "text": "ETH đáy thấp hơn, BTC giữ đáy (SMT tăng)", "i": [a["i"], b["i"]]})
    return out

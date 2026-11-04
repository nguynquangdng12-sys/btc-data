"""Phân tích BTC theo Sổ tay (ICT/SMC + Wyckoff + Key Volume + Liquidity) + đa khung + hợp lưu thời gian
+ dấu chân cá mập, rồi vẽ chart.

Cách dùng:  python scripts/analyze.py [--out out] [--capital 1000] [--now 2026-11-05T07:00Z] [--quiet]
Đầu ra (out/):
  chart_m15.png  điểm vào M15 + thời gian + dòng tiền (volume, CVD, OI)
  chart_h1.png   bối cảnh H1 + volume profile + VWAP + Fibonacci + vùng điểm
  chart_mtf.png  đa khung D1/H4/H1/M15
  chart_flow.png dấu chân cá mập (OI, CVD, funding, long/short, thanh lý, Coinbase premium)
  chart_m5.png   (khi có tín hiệu M5)
  summary.json, plan_history.json (theo dõi lệnh chờ qua các lần chạy)
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

import charts
import flowctx
import timectx
from core import (VN, atr, dealing_range, ema, equal_levels, fvgs, key_volume, load, load_json, order_blocks, rsi,
                  rsi_divergence, smt, spikes, structure, swings, volume_profile, vsa_bars, vwap_from, vwap_session)

TF_NAMES = ["D1", "H4", "H1", "M15", "M5"]


# ---------------------------------------------------------------- khung nhỏ: điểm vào


def ltf_plan(df, bias, liq, eqh, eql, capital, kz, n_sw=2, tf="M15"):
    """Tìm điểm vào trên khung nhỏ: tín hiệu đang có (live) hoặc lệnh chờ (plan) tại POI gần nhất."""
    sw = swings(df, n_sw)
    ev, trend = structure(df, sw, n_sw)
    fv = fvgs(df, 0.0003)
    ob = order_blocks(df, ev)
    a = float(atr(df).iloc[-1])
    price = float(df["close"].iloc[-1])
    res = {"tf": tf, "trend": trend, "sw": sw, "ev": ev, "fv": fv, "ob": ob, "atr": round(a, 1), "setup": None}
    rng_lo, rng_hi, eq = dealing_range(df, sw)
    res["range"] = [round(rng_lo, 1), round(rng_hi, 1), round(eq, 1)]
    if not bias:
        return res
    hs = [x for x in sw if x["kind"] == "H"]; ls = [x for x in sw if x["kind"] == "L"]
    if bias == "down":
        tg = sorted([q["price"] for q in liq if q["side"] == "L" and not q["swept"] and q["price"] < price] +
                    [x["price"] for x in eql if x["price"] < price], reverse=True)
    else:
        tg = sorted([q["price"] for q in liq if q["side"] == "H" and not q["swept"] and q["price"] > price] +
                    [x["price"] for x in eqh if x["price"] > price])

    def build(kind, entry, sl, note, poi):
        min_risk = max(a, 0.0025 * entry)  # SL tối thiểu: 1 ATR và 0,25% giá
        if abs(entry - sl) < min_risk:
            sl = entry + min_risk if bias == "down" else entry - min_risk
        risk = abs(entry - sl)
        if risk <= 0:
            return None
        tp1 = entry - 2 * risk if bias == "down" else entry + 2 * risk
        cands = [t for t in tg if abs(t - entry) >= 2 * risk and ((t < entry) if bias == "down" else (t > entry))]
        tp2 = cands[0] if cands else None
        return {"kind": kind, "dir": "SHORT" if bias == "down" else "LONG", "entry": round(entry, 1), "sl": round(sl, 1),
                "tp1": round(tp1, 1), "tp2": round(tp2, 1) if tp2 else None,
                "rr_tp2": round(abs(tp2 - entry) / risk, 2) if tp2 else None,
                "risk_pct_price": round(risk / entry * 100, 3), "sl_before_liq_60x": risk / entry < 0.012,
                "size_btc_2pct": round(capital * 0.02 / risk, 4), "size_btc_5pct": round(capital * 0.05 / risk, 4),
                "contracts_2pct": int(capital * 0.02 / risk / 0.01), "contracts_5pct": int(capital * 0.05 / risk / 0.01),
                "note": note, "poi": poi, "in_killzone": kz, "dist_pct": round(abs(entry - price) / price * 100, 2)}

    # 1) Tín hiệu đang có: quét thanh khoản + MSS bằng nến displacement + FVG
    recent = df.iloc[-48:]
    side = "H" if bias == "down" else "L"
    swept_now = [q["name"] for q in liq if q["side"] == side and
                 (bool((recent["high"] > q["price"]).any()) if side == "H" else bool((recent["low"] < q["price"]).any()))]
    mss = [e for e in ev if e["dir"] == bias and e["to"] >= len(df) - 24]
    if swept_now and mss:
        e = mss[-1]
        body = (df["close"] - df["open"]).abs()
        disp = body.iloc[e["to"]] > 1.5 * body.iloc[-50:].mean()
        gaps = [g for g in fv if g["dir"] == bias and g["i"] >= e["from"] and not g["filled"]]
        if disp and gaps:
            g = gaps[-1]
            ext = float(recent["high"].max()) if bias == "down" else float(recent["low"].min())
            sl = ext + 0.5 * a if bias == "down" else ext - 0.5 * a
            st = build("live", g["mid"], sl, f"quét {', '.join(swept_now)} → {e['type']} {tf} + FVG",
                       [round(g["bot"], 1), round(g["top"], 1)])
            if st:
                chased = (price < st["entry"] - 0.5 * (st["entry"] - st["tp1"])) if bias == "down" else \
                         (price > st["entry"] + 0.5 * (st["tp1"] - st["entry"]))
                st["chased"] = bool(chased)
                st["valid"] = bool(kz and st["sl_before_liq_60x"] and not chased and (st["rr_tp2"] or 0) >= 2)
                st["mss_i"], st["fvg_i"] = e["to"], g["i"]
                res["setup"] = st
                return res

    # 2) Lệnh chờ: POI gần nhất cùng hướng, ở premium (Short) / discount (Long) của range khung nhỏ
    pois = [dict(z, src="FVG") for z in fv if not z["filled"] and z["dir"] == bias] + \
           [dict(z, src="OB") for z in ob if z["valid"] and not z["tested"] and z["dir"] == bias]
    best = None
    for z in pois:
        mid = (z["top"] + z["bot"]) / 2
        if bias == "down" and not (z["bot"] > price and mid > eq):
            continue
        if bias == "up" and not (z["top"] < price and mid < eq):
            continue
        if abs(mid - price) / price > 0.03:
            continue
        if best is None or abs(mid - price) < abs((best["top"] + best["bot"]) / 2 - price):
            best = z
    if best:
        mid = (best["top"] + best["bot"]) / 2
        entry = mid if best["src"] == "FVG" else (best["bot"] if bias == "down" else best["top"])
        if bias == "down":
            prot = [x["price"] for x in hs if x["price"] > best["top"]]
            sl = (min(prot) if prot and min(prot) - best["top"] < 3 * a else best["top"]) + 0.5 * a
        else:
            prot = [x["price"] for x in ls if x["price"] < best["bot"]]
            sl = (max(prot) if prot and best["bot"] - max(prot) < 3 * a else best["bot"]) - 0.5 * a
        st = build("plan", entry, sl, f"lệnh chờ tại {best['src']} {tf} trong vùng " + ("premium" if bias == "down" else "discount"),
                   [round(best["bot"], 1), round(best["top"], 1)])
        if st:
            st["valid"] = False
            st["ok_rr"] = (st["rr_tp2"] or 0) >= 2
            st["poi_i"] = best["i"]
            res["setup"] = st
    return res


# ---------------------------------------------------------------- đa khung


def tf_state(name, df, sw, ev, trend, price, rng=None):
    lo, hi, eq = dealing_range(df, sw) if rng is None else (rng[0], rng[1], (rng[0] + rng[1]) / 2)
    e20, e50 = ema(df["close"], 20), ema(df["close"], 50)
    c = float(df["close"].iloc[-1])
    stack = "up" if c > e20.iloc[-1] > e50.iloc[-1] else ("down" if c < e20.iloc[-1] < e50.iloc[-1] else "mixed")
    pos = (price - lo) / (hi - lo) * 100 if hi > lo else 50
    pd_ = "premium" if price > eq else "discount"
    if pos > 100:
        pd_ = "trên đỉnh range"
    elif pos < 0:
        pd_ = "dưới đáy range"
    last = ev[-1] if ev else None
    return {"tf": name, "trend": trend, "last_event": (f"{last['type']} {'↑' if last['dir'] == 'up' else '↓'} {last['level']:,.0f} "
                                                      f"({df['dt'].iloc[last['to']].astimezone(VN):%d/%m %H:%M})") if last else None,
            "range": [round(lo, 1), round(hi, 1)], "pos_pct": round(min(max(pos, 0), 100), 0), "pd": pd_,
            "ema_stack": stack, "ema20": round(float(e20.iloc[-1]), 1), "ema50": round(float(e50.iloc[-1]), 1),
            "rsi14": round(float(rsi(df["close"]).iloc[-1]), 1)}


def wyckoff(d1, h4, trend_d1, price):
    w = h4.iloc[-90:-15] if len(h4) > 120 else h4.iloc[:-15]
    hi, lo = float(w["high"].max()), float(w["low"].min())
    last = h4.iloc[-15:]
    width = (hi - lo) / ((hi + lo) / 2) * 100
    prior = d1["close"].iloc[-40] if len(d1) > 40 else d1["close"].iloc[0]
    pre_trend = "giảm" if prior > hi else ("tăng" if prior < lo else "đi ngang")
    if last["low"].min() < lo and price > lo:
        ph = "Spring/Test (Phase C) – có thể là cuối tích lũy"
    elif last["high"].max() > hi and price < hi:
        ph = "UTAD/Upthrust (Phase C) – có thể là cuối phân phối"
    elif price > hi and trend_d1 == "up":
        ph = "Markup (Phase D–E) – giá đã thoát range lên"
    elif price < lo and trend_d1 == "down":
        ph = "Markdown – giá đã thoát range xuống"
    elif pre_trend == "giảm":
        ph = "Trong range sau xu hướng giảm (Phase B) – tích lũy hoặc tái phân phối"
    elif pre_trend == "tăng":
        ph = "Trong range sau xu hướng tăng (Phase B) – phân phối hoặc tái tích lũy"
    else:
        ph = "Range (Phase B)"
    return {"phase": ph, "h4_range": [round(lo, 1), round(hi, 1)], "range_width_pct": round(width, 1), "prior_trend": pre_trend,
            "note": "ước lượng tự động từ D1/H4"}


def po3(now, d1, m5, liq, opens):
    o = opens.get("Mở cửa ngày (07:00 VN)")
    if not o:
        return None
    price = float(m5["close"].iloc[-1])
    sess = {q["name"]: q for q in liq if q["name"].split(" ")[0] in ("Á", "London", "NY")}
    t = "ngày đang " + ("TĂNG (giá trên mở cửa)" if price > o else "GIẢM (giá dưới mở cửa)")
    sw_h, sw_l = sess.get("Á H", {}).get("swept"), sess.get("Á L", {}).get("swept")
    if sw_h and sw_l:
        t += "; đã quét cả 2 đầu phiên Á (ngày biến động mạnh)"
    elif sw_h:
        t += ("; đã quét đỉnh phiên Á rồi đảo xuống → PO3 giảm (A: Á, M: quét lên, D: xả xuống)" if price < o
              else "; đã vượt đỉnh phiên Á và giữ trên giá mở cửa → ngày tăng (PO3 tăng nếu đáy ngày đã hình thành)")
    elif sw_l:
        t += ("; đã quét đáy phiên Á rồi đảo lên → PO3 tăng (A: Á, M: quét xuống, D: đẩy lên)" if price > o
              else "; đã phá đáy phiên Á và giữ dưới giá mở cửa → ngày giảm (PO3 giảm nếu đỉnh ngày đã hình thành)")
    elif "Á H" in sess:
        t += "; chưa quét đỉnh/đáy phiên Á"
    swept_names = [k for k, q in sess.items() if q["swept"]]
    if swept_names:
        t += " · đã quét: " + ", ".join(swept_names)
    return {"daily_open": o, "price_vs_open_pct": round((price / o - 1) * 100, 2), "text": t}


# ---------------------------------------------------------------- theo dõi lệnh chờ


def track_plans(hist, setup, now, m5, m15, trend_d1, tctx):
    hist = [h for h in (hist or []) if isinstance(h, dict)]
    if setup:
        dup = [h for h in hist if h["dir"] == setup["dir"] and abs(h["entry"] / setup["entry"] - 1) < 0.0006
               and abs(h["sl"] / setup["sl"] - 1) < 0.0012 and
               (now - datetime.fromisoformat(h["created_utc"])).total_seconds() < 24 * 3600]
        if not dup:
            if tctx["in_window"]:
                exp, win, beg = tctx["window_end_utc"], tctx["in_window"], now.isoformat()
            elif tctx["next_window"]:
                nw = next(x for x in tctx["_intervals"] if x["kind"] == "kz" and x["start"] > now)
                exp, win, beg = nw["end"].isoformat(), nw["name"], nw["start"].isoformat()
            else:
                exp, win, beg = (now + timedelta(hours=4)).isoformat(), None, now.isoformat()
            hist.append({"created_utc": now.isoformat(), "created_vn": now.astimezone(VN).strftime("%d/%m %H:%M"),
                         "dir": setup["dir"], "kind": setup["kind"], "entry": setup["entry"], "sl": setup["sl"],
                         "tp1": setup["tp1"], "tp2": setup["tp2"], "window": win, "start_utc": beg, "expire_utc": exp,
                         "d1": trend_d1})
    hist = hist[-60:]
    out = []
    for h in hist:
        t0 = datetime.fromisoformat(h["created_utc"])
        if (now - t0).total_seconds() > 36 * 3600:
            continue
        short = h["dir"] == "SHORT"
        bars = m5[m5["dt"] >= t0] if t0 >= m5["dt"].iloc[0] else m15[m15["dt"] >= t0]
        q15 = m15[(m15["dt"] >= t0) & (m15["dt"] + pd.Timedelta(minutes=15) <= now)]
        inval = q15[(q15["close"] > h["sl"]) if short else (q15["close"] < h["sl"])]
        t_inval = inval["dt"].iloc[0] + pd.Timedelta(minutes=15) if len(inval) else None
        fill = bars[(bars["high"] >= h["entry"]) if short else (bars["low"] <= h["entry"])]
        t_fill = fill["dt"].iloc[0] if len(fill) else None
        exp = datetime.fromisoformat(h["expire_utc"])
        beg = datetime.fromisoformat(h.get("start_utc") or h["created_utc"])
        st, t_ev = "CHỜ KHỚP" + (f" (chỉ vào trong {h.get('window')})" if now < beg else ""), None
        if t_fill is None or (t_inval is not None and t_inval <= t_fill):
            if t_inval is not None:
                st, t_ev = "HỦY: giá đóng M15 vượt SL trước khi khớp", t_inval
            elif now > exp:
                st, t_ev = "HẾT HẠN: không khớp trong phiên", exp
            elif trend_d1 != h["d1"]:
                st = "HỦY: D1 đổi hướng"
        elif t_fill > exp:
            st, t_ev = "HẾT HẠN: không khớp trong phiên", exp
        elif t_fill < beg:
            st, t_ev = "BỎ: giá chạm entry trước giờ phiên (không vào lệnh)", t_fill
        else:
            after = bars[bars["dt"] >= t_fill]
            st, t_ev = "ĐÃ KHỚP, đang chạy", t_fill
            tp1_hit = False
            for r in after.itertuples():
                hit_sl = r.high >= h["sl"] if short else r.low <= h["sl"]
                hit_tp1 = r.low <= h["tp1"] if short else r.high >= h["tp1"]
                hit_tp2 = h["tp2"] and (r.low <= h["tp2"] if short else r.high >= h["tp2"])
                if hit_sl:
                    st, t_ev = ("ĐẠT TP1 rồi quay về SL" if tp1_hit else "THUA: chạm SL"), r.dt
                    break
                if hit_tp2:
                    st, t_ev = "THẮNG: đạt TP2", r.dt
                    break
                if hit_tp1 and not tp1_hit:
                    tp1_hit = True
                    st, t_ev = "ĐẠT TP1 (2R), đang chờ TP2", r.dt
        out.append({"created_vn": h["created_vn"], "dir": h["dir"], "entry": h["entry"], "sl": h["sl"], "tp1": h["tp1"],
                    "tp2": h["tp2"], "window": h.get("window"), "status": st,
                    "at_vn": t_ev.astimezone(VN).strftime("%d/%m %H:%M") if t_ev is not None else None})
    return hist, out[-6:]


# ---------------------------------------------------------------- phân tích chính


def analyze(capital, now, hist):
    d1, h4, h1, m15, m5 = (load(t) for t in ["1d", "4h", "1h", "15m", "5m"])
    e1h, e15 = load("1h", "eth"), load("15m", "eth")
    if m5["dt"].iloc[-1] > now:  # chạy thử với --now trong quá khứ: cắt dữ liệu tới thời điểm đó
        cut = lambda df: None if df is None else df[df["dt"] <= now].reset_index(drop=True)  # noqa: E731
        d1, h4, h1, m15, m5, e1h, e15 = (cut(x) for x in (d1, h4, h1, m15, m5, e1h, e15))
    meta = load_json("meta.json", {})
    flow = load_json("flow.json", {})
    cal = load_json("calendar.json", {})
    price = float(m5["close"].iloc[-1])

    # ---- D1
    sw_d1 = swings(d1, 2)
    ev_d1, trend_d1 = structure(d1, sw_d1, 2)
    hs = [s for s in sw_d1 if s["kind"] == "H"]; ls = [s for s in sw_d1 if s["kind"] == "L"]
    leg_hi, leg_lo = (hs[-1]["price"] if hs else d1["high"].max()), (ls[-1]["price"] if ls else d1["low"].min())
    eq = (leg_hi + leg_lo) / 2
    zone_pd = "premium" if price > eq else "discount"

    # ---- Thanh khoản
    done_d1 = d1[d1["confirm"] == 1]
    pdh, pdl = float(done_d1["high"].iloc[-1]), float(done_d1["low"].iloc[-1])
    wk = d1.set_index("dt").resample("W-SUN", label="left", closed="left").agg({"high": "max", "low": "min"})
    pwh, pwl = (float(wk["high"].iloc[-2]), float(wk["low"].iloc[-2])) if len(wk) >= 2 else (pdh, pdl)
    now_vn = now.astimezone(VN)

    def swept(level, side, frame):
        if level is None or not len(frame):
            return False
        return bool((frame["high"] > level).any()) if side == "H" else bool((frame["low"] < level).any())

    # PDH/PDL theo nến ngày OKX (00:00 UTC = 07:00 VN): "đã quét" tính từ đầu ngày giao dịch hiện tại
    day0 = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    today_m5 = m5[m5["dt"] >= day0]
    liq = [
        {"name": "PDH", "price": pdh, "side": "H", "swept": swept(pdh, "H", today_m5)},
        {"name": "PDL", "price": pdl, "side": "L", "swept": swept(pdl, "L", today_m5)},
        {"name": "PWH", "price": pwh, "side": "H", "swept": bool(d1["high"].iloc[-7:].max() > pwh)},
        {"name": "PWL", "price": pwl, "side": "L", "swept": bool(d1["low"].iloc[-7:].min() < pwl)},
    ]
    # Đỉnh/đáy các phiên (sáng Á, London, New York) đã kết thúc trong 20 giờ qua
    short_name = {"Phiên sáng (Á)": "Á", "London KZ": "London", "New York KZ": "NY"}
    for iv in timectx.intervals(now - timedelta(hours=26), now):
        if iv["kind"] != "kz" or iv["end"] > now or iv["end"] < now - timedelta(hours=20):
            continue
        seg = m5[(m5["dt"] >= iv["start"]) & (m5["dt"] < iv["end"])]
        if len(seg) < 12:
            continue
        aft = m5[m5["dt"] >= iv["end"]]
        nm = short_name.get(iv["name"], iv["name"])
        hi_, lo_ = float(seg["high"].max()), float(seg["low"].min())
        liq += [{"name": f"{nm} H", "price": hi_, "side": "H", "swept": swept(hi_, "H", aft)},
                {"name": f"{nm} L", "price": lo_, "side": "L", "swept": swept(lo_, "L", aft)}]

    # ---- H4, H1
    sw_h4 = swings(h4, 2); ev_h4, trend_h4 = structure(h4, sw_h4, 2)
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
    fib, pair = None, None
    for ih in range(len(hh) - 1, -1, -1):
        for il in range(len(ll) - 1, -1, -1):
            if abs(hh[ih]["price"] - ll[il]["price"]) >= 3 * atr_h1:
                pair = (hh[ih], ll[il]); break
        if pair:
            break
    if pair:
        hsw, lsw = pair
        a, b = (lsw, hsw) if lsw["i"] < hsw["i"] else (hsw, lsw)
        up = a["kind"] == "L"
        lo, hi = (a["price"], b["price"]) if up else (b["price"], a["price"])
        lv = {str(r): (hi - (hi - lo) * r if up else lo + (hi - lo) * r) for r in [0, 0.5, 0.618, 0.705, 0.79, 1]}
        fib = {"dir": "up" if up else "down", "from_i": a["i"], "to_i": b["i"], "levels": lv,
               "ote": sorted([lv["0.618"], lv["0.79"]])}

    # ---- VWAP, Volume Profile
    vw_day_m15 = vwap_session(m15)
    week0 = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=now.weekday())
    vw_week_h1 = vwap_from(h1, week0)
    anchor_sw = (hs[-1] if trend_d1 == "down" else ls[-1]) if (hs and ls) else None
    avwap = None
    if anchor_sw is not None:
        t_anchor = d1["dt"].iloc[anchor_sw["i"]]
        base = h1 if t_anchor >= h1["dt"].iloc[0] else h4
        avwap = vwap_from(base, t_anchor)
        avwap_val = float(avwap.iloc[-1])
        avwap = {"anchor_vn": t_anchor.astimezone(VN).strftime("%d/%m"), "value": round(avwap_val, 1),
                 "series_h1": vwap_from(h1, t_anchor) if t_anchor >= h1["dt"].iloc[0] else None,
                 "base": "H1" if base is h1 else "H4"}
    vp_h1 = volume_profile(h1.iloc[-120:], bins=70)
    vp_m15 = volume_profile(m15.iloc[-96:], bins=60)
    vwap = {"day": round(float(vw_day_m15.iloc[-1]), 1), "week": round(float(vw_week_h1.iloc[-1]), 1),
            "anchored_d1_leg": {k: v for k, v in avwap.items() if k != "series_h1"} if avwap else None}

    # ---- Thời gian & dòng tiền
    tctx = timectx.context(now, cal, m5, m15, d1)
    kz = tctx["in_window"]
    m15_res = ltf_plan(m15, trend_d1, liq, eqh, eql, capital, kz, 2, "M15")
    fl = flowctx.analyze_flow(flow, m15, h1, m15_res["sw"], now)

    # ---- Vùng H1 (điểm tự động tối đa 7; vĩ mô +1/0/−1 do người phân tích cộng)
    bias = trend_d1
    zones = []
    cand = [dict(z, src="FVG") for z in fv_h1 if not z["filled"]] + [dict(z, src="OB") for z in ob_h1 if z["valid"]]
    extra_lv = {"POC H1": vp_h1["poc"] if vp_h1 else None, "VAH H1": vp_h1["vah"] if vp_h1 else None,
                "VAL H1": vp_h1["val"] if vp_h1 else None, "VWAP tuần": vwap["week"],
                "AVWAP con sóng D1": avwap["value"] if avwap else None, **tctx["opens"]}
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
        pad = 0.001 * mid
        extra = [k for k, v in extra_lv.items() if v and z["bot"] - pad <= v <= z["top"] + pad]
        zones.append({"src": z["src"], "dir": z["dir"], "bot": round(z["bot"], 1), "top": round(z["top"], 1),
                      "mid": round(mid, 1), "i": z["i"], "score_auto": s, "why": why, "extra_confluence": extra,
                      "dist_pct": round(dist * 100, 2)})
    zones.sort(key=lambda z: (-z["score_auto"], z["dist_pct"]))
    zones = zones[:3]

    # ---- Đa khung
    sw_m5 = swings(m5, 2); ev_m5, trend_m5 = structure(m5, sw_m5, 2)
    mtf = [tf_state("D1", d1, sw_d1, ev_d1, trend_d1, price, (leg_lo, leg_hi)), tf_state("H4", h4, sw_h4, ev_h4, trend_h4, price),
           tf_state("H1", h1, sw_h1, ev_h1, trend_h1, price),
           tf_state("M15", m15, m15_res["sw"], m15_res["ev"], m15_res["trend"], price),
           tf_state("M5", m5, sw_m5, ev_m5, trend_m5, price)]
    align = sum(1 for x in mtf if bias and x["trend"] == bias)

    # ---- Phân kỳ, SMT, VSA, Wyckoff, PO3
    div = {"H1": rsi_divergence(h1, sw_h1, rsi(h1["close"])), "M15": rsi_divergence(m15, m15_res["sw"], rsi(m15["close"]))}
    smt_res = {"H1": smt(h1, e1h, sw_h1), "M15": smt(m15, e15, m15_res["sw"])}
    vsa = {"H1": vsa_bars(h1, 48, 2.0, sw_h1)[-5:], "M15": vsa_bars(m15, 96, 2.5, m15_res["sw"])[-6:]}
    wy = wyckoff(d1, h4, trend_d1, price)
    p3 = po3(now, d1, m5, liq, tctx["opens"])

    # ---- Tín hiệu M5 (tham khảo)
    signal = None
    m5_res = ltf_plan(m5, bias, liq, eqh, eql, capital, kz, 2, "M5")
    if m5_res["setup"] and m5_res["setup"]["kind"] == "live":
        signal = m5_res["setup"]

    # ---- Checklist cho lệnh M15
    st = m15_res["setup"]
    checklist = None
    if st:
        dsign = 1 if st["dir"] == "LONG" else -1
        zone_hit = next((z for z in zones if z["bot"] * 0.998 <= st["entry"] <= z["top"] * 1.002), None)
        tscore, twhy = timectx.time_score(tctx, st["dir"], st["entry"])
        agree = lambda lst: [x["text"] for x in lst if (x["type"] == "bullish") == (dsign > 0)]  # noqa: E731
        against = lambda lst: [x["text"] for x in lst if (x["type"] == "bullish") != (dsign > 0)]  # noqa: E731
        sm_ok = agree(smt_res["H1"] + smt_res["M15"]); sm_bad = against(smt_res["H1"] + smt_res["M15"])
        dv_ok = agree(div["H1"] + div["M15"]); dv_bad = against(div["H1"] + div["M15"])
        vw_side = (st["entry"] > vwap["day"]) if dsign < 0 else (st["entry"] < vwap["day"])
        vp_lv = [k for k, v in (("VAH H1", vp_h1 and vp_h1["vah"]), ("POC H1", vp_h1 and vp_h1["poc"]),
                                ("VAL H1", vp_h1 and vp_h1["val"]), ("POC M15", vp_m15 and vp_m15["poc"]))
                 if v and abs(v / st["entry"] - 1) < 0.0015]
        fp = fl["score"] * dsign
        checks = {
            "zone_h1": {"ok": bool(zone_hit and zone_hit["score_auto"] >= 4),
                        "text": (f"trong vùng {zone_hit['src']} H1 {zone_hit['score_auto']}/7" if zone_hit else "entry không nằm trong vùng H1 có điểm")},
            "mtf": {"ok": align >= 3, "text": f"{align}/5 khung cùng hướng D1"},
            "time": {"ok": tscore >= 1, "score": tscore, "text": "; ".join(twhy) or "ngoài cửa sổ phiên"},
            "footprint": {"ok": fp > 0, "score": fl["score"], "text": f"điểm dòng tiền {fl['score']:+d} → " +
                          ("ủng hộ" if fp > 0 else "chống lại" if fp < 0 else "trung tính")},
            "smt_div": {"ok": bool(sm_ok or dv_ok) and not (sm_bad or dv_bad), "text": "; ".join(sm_ok + dv_ok + [f"NGƯỢC: {x}" for x in sm_bad + dv_bad]) or "không có"},
            "pd_d1": {"ok": (zone_pd == "premium") if dsign < 0 else (zone_pd == "discount"),
                      "text": f"giá đang ở {zone_pd} của con sóng D1 (Short nên ở premium, Long ở discount)"},
            "vwap": {"ok": bool(vw_side), "text": ("entry trên" if st["entry"] > vwap["day"] else "entry dưới") + f" VWAP ngày {vwap['day']:,.0f}"},
            "volume_profile": {"ok": bool(vp_lv), "text": ", ".join(vp_lv) or "không trùng mức VP"},
        }
        hard = {
            "d1_direction": True, "rr_tp2_ge_2": (st["rr_tp2"] or 0) >= 2, "risk_lt_1_2pct": st["sl_before_liq_60x"],
            "not_chased": not st.get("chased", False), "no_big_news_30m": not tctx["no_trade_news"],
        }
        checklist = {"passed": sum(1 for v in checks.values() if v["ok"]), "total": len(checks), "checks": checks,
                     "hard_rules": hard, "hard_ok": all(hard.values())}

    hist, tracking = track_plans(hist, st, now, m5, m15, trend_d1, tctx)

    summary = {
        "time_vn": now_vn.strftime("%Y-%m-%d %H:%M"), "source": meta.get("source"), "data_updated_utc": meta.get("updated_utc"),
        "flow_updated_utc": flow.get("updated_utc"), "price": round(price, 1),
        "d1_trend": trend_d1, "d1_last_event": ev_d1[-1] if ev_d1 else None,
        "d1_leg": [round(leg_lo, 1), round(leg_hi, 1)], "equilibrium": round(eq, 1), "premium_discount": zone_pd,
        "h1_trend": trend_h1, "h1_last_events": ev_h1[-3:], "killzone_now": kz,
        "mtf": mtf, "mtf_alignment": f"{align}/5" if bias else None,
        "liquidity": [{**q, "price": round(q["price"], 1)} for q in liq],
        "eqh": [round(e["price"], 1) for e in eqh[-3:]], "eql": [round(e["price"], 1) for e in eql[-3:]],
        "key_volume_h1": kv_h1, "key_volume_h4": kv_h4, "fib_h1": fib, "zones": zones,
        "vwap": vwap, "volume_profile": {"H1_5d": {k: vp_h1[k] for k in ("poc", "vah", "val", "hvn", "lvn")} if vp_h1 else None,
                                         "M15_24h": {k: vp_m15[k] for k in ("poc", "vah", "val")} if vp_m15 else None},
        "rsi_divergence": div, "smt": smt_res, "vsa": vsa, "wyckoff": wy, "po3": p3,
        "time": {k: v for k, v in tctx.items() if not k.startswith("_")},
        "footprint": {"score": fl["score"], "signals": fl["signals"], "info": fl["info"]},
        "atr_h1": round(atr_h1, 1), "atr_m5": m5_res["atr"],
        "funding_rate": (flow.get("funding") or {}).get("rate", meta.get("funding_rate")),
        "open_interest_btc": (flow.get("oi_now") or {}).get("oi_btc", meta.get("open_interest_btc")),
        "signal": signal, "capital_assumed": capital,
        "m15_trend": m15_res["trend"], "m15_range": m15_res.get("range"), "m15_setup": st, "m15_checklist": checklist,
        "m15_last_events": m15_res["ev"][-3:], "plan_tracking": tracking,
    }
    ctx = dict(d1=d1, h4=h4, h1=h1, m15=m15, m5=m5, e1h=e1h, e15=e15, sw_d1=sw_d1, ev_d1=ev_d1, sw_h4=sw_h4, ev_h4=ev_h4,
               sw_h1=sw_h1, ev_h1=ev_h1, fv_h1=fv_h1, ob_h1=ob_h1, kv_h1=kv_h1, kv_h4=kv_h4, sp_h1=sp_h1, liq=liq,
               eqh=eqh, eql=eql, fib=fib, zones=zones, m15_res=m15_res, m5_res=m5_res, sw_m5=sw_m5, ev_m5=ev_m5,
               vw_day_m15=vw_day_m15, vw_week_h1=vw_week_h1, avwap=avwap, vp_h1=vp_h1, vp_m15=vp_m15, tctx=tctx,
               flow=fl, flow_raw=flow, div=div, smt=smt_res, vsa=vsa, mtf=mtf, eq_d1=eq)
    return summary, ctx, hist


def conclusion(s):
    st, sg = s["m15_setup"], s["signal"]
    if st and st["kind"] == "live":
        return f"TÍN HIỆU {st['dir']} M15" + ("" if st.get("valid") else " (CHƯA HỢP LỆ)")
    if st and st["kind"] == "plan":
        return f"LỆNH CHỜ {st['dir']} M15"
    if sg and sg.get("valid"):
        return f"TÍN HIỆU {sg['dir']} M5 (tham khảo)"
    return "CHỜ"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out"); ap.add_argument("--capital", type=float, default=1000)
    ap.add_argument("--now", help="giờ UTC để thử, ví dụ 2026-11-05T07:00Z"); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    now = datetime.fromisoformat(a.now.replace("Z", "+00:00")) if a.now else datetime.now(timezone.utc)
    hp = os.path.join(a.out, "plan_history.json")
    hist = json.load(open(hp)) if os.path.exists(hp) else []
    s, c, hist = analyze(a.capital, now, hist)
    s["conclusion"] = conclusion(s)
    charts.draw_all(s, c, a.out)
    json.dump(hist, open(hp, "w"), ensure_ascii=False, indent=0)
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1, default=str)
    if not a.quiet:
        print(json.dumps(s, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()

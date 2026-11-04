"""Dấu chân cá mập (dòng tiền lớn) từ dữ liệu OKX + Coinbase trong data/flow.json.

Mỗi tín hiệu có dấu: +1 ủng hộ tăng, -1 ủng hộ giảm, 0 trung tính.
"""
import numpy as np
import pandas as pd

from core import VN


def _df(rows, cols):
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=cols)
    d["dt"] = pd.to_datetime(d["ts"], unit="ms", utc=True)
    return d.sort_values("dt").reset_index(drop=True)


def _pct(a, b):
    return (a / b - 1) * 100 if b else 0.0


def _price_at(px, dt):
    k = px["dt"].searchsorted(dt, side="right") - 1
    return float(px["close"].iloc[max(k, 0)])


def analyze_flow(flow, m15, h1, sw_m15, now):
    sig, info, series = [], {}, {}
    if not flow:
        return {"signals": [], "score": 0, "info": {"note": "không có dữ liệu dòng tiền"}, "series": {}}
    price = float(m15["close"].iloc[-1])

    # 1) OI + giá
    oi = _df(flow.get("oi_15m"), ["ts", "oi_btc", "oi_usd"])
    if oi is not None and len(oi) > 20:
        series["oi_15m"] = oi
        oi1h = _df(flow.get("oi_1h"), ["ts", "oi_btc", "oi_usd"])
        if oi1h is not None:
            series["oi_1h"] = oi1h
        rows = {}
        for lab, bars in (("1h", 4), ("4h", 16), ("24h", 96)):
            if len(oi) <= bars:
                continue
            t_now, t_old = oi["dt"].iloc[-1], oi["dt"].iloc[-1 - bars]
            doi = _pct(oi["oi_btc"].iloc[-1], oi["oi_btc"].iloc[-1 - bars])
            dp = _pct(_price_at(m15, t_now + pd.Timedelta(minutes=14)), _price_at(m15, t_old + pd.Timedelta(minutes=14)))
            if dp > 0.3 and doi > 0.5:
                reg, s = "giá↑ OI↑: long mới vào, tiền mới đẩy giá", 1
            elif dp > 0.3 and doi < -0.5:
                reg, s = "giá↑ OI↓: short đóng lệnh (short cover), đà tăng kém bền", 0
            elif dp < -0.3 and doi > 0.5:
                reg, s = "giá↓ OI↑: short mới vào, tiền mới đẩy giá xuống", -1
            elif dp < -0.3 and doi < -0.5:
                reg, s = "giá↓ OI↓: long xả/bị thanh lý, đà giảm kém bền", 0
            elif doi > 1.0:
                reg, s = "giá đi ngang, OI↑: vị thế tích tụ, sắp có cú quét", 0
            elif doi < -1.0:
                reg, s = "giá đi ngang, OI↓: vị thế rút ra", 0
            else:
                reg, s = "OI ít thay đổi", 0
            rows[lab] = {"d_price_pct": round(dp, 2), "d_oi_pct": round(doi, 2), "regime": reg, "sign": s}
        info["oi"] = rows
        info["oi_btc_now"] = round(float(oi["oi_btc"].iloc[-1]), 0)
        if "4h" in rows:
            sig.append({"name": "OI 4h", "sign": rows["4h"]["sign"],
                        "text": f"OI {rows['4h']['d_oi_pct']:+.2f}%, giá {rows['4h']['d_price_pct']:+.2f}% → {rows['4h']['regime']}"})

    # 2) Taker mua/bán chủ động (CVD)
    tk = _df(flow.get("taker_15m"), ["ts", "sell", "buy"])
    if tk is not None and len(tk) > 20:
        tk["delta"] = tk["buy"] - tk["sell"]
        tk["cvd"] = tk["delta"].cumsum()
        series["taker_15m"] = tk
        tk1 = _df(flow.get("taker_1h"), ["ts", "sell", "buy"])
        if tk1 is not None:
            tk1["delta"] = tk1["buy"] - tk1["sell"]; tk1["cvd"] = tk1["delta"].cumsum()
            series["taker_1h"] = tk1
        d4 = float(tk["delta"].iloc[-16:].sum()); d24 = float(tk["delta"].iloc[-96:].sum())
        info["taker"] = {"delta_4h_btc": round(d4, 1), "delta_24h_btc": round(d24, 1),
                         "buy_ratio_4h": round(float(tk["buy"].iloc[-16:].sum() / max((tk["buy"] + tk["sell"]).iloc[-16:].sum(), 1e-9)), 3)}
        thr = 2 * float(tk["delta"].abs().iloc[-96:].median())  # ròng 4h phải lớn hơn ~2 nến 15m điển hình
        s = 1 if d4 > thr else (-1 if d4 < -thr else 0)
        sig.append({"name": "CVD 4h", "sign": s,
                    "text": f"taker ròng 4h {d4:+,.0f} BTC (24h {d24:+,.0f}) → " +
                            ("phe mua chủ động áp đảo" if s > 0 else "phe bán chủ động áp đảo" if s < 0 else "cân bằng")})
        # phân kỳ CVD tại 2 swing gần nhất M15
        cvd_at = lambda i: float(tk["cvd"].iloc[max(tk["dt"].searchsorted(m15["dt"].iloc[i], side="right") - 1, 0)])  # noqa: E731
        div = None
        n = len(m15)
        for kind in ("H", "L"):
            pts = [p for p in sw_m15 if p["kind"] == kind and p["i"] >= n - 80 and m15["dt"].iloc[p["i"]] >= tk["dt"].iloc[0]]
            if len(pts) < 2:
                continue
            a, b = pts[-2], pts[-1]
            ca, cb = cvd_at(a["i"]), cvd_at(b["i"])
            if kind == "H" and b["price"] > a["price"] and cb < ca:
                div = {"type": "bearish", "i": [a["i"], b["i"]], "text": "đỉnh M15 cao hơn nhưng CVD thấp hơn: lệnh bán limit hấp thụ ở đỉnh"}
            if kind == "L" and b["price"] < a["price"] and cb > ca:
                div = {"type": "bullish", "i": [a["i"], b["i"]], "text": "đáy M15 thấp hơn nhưng CVD cao hơn: lệnh mua limit hấp thụ ở đáy"}
        if div:
            info["cvd_divergence"] = div
            sig.append({"name": "Phân kỳ CVD", "sign": 1 if div["type"] == "bullish" else -1, "text": div["text"]})
    tk5 = _df(flow.get("taker_5m"), ["ts", "sell", "buy"])
    if tk5 is not None and len(tk5) > 20:
        tk5["delta"] = tk5["buy"] - tk5["sell"]
        med = float(tk5["delta"].abs().median()) or 1.0
        big = tk5[tk5["delta"].abs() >= 4 * med].tail(6)
        info["whale_market_orders_5m"] = [
            {"time_vn": r.dt.astimezone(VN).strftime("%d/%m %H:%M"), "delta_btc": round(float(r.delta), 1),
             "x_median": round(abs(float(r.delta)) / med, 1)} for r in big.itertuples()]

    # 3) Funding
    fr = (flow.get("funding") or {}).get("rate")
    fh = flow.get("funding_hist") or []
    if fr is not None:
        avg7 = float(np.mean([x[1] for x in fh[-21:]])) if fh else 0.0
        if fr > 0.0003:
            s, t = -1, "funding cao: phe long đông, dễ bị quét long"
        elif fr < -0.0001:
            s, t = 1, "funding âm: phe short đông, dễ bị ép short"
        else:
            s, t = 0, "funding bình thường"
        info["funding"] = {"rate_pct": round(fr * 100, 4), "avg7d_pct": round(avg7 * 100, 4)}
        series["funding_hist"] = _df(fh, ["ts", "rate"])
        sig.append({"name": "Funding", "sign": s, "text": f"{fr * 100:.4f}%/8h (TB 7 ngày {avg7 * 100:.4f}%) → {t}"})

    # 4) Long/short: top trader (cá mập) vs toàn bộ tài khoản (đám đông)
    tp = _df(flow.get("ls_top_pos_1h"), ["ts", "r"]); al = _df(flow.get("ls_all_acc_1h"), ["ts", "r"])
    ta = _df(flow.get("ls_top_acc_1h"), ["ts", "r"])
    if tp is not None and al is not None and len(tp) > 25 and len(al) > 25:
        series.update(ls_top_pos=tp, ls_all_acc=al, ls_top_acc=ta)
        dtp = _pct(tp["r"].iloc[-1], tp["r"].iloc[-25]); dal = _pct(al["r"].iloc[-1], al["r"].iloc[-25])
        s, t = 0, "không lệch rõ"
        if dtp > 3 and dal < 0:
            s, t = 1, "cá mập tăng vị thế long trong khi đám đông giảm long"
        elif dtp < -3 and dal > 0:
            s, t = -1, "cá mập giảm long/tăng short trong khi đám đông tăng long"
        elif al["r"].iloc[-1] > 2.5:
            s, t = -1, "đám đông nghiêng long rất mạnh (dễ bị quét long)"
        elif al["r"].iloc[-1] < 0.8:
            s, t = 1, "đám đông nghiêng short mạnh (dễ bị ép short)"
        info["long_short"] = {"top_trader_pos": round(float(tp["r"].iloc[-1]), 3), "top_trader_pos_24h_chg_pct": round(dtp, 1),
                              "all_accounts": round(float(al["r"].iloc[-1]), 3), "all_accounts_24h_chg_pct": round(dal, 1),
                              "top_trader_acc": round(float(ta["r"].iloc[-1]), 3) if ta is not None else None}
        sig.append({"name": "Long/Short", "sign": s,
                    "text": f"top trader (vị thế) {tp['r'].iloc[-1]:.2f} ({dtp:+.1f}%/24h), đám đông {al['r'].iloc[-1]:.2f} ({dal:+.1f}%/24h) → {t}"})

    # 5) Thanh lý
    lq = _df(flow.get("liquidations"), ["ts", "px", "btc", "pos", "side"])
    if lq is not None and len(lq):
        series["liq"] = lq
        last_dt = lq["dt"].iloc[-1]
        cover_h = (last_dt - lq["dt"].iloc[0]).total_seconds() / 3600
        res = {"covers_hours": round(cover_h, 1)}
        for lab, hrs in (("1h", 1), ("4h", 4), ("24h", 24)):
            w = lq[lq["dt"] >= now - pd.Timedelta(hours=hrs)]
            res[lab] = {"long_btc": round(float(w.loc[w["pos"] == "long", "btc"].sum()), 1),
                        "short_btc": round(float(w.loc[w["pos"] == "short", "btc"].sum()), 1)}
        w = lq[lq["dt"] >= now - pd.Timedelta(hours=24)]
        top = w.nlargest(5, "btc")
        res["largest"] = [{"time_vn": r.dt.astimezone(VN).strftime("%d/%m %H:%M"), "px": r.px, "btc": round(r.btc, 2),
                           "pos": r.pos} for r in top.itertuples()]
        bins = (w["px"] // 100 * 100)
        cl = w.groupby([bins, w["pos"]])["btc"].sum().sort_values(ascending=False).head(4)
        res["clusters"] = [{"price_bin": int(k[0]), "pos": k[1], "btc": round(float(v), 1)} for k, v in cl.items()]
        info["liquidations"] = res
        w6 = lq[lq["dt"] >= now - pd.Timedelta(hours=6)]
        lg = float(w6.loc[w6["pos"] == "long", "btc"].sum()); sh = float(w6.loc[w6["pos"] == "short", "btc"].sum())
        s, t = 0, f"6h: long bị thanh lý {lg:,.1f} BTC, short {sh:,.1f} BTC"
        if lg > 3 * max(sh, 1) and lg > 15:
            med = float(w6.loc[w6["pos"] == "long", "px"].median())
            if price > med:
                s, t = 1, t + " → long đã bị quét (thanh khoản bên bán) và giá hồi lên"
        elif sh > 3 * max(lg, 1) and sh > 15:
            med = float(w6.loc[w6["pos"] == "short", "px"].median())
            if price < med:
                s, t = -1, t + " → short đã bị quét (thanh khoản bên mua) và giá rơi lại"
        sig.append({"name": "Thanh lý", "sign": s, "text": t})

    # 6) Tường lệnh trong sổ lệnh (tham khảo, dễ bị rút/spoof)
    bk = flow.get("book")
    if bk and bk.get("bids") and bk.get("asks"):
        mid = (bk.get("best_bid", bk["bids"][0][0]) + bk.get("best_ask", bk["asks"][0][0])) / 2
        step = 50.0

        def walls(levels):
            d = {}
            for p, q in levels:
                if abs(p / mid - 1) <= 0.02:
                    k = round(p // step * step, 1)
                    d[k] = d.get(k, 0) + q
            return sorted(d.items(), key=lambda kv: -kv[1])

        wb, wa = walls(bk["bids"]), walls(bk["asks"])
        near = lambda lv, pct: sum(q for p, q in lv if abs(p / mid - 1) <= pct)  # noqa: E731
        b5, a5 = near(bk["bids"], 0.005), near(bk["asks"], 0.005)
        info["orderbook"] = {"mid": round(mid, 1), "bid_walls": [[p, round(q, 1)] for p, q in wb[:3]],
                             "ask_walls": [[p, round(q, 1)] for p, q in wa[:3]],
                             "imbalance_0_5pct": round(b5 / max(b5 + a5, 1e-9), 3),
                             "depth_pct": [round((bk["bids"][-1][0] / mid - 1) * 100, 2), round((bk["asks"][-1][0] / mid - 1) * 100, 2)]}

    # 7) Coinbase premium (cầu spot Mỹ)
    cb = flow.get("coinbase_1h")
    if cb and h1 is not None:
        c = _df(cb, ["ts", "cb_close", "cb_vol"])
        m = pd.merge(c, h1[["dt", "close"]], on="dt", how="inner")
        if len(m) > 30:
            m["prem"] = (m["cb_close"] / m["close"] - 1) * 100
            series["cb_prem"] = m
            cur = float(m["prem"].iloc[-1]); avg = float(m["prem"].iloc[-24:].mean())
            s = 1 if cur > 0.05 and avg > 0 else (-1 if cur < -0.05 and avg < 0 else 0)
            info["coinbase_premium"] = {"now_pct": round(cur, 3), "avg24h_pct": round(avg, 3)}
            sig.append({"name": "Coinbase premium", "sign": s,
                        "text": f"{cur:+.3f}% (TB 24h {avg:+.3f}%) → " +
                                ("cầu spot Mỹ mạnh" if s > 0 else "Mỹ bán spot / cầu yếu" if s < 0 else "trung tính")})

    score = int(sum(x["sign"] for x in sig))
    return {"signals": sig, "score": score, "info": info, "series": series}

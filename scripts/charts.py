"""Vẽ chart nền tối, dễ đọc trên điện thoại: M15 (điểm vào), H1 (bối cảnh), đa khung, dấu chân cá mập."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from core import VN, ema, key_volume, spikes  # noqa: E402

BG, FG, GRID = "#0f1117", "#d7dae0", "#232733"
UP, DN, BLUE, YEL, PURP, GRAY, ORG, CYAN, PINK = ("#26a69a", "#ef5350", "#42a5f5", "#ffca28", "#ab47bc", "#8a8f98",
                                                  "#ff9800", "#26c6da", "#f06292")
SESS_COL = {"Phiên sáng (Á)": CYAN, "London KZ": BLUE, "New York KZ": PURP}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


# ---------------------------------------------------------------- tiện ích


def style(a):
    a.set_facecolor(BG); a.tick_params(colors=FG, labelsize=8); a.grid(color=GRID, linewidth=0.6)
    for s in a.spines.values():
        s.set_color(GRID)


def candles(ax, df, width=0.7):
    for k, r in enumerate(df.itertuples()):
        col = UP if r.close >= r.open else DN
        ax.vlines(k, r.low, r.high, color=col, linewidth=0.8, zorder=3)
        ax.add_patch(plt.Rectangle((k - width / 2, min(r.open, r.close)), width, max(abs(r.close - r.open), 1e-6),
                                   color=col, linewidth=0, zorder=3))


def xticks(ax, w, n_ticks=10):
    step = max(len(w) // n_ticks, 1)
    pos = list(range(0, len(w), step))
    ax.set_xticks(pos)
    ax.set_xticklabels([w["dt"].iloc[p].astimezone(VN).strftime("%d/%m\n%H:%M") for p in pos], color=FG, fontsize=7.5)


def label(ax, x, y, text, color, ha="left", va="center", size=8.5, z=6):
    ax.text(x, y, text, color=color, fontsize=size, ha=ha, va=va, zorder=z,
            bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec=color, lw=0.6, alpha=0.85))


def place_right(ax, items, x, lo, hi, gap_frac=0.026):
    """items: (y, text, color). Dời nhãn để không chồng nhau, kẻ đường dẫn về mức giá thật."""
    if not items:
        return
    gap = (hi - lo) * gap_frac
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + gap)
    over = ys[-1] - hi
    if over > 0:
        ys = [y - over for y in ys]
        for k in range(len(ys) - 2, -1, -1):
            ys[k] = min(ys[k], ys[k + 1] - gap)
    for (y0, txt, col), y in zip(items, ys):
        if abs(y - y0) > gap * 0.1:
            ax.plot([x - 0.3, x + 0.6], [y0, y], color=col, lw=0.5, alpha=0.7, clip_on=False)
        label(ax, x + 0.8, y, txt, col, size=7.3)


class XMap:
    """Đổi thời gian -> toạ độ x (số thứ tự nến) cho khung đang vẽ, kể cả thời điểm tương lai."""

    def __init__(self, w, minutes):
        self.t = w["dt"].reset_index(drop=True)
        self.bar = pd.Timedelta(minutes=minutes)
        self.n = len(w)

    def __call__(self, dt):
        dt = pd.Timestamp(dt)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        if dt <= self.t.iloc[0]:
            return (dt - self.t.iloc[0]) / self.bar
        if dt >= self.t.iloc[-1]:
            return self.n - 1 + (dt - self.t.iloc[-1]) / self.bar
        k = int(self.t.searchsorted(dt, side="right") - 1)
        return k + (dt - self.t.iloc[k]) / self.bar


def twin(a, color=YEL):
    """Trục phụ bên trái (màu riêng), trục chính giữ bên phải."""
    a2 = a.twinx(); style(a2); a2.grid(False)
    a.yaxis.tick_right(); a2.yaxis.tick_left()
    a2.tick_params(axis="y", colors=color, labelsize=7)
    return a2


def plabel(a, text):
    a.text(0.003, 0.96, text, transform=a.transAxes, color=FG, fontsize=7.8, va="top", ha="left", zorder=9,
           bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec=GRID, alpha=0.8))


def legend(ax, handles, ncol=4, loc="upper left"):
    ax.legend(handles=handles, loc=loc, fontsize=7.2, facecolor=BG, edgecolor=GRID, labelcolor=FG, ncol=ncol,
              framealpha=0.85)


def trend_txt(t):
    return "↑" if t == "up" else ("↓" if t == "down" else "–")


def infobox(ax, lines, loc=(0.005, 0.015)):
    ax.text(loc[0], loc[1], "\n".join(lines), transform=ax.transAxes, color=FG, fontsize=7.6, va="bottom", ha="left",
            zorder=9, bbox=dict(boxstyle="round,pad=0.35", fc=BG, ec=GRID, alpha=0.88))


# ---------------------------------------------------------------- phần chung của chart nến


def time_layer(ax, xm, tctx, n, ylim, show_macro=True, news=True):
    lo, hi = ylim
    for iv in tctx["_intervals"]:
        x0, x1 = xm(iv["start"]), xm(iv["end"])
        if x1 < -1 or x0 > n + 25:
            continue
        if iv["kind"] == "kz":
            ax.axvspan(x0 - 0.5, x1 - 0.5, color=SESS_COL.get(iv["name"], BLUE), alpha=0.07, lw=0, zorder=0)
            if -1 <= x0 <= n + 20:
                ax.text(x0 - 0.3, lo + (hi - lo) * 0.005, iv["name"].replace(" KZ", "").replace("Phiên ", ""),
                        color=SESS_COL.get(iv["name"], BLUE), fontsize=6.6, va="bottom", ha="left", alpha=0.9, zorder=2)
        elif iv["kind"] == "sb":
            ax.axvspan(x0 - 0.5, x1 - 0.5, ymin=0.965, ymax=1.0, color=YEL, alpha=0.55, lw=0, zorder=1)
        elif iv["kind"] == "macro" and show_macro:
            ax.axvspan(x0 - 0.5, x1 - 0.5, ymin=0.975, ymax=1.0, color=ORG, alpha=0.45, lw=0, zorder=1)
    if news:
        for e in tctx["_events"]:
            x = xm(e["dt"])
            if x < 0 or x > n + 24:
                continue
            col = DN if e["big"] else (ORG if e["impact"] == "High" else GRAY)
            if x <= n - 0.5:  # tin đã ra: vạch dọc + tên
                ax.axvline(x - 0.5, color=col, lw=0.9 if e["big"] else 0.6, ls="--", alpha=0.8, zorder=1)
                ax.text(x - 0.3, lo + (hi - lo) * 0.04, f"{e['time_vn'][-5:]} {e['country']} {e['title'][:22]}", color=col,
                        fontsize=6.5, rotation=90, va="bottom", ha="left", alpha=0.95, zorder=2)
            else:  # tin sắp ra: tam giác ở mép trên (chi tiết trong ô thông tin)
                ax.scatter([x - 0.5], [hi - (hi - lo) * 0.05], marker="v", s=55 if e["big"] else 30, color=col, zorder=6)


def plan_layer(ax, plan, n, x1_frac=0.84):
    if not plan:
        return
    x1 = n * x1_frac
    lo, hi = ax.get_ylim()
    for name, v, col in (("Entry", plan["entry"], BLUE), ("SL", plan["sl"], DN), ("TP1 2R", plan["tp1"], UP), ("TP2", plan.get("tp2"), UP)):
        if v and lo <= v <= hi:
            ax.hlines(v, x1, n + 0.5, color=col, lw=1.8, zorder=7)
            ax.text(x1, v, f" {name} {v:,.1f} ", color=BG, fontsize=8.3, va="center", ha="right", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc=col, ec=col))
    ax.add_patch(plt.Rectangle((x1, min(plan["entry"], plan["sl"])), n + 0.5 - x1, abs(plan["entry"] - plan["sl"]),
                               color=DN, alpha=0.13, zorder=1))
    ax.add_patch(plt.Rectangle((x1, min(plan["entry"], plan["tp1"])), n + 0.5 - x1, abs(plan["tp1"] - plan["entry"]),
                               color=UP, alpha=0.10, zorder=1))


def structure_layer(ax, w, x0, n, sw, ev, fv, ob, pad, X):
    for g in fv:
        if g["end"] < x0 or g["filled"]:
            continue
        col = UP if g["dir"] == "up" else DN
        ax.add_patch(plt.Rectangle((X(g["i"]) - 0.5, g["bot"]), n - X(g["i"]) + 0.5, g["top"] - g["bot"],
                                   color=col, alpha=0.14, lw=0, zorder=1))
        ax.hlines(g["mid"], X(g["i"]), n, color=col, lw=0.6, ls="--", alpha=0.7, zorder=1)
        if g["i"] >= x0:
            ax.text(X(g["i"]) + 0.5, g["top"], "FVG", color=col, fontsize=6.8, va="bottom", zorder=2)
    for o in ob:
        if o["i"] < x0 or not o["valid"]:
            continue
        col = UP if o["dir"] == "up" else DN
        ax.add_patch(plt.Rectangle((X(o["i"]) - 0.5, o["bot"]), n - X(o["i"]) + 0.5, o["top"] - o["bot"],
                                   fill=False, ec=col, lw=1.4, zorder=2))
        ax.text(X(o["i"]), o["bot"], "OB", color=col, fontsize=7.3, va="top", fontweight="bold", zorder=2)
    for e in ev:
        if e["from"] < x0:
            continue
        col = UP if e["dir"] == "up" else DN
        ax.hlines(e["level"], X(e["from"]), X(e["to"]), color=col, lw=1.2, ls="-" if e["type"] == "BOS" else "--", zorder=4)
        ax.text((X(e["from"]) + X(e["to"])) / 2, e["level"], e["type"], color=col, fontsize=7.8, ha="center",
                va="bottom" if e["dir"] == "up" else "top", fontweight="bold", zorder=4)
    for s in sw:
        if s["i"] < x0:
            continue
        top = s["kind"] == "H"
        ax.text(X(s["i"]), s["price"] + (pad * 0.22 if top else -pad * 0.22), s["label"], color=FG, fontsize=6.6,
                ha="center", va="bottom" if top else "top", zorder=4)


def liquidity_layer(c, R, ylim):
    lo, hi = ylim
    out = []
    items = c["liq"] + [{"name": "EQH $$$", "price": e["price"], "side": "H", "swept": False} for e in c["eqh"][-2:]] + \
        [{"name": "EQL $$$", "price": e["price"], "side": "L", "swept": False} for e in c["eql"][-2:]]
    for q in items:
        if lo <= q["price"] <= hi:
            col = DN if q["side"] == "H" else UP
            out.append((q["price"], col, q["swept"]))
            R.append((q["price"], f"{q['name']} {q['price']:,.0f}" + (" swept" if q["swept"] else ""), col if not q["swept"] else GRAY))
    return out


def pair_lines(ax, X, x0, items, df, color, tag, k=0):
    """Nối 2 swing của phân kỳ / SMT (k: thứ tự để dời nhãn khỏi chồng nhau)."""
    lo_, hi_ = ax.get_ylim(); dy = (hi_ - lo_) * 0.03 * k
    for it in items:
        i, j = it["i"]
        if i < x0:
            continue
        hi = it["type"] == "bearish"
        ya = float(df["high"].iloc[i] if hi else df["low"].iloc[i]); yb = float(df["high"].iloc[j] if hi else df["low"].iloc[j])
        ax.plot([X(i), X(j)], [ya, yb], color=color, lw=1.3, ls=(0, (3, 2)), zorder=5)
        ax.text(X(j) + 0.6, yb + (dy if hi else -dy), tag, color=color, fontsize=7, va="bottom" if hi else "top", zorder=5,
                fontweight="bold", clip_on=True)


def subplot_grid(heights, w=16, h=11.5):
    fig = plt.figure(figsize=(w, h), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.05)
    axes = [fig.add_subplot(gs[0, 0])]
    for k in range(1, len(heights)):
        axes.append(fig.add_subplot(gs[k, 0], sharex=axes[0]))
    for a in axes:
        style(a); a.yaxis.tick_right()
    for a in axes[:-1]:
        plt.setp(a.get_xticklabels(), visible=False)
    return fig, axes


def map_series(w, xm, df, col):
    """Ghép chuỗi theo thời gian (dt) vào trục x của khung đang vẽ."""
    if df is None or not len(df):
        return np.array([]), np.array([])
    d = df[(df["dt"] >= w["dt"].iloc[0]) & (df["dt"] <= w["dt"].iloc[-1] + pd.Timedelta(hours=1))]
    return np.array([xm(t) for t in d["dt"]]), d[col].values


# ---------------------------------------------------------------- M15: điểm vào


def draw_m15(s, c, path, n=160):
    df = c["m15"]; r = c["m15_res"]; tctx = c["tctx"]; fl = c["flow"]
    x0 = max(len(df) - n, 0); w = df.iloc[x0:].reset_index(drop=True); n = len(w)
    X = lambda i: i - x0  # noqa: E731
    xm = XMap(w, 15)
    plan = s["m15_setup"]
    fig, (ax, av, ad, ao) = subplot_grid([6, 1.3, 1.3, 1.2])
    lo, hi = w["low"].min(), w["high"].max()
    if plan:
        lo = min(lo, plan["sl"], plan["tp1"], plan["entry"]); hi = max(hi, plan["sl"], plan["tp1"], plan["entry"])
    pad = (hi - lo) * 0.07
    ylim = (lo - pad, hi + pad)
    ax.set_ylim(*ylim); ax.set_xlim(-1, n + 24)
    R = []
    time_layer(ax, xm, tctx, n, ylim)

    # mức giá mở cửa
    an = tctx["_anchors"]
    for key, t0, col in (("Mở cửa ngày (07:00 VN)", an["day"], GRAY), ("NY midnight open", an["ny_mid"], ORG),
                         ("NY 08:30 open", an["ny_830"], ORG)):
        v = tctx["opens"].get(key)
        if v and ylim[0] <= v <= ylim[1]:
            ax.hlines(v, max(xm(t0), 0), n, color=col, lw=0.9, ls=(0, (1, 2)), zorder=2)
            R.append((v, f"{key.replace(' (07:00 VN)', '')} {v:,.0f}", col))

    # VWAP
    vd = c["vw_day_m15"].iloc[x0:].values
    ax.plot(range(n), vd, color=PINK, lw=1.0, alpha=0.9, zorder=2)
    R.append((float(vd[-1]), f"VWAP ngày {vd[-1]:,.0f}", PINK))
    vw = pd.DataFrame({"dt": c["h1"]["dt"], "v": c["vw_week_h1"]}).dropna()
    xs, ys = map_series(w, xm, vw, "v")
    if len(xs):
        ax.plot(xs, ys, color=CYAN, lw=0.9, alpha=0.8, ls="-.", zorder=2)
        R.append((float(ys[-1]), f"VWAP tuần {ys[-1]:,.0f}", CYAN))

    structure_layer(ax, w, x0, n, r["sw"], r["ev"], r["fv"], r["ob"], pad, X)
    kv = key_volume(df, 96)
    if kv["i"] >= x0:
        ax.add_patch(plt.Rectangle((X(kv["i"]) - 0.5, kv["bot"]), n - X(kv["i"]) + 0.5, kv["top"] - kv["bot"], color=YEL, alpha=0.15, lw=0))
        label(ax, X(kv["i"]), kv["top"], "KV M15", YEL, va="bottom", size=7)
    for p, col, sw_ in liquidity_layer(c, R, ylim):
        ax.hlines(p, 0, n, color=col, lw=0.9, ls=(0, (6, 3)), alpha=0.35 if sw_ else 0.9, zorder=2)

    # Volume profile M15 24h: POC/VAH/VAL
    vp = c["vp_m15"]
    if vp:
        for k, col in (("poc", YEL), ("vah", GRAY), ("val", GRAY)):
            if ylim[0] <= vp[k] <= ylim[1]:
                ax.hlines(vp[k], n - 96, n, color=col, lw=0.8, ls=":", alpha=0.9, zorder=2)
                R.append((vp[k], f"{k.upper()} 24h {vp[k]:,.0f}", col))

    # Tường lệnh
    ob_info = s["footprint"]["info"].get("orderbook")
    if ob_info:
        for side, col in (("bid_walls", UP), ("ask_walls", DN)):
            for p, q in ob_info[side][:2]:
                if ylim[0] <= p <= ylim[1]:
                    ax.hlines(p + 25, n - 8, n + 0.5, color=col, lw=3.5, alpha=0.55, zorder=5)
                    R.append((p + 25, f"Tường {'mua' if side == 'bid_walls' else 'bán'} {p:,.0f} ({q:,.0f} BTC)", col))

    # Thanh lý (bong bóng)
    lq = fl["series"].get("liq")
    if lq is not None and len(lq):
        lw_ = lq[(lq["dt"] >= w["dt"].iloc[0]) & (lq["dt"] <= w["dt"].iloc[-1] + pd.Timedelta(minutes=15)) &
                 (lq["px"] >= ylim[0]) & (lq["px"] <= ylim[1])].copy()
        if len(lw_):
            lw_["bar"] = lw_["dt"].dt.floor("15min")
            g = lw_.groupby(["bar", "pos"]).agg(btc=("btc", "sum"), px=("px", "median")).reset_index()
            g = g[g["btc"] >= max(3.0, float(g["btc"].quantile(0.8)) if len(g) > 5 else 3.0)]
            for rr in g.itertuples():
                x = xm(rr.bar)
                col = DN if rr.pos == "long" else UP
                ax.scatter([x], [rr.px], s=min(20 + rr.btc * 6, 600), color=col, alpha=0.35, edgecolors=col, lw=0.8, zorder=6)
                if rr.btc >= 20:
                    ax.text(x, rr.px, f"{rr.btc:,.0f}", color=FG, fontsize=6.5, ha="center", va="center", zorder=7, clip_on=True)

    # VSA, phân kỳ, SMT
    for v in s["vsa"]["M15"]:
        if v["i"] >= x0:
            up_ = v["bias"] > 0
            y = float(df["low"].iloc[v["i"]]) - pad * 0.45 if up_ else float(df["high"].iloc[v["i"]]) + pad * 0.45
            ax.scatter([X(v["i"])], [y], marker="^" if up_ else "v", s=40, color=YEL, zorder=6)
    pair_lines(ax, X, x0, c["div"]["M15"], df, ORG, "Div RSI", 0)
    pair_lines(ax, X, x0, c["smt"]["M15"], df, CYAN, "SMT", 1)
    cd = fl["info"].get("cvd_divergence")
    if cd and max(cd["i"]) < len(df):
        pair_lines(ax, X, x0, [cd], df, PINK, "Div CVD", 2)

    candles(ax, w)
    ax.axhline(s["price"], color=FG, lw=0.6, alpha=0.6)
    R.append((s["price"], f"Giá {s['price']:,.1f}", FG))
    place_right(ax, [t for t in R if ylim[0] <= t[0] <= ylim[1]], n + 1, *ylim)
    plan_layer(ax, plan, n)

    # info box
    mt = " ".join(f"{m['tf']}{trend_txt(m['trend'])}" for m in s["mtf"])
    tm = s["time"]
    lines = [f"Đa khung: {mt}  ({s['mtf_alignment'] or '?'} cùng D1)",
             "Thời gian: " + (f"{tm['in_window']} đến {tm['window_end_vn']}" if tm["in_window"] else
                              (f"ngoài phiên · {tm['next_window']['name']} mở {tm['next_window']['start_vn']}" if tm["next_window"] else "ngoài phiên")) +
             (f" · {tm['silver_bullet_now']}" if tm["silver_bullet_now"] else "") + (f" · {tm['macro_now']}" if tm["macro_now"] else ""),
             f"Dấu chân cá mập: {s['footprint']['score']:+d} · " + " | ".join(f"{x['name']} {'+' if x['sign'] > 0 else '−' if x['sign'] < 0 else '0'}" for x in s["footprint"]["signals"])]
    up, seen = [], set()
    for e in tm["news_upcoming"]:
        if (e["impact"] == "High" or e["big"]) and (e["time_vn"], e["country"]) not in seen:
            seen.add((e["time_vn"], e["country"])); up.append(e)
    if up:
        lines.append("Tin sắp tới (▼): " + "; ".join(f"{e['time_vn']} {e['country']} {e['title'][:28]}" for e in up[:3]))
    if tm["no_trade_news"]:
        lines.append("⚠ Trong ±30 phút tin lớn: KHÔNG mở lệnh")
    if plan and s.get("m15_checklist"):
        ck = s["m15_checklist"]
        lines.append(f"Checklist lệnh: {ck['passed']}/{ck['total']} · quy tắc cứng: {'ĐẠT' if ck['hard_ok'] else 'CHƯA ĐẠT'}")
    infobox(ax, lines)

    # volume
    sma = df["volume"].rolling(20).mean().iloc[x0:].values
    cols = [UP if rr.close >= rr.open else DN for rr in w.itertuples()]
    av.bar(range(n), w["volume"], color=cols, width=0.7, alpha=0.6)
    av.plot(range(n), sma, color=GRAY, lw=0.8)
    for i in spikes(df):
        if i >= x0:
            av.bar(X(i), df["volume"][i], color=YEL, width=0.7)
    plabel(av, "Volume (vàng = đột biến ≥2× TB20)")

    # delta & CVD
    tk = fl["series"].get("taker_15m")
    if tk is not None:
        xs, d = map_series(w, xm, tk, "delta")
        ad.bar(xs, d, color=[UP if v >= 0 else DN for v in d], width=0.7, alpha=0.7)
        a2 = twin(ad)
        xs2, cv = map_series(w, xm, tk, "cvd")
        a2.plot(xs2, cv, color=YEL, lw=1.1)
        plabel(ad, "Taker mua−bán mỗi 15m (cột) · CVD (vàng, trục trái)")
    else:
        ad.text(0.5, 0.5, "không có dữ liệu taker", transform=ad.transAxes, color=GRAY, ha="center")
    oi = fl["series"].get("oi_15m")
    if oi is not None:
        xs, v = map_series(w, xm, oi, "oi_btc")
        ao.plot(xs, v, color=BLUE, lw=1.2)
        ao.fill_between(xs, v, v.min() if len(v) else 0, color=BLUE, alpha=0.08)
        plabel(ao, "Open Interest (BTC)")
    xticks(ao, w)
    ax.set_title(f"{s['source']} · M15 · {s['time_vn']} (giờ VN) · D1 {trend_txt(s['d1_trend'])} · {s['premium_discount']} · {s['conclusion']}",
                 color=FG, fontsize=12, loc="left")
    legend(ax, [Line2D([], [], color=FG, lw=1.2, label="BOS (liền) / CHoCH (đứt)"), Patch(color=UP, alpha=0.3, label="FVG"),
                Patch(fill=False, ec=FG, label="Order Block"), Line2D([], [], color=PINK, label="VWAP ngày"),
                Line2D([], [], color=CYAN, ls="-.", label="VWAP tuần"), Line2D([], [], color=ORG, ls=":", label="Giá mở cửa NY"),
                Line2D([], [], marker="o", ls="", color=DN, alpha=0.5, label="Thanh lý long"),
                Line2D([], [], marker="o", ls="", color=UP, alpha=0.5, label="Thanh lý short"),
                Patch(color=YEL, alpha=0.6, label="Silver Bullet (dải trên)"), Patch(color=ORG, alpha=0.5, label="ICT macro"),
                Line2D([], [], color=DN, ls="--", label="Tin lớn")], ncol=6)
    fig.savefig(path, facecolor=BG, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- H1: bối cảnh


def draw_h1(s, c, path, n=120):
    h1 = c["h1"]; tctx = c["tctx"]; fl = c["flow"]
    x0 = len(h1) - n; w = h1.iloc[x0:].reset_index(drop=True)
    X = lambda i: max(i - x0, 0)  # noqa: E731
    xm = XMap(w, 60)
    fig = plt.figure(figsize=(16, 11), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(3, 2, height_ratios=[6, 1.3, 1.3], width_ratios=[12, 1.6], hspace=0.05, wspace=0.02)
    ax = fig.add_subplot(gs[0, 0]); avp = fig.add_subplot(gs[0, 1], sharey=ax)
    av = fig.add_subplot(gs[1, 0], sharex=ax); ao = fig.add_subplot(gs[2, 0], sharex=ax)
    for a in (ax, avp, av, ao):
        style(a)
    for a in (ax, av, ao):
        a.yaxis.tick_right()
    plt.setp(ax.get_xticklabels(), visible=False); plt.setp(av.get_xticklabels(), visible=False)
    avp.yaxis.tick_right(); avp.set_xticks([])
    ymin, ymax = w["low"].min(), w["high"].max()
    for z in s["zones"]:
        if z["score_auto"] >= 4 and abs(z["mid"] / s["price"] - 1) < 0.02:
            ymin, ymax = min(ymin, z["bot"]), max(ymax, z["top"])
    pad = (ymax - ymin) * 0.08
    ylim = (ymin - pad, ymax + pad)
    ax.set_ylim(*ylim); ax.set_xlim(-1, n + 22)
    R = []
    time_layer(ax, xm, tctx, n, ylim, show_macro=False)

    f = c["fib"]
    if f and f["from_i"] >= x0:
        for rr, v in f["levels"].items():
            ax.hlines(v, X(f["from_i"]), n, color=GRAY, lw=0.7, ls=":")
            if rr in ("0.5", "0.618", "0.79"):
                R.append((v, f"Fib {rr} {v:,.0f}", GRAY))
        ax.add_patch(plt.Rectangle((X(f["from_i"]), f["ote"][0]), n - X(f["from_i"]), f["ote"][1] - f["ote"][0], color=YEL, alpha=0.07, lw=0))
        ax.text(X(f["from_i"]), f["levels"]["0.5"], " Premium ↑ / Discount ↓ (OTE vàng)", color=GRAY, fontsize=7.3, va="bottom")

    structure_layer(ax, w, x0, n, c["sw_h1"], c["ev_h1"], c["fv_h1"], c["ob_h1"], pad, X)
    for kv, name in ((c["kv_h1"], "KV H1"), (c["kv_h4"], "KV H4")):
        if name == "KV H4":
            t4 = c["h4"]["dt"][kv["i"]]
            start = int((h1["dt"] >= t4).idxmax()) if (h1["dt"] >= t4).any() else x0
            for v, nm in ((kv["top"], "KV H4 đỉnh"), (kv["bot"], "KV H4 đáy")):
                ax.hlines(v, X(start), n, color=YEL, lw=1.0, ls="-.", alpha=0.8)
                R.append((v, f"{nm} {v:,.0f}" + (" (phá)" if kv["broken"] else ""), YEL))
        elif kv["i"] >= x0:
            ax.add_patch(plt.Rectangle((X(kv["i"]) - 0.5, kv["bot"]), n - X(kv["i"]) + 0.5, kv["top"] - kv["bot"], color=YEL, alpha=0.16, lw=0))
            label(ax, X(kv["i"]), kv["top"], name, YEL, va="bottom", size=7.3)
    for p, col, sw_ in liquidity_layer(c, R, ylim):
        ax.hlines(p, 0, n, color=col, lw=0.9, ls=(0, (6, 3)), alpha=0.35 if sw_ else 0.9)

    # VWAP tuần + AVWAP con sóng D1
    vwk = c["vw_week_h1"].iloc[x0:].values
    ax.plot(range(n), vwk, color=CYAN, lw=1.0, ls="-.", alpha=0.9)
    if not np.isnan(vwk[-1]):
        R.append((float(vwk[-1]), f"VWAP tuần {vwk[-1]:,.0f}", CYAN))
    av_ = c["avwap"]
    if av_ and av_.get("series_h1") is not None:
        sv = av_["series_h1"].iloc[x0:].values
        ax.plot(range(n), sv, color=PINK, lw=1.0, alpha=0.9)
        R.append((float(sv[-1]), f"AVWAP từ {av_['anchor_vn']} {sv[-1]:,.0f}", PINK))
    elif av_ and ylim[0] <= av_["value"] <= ylim[1]:
        ax.axhline(av_["value"], color=PINK, lw=0.9, ls="--")
        R.append((av_["value"], f"AVWAP từ {av_['anchor_vn']} {av_['value']:,.0f}", PINK))

    # Volume profile 5 ngày
    vp = c["vp_h1"]
    if vp:
        mids, hist = np.array(vp["mids"]), np.array(vp["hist"])
        inva = (mids >= vp["val"]) & (mids <= vp["vah"])
        avp.barh(mids, hist, height=vp["step"] * 0.9, color=np.where(inva, BLUE, GRAY), alpha=0.6)
        avp.barh([vp["poc"]], [hist.max()], height=vp["step"] * 0.9, color=YEL, alpha=0.9)
        for k in ("poc", "vah", "val"):
            ax.axhline(vp[k], color=YEL if k == "poc" else BLUE, lw=0.7, ls=":", alpha=0.7)
            R.append((vp[k], f"{k.upper()} 5N {vp[k]:,.0f}", YEL if k == "poc" else BLUE))
        avp.set_title("Volume Profile 5N", color=FG, fontsize=8)
    avp.set_xlim(left=0)

    pair_lines(ax, X, x0, c["div"]["H1"], h1, ORG, "Div RSI", 0)
    pair_lines(ax, X, x0, c["smt"]["H1"], h1, CYAN, "SMT", 1)
    candles(ax, w)
    for z in s["zones"]:
        if z["score_auto"] < 4:
            continue
        col = UP if z["dir"] == "up" else DN
        ax.add_patch(plt.Rectangle((max(X(z["i"]), 0) - 0.5, z["bot"]), n - max(X(z["i"]), 0) + 0.5, z["top"] - z["bot"],
                                   fill=True, fc=col, alpha=0.08, ec=col, lw=1.0, ls="--", zorder=1))
        grade = "A+" if z["score_auto"] >= 6 else "B"
        label(ax, n - 1, z["mid"], f"Vùng {grade} {z['score_auto']}/7 (+vĩ mô)", col, ha="right", size=8)
    ax.axhline(s["price"], color=FG, lw=0.6, alpha=0.6)
    R.append((s["price"], f"Giá {s['price']:,.1f}", FG))
    place_right(ax, [t for t in R if ylim[0] <= t[0] <= ylim[1]], n + 1, *ylim)

    wy, p3 = s["wyckoff"], s["po3"]
    lines = [f"Wyckoff (ước lượng): {wy['phase']}", f"PO3: {p3['text']}" if p3 else "",
             f"D1 {trend_txt(s['d1_trend'])} · con sóng {s['d1_leg'][0]:,.0f}–{s['d1_leg'][1]:,.0f} · EQ {s['equilibrium']:,.0f} → {s['premium_discount']}"]
    infobox(ax, [x for x in lines if x])

    sma = h1["volume"].rolling(20).mean().iloc[x0:].values
    av.bar(range(n), w["volume"], color=[UP if rr.close >= rr.open else DN for rr in w.itertuples()], width=0.7, alpha=0.6)
    av.plot(range(n), sma, color=GRAY, lw=0.8)
    for i in c["sp_h1"]:
        if i >= x0:
            av.bar(X(i), h1["volume"][i], color=YEL, width=0.7)
    plabel(av, "Volume (vàng = đột biến ≥2× TB20)")
    oi = fl["series"].get("oi_1h"); tk = fl["series"].get("taker_1h")
    plabel(ao, "OI BTC (xanh, trục phải) · CVD taker (vàng, trục trái)")
    if oi is not None:
        xs, v = map_series(w, xm, oi, "oi_btc")
        ao.plot(xs, v, color=BLUE, lw=1.2)
    if tk is not None:
        a2 = twin(ao)
        xs, cv = map_series(w, xm, tk, "cvd")
        a2.plot(xs, cv, color=YEL, lw=1.0)
    xticks(ao, w)
    ax.set_title(f"{s['source']} · H1 · {s['time_vn']} (giờ VN) · D1 {trend_txt(s['d1_trend'])} · {s['premium_discount']} · {s['conclusion']}",
                 color=FG, fontsize=12, loc="left")
    legend(ax, [Line2D([], [], color=FG, lw=1.2, label="BOS (liền) / CHoCH (đứt)"), Patch(color=UP, alpha=0.3, label="FVG"),
                Patch(fill=False, ec=FG, label="Order Block"), Patch(color=YEL, alpha=0.4, label="Key Volume / spike"),
                Line2D([], [], color=DN, ls="--", label="BSL"), Line2D([], [], color=UP, ls="--", label="SSL"),
                Line2D([], [], color=CYAN, ls="-.", label="VWAP tuần"), Line2D([], [], color=PINK, label="AVWAP con sóng D1"),
                Patch(color=CYAN, alpha=0.25, label="Phiên sáng"), Patch(color=BLUE, alpha=0.25, label="London KZ"),
                Patch(color=PURP, alpha=0.25, label="New York KZ"), Line2D([], [], color=YEL, label="OI (xanh) / CVD (vàng)")], ncol=6)
    fig.savefig(path, facecolor=BG, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- đa khung


def draw_mtf(s, c, path):
    specs = [("D1", c["d1"], c["sw_d1"], c["ev_d1"], 90, "%d/%m"), ("H4", c["h4"], c["sw_h4"], c["ev_h4"], 120, "%d/%m"),
             ("H1", c["h1"], c["sw_h1"], c["ev_h1"], 120, "%d/%m %Hh"),
             ("M15", c["m15"], c["m15_res"]["sw"], c["m15_res"]["ev"], 128, "%d/%m %H:%M")]
    fig, axs = plt.subplots(2, 2, figsize=(16, 10.5), dpi=100, facecolor=BG)
    st = {m["tf"]: m for m in s["mtf"]}
    for ax, (name, df, sw, ev, n, fmt) in zip(axs.flat, specs):
        style(ax); ax.yaxis.tick_right()
        x0 = max(len(df) - n, 0); w = df.iloc[x0:].reset_index(drop=True); n = len(w)
        X = lambda i, x0=x0: i - x0  # noqa: E731
        m = st[name]
        lo, hi = m["range"]
        if name == "D1":
            lo, hi = s["d1_leg"]
        eqv = (lo + hi) / 2
        ymin, ymax = min(w["low"].min(), lo), max(w["high"].max(), hi)
        pad = (ymax - ymin) * 0.06
        ax.set_ylim(ymin - pad, ymax + pad); ax.set_xlim(-1, n + 3)
        ax.axhspan(eqv, hi, color=DN, alpha=0.05); ax.axhspan(lo, eqv, color=UP, alpha=0.05)
        for v, t in ((hi, "đỉnh range"), (eqv, "EQ 50%"), (lo, "đáy range")):
            ax.axhline(v, color=GRAY, lw=0.7, ls="--" if t == "EQ 50%" else ":")
            ax.text(0, v, f" {t} {v:,.0f}", color=GRAY, fontsize=6.8, va="bottom")
        ax.plot(range(n), ema(df["close"], 20).iloc[x0:].values, color=YEL, lw=0.9, alpha=0.85)
        ax.plot(range(n), ema(df["close"], 50).iloc[x0:].values, color=BLUE, lw=0.9, alpha=0.85)
        for e in ev:
            if e["from"] < x0:
                continue
            col = UP if e["dir"] == "up" else DN
            ax.hlines(e["level"], X(e["from"]), X(e["to"]), color=col, lw=1.1, ls="-" if e["type"] == "BOS" else "--")
            ax.text(X(e["to"]), e["level"], " " + e["type"], color=col, fontsize=6.8, va="center", fontweight="bold")
        for p in sw:
            if p["i"] >= x0:
                top = p["kind"] == "H"
                ax.text(X(p["i"]), p["price"] + (pad * 0.3 if top else -pad * 0.3), p["label"], color=FG, fontsize=6,
                        ha="center", va="bottom" if top else "top")
        candles(ax, w)
        ax.axhline(s["price"], color=FG, lw=0.6, alpha=0.6)
        step = max(n // 6, 1); pos = list(range(0, n, step))
        ax.set_xticks(pos); ax.set_xticklabels([w["dt"].iloc[p].astimezone(VN).strftime(fmt) for p in pos], color=FG, fontsize=7)
        tcol = UP if m["trend"] == "up" else DN if m["trend"] == "down" else GRAY
        ax.set_title(f"{name} {trend_txt(m['trend'])}  ·  {m['last_event'] or '—'}  ·  {m['pd']} {m['pos_pct']:.0f}%  ·  EMA {m['ema_stack']}  ·  RSI {m['rsi14']}",
                     color=tcol, fontsize=9.5, loc="left")
    wy, p3 = s["wyckoff"], s["po3"]
    sm = [x["text"] for k in ("H1", "M15") for x in s["smt"][k]] + [x["text"] for k in ("H1", "M15") for x in s["rsi_divergence"][k]]
    fig.suptitle(f"Đa khung · {s['time_vn']} (giờ VN) · " + " ".join(f"{m['tf']}{trend_txt(m['trend'])}" for m in s["mtf"]) +
                 f" · {s['mtf_alignment'] or '?'} cùng hướng D1 · Wyckoff: {wy['phase'].split(' –')[0]}",
                 color=FG, fontsize=12.5, x=0.01, ha="left", y=0.995)
    foot = (f"PO3: {p3['text']}" if p3 else "") + (("   ·   SMT/Phân kỳ: " + "; ".join(sm)) if sm else "")
    fig.text(0.01, 0.005, foot, color=FG, fontsize=8.5, ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, facecolor=BG, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- dấu chân cá mập


def draw_flow(s, c, path, days=4):
    fl = c["flow"]; se = fl["series"]; h1 = c["h1"]
    t0 = h1["dt"].iloc[-1] - pd.Timedelta(days=days)
    fig = plt.figure(figsize=(16, 14), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(7, 1, height_ratios=[1.05, 2.2, 1.2, 1.2, 1, 1.1, 1.2], hspace=0.09)
    head = fig.add_subplot(gs[0, 0]); head.set_facecolor(BG); head.axis("off")
    axs = [fig.add_subplot(gs[1, 0])]
    for k in range(2, 7):
        axs.append(fig.add_subplot(gs[k, 0], sharex=axs[0]))
    for a in axs:
        style(a); a.yaxis.tick_right()
    for a in axs[:-1]:
        plt.setp(a.get_xticklabels(), visible=False)
    tz = VN
    head.set_title(f"Dấu chân cá mập · {s['time_vn']} (giờ VN) · điểm dòng tiền {s['footprint']['score']:+d} "
                   f"(+ ủng hộ tăng / − ủng hộ giảm)", color=FG, fontsize=12.5, loc="left")
    for k, x in enumerate(s["footprint"]["signals"]):
        col = UP if x["sign"] > 0 else DN if x["sign"] < 0 else GRAY
        txt = f"{'+' if x['sign'] > 0 else '−' if x['sign'] < 0 else '0'}  {x['name']}: {x['text']}"
        head.text(0.0, 0.92 - k * 0.165, txt if len(txt) <= 112 else txt[:110] + "…",
                  color=col, fontsize=8.6, transform=head.transAxes, va="top")
    info = s["footprint"]["info"]
    extra = []
    if info.get("orderbook"):
        ob = info["orderbook"]
        extra.append("Tường mua: " + ", ".join(f"{p:,.0f} ({q:,.0f} BTC)" for p, q in ob["bid_walls"]))
        extra.append("Tường bán: " + ", ".join(f"{p:,.0f} ({q:,.0f} BTC)" for p, q in ob["ask_walls"]))
        extra.append(f"Mất cân bằng sổ lệnh ±0,5%: {ob['imbalance_0_5pct'] * 100:.0f}% bên mua")
    if info.get("whale_market_orders_5m"):
        extra.append("Lệnh thị trường lớn (5m): " + ", ".join(f"{x['time_vn'][-5:]} {x['delta_btc']:+,.0f}" for x in info["whale_market_orders_5m"][-4:]))
    for k, t in enumerate(extra):
        head.text(0.70, 0.92 - k * 0.165, t if len(t) <= 70 else t[:68] + "…", color=FG, fontsize=8.2,
                  transform=head.transAxes, va="top")

    w = h1[h1["dt"] >= t0]
    a = axs[0]
    a.plot(w["dt"], w["close"], color=FG, lw=1.2)
    a.fill_between(w["dt"], w["low"], w["high"], color=GRAY, alpha=0.15, lw=0)
    plabel(a, "Giá H1 (trắng, trục phải) · Coinbase premium % (tím, trục trái)")
    cb = se.get("cb_prem")
    if cb is not None:
        cbw = cb[cb["dt"] >= t0]
        a2 = twin(a, PURP)
        a2.plot(cbw["dt"], cbw["prem"], color=PURP, lw=1.1)
        a2.fill_between(cbw["dt"], cbw["prem"], 0, color=PURP, alpha=0.18, lw=0)
        a2.axhline(0, color=PURP, lw=0.6, ls=":")

    a = axs[1]
    oi = se.get("oi_1h")
    if oi is not None:
        o = oi[oi["dt"] >= t0]
        a.plot(o["dt"], o["oi_btc"], color=BLUE, lw=1.3)
        d = o["oi_btc"].diff()
        a2 = twin(a, GRAY)
        a2.bar(o["dt"], d, width=0.035, color=[UP if v >= 0 else DN for v in d.fillna(0)], alpha=0.4)
    plabel(a, "Open Interest BTC (xanh, trục phải) · ΔOI mỗi giờ (cột, trục trái)")

    a = axs[2]
    tk = se.get("taker_1h")
    if tk is not None:
        t = tk[tk["dt"] >= t0].copy()
        a.bar(t["dt"], t["delta"], width=0.035, color=[UP if v >= 0 else DN for v in t["delta"]], alpha=0.7)
        a2 = twin(a)
        a2.plot(t["dt"], t["delta"].cumsum(), color=YEL, lw=1.2)
    plabel(a, "Taker mua−bán mỗi giờ (cột, trục phải) · CVD (vàng, trục trái)")

    a = axs[3]
    fh = se.get("funding_hist")
    if fh is not None:
        f = fh[fh["dt"] >= t0 - pd.Timedelta(hours=8)]
        a.bar(f["dt"], f["rate"] * 100, width=0.25, color=[UP if v >= 0 else DN for v in f["rate"]], alpha=0.7, align="edge")
        a.axhline(0.01, color=GRAY, lw=0.6, ls=":"); a.axhline(0, color=GRAY, lw=0.6)
    plabel(a, "Funding %/8h (vạch chấm = 0,01% chuẩn)")

    a = axs[4]
    for key, col, lab in (("ls_top_pos", YEL, "Top trader (vị thế)"), ("ls_all_acc", BLUE, "Đám đông (tài khoản)"),
                          ("ls_top_acc", GRAY, "Top trader (tài khoản)")):
        d = se.get(key)
        if d is not None:
            d = d[d["dt"] >= t0]
            a.plot(d["dt"], d["r"], color=col, lw=1.2, label=lab)
    a.axhline(1, color=GRAY, lw=0.6, ls=":")
    plabel(a, "Tỷ lệ Long/Short")
    a.legend(loc="upper right", fontsize=7, facecolor=BG, edgecolor=GRID, labelcolor=FG, ncol=3)

    a = axs[5]
    lq = se.get("liq")
    if lq is not None and len(lq):
        l = lq[lq["dt"] >= t0].copy()
        l["h"] = l["dt"].dt.floor("h")
        g = l.groupby(["h", "pos"])["btc"].sum().unstack(fill_value=0)
        if "long" in g:
            a.bar(g.index, -g["long"], width=0.035, color=DN, alpha=0.75, label="Long bị thanh lý")
        if "short" in g:
            a.bar(g.index, g["short"], width=0.035, color=UP, alpha=0.75, label="Short bị thanh lý")
        a.axhline(0, color=GRAY, lw=0.6)
        cov = s["footprint"]["info"].get("liquidations", {}).get("covers_hours")
        if cov:
            a.text(0.997, 0.96, f"dữ liệu thanh lý ~{cov} giờ gần nhất", transform=a.transAxes, color=GRAY, fontsize=7,
                   ha="right", va="top")
    plabel(a, "Thanh lý mỗi giờ (BTC): đỏ = long bị thanh lý, xanh = short bị thanh lý")
    a.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%H:%M", tz=tz))
    a.tick_params(axis="x", colors=FG, labelsize=7.5)
    fig.savefig(path, facecolor=BG, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- M5 (tham khảo khi có tín hiệu)


def draw_m5(s, c, path, n=150):
    sg = s["signal"]; m5 = c["m5"]; r = c["m5_res"]
    x0 = max(len(m5) - n, 0); w = m5.iloc[x0:].reset_index(drop=True); n = len(w)
    X = lambda i: i - x0  # noqa: E731
    fig, (ax, av) = subplot_grid([5, 1], h=9.5)
    lo = min(w["low"].min(), sg["sl"]); hi = max(w["high"].max(), sg["tp1"], sg["tp2"] or 0)
    pad = (hi - lo) * 0.06; ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(-1, n + 18)
    structure_layer(ax, w, x0, n, r["sw"], r["ev"], r["fv"], r["ob"], pad, X)
    R = []
    for p, col, sw_ in liquidity_layer(c, R, (lo - pad, hi + pad)):
        ax.hlines(p, 0, n, color=col, lw=0.8, ls=(0, (6, 3)), alpha=0.5)
    candles(ax, w)
    place_right(ax, R, n + 1, lo - pad, hi + pad)
    plan_layer(ax, sg, n, 0.6)
    av.bar(range(n), w["volume"], color=[UP if rr.close >= rr.open else DN for rr in w.itertuples()], width=0.7, alpha=0.6)
    xticks(av, w)
    ax.set_title(f"{s['source']} · M5 (tham khảo) · {s['time_vn']} · {sg['dir']} · {sg['note']} · R:R TP2 {sg['rr_tp2']}",
                 color=FG, fontsize=12, loc="left")
    fig.savefig(path, facecolor=BG, bbox_inches="tight"); plt.close(fig)


def draw_all(s, c, out):
    jobs = [("chart_m15.png", draw_m15), ("chart_h1.png", draw_h1), ("chart_mtf.png", draw_mtf), ("chart_flow.png", draw_flow)]
    if s.get("signal"):
        jobs.append(("chart_m5.png", draw_m5))
    elif os.path.exists(os.path.join(out, "chart_m5.png")):
        os.remove(os.path.join(out, "chart_m5.png"))
    errs = []
    for name, fn in jobs:
        try:
            fn(s, c, os.path.join(out, name))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            errs.append(f"{name}: {e}")
            plt.close("all")
    s["chart_errors"] = errs

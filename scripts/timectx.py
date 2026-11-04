"""Hợp lưu thời gian: phiên/kill zone, Silver Bullet, ICT macro, giá mở cửa quan trọng, lịch tin, funding."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

VNZ = ZoneInfo("Asia/Ho_Chi_Minh")
NYZ = ZoneInfo("America/New_York")

# (tên, múi giờ, bắt đầu, kết thúc, loại)  — kz: cửa sổ giao dịch chính; sb: Silver Bullet
SESSIONS = [
    ("Phiên sáng (Á)", VNZ, (7, 0), (11, 0), "kz"),
    ("London KZ", NYZ, (2, 0), (5, 0), "kz"),
    ("New York KZ", NYZ, (7, 0), (10, 0), "kz"),
    ("Silver Bullet London", NYZ, (3, 0), (4, 0), "sb"),
    ("Silver Bullet AM", NYZ, (10, 0), (11, 0), "sb"),
    ("Silver Bullet PM", NYZ, (14, 0), (15, 0), "sb"),
]
# ICT macro (giờ New York)
MACROS = [((2, 33), (3, 0)), ((4, 3), (4, 30)), ((8, 50), (9, 10)), ((9, 50), (10, 10)), ((10, 50), (11, 10)),
          ((11, 50), (12, 10)), ((13, 10), (13, 40)), ((15, 15), (15, 45))]
BIG_COUNTRIES = ("USD", "JPY", "CNY", "ALL")


def vn(dt):
    return dt.astimezone(VNZ).strftime("%H:%M")


def intervals(t0, t1):
    """Mọi phiên/SB/macro giao với [t0, t1] (UTC)."""
    out = []
    day = (t0 - timedelta(days=1)).date()
    while day <= (t1 + timedelta(days=1)).date():
        for name, tz, a, b, kind in SESSIONS:
            s = datetime(day.year, day.month, day.day, a[0], a[1], tzinfo=tz).astimezone(timezone.utc)
            e = datetime(day.year, day.month, day.day, b[0], b[1], tzinfo=tz).astimezone(timezone.utc)
            if e > t0 and s < t1:
                out.append({"name": name, "kind": kind, "start": s, "end": e})
        for a, b in MACROS:
            s = datetime(day.year, day.month, day.day, a[0], a[1], tzinfo=NYZ).astimezone(timezone.utc)
            e = datetime(day.year, day.month, day.day, b[0], b[1], tzinfo=NYZ).astimezone(timezone.utc)
            if e > t0 and s < t1:
                out.append({"name": f"Macro {a[0]:02d}:{a[1]:02d}", "kind": "macro", "start": s, "end": e})
        day += timedelta(days=1)
    uniq = {(x["name"], x["start"]): x for x in out}
    return sorted(uniq.values(), key=lambda x: x["start"])


def parse_events(cal, t0, t1):
    out = []
    for e in (cal or {}).get("events", []):
        try:
            dt = datetime.fromisoformat(e["date"]).astimezone(timezone.utc)
        except Exception:  # noqa: BLE001
            continue
        if not (t0 <= dt <= t1):
            continue
        big = e.get("impact") == "High" and e.get("country") in BIG_COUNTRIES
        out.append({"dt": dt, "time_vn": dt.astimezone(VNZ).strftime("%d/%m %H:%M"), "title": e.get("title"),
                    "country": e.get("country"), "impact": e.get("impact"), "big": big,
                    "forecast": e.get("forecast") or "", "previous": e.get("previous") or ""})
    return out


def price_at(df, dt):
    """Giá mở cửa của nến chứa thời điểm dt (None nếu ngoài dữ liệu)."""
    if df is None or not len(df) or dt < df["dt"].iloc[0] or dt > df["dt"].iloc[-1] + timedelta(hours=1):
        return None
    k = df["dt"].searchsorted(dt, side="right") - 1
    return float(df["open"].iloc[max(k, 0)])


def key_opens(now, m5, m15, d1):
    ny = now.astimezone(NYZ)
    ny_mid = datetime(ny.year, ny.month, ny.day, tzinfo=NYZ).astimezone(timezone.utc)
    ny_830 = datetime(ny.year, ny.month, ny.day, 8, 30, tzinfo=NYZ).astimezone(timezone.utc)
    utc_day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    week = utc_day - timedelta(days=utc_day.weekday())
    month = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    src = lambda dt: price_at(m5, dt) or price_at(m15, dt)  # noqa: E731
    out = {}
    out["Mở cửa ngày (07:00 VN)"] = price_at(d1, utc_day) or src(utc_day)
    out["NY midnight open"] = src(ny_mid)
    if now >= ny_830:
        out["NY 08:30 open"] = src(ny_830)
    out["Mở cửa tuần"] = price_at(d1, week)
    out["Mở cửa tháng"] = price_at(d1, month)
    return {k: round(v, 1) for k, v in out.items() if v}, {"ny_mid": ny_mid, "ny_830": ny_830, "day": utc_day}


def context(now, cal, m5, m15, d1):
    ivs = intervals(now - timedelta(hours=60), now + timedelta(hours=30))
    cur = [x for x in ivs if x["start"] <= now < x["end"]]
    kz_now = next((x for x in cur if x["kind"] == "kz"), None)
    nxt_kz = next((x for x in ivs if x["kind"] == "kz" and x["start"] > now), None)
    opens, anchors = key_opens(now, m5, m15, d1)
    ev = parse_events(cal, now - timedelta(hours=60), now + timedelta(hours=30))
    upcoming = [e for e in ev if e["dt"] >= now - timedelta(minutes=30)]
    big_near = [e for e in ev if e["big"] and abs((e["dt"] - now).total_seconds()) <= 30 * 60]
    next_big = next((e for e in ev if e["big"] and e["dt"] > now), None)
    nf = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    funding_next = min(t for t in (nf + timedelta(hours=h) for h in (0, 8, 16, 24)) if t > now)
    dow = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"][now.astimezone(VNZ).weekday()]
    weekly_note = {
        "Thứ Hai": "đầu tuần: thường tạo range, dễ quét 2 đầu",
        "Thứ Ba": "hay tạo đỉnh/đáy tuần (ICT weekly profile)",
        "Thứ Tư": "hay tạo đỉnh/đáy tuần hoặc đảo chiều giữa tuần",
        "Thứ Năm": "tiếp diễn xu hướng tuần hoặc đảo chiều theo tin",
        "Thứ Sáu": "chốt lời cuối tuần, cẩn thận đảo chiều chiều thứ Sáu NY",
        "Thứ Bảy": "cuối tuần: thanh khoản mỏng, hạn chế giao dịch",
        "Chủ Nhật": "cuối tuần: thanh khoản mỏng, chờ mở cửa tuần",
    }[dow]
    return {
        "now_vn": now.astimezone(VNZ).strftime("%Y-%m-%d %H:%M"),
        "in_window": kz_now["name"] if kz_now else None,
        "window_end_vn": vn(kz_now["end"]) if kz_now else None,
        "window_end_utc": kz_now["end"].isoformat() if kz_now else None,
        "next_window": {"name": nxt_kz["name"], "start_vn": vn(nxt_kz["start"]), "end_vn": vn(nxt_kz["end"]),
                        "in_min": int((nxt_kz["start"] - now).total_seconds() // 60)} if nxt_kz else None,
        "silver_bullet_now": next((x["name"] for x in cur if x["kind"] == "sb"), None),
        "macro_now": next((x["name"] for x in cur if x["kind"] == "macro"), None),
        "windows_today_vn": [f"{x['name']} {vn(x['start'])}–{vn(x['end'])}" for x in ivs
                             if x["kind"] in ("kz", "sb") and now - timedelta(hours=2) <= x["end"]
                             and x["start"] <= now + timedelta(hours=18)],
        "opens": opens,
        "news_upcoming": [{k: v for k, v in e.items() if k != "dt"} for e in upcoming[:10]],
        "news_block": [f"{e['time_vn']} {e['country']} {e['title']}" for e in big_near],
        "no_trade_news": bool(big_near),
        "next_big_news": ({"time_vn": next_big["time_vn"], "title": f"{next_big['country']} {next_big['title']}",
                           "in_min": int((next_big["dt"] - now).total_seconds() // 60)} if next_big else None),
        "funding_next_vn": vn(funding_next), "funding_in_min": int((funding_next - now).total_seconds() // 60),
        "day_of_week": f"{dow}: {weekly_note}",
        "_intervals": ivs, "_events": ev, "_anchors": anchors,
    }


def time_score(ctx, direction, entry):
    """0–3 điểm: trong cửa sổ phiên, trong Silver Bullet/macro, entry đúng phía giá mở cửa (PO3)."""
    pts, why = 0, []
    if ctx["in_window"]:
        pts += 1; why.append(f"trong {ctx['in_window']} (đến {ctx['window_end_vn']})")
    elif ctx["next_window"] and ctx["next_window"]["in_min"] <= 45:
        why.append(f"{ctx['next_window']['name']} mở sau {ctx['next_window']['in_min']} phút")
    if ctx["silver_bullet_now"] or ctx["macro_now"]:
        pts += 1; why.append(ctx["silver_bullet_now"] or ctx["macro_now"])
    o = ctx["opens"]
    refs = [v for k, v in o.items() if k in ("Mở cửa ngày (07:00 VN)", "NY midnight open")]
    if entry and refs and direction in ("LONG", "SHORT"):
        if direction == "SHORT" and all(entry > r for r in refs):
            pts += 1; why.append("entry trên giá mở cửa ngày & NY midnight (PO3: bán ở premium của ngày)")
        elif direction == "LONG" and all(entry < r for r in refs):
            pts += 1; why.append("entry dưới giá mở cửa ngày & NY midnight (PO3: mua ở discount của ngày)")
    if ctx["no_trade_news"]:
        why.append("ĐANG TRONG ±30 phút tin lớn: " + "; ".join(ctx["news_block"]))
    return pts, why

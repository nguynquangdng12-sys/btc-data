"""Gửi cảnh báo Telegram ngay khi có tín hiệu (chạy sau analyze.py trong mỗi lượt 5 phút).

Cần 2 secret của GitHub: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID. Thiếu thì bỏ qua, không làm hỏng lượt chạy.
Chống gửi trùng bằng out/notify_state.json (được giữ qua các lượt trên nhánh live).

Các loại cảnh báo:
- TÍN HIỆU M15 mới (kind "live"): ghi rõ hợp lệ hay chưa, lý do nếu chưa (trong phiên mới gửi).
- LỆNH CHỜ mới (kind "plan").
- CHUẨN BỊ: giá cách entry của lệnh chờ ≤ 0,3%.
- Lệnh chờ đổi trạng thái (khớp / TP / SL / hủy / hết hạn).
Lần chạy đầu tiên chỉ ghi trạng thái, không gửi (tránh spam lịch sử cũ).

Chạy thử không gửi: python scripts/notify.py --dry-run
"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
import uuid

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.join(ROOT, "out")
STATE = os.path.join(OUT, "notify_state.json")
PREP_PCT = 0.3
HARD_TEXT = {
    "d1_direction": "ngược hướng D1", "rr_tp2_ge_2": "R:R TP2 < 2",
    "risk_lt_1_2pct": "risk ≥ 1,2% (SL quá xa)", "not_chased": "đuổi giá", "no_big_news_30m": "trong ±30 phút tin lớn",
}


def fmt(x):
    return f"{x:,.1f}".replace(",", "_").replace(".", ",").replace("_", ".") if isinstance(x, (int, float)) else str(x)


def setup_key(st):
    return f"{st['kind']}|{st['dir']}|{round(st['entry'] / 50) * 50:.0f}"


def plan_key(p):
    return f"{p.get('created_vn')}|{p.get('dir')}|{p.get('entry')}"


def levels(d):
    return (f"Entry: {fmt(d['entry'])}\nSL: {fmt(d['sl'])}"
            + (f" (risk {str(d['risk_pct_price']).replace('.', ',')}%)" if d.get("risk_pct_price") is not None else "")
            + f"\nTP1: {fmt(d['tp1'])} · TP2: {fmt(d['tp2'])}"
            + (f" (R:R TP2 {str(d['rr_tp2']).replace('.', ',')})" if d.get("rr_tp2") else ""))


def build_alerts(s, state):
    alerts = []  # (key, text, gửi kèm chart?)
    price, t = s.get("price"), s.get("time") or {}
    win = t.get("in_window")
    head = f"[{(s.get('time_vn') or '')[-5:]}] BTC OKX {fmt(price)}"
    st, ck = s.get("m15_setup"), s.get("m15_checklist") or {}

    if st and st.get("kind") == "live" and win:
        bad = [HARD_TEXT.get(k, k) for k, v in (ck.get("hard_rules") or {}).items() if not v]
        ok = st.get("valid") and ck.get("hard_ok")
        txt = (f"{head}\nTÍN HIỆU {st['dir']} M15 · {win} · {'HỢP LỆ' if ok else 'CHƯA HỢP LỆ'}\n{levels(st)}\n"
               f"Checklist {ck.get('passed', '?')}/{ck.get('total', 8)} · đa khung {s.get('mtf_alignment')} · "
               f"dòng tiền {(s.get('footprint') or {}).get('score')}\n"
               + (f"Lý do chưa hợp lệ: {', '.join(bad)}\n" if bad else "")
               + f"Mẫu: {st.get('note', '')}\nHĐ 2%: {st.get('contracts_2pct')} · 5%: {st.get('contracts_5pct')}")
        alerts.append((setup_key(st), txt, True))
    elif st and st.get("kind") == "plan":
        txt = (f"{head}\nLỆNH CHỜ {st['dir']} M15 (limit, chưa vào lệnh)\n{levels(st)}\n"
               f"Chỉ khớp trong phiên ({win or 'phiên kế tiếp'}), hủy nếu hết phiên chưa khớp.\n"
               f"HĐ 2%: {st.get('contracts_2pct')} · 5%: {st.get('contracts_5pct')}")
        alerts.append((setup_key(st), txt, True))

    for p in s.get("plan_tracking") or []:
        k, status = plan_key(p), p.get("status") or ""
        if status == "CHỜ KHỚP" and price and p.get("entry"):
            dist = abs(price / p["entry"] - 1) * 100
            if dist <= PREP_PCT and win:
                alerts.append((k + "|prep", f"{head}\nCHUẨN BỊ {p['dir']}: giá cách entry {dist:.2f}%\n{levels(p)}", False))
        if status != "CHỜ KHỚP":
            alerts.append((k + "|status|" + status, f"{head}\nLệnh chờ {p.get('dir')} {fmt(p.get('entry'))} (tạo {p.get('created_vn')}): {status}", False))
    return alerts


def tg(method, data, files=None):
    tok = os.environ["TELEGRAM_BOT_TOKEN"]
    url = f"https://api.telegram.org/bot{tok}/{method}"
    if files:
        b = uuid.uuid4().hex
        body = b""
        for k, v in data.items():
            body += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        for k, path in files.items():
            body += (f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{os.path.basename(path)}\"\r\n"
                     "Content-Type: image/png\r\n\r\n").encode() + open(path, "rb").read() + b"\r\n"
        body += f"--{b}--\r\n".encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    else:
        req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode())
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sp = os.path.join(OUT, "summary.json")
    if not os.path.exists(sp) or os.path.exists(os.path.join(OUT, "ANALYZE_FAILED")):
        print("notify: không có summary hợp lệ, bỏ qua"); return
    s = json.load(open(sp))
    first = not os.path.exists(STATE)
    state = {} if first else json.load(open(STATE))
    order = list(state.get("sent", []))
    sent = set(order)
    new = [x for x in build_alerts(s, state) if x[0] not in sent]
    can_send = bool(os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID")) and not a.dry_run
    if first and not a.dry_run:
        print(f"notify: lần đầu, ghi {len(new)} mục làm mốc, không gửi")
    for key, text, chart in new:
        if a.dry_run or not can_send or first:
            print("---", key, "\n" + text)
            if not can_send and not first and not a.dry_run:
                continue  # chưa cấu hình: không đánh dấu đã gửi để khi cấu hình xong vẫn nhận tin mới
        else:
            try:
                chat = os.environ["TELEGRAM_CHAT_ID"]
                tg("sendMessage", {"chat_id": chat, "text": text})
                img = os.path.join(OUT, "chart_m15.png")
                if chart and os.path.exists(img):
                    tg("sendPhoto", {"chat_id": chat, "caption": "M15"}, {"photo": img})
                print("notify: đã gửi", key)
            except Exception as e:  # noqa: BLE001
                print("notify: lỗi gửi", key, str(e).replace(os.environ.get("TELEGRAM_BOT_TOKEN", "#"), "***")[:300])
                continue
        sent.add(key); order.append(key)
    if not a.dry_run:
        state["sent"] = list(dict.fromkeys(order))[-500:]
        json.dump(state, open(STATE, "w"), ensure_ascii=False, indent=0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001  cảnh báo không được làm hỏng lượt chạy
        print("notify: lỗi", e, file=sys.stderr)

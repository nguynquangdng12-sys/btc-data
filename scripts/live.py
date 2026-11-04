"""Vòng chạy trên GitHub Actions: tải dữ liệu -> phân tích -> đẩy kết quả lên nhánh `live`.

- Trong cửa sổ phiên (sáng 06:00–11:05 giờ VN; London 01:00–05:05 và New York 06:00–10:05 giờ New York,
  tự theo giờ mùa hè/đông của Mỹ) lặp lại mỗi 5 phút, tối đa ~55 phút mỗi lần chạy workflow.
- Ngoài phiên: chạy một lần.
- Nhánh `live` (force-push, chỉ giữ 1 commit): out/*.png, out/summary.json, out/plan_history.json, data/*.
- Nhánh main: commit dữ liệu tối đa mỗi ~15 phút (giữ tương thích).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
VN = ZoneInfo("Asia/Ho_Chi_Minh")
NY = ZoneInfo("America/New_York")
WINDOWS = [(VN, (6, 0), (11, 5)), (NY, (1, 0), (5, 5)), (NY, (6, 0), (10, 5))]
MAX_LOOP = timedelta(minutes=55)


def sh(cmd, cwd=ROOT, check=False, quiet=False):
    r = subprocess.run(cmd, cwd=cwd, shell=True, text=True, capture_output=True)
    if not quiet and (r.stdout.strip() or r.stderr.strip()):
        print((r.stdout + r.stderr).strip()[-3000:])
    if check and r.returncode:
        raise RuntimeError(f"lỗi: {cmd}")
    return r


def active_until(now):
    ends = []
    for tz, (h0, m0), (h1, m1) in WINDOWS:
        loc = now.astimezone(tz)
        a = loc.replace(hour=h0, minute=m0, second=0, microsecond=0)
        b = loc.replace(hour=h1, minute=m1, second=0, microsecond=0)
        if a <= loc < b:
            ends.append(b.astimezone(timezone.utc))
    return max(ends) if ends else None


def remote_url():
    tok, repo = os.environ.get("GH_TOKEN"), os.environ.get("REPO")
    return f"https://x-access-token:{tok}@github.com/{repo}.git" if tok and repo else None


def restore_history():
    """Lấy plan_history.json (theo dõi lệnh chờ) và calendar.json (cache lịch tin) từ nhánh live."""
    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    if sh("git fetch -q --depth 1 origin live", quiet=True).returncode:
        print("Chưa có nhánh live")
        return
    for src, dst, head in (("out/plan_history.json", "out/plan_history.json", "["), ("data/calendar.json", "data/calendar.json", "{")):
        r = sh(f"git show FETCH_HEAD:{src}", quiet=True)
        if r.returncode == 0 and r.stdout.strip().startswith(head):
            open(os.path.join(ROOT, dst), "w").write(r.stdout)
            print("Đã khôi phục", src)


def publish_live(stamp):
    url = remote_url()
    if not url:
        print("Không có GH_TOKEN/REPO: bỏ qua đẩy nhánh live")
        return
    tmp = tempfile.mkdtemp()
    try:
        shutil.copytree(os.path.join(ROOT, "out"), os.path.join(tmp, "out"))
        shutil.copytree(os.path.join(ROOT, "data"), os.path.join(tmp, "data"))
        open(os.path.join(tmp, "README.md"), "w").write(
            "# btc-data · live\n\nNhánh tự động (force-push mỗi lần chạy). Chart và summary mới nhất trong out/.\n")
        sh("git init -q && git checkout -q --orphan live && git add -A && "
           "git -c user.name=btc-data-bot -c user.email=actions@users.noreply.github.com "
           f"commit -q -m 'live: {stamp}'", cwd=tmp, check=True)
        r = sh(f"git push -q -f {url} live", cwd=tmp, quiet=True)
        print("Đẩy nhánh live:", "OK" if r.returncode == 0 else ("LỖI " + r.stderr.replace(os.environ.get('GH_TOKEN', '#'), '***')[-500:]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def commit_main(stamp):
    sh("git add data/btc_*.csv data/meta.json")
    if sh("git diff --cached --quiet", quiet=True).returncode == 0:
        return
    sh("git -c user.name=btc-data-bot -c user.email=actions@users.noreply.github.com "
       f"commit -q -m 'data: {stamp}'")
    for _ in range(3):
        sh("git pull -q --rebase -X theirs origin main", quiet=True)
        if sh("git push -q", quiet=True).returncode == 0:
            print("Commit dữ liệu lên main: OK")
            return
        time.sleep(3)
    print("Commit dữ liệu lên main: LỖI (bỏ qua)")


def run_once(it):
    t0 = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    print(f"\n===== Lượt {it + 1} · {stamp}")
    r = subprocess.run([sys.executable, "scripts/fetch.py"], cwd=ROOT)
    if r.returncode:
        print("fetch lỗi: bỏ qua lượt này")
        return
    r = subprocess.run([sys.executable, "scripts/analyze.py", "--out", "out", "--quiet"], cwd=ROOT)
    if r.returncode:
        print("analyze lỗi: vẫn đẩy dữ liệu, kèm cờ lỗi")
        open(os.path.join(ROOT, "out", "ANALYZE_FAILED"), "w").write(stamp)
    elif os.path.exists(os.path.join(ROOT, "out", "ANALYZE_FAILED")):
        os.remove(os.path.join(ROOT, "out", "ANALYZE_FAILED"))
    publish_live(stamp)
    if it % 3 == 0:
        commit_main(stamp)
    print(f"Lượt {it + 1} xong sau {time.time() - t0:.0f}s")


def main():
    start = datetime.now(timezone.utc)
    until = active_until(start)
    if until:
        until = min(until, start + MAX_LOOP)
        print(f"Trong cửa sổ phiên: lặp mỗi 5 phút đến {until:%H:%M} UTC")
    restore_history()
    it = 0
    while True:
        run_once(it)
        it += 1
        if not until:
            break
        now = datetime.now(timezone.utc)
        nxt = now.replace(second=25, microsecond=0) + timedelta(minutes=5 - now.minute % 5)
        if nxt - now > timedelta(minutes=5):
            nxt -= timedelta(minutes=5)
        if nxt >= until:
            break
        time.sleep(max((nxt - now).total_seconds(), 5))


if __name__ == "__main__":
    main()

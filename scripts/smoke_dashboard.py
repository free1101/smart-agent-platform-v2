"""控制台浏览器冒烟:真实启动 Streamlit,用 Playwright 驱动系统浏览器逐页检查渲染。

为什么需要这个脚本:
- dashboard 的很多问题是「只有真正跑起来才暴露」的,例如 `dashboard/app.py` 与顶层
  `app/` 包同名导致的循环导入、页面调用门面函数时参数不匹配、Streamlit trace 时间线
  全 0。pytest 只覆盖 `app/`,`scripts/smoke_dashboard.py` 补上渲染层这一环。
- 逐页截图保存在 `data/smoke/dashboard/`,页面报错时可直接看图定位。

用法:
    python scripts/smoke_dashboard.py
    python scripts/smoke_dashboard.py --port 8510 --browser chrome
    python scripts/smoke_dashboard.py --no-shots --headed

前置:
    pip install streamlit playwright
    浏览器复用系统已安装的 Edge / Chrome(`channel`),无需 `playwright install`。

退出码:
    0 = 所有页面无异常;1 = 有页面异常或服务未启动。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DEFAULT_SHOTS_DIR = PROJECT_ROOT / "data" / "smoke" / "dashboard"

# 浏览器 channel 尝试顺序:优先复用系统浏览器,最后才用 Playwright 自带 chromium
CHANNELS = ("msedge", "chrome", None)


def _wait_health(base_url: str, timeout: float) -> bool:
    """轮询 Streamlit 健康端点,直到服务就绪或超时。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/_stcore/health", timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - 服务尚未监听属于正常过程
            time.sleep(1)
    return False


def _start_server(port: int, log_path: Path | None) -> subprocess.Popen:
    """以子进程启动 Streamlit(项目根为 cwd),返回进程句柄。"""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(DASHBOARD_DIR / "app.py"),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    log = open(log_path, "w", encoding="utf-8") if log_path else subprocess.DEVNULL
    return subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def _stop_server(proc: subprocess.Popen) -> None:
    """关闭 Streamlit 进程树。

    Windows 上 `terminate()` 只结束父进程,Streamlit 派生的子进程仍占着端口,
    必须用 `taskkill /T` 按树结束,否则下次运行会撞端口。
    """
    if proc.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _discover_pages() -> list[tuple[str, str]]:
    """首页 + dashboard/pages/ 下每个页面,URL 由文件名派生(与 Streamlit 一致)。"""
    pages_dir = DASHBOARD_DIR / "pages"
    targets: list[tuple[str, str]] = [("/", "首页")]
    if pages_dir.is_dir():
        for path in sorted(pages_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            targets.append((f"/{path.stem}", path.stem))
    return targets


def _launch_browser(playwright, choice: str, headless: bool):
    """按 channel 优先级启动浏览器;`--browser` 显式指定时只用指定值。"""
    if choice != "auto":
        candidates: tuple = (None,) if choice == "chromium" else (choice,)
    else:
        candidates = CHANNELS

    errors: list[str] = []
    for channel in candidates:
        try:
            if channel is None:
                return playwright.chromium.launch(headless=headless)
            return playwright.chromium.launch(channel=channel, headless=headless)
        except Exception as exc:  # noqa: BLE001 - 换下一个 channel 继续尝试
            errors.append(f"{channel or 'chromium'}: {exc}")
    raise RuntimeError(
        "无法启动浏览器,请安装 Edge/Chrome,或执行 `playwright install chromium`。"
        f"尝试记录: {errors}"
    )


def _check_pages(
    base_url: str,
    pages: list[tuple[str, str]],
    shots_dir: Path | None,
    wait_ms: int,
    browser_choice: str,
    headless: bool,
) -> dict:
    """逐页访问并采集渲染异常。返回结构化报告。"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - 依赖缺失时给出明确指引
        raise RuntimeError("需要 Playwright,请执行: pip install playwright") from exc

    report: dict = {"pages": [], "console_errors": [], "console_noise": []}

    def _on_console(msg) -> None:
        if msg.type != "error":
            return
        text = msg.text[:200]
        # Streamlit 自身会请求 favicon、试探 MPA 路由,这些提示与页面渲染无关
        noise = ("Failed to load resource" in text and "404" in text) or (
            "does not seem to exist" in text
        )
        (report["console_noise"] if noise else report["console_errors"]).append(text)

    with sync_playwright() as playwright:
        browser = _launch_browser(playwright, browser_choice, headless)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.set_default_timeout(30000)
        page.on("console", _on_console)
        page.on("pageerror", lambda e: report["console_errors"].append(f"pageerror: {str(e)[:200]}"))

        try:
            for index, (href, label) in enumerate(pages):
                page.goto(base_url + href, wait_until="domcontentloaded")
                page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=30000)
                try:
                    # 冷启动首屏可能慢于固定等待,等标题出现再截图,避免取到空标题
                    page.wait_for_selector("h1", timeout=15000)
                except Exception:  # noqa: BLE001 - 没有 h1 的页面不算失败
                    pass
                page.wait_for_timeout(wait_ms)

                exc = page.locator('[data-testid="stException"]')
                texts = [exc.nth(i).inner_text()[:400] for i in range(exc.count())]
                h1 = page.locator("h1").first
                title = h1.inner_text().strip() if h1.count() else ""

                shot = ""
                if shots_dir is not None:
                    shot_path = shots_dir / f"{index:02d}_{href.strip('/') or 'home'}.png"
                    page.screenshot(path=str(shot_path), full_page=True)
                    shot = str(shot_path.relative_to(PROJECT_ROOT))

                report["pages"].append(
                    {
                        "href": href,
                        "label": label,
                        "title": title,
                        "exceptions": exc.count(),
                        "detail": texts,
                        "screenshot": shot,
                    }
                )
        finally:
            browser.close()

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="控制台浏览器冒烟(Streamlit + Playwright)")
    parser.add_argument("--port", type=int, default=8503, help="临时 Streamlit 端口")
    parser.add_argument("--browser", default="auto", choices=["auto", "msedge", "chrome", "chromium"])
    parser.add_argument("--no-shots", action="store_true", help="不落截图")
    parser.add_argument("--headed", action="store_true", help="显示浏览器窗口(默认无头)")
    parser.add_argument("--wait-ms", type=int, default=3500, help="每页渲染等待毫秒")
    parser.add_argument("--timeout", type=float, default=60.0, help="等待服务就绪的秒数")
    parser.add_argument("--json", action="store_true", help="额外输出 JSON 报告")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    base_url = f"http://127.0.0.1:{args.port}"
    shots_dir = None if args.no_shots else DEFAULT_SHOTS_DIR
    if shots_dir is not None:
        shots_dir.mkdir(parents=True, exist_ok=True)

    log_path = PROJECT_ROOT / "data" / "smoke" / "streamlit.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    pages = _discover_pages()
    server = _start_server(args.port, log_path)
    try:
        if not _wait_health(base_url, args.timeout):
            print(f"[失败] Streamlit 未在 {args.timeout:.0f}s 内就绪,日志: {log_path}")
            return 1
        report = _check_pages(
            base_url, pages, shots_dir, args.wait_ms, args.browser, not args.headed
        )
    except RuntimeError as exc:
        print(f"[失败] {exc}")
        return 1
    finally:
        _stop_server(server)

    broken = [item for item in report["pages"] if item["exceptions"]]
    print(f"{'页面':<22}{'标题':<12}{'异常'}")
    print("-" * 44)
    for item in report["pages"]:
        print(f"{item['href']:<22}{item['title'] or '-':<12}{item['exceptions']}")
    for item in broken:
        print(f"\n[异常] {item['href']}")
        for text in item["detail"]:
            print("  " + text.replace("\n", "\n  ")[:600])

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["console_errors"]:
        print(f"\n浏览器 console 错误 {len(report['console_errors'])} 条(前 5 条):")
        for text in report["console_errors"][:5]:
            print("  " + text)
    if report["console_noise"]:
        print(f"\n控制台提示 {len(report['console_noise'])} 条(Streamlit 自身路由/静态资源,不影响渲染)")
    if shots_dir is not None:
        print(f"\n截图: {shots_dir.relative_to(PROJECT_ROOT)}")

    total = len(report["pages"])
    ok = total - len(broken)
    print(f"\n结果: {ok}/{total} 页面正常")
    return 0 if not broken else 1


if __name__ == "__main__":
    raise SystemExit(main())

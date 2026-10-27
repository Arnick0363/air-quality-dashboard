"""下载全国城市小时文件，汇总成日值，并计算空间自相关和次日预测。

原始 CSV 不保留。月度中间结果在 data/raw/china_months，已被 gitignore。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from calendar import monthrange
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.china_clean import city_day_from_wide
from src.china_figures import save_china_figures
from src.china_settings import CHINA_END, CHINA_START, HOURLY_TYPES, LOCATION_LIST_URLS
from src.china_summary import build_summary, render_markdown
from src.forecast_model import evaluate_forecast
from src.paths import (
    CHINA_DAILY_PATH,
    CHINA_FINDINGS_MD,
    CHINA_FORECAST_PATH,
    CHINA_LOCAL_PATH,
    CHINA_META_PATH,
    CHINA_MONTHS,
    CHINA_SUMMARY_JSON,
    LOCATION_LIST,
)
from src.regions import match_cities, read_location_list

BASE_URL = "https://quotsoft.net/air/data/china_cities_{day}.csv"
HEADERS = {"User-Agent": "air-quality-dashboard educational research"}


def month_spans(start: date, end: date):
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        last = date(cursor.year, cursor.month, monthrange(cursor.year, cursor.month)[1])
        yield max(cursor, start), min(last, end)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)


def fetch_day(day: date) -> tuple[date, pd.DataFrame | None, dict, str]:
    url = BASE_URL.format(day=day.strftime("%Y%m%d"))
    last_error = "error"
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=60, headers=HEADERS)
            if response.status_code == 404:
                return day, None, {}, "missing"
            response.raise_for_status()
            text = response.content.decode("utf-8-sig", errors="replace")
            if "date" not in text.split("\n", 1)[0]:
                last_error = "bad-header"
                continue
            frame = pd.read_csv(io.StringIO(text))
            daily, qc = city_day_from_wide(frame)
            daily.insert(0, "date", pd.Timestamp(day))
            return day, daily, qc, "ok"
        except Exception as exc:  # 网络失败要重试，不让一天毁掉整月
            last_error = f"{type(exc).__name__}"
            if attempt == 2:
                return day, None, {}, last_error
    return day, None, {}, last_error


def download_months(start: date, end: date, workers: int) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    CHINA_MONTHS.mkdir(parents=True, exist_ok=True)
    report = {"ok": 0, "missing": 0, "error": 0, "missing_days": [], "invalid_hours": {name: 0 for name in HOURLY_TYPES}}
    for pass_id in (1, 2):
        print(f"PASS {pass_id}", flush=True)
        if pass_id == 2:
            report["error"] = 0
            report["missing"] = 0
            report["missing_days"] = []
        _download_pass(start, end, workers, report)
    return report


def _download_pass(start: date, end: date, workers: int, report: dict) -> None:
    from concurrent.futures import ThreadPoolExecutor

    for first, last in month_spans(start, end):
        path = CHINA_MONTHS / f"{first:%Y-%m}.parquet"
        have: set[date] = set()
        old = None
        if path.exists() and path.stat().st_size > 0:
            old = pd.read_parquet(path)
            have = set(pd.to_datetime(old["date"]).dt.date)
        days = []
        cursor = first
        while cursor <= last:
            if cursor not in have:
                days.append(cursor)
            cursor += timedelta(days=1)
        if not days:
            print(f"SKIP {path.name} days={len(have)}", flush=True)
            continue
        print(f"GET {first:%Y-%m} missing={len(days)} have={len(have)}", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(fetch_day, days))
        pieces = [] if old is None else [old]
        ok = missing = error = 0
        for day, frame, qc, status in results:
            if status == "ok" and frame is not None:
                ok += 1
                report["ok"] += 1
                pieces.append(frame)
                for key, value in qc.items():
                    report["invalid_hours"][key] = report["invalid_hours"].get(key, 0) + int(value)
            elif status == "missing":
                missing += 1
                report["missing"] += 1
                report["missing_days"].append(day.isoformat())
            else:
                error += 1
                report["error"] += 1
                report["missing_days"].append(f"{day.isoformat()}:{status}")
        if len(pieces) == (0 if old is None else 1):
            print(f"EMPTY {first:%Y-%m} missing={missing} error={error}", flush=True)
            continue
        month = pd.concat(pieces, ignore_index=True)
        month["date"] = pd.to_datetime(month["date"])
        month = month.drop_duplicates(["date", "city"], keep="last")
        month.to_parquet(path, index=False)
        print(
            f"WROTE {path.name} rows={len(month)} dates={month['date'].nunique()} ok={ok} missing={missing} error={error}",
            flush=True,
        )


def load_months() -> pd.DataFrame:
    files = sorted(CHINA_MONTHS.glob("*.parquet"))
    if not files:
        raise RuntimeError("没有月度日值。先完成下载。")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.drop_duplicates(["date", "city"], keep="last")
    return frame.sort_values(["date", "city"]).reset_index(drop=True)


def ensure_gazetteer() -> Path:
    if LOCATION_LIST.exists() and LOCATION_LIST.stat().st_size > 1000:
        return LOCATION_LIST
    LOCATION_LIST.parent.mkdir(parents=True, exist_ok=True)
    last = "没有尝试"
    for url in LOCATION_LIST_URLS:
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            LOCATION_LIST.write_bytes(response.content)
            print(f"GAZETTEER {url} bytes={len(response.content)}", flush=True)
            return LOCATION_LIST
        except Exception as exc:
            last = f"{url} {exc}"
            print(f"GAZETTEER_FAIL {last}", flush=True)
    raise RuntimeError(f"城市坐标表下载失败：{last}")


def recount_files(frame: pd.DataFrame, start: date, end: date, report: dict) -> dict:
    """跳过的月份不会出现在本次下载计数里，这里按日期把缺失补齐。"""
    have = set(pd.to_datetime(frame["date"]).dt.date.unique())
    missing = []
    cursor = start
    while cursor <= end:
        if cursor not in have:
            missing.append(cursor.isoformat())
        cursor += timedelta(days=1)
    report["missing_days"] = missing
    report["ok"] = len(have)
    report["missing"] = len(missing)
    return report


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return json_ready(value.item())
        except Exception:
            pass
    return value


def analyze(start: date, end: date, report: dict) -> None:
    qc_path = CHINA_MONTHS.parent / "china_qc.json"
    if report.get("invalid_hours") and any(report.get("invalid_hours", {}).values()):
        qc_path.write_text(json.dumps(json_ready(report), ensure_ascii=False), encoding="utf-8")
    elif qc_path.exists():
        saved = json.loads(qc_path.read_text(encoding="utf-8"))
        report.setdefault("invalid_hours", saved.get("invalid_hours", {}))
    daily = load_months()
    daily = daily.loc[(daily["date"] >= pd.Timestamp(start)) & (daily["date"] <= pd.Timestamp(end))].copy()
    report = recount_files(daily, start, end, report)
    gazetteer = read_location_list(ensure_gazetteer())
    meta = match_cities(sorted(daily["city"].unique()), gazetteer)
    unmatched = meta.loc[meta["lat"].isna(), "city"].tolist()
    print(f"CITIES {len(meta)} UNMATCHED {len(unmatched)} {unmatched[:30]}", flush=True)
    print("FORECAST start", flush=True)
    ml, predictions = evaluate_forecast(daily, meta)
    print("SPATIAL start", flush=True)
    summary, local = build_summary(daily, meta, report, ml)
    CHINA_DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(CHINA_DAILY_PATH, index=False)
    meta.to_parquet(CHINA_META_PATH, index=False)
    local.to_parquet(CHINA_LOCAL_PATH, index=False)
    if not predictions.empty:
        predictions.to_parquet(CHINA_FORECAST_PATH, index=False)
    CHINA_SUMMARY_JSON.write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    CHINA_FINDINGS_MD.write_text(render_markdown(summary), encoding="utf-8")
    save_china_figures(summary)
    print("PREPARE_CHINA_OK", flush=True)
    print(f"daily_bytes={CHINA_DAILY_PATH.stat().st_size}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=CHINA_START.isoformat())
    parser.add_argument("--end", default=CHINA_END.isoformat())
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    report = {"ok": 0, "missing": 0, "error": 0, "missing_days": [], "invalid_hours": {}}
    if not args.analyze_only:
        report = download_months(start, end, args.workers)
    analyze(start, end, report)


if __name__ == "__main__":
    main()

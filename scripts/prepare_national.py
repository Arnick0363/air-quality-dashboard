"""下载全国城市小时数据，整理成日值，并计算空间统计和预测。

原始逐小时文件不保留。中断后已完成的月份会跳过。
"""

from __future__ import annotations

import json
import sys
import time
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.national_analyze import analyze_national
from src.national_clean import load_qweather_gazetteer, match_cities, parse_china_cities_csv
from src.national_report import dump_summary, upsert_national_findings
from src.paths import (
    FINDINGS_MD,
    NATIONAL_CITIES_PATH,
    NATIONAL_DAILY_PATH,
    NATIONAL_LISA_PATH,
    NATIONAL_PARTS,
    NATIONAL_PRED_PATH,
    NATIONAL_SUMMARY_JSON,
)
from src.settings import MONTH_TO_SEASON

START = date(2023, 1, 1)
END = date(2024, 12, 31)
GAZETTEER_URL = "https://raw.githubusercontent.com/qwd/LocationList/master/China-City-List-latest.csv"
CITY_URL = "https://quotsoft.net/air/data/china_cities_{ymd}.csv"
WORKERS = 2


def _month_starts(start: date, end: date) -> list[date]:
    cursor = date(start.year, start.month, 1)
    months = []
    while cursor <= end:
        months.append(cursor)
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return months


def _days_in_month(month_start: date, start: date, end: date) -> list[date]:
    last = monthrange(month_start.year, month_start.month)[1]
    days = []
    for day_number in range(1, last + 1):
        current = date(month_start.year, month_start.month, day_number)
        if start <= current <= end:
            days.append(current)
    return days


def _fetch_day(current: date) -> tuple[date, pd.DataFrame | None, str]:
    url = CITY_URL.format(ymd=current.strftime("%Y%m%d"))
    last_error = "failed"
    for attempt in range(6):
        try:
            response = requests.get(
                url,
                timeout=45,
                headers={"User-Agent": "air-quality-dashboard research download"},
            )
            if response.status_code == 404:
                return current, None, "404"
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"http {response.status_code}"
                time.sleep(8 + attempt * 7)
                continue
            if response.status_code != 200:
                last_error = f"http {response.status_code}"
                time.sleep(2 * (attempt + 1))
                continue
            response.encoding = "utf-8"
            text = response.text
            if "type" not in text[:1500] or "PM2.5" not in text[:8000]:
                return current, None, "not-csv"
            time.sleep(0.35)
            return current, parse_china_cities_csv(text), "ok"
        except (requests.RequestException, pd.errors.ParserError, ValueError, UnicodeError) as exc:
            last_error = type(exc).__name__
            time.sleep(4 + attempt * 4)
    return current, None, last_error


def _download_month(month_start: date, start: date, end: date) -> dict:
    NATIONAL_PARTS.mkdir(parents=True, exist_ok=True)
    tag = f"{month_start:%Y-%m}"
    manifest_path = NATIONAL_PARTS / f"{tag}.json"
    data_path = NATIONAL_PARTS / f"{tag}.parquet"
    days = _days_in_month(month_start, start, end)
    existing = None
    if manifest_path.exists() and data_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        retryable = {
            day for day, status in manifest.get("errors", {}).items()
            if status != "404"
        }
        if not retryable:
            print(f"跳过已完成的 {tag}", flush=True)
            return manifest
        existing = pd.read_parquet(data_path)
        have = set(pd.to_datetime(existing["date"]).dt.date)
        days = [current for current in days if current not in have]
        print(f"{tag} 补下 {len(days)} 天", flush=True)
        if not days:
            return manifest
    frames = []
    errors = {}
    if existing is not None and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for day, status in previous.get("errors", {}).items():
            if status == "404":
                errors[day] = status
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_fetch_day, current) for current in days]
        done = 0
        for future in as_completed(futures):
            current, frame, status = future.result()
            done += 1
            if frame is None:
                errors[current.isoformat()] = status
            else:
                frames.append(frame)
            if done % 8 == 0 or done == len(days):
                print(f"{tag} {done}/{len(days)}", flush=True)
    parts = []
    if existing is not None:
        parts.append(existing)
    parts.extend(frames)
    if not parts:
        raise RuntimeError(f"{tag} 没有任何成功的城市日文件")
    month = pd.concat(parts, ignore_index=True)
    month.to_parquet(data_path, index=False)
    manifest = {
        "month": tag,
        "ok_days": int(month["date"].nunique()),
        "errors": errors,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _load_gazetteer() -> pd.DataFrame:
    urls = [
        GAZETTEER_URL,
        "https://cdn.jsdelivr.net/gh/qwd/LocationList@master/China-City-List-latest.csv",
    ]
    last_error = "没有尝试"
    for url in urls:
        try:
            response = requests.get(url, timeout=60, headers={"User-Agent": "air-quality-dashboard"})
            response.raise_for_status()
            response.encoding = "utf-8"
            gazetteer = load_qweather_gazetteer(response.text)
            print(f"坐标表 {len(gazetteer)} 行，来自 {url}", flush=True)
            return gazetteer
        except (requests.RequestException, ValueError, pd.errors.ParserError) as exc:
            last_error = f"{url}：{type(exc).__name__} {exc}"
            print(last_error, flush=True)
    raise RuntimeError("城市坐标表下载失败：" + last_error)


def _attach(daily: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["city", "province", "region", "lon", "lat", "matched", "adcode", "source_name"]
    out = daily.merge(cities[meta_cols], on="city", how="left")
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["season"] = out["month"].map(MONTH_TO_SEASON)
    out["matched"] = out["matched"].fillna(False)
    out["province"] = out["province"].fillna("未匹配坐标")
    out["region"] = out["region"].fillna("未匹配")
    return out


def main() -> None:
    print("读取城市坐标表", flush=True)
    gazetteer = _load_gazetteer()
    months = _month_starts(START, END)
    _download_month(months[0], START, END)
    first = pd.read_parquet(NATIONAL_PARTS / f"{months[0]:%Y-%m}.parquet")
    preview = match_cities(sorted(first["city"].dropna().unique().tolist()), gazetteer)
    preview_n = int(preview["matched"].sum())
    print(f"首月城市 {len(preview)}，对上坐标 {preview_n}", flush=True)
    if preview_n < 80:
        missing = preview.loc[~preview["matched"], "city"].head(40).tolist()
        raise RuntimeError("对上坐标的城市太少，先停下：" + "、".join(map(str, missing)))
    for month_start in months[1:]:
        _download_month(month_start, START, END)
    frames = [pd.read_parquet(path) for path in sorted(NATIONAL_PARTS.glob("*.parquet"))]
    if not frames:
        raise RuntimeError("没有下载到任何月份")
    daily = pd.concat(frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"])
    city_names = sorted(daily["city"].dropna().unique().tolist())
    cities = match_cities(city_names, gazetteer)
    matched_n = int(cities["matched"].sum())
    print(f"城市 {len(cities)}，对上坐标 {matched_n}", flush=True)
    if matched_n < 80:
        missing = cities.loc[~cities["matched"], "city"].head(40).tolist()
        raise RuntimeError("对上坐标的城市太少，停止分析，避免用残缺地图做结论：" + "、".join(missing))
    full = _attach(daily, cities)
    present = set(pd.to_datetime(full["date"]).dt.date.unique())
    expected = []
    cursor = START
    while cursor <= END:
        expected.append(cursor)
        cursor += timedelta(days=1)
    missing_dates = [item.isoformat() for item in expected if item not in present]
    print(f"缺失日期 {len(missing_dates)} 天，开始空间统计和模型", flush=True)
    if len(present) < 300:
        raise RuntimeError(f"成功日期只有 {len(present)} 天，不写结论")
    result = analyze_national(daily, cities, missing_dates=missing_dates)
    NATIONAL_DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    NATIONAL_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(NATIONAL_DAILY_PATH, index=False)
    cities.to_parquet(NATIONAL_CITIES_PATH, index=False)
    if not result["lisa"].empty:
        result["lisa"].to_parquet(NATIONAL_LISA_PATH, index=False)
    result["predictions"].to_parquet(NATIONAL_PRED_PATH, index=False)
    dump_summary(NATIONAL_SUMMARY_JSON, result["summary"])
    upsert_national_findings(FINDINGS_MD, result["summary"])
    pm = result["summary"]["pm25"]["city_day_mean"]
    moran_i = result["summary"]["moran_period_mean"].get("I")
    print(f"完成。城市日均 PM2.5 {pm}，Moran I {moran_i}", flush=True)
    print(f"写出 {NATIONAL_DAILY_PATH}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        OUTPUTS_ERROR = ROOT / "outputs" / "national_prepare_error.txt"
        OUTPUTS_ERROR.parent.mkdir(parents=True, exist_ok=True)
        OUTPUTS_ERROR.write_text(traceback.format_exc(), encoding="utf-8")
        raise

"""下载、清洗、出图，并写出可引用的分析摘要。

在项目根目录运行：
    python scripts/prepare_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

from scripts.download_data import ensure_raw_csvs
from src.clean import clean_hourly, read_station_csvs
from src.figures import save_all
from src.metrics import add_o3_rolling, station_daily
from src.paths import (
    CLEANING_JSON,
    DAILY_PATH,
    DATA_PROCESSED,
    FIG_DIR,
    FINDINGS_MD,
    HOURLY_PATH,
    SUMMARY_JSON,
)
from src.report import build_summary, render_findings_md

KEEP_COLUMNS = [
    "station", "datetime", "date", "year", "month", "day", "hour", "season", "is_weekend",
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "O3_roll8",
    "TEMP", "PRES", "DEWP", "RAIN", "WSPM", "wd",
]


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    csvs = ensure_raw_csvs()
    print(f"读取 {len(csvs)} 个 CSV", flush=True)
    raw = read_station_csvs(csvs)
    hourly, cleaning = clean_hourly(raw)
    hourly = add_o3_rolling(hourly)
    hourly["wd"] = hourly["wd"].astype("string")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    hourly[KEEP_COLUMNS].to_parquet(HOURLY_PATH, index=False)
    daily = station_daily(hourly)
    daily.to_parquet(DAILY_PATH, index=False)
    summary, extras = build_summary(hourly, daily, cleaning)
    CLEANING_JSON.parent.mkdir(parents=True, exist_ok=True)
    CLEANING_JSON.write_text(json.dumps(_json_ready(cleaning), ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_JSON.write_text(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    FINDINGS_MD.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_MD.write_text(render_findings_md(summary), encoding="utf-8")
    figures = save_all(summary, extras, FIG_DIR)
    pm = summary["pm25"]
    print("PREPARE_OK", flush=True)
    print(f"hourly={HOURLY_PATH} bytes={HOURLY_PATH.stat().st_size}", flush=True)
    print(f"daily_rows={len(daily)} figures={figures}", flush=True)
    print(
        f"PM2.5 city mean={pm['mean']} winter={pm['winter_mean']} summer={pm['summer_mean']} "
        f"high={pm['highest_station']} {pm['highest_value']} low={pm['lowest_station']} {pm['lowest_value']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

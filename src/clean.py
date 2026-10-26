"""把 12 个站点的小时宽表收成一张长表，并处理缺失与不可能值。

长表的含义：一行是一个站点的一个小时，而不是 12 个分散文件。
污染物仍放在各自的列里，方便按指标筛选。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.settings import (
    EXPECTED_END,
    EXPECTED_START,
    INTERP_COLUMNS,
    INTERP_LIMIT_HOURS,
    METEO,
    MONTH_TO_SEASON,
    POLLUTANTS,
    STATIONS,
)

RAW_COLUMNS = [
    "No", "year", "month", "day", "hour",
    *POLLUTANTS, *METEO, "wd", "station",
]


def _rule_count(report: dict, name: str, mask: pd.Series) -> None:
    report["rules"][name] = int(mask.fillna(False).sum())


def interpolate_short_gaps(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    limit: int = INTERP_LIMIT_HOURS,
) -> pd.DataFrame:
    """只填补连续长度不超过 limit 的内部缺口。更长的缺口整段保留为缺失。

    pandas 的 interpolate(limit=...) 会从缺口一端填进 limit 个点，长停机仍会被补上一段。
    这里改成按连续缺失段判断：段长超过 limit，或者贴着序列两端，就完全不补。
    """
    out = frame.copy()
    cols = columns or [col for col in INTERP_COLUMNS if col in out.columns]
    for col in cols:
        values = out[col].to_numpy(dtype=float, na_value=np.nan)
        missing = np.isnan(values)
        if not missing.any():
            continue
        padded = np.concatenate(([False], missing, [False]))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        filled = values.copy()
        n = len(values)
        for start, end in zip(starts, ends):
            gap = int(end - start)
            if gap > limit or start == 0 or end >= n:
                continue
            left = values[start - 1]
            right = values[end]
            if np.isnan(left) or np.isnan(right):
                continue
            filled[start:end] = np.linspace(left, right, gap + 2)[1:-1]
        out[col] = filled
    return out


def clean_hourly(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """清洗已合并的原始表。返回 (小时长表, 清洗报告)。"""
    df = raw.copy()
    df.columns = [str(col).strip() for col in df.columns]
    missing_cols = [col for col in ("year", "month", "day", "hour", "station", *POLLUTANTS, *METEO) if col not in df.columns]
    if missing_cols:
        raise ValueError(f"原始数据缺少字段: {missing_cols}")

    report: dict = {"rows_in": int(len(df)), "rules": {}}
    for col in POLLUTANTS + METEO:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["station"] = df["station"].astype(str).str.strip()
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]], errors="coerce")
    bad_time = df["datetime"].isna()
    _rule_count(report, "unparseable_datetime", bad_time)
    df = df.loc[~bad_time].copy()

    # 物理上不可能的取值先挖成缺失，再考虑短缺口插值。高污染数值保留。
    numeric = POLLUTANTS + METEO
    sentinel = pd.Series(False, index=df.index)
    for col in numeric:
        sentinel = sentinel | df[col].le(-999) | df[col].ge(9999)
    _rule_count(report, "sentinel_9999", sentinel)
    df.loc[sentinel, numeric] = np.nan

    for col in POLLUTANTS + ["RAIN", "WSPM"]:
        bad = df[col] < 0
        _rule_count(report, f"{col}_negative", bad)
        df.loc[bad, col] = np.nan

    bad_temp = ~df["TEMP"].between(-40, 50) & df["TEMP"].notna()
    _rule_count(report, "TEMP_out_of_range", bad_temp)
    df.loc[bad_temp, "TEMP"] = np.nan

    bad_pres = ~df["PRES"].between(850, 1100) & df["PRES"].notna()
    _rule_count(report, "PRES_out_of_range", bad_pres)
    df.loc[bad_pres, "PRES"] = np.nan

    bad_dewp = ~df["DEWP"].between(-50, 40) & df["DEWP"].notna()
    _rule_count(report, "DEWP_out_of_range", bad_dewp)
    df.loc[bad_dewp, "DEWP"] = np.nan

    # 露点明显高于气温，视为传感器矛盾，只改露点。
    dewp_gt_temp = df["DEWP"].notna() & df["TEMP"].notna() & (df["DEWP"] > df["TEMP"] + 2)
    _rule_count(report, "DEWP_above_TEMP", dewp_gt_temp)
    df.loc[dewp_gt_temp, "DEWP"] = np.nan

    bad_wind = df["WSPM"] > 40
    _rule_count(report, "WSPM_above_40", bad_wind)
    df.loc[bad_wind, "WSPM"] = np.nan

    both = df["PM2.5"].notna() & df["PM10"].notna()
    pm_invert = both & (df["PM2.5"] > df["PM10"] + 1)
    report["pm25_gt_pm10_rows"] = int(pm_invert.sum())
    report["pm25_gt_pm10_rate_among_pairs"] = float(pm_invert.sum() / both.sum()) if both.any() else 0.0

    report["missing_rate_before_interp"] = {
        col: float(df[col].isna().mean()) for col in numeric
    }

    dup_n = int(df.duplicated(["station", "datetime"]).sum())
    report["duplicate_station_hours"] = dup_n
    df = df.drop_duplicates(["station", "datetime"], keep="first")

    expected = pd.date_range(EXPECTED_START, EXPECTED_END, freq="h")
    report["expected_hours_per_station"] = int(len(expected))
    unknown = sorted(set(df["station"]) - set(STATIONS))
    report["unknown_stations"] = unknown
    if unknown:
        raise ValueError(f"出现未登记的站点: {unknown}")

    found = set(df["station"])
    missing_stations = [station for station in STATIONS if station not in found]
    report["missing_stations"] = missing_stations
    if missing_stations:
        raise ValueError(f"缺少站点文件或站点列为空: {missing_stations}")

    pieces: list[pd.DataFrame] = []
    present = []
    for station in STATIONS:
        part = df.loc[df["station"] == station].set_index("datetime").sort_index()
        present.append(station)
        part = interpolate_short_gaps(part.reindex(expected))
        part["station"] = station
        part["wd"] = part["wd"] if "wd" in part.columns else np.nan
        pieces.append(part.reset_index().rename(columns={"index": "datetime"}))

    out = pd.concat(pieces, ignore_index=True)
    out["year"] = out["datetime"].dt.year.astype(int)
    out["month"] = out["datetime"].dt.month.astype(int)
    out["day"] = out["datetime"].dt.day.astype(int)
    out["hour"] = out["datetime"].dt.hour.astype(int)
    out["season"] = out["month"].map(MONTH_TO_SEASON)
    out["date"] = out["datetime"].dt.floor("D")
    out["is_weekend"] = out["datetime"].dt.dayofweek >= 5

    report["stations_present"] = present
    report["rows_out"] = int(len(out))
    report["missing_rate_after_interp"] = {
        col: float(out[col].isna().mean()) for col in numeric
    }
    report["interp_limit_hours"] = INTERP_LIMIT_HOURS
    report["rain_not_interpolated"] = True
    return out, report


def read_station_csvs(csv_paths: list) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        frame = pd.read_csv(path, na_values=["NA", "NaN", "null", ""], keep_default_na=True)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("没有读到任何站点 CSV")
    return pd.concat(frames, ignore_index=True)

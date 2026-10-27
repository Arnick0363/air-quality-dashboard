"""把王晓磊归档的单日宽表收成城市日值。

原始文件每个小时、每种污染物一行，后面是城市列。
日均值只在当天有效小时不少于 18 个时保留。
"""

from __future__ import annotations

import pandas as pd

from src.china_settings import HOUR_LIMITS, HOURLY_TYPES, MIN_VALID_HOURS

ID_COLUMNS = {"date", "hour", "type"}


def city_columns(frame: pd.DataFrame) -> list[str]:
    names = []
    for name in frame.columns:
        text = str(name).strip()
        if text in ID_COLUMNS or not text or text.startswith("Unnamed"):
            continue
        names.append(name)
    return names


def _block(frame: pd.DataFrame, pollutant: str, cities: list[str]) -> pd.DataFrame | None:
    sub = frame.loc[frame["type"].astype(str).str.strip() == pollutant, cities]
    if sub.empty:
        return None
    values = sub.apply(pd.to_numeric, errors="coerce")
    limit = HOUR_LIMITS[pollutant]
    invalid = values.notna() & ((values < 0) | (values > limit))
    cleaned = values.mask(invalid)
    cleaned.attrs["invalid_hours"] = int(invalid.to_numpy().sum())
    return cleaned


def city_day_from_wide(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """返回当天每个城市一行，以及被判为无效的小时计数。"""
    if "type" not in frame.columns:
        raise ValueError("城市日文件缺少 type 列")
    cities = city_columns(frame)
    qc = {name: 0 for name in HOURLY_TYPES}
    pieces: dict[str, pd.Series] = {}
    counts: dict[str, pd.Series] = {}

    for pollutant, column, count_column in (
        ("PM2.5", "pm25", "n_pm25"),
        ("PM10", "pm10", "n_pm10"),
        ("SO2", "so2", "n_so2"),
        ("NO2", "no2", "n_no2"),
        ("CO", "co", "n_co"),
        ("O3", "o3", "n_o3"),
    ):
        block = _block(frame, pollutant, cities)
        if block is None:
            mean = pd.Series(pd.NA, index=cities, dtype="Float64")
            count = pd.Series(0, index=cities, dtype="int64")
        else:
            qc[pollutant] += int(block.attrs.get("invalid_hours", 0))
            count = block.notna().sum(axis=0).astype("int64")
            mean = block.mean(axis=0, skipna=True)
            mean = mean.where(count >= MIN_VALID_HOURS)
        pieces[column] = mean
        counts[count_column] = count

    o3_count = counts["n_o3"]
    mda = _block(frame, "O3_8h_24h", cities)
    if mda is None:
        o3_mda8 = pd.Series(pd.NA, index=cities, dtype="Float64")
    else:
        hour = pd.to_numeric(
            frame.loc[frame["type"].astype(str).str.strip() == "O3_8h_24h", "hour"],
            errors="coerce",
        )
        order = hour.reset_index(drop=True).to_numpy()
        latest_pos = int(pd.Series(order).idxmax()) if len(order) else 0
        o3_mda8 = mda.iloc[latest_pos]
        o3_mda8 = o3_mda8.where(o3_count >= MIN_VALID_HOURS)

    out = pd.DataFrame({"city": [str(name).strip() for name in cities]})
    for column, series in pieces.items():
        out[column] = series.to_numpy()
    out["o3_mda8"] = o3_mda8.to_numpy()
    for column, series in counts.items():
        out[column] = series.to_numpy()
    return out, qc

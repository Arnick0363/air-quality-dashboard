"""由小时长表计算日值、城市序列、相关性和极端污染过程。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.settings import (
    METEO,
    MIN_VALID_HOURS,
    O3_ROLL_MIN_PERIODS,
    O3_ROLL_WINDOW,
    POLLUTANTS,
    STANDARDS,
    value_column,
)

DAILY_VALUE_COLUMNS = POLLUTANTS + METEO


def add_o3_rolling(hourly: pd.DataFrame) -> pd.DataFrame:
    """在完整小时网格上计算前向 8 小时滑动平均。

    必须在按季节或日期删行之前调用，否则滑动窗口会跨过被删掉的月份。
    """
    df = hourly.sort_values(["station", "datetime"]).copy()
    df["O3_roll8"] = df.groupby("station", sort=False)["O3"].transform(
        lambda series: series.rolling(O3_ROLL_WINDOW, min_periods=O3_ROLL_MIN_PERIODS).mean()
    )
    return df


def station_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """每个站点每一天一行。有效小时不足的日均值置为缺失，避免残缺日拉偏均值。"""
    df = hourly.copy()
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["datetime"]).dt.floor("D")
    if "O3_roll8" not in df.columns:
        df = add_o3_rolling(df)

    named = {}
    for col in DAILY_VALUE_COLUMNS:
        named[col] = (col, "mean")
        named[f"{col}_n"] = (col, "count")
    named["O3_MDA8"] = ("O3_roll8", "max")
    named["O3_roll_n"] = ("O3_roll8", "count")
    if "season" in df.columns:
        named["season"] = ("season", "first")
        named["year"] = ("year", "first")
        named["month"] = ("month", "first")
    else:
        pass

    daily = df.groupby(["station", "date"], as_index=False).agg(**named)
    if "year" not in daily.columns:
        daily["year"] = daily["date"].dt.year
        daily["month"] = daily["date"].dt.month
    daily["is_weekend"] = daily["date"].dt.dayofweek >= 5
    for col in DAILY_VALUE_COLUMNS:
        daily.loc[daily[f"{col}_n"] < MIN_VALID_HOURS, col] = np.nan
    # 臭氧日最大 8 小时值同样要求当天原始臭氧小时数达标。
    daily.loc[daily["O3_n"] < MIN_VALID_HOURS, "O3_MDA8"] = np.nan
    if "RAIN_n" in daily.columns:
        rain_sum = df.groupby(["station", "date"])["RAIN"].sum(min_count=1).reset_index(name="RAIN_sum")
        daily = daily.merge(rain_sum, on=["station", "date"], how="left")
        daily.loc[daily["RAIN_n"] < MIN_VALID_HOURS, "RAIN_sum"] = np.nan
    return daily


def required_stations(n_selected: int) -> int:
    """全网分析要求至少 8 站；站点变少时改为不少于一半，避免一张图被单站左右。"""
    if n_selected >= 10:
        return 8
    return max(1, int(np.ceil(n_selected / 2)))


def city_daily(daily: pd.DataFrame, pollutant: str, min_stations: int | None = None) -> pd.DataFrame:
    """把站点-日平均成城市-日。"""
    col = value_column(pollutant)
    work = daily.dropna(subset=[col]).copy()
    if work.empty:
        return pd.DataFrame(columns=["date", col, "n_stations", "year", "month", "season"])
    n_selected = int(daily["station"].nunique())
    need = min_stations if min_stations is not None else required_stations(n_selected)
    agg_map = {
        col: (col, "mean"),
        "n_stations": (col, "count"),
        "year": ("year", "first"),
        "month": ("month", "first"),
    }
    if "season" in work.columns:
        agg_map["season"] = ("season", "first")
    for extra in ("TEMP", "DEWP", "WSPM", "RAIN", "PRES", "PM2.5", "PM10", "SO2", "NO2", "CO", "O3"):
        if extra in work.columns and extra != col:
            agg_map[extra] = (extra, "mean")
    out = work.groupby("date", as_index=False).agg(**agg_map)
    out.loc[out["n_stations"] < need, col] = np.nan
    return out.sort_values("date")


def pearson(x: pd.Series, y: pd.Series) -> dict | None:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    n = int(len(frame))
    if n < 5 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return None
    r, p = stats.pearsonr(frame["x"], frame["y"])
    return {"r": float(r), "p": float(p), "n": n}


def correlation_matrix(city: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    use = [col for col in columns if col in city.columns]
    return city[use].corr(method="pearson", min_periods=5)


def find_episodes(city: pd.DataFrame, value_col: str, threshold: float) -> list[dict]:
    """把超过阈值的连续日期合并成一次污染过程。中间若有未超标或无效日，则断开。"""
    if city.empty or value_col not in city.columns:
        return []
    df = city.dropna(subset=[value_col]).sort_values("date").copy()
    over = df.loc[df[value_col] > threshold].copy()
    if over.empty:
        return []
    gap = over["date"].diff().dt.days.fillna(1)
    group_id = (gap > 1).cumsum()
    episodes = []
    for _, group in over.groupby(group_id):
        peak_idx = group[value_col].idxmax()
        item = {
            "start": pd.Timestamp(group["date"].iloc[0]).date().isoformat(),
            "end": pd.Timestamp(group["date"].iloc[-1]).date().isoformat(),
            "days": int(len(group)),
            "peak": float(group[value_col].max()),
            "peak_date": pd.Timestamp(group.loc[peak_idx, "date"]).date().isoformat(),
            "mean": float(group[value_col].mean()),
        }
        for extra in ("WSPM", "DEWP", "TEMP"):
            if extra in group.columns and group[extra].notna().any():
                item[extra.lower()] = float(group[extra].mean())
        episodes.append(item)
    episodes.sort(key=lambda row: (row["peak"], row["days"]), reverse=True)
    return episodes


def diurnal_profile(hourly: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    work = hourly.dropna(subset=[pollutant])
    if work.empty:
        return pd.DataFrame(columns=["hour", "value", "n"])
    out = work.groupby("hour", as_index=False).agg(value=(pollutant, "mean"), n=(pollutant, "count"))
    return out.sort_values("hour")


def seasonal_means(daily: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    col = value_column(pollutant)
    work = daily.dropna(subset=[col])
    if work.empty or "season" not in work.columns:
        return pd.DataFrame(columns=["season", "value", "n"])
    # 先站点-日，再对所有站点-日取平均，与城市-日平均在站点覆盖均匀时接近。
    # 这里用城市-日，避免多站点重复加权不一致：调用方应传入城市日表或站点日表。
    return work.groupby("season", as_index=False).agg(value=(col, "mean"), n=(col, "count"))


def station_means(daily: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    col = value_column(pollutant)
    work = daily.dropna(subset=[col])
    if work.empty:
        return pd.DataFrame(columns=["station", "value", "n_days"])
    return (
        work.groupby("station", as_index=False)
        .agg(value=(col, "mean"), n_days=(col, "count"))
        .sort_values("value", ascending=False)
    )


def exceedance(daily: pd.DataFrame, pollutant: str) -> dict:
    col = value_column(pollutant)
    limit = STANDARDS[pollutant]["limit"]
    work = daily.dropna(subset=[col])
    n = int(len(work))
    if n == 0:
        return {"n": 0, "n_over": 0, "rate": None, "limit": limit, "column": col}
    n_over = int((work[col] > limit).sum())
    return {
        "n": n,
        "n_over": n_over,
        "rate": n_over / n,
        "limit": limit,
        "column": col,
    }


def format_number(value: float | None, pollutant: str | None = None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if pollutant == "CO":
        return f"{value:,.0f}"
    if pollutant == "WSPM":
        return f"{value:.2f}"
    return f"{value:.1f}"

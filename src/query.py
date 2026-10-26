"""按页面筛选条件切小时表和日表。筛选只作用于已经清洗好的长表。"""

from __future__ import annotations

import pandas as pd

from src.settings import SEASON_MONTHS


def _months(season_mode: str) -> list[int] | None:
    if season_mode in SEASON_MONTHS:
        return list(SEASON_MONTHS[season_mode])
    return None


def subset_hourly(
    hourly: pd.DataFrame,
    station_ids: list[str],
    years: list[int],
    start,
    end,
    season_mode: str = "全年",
) -> pd.DataFrame:
    if hourly.empty or not station_ids or not years:
        return hourly.iloc[0:0]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(hours=23)
    mask = hourly["station"].isin(station_ids)
    mask &= hourly["year"].isin([int(year) for year in years])
    mask &= hourly["datetime"].between(start_ts, end_ts)
    months = _months(season_mode)
    if months is not None:
        mask &= hourly["month"].isin(months)
    return hourly.loc[mask]


def subset_daily(
    daily: pd.DataFrame,
    station_ids: list[str],
    years: list[int],
    start,
    end,
    season_mode: str = "全年",
) -> pd.DataFrame:
    if daily.empty or not station_ids or not years:
        return daily.iloc[0:0]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = daily["station"].isin(station_ids)
    mask &= daily["year"].isin([int(year) for year in years])
    mask &= daily["date"].between(start_ts, end_ts)
    months = _months(season_mode)
    if months is not None:
        mask &= daily["month"].isin(months)
    return daily.loc[mask]

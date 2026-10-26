"""用小型构造数据检查清洗规则、日有效性和插值，不依赖完整数据集。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.clean import interpolate_short_gaps
from src.metrics import exceedance, find_episodes, required_stations, station_daily
from src.settings import MIN_VALID_HOURS, MONTH_TO_SEASON, pm25_category
from src.spatial import idw_predict


def test_short_gap_is_filled_and_long_gap_is_not():
    index = pd.date_range("2014-01-01", periods=30, freq="h")
    frame = pd.DataFrame({"PM2.5": np.arange(30, dtype=float), "RAIN": np.arange(30, dtype=float)}, index=index)
    frame.loc[index[5:8], "PM2.5"] = np.nan
    frame.loc[index[5:8], "RAIN"] = np.nan
    frame.loc[index[12:24], "PM2.5"] = np.nan
    filled = interpolate_short_gaps(frame, columns=["PM2.5"], limit=6)
    assert filled.loc[index[5:8], "PM2.5"].notna().all()
    assert filled.loc[index[12:24], "PM2.5"].isna().all()
    assert filled.loc[index[5:8], "RAIN"].isna().all()


def test_negative_does_not_survive_as_anchor_when_left_as_nan():
    index = pd.date_range("2014-01-01", periods=5, freq="h")
    frame = pd.DataFrame({"PM2.5": [10.0, -5.0, 30.0, 40.0, 50.0]}, index=index)
    frame.loc[frame["PM2.5"] < 0, "PM2.5"] = np.nan
    filled = interpolate_short_gaps(frame, columns=["PM2.5"], limit=6)
    assert filled.loc[index[1], "PM2.5"] == 20.0


def test_daily_validity_needs_18_hours():
    hours = pd.date_range("2014-06-01", periods=48, freq="h")
    frame = pd.DataFrame({
        "station": "Dongsi",
        "datetime": hours,
        "PM2.5": 10.0,
        "PM10": 20.0,
        "SO2": 5.0,
        "NO2": 15.0,
        "CO": 400.0,
        "O3": 40.0,
        "TEMP": 20.0,
        "PRES": 1010.0,
        "DEWP": 10.0,
        "RAIN": 0.0,
        "WSPM": 2.0,
        "year": hours.year,
        "month": hours.month,
        "season": [MONTH_TO_SEASON[m] for m in hours.month],
    })
    frame.loc[frame["datetime"].dt.day == 2, "PM2.5"] = np.nan
    frame.loc[frame["datetime"].dt.hour < 8, "PM2.5"] = np.nan
    # 6 月 1 日只剩 16 个小时，6 月 2 日全部缺失。
    frame["O3_roll8"] = frame["O3"]
    daily = station_daily(frame)
    june1 = daily.loc[daily["date"] == pd.Timestamp("2014-06-01"), "PM2.5"]
    assert june1.isna().all()
    assert int(daily.loc[daily["date"] == pd.Timestamp("2014-06-01"), "PM2.5_n"].iloc[0]) < MIN_VALID_HOURS


def test_episodes_split_on_gap():
    dates = pd.to_datetime(["2014-01-01", "2014-01-02", "2014-01-03", "2014-01-05", "2014-01-06"])
    city = pd.DataFrame({
        "date": dates,
        "PM2.5": [180, 190, 160, 200, 170],
        "WSPM": [1.0, 1.2, 1.1, 0.8, 0.9],
        "DEWP": [-5, -4, -3, -6, -5],
        "TEMP": [-1, 0, 1, -2, -1],
    })
    episodes = find_episodes(city, "PM2.5", 150)
    assert len(episodes) == 2
    assert episodes[0]["peak_date"] == "2014-01-05"
    assert episodes[1]["days"] == 3


def test_idw_returns_exact_value_on_station():
    pred = idw_predict([40.0, 40.2], [116.0, 116.2], [10.0, 30.0], [40.0], [116.0])
    assert pred[0] == 10.0


def test_required_stations_and_pm25_bands():
    assert required_stations(12) == 8
    assert required_stations(3) == 2
    assert pm25_category(35) == "优"
    assert pm25_category(75) == "良"
    assert pm25_category(151) == "重度污染"
    assert exceedance(pd.DataFrame({"PM2.5": [10, 80, 90]}), "PM2.5")["n_over"] == 2

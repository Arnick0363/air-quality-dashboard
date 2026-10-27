"""全国层的规则测试，不下载城市日文件。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.autocorr import knn_weights, moran_test, morans_i
from src.china_clean import city_day_from_wide
from src.forecast_model import impute_with_train_median, make_supervised, regression_scores
from src.regions import assign_region, match_cities


def _wide(hours, values_by_type):
    rows = []
    for pollutant, cities in values_by_type.items():
        for hour in range(hours):
            row = {"date": 20200101, "hour": hour, "type": pollutant}
            for city, series in cities.items():
                row[city] = series[hour]
            rows.append(row)
    return pd.DataFrame(rows)


def test_daily_mean_needs_18_valid_hours_and_drops_negative():
    good = [10.0] * 24
    short = [20.0] * 10 + [np.nan] * 14
    dirty = [10.0, -5.0] + [10.0] * 22
    frame = _wide(24, {"PM2.5": {"甲": good, "乙": short, "丙": dirty}})
    daily, qc = city_day_from_wide(frame)
    got = daily.set_index("city")
    assert got.loc["甲", "pm25"] == 10
    assert got.loc["甲", "n_pm25"] == 24
    assert pd.isna(got.loc["乙", "pm25"])
    assert got.loc["乙", "n_pm25"] == 10
    assert got.loc["丙", "n_pm25"] == 23
    assert qc["PM2.5"] == 1


def test_ozone_mda8_uses_last_hour_and_hourly_count():
    hours = 24
    o3 = [40.0] * hours
    mda = list(range(hours))
    sparse = [30.0] * 5 + [np.nan] * 19
    frame = _wide(hours, {
        "O3": {"甲": o3, "乙": sparse},
        "O3_8h_24h": {"甲": mda, "乙": mda},
    })
    daily, _qc = city_day_from_wide(frame)
    got = daily.set_index("city")
    assert got.loc["甲", "o3_mda8"] == 23
    assert pd.isna(got.loc["乙", "o3_mda8"])


def test_region_priority_is_mutually_exclusive():
    assert assign_region("深圳", "广东") == "珠三角"
    assert assign_region("南京", "江苏") == "长三角"
    assert assign_region("合肥", "安徽") == "长三角"
    assert assign_region("成都", "四川") == "成渝"
    assert assign_region("攀枝花", "四川") == "其他"
    assert assign_region("哈尔滨", "黑龙江") == "东北"
    assert assign_region("北京", "北京") == "华北"
    assert assign_region("西安", "陕西") == "西北"
    assert assign_region("武汉", "湖北") == "其他"


def test_prefecture_code_beats_province_code_when_names_collide():
    gazetteer = pd.DataFrame({
        "Location_Name_ZH": ["吉林", "吉林"],
        "AD_code": ["220000", "220200"],
        "Longitude": [126.1, 126.57],
        "Latitude": [43.9, 43.84],
        "province": ["吉林", "吉林"],
    })
    matched = match_cities(["吉林"], gazetteer)
    assert matched.loc[0, "ad_code"] == "220200"
    assert matched.loc[0, "region"] == "东北"
    assert abs(matched.loc[0, "lon"] - 126.57) < 1e-6


def test_long_autonomous_name_uses_contained_gazetteer_name():
    gazetteer = pd.DataFrame({
        "Location_Name_ZH": ["临夏", "大理州"],
        "AD_code": ["622900", "532900"],
        "Longitude": [103.2, 100.2],
        "Latitude": [35.6, 25.6],
        "province": ["甘肃", "云南"],
    })
    matched = match_cities(["临夏回族自治州", "大理州"], gazetteer).set_index("city")
    assert matched.loc["大理州", "ad_code"] == "532900"
    assert abs(float(matched.loc["临夏回族自治州", "lon"]) - 103.2) < 1e-6


def test_gradient_has_positive_moran_and_constant_does_not():
    lat = np.linspace(20, 45, 16)
    lon = np.linspace(100, 120, 16)
    weights = knn_weights(lat, lon, k=3)
    assert weights.shape == (16, 16)
    assert np.allclose(weights.sum(axis=1), 1)
    assert morans_i(lon, weights) > 0.4
    assert morans_i(np.ones(16), weights) is None
    tested = moran_test(lon, weights, n_perm=49, seed=1)
    assert 0 < tested["p"] <= 1


def test_supervised_split_follows_target_date_and_skips_gaps():
    days = pd.date_range("2023-11-01", "2025-01-02", freq="D")
    frame = pd.DataFrame({
        "date": days,
        "city": "甲",
        "pm25": np.linspace(10, 20, len(days)),
        "pm10": 30.0,
        "so2": 8.0,
        "no2": 20.0,
        "co": 0.8,
        "o3_mda8": 90.0,
    })
    frame = frame.loc[frame["date"] != "2023-12-30"]
    meta = pd.DataFrame({"city": ["甲"], "lat": [40.0], "lon": [116.0], "region": ["华北"]})
    supervised = make_supervised(frame, meta)
    assert supervised.loc[supervised["date"] == "2023-12-29"].empty
    assert supervised.loc[supervised["date"] == "2023-12-31"].empty
    first_valid = supervised.loc[supervised["date"] == "2024-01-01"].iloc[0]
    assert first_valid["split"] == "valid"
    assert first_valid["target_date"] == pd.Timestamp("2024-01-02")
    train_targets = supervised.loc[supervised["split"] == "train", "target_date"]
    test_targets = supervised.loc[supervised["split"] == "test", "target_date"]
    assert train_targets.max() < test_targets.min()
    assert train_targets.dt.year.max() <= 2023
    assert set(test_targets.dt.year.unique()) == {2025}


def test_impute_median_comes_only_from_train():
    train = pd.DataFrame({"pm25": [1.0, 3.0], "pm10": [10.0, np.nan], "so2": [1.0, 1.0],
                          "no2": [1.0, 1.0], "co": [1.0, 1.0], "o3_mda8": [1.0, 1.0],
                          "pm25_lag1": [1.0, 1.0], "pm25_lag7": [1.0, 1.0], "pm25_roll7": [1.0, 1.0],
                          "month_sin": [0.0, 0.0], "month_cos": [1.0, 1.0], "dow": [1, 2],
                          "lat": [1.0, 1.0], "lon": [2.0, 2.0]})
    test = train.copy()
    test["pm10"] = [9999.0, np.nan]
    filled_train, (filled_test,), median = impute_with_train_median(train, [test])
    assert median["pm10"] == 10
    assert filled_test["pm10"].iloc[1] == 10
    assert filled_train["pm10"].iloc[1] == 10


def test_regression_scores_mae():
    scores = regression_scores(np.array([1.0, 2.0]), np.array([2.0, 2.0]))
    assert scores["mae"] == 0.5
    assert scores["n"] == 2

"""全国城市日值、空间权重和时间切分。不下载数据。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.national_clean import (
    aggregate_city_hours,
    assign_region,
    match_cities,
    normalize_place_name,
    parse_china_cities_csv,
)
from src.national_model import FEATURE_COLUMNS, build_supervised, time_split
from src.spatial_stats import global_moran_test, inverse_distance_weights, knn_weights, spatial_lag_matrix


def _gazetteer(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["norm_name"] = frame["source_name"].map(normalize_place_name)
    frame["norm_adm2"] = frame["adm2"].map(normalize_place_name)
    frame["level"] = frame["adcode"].map(
        lambda code: "省级" if int(code) % 10000 == 0 else "地级" if int(code) % 100 == 0 else "县级"
    )
    return frame


def test_normalize_place_name_strips_suffix_and_alias():
    assert normalize_place_name("北京市") == "北京"
    assert normalize_place_name("巴音郭楞州") == "巴音郭楞"
    assert normalize_place_name("朝阳市") == "朝阳"


def test_match_prefers_prefecture_and_keeps_county_city():
    gazetteer = _gazetteer([
        {"source_name": "朝阳", "province": "北京市", "adm2": "北京市", "lat": 39.92, "lon": 116.44, "adcode": 110105},
        {"source_name": "朝阳", "province": "辽宁省", "adm2": "朝阳市", "lat": 41.57, "lon": 120.45, "adcode": 211300},
        {"source_name": "北京", "province": "北京市", "adm2": "北京市", "lat": 39.90, "lon": 116.40, "adcode": 110000},
        {"source_name": "昆山", "province": "江苏省", "adm2": "苏州市", "lat": 31.38, "lon": 120.98, "adcode": 320583},
        {"source_name": "重复", "province": "江苏省", "adm2": "南京市", "lat": 32.0, "lon": 118.8, "adcode": 320102},
        {"source_name": "重复", "province": "浙江省", "adm2": "杭州市", "lat": 30.2, "lon": 120.1, "adcode": 330102},
    ])
    matched = match_cities(["朝阳", "北京", "昆山", "重复", "不存在"], gazetteer).set_index("city")
    assert matched.loc["朝阳", "province"] == "辽宁省"
    assert matched.loc["北京", "adcode"] == 110000
    assert matched.loc["昆山", "matched"]
    assert not bool(matched.loc["重复", "matched"])
    assert not bool(matched.loc["不存在", "matched"])
    assert assign_region("广东省", "深圳") == "珠三角"
    assert assign_region("广东省", "广州") == "珠三角"
    assert assign_region("广东省", "惠州") == "珠三角"
    assert assign_region("广东省", "韶关") == "其他"
    assert assign_region("山西省", "太原") == "华北"
    assert assign_region("四川省", "成都") == "成渝"


def test_daily_mean_requires_18_valid_hours():
    rows = []
    for hour in range(24):
        rows.append({"date": "2024-01-01", "hour": hour, "city": "甲", "type": "PM2.5", "value": 10 + hour})
        if hour < 10:
            rows.append({"date": "2024-01-01", "hour": hour, "city": "乙", "type": "PM2.5", "value": 50})
        rows.append({"date": "2024-01-01", "hour": hour, "city": "甲", "type": "CO", "value": -1 if hour == 0 else 1.2})
    daily = aggregate_city_hours(pd.DataFrame(rows), min_hours=18)
    jia = daily.loc[daily["city"] == "甲"].iloc[0]
    yi = daily.loc[daily["city"] == "乙"].iloc[0]
    assert jia["PM2.5_n"] == 24
    assert abs(jia["PM2.5"] - np.mean(np.arange(10, 34))) < 1e-9
    assert np.isnan(yi["PM2.5"])
    assert yi["PM2.5_n"] == 10
    assert jia["CO_n"] == 23
    assert abs(jia["CO"] - 1.2) < 1e-9


def test_parse_wide_csv_drops_duplicate_city_column():
    hours = "\n".join(
        f"20240102,{hour},PM2.5,12,30" for hour in range(20)
    )
    text = "date,hour,type,甲,甲\n" + hours + "\n"
    daily = parse_china_cities_csv(text, min_hours=18)
    assert set(daily["city"]) == {"甲"}
    assert daily.iloc[0]["PM2.5_n"] == 20


def test_knn_weights_are_row_standardized_without_self():
    lat = np.array([0, 0, 0, 0, 1], dtype=float)
    lon = np.array([0, 1, 2, 3, 0], dtype=float)
    weights = knn_weights(lat, lon, k=2)
    assert weights.shape == (5, 5)
    assert np.allclose(weights.sum(axis=1), 1)
    assert np.allclose(np.diag(weights), 0)


def test_moran_sign_follows_smooth_and_checkerboard():
    lat, lon, smooth, checks = [], [], [], []
    for i in range(6):
        for j in range(6):
            lat.append(30 + i * 0.4)
            lon.append(110 + j * 0.4)
            smooth.append(float(i))
            checks.append(float((i + j) % 2))
    weights = knn_weights(lat, lon, k=4)
    smooth_result = global_moran_test(smooth, weights, n_perm=99, seed=1)
    check_result = global_moran_test(checks, weights, n_perm=99, seed=1)
    assert smooth_result["I"] > 0
    assert check_result["I"] < 0
    assert 0 < smooth_result["p_value"] <= 1
    assert abs(smooth_result["expected"] - (-1 / 35)) < 1e-12


def test_inverse_distance_drops_isolated_point():
    lat = [30, 30.2, 30.4, 45]
    lon = [110, 110.2, 110.4, 80]
    weights, kept = inverse_distance_weights(lat, lon, cutoff_km=100)
    assert 3 not in set(kept.tolist())
    assert weights.shape[0] == 3
    assert np.allclose(weights.sum(axis=1), 1)


def test_supervised_features_use_yesterday_not_same_day():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    panel = pd.DataFrame({
        "date": list(dates) * 2,
        "city": ["甲", "甲", "乙", "乙"],
        "PM2.5": [10.0, 12.0, 30.0, 40.0],
        "lat": [30.0, 30.0, 31.0, 31.0],
        "lon": [110.0, 110.0, 111.0, 111.0],
    })
    supervised = build_supervised(panel, k=1)
    day2 = supervised.loc[supervised["date"] == pd.Timestamp("2024-01-02")].set_index("city")
    assert abs(day2.loc["甲", "lag1"] - 10) < 1e-9
    assert abs(day2.loc["甲", "spatial_lag1"] - 30) < 1e-9
    assert "PM2.5" not in FEATURE_COLUMNS
    assert "y" not in FEATURE_COLUMNS
    leaked = spatial_lag_matrix(np.array([12.0, 40.0]), np.array([[0, 1], [1, 0]]))
    assert abs(day2.loc["甲", "spatial_lag1"] - leaked[0]) > 1


def test_time_split_does_not_shuffle_future_into_train():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    frame = pd.DataFrame({"date": dates, "y": np.arange(40)})
    train, test, cutoff = time_split(frame, test_fraction=0.25)
    assert train["date"].max() < cutoff
    assert test["date"].min() >= cutoff
    assert train["date"].max() < test["date"].min()

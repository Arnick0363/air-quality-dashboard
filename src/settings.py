"""站点、污染物、空气质量标准与清洗规则。

站点经纬度取自公开文献中的国控点位置表（保留三位小数），两处一致：
- npj Urban Sustainability (2022) Table 1，DOI 相关论文
  Changing weekend effects of air pollutants in Beijing
- Foreback 等，ACP 2022 补充材料 Table S1

功能分区沿用同一监测网络分类：城区评价点、郊区评价点、背景点。
坐标不是本仓库实测，地图只作描述性空间展示。
"""

from __future__ import annotations

# 日均值至少需要的有效小时数（24 小时的 75%）。这是分析约定，不是原始数据字段。
MIN_VALID_HOURS = 18
# 只填补不超过 6 小时的内部缺口，避免把长时段故障补成平滑曲线。
INTERP_LIMIT_HOURS = 6
# 臭氧 8 小时滑动平均至少需要 6 个有效小时。
O3_ROLL_WINDOW = 8
O3_ROLL_MIN_PERIODS = 6
# 逆距离权重：幂次，以及超过该距离的格点不着色。
IDW_POWER = 2.0
IDW_MAX_KM = 40.0

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
METEO = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
# 降水是间歇量，线性插值会造出并不存在的降雨，因此不参与插值。
INTERP_COLUMNS = POLLUTANTS + ["TEMP", "PRES", "DEWP", "WSPM"]

# GB 3095-2012 二级标准。CO 数据集单位是 μg/m³，标准原文为 4 mg/m³。
# O3 用的是日最大 8 小时平均，不是日均。
STANDARDS = {
    "PM2.5": {
        "limit": 75.0,
        "unit": "μg/m³",
        "agg": "daily_mean",
        "rule": "GB 3095-2012 二级，24 小时平均",
    },
    "PM10": {
        "limit": 150.0,
        "unit": "μg/m³",
        "agg": "daily_mean",
        "rule": "GB 3095-2012 二级，24 小时平均",
    },
    "SO2": {
        "limit": 150.0,
        "unit": "μg/m³",
        "agg": "daily_mean",
        "rule": "GB 3095-2012 二级，24 小时平均",
    },
    "NO2": {
        "limit": 80.0,
        "unit": "μg/m³",
        "agg": "daily_mean",
        "rule": "GB 3095-2012 二级，24 小时平均",
    },
    "CO": {
        "limit": 4000.0,
        "unit": "μg/m³",
        "agg": "daily_mean",
        "rule": "GB 3095-2012 二级，24 小时平均 4 mg/m³（换算为 4000 μg/m³）",
    },
    "O3": {
        "limit": 160.0,
        "unit": "μg/m³",
        "agg": "mda8",
        "rule": "GB 3095-2012 二级，日最大 8 小时平均",
    },
}

# HJ 633-2012 中与 PM2.5 24 小时浓度对应的空气质量类别边界（右闭）。
PM25_BANDS = [
    (35, "优"),
    (75, "良"),
    (115, "轻度污染"),
    (150, "中度污染"),
    (250, "重度污染"),
    (float("inf"), "严重污染"),
]

SEASON_MONTHS = {
    "春": (3, 4, 5),
    "夏": (6, 7, 8),
    "秋": (9, 10, 11),
    "冬": (12, 1, 2),
}
MONTH_TO_SEASON = {
    month: season for season, months in SEASON_MONTHS.items() for month in months
}
SEASON_ORDER = ["春", "夏", "秋", "冬"]

WIND_ORDER = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]

# 重度污染过程：PM2.5 日均超过该值的连续天合并为一次事件。
EPISODE_PM25 = 150.0

STATIONS: dict[str, dict] = {
    "Aotizhongxin": {"zh": "奥体中心", "lon": 116.397, "lat": 39.982, "zone": "城区"},
    "Changping": {"zh": "昌平", "lon": 116.230, "lat": 40.217, "zone": "郊区"},
    "Dingling": {"zh": "定陵", "lon": 116.220, "lat": 40.292, "zone": "背景点"},
    "Dongsi": {"zh": "东四", "lon": 116.417, "lat": 39.929, "zone": "城区"},
    "Guanyuan": {"zh": "官园", "lon": 116.339, "lat": 39.929, "zone": "城区"},
    "Gucheng": {"zh": "古城", "lon": 116.184, "lat": 39.914, "zone": "城区"},
    "Huairou": {"zh": "怀柔", "lon": 116.628, "lat": 40.328, "zone": "郊区"},
    "Nongzhanguan": {"zh": "农展馆", "lon": 116.461, "lat": 39.937, "zone": "城区"},
    "Shunyi": {"zh": "顺义", "lon": 116.655, "lat": 40.127, "zone": "郊区"},
    "Tiantan": {"zh": "天坛", "lon": 116.407, "lat": 39.886, "zone": "城区"},
    "Wanliu": {"zh": "万柳", "lon": 116.287, "lat": 39.987, "zone": "城区"},
    "Wanshouxigong": {"zh": "万寿西宫", "lon": 116.352, "lat": 39.878, "zone": "城区"},
}

STATION_IDS = list(STATIONS)
ZH_TO_ID = {meta["zh"]: sid for sid, meta in STATIONS.items()}
ID_TO_ZH = {sid: meta["zh"] for sid, meta in STATIONS.items()}

EXPECTED_START = "2013-03-01 00:00:00"
EXPECTED_END = "2017-02-28 23:00:00"
COMPLETE_YEARS = [2014, 2015, 2016]
PARTIAL_YEARS = [2013, 2017]


def station_zh(station_id: str) -> str:
    return ID_TO_ZH.get(station_id, station_id)


def pollutant_label(name: str) -> str:
    return name


def value_column(pollutant: str) -> str:
    """日表里用于超标和时序的列。臭氧用日最大 8 小时平均。"""
    if pollutant == "O3":
        return "O3_MDA8"
    return pollutant


def pm25_category(value: float) -> str:
    if value != value:  # NaN
        return "缺失"
    for upper, name in PM25_BANDS:
        if value <= upper:
            return name
    return "严重污染"

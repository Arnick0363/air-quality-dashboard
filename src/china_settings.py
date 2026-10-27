"""全国城市层的口径。和北京 12 站小时数据分开，避免两套数字混用。"""

from __future__ import annotations

from datetime import date

# 2014 年只有 5 月以后，抽查 2015-01-01 的单日文件为 404。
# 全国层只用完整日历年，方便年与年比较。
CHINA_START = date(2016, 1, 1)
CHINA_END = date(2025, 12, 31)

MIN_VALID_HOURS = 18
YEAR_MIN_DAYS = 240
SEASON_MIN_DAYS = 300
LONG_TERM_MIN_DAYS = 2000
KNN_K = 8
N_PERM = 199
RANDOM_SEED = 0

# 按目标日落在哪一年划分，而不是按特征日。2023-12-31 预测的是 2024-01-01，归入验证年。
VALID_YEAR = 2024
TEST_YEAR = 2025

PM25_DAY_LIMIT = 75.0

# 小时值超出这个范围视为无效小时，不参与日均。阈值写在清洗说明里。
HOUR_LIMITS = {
    "PM2.5": 2000.0,
    "PM10": 3000.0,
    "SO2": 2000.0,
    "NO2": 1500.0,
    "O3": 1200.0,
    "CO": 20.0,
    "O3_8h_24h": 1200.0,
}

HOURLY_TYPES = ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3")

# 页面上可选的指标。臭氧用日最大 8 小时，不用小时均值的日平均。
POLLUTANT_COLUMNS = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "SO2": "so2",
    "NO2": "no2",
    "CO": "co",
    "O3": "o3_mda8",
}

POLLUTANT_UNITS = {
    "PM2.5": "μg/m³",
    "PM10": "μg/m³",
    "SO2": "μg/m³",
    "NO2": "μg/m³",
    "CO": "mg/m³",
    "O3": "μg/m³",
}

# GB 3095-2012 二级，日尺度参照。CO 在这套城市文件里本来就是毫克/立方米。
DAY_LIMITS = {
    "PM2.5": 75.0,
    "PM10": 150.0,
    "SO2": 150.0,
    "NO2": 80.0,
    "CO": 4.0,
    "O3": 160.0,
}

# 珠三角九市。长三角按沪苏浙皖全部地级单位，不是只取核心 26 城。
# 成渝只取规划中的双城经济圈城市，不把攀枝花、甘孜等整个四川算进去。
PEARL_RIVER = {"广州", "深圳", "珠海", "佛山", "惠州", "东莞", "中山", "江门", "肇庆"}
YANGTZE_PROVINCES = {"上海", "江苏", "浙江", "安徽"}
CHENGDU_CHONGQING = {
    "成都", "自贡", "泸州", "德阳", "绵阳", "遂宁", "内江", "乐山",
    "南充", "眉山", "宜宾", "广安", "达州", "雅安", "资阳", "重庆",
}
NORTHEAST = {"辽宁", "吉林", "黑龙江"}
NORTH = {"北京", "天津", "河北", "山西", "内蒙古"}
NORTHWEST = {"陕西", "甘肃", "青海", "宁夏", "新疆"}

REGION_ORDER = ["东北", "华北", "长三角", "珠三角", "成渝", "西北", "其他"]

LOCATION_LIST_URLS = (
    "https://raw.githubusercontent.com/qwd/LocationList/master/China-City-List-latest.csv",
    "https://cdn.jsdelivr.net/gh/qwd/LocationList@master/China-City-List-latest.csv",
)

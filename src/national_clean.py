"""全国城市小时宽表折成日值，并把城市名对到公开坐标表。

日值规则和北京站点一致的地方只有一条：至少 18 个有效小时才保留当天。
这里的一行是一座监测城市，不是北京的一个国控站。
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

HOURLY_TYPES = ("PM2.5", "PM10", "SO2", "NO2", "O3", "CO", "AQI")
MIN_CITY_HOURS = 18

# 超过这些值的小时按哨兵或坏值丢掉，不拿来算日均。真实重污染小时远低于这些上限。
VALUE_CAPS = {
    "PM2.5": 3000.0,
    "PM10": 5000.0,
    "SO2": 3000.0,
    "NO2": 3000.0,
    "O3": 2000.0,
    "CO": 50.0,
    "AQI": 2000.0,
}

# 监测城市简称 -> 坐标表里的地名。只改名字，不手写经纬度。
PLACE_ALIASES = {
    "巴音郭楞州": "巴音郭楞",
    "博尔塔拉州": "博尔塔拉",
    "克孜勒苏州": "克孜勒苏",
    "克孜勒苏柯尔克孜州": "克孜勒苏",
    "伊犁哈萨克州": "伊犁",
    "伊犁州": "伊犁",
    "昌吉州": "昌吉",
    "黔东南州": "黔东南",
    "黔南州": "黔南",
    "黔西南州": "黔西南",
    "西双版纳州": "西双版纳",
    "德宏州": "德宏",
    "大理州": "大理",
    "红河州": "红河",
    "怒江州": "怒江",
    "迪庆州": "迪庆",
    "楚雄州": "楚雄",
    "文山州": "文山",
    "湘西州": "湘西",
    "延边州": "延边",
    "恩施州": "恩施",
    "凉山州": "凉山",
    "阿坝州": "阿坝",
    "甘孜州": "甘孜",
    "甘南州": "甘南",
    "临夏州": "临夏",
    "海北州": "海北",
    "海南州": "海南",
    "海西州": "海西",
    "黄南州": "黄南",
    "果洛州": "果洛",
    "玉树州": "玉树",
    "兴安盟": "兴安",
    "锡林郭勒盟": "锡林郭勒",
    "阿拉善盟": "阿拉善",
    "襄樊": "襄阳",
}

SUFFIXES = (
    "特别行政区",
    "维吾尔自治区",
    "回族自治区",
    "壮族自治区",
    "自治州",
    "自治区",
    "地区",
    "盟",
    "市",
    "州",
    "县",
    "区",
)

MUNICIPALITIES = {"北京", "上海", "天津", "重庆"}
NORTH_CHINA = {"北京市", "天津市", "河北省", "山西省", "内蒙古自治区"}
YANGTZE_DELTA = {"上海市", "江苏省", "浙江省", "安徽省"}
CHENGDU_CHONGQING = {"四川省", "重庆市"}
NORTHWEST = {"陕西省", "甘肃省", "青海省", "宁夏回族自治区", "新疆维吾尔自治区"}
PEARL_RIVER_DELTA = {"广州", "深圳", "佛山", "东莞", "中山", "珠海", "惠州", "江门", "肇庆"}


def normalize_place_name(name: str) -> str:
    """去掉行政区后缀，便于和坐标表对齐。不猜测经纬度。"""
    text = re.sub(r"\s+", "", str(name)).strip()
    text = PLACE_ALIASES.get(text, text)
    changed = True
    while changed and text:
        changed = False
        for suffix in SUFFIXES:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                changed = True
                break
    return text


def clean_hourly_values(values: pd.Series, pollutant: str) -> pd.Series:
    number = pd.to_numeric(values, errors="coerce")
    cap = VALUE_CAPS[pollutant]
    return number.mask((number < 0) | (number > cap))


def aggregate_city_hours(long_frame: pd.DataFrame, min_hours: int = MIN_CITY_HOURS) -> pd.DataFrame:
    """长表 date, hour, city, type, value -> 每个城市每天一行。

    有效小时不足 min_hours 的指标留空，不把半天数据当成日均。
    """
    required = {"date", "hour", "city", "type", "value"}
    missing = required - set(long_frame.columns)
    if missing:
        raise ValueError(f"缺少列: {sorted(missing)}")
    frame = long_frame.loc[long_frame["type"].isin(HOURLY_TYPES)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["date", "city", *HOURLY_TYPES])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["city"] = frame["city"].astype(str).str.strip()
    cleaned = []
    for pollutant, part in frame.groupby("type", sort=False):
        part = part.copy()
        part["value"] = clean_hourly_values(part["value"], pollutant)
        cleaned.append(part)
    frame = pd.concat(cleaned, ignore_index=True)
    grouped = frame.groupby(["date", "city", "type"], sort=False)["value"]
    means = grouped.mean()
    counts = grouped.count()
    mean_wide = means.unstack("type")
    count_wide = counts.unstack("type").reindex(mean_wide.index)
    for pollutant in HOURLY_TYPES:
        if pollutant not in mean_wide.columns:
            mean_wide[pollutant] = np.nan
            count_wide[pollutant] = 0
    count_wide = count_wide.fillna(0)
    for pollutant in HOURLY_TYPES:
        thin = count_wide[pollutant] < min_hours
        mean_wide.loc[thin, pollutant] = np.nan
    out = mean_wide[list(HOURLY_TYPES)].copy()
    for pollutant in HOURLY_TYPES:
        out[f"{pollutant}_n"] = count_wide[pollutant].astype(int).to_numpy()
    out = out.reset_index()
    return out


def parse_china_cities_csv(text: str, min_hours: int = MIN_CITY_HOURS) -> pd.DataFrame:
    """解析 quotsoft 的全国城市小时宽表，返回城市日值。"""
    frame = pd.read_csv(io.StringIO(text))
    frame.columns = [str(col).strip() for col in frame.columns]
    renamed = []
    seen: set[str] = set()
    for name in frame.columns:
        base = re.sub(r"\.\d+$", "", name)
        if name in seen or base in seen:
            renamed.append(f"{name}__dup")
            continue
        seen.add(name)
        seen.add(base)
        renamed.append(name)
    frame.columns = renamed
    if not {"date", "hour", "type"}.issubset(frame.columns):
        raise ValueError("城市小时表缺少 date、hour 或 type")
    date_number = pd.to_numeric(frame["date"], errors="coerce")
    if date_number.notna().mean() > 0.9:
        frame["date"] = pd.to_datetime(date_number.round().astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    else:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    cities = [col for col in frame.columns if col not in {"date", "hour", "type"} and "__dup" not in col]
    long = frame.melt(id_vars=["date", "hour", "type"], value_vars=cities, var_name="city", value_name="value")
    return aggregate_city_hours(long, min_hours=min_hours)


def _admin_level(adcode: int) -> str:
    if adcode % 10000 == 0:
        return "省级"
    if adcode % 100 == 0:
        return "地级"
    return "县级"


def load_qweather_gazetteer(text: str) -> pd.DataFrame:
    """读和风天气城市列表。第一行是版本说明，不是表头。"""
    lines = text.splitlines()
    header_at = next((i for i, line in enumerate(lines) if line.startswith("Location_ID")), None)
    if header_at is None:
        raise ValueError("坐标表里没有 Location_ID 表头")
    frame = pd.read_csv(io.StringIO("\n".join(lines[header_at:])))
    keep = frame.rename(columns={
        "Location_Name_ZH": "source_name",
        "Adm1_Name_ZH": "province",
        "Adm2_Name_ZH": "adm2",
        "Latitude": "lat",
        "Longitude": "lon",
        "AD_code": "adcode",
    })
    keep = keep.loc[:, ["source_name", "province", "adm2", "lat", "lon", "adcode"]].copy()
    keep["lat"] = pd.to_numeric(keep["lat"], errors="coerce")
    keep["lon"] = pd.to_numeric(keep["lon"], errors="coerce")
    keep["adcode"] = pd.to_numeric(keep["adcode"], errors="coerce")
    keep = keep.dropna(subset=["source_name", "province", "lat", "lon", "adcode"])
    keep["adcode"] = keep["adcode"].astype(int)
    keep["norm_name"] = keep["source_name"].map(normalize_place_name)
    keep["norm_adm2"] = keep["adm2"].map(lambda value: normalize_place_name(value) if pd.notna(value) else "")
    keep["level"] = keep["adcode"].map(_admin_level)
    keep = keep[(keep["lat"].between(15, 55)) & (keep["lon"].between(70, 140))]
    return keep.reset_index(drop=True)


def _prefer_row(candidates: pd.DataFrame) -> pd.Series | None:
    if candidates.empty:
        return None
    pool = candidates
    outside = pool[pool["level"] != "省级"]
    municipalities = pool[pool["norm_name"].isin(MUNICIPALITIES) & (pool["level"] == "省级")]
    if len(municipalities) == 1 and pool["norm_name"].iloc[0] in MUNICIPALITIES:
        named = pool[pool["norm_name"].isin(MUNICIPALITIES) & (pool["level"] == "省级")]
        if len(named):
            return named.iloc[0]
    prefectures = pool[pool["level"] == "地级"]
    # 同名区县和地级市同时存在时，监测城市名指地级市，例如朝阳指辽宁朝阳。
    if len(prefectures) and len(pool) > len(prefectures):
        non_muni_district = prefectures[~prefectures["province"].isin(["北京市", "上海市", "天津市", "重庆市"])]
        pool = non_muni_district if len(non_muni_district) else prefectures
    elif len(prefectures) == 1:
        return prefectures.iloc[0]
    if len(pool) == 1:
        return pool.iloc[0]
    same_adm = pool[pool["norm_name"] == pool["norm_adm2"]]
    if len(same_adm) == 1:
        return same_adm.iloc[0]
    if len(pool) > 1:
        return None
    return None


def match_cities(city_names: list[str], gazetteer: pd.DataFrame) -> pd.DataFrame:
    """把监测城市名对到唯一坐标。对不上或重名无法消解的，坐标留空。"""
    by_name: dict[str, pd.DataFrame] = {}
    for key, part in gazetteer.groupby("norm_name"):
        by_name[str(key)] = part
    by_adm2: dict[str, pd.DataFrame] = {}
    for key, part in gazetteer.groupby("norm_adm2"):
        if key:
            by_adm2[str(key)] = part
    rows = []
    for raw_name in city_names:
        key = normalize_place_name(raw_name)
        chosen = _prefer_row(by_name.get(key, gazetteer.iloc[0:0]))
        if chosen is None:
            adm = by_adm2.get(key, gazetteer.iloc[0:0])
            seats = adm[adm["level"].isin(["地级", "省级"])]
            if key in MUNICIPALITIES:
                chosen = _prefer_row(adm)
            elif len(seats) == 1:
                chosen = seats.iloc[0]
            elif len(adm) == 1:
                chosen = adm.iloc[0]
        if chosen is None:
            rows.append({
                "city": raw_name,
                "province": pd.NA,
                "lon": np.nan,
                "lat": np.nan,
                "adcode": pd.NA,
                "source_name": pd.NA,
                "matched": False,
            })
            continue
        rows.append({
            "city": raw_name,
            "province": chosen["province"],
            "lon": float(chosen["lon"]),
            "lat": float(chosen["lat"]),
            "adcode": int(chosen["adcode"]),
            "source_name": chosen["source_name"],
            "matched": True,
        })
    out = pd.DataFrame(rows)
    out["region"] = [
        assign_region(province, city) for province, city in zip(out["province"], out["city"])
    ]
    out["coord_source"] = "和风天气城市列表 Latitude/Longitude"
    return out


def assign_region(province, city: str) -> str:
    """区域是事先写死的分组，用来对比，不是聚类结果。

    广州、惠州的「州」是城市名本身。先按原名判断珠三角，再做后缀归一。
    """
    raw = re.sub(r"\s+", "", str(city)).strip()
    raw = PLACE_ALIASES.get(raw, raw)
    if raw in PEARL_RIVER_DELTA or raw.removesuffix("市") in PEARL_RIVER_DELTA:
        return "珠三角"
    city_key = normalize_place_name(city)
    if city_key in PEARL_RIVER_DELTA:
        return "珠三角"
    if pd.isna(province):
        return "未匹配"
    if province in NORTH_CHINA:
        return "华北"
    if province in YANGTZE_DELTA:
        return "长三角"
    if province in CHENGDU_CHONGQING:
        return "成渝"
    if province in NORTHWEST:
        return "西北"
    return "其他"

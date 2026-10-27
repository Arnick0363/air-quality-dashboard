"""城市代表点与分析区域。

坐标来自和风天气公开城市列表里的城市级代表点，不是监测站坐标。
一个城市只进一个区域，避免对比时重复计数。
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from src.china_settings import (
    CHENGDU_CHONGQING,
    NORTHEAST,
    NORTH,
    NORTHWEST,
    PEARL_RIVER,
    YANGTZE_PROVINCES,
)

ALIASES = {
    "湘西州": "湘西",
    "西双版纳州": "西双版纳",
    "延边州": "延边",
    "恩施土家族苗族自治州": "恩施州",
    "玉树藏族自治州": "玉树州",
    "阿坝州": "阿坝",
    "甘孜州": "甘孜",
    "凉山州": "凉山",
    "临夏州": "临夏",
    "甘南州": "甘南",
    "海北州": "海北",
    "黄南州": "黄南",
    "海南州": "海南",
    "果洛州": "果洛",
    "玉树州": "玉树",
    "海西州": "海西",
    "昌吉州": "昌吉",
    "博州": "博尔塔拉",
    "巴州": "巴音郭楞",
    "伊犁州": "伊犁",
    "克孜勒苏柯尔克孜自治州": "克州",
    "伊犁哈萨克州": "伊犁",
    "锡林郭勒盟": "锡林郭勒",
    "阿拉善盟": "阿拉善",
    "兴安盟": "兴安",
    "黔西南州": "黔西南",
    "黔东南州": "黔东南",
    "黔南州": "黔南",
    "大理州": "大理",
    "德宏州": "德宏",
    "怒江州": "怒江",
    "迪庆州": "迪庆",
    "楚雄州": "楚雄",
    "红河州": "红河",
    "文山州": "文山",
}


def assign_region(city: str, province: str | None) -> str:
    if city in PEARL_RIVER:
        return "珠三角"
    if province in YANGTZE_PROVINCES:
        return "长三角"
    if city in CHENGDU_CHONGQING:
        return "成渝"
    if province in NORTHEAST:
        return "东北"
    if province in NORTH:
        return "华北"
    if province in NORTHWEST:
        return "西北"
    return "其他"


def shorten_province(name: str | None) -> str | None:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    text = str(name).strip()
    for suffix in ("维吾尔自治区", "壮族自治区", "回族自治区", "特别行政区", "自治区", "省", "市"):
        if text.endswith(suffix) and len(text) > len(suffix):
            text = text[: -len(suffix)]
            break
    return text or None


def read_location_list(path: Path) -> pd.DataFrame:
    raw = path.read_text(encoding="utf-8-sig")
    lines = raw.splitlines()
    if lines and lines[0].startswith("China-City-List"):
        raw = "\n".join(lines[1:])
    frame = pd.read_csv(io.StringIO(raw))
    frame["AD_code"] = frame["AD_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    frame["province"] = frame["Adm1_Name_ZH"].map(shorten_province)
    return frame


def _prefer_row(exact: pd.DataFrame) -> pd.Series:
    codes = exact["AD_code"].astype(str)
    prefecture = exact[codes.str.endswith("00") & ~codes.str.endswith("0000")]
    if len(prefecture) >= 1:
        return prefecture.iloc[0]
    municipality = exact[codes.str.endswith("0000")]
    if len(municipality) >= 1:
        return municipality.iloc[0]
    return exact.iloc[0]


def resolve_name(city: str, lookup: dict) -> str | None:
    """先用全名，再用别名。自治州和地区才允许用名称里包含的更短地名。"""
    if city in lookup:
        return city
    alias = ALIASES.get(city)
    if alias in lookup:
        return alias
    if city.endswith(("自治州", "地区", "盟", "州")):
        contained = [name for name in lookup if name and name in city and len(name) >= 2]
        if contained:
            return max(contained, key=len)
    return None


def match_cities(cities: list[str], gazetteer: pd.DataFrame) -> pd.DataFrame:
    lookup = {}
    for name, group in gazetteer.groupby(gazetteer["Location_Name_ZH"].astype(str).str.strip()):
        lookup[str(name)] = group

    rows = []
    for city in cities:
        key = resolve_name(city, lookup)
        group = None if key is None else lookup.get(key)
        if group is None:
            rows.append({
                "city": city,
                "lon": pd.NA,
                "lat": pd.NA,
                "province": None,
                "region": "其他",
                "ad_code": None,
                "coord_note": "城市列表中没有同名代表点",
            })
            continue
        chosen = _prefer_row(group)
        province = chosen["province"]
        code = str(chosen["AD_code"])
        if code.endswith("0000"):
            level = "直辖市或省级代表点"
        elif code.endswith("00"):
            level = "地级代表点"
        else:
            level = "区县代表点"
        rows.append({
            "city": city,
            "lon": float(chosen["Longitude"]),
            "lat": float(chosen["Latitude"]),
            "province": province,
            "region": assign_region(city, province),
            "ad_code": code,
            "coord_note": f"和风天气城市列表，{level}",
        })
    return pd.DataFrame(rows)

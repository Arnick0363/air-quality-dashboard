"""全国城市页共用的读数、筛选和空状态。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd
import streamlit as st

from src.paths import NATIONAL_DAILY_PATH, NATIONAL_LISA_PATH, NATIONAL_PRED_PATH, NATIONAL_SUMMARY_JSON

METRICS = {
    "PM2.5": {"unit": "μg/m³", "limit": 75.0, "label": "PM2.5 日均"},
    "PM10": {"unit": "μg/m³", "limit": 150.0, "label": "PM10 日均"},
    "SO2": {"unit": "μg/m³", "limit": 150.0, "label": "SO2 日均"},
    "NO2": {"unit": "μg/m³", "limit": 80.0, "label": "NO2 日均"},
    "CO": {"unit": "mg/m³", "limit": 4.0, "label": "CO 日均"},
    "O3": {"unit": "μg/m³", "limit": None, "label": "O3 小时值的日均"},
    "AQI": {"unit": "无量纲", "limit": None, "label": "小时 AQI 的日均"},
}

NATIONAL_EMPTY = {
    "file": "还没有全国城市日数据。",
    "provinces": "还没有选择省份。至少选一个省份后，地图和数字会按这个范围重算。",
    "date": "请选择开始和结束两个日期，并且开始不晚于结束。",
    "rows": "这个省份、城市、季节和日期交在一起之后，没有城市日记录。可以改回「全年」或放宽日期。",
    "metric": "当前筛选里，这个指标没有达到「每天至少 18 个有效小时」的城市日。",
}


@dataclass
class NationalFilter:
    ok: bool
    reason: str
    provinces: list[str]
    cities: list[str]
    metric: str
    start: date | None
    end: date | None
    season: str


def missing_national_page() -> None:
    st.markdown(
        """
        <div class="error-box">
          <h3>还没有全国城市日数据</h3>
          <p>仪表盘的全国页面需要 <code>data/processed/national_city_day.parquet</code>。原始逐小时文件不会留下，脚本会边下载边折成日值。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code("python scripts/prepare_national.py", language="text")
    st.caption("这一步要联网，大约覆盖 2023–2024 年的监测城市。北京站点页不依赖它。")


def read_national():
    if not NATIONAL_DAILY_PATH.exists() or not NATIONAL_SUMMARY_JSON.exists():
        return None
    daily = pd.read_parquet(NATIONAL_DAILY_PATH)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["province"] = daily["province"].fillna("未匹配坐标")
    daily["region"] = daily["region"].fillna("未匹配")
    daily["province"] = daily["province"].fillna("未匹配坐标")
    daily["region"] = daily["region"].fillna("未匹配")
    summary = json.loads(NATIONAL_SUMMARY_JSON.read_text(encoding="utf-8"))
    lisa = pd.read_parquet(NATIONAL_LISA_PATH) if NATIONAL_LISA_PATH.exists() else pd.DataFrame()
    pred = pd.read_parquet(NATIONAL_PRED_PATH) if NATIONAL_PRED_PATH.exists() else pd.DataFrame()
    if not pred.empty:
        pred["date"] = pd.to_datetime(pred["date"])
    return daily, summary, lisa, pred


@st.cache_data(show_spinner="正在读取全国城市日数据…")
def cached_national():
    return read_national()


def render_national_sidebar(daily: pd.DataFrame) -> NationalFilter:
    st.sidebar.header("全国城市")
    st.sidebar.caption("筛选的是监测城市日值。北京 12 个国控站在上面的「北京站点」。")
    provinces = sorted(province for province in daily["province"].dropna().unique())
    chosen_provinces = st.sidebar.multiselect(
        "省份",
        provinces,
        default=provinces,
        key="nat_provinces",
        placeholder="请选择省份",
    )
    available = sorted(daily.loc[daily["province"].isin(chosen_provinces), "city"].unique())
    city_kwargs = {"placeholder": "可搜索城市", "key": "nat_cities"}
    if "nat_cities" in st.session_state:
        kept = [city for city in st.session_state["nat_cities"] if city in available]
        if kept != list(st.session_state["nat_cities"]):
            st.session_state["nat_cities"] = kept
    else:
        city_kwargs["default"] = []
    chosen_cities = st.sidebar.multiselect(
        "城市（留空表示当前省份的全部）",
        available,
        **city_kwargs,
    )
    metric = st.sidebar.selectbox(
        "指标",
        list(METRICS),
        format_func=lambda name: f"{METRICS[name]['label']}（{METRICS[name]['unit']}）",
        key="nat_metric",
    )
    season = st.sidebar.radio("季节", ["全年", "春", "夏", "秋", "冬"], horizontal=True, key="nat_season")
    min_date = daily["date"].min().date()
    max_date = daily["date"].max().date()
    picked = st.sidebar.date_input(
        "时间范围",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="nat_dates",
    )
    st.sidebar.caption(
        f"全库 {daily['city'].nunique()} 个城市名 · {daily['date'].nunique()} 天 · "
        f"{min_date.isoformat()} 至 {max_date.isoformat()}"
    )
    st.sidebar.caption("日均要求当天至少 18 个有效小时。没有坐标的城市在省份里标成「未匹配坐标」，不会画到地图上。")
    start = end = None
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        start, end = picked
    elif isinstance(picked, (date, datetime)):
        start = end = picked if isinstance(picked, date) else picked.date()
    if not chosen_provinces:
        return NationalFilter(False, "provinces", [], chosen_cities, metric, start, end, season)
    if start is None or end is None or start > end:
        return NationalFilter(False, "date", chosen_provinces, chosen_cities, metric, start, end, season)
    return NationalFilter(True, "", chosen_provinces, chosen_cities, metric, start, end, season)


def apply_filter(daily: pd.DataFrame, state: NationalFilter) -> pd.DataFrame:
    part = daily.loc[daily["province"].isin(state.provinces)]
    if state.cities:
        part = part.loc[part["city"].isin(state.cities)]
    start = pd.Timestamp(state.start)
    end = pd.Timestamp(state.end)
    part = part.loc[(part["date"] >= start) & (part["date"] <= end)]
    if state.season != "全年":
        part = part.loc[part["season"] == state.season]
    return part


def format_metric(value, metric: str) -> str:
    if value is None or value != value:
        return "—"
    if metric == "CO":
        return f"{float(value):.2f}"
    if metric == "AQI":
        return f"{float(value):.0f}"
    return f"{float(value):.1f}"


def national_empty(reason: str) -> None:
    text = NATIONAL_EMPTY.get(reason, reason)
    st.markdown(
        f'<div class="empty-box"><h3>这个筛选下没有可计算的结果</h3><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


def show_map(fmap, signature: str, height: int = 560) -> None:
    from streamlit_folium import st_folium

    try:
        st_folium(fmap, height=height, use_container_width=True, returned_objects=[], key=signature)
    except TypeError:
        st_folium(fmap, height=height, width=900, returned_objects=[], key=signature)

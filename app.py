"""北京空气质量仪表盘入口。

运行：streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.ui import inject_css, load_hourly, missing_data_page, render_sidebar

st.set_page_config(
    page_title="北京空气质量时空分析",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_css()

hourly = load_hourly()
if hourly is None:
    missing_data_page()
    st.stop()

st.session_state["filter_state"] = render_sidebar(hourly)

pages = [
    st.Page("views/overview.py", title="总览", default=True),
    st.Page("views/eda.py", title="趋势与季节"),
    st.Page("views/spatial.py", title="空间分布"),
    st.Page("views/advanced.py", title="聚类与关联"),
]
try:
    navigation = st.navigation(pages, position="top")
except TypeError:
    navigation = st.navigation(pages)
navigation.run()

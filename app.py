"""中国城市空气质量仪表盘入口。

运行：streamlit run app.py
全国城市是日尺度，北京 12 站是小时尺度。
没有许可清楚的检测图像和标注，所以这里不设 YOLO 页面。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.ui import inject_css

st.set_page_config(
    page_title="中国城市空气质量时空分析",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="auto",
)
inject_css()

pages = {
    "全国城市": [
        st.Page("views/national_overview.py", title="总览", default=True),
        st.Page("views/national_compare.py", title="区域与季节"),
        st.Page("views/national_spatial.py", title="空间关联"),
        st.Page("views/national_model.py", title="城市预测"),
    ],
    "北京站点": [
        st.Page("views/overview.py", title="总览"),
        st.Page("views/eda.py", title="趋势与季节"),
        st.Page("views/spatial.py", title="空间分布"),
        st.Page("views/advanced.py", title="聚类与关联"),
    ],
}
try:
    navigation = st.navigation(pages, position="top")
except TypeError:
    flat = [page for group in pages.values() for page in group]
    navigation = st.navigation(flat)
navigation.run()

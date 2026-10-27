"""项目路径。所有脚本与页面都从这里取目录，避免写死盘符。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
HOURLY_PATH = DATA_PROCESSED / "hourly.parquet"
DAILY_PATH = DATA_PROCESSED / "daily.parquet"
NATIONAL_DAILY_PATH = DATA_PROCESSED / "national_city_day.parquet"
NATIONAL_CITIES_PATH = DATA_PROCESSED / "national_cities.parquet"
NATIONAL_LISA_PATH = DATA_PROCESSED / "national_lisa.parquet"
NATIONAL_PRED_PATH = DATA_PROCESSED / "national_predictions.parquet"
NATIONAL_PARTS = DATA_RAW / "national_parts"
CLEANING_JSON = OUTPUTS / "cleaning_report.json"
SUMMARY_JSON = OUTPUTS / "analysis_summary.json"
NATIONAL_SUMMARY_JSON = OUTPUTS / "national_summary.json"
YOLO_DIR = OUTPUTS / "yolo"
FINDINGS_MD = ROOT / "docs" / "findings.md"
CHINA_MONTHS = DATA_RAW / "china_months"
CHINA_DAILY_PATH = DATA_PROCESSED / "china_city_daily.parquet"
CHINA_META_PATH = DATA_PROCESSED / "china_city_meta.parquet"
CHINA_LOCAL_PATH = DATA_PROCESSED / "china_local_moran.parquet"
CHINA_FORECAST_PATH = DATA_PROCESSED / "china_forecast_test.parquet"
CHINA_SUMMARY_JSON = OUTPUTS / "china_summary.json"
CHINA_FINDINGS_MD = ROOT / "docs" / "china-findings.md"
LOCATION_LIST = DATA_RAW / "China-City-List-latest.csv"
INTERVIEW_PDF = ROOT / "docs" / "must-spatial-big-data-interview.pdf"

# 官方压缩包解压后的目录名
RAW_SUBDIR = DATA_RAW / "PRSA_Data_20130301-20170228"
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
    "PRSA2017_Data_20130301-20170228.zip"
)
UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data"
UCI_DOI = "https://doi.org/10.24432/C5RK5G"

"""项目路径。所有脚本与页面都从这里取目录，避免写死盘符。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIG_DIR = OUTPUTS / "figures"
HOURLY_PATH = DATA_PROCESSED / "hourly.parquet"
DAILY_PATH = DATA_PROCESSED / "daily.parquet"
CLEANING_JSON = OUTPUTS / "cleaning_report.json"
SUMMARY_JSON = OUTPUTS / "analysis_summary.json"
FINDINGS_MD = ROOT / "docs" / "findings.md"

# 官方压缩包解压后的目录名
RAW_SUBDIR = DATA_RAW / "PRSA_Data_20130301-20170228"
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/"
    "PRSA2017_Data_20130301-20170228.zip"
)
UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data"
UCI_DOI = "https://doi.org/10.24432/C5RK5G"

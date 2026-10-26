"""下载北京多站点空气质量原始 CSV。

优先使用已经放在 data/raw 的官方压缩包；没有则从 UCI 续传下载。
UCI 不通时，再尝试 Hugging Face 上同一公开数据集的 CSV 镜像。
"""

from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.paths import DATA_RAW, RAW_SUBDIR, UCI_ZIP_URL
from src.settings import STATION_IDS

ZIP_PATH = DATA_RAW / "PRSA2017_Data_20130301-20170228.zip"
HF_PREFIXES = [
    "https://hf-mirror.com/datasets/soizhiwen/time-series/resolve/main/BeijingAirQuality/",
    "https://huggingface.co/datasets/soizhiwen/time-series/resolve/main/BeijingAirQuality/",
]


def _session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "air-quality-dashboard/prepare"
    return session


def _find_csvs(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    files = sorted(folder.rglob("PRSA_Data_*.csv"))
    return files


def _zip_has_csvs(path: Path) -> bool:
    if not path.exists() or not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".csv")]
    return len(names) >= 12


def download_uci_zip(path: Path = ZIP_PATH, url: str = UCI_ZIP_URL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    session = _session()
    last_report = 0
    for attempt in range(1, 40):
        have = path.stat().st_size if path.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 90)) as resp:
                print(f"UCI attempt={attempt} status={resp.status_code} have={have}", flush=True)
                if resp.status_code == 200 and have:
                    path.unlink(missing_ok=True)
                    have = 0
                if resp.status_code not in (200, 206):
                    resp.raise_for_status()
                mode = "ab" if resp.status_code == 206 else "wb"
                with path.open(mode) as handle:
                    for chunk in resp.iter_content(64 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        have += len(chunk)
                        if have - last_report >= 512 * 1024:
                            print(f"bytes={have}", flush=True)
                            last_report = have
            if _zip_has_csvs(path):
                print(f"ZIP_OK {path} bytes={path.stat().st_size}", flush=True)
                return path
            print("压缩包不完整，准备重试", flush=True)
            path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"UCI error attempt={attempt}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(min(8, attempt))
    raise RuntimeError(f"无法从 UCI 下载数据集: {url}")


def _download_hf_csvs(target: Path) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    session = _session()
    saved = []
    for station in STATION_IDS:
        name = f"PRSA_Data_{station}_20130301-20170228.csv"
        dest = target / name
        if dest.exists() and dest.stat().st_size > 100_000:
            saved.append(dest)
            continue
        last_error = None
        for prefix in HF_PREFIXES:
            try:
                with session.get(prefix + name, stream=True, timeout=(20, 60)) as resp:
                    resp.raise_for_status()
                    with dest.open("wb") as handle:
                        for chunk in resp.iter_content(64 * 1024):
                            if chunk:
                                handle.write(chunk)
                if dest.stat().st_size > 100_000:
                    print(f"CSV_OK {name} via {prefix}", flush=True)
                    saved.append(dest)
                    last_error = None
                    break
            except Exception as exc:
                last_error = exc
                dest.unlink(missing_ok=True)
        if last_error is not None and dest not in saved:
            raise RuntimeError(f"镜像下载失败 {name}: {last_error}")
    return saved


def extract_zip(path: Path, dest: Path = DATA_RAW) -> list[Path]:
    with zipfile.ZipFile(path) as archive:
        archive.extractall(dest)
    files = _find_csvs(dest)
    if len(files) < 12:
        raise FileNotFoundError(f"解压后只找到 {len(files)} 个 CSV")
    return files


def ensure_raw_csvs() -> list[Path]:
    """返回 12 个站点 CSV 的路径。"""
    existing = _find_csvs(DATA_RAW)
    if len(existing) >= 12:
        print(f"使用已有 CSV {len(existing)} 个", flush=True)
        return existing
    candidates = [ZIP_PATH, DATA_RAW / "prsa.zip"]
    candidates += list(DATA_RAW.glob("*.zip"))
    for candidate in candidates:
        if _zip_has_csvs(candidate):
            print(f"解压 {candidate}", flush=True)
            return extract_zip(candidate)
    print("未找到本地压缩包，开始下载 UCI 官方文件", flush=True)
    try:
        zpath = download_uci_zip()
        return extract_zip(zpath)
    except Exception as exc:
        print(f"UCI 下载失败，改试公开 CSV 镜像: {exc}", flush=True)
        files = _download_hf_csvs(RAW_SUBDIR)
        if len(files) < 12:
            raise
        return files


if __name__ == "__main__":
    paths = ensure_raw_csvs()
    print(f"READY {len(paths)} files")
    for path in paths:
        print(path)

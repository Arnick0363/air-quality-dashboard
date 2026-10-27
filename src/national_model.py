"""用截至前一日的信息预测城市当日 PM2.5。

特征里没有当天的污染物，也没有把样本随机打乱。测试集是最后一段日期。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from src.spatial_stats import knn_weights, spatial_lag_matrix

FEATURE_COLUMNS = [
    "lag1",
    "lag7",
    "spatial_lag1",
    "month_sin",
    "month_cos",
    "dow",
    "lat",
    "lon",
]

FEATURE_LABELS = {
    "lag1": "昨日 PM2.5",
    "lag7": "7 日前 PM2.5",
    "spatial_lag1": "邻近城市昨日 PM2.5",
    "month_sin": "月份正弦",
    "month_cos": "月份余弦",
    "dow": "星期",
    "lat": "纬度",
    "lon": "经度",
}


def _wide_to_long(wide: pd.DataFrame, name: str) -> pd.DataFrame:
    out = wide.stack().rename(name).reset_index()
    out.columns = ["date", "city", name]
    return out


def build_supervised(panel: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """构造预测表。y 是当日 PM2.5，特征只使用更早的浓度、日历和固定坐标。"""
    required = {"date", "city", "PM2.5", "lat", "lon"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"缺少列: {sorted(missing)}")
    frame = panel.loc[:, ["date", "city", "PM2.5", "lat", "lon"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=["lat", "lon"])
    meta = frame.groupby("city", as_index=False)[["lat", "lon"]].first()
    if len(meta) < 2:
        raise ValueError("有坐标的城市少于 2 个，做不了空间滞后")
    order = meta["city"].tolist()
    weights = knn_weights(meta["lat"].to_numpy(), meta["lon"].to_numpy(), k=k)
    wide = frame.pivot_table(index="date", columns="city", values="PM2.5", aggfunc="mean")
    wide = wide.reindex(columns=order).sort_index()
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))
    lag1 = wide.shift(1)
    lag7 = wide.shift(7)
    lagged_space = pd.DataFrame(
        spatial_lag_matrix(lag1.to_numpy(dtype=float), weights),
        index=wide.index,
        columns=order,
    )
    base = frame.drop_duplicates(["date", "city"]).rename(columns={"PM2.5": "y"})
    base = base.merge(_wide_to_long(lag1, "lag1"), on=["date", "city"], how="left")
    base = base.merge(_wide_to_long(lag7, "lag7"), on=["date", "city"], how="left")
    base = base.merge(_wide_to_long(lagged_space, "spatial_lag1"), on=["date", "city"], how="left")
    month = base["date"].dt.month.to_numpy()
    base["month_sin"] = np.sin(2 * np.pi * month / 12)
    base["month_cos"] = np.cos(2 * np.pi * month / 12)
    base["dow"] = base["date"].dt.dayofweek.astype(float)
    return base.reset_index(drop=True)


def time_split(frame: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """按日期切。训练集的每一天都早于测试集，不打乱。"""
    if not 0 < test_fraction < 0.5:
        raise ValueError("测试比例要在 0 和 0.5 之间")
    dates = np.array(sorted(pd.to_datetime(frame["date"]).unique()))
    if len(dates) < 30:
        raise ValueError("可用日期少于 30 天，时间切分不稳定")
    cut_at = int(round(len(dates) * (1 - test_fraction)))
    cut_at = min(max(cut_at, 1), len(dates) - 1)
    cutoff = pd.Timestamp(dates[cut_at])
    train = frame.loc[frame["date"] < cutoff].copy()
    test = frame.loc[frame["date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("时间切分后训练或测试为空")
    return train, test, cutoff


def _mae_rmse(y_true, y_pred) -> tuple[float, float]:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))


def fit_pm25_model(supervised: pd.DataFrame, test_fraction: float = 0.25, seed: int = 42) -> dict:
    """梯度提升树对照持久性基线。持久性基线就是预测成昨天的值。"""
    usable = supervised.dropna(subset=["y", *FEATURE_COLUMNS]).copy()
    if usable["city"].nunique() < 5:
        raise ValueError("有坐标的城市少于 5 个，不做预测")
    train, test, cutoff = time_split(usable, test_fraction=test_fraction)
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=250,
        min_samples_leaf=40,
        random_state=seed,
    )
    model.fit(train[FEATURE_COLUMNS], train["y"])
    train_pred = model.predict(train[FEATURE_COLUMNS])
    test_pred = model.predict(test[FEATURE_COLUMNS])
    train_mae, train_rmse = _mae_rmse(train["y"], train_pred)
    test_mae, test_rmse = _mae_rmse(test["y"], test_pred)
    persist_mae, persist_rmse = _mae_rmse(test["y"], test["lag1"])
    sample_n = min(12000, len(test))
    sample = test.sample(n=sample_n, random_state=seed)
    importance = permutation_importance(
        model,
        sample[FEATURE_COLUMNS],
        sample["y"],
        n_repeats=5,
        random_state=seed,
        scoring="neg_mean_absolute_error",
    )
    ranked = []
    for name, mean, std in zip(FEATURE_COLUMNS, importance.importances_mean, importance.importances_std):
        ranked.append({
            "feature": name,
            "label": FEATURE_LABELS[name],
            "mae_increase": float(mean),
            "mae_increase_std": float(std),
        })
    ranked.sort(key=lambda item: item["mae_increase"], reverse=True)
    predictions = test.loc[:, ["date", "city", "y", "lag1"]].copy()
    predictions["y_model"] = test_pred
    predictions["y_persist"] = test["lag1"].to_numpy()
    summary = {
        "task": "用前一日及更早的信息预测当日城市 PM2.5",
        "model": "sklearn HistGradientBoostingRegressor",
        "features": FEATURE_COLUMNS,
        "cutoff": cutoff.date().isoformat(),
        "train_start": pd.Timestamp(train["date"].min()).date().isoformat(),
        "train_end": pd.Timestamp(train["date"].max()).date().isoformat(),
        "test_start": pd.Timestamp(test["date"].min()).date().isoformat(),
        "test_end": pd.Timestamp(test["date"].max()).date().isoformat(),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_cities": int(usable["city"].nunique()),
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "persist_mae": persist_mae,
        "persist_rmse": persist_rmse,
        "importance_sample_n": int(sample_n),
        "importance": ranked,
        "seed": seed,
    }
    return {"summary": summary, "predictions": predictions}

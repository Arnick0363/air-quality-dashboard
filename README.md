# 北京多站点空气质量时空分析

训练项目。用 2013–2017 年北京 12 个国控站的公开小时数据，做时间趋势、季节与日变化、站点差异、气象关联和空间插值，并用 Streamlit 做成可筛选的中文仪表盘。

全周期城市日均 PM2.5 为 **79.4 μg/m³**。冬季 **94.2**，夏季 **64.6**，冬季是夏季的 **1.46** 倍。完整年份里，2016 年（71.8）比 2014 年（84.5）低 12.7 μg/m³。

![总览页：默认筛选下的指标卡](docs/screenshots/overview-desktop.png)

本地窄屏下指标卡改为上下排列，侧栏收起：

![窄屏总览](docs/screenshots/overview-narrow.png)

本地演示：安装依赖后运行 `streamlit run app.py`，打开 http://localhost:8501 。处理后的数据已在仓库里，不必先下载。  
在线链接：尚未部署。

## 数据来源

- 数据集：[Beijing Multi-Site Air-Quality Data](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)，UCI，DOI [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G)
- 论文：Zhang 等，2017，*Cautionary tales on air-quality improvement in Beijing*，Proceedings of the Royal Society A
- 时段：2013-03-01 00:00 至 2017-02-28 23:00，12 站 × 35064 小时 = 420768 条
- 污染物：PM2.5、PM10、SO2、NO2、CO、O3（μg/m³）
- 气象：气温、气压、露点、降水、风向、风速
- 字段、站点坐标和清洗规则见 [docs/data-dictionary.md](docs/data-dictionary.md)

处理后的 `data/processed/hourly.parquet` 与 `daily.parquet` 随仓库提供，克隆后即可打开仪表盘。原始压缩包和 CSV 仍不进入版本库。若要按 UCI 原始文件重新清洗，再运行 `python scripts/prepare_data.py`，它会下载压缩包并覆盖这两份 parquet。

## 复现步骤

在项目根目录执行。Windows 可以用 PowerShell。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

仓库里已经有：

- `data/processed/hourly.parquet`、`daily.parquet`（约 7.3 MB + 0.6 MB）
- `outputs/figures/` 下的静态图
- `docs/findings.md` 与 `outputs/analysis_summary.json`

要从原始压缩包重做清洗时，再运行 `.\.venv\Scripts\python.exe scripts\prepare_data.py`。这一步需要联网下载，原始包仍只留在本地。

测试（不下载数据）：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

仪表盘有四页：总览、趋势与季节、空间分布、聚类与关联。侧栏可以改城市（只有北京）、年份、站点、污染物、季节范围和日期。臭氧的日序列和超标用日最大 8 小时平均，日变化图仍用小时浓度。没有选站点、没有有效日，或地图组件失败时，页面给出空状态或错误说明，而不是空白图。

## 结果与结论

下面的数字来自清洗后的全库计算，完整列表和表格在 [docs/findings.md](docs/findings.md)。日均值要求当天至少 18 个有效小时；城市日至少 8 个站点有效。2013 与 2017 不是完整年，年均比较只用 2014–2016。

1. 冬季城市日均 PM2.5 为 94.2 μg/m³（359 天），夏季为 64.6 μg/m³（368 天），冬季是夏季的 1.46 倍。跨年平均后，最高的月份是 12 月（103.9），最低是 8 月（53.5）。
2. 2014 年城市日均 PM2.5 为 84.5 μg/m³（363 天），2016 年为 71.8 μg/m³（365 天），低 12.7 μg/m³（-15.0%）。
3. 全周期 1458 个有效城市日里，日均超过 75 μg/m³ 的有 584 天（40.1%），超过 150 μg/m³ 的有 174 天（11.9%），并成 84 次重度过程。峰值最高的一次是 2015-11-27 至 2015-12-01，峰值日 2015-12-01 的城市日均是 472.9 μg/m³；这一天 12 个站都有 24 小时数据，站日均落在 379.7 到 519.8 之间。
4. 这些重度过程按持续天数加权的平均风速是 1.21 m/s，全部有效城市日是 1.73 m/s。过程期间平均露点 1.31 °C，全周期是 2.48 °C。城市日尺度上，PM2.5 与风速 r=-0.43（n=1458），与露点只有 r=0.08。能稳定看到的是风速偏低与高污染同现，不是露点越高污染越高。
5. 站点日均 PM2.5 最高是东四（86.0），最低是定陵（65.7）。层次聚类取 k=2（轮廓系数 0.453）：定陵、怀柔为较低组，其余 10 站为较高组。留一法里，逆距离加权 RMSE 3.81，最近邻 4.73。这只说明长期平均在站点之间比较平滑，不能当成小时污染地图。

![城市日均 PM2.5 的季节平均](outputs/figures/seasonal_pm25.png)

![空间分布：站点、热力与逆距离加权](docs/screenshots/spatial-map.png)

![各站 PM2.5](outputs/figures/station_rank_pm25.png)

![月均序列](outputs/figures/monthly_pm25.png)

![站点聚类](outputs/figures/cluster_dendrogram.png)

其他污染物的城市日尺度平均（臭氧为日最大 8 小时平均）：PM10 104.5、SO2 15.8、NO2 50.5、CO 1228、O3 99.9。SO2 没有城市日超过 150 μg/m³；PM2.5、PM10、NO2、CO、O3 的超标天数见 findings 里的表。

小时平均的 PM2.5 在 22 时最高（89.0），7 时最低（73.3）。工作日城市日均 77.6，周末 83.8。

## 局限与致谢

- 短缺口才线性插值（连续不超过 6 小时，整段更长的不补）。插值前 PM2.5 缺失率 2.09%，插值后 1.06%。降水和风向不插值。
- 约 4.97% 的成对小时里 PM2.5 高于 PM10，已保留，没有当成异常删除。
- 日序列有自相关，相关和季节差只作描述。
- 空间插值使用文献中的站点坐标，格点距最近站点超过 40 公里不着色。
- 标准对照的是 GB 3095-2012 二级限值，用来给筛选结果一个参照，不是正式评价报告。

交互形式借鉴了 Febrianss 的公开项目 [beijing-prsa-2019](https://github.com/fbrianzy/beijing-prsa-2019)（MIT License, Copyright (c) 2026 Febrianss）。本仓库的清洗、指标、页面和结论文案是单独写的，没有复制该仓库源码。

## 在线演示

尚未部署，本 README 不提供在线地址。处理后的 parquet 已随仓库提供，克隆后按上面的步骤即可在本地运行。原始压缩包仍需本地下载，只有重做清洗时才需要。

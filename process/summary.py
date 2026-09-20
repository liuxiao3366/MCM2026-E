import pandas as pd
import numpy as np

# 1. 读取数据
path = "Attachment/Attachment1.xlsx"
df = pd.read_excel(path, sheet_name="Sheet1")

# 2. 基本检查
print(df.head())
print(df.columns)
print(df.shape)

# 3. 按方案汇总
summary = (
    df.groupby("方案ID", as_index=False)
      .agg(
          投放天数=("日期", "nunique"),

          总消费=("消费额", "sum"),
          总展现=("展现量", "sum"),
          总点击=("点击量", "sum"),

          上方位展现=("上方位展现量", "sum"),
          上方首位展现=("上方首位展现量", "sum"),
          上方位点击=("上方位点击量", "sum"),
          上方位消费=("上方位消费额", "sum"),
      )
)

# 4. 构造效率指标

# 点击率：看到广告的人中有多少进行了点击
summary["CTR"] = (
    summary["总点击"] / summary["总展现"]
)

# 单次点击成本
summary["CPC"] = np.where(
    summary["总点击"] > 0,
    summary["总消费"] / summary["总点击"],
    np.nan
)

# 广告进入搜索结果上方区域的比例
summary["上方位展现率"] = (
    summary["上方位展现"] / summary["总展现"]
)

# 广告位于第一位的比例
summary["首位展现率"] = (
    summary["上方首位展现"] / summary["总展现"]
)

# 上方位置广告的点击率
summary["上方位CTR"] = np.where(
    summary["上方位展现"] > 0,
    summary["上方位点击"] / summary["上方位展现"],
    np.nan
)

# 上方位置平均点击成本
summary["上方位CPC"] = np.where(
    summary["上方位点击"] > 0,
    summary["上方位消费"] / summary["上方位点击"],
    np.nan
)

# 5. 排序，方便看
summary = summary.sort_values(
    "总消费",
    ascending=False
)

# 6. 打印
pd.set_option("display.max_columns", None)
print(summary)

# 7. 保存
summary.to_excel(
    "方案级汇总.xlsx",
    index=False
)



unit_summary = (
    df.groupby(["方案ID", "推广单元ID"], as_index=False)
      .agg(
          投放天数=("日期", "nunique"),
          总消费=("消费额", "sum"),
          总展现=("展现量", "sum"),
          总点击=("点击量", "sum"),
          上方位展现=("上方位展现量", "sum"),
          上方首位展现=("上方首位展现量", "sum"),
          上方位点击=("上方位点击量", "sum"),
          上方位消费=("上方位消费额", "sum"),
      )
)

unit_summary["CTR"] = unit_summary["总点击"] / unit_summary["总展现"]

unit_summary["CPC"] = (
    unit_summary["总消费"] / unit_summary["总点击"]
)

unit_summary["上方位展现率"] = (
    unit_summary["上方位展现"] / unit_summary["总展现"]
)

unit_summary["首位展现率"] = (
    unit_summary["上方首位展现"] / unit_summary["总展现"]
)

print(
    unit_summary.sort_values("CPC")
)
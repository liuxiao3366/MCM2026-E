"""构造问题一的日级汇总数据，并绘制全年时间序列图。

输出：
1. data/processed/daily_summary.csv
2. figures/q1_daily_time_series.png

本脚本只进行数据汇总和可视化，不进行任何模型拟合。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CANDIDATES = (
    PROJECT_ROOT / "data" / "raw" / "Attachment" / "Attachment1.xlsx",
    PROJECT_ROOT / "Attachment" / "Attachment1.xlsx",
)
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "daily_summary.csv"
OUTPUT_FIGURE = PROJECT_ROOT / "figures" / "q1_daily_time_series.png"
ROLLING_WINDOW = 7


def find_input_file() -> Path:
    """返回附件 1 的实际路径，并兼容项目整理前后的目录结构。"""
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    candidates = "\n".join(f"- {path}" for path in INPUT_CANDIDATES)
    raise FileNotFoundError(f"未找到 Attachment1.xlsx，已检查：\n{candidates}")


def require_columns(df: pd.DataFrame, required: set[str], sheet_name: str) -> None:
    """检查工作表是否包含所需字段。"""
    missing = required.difference(df.columns)
    if missing:
        missing_text = "、".join(sorted(missing))
        raise ValueError(f"{sheet_name} 缺少字段：{missing_text}")


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """安全计算比率；分母为零时返回 NaN。"""
    return numerator.div(denominator.replace(0, np.nan))


def load_and_summarize(input_file: Path) -> pd.DataFrame:
    """读取 Sheet1、Sheet2，生成日期粒度的汇总表。"""
    sheet1 = pd.read_excel(input_file, sheet_name="Sheet1")
    sheet2 = pd.read_excel(input_file, sheet_name="Sheet2")

    # 原始 Sheet2 的“新注册数”字段可能含有尾部空格，统一清理列名。
    sheet1.columns = sheet1.columns.astype(str).str.strip()
    sheet2.columns = sheet2.columns.astype(str).str.strip()

    sheet1_required = {
        "日期",
        "展现量",
        "点击量",
        "消费额",
        "上方位展现量",
        "上方首位展现量",
    }
    sheet2_required = {"日期", "新注册数"}
    require_columns(sheet1, sheet1_required, "Sheet1")
    require_columns(sheet2, sheet2_required, "Sheet2")

    sheet1["日期"] = pd.to_datetime(sheet1["日期"], errors="coerce")
    sheet2["日期"] = pd.to_datetime(sheet2["日期"], errors="coerce")
    if sheet1["日期"].isna().any() or sheet2["日期"].isna().any():
        raise ValueError("日期字段存在无法解析的值，请检查附件 1。")

    sheet1_numeric_columns = [
        "展现量",
        "点击量",
        "消费额",
        "上方位展现量",
        "上方首位展现量",
    ]
    for column in sheet1_numeric_columns:
        sheet1[column] = pd.to_numeric(sheet1[column], errors="raise")
    sheet2["新注册数"] = pd.to_numeric(sheet2["新注册数"], errors="raise")

    daily = (
        sheet1.groupby("日期", as_index=False)[sheet1_numeric_columns]
        .sum()
        .rename(
            columns={
                "展现量": "总展现量",
                "点击量": "总点击量",
                "消费额": "总消费额",
            }
        )
    )

    daily["CTR"] = safe_ratio(daily["总点击量"], daily["总展现量"])
    daily["CPC"] = safe_ratio(daily["总消费额"], daily["总点击量"])
    daily["上方位展现率"] = safe_ratio(daily["上方位展现量"], daily["总展现量"])
    daily["首位展现率"] = safe_ratio(daily["上方首位展现量"], daily["总展现量"])

    registrations = sheet2[["日期", "新注册数"]].copy()
    if registrations["日期"].duplicated().any():
        raise ValueError("Sheet2 存在重复日期，无法进行一对一合并。")

    daily = daily.merge(
        registrations,
        on="日期",
        how="left",
        validate="one_to_one",
    )
    if daily["新注册数"].isna().any():
        missing_dates = daily.loc[daily["新注册数"].isna(), "日期"]
        missing_text = "、".join(missing_dates.dt.strftime("%Y-%m-%d"))
        raise ValueError(f"以下投放日期缺少新注册数：{missing_text}")

    output_columns = [
        "日期",
        "总展现量",
        "总点击量",
        "总消费额",
        "上方位展现量",
        "上方首位展现量",
        "CTR",
        "CPC",
        "上方位展现率",
        "首位展现率",
        "新注册数",
    ]
    return daily[output_columns].sort_values("日期").reset_index(drop=True)


def add_rolling_series(daily: pd.DataFrame) -> pd.DataFrame:
    """为绘图构造 7 日移动指标，不改变最终 CSV 的字段。"""
    plot_data = daily.copy()
    rolling = plot_data[
        ["总展现量", "总点击量", "总消费额", "新注册数"]
    ].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW)

    plot_data["总消费额_7日"] = rolling["总消费额"].mean()
    plot_data["总点击量_7日"] = rolling["总点击量"].mean()
    plot_data["新注册数_7日"] = rolling["新注册数"].mean()

    rolling_impressions = rolling["总展现量"].sum()
    rolling_clicks = rolling["总点击量"].sum()
    rolling_cost = rolling["总消费额"].sum()
    plot_data["CTR_7日"] = safe_ratio(rolling_clicks, rolling_impressions)
    plot_data["CPC_7日"] = safe_ratio(rolling_cost, rolling_clicks)
    return plot_data


def plot_time_series(daily: pd.DataFrame, output_file: Path) -> None:
    """绘制五项日级指标及其 7 日移动线。"""
    plot_data = add_rolling_series(daily)

    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "PingFang SC",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    figure, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=True)
    dates = plot_data["日期"]
    panels = [
        ("总消费额", "总消费额_7日", "每日总消费", "金额（元）", False),
        ("总点击量", "总点击量_7日", "每日总点击", "点击量", False),
        ("CTR", "CTR_7日", "每日点击率（CTR）", "CTR（%）", True),
        ("CPC", "CPC_7日", "每日单次点击成本（CPC）", "CPC（元/次）", False),
        ("新注册数", "新注册数_7日", "每日新注册数", "新注册数", False),
    ]

    for axis, (raw_column, rolling_column, title, ylabel, as_percent) in zip(
        axes, panels
    ):
        scale = 100 if as_percent else 1
        axis.plot(
            dates,
            plot_data[raw_column] * scale,
            color="#8FB9DD",
            linewidth=0.8,
            alpha=0.65,
            label="每日值",
        )
        axis.plot(
            dates,
            plot_data[rolling_column] * scale,
            color="#C83E4D",
            linewidth=2.0,
            label="7 日移动线",
        )
        axis.set_title(title, loc="left", fontsize=12)
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
        axis.legend(loc="upper right", frameon=False)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].set_xlabel("日期")
    figure.suptitle("2025 年 SEM 投放与新注册数全年时间序列", fontsize=16, y=0.995)
    figure.autofmt_xdate(rotation=30, ha="right")
    figure.tight_layout(rect=(0, 0, 1, 0.985))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    input_file = find_input_file()
    daily = load_and_summarize(input_file)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        float_format="%.6f",
    )
    plot_time_series(daily, OUTPUT_FIGURE)

    print(f"读取文件：{input_file}")
    print(f"日级汇总：{OUTPUT_CSV}（{len(daily)} 行）")
    print(f"时间序列图：{OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()

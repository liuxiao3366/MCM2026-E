"""问题一：星期效应与月份—方案分布的描述性统计和可视化。

本脚本只进行数据聚合、描述性统计和绘图，不进行模型拟合、聚类、
机器学习或综合评价。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.set_loglevel("error")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CANDIDATES = (
    PROJECT_ROOT / "data" / "raw" / "Attachment" / "Attachment1.xlsx",
    PROJECT_ROOT / "Attachment" / "Attachment1.xlsx",
)
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"

WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
MONTHS = list(range(1, 13))


def find_input_file() -> Path:
    """返回附件 1 的实际路径，并兼容项目整理前后的目录结构。"""
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    checked = "\n".join(f"- {path}" for path in INPUT_CANDIDATES)
    raise FileNotFoundError(f"未找到 Attachment1.xlsx，已检查：\n{checked}")


def require_columns(df: pd.DataFrame, required: set[str], sheet_name: str) -> None:
    """检查工作表是否包含所需字段。"""
    missing = required.difference(df.columns)
    if missing:
        missing_text = "、".join(sorted(missing))
        raise ValueError(f"{sheet_name} 缺少字段：{missing_text}")


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """计算比率，分母为零时返回 NaN。"""
    return numerator.div(denominator.replace(0, np.nan))


def configure_chinese_font() -> None:
    """设置常见中文字体回退顺序。"""
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "PingFang SC",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_data(input_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取并校验 Sheet1、Sheet2。"""
    sheet1 = pd.read_excel(input_file, sheet_name="Sheet1")
    sheet2 = pd.read_excel(input_file, sheet_name="Sheet2")
    sheet1.columns = sheet1.columns.astype(str).str.strip()
    sheet2.columns = sheet2.columns.astype(str).str.strip()

    sheet1_required = {
        "日期",
        "方案ID",
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

    sheet1_numeric = [
        "展现量",
        "点击量",
        "消费额",
        "上方位展现量",
        "上方首位展现量",
    ]
    for column in sheet1_numeric:
        sheet1[column] = pd.to_numeric(sheet1[column], errors="raise")
    sheet2["新注册数"] = pd.to_numeric(sheet2["新注册数"], errors="raise")

    return sheet1, sheet2


def build_daily_summary(
    sheet1: pd.DataFrame, sheet2: pd.DataFrame
) -> pd.DataFrame:
    """构造日期粒度指标，并增加星期和月份字段。"""
    daily = (
        sheet1.groupby("日期", as_index=False)
        .agg(
            总消费额=("消费额", "sum"),
            总展现量=("展现量", "sum"),
            总点击量=("点击量", "sum"),
            上方位展现量=("上方位展现量", "sum"),
            上方首位展现量=("上方首位展现量", "sum"),
        )
        .sort_values("日期")
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
        dates = daily.loc[daily["新注册数"].isna(), "日期"]
        missing_text = "、".join(dates.dt.strftime("%Y-%m-%d"))
        raise ValueError(f"以下投放日期缺少新注册数：{missing_text}")

    daily["星期几"] = pd.Categorical(
        daily["日期"].dt.dayofweek.map(dict(enumerate(WEEKDAYS))),
        categories=WEEKDAYS,
        ordered=True,
    )
    daily["月份"] = daily["日期"].dt.month
    return daily.reset_index(drop=True)


def build_weekday_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """计算周一至周日的描述性统计。"""
    summary = (
        daily.groupby("星期几", observed=False)
        .agg(
            样本天数=("日期", "size"),
            总消费额_均值=("总消费额", "mean"),
            总消费额_中位数=("总消费额", "median"),
            总点击量_均值=("总点击量", "mean"),
            总点击量_中位数=("总点击量", "median"),
            CTR_均值=("CTR", "mean"),
            CTR_中位数=("CTR", "median"),
            CPC_均值=("CPC", "mean"),
            CPC_中位数=("CPC", "median"),
            新注册数_均值=("新注册数", "mean"),
            新注册数_中位数=("新注册数", "median"),
        )
        .reindex(WEEKDAYS)
        .reset_index()
    )
    return summary


def save_boxplot(
    daily: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    filename: str,
    as_percent: bool = False,
) -> None:
    """按星期绘制单个指标的箱线图。"""
    scale = 100 if as_percent else 1
    values = [
        (daily.loc[daily["星期几"] == weekday, column] * scale).dropna().to_numpy()
        for weekday in WEEKDAYS
    ]

    figure, axis = plt.subplots(figsize=(10, 6))
    boxplot = axis.boxplot(
        values,
        tick_labels=WEEKDAYS,
        patch_artist=True,
        widths=0.62,
        medianprops={"color": "#B22222", "linewidth": 1.8},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
        flierprops={
            "marker": "o",
            "markersize": 3.5,
            "markerfacecolor": "#777777",
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
    )
    colors = ["#8EC9E6", "#8EC9E6", "#8EC9E6", "#8EC9E6", "#8EC9E6", "#F3C77B", "#F3C77B"]
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    axis.set_title(title)
    axis.set_xlabel("星期")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_weekday_boxplots(daily: pd.DataFrame) -> None:
    """生成四张星期效应箱线图。"""
    save_boxplot(
        daily,
        "总消费额",
        "星期与每日总消费分布",
        "每日总消费（元）",
        "q1_weekday_total_cost_boxplot.png",
    )
    save_boxplot(
        daily,
        "CTR",
        "星期与每日点击率分布",
        "CTR（%）",
        "q1_weekday_ctr_boxplot.png",
        as_percent=True,
    )
    save_boxplot(
        daily,
        "CPC",
        "星期与每日单次点击成本分布",
        "CPC（元/次）",
        "q1_weekday_cpc_boxplot.png",
    )
    save_boxplot(
        daily,
        "新注册数",
        "星期与每日新注册数分布",
        "新注册数",
        "q1_weekday_registrations_boxplot.png",
    )


def build_month_plan_summary(sheet1: pd.DataFrame) -> pd.DataFrame:
    """按月份和方案 ID 聚合投放总量及比率。"""
    data = sheet1.copy()
    data["月份"] = data["日期"].dt.month
    summary = (
        data.groupby(["月份", "方案ID"], as_index=False)
        .agg(
            投放天数=("日期", "nunique"),
            总消费=("消费额", "sum"),
            总点击=("点击量", "sum"),
            总展现=("展现量", "sum"),
        )
        .sort_values(["方案ID", "月份"])
    )
    summary["CTR"] = safe_ratio(summary["总点击"], summary["总展现"])
    summary["CPC"] = safe_ratio(summary["总消费"], summary["总点击"])
    return summary


def make_pivot(
    monthly: pd.DataFrame,
    value: str,
    plan_ids: list[int],
) -> pd.DataFrame:
    """将月度方案长表转换为固定 12 个月的宽表。"""
    return (
        monthly.pivot(index="方案ID", columns="月份", values=value)
        .reindex(index=plan_ids, columns=MONTHS)
    )


def save_heatmap(
    pivot: pd.DataFrame,
    title: str,
    colorbar_label: str,
    filename: str,
    value_format: str,
    as_percent: bool = False,
) -> None:
    """绘制带数值标注的月份—方案热力图，缺失组合显示为灰色。"""
    display_values = pivot.astype(float).to_numpy(copy=True)
    if as_percent:
        display_values *= 100
    masked_values = np.ma.masked_invalid(display_values)

    colormap = plt.get_cmap("YlOrRd").copy()
    colormap.set_bad(color="#E6E6E6")

    figure, axis = plt.subplots(figsize=(14, 5.8))
    image = axis.imshow(masked_values, aspect="auto", cmap=colormap)
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(colorbar_label)

    axis.set_xticks(range(len(MONTHS)), labels=[f"{month}月" for month in MONTHS])
    axis.set_yticks(
        range(len(pivot.index)),
        labels=[str(plan_id) for plan_id in pivot.index],
    )
    axis.set_xlabel("月份")
    axis.set_ylabel("方案ID")
    axis.set_title(title)

    finite_values = display_values[np.isfinite(display_values)]
    midpoint = (finite_values.min() + finite_values.max()) / 2 if finite_values.size else 0
    for row in range(display_values.shape[0]):
        for column in range(display_values.shape[1]):
            value = display_values[row, column]
            if not np.isfinite(value):
                text = "—"
                color = "#666666"
            else:
                text = format(value, value_format)
                if as_percent:
                    text += "%"
                color = "white" if value > midpoint else "black"
            axis.text(column, row, text, ha="center", va="center", fontsize=8, color=color)

    figure.tight_layout()
    figure.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_month_plan_heatmaps(
    monthly: pd.DataFrame,
    plan_ids: list[int],
) -> dict[str, pd.DataFrame]:
    """生成四张月份—方案热力图并返回对应宽表。"""
    pivots = {
        "总消费": make_pivot(monthly, "总消费", plan_ids),
        "总点击": make_pivot(monthly, "总点击", plan_ids),
        "CTR": make_pivot(monthly, "CTR", plan_ids),
        "CPC": make_pivot(monthly, "CPC", plan_ids),
    }
    save_heatmap(
        pivots["总消费"],
        "各方案月度总消费",
        "总消费（元）",
        "q1_month_plan_total_cost_heatmap.png",
        ",.0f",
    )
    save_heatmap(
        pivots["总点击"],
        "各方案月度总点击",
        "总点击量",
        "q1_month_plan_total_clicks_heatmap.png",
        ",.0f",
    )
    save_heatmap(
        pivots["CTR"],
        "各方案月度点击率（CTR）",
        "CTR（%）",
        "q1_month_plan_ctr_heatmap.png",
        ".1f",
        as_percent=True,
    )
    save_heatmap(
        pivots["CPC"],
        "各方案月度单次点击成本（CPC）",
        "CPC（元/次）",
        "q1_month_plan_cpc_heatmap.png",
        ".2f",
    )
    return pivots


def save_tables(
    weekday_summary: pd.DataFrame,
    monthly: pd.DataFrame,
    active_status: pd.DataFrame,
    active_days: pd.DataFrame,
    monthly_cost: pd.DataFrame,
) -> None:
    """保存星期效应与月份—方案统计结果。"""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    weekday_summary.to_csv(
        TABLE_DIR / "q1_weekday_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    monthly.to_csv(
        TABLE_DIR / "q1_month_plan_summary.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    active_status.to_csv(
        TABLE_DIR / "q1_plan_month_active.csv",
        encoding="utf-8-sig",
    )
    active_days.to_csv(
        TABLE_DIR / "q1_plan_month_active_days.csv",
        encoding="utf-8-sig",
    )
    monthly_cost.to_csv(
        TABLE_DIR / "q1_plan_month_cost.csv",
        encoding="utf-8-sig",
        float_format="%.2f",
    )


def print_requested_tables(
    weekday_summary: pd.DataFrame,
    active_days: pd.DataFrame,
    monthly_cost: pd.DataFrame,
) -> None:
    """仅打印用户指定的三张结果表。"""
    weekday_for_print = weekday_summary.copy()
    weekday_for_print["CTR_均值"] *= 100
    weekday_for_print["CTR_中位数"] *= 100

    print("星期效应汇总表（CTR 单位：%）")
    print(
        weekday_for_print.to_string(
            index=False,
            formatters={
                "总消费额_均值": "{:,.2f}".format,
                "总消费额_中位数": "{:,.2f}".format,
                "总点击量_均值": "{:,.2f}".format,
                "总点击量_中位数": "{:,.2f}".format,
                "CTR_均值": "{:.3f}".format,
                "CTR_中位数": "{:.3f}".format,
                "CPC_均值": "{:.3f}".format,
                "CPC_中位数": "{:.3f}".format,
                "新注册数_均值": "{:,.2f}".format,
                "新注册数_中位数": "{:,.2f}".format,
            },
        )
    )
    print("\n各方案各月投放天数表")
    print(active_days.to_string())
    print("\n各方案各月消费额表（元）")
    print(monthly_cost.to_string(float_format=lambda value: f"{value:,.2f}"))


def main() -> None:
    configure_chinese_font()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    sheet1, sheet2 = load_data(find_input_file())
    daily = build_daily_summary(sheet1, sheet2)
    weekday_summary = build_weekday_summary(daily)
    plot_weekday_boxplots(daily)

    monthly = build_month_plan_summary(sheet1)
    plan_ids = sorted(sheet1["方案ID"].unique().tolist())
    pivots = plot_month_plan_heatmaps(monthly, plan_ids)

    active_days = make_pivot(monthly, "投放天数", plan_ids).fillna(0).astype(int)
    active_status = active_days.gt(0).replace({True: "有投放", False: "无投放"})
    monthly_cost = pivots["总消费"].fillna(0.0)

    month_labels = {month: f"{month}月" for month in MONTHS}
    active_days = active_days.rename(columns=month_labels)
    active_status = active_status.rename(columns=month_labels)
    monthly_cost = monthly_cost.rename(columns=month_labels)
    active_days.index.name = "方案ID"
    active_status.index.name = "方案ID"
    monthly_cost.index.name = "方案ID"

    save_tables(
        weekday_summary,
        monthly,
        active_status,
        active_days,
        monthly_cost,
    )
    print_requested_tables(weekday_summary, active_days, monthly_cost)


if __name__ == "__main__":
    main()

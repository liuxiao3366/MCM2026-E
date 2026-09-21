"""问题一：广告位置、出价结果代理与预算配置合理性分析。

数据来源：附件1.xlsx 的 Sheet1，以及已生成的关键词推广单元汇总表。
本脚本只进行描述统计、非参数配对检验、相关分析和可视化：

1. 比较上方位与非上方位的 CTR、CPC；
2. 按方案和推广单元先求和、再计算整体位置指标；
3. 分析广告位置比例与整体 CTR、CPC 的关联；
4. 用实际消费占比描述预算配置，用 CPC 和广告位置描述出价结果代理。

附件不含真实竞价金额，因而不估计竞价；附件也不含首位点击量和首位消费额，
因而不计算首位 CTR 或首位 CPC。所有结果仅表示统计关联，不表示因果关系。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.set_loglevel("error")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CANDIDATES = [
    PROJECT_ROOT / "附件1.xlsx",
    PROJECT_ROOT / "data" / "raw" / "附件1.xlsx",
    PROJECT_ROOT / "data" / "raw" / "Attachment" / "Attachment1.xlsx",
]
KEYWORD_UNIT_CSV = (
    PROJECT_ROOT / "output" / "tables" / "q1_keyword_unit_summary.csv"
)
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"

PAIRED_SUMMARY_CSV = TABLE_DIR / "q1_position_paired_summary.csv"
PAIRED_TESTS_CSV = TABLE_DIR / "q1_position_paired_tests.csv"
CORRELATIONS_CSV = TABLE_DIR / "q1_position_correlations.csv"
PLAN_SUMMARY_CSV = TABLE_DIR / "q1_position_plan_summary.csv"
UNIT_SUMMARY_CSV = TABLE_DIR / "q1_position_unit_summary.csv"
BUDGET_EFFICIENCY_CSV = TABLE_DIR / "q1_budget_efficiency_by_unit.csv"

POSITION_FIGURES = {
    ("上方位展现率", "CTR"): FIGURE_DIR / "q1_upper_impression_rate_vs_ctr.png",
    ("上方位展现率", "CPC"): FIGURE_DIR / "q1_upper_impression_rate_vs_cpc.png",
    ("首位展现率", "CTR"): FIGURE_DIR / "q1_first_impression_rate_vs_ctr.png",
    ("首位展现率", "CPC"): FIGURE_DIR / "q1_first_impression_rate_vs_cpc.png",
}
BUDGET_SHARE_FIGURE = FIGURE_DIR / "q1_budget_share_vs_click_share.png"
BUDGET_RATIO_FIGURE = FIGURE_DIR / "q1_budget_share_vs_click_budget_ratio.png"

REQUIRED_COLUMNS = [
    "日期",
    "方案ID",
    "推广单元ID",
    "展现量",
    "点击量",
    "消费额",
    "上方位展现量",
    "上方首位展现量",
    "上方位点击量",
    "上方位消费额",
]
NUMERIC_COLUMNS = [
    "展现量",
    "点击量",
    "消费额",
    "上方位展现量",
    "上方首位展现量",
    "上方位点击量",
    "上方位消费额",
]
KEYWORD_QUALITY_COLUMNS = [
    "zero_effect比例",
    "中位数浏览深度",
    "中位数跳出率",
    "中位数平均访问时长_秒",
]


def find_input_file() -> Path:
    """按常用位置查找附件1，兼容当前项目的英文文件名。"""
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in INPUT_CANDIDATES)
    raise FileNotFoundError(f"未找到附件1.xlsx，已检查：\n{searched}")


def normalize_identifier(value: object) -> str:
    """稳定转换方案和推广单元编码，避免整数编码出现 .0。"""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """逐项安全除法；分母为0或缺失时返回 NaN。"""
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid = numerator.notna() & denominator.notna() & denominator.ne(0)
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def load_sheet1() -> pd.DataFrame:
    """读取、校验 Sheet1 并计算记录级位置、CTR 和 CPC 指标。"""
    data = pd.read_excel(find_input_file(), sheet_name="Sheet1", dtype=object)
    data.columns = data.columns.map(lambda column: str(column).strip())
    missing = sorted(set(REQUIRED_COLUMNS).difference(data.columns))
    if missing:
        raise ValueError(f"Sheet1 缺少字段：{'、'.join(missing)}")
    data = data[REQUIRED_COLUMNS].copy()

    data["日期"] = pd.to_datetime(data["日期"], errors="coerce")
    if data["日期"].isna().any():
        raise ValueError("Sheet1 存在无法解析的日期。")
    for column in ["方案ID", "推广单元ID"]:
        data[column] = data[column].map(normalize_identifier).astype("string")
        if data[column].eq("").any():
            raise ValueError(f"Sheet1 的 {column} 存在空编码。")
    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid = data[column].notna() & converted.isna()
        if invalid.any():
            examples = data.loc[invalid, column].astype(str).unique()[:5]
            raise ValueError(f"Sheet1 的 {column} 存在非数值：{examples.tolist()}")
        data[column] = converted.astype(float)

    if data[NUMERIC_COLUMNS].lt(0).any().any():
        raise ValueError("Sheet1 的展现、点击或消费字段存在负值。")
    consistency_checks = {
        "上方位展现量超过总展现量": data["上方位展现量"] > data["展现量"],
        "上方首位展现量超过上方位展现量": (
            data["上方首位展现量"] > data["上方位展现量"]
        ),
        "上方位点击量超过总点击量": data["上方位点击量"] > data["点击量"],
        "上方位消费额超过总消费额": data["上方位消费额"] > data["消费额"] + 1e-9,
    }
    failed = {name: int(mask.sum()) for name, mask in consistency_checks.items() if mask.any()}
    if failed:
        raise ValueError(f"Sheet1 位置字段不满足总量约束：{failed}")

    data["非上方位展现量"] = data["展现量"] - data["上方位展现量"]
    data["非上方位点击量"] = data["点击量"] - data["上方位点击量"]
    data["非上方位消费额"] = data["消费额"] - data["上方位消费额"]

    data["上方位CTR"] = safe_divide(data["上方位点击量"], data["上方位展现量"])
    data["非上方位CTR"] = safe_divide(
        data["非上方位点击量"], data["非上方位展现量"]
    )
    data["上方位CPC"] = safe_divide(data["上方位消费额"], data["上方位点击量"])
    data["非上方位CPC"] = safe_divide(
        data["非上方位消费额"], data["非上方位点击量"]
    )
    data["上方位展现率"] = safe_divide(data["上方位展现量"], data["展现量"])
    data["首位展现率"] = safe_divide(data["上方首位展现量"], data["展现量"])
    data["CTR"] = safe_divide(data["点击量"], data["展现量"])
    data["CPC"] = safe_divide(data["消费额"], data["点击量"])
    return data.sort_values(["日期", "方案ID", "推广单元ID"]).reset_index(drop=True)


def safe_mean(values: pd.Series) -> float:
    """存在有效值时计算均值。"""
    valid = values.dropna()
    return float(valid.mean()) if len(valid) else np.nan


def safe_median(values: pd.Series) -> float:
    """存在有效值时计算中位数。"""
    valid = values.dropna()
    return float(valid.median()) if len(valid) else np.nan


def paired_rank_biserial(differences: np.ndarray) -> float:
    """计算配对秩二分相关；正值表示上方位指标整体较高。"""
    nonzero = differences[np.isfinite(differences) & (differences != 0)]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def build_paired_results(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """汇总位置配对差异，并进行 Wilcoxon 配对符号秩检验。"""
    comparisons = [
        ("CTR", "上方位CTR", "非上方位CTR"),
        ("CPC", "上方位CPC", "非上方位CPC"),
    ]
    summary_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []

    for metric, upper_column, nonupper_column in comparisons:
        pairs = data[[upper_column, nonupper_column]].dropna().copy()
        upper = pairs[upper_column]
        nonupper = pairs[nonupper_column]
        differences = upper - nonupper
        ratios = safe_divide(upper, nonupper)

        summary_rows.append(
            {
                "比较指标": metric,
                "有效配对数": len(pairs),
                "比值有效配对数": int(ratios.notna().sum()),
                "上方位均值": safe_mean(upper),
                "上方位中位数": safe_median(upper),
                "非上方位均值": safe_mean(nonupper),
                "非上方位中位数": safe_median(nonupper),
                "配对差值定义": "上方位-非上方位",
                "配对差值均值": safe_mean(differences),
                "配对差值中位数": safe_median(differences),
                "配对比值定义": "上方位/非上方位",
                "配对比值均值": safe_mean(ratios),
                "配对比值中位数": safe_median(ratios),
            }
        )

        nonzero_count = int(differences.ne(0).sum())
        if nonzero_count:
            result = stats.wilcoxon(
                upper,
                nonupper,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            statistic = float(result.statistic)
            p_value = float(result.pvalue)
        else:
            statistic = 0.0
            p_value = 1.0
        test_rows.append(
            {
                "比较指标": metric,
                "检验方法": "Wilcoxon配对符号秩检验",
                "备择假设": "双侧",
                "有效配对数": len(pairs),
                "非零差值数": nonzero_count,
                "统计量名称": "W",
                "统计量": statistic,
                "p值": p_value,
                "效应量名称": "paired_rank_biserial_r",
                "效应量": paired_rank_biserial(differences.to_numpy(dtype=float)),
                "效应量方向": "正值表示上方位指标整体高于非上方位",
                "解释限制": "位置与表现的统计关联，不表示因果关系",
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(test_rows)


def aggregate_position(
    data: pd.DataFrame,
    group_columns: list[str],
    level_name: str,
) -> pd.DataFrame:
    """按指定层级先汇总原始量，再计算整体比率。"""
    sums = [
        "展现量",
        "点击量",
        "消费额",
        "上方位展现量",
        "上方首位展现量",
        "上方位点击量",
        "上方位消费额",
        "非上方位展现量",
        "非上方位点击量",
        "非上方位消费额",
    ]
    grouped = data.groupby(group_columns, observed=True, sort=True)
    result = grouped[sums].sum().reset_index()
    counts = grouped.size().rename("Sheet1记录数").reset_index()
    active_days = grouped["日期"].nunique().rename("有投放记录天数").reset_index()
    result = result.merge(counts, on=group_columns, validate="one_to_one")
    result = result.merge(active_days, on=group_columns, validate="one_to_one")
    result.insert(0, "聚合层级", level_name)

    result["上方位CTR"] = safe_divide(result["上方位点击量"], result["上方位展现量"])
    result["非上方位CTR"] = safe_divide(
        result["非上方位点击量"], result["非上方位展现量"]
    )
    result["上方位CPC"] = safe_divide(result["上方位消费额"], result["上方位点击量"])
    result["非上方位CPC"] = safe_divide(
        result["非上方位消费额"], result["非上方位点击量"]
    )
    result["上方位展现率"] = safe_divide(result["上方位展现量"], result["展现量"])
    result["首位展现率"] = safe_divide(result["上方首位展现量"], result["展现量"])
    result["总CTR"] = safe_divide(result["点击量"], result["展现量"])
    result["总CPC"] = safe_divide(result["消费额"], result["点击量"])
    return result


POSITION_CORRELATION_SPECS = [
    ("上方位展现率", "CTR"),
    ("上方位展现率", "CPC"),
    ("首位展现率", "CTR"),
    ("首位展现率", "CPC"),
]


def build_position_correlations(data: pd.DataFrame) -> pd.DataFrame:
    """计算记录级广告位置比例与总体 CTR、CPC 的 Spearman 相关。"""
    rows: list[dict[str, object]] = []
    for first, second in POSITION_CORRELATION_SPECS:
        pairs = data[[first, second]].dropna()
        if len(pairs) < 3 or pairs[first].nunique() < 2 or pairs[second].nunique() < 2:
            coefficient, p_value = np.nan, np.nan
        else:
            result = stats.spearmanr(pairs[first], pairs[second])
            coefficient, p_value = float(result.statistic), float(result.pvalue)
        rows.append(
            {
                "分析层级": "Sheet1记录（日期×方案ID×推广单元ID）",
                "变量1": first,
                "变量2": second,
                "方法": "Spearman",
                "样本量": len(pairs),
                "rho": coefficient,
                "p值": p_value,
                "p值报告": format_p_value(p_value),
                "解释限制": "统计关联，不表示位置造成表现变化",
            }
        )
    return pd.DataFrame(rows)


def load_keyword_unit_summary() -> pd.DataFrame:
    """读取关键词推广单元质量指标并校验一对一连接键。"""
    if not KEYWORD_UNIT_CSV.exists():
        raise FileNotFoundError(
            f"未找到 {KEYWORD_UNIT_CSV}，请先运行 05_q1_keyword_management.py。"
        )
    keyword = pd.read_csv(
        KEYWORD_UNIT_CSV,
        encoding="utf-8-sig",
        dtype={"方案ID": "string", "推广单元ID": "string"},
    )
    required = {"方案ID", "推广单元ID", *KEYWORD_QUALITY_COLUMNS}
    missing = sorted(required.difference(keyword.columns))
    if missing:
        raise ValueError(f"关键词单元汇总缺少字段：{'、'.join(missing)}")
    if keyword.duplicated(["方案ID", "推广单元ID"]).any():
        raise ValueError("关键词单元汇总的方案ID与推广单元ID组合不唯一。")
    return keyword[["方案ID", "推广单元ID", *KEYWORD_QUALITY_COLUMNS]].copy()


def build_budget_efficiency(
    unit_summary: pd.DataFrame,
    keyword_summary: pd.DataFrame,
) -> pd.DataFrame:
    """构造12个推广单元的预算、点击贡献和关键词质量联合表。"""
    budget = unit_summary.copy()
    total_cost = float(budget["消费额"].sum())
    total_clicks = float(budget["点击量"].sum())
    budget["预算占比"] = budget["消费额"] / total_cost if total_cost else np.nan
    budget["点击贡献占比"] = budget["点击量"] / total_clicks if total_clicks else np.nan
    budget["点击贡献预算比"] = safe_divide(budget["点击贡献占比"], budget["预算占比"])

    budget = budget.merge(
        keyword_summary,
        on=["方案ID", "推广单元ID"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not budget["_merge"].eq("both").all():
        missing_units = budget.loc[
            budget["_merge"].ne("both"), ["方案ID", "推广单元ID"]
        ].to_dict("records")
        raise ValueError(f"以下Sheet1推广单元未匹配关键词质量表：{missing_units}")
    budget = budget.drop(columns="_merge")

    output_columns = [
        "方案ID",
        "推广单元ID",
        "Sheet1记录数",
        "有投放记录天数",
        "消费额",
        "点击量",
        "展现量",
        "预算占比",
        "点击贡献占比",
        "点击贡献预算比",
        "总CTR",
        "总CPC",
        "上方位展现率",
        "首位展现率",
        *KEYWORD_QUALITY_COLUMNS,
    ]
    return budget[output_columns].sort_values(
        ["预算占比", "方案ID", "推广单元ID"],
        ascending=[False, True, True],
        kind="mergesort",
    )


def configure_chinese_font() -> None:
    """配置常见中文字体回退。"""
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "PingFang SC",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def format_p_value(value: float) -> str:
    """格式化图题中的 p 值，避免浮点下溢显示为精确的0。"""
    if value == 0:
        return "<1e-300"
    return f"{value:.3g}"


def save_position_scatterplots(
    data: pd.DataFrame,
    correlations: pd.DataFrame,
) -> None:
    """分别保存四张广告位置比例与 CTR/CPC 的散点图。"""
    lookup = correlations.set_index(["变量1", "变量2"])
    for first, second in POSITION_CORRELATION_SPECS:
        pairs = data[[first, second]].dropna()
        figure, axis = plt.subplots(figsize=(8, 6))
        x = pairs[first] * 100
        y = pairs[second] * 100 if second == "CTR" else pairs[second]
        axis.scatter(x, y, s=15, alpha=0.28, color="#4C72B0", edgecolors="none")
        row = lookup.loc[(first, second)]
        axis.set_title(
            f"{first}与{second}的记录级关系\n"
            f"Spearman ρ={row['rho']:.3f}, "
            f"p={format_p_value(row['p值'])}, n={int(row['样本量'])}"
        )
        axis.set_xlabel(f"{first}（%）")
        axis.set_ylabel("CTR（%）" if second == "CTR" else "CPC（元/次）")
        axis.grid(linestyle="--", linewidth=0.6, alpha=0.35)
        figure.tight_layout()
        figure.savefig(POSITION_FIGURES[(first, second)], dpi=300, bbox_inches="tight")
        plt.close(figure)


def annotate_units(axis: plt.Axes, budget: pd.DataFrame, x: str, y: str) -> None:
    """标注预算占比至少1%的主要单元，避免小占比单元标签重叠。"""
    for row in budget.itertuples(index=False):
        x_value = getattr(row, x)
        y_value = getattr(row, y)
        if row.预算占比 >= 0.01 and pd.notna(x_value) and pd.notna(y_value):
            axis.annotate(
                str(row.推广单元ID),
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
                alpha=0.85,
            )


def save_budget_scatterplots(budget: pd.DataFrame) -> None:
    """分别保存预算占比—点击贡献和预算占比—贡献预算比散点图。"""
    figure, axis = plt.subplots(figsize=(8, 6.5))
    axis.scatter(
        budget["预算占比"],
        budget["点击贡献占比"],
        s=55,
        alpha=0.80,
        color="#4C72B0",
        edgecolors="white",
        linewidths=0.6,
    )
    limit = max(float(budget["预算占比"].max()), float(budget["点击贡献占比"].max())) * 1.08
    axis.plot([0, limit], [0, limit], linestyle="--", color="#C44E52", label="y=x")
    annotate_units(axis, budget, "预算占比", "点击贡献占比")
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axis.set_xlabel("预算占比（实际消费占比）")
    axis.set_ylabel("点击贡献占比")
    axis.set_title("推广单元预算占比与点击贡献占比")
    axis.grid(linestyle="--", linewidth=0.6, alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(BUDGET_SHARE_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6.5))
    axis.scatter(
        budget["预算占比"],
        budget["点击贡献预算比"],
        s=55,
        alpha=0.80,
        color="#55A868",
        edgecolors="white",
        linewidths=0.6,
    )
    annotate_units(axis, budget, "预算占比", "点击贡献预算比")
    axis.axhline(1, linestyle="--", color="#C44E52", label="点击贡献占比=预算占比")
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    axis.set_xlabel("预算占比（实际消费占比）")
    axis.set_ylabel("点击贡献预算比")
    axis.set_title("推广单元预算占比与点击贡献预算比")
    axis.grid(linestyle="--", linewidth=0.6, alpha=0.35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(BUDGET_RATIO_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_csv(dataframe: pd.DataFrame, path: Path) -> None:
    """统一保存 UTF-8 BOM CSV。"""
    dataframe.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )


def print_core_results(
    paired_summary: pd.DataFrame,
    paired_tests: pd.DataFrame,
    correlations: pd.DataFrame,
    budget: pd.DataFrame,
) -> None:
    """终端只打印核心汇总和检验结果。"""
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print("位置表现配对汇总")
        print(paired_summary.to_string(index=False))

        print("\nWilcoxon配对检验")
        print(
            paired_tests[
                [
                    "比较指标",
                    "有效配对数",
                    "非零差值数",
                    "统计量",
                    "p值",
                    "效应量",
                ]
            ].to_string(index=False)
        )

        print("\n位置策略相关性")
        print(
            correlations[
                ["变量1", "变量2", "样本量", "rho", "p值报告"]
            ].to_string(index=False)
        )

        print("\n推广单元预算效率核心汇总")
        print(
            budget[
                [
                    "方案ID",
                    "推广单元ID",
                    "预算占比",
                    "点击贡献占比",
                    "点击贡献预算比",
                    "总CTR",
                    "总CPC",
                    "zero_effect比例",
                    "中位数浏览深度",
                    "中位数跳出率",
                    "中位数平均访问时长_秒",
                ]
            ].to_string(index=False)
        )
        print("注：CPC和广告位置仅作为出价结果代理，不代表真实竞价金额；统计关联不表示因果关系。")


def main() -> None:
    data = load_sheet1()
    paired_summary, paired_tests = build_paired_results(data)
    plan_summary = aggregate_position(data, ["方案ID"], "方案ID")
    unit_summary = aggregate_position(data, ["方案ID", "推广单元ID"], "推广单元ID")
    correlations = build_position_correlations(data)
    keyword_summary = load_keyword_unit_summary()
    budget = build_budget_efficiency(unit_summary, keyword_summary)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(paired_summary, PAIRED_SUMMARY_CSV)
    save_csv(paired_tests, PAIRED_TESTS_CSV)
    save_csv(correlations, CORRELATIONS_CSV)
    save_csv(plan_summary, PLAN_SUMMARY_CSV)
    save_csv(unit_summary, UNIT_SUMMARY_CSV)
    save_csv(budget, BUDGET_EFFICIENCY_CSV)

    configure_chinese_font()
    save_position_scatterplots(data, correlations)
    save_budget_scatterplots(budget)
    print_core_results(paired_summary, paired_tests, correlations, budget)


if __name__ == "__main__":
    main()

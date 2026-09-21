"""问题一：用局部星期匹配和控制变量模型复核假日关联。

本脚本不重复已有描述统计和全局 Kruskal-Wallis 检验，包含：
1. 五个主要假期的前后各 14 天同星期局部匹配；
2. Wilcoxon 配对检验、Holm 多重比较校正和配对秩二分效应量；
3. 控制月份与星期后的假日类型 OLS 模型（Newey-West HAC 标准误）；
4. 元旦的局部描述，不对单日样本作正式显著性推断。

所有结果只用于描述时间关联结构，不用于识别因果效应。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "daily_summary.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "tables"

MATCHES_CSV = OUTPUT_DIR / "q1_holiday_local_matches.csv"
LOCAL_COMPARISONS_CSV = OUTPUT_DIR / "q1_holiday_local_comparisons.csv"
MODEL_COEFFICIENTS_CSV = OUTPUT_DIR / "q1_holiday_control_model_coefficients.csv"
MODEL_FIT_CSV = OUTPUT_DIR / "q1_holiday_control_model_fit.csv"

ALPHA = 0.05
LOCAL_WINDOW_DAYS = 14
HAC_MAX_LAG = 7
WEEKDAY_NAMES = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
METRICS = ["总消费额", "总点击量", "CTR", "CPC", "新注册数"]

# 与国务院办公厅 2025 年放假通知一致。
HOLIDAY_RANGES = {
    "元旦": ("2025-01-01", "2025-01-01"),
    "春节": ("2025-01-28", "2025-02-04"),
    "清明节": ("2025-04-04", "2025-04-06"),
    "劳动节": ("2025-05-01", "2025-05-05"),
    "端午节": ("2025-05-31", "2025-06-02"),
    "国庆/中秋假期": ("2025-10-01", "2025-10-08"),
}
FORMAL_HOLIDAYS = ["春节", "清明节", "劳动节", "端午节", "国庆/中秋假期"]
ALL_HOLIDAYS = ["元旦", *FORMAL_HOLIDAYS]


def load_daily_summary() -> pd.DataFrame:
    """读取并校验连续的日级汇总表。"""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"未找到日级汇总文件：{INPUT_CSV}")

    daily = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    required = {"日期", *METRICS}
    missing = required.difference(daily.columns)
    if missing:
        missing_text = "、".join(sorted(missing))
        raise ValueError(f"daily_summary.csv 缺少字段：{missing_text}")

    daily["日期"] = pd.to_datetime(daily["日期"], errors="coerce")
    if daily["日期"].isna().any():
        raise ValueError("daily_summary.csv 中存在无法解析的日期。")
    if daily["日期"].duplicated().any():
        raise ValueError("daily_summary.csv 中存在重复日期。")
    for metric in METRICS:
        daily[metric] = pd.to_numeric(daily[metric], errors="raise")

    daily = daily.sort_values("日期").reset_index(drop=True)
    if not daily["日期"].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError("日期序列不连续，不能按自然日构造局部窗口。")
    return daily


def build_holiday_lookup() -> dict[pd.Timestamp, str]:
    """展开官方假期区间。"""
    lookup: dict[pd.Timestamp, str] = {}
    for holiday, (start, end) in HOLIDAY_RANGES.items():
        for date in pd.date_range(start, end, freq="D"):
            if date in lookup:
                raise ValueError(f"假期日期重复：{date:%Y-%m-%d}")
            lookup[date] = holiday
    return lookup


def add_holiday_type(daily: pd.DataFrame) -> pd.DataFrame:
    """增加用于控制模型的假日类型；全部非节假日共用基准类别。"""
    lookup = build_holiday_lookup()
    categories = ["非节假日", *ALL_HOLIDAYS]
    result = daily.copy()
    result["假日类型"] = pd.Categorical(
        result["日期"].map(lambda date: lookup.get(date, "非节假日")),
        categories=categories,
        ordered=True,
    )
    return result


def choose_same_weekday_matches(
    holiday_dates: pd.DatetimeIndex,
    candidates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, pd.Timestamp]:
    """在一个局部窗口内不放回地匹配最近同星期日期。"""
    unused = set(candidates)
    matches: dict[pd.Timestamp, pd.Timestamp] = {}
    for holiday_date in holiday_dates:
        same_weekday = [date for date in unused if date.dayofweek == holiday_date.dayofweek]
        if not same_weekday:
            raise ValueError(
                f"{holiday_date:%Y-%m-%d} 在局部窗口中没有可用的同星期对照。"
            )
        matched = min(
            same_weekday,
            key=lambda date: (abs((date - holiday_date).days), date),
        )
        matches[holiday_date] = matched
        unused.remove(matched)
    return matches


def build_formal_holiday_matches(daily: pd.DataFrame) -> pd.DataFrame:
    """为五个主要假期的每一天匹配一个前置和一个后置同星期对照。"""
    holiday_lookup = build_holiday_lookup()
    all_holiday_dates = set(holiday_lookup)
    available_dates = set(daily["日期"])
    rows: list[dict[str, object]] = []

    for holiday in FORMAL_HOLIDAYS:
        start_text, end_text = HOLIDAY_RANGES[holiday]
        start, end = pd.Timestamp(start_text), pd.Timestamp(end_text)
        holiday_dates = pd.date_range(start, end, freq="D")
        before_candidates = pd.DatetimeIndex(
            [
                date
                for date in pd.date_range(
                    start - pd.Timedelta(days=LOCAL_WINDOW_DAYS),
                    start - pd.Timedelta(days=1),
                    freq="D",
                )
                if date in available_dates and date not in all_holiday_dates
            ]
        )
        after_candidates = pd.DatetimeIndex(
            [
                date
                for date in pd.date_range(
                    end + pd.Timedelta(days=1),
                    end + pd.Timedelta(days=LOCAL_WINDOW_DAYS),
                    freq="D",
                )
                if date in available_dates and date not in all_holiday_dates
            ]
        )
        before_matches = choose_same_weekday_matches(holiday_dates, before_candidates)
        after_matches = choose_same_weekday_matches(holiday_dates, after_candidates)

        for holiday_date in holiday_dates:
            before_date = before_matches[holiday_date]
            after_date = after_matches[holiday_date]
            rows.append(
                {
                    "假期": holiday,
                    "假期日期": holiday_date,
                    "星期": WEEKDAY_NAMES[holiday_date.dayofweek],
                    "前置对照日期": before_date,
                    "前置间隔天数": int((holiday_date - before_date).days),
                    "后置对照日期": after_date,
                    "后置间隔天数": int((after_date - holiday_date).days),
                    "匹配规则": (
                        "候选窗口按整个假期区间边界定义：开始日前1—14天、"
                        "结束日后1—14天的非节假日；每个假日按同星期在两侧"
                        "分别不放回最近匹配；故相对单个假日的间隔可超过14天"
                    ),
                }
            )

    matches = pd.DataFrame(rows)
    value_lookup = daily.set_index("日期")
    for metric in METRICS:
        matches[f"{metric}_假期值"] = matches["假期日期"].map(value_lookup[metric])
        matches[f"{metric}_前置对照值"] = matches["前置对照日期"].map(value_lookup[metric])
        matches[f"{metric}_后置对照值"] = matches["后置对照日期"].map(value_lookup[metric])
        matches[f"{metric}_配对对照均值"] = matches[
            [f"{metric}_前置对照值", f"{metric}_后置对照值"]
        ].mean(axis=1)
    return matches


def paired_rank_biserial(differences: np.ndarray) -> float:
    """计算 Wilcoxon 配对检验的秩二分相关，正值表示假期偏高。"""
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def holm_adjust(p_values: list[float]) -> np.ndarray:
    """使用 Holm step-down 方法校正 p 值。"""
    p_array = np.asarray(p_values, dtype=float)
    order = np.argsort(p_array)
    adjusted_sorted = np.empty(len(p_array), dtype=float)
    running_max = 0.0
    for rank, original_index in enumerate(order):
        adjusted_value = (len(p_array) - rank) * p_array[original_index]
        running_max = max(running_max, adjusted_value)
        adjusted_sorted[rank] = min(running_max, 1.0)
    adjusted = np.empty(len(p_array), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def build_local_comparisons(
    daily: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """计算局部中位数、相对变化、Wilcoxon 检验和配对效应量。"""
    rows: list[dict[str, object]] = []

    for metric in METRICS:
        for holiday in FORMAL_HOLIDAYS:
            holiday_matches = matches.loc[matches["假期"] == holiday]
            holiday_values = holiday_matches[f"{metric}_假期值"].to_numpy(dtype=float)
            control_values = holiday_matches[f"{metric}_配对对照均值"].to_numpy(dtype=float)
            differences = holiday_values - control_values
            test = stats.wilcoxon(
                holiday_values,
                control_values,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            holiday_median = float(np.median(holiday_values))
            control_median = float(np.median(control_values))
            relative_change = (
                (holiday_median - control_median) / control_median
                if control_median != 0
                else np.nan
            )
            rows.append(
                {
                    "指标": metric,
                    "假期": holiday,
                    "推断类型": "正式配对检验",
                    "假期样本数": len(holiday_values),
                    "局部配对数": len(control_values),
                    "假期中位数": holiday_median,
                    "局部对照中位数": control_median,
                    "中位数相对变化率": relative_change,
                    "检验方法": "Wilcoxon配对符号秩检验",
                    "统计量名称": "W",
                    "统计量": float(test.statistic),
                    "原始p值": float(test.pvalue),
                    "校正方法": "Holm（同一指标的5个假期）",
                    "校正p值": np.nan,
                    "校正后显著": False,
                    "效应量名称": "paired_rank_biserial_r",
                    "效应量": paired_rank_biserial(differences),
                    "说明": "正效应量表示假期值整体高于同星期局部对照",
                }
            )

    comparisons = pd.DataFrame(rows)
    for metric in METRICS:
        mask = comparisons["指标"] == metric
        adjusted = holm_adjust(comparisons.loc[mask, "原始p值"].tolist())
        comparisons.loc[mask, "校正p值"] = adjusted
        comparisons.loc[mask, "校正后显著"] = adjusted < ALPHA

    # 元旦仅作描述：受数据起点限制，使用其后 14 天中的非节假日作为局部参照。
    holiday_lookup = build_holiday_lookup()
    new_year_date = pd.Timestamp("2025-01-01")
    new_year_controls = daily.loc[
        daily["日期"].between("2025-01-02", "2025-01-15")
        & ~daily["日期"].isin(holiday_lookup),
    ]
    descriptive_rows: list[dict[str, object]] = []
    for metric in METRICS:
        holiday_value = float(
            daily.loc[daily["日期"] == new_year_date, metric].iloc[0]
        )
        control_median = float(new_year_controls[metric].median())
        relative_change = (
            (holiday_value - control_median) / control_median
            if control_median != 0
            else np.nan
        )
        descriptive_rows.append(
            {
                "指标": metric,
                "假期": "元旦",
                "推断类型": "仅描述",
                "假期样本数": 1,
                "局部配对数": len(new_year_controls),
                "假期中位数": holiday_value,
                "局部对照中位数": control_median,
                "中位数相对变化率": relative_change,
                "检验方法": "未检验",
                "统计量名称": "",
                "统计量": np.nan,
                "原始p值": np.nan,
                "校正方法": "不适用",
                "校正p值": np.nan,
                "校正后显著": False,
                "效应量名称": "未计算",
                "效应量": np.nan,
                "说明": "元旦仅1天；局部参照为数据范围内其后14天的非节假日",
            }
        )
    return pd.concat([comparisons, pd.DataFrame(descriptive_rows)], ignore_index=True)


def build_control_design(
    daily: pd.DataFrame,
    outcome: str,
) -> tuple[pd.Series, pd.DataFrame, list[str], str]:
    """构造假日类型、月份和星期控制变量设计矩阵。"""
    holiday_dummies = pd.get_dummies(
        daily["假日类型"],
        prefix="假日",
        drop_first=True,
        dtype=float,
    )
    month = pd.Categorical(daily["日期"].dt.month, categories=range(1, 13), ordered=True)
    month_dummies = pd.get_dummies(
        month,
        prefix="月份",
        drop_first=True,
        dtype=float,
    )
    weekday = pd.Categorical(daily["日期"].dt.dayofweek, categories=range(7), ordered=True)
    weekday_dummies = pd.get_dummies(
        weekday,
        prefix="星期",
        drop_first=True,
        dtype=float,
    )

    design_parts: list[pd.DataFrame | pd.Series] = []
    if outcome == "新注册数":
        clicks_scaled = (daily["总点击量"] / 1000.0).rename("同日总点击量_千次")
        design_parts.append(clicks_scaled)
    design_parts.extend([holiday_dummies, month_dummies, weekday_dummies])
    design = pd.concat(design_parts, axis=1)
    design = sm.add_constant(design, has_constant="add")

    if outcome == "CTR":
        response = daily[outcome] * 100
        outcome_unit = "百分点"
    elif outcome == "CPC":
        response = daily[outcome]
        outcome_unit = "元/次"
    elif outcome == "总消费额":
        response = daily[outcome]
        outcome_unit = "元"
    elif outcome == "总点击量":
        response = daily[outcome]
        outcome_unit = "次"
    else:
        response = daily[outcome]
        outcome_unit = "人"

    holiday_columns = [column for column in holiday_dummies.columns]
    return response.astype(float), design.astype(float), holiday_columns, outcome_unit


def fit_control_model(
    daily: pd.DataFrame,
    outcome: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """拟合一个控制月份、星期的假日关联模型。"""
    response, design, holiday_columns, outcome_unit = build_control_design(daily, outcome)
    fitted = sm.OLS(response, design).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_MAX_LAG},
        use_t=True,
    )
    confidence_intervals = fitted.conf_int(alpha=ALPHA)
    sample_counts = daily["假日类型"].value_counts()

    coefficient_rows: list[dict[str, object]] = []
    for column in holiday_columns:
        holiday = column.removeprefix("假日_")
        is_new_year = holiday == "元旦"
        coefficient_rows.append(
            {
                "模型": f"{outcome}~假日类型+月份+星期"
                if outcome != "新注册数"
                else "新注册数~同日总点击+假日类型+月份+星期",
                "因变量": outcome,
                "因变量单位": outcome_unit,
                "假日变量": holiday,
                "参考组": "非节假日",
                "假日样本天数": int(sample_counts[holiday]),
                "系数": float(fitted.params[column]),
                "HAC标准误": np.nan if is_new_year else float(fitted.bse[column]),
                "t值": np.nan if is_new_year else float(fitted.tvalues[column]),
                "p值": np.nan if is_new_year else float(fitted.pvalues[column]),
                "95%置信区间下限": (
                    np.nan if is_new_year else float(confidence_intervals.loc[column, 0])
                ),
                "95%置信区间上限": (
                    np.nan if is_new_year else float(confidence_intervals.loc[column, 1])
                ),
                "协方差估计": f"Newey-West HAC(maxlags={HAC_MAX_LAG})",
                "推断限制": (
                    "单日样本：仅保留系数作描述，不报告标准误、置信区间和p值"
                    if is_new_year
                    else ""
                ),
            }
        )

    # 元旦哑变量仍留在设计矩阵中，以免该日进入非节假日参考组；
    # 但单日样本不参与正式的联合显著性检验。
    formal_holiday_columns = [
        column for column in holiday_columns if column != "假日_元旦"
    ]
    parameter_names = list(fitted.params.index)
    restriction = np.zeros((len(formal_holiday_columns), len(parameter_names)))
    for row, column in enumerate(formal_holiday_columns):
        restriction[row, parameter_names.index(column)] = 1
    joint_test = fitted.wald_test(restriction, use_f=False, scalar=True)

    click_coefficient = np.nan
    click_p_value = np.nan
    click_ci_low = np.nan
    click_ci_high = np.nan
    if outcome == "新注册数":
        click_name = "同日总点击量_千次"
        click_coefficient = float(fitted.params[click_name])
        click_p_value = float(fitted.pvalues[click_name])
        click_ci_low = float(confidence_intervals.loc[click_name, 0])
        click_ci_high = float(confidence_intervals.loc[click_name, 1])

    fit_row = {
        "模型": coefficient_rows[0]["模型"],
        "因变量": outcome,
        "因变量单位": outcome_unit,
        "样本量": int(fitted.nobs),
        "R2": float(fitted.rsquared),
        "调整R2": float(fitted.rsquared_adj),
        "AIC": float(fitted.aic),
        "BIC": float(fitted.bic),
        "Durbin_Watson": float(durbin_watson(fitted.resid)),
        "协方差估计": f"Newey-West HAC(maxlags={HAC_MAX_LAG})",
        "假日变量参考组": "非节假日",
        "月份参考组": "1月",
        "星期参考组": "周一",
        "五个主要假日变量联合Wald统计量": float(
            np.asarray(joint_test.statistic).squeeze()
        ),
        "五个主要假日变量联合自由度": len(formal_holiday_columns),
        "五个主要假日变量联合p值": float(joint_test.pvalue),
        "同日点击系数_每1000次": click_coefficient,
        "同日点击p值": click_p_value,
        "同日点击95%CI下限": click_ci_low,
        "同日点击95%CI上限": click_ci_high,
    }
    return coefficient_rows, fit_row


def format_p_value(value: float) -> str:
    """格式化终端摘要中的 p 值。"""
    return f"{value:.3e}" if value < 0.001 else f"{value:.4f}"


def print_summary(
    matches: pd.DataFrame,
    comparisons: pd.DataFrame,
    model_fit: pd.DataFrame,
) -> None:
    """打印简洁的过程与模型汇总，不作业务或因果解释。"""
    print("局部匹配摘要")
    counts = matches.groupby("假期").size().reindex(FORMAL_HOLIDAYS)
    for holiday, count in counts.items():
        print(f"{holiday}: {count}个假期日，每日匹配1个前置和1个后置同星期对照。")
    formal = comparisons.loc[comparisons["推断类型"] == "正式配对检验"]
    significant_counts = formal.groupby("指标")["校正后显著"].sum().reindex(METRICS)
    print("Holm校正后显著的局部假期比较数（共5个/指标）：")
    print(significant_counts.to_string())
    print("元旦仅保存描述性结果，未进行正式显著性检验。")

    print("\n控制月份和星期后的模型摘要")
    for row in model_fit.itertuples(index=False):
        print(
            f"{row.因变量}: 调整R2={row.调整R2:.4f}, "
            f"五个主要假日联合Wald={row.五个主要假日变量联合Wald统计量:.4f}, "
            f"联合p={format_p_value(row.五个主要假日变量联合p值)}。"
        )
    print("注：局部匹配与控制模型均只反映统计关联，不构成假日的因果效应。")


def main() -> None:
    daily = add_holiday_type(load_daily_summary())
    matches = build_formal_holiday_matches(daily)
    local_comparisons = build_local_comparisons(daily, matches)

    coefficient_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    for outcome in METRICS:
        coefficients, fit = fit_control_model(daily, outcome)
        coefficient_rows.extend(coefficients)
        fit_rows.append(fit)
    coefficients_df = pd.DataFrame(coefficient_rows)
    model_fit_df = pd.DataFrame(fit_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matches.to_csv(
        MATCHES_CSV,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
        float_format="%.12g",
    )
    local_comparisons.to_csv(
        LOCAL_COMPARISONS_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    coefficients_df.to_csv(
        MODEL_COEFFICIENTS_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )
    model_fit_df.to_csv(
        MODEL_FIT_CSV,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )

    print_summary(matches, local_comparisons, model_fit_df)


if __name__ == "__main__":
    main()

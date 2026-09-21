"""问题一：关键词管理与运用合理性的描述性分析。

仅使用附件1.xlsx 的 Sheet3，完成数据质量、集中度、成本与流量质量关系、
重复投放关键词和推广单元关键词结构分析。不执行聚类、综合评价或问题2分类。

关键词投放记录以 (方案ID, 推广单元ID, 关键词) 三元组标识；关键词编码本身
不被当作唯一主键。跳出率和平均访问时长中的“/”表示指标不可定义，不插补。
"""

from __future__ import annotations

from datetime import time
from math import ceil
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
matplotlib.set_loglevel("error")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CANDIDATES = [
    PROJECT_ROOT / "附件1.xlsx",
    PROJECT_ROOT / "data" / "raw" / "附件1.xlsx",
    PROJECT_ROOT / "data" / "raw" / "Attachment" / "Attachment1.xlsx",
]
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"

BASIC_STATS_CSV = TABLE_DIR / "q1_keyword_basic_stats.csv"
ANOMALIES_CSV = TABLE_DIR / "q1_keyword_anomalies.csv"
CONCENTRATION_CSV = TABLE_DIR / "q1_keyword_concentration.csv"
GINI_COMPARISON_CSV = TABLE_DIR / "q1_keyword_gini_comparison.csv"
CORRELATIONS_CSV = TABLE_DIR / "q1_keyword_correlations.csv"
DUPLICATE_SUMMARY_CSV = TABLE_DIR / "q1_keyword_duplicate_summary.csv"
DUPLICATE_TOP20_CSV = TABLE_DIR / "q1_keyword_duplicate_top20.csv"
UNIT_SUMMARY_CSV = TABLE_DIR / "q1_keyword_unit_summary.csv"

LORENZ_FIGURE = FIGURE_DIR / "q1_keyword_lorenz_curves.png"
SCATTER_FIGURE = FIGURE_DIR / "q1_keyword_relationship_scatterplots.png"

REQUIRED_COLUMNS = [
    "关键词",
    "方案ID",
    "推广单元ID",
    "消费额",
    "点击量",
    "浏览量",
    "跳出率",
    "平均访问时长",
]
ID_COLUMNS = ["方案ID", "推广单元ID", "关键词"]
NUMERIC_COLUMNS = ["消费额", "点击量", "浏览量"]
TOP_PROPORTIONS = [0.01, 0.05, 0.10, 0.20]


def find_input_file() -> Path:
    """按常用位置查找附件1，兼容当前项目的英文文件名。"""
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    searched = "\n".join(str(path) for path in INPUT_CANDIDATES)
    raise FileNotFoundError(f"未找到附件1.xlsx，已检查：\n{searched}")


def normalize_identifier(value: object) -> str:
    """将 Excel 中的编码稳定转换为字符串，避免整数出现 .0。"""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def is_undefined_marker(value: object) -> bool:
    """识别题目中表示指标不可定义的斜杠或空值。"""
    if pd.isna(value):
        return True
    return str(value).strip() in {"", "/"}


def parse_rate(value: object) -> float:
    """将跳出率转换为 0--1 尺度；斜杠保持为 NaN。"""
    if is_undefined_marker(value):
        return np.nan
    text = str(value).strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    parsed = float(value)
    # 兼容未来可能出现的 61.09 形式；当前 Sheet3 已是 0.6109 形式。
    return parsed / 100.0 if 1 < parsed <= 100 else parsed


def parse_duration_seconds(value: object) -> float:
    """将访问时长统一为秒；斜杠保持为 NaN。"""
    if is_undefined_marker(value):
        return np.nan
    if isinstance(value, time):
        return float(value.hour * 3600 + value.minute * 60 + value.second)
    if isinstance(value, pd.Timedelta):
        return float(value.total_seconds())
    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        # Excel 时间通常以“天的小数”存储；整数 0 是有效的 0 秒。
        if 0 < abs(numeric) < 1:
            return numeric * 86400.0
        return numeric
    duration = pd.to_timedelta(str(value).strip(), errors="raise")
    return float(duration.total_seconds())


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """逐行安全除法，分母为 0 时返回 NaN。"""
    result = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid = denominator.ne(0) & denominator.notna() & numerator.notna()
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def load_and_prepare() -> pd.DataFrame:
    """读取 Sheet3、校验字段并构造关键词级派生指标与质量标记。"""
    input_file = find_input_file()
    raw = pd.read_excel(input_file, sheet_name="Sheet3", dtype=object)
    raw.columns = raw.columns.map(lambda column: str(column).strip())

    missing = sorted(set(REQUIRED_COLUMNS).difference(raw.columns))
    if missing:
        raise ValueError(f"Sheet3 缺少字段：{'、'.join(missing)}")

    data = raw[REQUIRED_COLUMNS].copy()
    for column in ID_COLUMNS:
        data[column] = data[column].map(normalize_identifier).astype("string")
        if data[column].eq("").any():
            raise ValueError(f"字段 {column} 存在空编码，无法构造投放记录标识。")

    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(data[column], errors="coerce")
        invalid = data[column].notna() & converted.isna()
        if invalid.any():
            examples = data.loc[invalid, column].astype(str).unique()[:5]
            raise ValueError(f"字段 {column} 存在非数值：{examples.tolist()}")
        data[column] = converted.astype(float)

    data["跳出率不可定义"] = data["跳出率"].map(is_undefined_marker)
    data["平均访问时长不可定义"] = data["平均访问时长"].map(is_undefined_marker)
    data["跳出率"] = data["跳出率"].map(parse_rate).astype(float)
    data["平均访问时长_秒"] = data["平均访问时长"].map(
        parse_duration_seconds
    ).astype(float)
    data = data.drop(columns="平均访问时长")

    data["CPC"] = safe_divide(data["消费额"], data["点击量"])
    data["浏览深度"] = safe_divide(data["浏览量"], data["点击量"])
    data["zero_effect"] = (
        data["消费额"].eq(0) & data["点击量"].eq(0) & data["浏览量"].eq(0)
    )
    data["有消费但0点击"] = data["消费额"].gt(0) & data["点击量"].eq(0)
    data["0点击但浏览量大于0"] = data["点击量"].eq(0) & data["浏览量"].gt(0)
    data["唯一标识重复"] = data.duplicated(ID_COLUMNS, keep=False)
    data["跳出率范围异常"] = data["跳出率"].notna() & ~data["跳出率"].between(0, 1)
    data["平均访问时长为负"] = (
        data["平均访问时长_秒"].notna() & data["平均访问时长_秒"].lt(0)
    )
    return data


def build_basic_stats(data: pd.DataFrame) -> pd.DataFrame:
    """汇总基础规模、零效果和数据质量统计。"""
    total = len(data)
    keyword_profile = data.groupby("关键词", observed=True).agg(
        推广单元数=("推广单元ID", "nunique"),
        方案数=("方案ID", "nunique"),
        投放实例数=("关键词", "size"),
    )

    def row(name: str, value: int, note: str, ratio: float | None = None) -> dict:
        return {
            "统计项": name,
            "数量": int(value),
            "占总记录比例": np.nan if ratio is None else float(ratio),
            "说明": note,
        }

    rows = [
        row("Sheet3总记录数", total, "原始投放记录行数", 1.0),
        row("不同关键词编码数量", data["关键词"].nunique(), "关键词不是唯一主键"),
        row(
            "不同投放记录三元组数量",
            data[ID_COLUMNS].drop_duplicates().shape[0],
            "唯一标识=(方案ID,推广单元ID,关键词)",
        ),
        row(
            "唯一标识重复记录数",
            int(data["唯一标识重复"].sum()),
            "仅标记，不删除",
            data["唯一标识重复"].mean(),
        ),
        row(
            "zero_effect记录数",
            int(data["zero_effect"].sum()),
            "消费额、点击量、浏览量均为0",
            data["zero_effect"].mean(),
        ),
        row(
            "有消费但0点击记录数",
            int(data["有消费但0点击"].sum()),
            "异常仅标记，不删除",
            data["有消费但0点击"].mean(),
        ),
        row(
            "0点击但浏览量大于0记录数",
            int(data["0点击但浏览量大于0"].sum()),
            "异常明细另存q1_keyword_anomalies.csv",
            data["0点击但浏览量大于0"].mean(),
        ),
        row(
            "跳出率不可定义记录数",
            int(data["跳出率不可定义"].sum()),
            "原值为/或空，不插补",
            data["跳出率不可定义"].mean(),
        ),
        row(
            "平均访问时长不可定义记录数",
            int(data["平均访问时长不可定义"].sum()),
            "原值为/或空，不插补",
            data["平均访问时长不可定义"].mean(),
        ),
        row(
            "跨多个推广单元关键词数量",
            int(keyword_profile["推广单元数"].gt(1).sum()),
            "按关键词编码统计",
        ),
        row(
            "跨多个方案关键词数量",
            int(keyword_profile["方案数"].gt(1).sum()),
            "按关键词编码统计",
        ),
    ]
    return pd.DataFrame(rows)


def build_anomaly_details(data: pd.DataFrame) -> pd.DataFrame:
    """输出异常记录明细，不从后续原始记录集合中删除。"""
    anomaly_flags = [
        "有消费但0点击",
        "0点击但浏览量大于0",
        "唯一标识重复",
        "跳出率范围异常",
        "平均访问时长为负",
    ]
    masks = data[anomaly_flags]
    anomaly_mask = masks.any(axis=1)
    columns = [
        *ID_COLUMNS,
        "消费额",
        "点击量",
        "浏览量",
        "CPC",
        "浏览深度",
        "跳出率",
        "平均访问时长_秒",
        *anomaly_flags,
    ]
    return data.loc[anomaly_mask, columns].copy()


def gini_coefficient(values: pd.Series) -> float:
    """计算允许含 0 的非负序列 Gini 系数。"""
    array = values.dropna().to_numpy(dtype=float)
    if np.any(array < 0):
        raise ValueError("Gini 系数要求输入非负，数据中发现负值。")
    if len(array) == 0 or array.sum() == 0:
        return np.nan
    array = np.sort(array)
    n = len(array)
    positions = np.arange(1, n + 1)
    return float((2 * np.sum(positions * array) / (n * array.sum())) - (n + 1) / n)


def contribution_share(values: pd.Series, total: float) -> float:
    """安全计算贡献比例。"""
    return float(values.sum() / total) if total != 0 else np.nan


def build_concentration_scopes(
    data: pd.DataFrame,
) -> list[tuple[str, str, pd.DataFrame, list[str]]]:
    """构造全部记录、实际效果记录和独立关键词三种集中度口径。"""
    effective = data.loc[~data["zero_effect"]].copy()
    keyword_aggregated = (
        data.groupby("关键词", observed=True, sort=True)[["消费额", "点击量", "浏览量"]]
        .sum()
        .reset_index()
    )
    return [
        ("全部投放记录（含zero_effect）", "关键词投放记录", data, ID_COLUMNS),
        ("有实际效果记录（排除zero_effect）", "关键词投放记录", effective, ID_COLUMNS),
        ("独立关键词（按关键词编码聚合）", "独立关键词", keyword_aggregated, ["关键词"]),
    ]


def build_concentration(
    scopes: list[tuple[str, str, pd.DataFrame, list[str]]],
) -> pd.DataFrame:
    """在三种口径下计算头部对象的消费、点击和浏览贡献率。"""
    rows: list[dict[str, object]] = []
    for scope_name, object_name, scope_data, key_columns in scopes:
        totals = {
            metric: float(scope_data[metric].sum())
            for metric in ["消费额", "点击量", "浏览量"]
        }
        gini = {
            "消费额": gini_coefficient(scope_data["消费额"]),
            "点击量": gini_coefficient(scope_data["点击量"]),
        }
        for sort_metric in ["消费额", "点击量"]:
            ordered = scope_data.sort_values(
                [sort_metric, *key_columns],
                ascending=[False, *([True] * len(key_columns))],
                kind="mergesort",
            )
            for proportion in TOP_PROPORTIONS:
                selected_count = max(1, ceil(len(ordered) * proportion))
                selected = ordered.head(selected_count)
                rows.append(
                    {
                        "分析口径": scope_name,
                        "分析对象": object_name,
                        "总体数量": len(ordered),
                        "排序依据": f"{sort_metric}降序",
                        "头部比例": proportion,
                        "入选对象数": selected_count,
                        "实际入选比例": selected_count / len(ordered),
                        "总消费额贡献率": contribution_share(
                            selected["消费额"], totals["消费额"]
                        ),
                        "总点击量贡献率": contribution_share(
                            selected["点击量"], totals["点击量"]
                        ),
                        "总浏览量贡献率": contribution_share(
                            selected["浏览量"], totals["浏览量"]
                        ),
                        "排序指标Gini系数": gini[sort_metric],
                        "边界处理": "对象数向上取整；并列值按对象编码稳定排序",
                    }
                )
    return pd.DataFrame(rows)


def build_gini_comparison(
    scopes: list[tuple[str, str, pd.DataFrame, list[str]]],
) -> pd.DataFrame:
    """单独汇总三种分析口径下的消费额和点击量 Gini。"""
    rows: list[dict[str, object]] = []
    for scope_name, object_name, scope_data, _ in scopes:
        for metric in ["消费额", "点击量"]:
            rows.append(
                {
                    "分析口径": scope_name,
                    "分析对象": object_name,
                    "总体数量": len(scope_data),
                    "指标": metric,
                    "该指标零值对象数": int(scope_data[metric].eq(0).sum()),
                    "该指标零值对象比例": float(scope_data[metric].eq(0).mean()),
                    "Gini系数": gini_coefficient(scope_data[metric]),
                    "zero_effect处理": (
                        "排除消费额、点击量、浏览量均为0的记录"
                        if scope_name.startswith("有实际效果")
                        else "先按关键词汇总，保留汇总后为0的关键词"
                        if scope_name.startswith("独立关键词")
                        else "保留"
                    ),
                }
            )
    return pd.DataFrame(rows)


def lorenz_coordinates(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """返回 Lorenz 曲线的累计记录比例与累计指标比例。"""
    array = np.sort(values.dropna().to_numpy(dtype=float))
    if np.any(array < 0):
        raise ValueError("Lorenz 曲线要求输入非负，数据中发现负值。")
    population = np.arange(len(array) + 1) / len(array)
    cumulative = np.insert(np.cumsum(array), 0, 0.0)
    if cumulative[-1] == 0:
        return population, np.full_like(population, np.nan, dtype=float)
    return population, cumulative / cumulative[-1]


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


def save_lorenz_plot(
    scopes: list[tuple[str, str, pd.DataFrame, list[str]]],
) -> None:
    """并列绘制三种分析口径下的消费额和点击量 Lorenz 曲线。"""
    configure_chinese_font()
    figure, axes = plt.subplots(1, 3, figsize=(19, 6.3), sharex=True, sharey=True)
    colors = {"消费额": "#C44E52", "点击量": "#4C72B0"}
    for axis, (scope_name, object_name, scope_data, _) in zip(axes, scopes):
        for metric in ["消费额", "点击量"]:
            x, y = lorenz_coordinates(scope_data[metric])
            gini = gini_coefficient(scope_data[metric])
            axis.plot(
                x,
                y,
                linewidth=2.2,
                color=colors[metric],
                label=f"{metric}（Gini={gini:.3f}）",
            )
        axis.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="#666666",
            linewidth=1.2,
            label="完全均等线",
        )
        axis.set_title(f"{scope_name}\n$n={len(scope_data)}$")
        axis.set_xlabel(f"累计{object_name}比例")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(linestyle="--", linewidth=0.6, alpha=0.4)
        axis.legend(fontsize=9)
    axes[0].set_ylabel("累计指标贡献比例")
    figure.suptitle("不同统计口径下的消费额与点击量 Lorenz 曲线", fontsize=16)
    figure.tight_layout()
    figure.savefig(LORENZ_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)


CORRELATION_SPECS = [
    ("消费额", "点击量"),
    ("CPC", "浏览深度"),
    ("CPC", "跳出率"),
    ("CPC", "平均访问时长_秒"),
    ("点击量", "浏览量"),
    ("浏览深度", "跳出率"),
    ("浏览深度", "平均访问时长_秒"),
]


def safe_spearman(first: pd.Series, second: pd.Series) -> tuple[int, float, float]:
    """对成对有效且非常数的数据计算 Spearman 相关。"""
    pairs = pd.concat([first, second], axis=1).dropna()
    if len(pairs) < 3 or pairs.iloc[:, 0].nunique() < 2 or pairs.iloc[:, 1].nunique() < 2:
        return len(pairs), np.nan, np.nan
    result = stats.spearmanr(pairs.iloc[:, 0], pairs.iloc[:, 1])
    return len(pairs), float(result.statistic), float(result.pvalue)


def build_correlations(data: pd.DataFrame) -> pd.DataFrame:
    """在点击量为正的记录上计算指定 Spearman 相关。"""
    # 浏览量为 0 是有效观测，对应浏览深度 0；只有点击量为 0 时比率不可定义。
    valid = data.loc[data["点击量"].gt(0)].copy()
    rows: list[dict[str, object]] = []
    for first, second in CORRELATION_SPECS:
        sample_size, coefficient, p_value = safe_spearman(valid[first], valid[second])
        rows.append(
            {
                "变量1": first,
                "变量2": second,
                "方法": "Spearman",
                "样本量": sample_size,
                "相关系数": coefficient,
                "p值": p_value,
                "样本口径": "点击量>0；浏览量可为0；其余指标按成对非缺失",
                "解释限制": "统计关联，不表示因果关系",
            }
        )
    return pd.DataFrame(rows)


def save_scatterplots(data: pd.DataFrame, correlations: pd.DataFrame) -> None:
    """将七组相关关系绘制在一张多面板散点图中。"""
    valid = data.loc[data["点击量"].gt(0)].copy()
    corr_lookup = correlations.set_index(["变量1", "变量2"])
    log_columns = {"消费额", "点击量", "浏览量"}

    figure, axes = plt.subplots(3, 3, figsize=(16, 14))
    axes_flat = axes.ravel()
    for axis, (first, second) in zip(axes_flat, CORRELATION_SPECS):
        pairs = valid[[first, second]].dropna()
        x = np.log1p(pairs[first]) if first in log_columns else pairs[first]
        y = np.log1p(pairs[second]) if second in log_columns else pairs[second]
        axis.scatter(x, y, s=13, alpha=0.30, color="#4C72B0", edgecolors="none")
        first_label = f"log1p({first})" if first in log_columns else first
        second_label = f"log1p({second})" if second in log_columns else second
        row = corr_lookup.loc[(first, second)]
        axis.set_title(f"{first} 与 {second}\nSpearman ρ={row['相关系数']:.3f}, n={int(row['样本量'])}")
        axis.set_xlabel(first_label)
        axis.set_ylabel(second_label)
        axis.grid(linestyle="--", linewidth=0.5, alpha=0.3)
    for axis in axes_flat[len(CORRELATION_SPECS) :]:
        axis.remove()
    figure.suptitle("关键词成本与流量质量关系", fontsize=16, y=1.01)
    figure.tight_layout()
    figure.savefig(SCATTER_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(figure)


def positive_max_min_ratio(values: pd.Series) -> tuple[int, float]:
    """计算至少两个正且非缺失值的最大/最小比。"""
    positive = values.dropna().loc[lambda series: series.gt(0)]
    if len(positive) < 2:
        return len(positive), np.nan
    return len(positive), float(positive.max() / positive.min())


def safe_median(values: pd.Series) -> float:
    """仅在存在非缺失值时计算中位数，避免全缺失组产生运行警告。"""
    valid = values.dropna()
    return float(valid.median()) if len(valid) else np.nan


def build_duplicate_summary(data: pd.DataFrame) -> pd.DataFrame:
    """按关键词汇总跨投放实例的表现分布和组内变异。"""
    profile = data.groupby("关键词", observed=True).agg(
        推广单元数=("推广单元ID", "nunique"),
        方案数=("方案ID", "nunique"),
    )
    duplicate_keywords = profile.index[
        profile["推广单元数"].gt(1) | profile["方案数"].gt(1)
    ]
    duplicate_data = data.loc[data["关键词"].isin(duplicate_keywords)]
    rows: list[dict[str, object]] = []

    for keyword, group in duplicate_data.groupby("关键词", sort=True, observed=True):
        cpc_count, cpc_ratio = positive_max_min_ratio(group["CPC"])
        depth_count, depth_ratio = positive_max_min_ratio(group["浏览深度"])
        ratios = np.asarray([cpc_ratio, depth_ratio], dtype=float)
        maximum_ratio = float(np.nanmax(ratios)) if np.isfinite(ratios).any() else np.nan
        rows.append(
            {
                "关键词": keyword,
                "投放实例数": len(group),
                "方案数": group["方案ID"].nunique(),
                "推广单元数": group["推广单元ID"].nunique(),
                "方案ID列表": "；".join(sorted(group["方案ID"].unique())),
                "推广单元ID列表": "；".join(sorted(group["推广单元ID"].unique())),
                "总消费额": group["消费额"].sum(),
                "总点击量": group["点击量"].sum(),
                "总浏览量": group["浏览量"].sum(),
                "CPC有效正值数": cpc_count,
                "CPC最小正值": group.loc[group["CPC"].gt(0), "CPC"].min(),
                "CPC中位数": safe_median(group["CPC"]),
                "CPC最大值": group["CPC"].max(),
                "CPC最大最小比": cpc_ratio,
                "点击量最小值": group["点击量"].min(),
                "点击量中位数": safe_median(group["点击量"]),
                "点击量最大值": group["点击量"].max(),
                "点击量极差": group["点击量"].max() - group["点击量"].min(),
                "浏览深度有效正值数": depth_count,
                "浏览深度最小正值": group.loc[group["浏览深度"].gt(0), "浏览深度"].min(),
                "浏览深度中位数": safe_median(group["浏览深度"]),
                "浏览深度最大值": group["浏览深度"].max(),
                "浏览深度最大最小比": depth_ratio,
                "跳出率最小值": group["跳出率"].min(),
                "跳出率中位数": safe_median(group["跳出率"]),
                "跳出率最大值": group["跳出率"].max(),
                "平均访问时长最小值_秒": group["平均访问时长_秒"].min(),
                "平均访问时长中位数_秒": safe_median(group["平均访问时长_秒"]),
                "平均访问时长最大值_秒": group["平均访问时长_秒"].max(),
                "组内最大倍率": maximum_ratio,
                "差异排序依据": "max(CPC最大最小比,浏览深度最大最小比)；仅使用至少2个正值",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["组内最大倍率", "点击量极差", "关键词"],
        ascending=[False, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def build_duplicate_top20(
    data: pd.DataFrame,
    duplicate_summary: pd.DataFrame,
) -> pd.DataFrame:
    """选出组内倍率差异最大的20个关键词，并输出其全部投放实例。"""
    ranked = duplicate_summary.head(20).copy()
    ranked["差异排名"] = np.arange(1, len(ranked) + 1)
    rank_columns = ranked[
        ["关键词", "差异排名", "组内最大倍率", "CPC最大最小比", "浏览深度最大最小比"]
    ]
    details = data.merge(rank_columns, on="关键词", how="inner", validate="many_to_one")
    output_columns = [
        "差异排名",
        "关键词",
        "方案ID",
        "推广单元ID",
        "消费额",
        "点击量",
        "浏览量",
        "CPC",
        "浏览深度",
        "跳出率",
        "平均访问时长_秒",
        "zero_effect",
        "组内最大倍率",
        "CPC最大最小比",
        "浏览深度最大最小比",
    ]
    return details[output_columns].sort_values(
        ["差异排名", "方案ID", "推广单元ID"],
        kind="mergesort",
    )


def build_unit_summary(data: pd.DataFrame) -> pd.DataFrame:
    """按方案ID与推广单元ID汇总关键词质量结构。"""
    summary = (
        data.groupby(["方案ID", "推广单元ID"], observed=True, sort=True)
        .agg(
            关键词投放记录数=("关键词", "size"),
            不同关键词数=("关键词", "nunique"),
            zero_effect记录数=("zero_effect", "sum"),
            zero_effect比例=("zero_effect", "mean"),
            总消费=("消费额", "sum"),
            总点击=("点击量", "sum"),
            中位数CPC=("CPC", "median"),
            中位数浏览深度=("浏览深度", "median"),
            中位数跳出率=("跳出率", "median"),
            中位数平均访问时长_秒=("平均访问时长_秒", "median"),
            跳出率不可定义数=("跳出率不可定义", "sum"),
            平均访问时长不可定义数=("平均访问时长不可定义", "sum"),
        )
        .reset_index()
    )
    return summary


def save_csv(dataframe: pd.DataFrame, path: Path) -> None:
    """统一保存为适合 Excel 打开的 UTF-8 BOM CSV。"""
    dataframe.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.12g",
    )


def print_results(
    basic_stats: pd.DataFrame,
    concentration: pd.DataFrame,
    gini_comparison: pd.DataFrame,
    correlations: pd.DataFrame,
    duplicate_summary: pd.DataFrame,
    data: pd.DataFrame,
    unit_summary: pd.DataFrame,
) -> None:
    """仅打印题目要求的五类简洁结果。"""
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print("基础统计")
        print(basic_stats.to_string(index=False))

        print("\n前1%、5%、10%、20%集中度")
        print(
            concentration[
                [
                    "分析口径",
                    "总体数量",
                    "排序依据",
                    "头部比例",
                    "入选对象数",
                    "总消费额贡献率",
                    "总点击量贡献率",
                    "总浏览量贡献率",
                ]
            ].to_string(index=False)
        )
        print("\nGini口径对照")
        print(
            gini_comparison[
                [
                    "分析口径",
                    "分析对象",
                    "总体数量",
                    "指标",
                    "该指标零值对象数",
                    "该指标零值对象比例",
                    "Gini系数",
                ]
            ].to_string(index=False)
        )

        print("\n相关性摘要")
        print(
            correlations[["变量1", "变量2", "样本量", "相关系数", "p值"]].to_string(
                index=False
            )
        )

        keyword_profile = data.groupby("关键词", observed=True).agg(
            投放实例数=("关键词", "size"),
            推广单元数=("推广单元ID", "nunique"),
            方案数=("方案ID", "nunique"),
        )
        duplicate_print = pd.DataFrame(
            {
                "统计项": [
                    "跨推广单元或方案重复关键词数量",
                    "跨多个推广单元关键词数量",
                    "跨多个方案关键词数量",
                    "进入差异Top20的关键词数量",
                ],
                "数量": [
                    int(
                        (
                            keyword_profile["推广单元数"].gt(1)
                            | keyword_profile["方案数"].gt(1)
                        ).sum()
                    ),
                    int(keyword_profile["推广单元数"].gt(1).sum()),
                    int(keyword_profile["方案数"].gt(1).sum()),
                    min(20, len(duplicate_summary)),
                ],
            }
        )
        print("\n重复关键词摘要")
        print(duplicate_print.to_string(index=False))

        print("\n各推广单元关键词质量摘要")
        print(unit_summary.to_string(index=False))


def main() -> None:
    data = load_and_prepare()
    basic_stats = build_basic_stats(data)
    anomalies = build_anomaly_details(data)
    concentration_scopes = build_concentration_scopes(data)
    concentration = build_concentration(concentration_scopes)
    gini_comparison = build_gini_comparison(concentration_scopes)
    correlations = build_correlations(data)
    duplicate_summary = build_duplicate_summary(data)
    duplicate_top20 = build_duplicate_top20(data, duplicate_summary)
    unit_summary = build_unit_summary(data)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(basic_stats, BASIC_STATS_CSV)
    save_csv(anomalies, ANOMALIES_CSV)
    save_csv(concentration, CONCENTRATION_CSV)
    save_csv(gini_comparison, GINI_COMPARISON_CSV)
    save_csv(correlations, CORRELATIONS_CSV)
    save_csv(duplicate_summary, DUPLICATE_SUMMARY_CSV)
    save_csv(duplicate_top20, DUPLICATE_TOP20_CSV)
    save_csv(unit_summary, UNIT_SUMMARY_CSV)
    save_lorenz_plot(concentration_scopes)
    save_scatterplots(data, correlations)

    print_results(
        basic_stats,
        concentration,
        gini_comparison,
        correlations,
        duplicate_summary,
        data,
        unit_summary,
    )


if __name__ == "__main__":
    main()

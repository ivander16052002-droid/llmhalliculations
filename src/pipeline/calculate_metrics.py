from pathlib import Path
import warnings

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Explicit final annotated files.
# Do not auto-discover files here: the project directory contains drafts,
# prefilled files, duplicates and intermediate annotation files.
INPUT_FILES = [
    DATA_DIR / "annotated" / "llama3_quote_full_20260517_v1_annotation_blank.xlsx",
    DATA_DIR / "annotated" / "qwen_quote_full_20260517_v1_annotation_blank.xlsx",
    DATA_DIR / "annotated" / "yandex_quote_full_20260505_v1_annotation_blank_v2.xlsx",
]

OUT_DIR = DATA_DIR / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "svr",
    "fr",
    "mr",
    "mr_author",
    "mr_title",
    "mr_year",
    "mr_container",
    "qvr",
    "esr",
    "lvr",
]

TRUE_VALUES = {"истина", "true", "1", "да", "yes"}
FALSE_VALUES = {"ложь", "false", "0", "нет", "no"}
NA_VALUES = {"", "na", "nan", "none", "null", "н/д", "нд", "-"}


def norm_metric_value(x):
    """
    Normalize final annotation metric values:
    ИСТИНА / TRUE / 1 / 1.0 -> 1
    ЛОЖЬ / FALSE / 0 / 0.0 -> 0
    NA / null / empty -> pd.NA

    For metric columns, unexpected values fail loudly.
    """
    if pd.isna(x):
        return pd.NA

    if isinstance(x, (int, float, np.integer, np.floating)):
        if float(x) == 1.0:
            return 1
        if float(x) == 0.0:
            return 0
        raise ValueError(f"Unexpected numeric metric value: {x!r}")

    s = str(x).strip()
    sl = s.lower()

    try:
        numeric = float(sl)
        if numeric == 1.0:
            return 1
        if numeric == 0.0:
            return 0
    except ValueError:
        pass

    if sl in TRUE_VALUES:
        return 1
    if sl in FALSE_VALUES:
        return 0
    if sl in NA_VALUES:
        return pd.NA

    raise ValueError(f"Unexpected metric value: {x!r}")


def norm_bool_soft(x):
    """
    Softer normalization for helper columns:
    source_present, quote_present, locator_present.

    These helper columns are not final metric columns, so unknown values
    are treated as NA instead of crashing the script.
    """
    if pd.isna(x):
        return pd.NA

    if isinstance(x, (int, float, np.integer, np.floating)):
        if float(x) == 1.0:
            return 1
        if float(x) == 0.0:
            return 0
        return pd.NA

    s = str(x).strip()
    sl = s.lower()

    try:
        numeric = float(sl)
        if numeric == 1.0:
            return 1
        if numeric == 0.0:
            return 0
    except ValueError:
        pass

    if sl in TRUE_VALUES:
        return 1
    if sl in FALSE_VALUES:
        return 0
    if sl in NA_VALUES:
        return pd.NA

    return pd.NA


def is_nonempty_text(x) -> bool:
    """
    True if a raw text field is non-empty and not a null-like string.

    Used for counting how often the model output source / quote / locator
    fields at all, regardless of later validation.
    """
    if pd.isna(x):
        return False

    s = str(x).strip()
    if s.lower() in NA_VALUES:
        return False

    return bool(s)


def read_one_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file type: {path}")


def collect_input_files() -> list[Path]:
    """
    Return only the three final annotated files explicitly listed in INPUT_FILES.
    """
    files = []

    for path in INPUT_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")
        files.append(path)

    print("Input annotation files:")
    for path in files:
        print(f"  {path}")

    return files


def load_annotations() -> pd.DataFrame:
    files = collect_input_files()

    frames = []
    for path in files:
        df = read_one_file(path)
        df.columns = [str(c).strip() for c in df.columns]
        df["annotation_file"] = path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    required_columns = [
        "run_id",
        "question_id",
        "discipline",
        "model",
        "prompt_type",
        "status",
    ]

    missing_required = [c for c in required_columns if c not in combined.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    missing_metrics = [m for m in METRICS if m not in combined.columns]
    if missing_metrics:
        raise ValueError(f"Missing metric columns: {missing_metrics}")

    return combined


def normalize_annotations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["model"] = df["model"].astype(str).str.strip().str.lower()
    df["prompt_type"] = df["prompt_type"].astype(str).str.strip().str.lower()
    df["status_norm"] = df["status"].astype(str).str.strip().str.lower()

    for metric in METRICS:
        df[metric] = df[metric].map(norm_metric_value).astype("Int64")

    for helper_col in ["source_present", "quote_present", "locator_present"]:
        if helper_col in df.columns:
            df[f"{helper_col}_norm"] = (
                df[helper_col].map(norm_bool_soft).astype("Int64")
            )

    return df


def count_positive(series: pd.Series) -> int:
    return int((series == 1).sum())


def count_applicable(series: pd.Series) -> int:
    return int(series.notna().sum())


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return np.nan
    return numerator / denominator


def raw_nonempty_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].map(is_nonempty_text).sum())


def helper_true_count(df: pd.DataFrame, column: str) -> int:
    norm_col = f"{column}_norm"
    if norm_col not in df.columns:
        return 0
    return int((df[norm_col] == 1).sum())


def summarize_group(group: pd.DataFrame) -> pd.Series:
    n_responses = len(group)

    answered = int((group["status_norm"] == "answered").sum())
    partial = int((group["status_norm"] == "partial").sum())
    refusal = int(
        group["status_norm"]
        .isin(["refusal", "full_refusal", "full refusal"])
        .sum()
    )
    other_status = n_responses - answered - partial - refusal

    # Raw / helper output coverage.
    # These are not validity metrics. They describe whether the model output
    # source/quote/locator-like fields at all.
    source_present_raw_n = helper_true_count(group, "source_present")
    quote_present_raw_n = helper_true_count(group, "quote_present")
    locator_present_raw_n = helper_true_count(group, "locator_present")

    source_reference_nonempty_n = raw_nonempty_count(group, "source_reference")
    source_id_nonempty_n = raw_nonempty_count(group, "source_id")
    source_authors_nonempty_n = raw_nonempty_count(group, "source_authors")
    source_title_nonempty_n = raw_nonempty_count(group, "source_title")
    source_year_nonempty_n = raw_nonempty_count(group, "source_year")
    source_container_nonempty_n = raw_nonempty_count(group, "source_container")
    source_locator_nonempty_n = raw_nonempty_count(group, "source_locator")
    quote_id_nonempty_n = raw_nonempty_count(group, "quote_id")
    quote_text_nonempty_n = raw_nonempty_count(group, "quote_text")

    row = {
        "n_responses": n_responses,
        "answered": answered,
        "partial": partial,
        "refusal": refusal,
        "other_status": other_status,
        "partial_rate_total": safe_rate(partial, n_responses),
        "refusal_rate_total": safe_rate(refusal, n_responses),

        # Raw output coverage.
        "source_present_raw_n": source_present_raw_n,
        "quote_present_raw_n": quote_present_raw_n,
        "locator_present_raw_n": locator_present_raw_n,
        "source_reference_nonempty_n": source_reference_nonempty_n,
        "source_id_nonempty_n": source_id_nonempty_n,
        "source_authors_nonempty_n": source_authors_nonempty_n,
        "source_title_nonempty_n": source_title_nonempty_n,
        "source_year_nonempty_n": source_year_nonempty_n,
        "source_container_nonempty_n": source_container_nonempty_n,
        "source_locator_nonempty_n": source_locator_nonempty_n,
        "quote_id_nonempty_n": quote_id_nonempty_n,
        "quote_text_nonempty_n": quote_text_nonempty_n,

        "source_present_raw_rate_total": safe_rate(source_present_raw_n, n_responses),
        "quote_present_raw_rate_total": safe_rate(quote_present_raw_n, n_responses),
        "locator_present_raw_rate_total": safe_rate(locator_present_raw_n, n_responses),
        "source_reference_nonempty_rate_total": safe_rate(
            source_reference_nonempty_n, n_responses
        ),
        "quote_text_nonempty_rate_total": safe_rate(
            quote_text_nonempty_n, n_responses
        ),
        "source_locator_nonempty_rate_total": safe_rate(
            source_locator_nonempty_n, n_responses
        ),
    }

    # Metric counts and rates.
    # metric_den = count of non-NA values
    # metric_n = count of positive values
    # metric_rate = metric_n / metric_den
    for metric in METRICS:
        den = count_applicable(group[metric])
        num = count_positive(group[metric])

        row[f"{metric}_den"] = den
        row[f"{metric}_n"] = num
        row[f"{metric}_rate"] = safe_rate(num, den)

    # Compatibility aliases for figures and chapter tables.
    # SVR/FR share the same conceptual denominator: cases where the model
    # produced a checkable-looking source claim. In clean annotations,
    # these denominators should normally match.
    row["source_metric_n"] = row["svr_den"]
    row["n_source_metric"] = row["svr_den"]

    row["svr_valid_n"] = row["svr_n"]
    row["fr_fabricated_n"] = row["fr_n"]

    row["mr_error_n"] = row["mr_n"]
    row["qvr_valid_n"] = row["qvr_n"]
    row["esr_supported_n"] = row["esr_n"]

    # Sequential applicability aliases.
    row["source_identified_n"] = row["svr_n"]
    row["quote_checkable_n"] = row["qvr_den"]
    row["quote_verified_n"] = row["qvr_n"]
    row["claim_supported_n"] = row["esr_n"]
    row["locator_checkable_n"] = row["lvr_den"]
    row["locator_verified_n"] = row["lvr_n"]

    return pd.Series(row)


def add_warning_checks(summary: pd.DataFrame, label: str) -> None:
    """
    Warnings only. They help catch inconsistent annotations,
    but do not stop the pipeline.
    """
    if "svr_den" in summary.columns and "fr_den" in summary.columns:
        mismatch = summary[summary["svr_den"] != summary["fr_den"]]
        if not mismatch.empty:
            warnings.warn(
                f"[{label}] SVR and FR denominators differ in some groups. "
                "Check whether source-level NA rules were applied consistently."
            )

    if "qvr_den" in summary.columns and "svr_n" in summary.columns:
        impossible = summary[summary["qvr_den"] > summary["svr_n"]]
        if not impossible.empty:
            warnings.warn(
                f"[{label}] qvr_den is greater than svr_n in some groups. "
                "This may be valid only if citation checking was allowed beyond "
                "SVR-positive cases."
            )

    if "esr_den" in summary.columns and "qvr_n" in summary.columns:
        impossible = summary[summary["esr_den"] > summary["qvr_n"]]
        if not impossible.empty:
            warnings.warn(
                f"[{label}] esr_den is greater than qvr_n in some groups. "
                "Check sequential metric rules."
            )


def summarize_by(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Compatible with both older and newer pandas versions.
    """
    try:
        summary = (
            df.groupby(group_cols, dropna=False)
            .apply(summarize_group, include_groups=False)
            .reset_index()
        )
    except TypeError:
        summary = (
            df.groupby(group_cols, dropna=False)
            .apply(summarize_group)
            .reset_index()
        )

    add_warning_checks(summary, " / ".join(group_cols))

    return summary


def make_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_group(df).to_frame().T
    summary.insert(0, "scope", "overall")
    add_warning_checks(summary, "overall")
    return summary


def order_columns(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    base_cols = group_cols + [
        "n_responses",
        "answered",
        "partial",
        "refusal",
        "other_status",
        "partial_rate_total",
        "refusal_rate_total",

        # Raw output coverage.
        "source_present_raw_n",
        "quote_present_raw_n",
        "locator_present_raw_n",
        "source_reference_nonempty_n",
        "source_id_nonempty_n",
        "source_authors_nonempty_n",
        "source_title_nonempty_n",
        "source_year_nonempty_n",
        "source_container_nonempty_n",
        "source_locator_nonempty_n",
        "quote_id_nonempty_n",
        "quote_text_nonempty_n",
        "source_present_raw_rate_total",
        "quote_present_raw_rate_total",
        "locator_present_raw_rate_total",
        "source_reference_nonempty_rate_total",
        "quote_text_nonempty_rate_total",
        "source_locator_nonempty_rate_total",

        # Main source-level aliases.
        "source_metric_n",
        "n_source_metric",
        "svr_n",
        "svr_den",
        "svr_rate",
        "fr_n",
        "fr_den",
        "fr_rate",

        # Attribution.
        "mr_n",
        "mr_den",
        "mr_rate",
        "mr_author_n",
        "mr_author_den",
        "mr_author_rate",
        "mr_title_n",
        "mr_title_den",
        "mr_title_rate",
        "mr_year_n",
        "mr_year_den",
        "mr_year_rate",
        "mr_container_n",
        "mr_container_den",
        "mr_container_rate",

        # Quote/evidence/locator.
        "qvr_n",
        "qvr_den",
        "qvr_rate",
        "esr_n",
        "esr_den",
        "esr_rate",
        "lvr_n",
        "lvr_den",
        "lvr_rate",

        # Sequential aliases for figures.
        "source_identified_n",
        "quote_checkable_n",
        "quote_verified_n",
        "claim_supported_n",
        "locator_checkable_n",
        "locator_verified_n",
    ]

    existing_first = [c for c in base_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_first]

    return df[existing_first + remaining]


def save_outputs(
    combined: pd.DataFrame,
    overall: pd.DataFrame,
    by_model: pd.DataFrame,
    by_prompt: pd.DataFrame,
    by_model_prompt: pd.DataFrame,
    by_model_discipline: pd.DataFrame,
) -> None:
    combined_path = OUT_DIR / "combined_annotations.csv"
    overall_path = OUT_DIR / "metrics_overall.csv"
    by_model_path = OUT_DIR / "metrics_by_model.csv"
    by_prompt_path = OUT_DIR / "metrics_by_prompt.csv"
    by_model_prompt_path = OUT_DIR / "metrics_by_model_prompt.csv"
    by_model_discipline_path = OUT_DIR / "metrics_by_model_discipline.csv"
    excel_path = OUT_DIR / "metrics_summary.xlsx"

    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")
    by_model.to_csv(by_model_path, index=False, encoding="utf-8-sig")
    by_prompt.to_csv(by_prompt_path, index=False, encoding="utf-8-sig")
    by_model_prompt.to_csv(by_model_prompt_path, index=False, encoding="utf-8-sig")
    by_model_discipline.to_csv(
        by_model_discipline_path,
        index=False,
        encoding="utf-8-sig",
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="overall", index=False)
        by_model.to_excel(writer, sheet_name="by_model", index=False)
        by_prompt.to_excel(writer, sheet_name="by_prompt", index=False)
        by_model_prompt.to_excel(writer, sheet_name="by_model_prompt", index=False)
        by_model_discipline.to_excel(
            writer,
            sheet_name="by_model_discipline",
            index=False,
        )
        combined.to_excel(writer, sheet_name="combined_annotations", index=False)

    print("Saved:")
    print(f"  {combined_path}")
    print(f"  {overall_path}")
    print(f"  {by_model_path}")
    print(f"  {by_prompt_path}")
    print(f"  {by_model_prompt_path}")
    print(f"  {by_model_discipline_path}")
    print(f"  {excel_path}")


def print_short_report(by_model: pd.DataFrame, by_model_prompt: pd.DataFrame) -> None:
    model_cols = [
        "model",
        "n_responses",
        "answered",
        "partial",
        "refusal",
        "source_metric_n",
        "svr_n",
        "svr_den",
        "svr_rate",
        "fr_n",
        "fr_den",
        "fr_rate",
        "mr_n",
        "mr_den",
        "mr_rate",
        "mr_author_n",
        "mr_title_n",
        "mr_year_n",
        "mr_container_n",
        "qvr_n",
        "qvr_den",
        "qvr_rate",
        "esr_n",
        "esr_den",
        "esr_rate",
        "lvr_n",
        "lvr_den",
        "lvr_rate",
        "quote_text_nonempty_n",
        "source_locator_nonempty_n",
    ]

    prompt_cols = [
        "model",
        "prompt_type",
        "n_responses",
        "answered",
        "partial",
        "refusal",
        "source_metric_n",
        "svr_n",
        "svr_den",
        "svr_rate",
        "fr_n",
        "fr_den",
        "fr_rate",
        "mr_n",
        "mr_den",
        "mr_rate",
        "qvr_n",
        "qvr_den",
        "qvr_rate",
        "esr_n",
        "esr_den",
        "esr_rate",
        "lvr_n",
        "lvr_den",
        "lvr_rate",
    ]

    print("\nMetrics by model:")
    print(
        by_model[[c for c in model_cols if c in by_model.columns]]
        .to_string(index=False)
    )

    print("\nMetrics by model × prompt:")
    print(
        by_model_prompt[[c for c in prompt_cols if c in by_model_prompt.columns]]
        .to_string(index=False)
    )


def main():
    combined = load_annotations()
    combined = normalize_annotations(combined)

    overall = make_overall_summary(combined)
    by_model = summarize_by(combined, ["model"])
    by_prompt = summarize_by(combined, ["prompt_type"])
    by_model_prompt = summarize_by(combined, ["model", "prompt_type"])
    by_model_discipline = summarize_by(combined, ["model", "discipline"])

    overall = order_columns(overall, ["scope"])
    by_model = order_columns(by_model, ["model"])
    by_prompt = order_columns(by_prompt, ["prompt_type"])
    by_model_prompt = order_columns(by_model_prompt, ["model", "prompt_type"])
    by_model_discipline = order_columns(
        by_model_discipline,
        ["model", "discipline"],
    )

    save_outputs(
        combined=combined,
        overall=overall,
        by_model=by_model,
        by_prompt=by_prompt,
        by_model_prompt=by_model_prompt,
        by_model_discipline=by_model_discipline,
    )

    print_short_report(by_model, by_model_prompt)


if __name__ == "__main__":
    main()
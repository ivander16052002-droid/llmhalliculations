from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

ORIGINAL_SAMPLE_PATH = DATA_DIR / "reannotation" / "reannotation_sample_with_original_labels.xlsx"
REANNOTATED_PATH = DATA_DIR / "reannotation" / "reannotation_sample_BLIND_annotated.xlsx"
OUT_DIR = DATA_DIR / "reannotation"
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


def norm_value(x):
    """
    Normalize values from Excel:
    ИСТИНА/TRUE/1 -> 1
    ЛОЖЬ/FALSE/0 -> 0
    NA/null/empty -> pd.NA
    """
    if pd.isna(x):
        return pd.NA

    s = str(x).strip()

    if s == "":
        return pd.NA

    sl = s.lower()

    if sl in {"истина", "true", "1", "да", "yes"}:
        return 1

    if sl in {"ложь", "false", "0", "нет", "no"}:
        return 0

    if sl in {"na", "nan", "none", "null", "н/д", "нд", "-"}:
        return pd.NA

    # Если случайно попало что-то странное — считаем как NA,
    # чтобы не ломать сравнение.
    return pd.NA


def to_category(series: pd.Series) -> pd.Series:
    """
    Convert 0/1/NA to comparable categorical values.
    NA is treated as its own category.
    """
    normalized = series.map(norm_value)
    return normalized.astype("object").where(~normalized.isna(), "NA")


def strict_agreement(a: pd.Series, b: pd.Series) -> float:
    """
    Agreement where NA is counted as a normal category.
    """
    aa = to_category(a)
    bb = to_category(b)
    return float((aa == bb).mean())


def agreement_excluding_double_na(a: pd.Series, b: pd.Series) -> float:
    """
    Agreement excluding rows where both annotations are NA.
    Useful because many metrics are NA by design.
    """
    aa = to_category(a)
    bb = to_category(b)

    mask = ~((aa == "NA") & (bb == "NA"))

    if mask.sum() == 0:
        return np.nan

    return float((aa[mask] == bb[mask]).mean())


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """
    Cohen's kappa for categories 0 / 1 / NA.
    """
    aa = to_category(a)
    bb = to_category(b)

    n = len(aa)
    if n == 0:
        return np.nan

    observed = (aa == bb).mean()

    categories = sorted(set(aa.unique()) | set(bb.unique()), key=str)

    expected = 0.0
    for c in categories:
        pa = (aa == c).mean()
        pb = (bb == c).mean()
        expected += pa * pb

    if expected == 1:
        return np.nan

    return float((observed - expected) / (1 - expected))


def main():
    if not ORIGINAL_SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Original sample not found: {ORIGINAL_SAMPLE_PATH}")

    if not REANNOTATED_PATH.exists():
        raise FileNotFoundError(f"Reannotated file not found: {REANNOTATED_PATH}")

    original = pd.read_excel(ORIGINAL_SAMPLE_PATH)
    reann = pd.read_excel(REANNOTATED_PATH)

    original.columns = [str(c).strip() for c in original.columns]
    reann.columns = [str(c).strip() for c in reann.columns]

    if "row_id" not in original.columns:
        raise ValueError("Column 'row_id' not found in original sample.")

    if "row_id" not in reann.columns:
        raise ValueError("Column 'row_id' not found in reannotated sample.")

    re_cols = ["row_id"] + [f"re_{m}" for m in METRICS if f"re_{m}" in reann.columns]

    df = original.merge(
        reann[re_cols],
        on="row_id",
        how="left",
    )

    summary_rows = []

    for metric in METRICS:
        re_metric = f"re_{metric}"

        if metric not in df.columns:
            print(f"Warning: original metric column missing: {metric}")
            continue

        if re_metric not in df.columns:
            print(f"Warning: reannotation metric column missing: {re_metric}")
            continue

        orig_cat = to_category(df[metric])
        re_cat = to_category(df[re_metric])

        disagreements_mask = orig_cat != re_cat

        row = {
            "metric": metric,
            "n": len(df),
            "original_non_na": int((orig_cat != "NA").sum()),
            "reannotation_non_na": int((re_cat != "NA").sum()),
            "strict_agreement_including_NA": strict_agreement(df[metric], df[re_metric]),
            "agreement_excluding_double_NA": agreement_excluding_double_na(df[metric], df[re_metric]),
            "cohen_kappa_0_1_NA": cohen_kappa(df[metric], df[re_metric]),
            "disagreements": int(disagreements_mask.sum()),
        }

        summary_rows.append(row)

        df[f"orig_{metric}_norm"] = orig_cat
        df[f"re_{metric}_norm"] = re_cat
        df[f"{metric}_match"] = ~disagreements_mask

    summary = pd.DataFrame(summary_rows)

    match_cols = [f"{m}_match" for m in METRICS if f"{m}_match" in df.columns]

    if match_cols:
        disagreements = df[~df[match_cols].all(axis=1)].copy()
    else:
        disagreements = pd.DataFrame()

    summary_path = OUT_DIR / "reannotation_agreement_summary.csv"
    disagreements_path = OUT_DIR / "reannotation_disagreements.xlsx"
    report_path = OUT_DIR / "reannotation_agreement.xlsx"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    disagreements.to_excel(disagreements_path, index=False)

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        disagreements.to_excel(writer, sheet_name="disagreements", index=False)

    print("Done.")
    print("\nAgreement summary:")
    print(summary.to_string(index=False))

    print(f"\nSaved:")
    print(f"  {summary_path}")
    print(f"  {disagreements_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
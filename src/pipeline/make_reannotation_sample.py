from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

INPUT_PATH = DATA_DIR / "metrics" / "combined_annotations.csv"
OUT_DIR = DATA_DIR / "reannotation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_SHARE = 0.30
RANDOM_SEED = 42

ANNOTATION_COLS = [
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
    "source_language",
    "comment",
]


def infer_model_from_run_id(run_id: str) -> str:
    s = str(run_id).lower()
    if "yandex" in s:
        return "yandex"
    if "llama" in s:
        return "llama3"
    if "qwen" in s:
        return "qwen"
    return ""


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    print(f"Reading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    # Clean column names
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    print(f"Shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())

    # Иногда model может потеряться, но run_id есть.
    if "model" not in df.columns:
        if "run_id" in df.columns:
            print("\nColumn 'model' not found. Inferring model from run_id...")
            df["model"] = df["run_id"].map(infer_model_from_run_id)
        else:
            raise KeyError(
                "Column 'model' not found, and 'run_id' is also missing. "
                "Check that combined_annotations.csv is the correct file."
            )

    if "prompt_type" not in df.columns:
        raise KeyError(
            "Column 'prompt_type' not found. "
            "Check that combined_annotations.csv is the correct file."
        )

    if "question_id" not in df.columns:
        print("\nColumn 'question_id' not found. Creating sequential question_id for sorting only.")
        df["question_id"] = range(1, len(df) + 1)

    df["model"] = df["model"].astype(str).str.strip().str.lower()
    df["prompt_type"] = df["prompt_type"].astype(str).str.strip().str.lower()

    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(1, len(df) + 1))

    # Стратифицированная выборка по model × prompt_type
    sample_parts = []

    for (model, prompt_type), group in df.groupby(["model", "prompt_type"], dropna=False):
        n = round(len(group) * SAMPLE_SHARE)
        n = max(1, n)

        part = group.sample(n=n, random_state=RANDOM_SEED).copy()

        # На всякий случай явно возвращаем значения группировки в колонки
        part["model"] = model
        part["prompt_type"] = prompt_type

        sample_parts.append(part)

    sample = pd.concat(sample_parts, ignore_index=True)

    sample = sample.sort_values(
        by=["model", "prompt_type", "question_id"],
        kind="stable"
    ).reset_index(drop=True)

    full_sample_path = OUT_DIR / "reannotation_sample_with_original_labels.xlsx"
    sample.to_excel(full_sample_path, index=False)

    blind = sample.copy()

    for col in ANNOTATION_COLS:
        if col in blind.columns:
            blind = blind.drop(columns=[col])

    for col in ANNOTATION_COLS:
        blind[f"re_{col}"] = ""

    blind_path = OUT_DIR / "reannotation_sample_BLIND.xlsx"
    blind.to_excel(blind_path, index=False)

    print("\nDone.")
    print(f"Sample size: {len(sample)}")
    print("\nBy model/prompt:")
    print(sample.groupby(["model", "prompt_type"]).size().to_string())
    print(f"\nBlind file: {blind_path}")
    print(f"Original labels file: {full_sample_path}")


if __name__ == "__main__":
    main()
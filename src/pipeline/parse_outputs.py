from __future__ import annotations

import argparse
import csv
from dataclasses import asdict

import pandas as pd

from src.config import PROJECT_ROOT
from src.parser import parse_model_response
from src.utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/raw_responses/yandex_test_run.jsonl",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/parsed_responses/yandex_test_run_parsed.csv",
    )
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input_file
    output_path = PROJECT_ROOT / args.output_file

    records = read_jsonl(input_path)

    parsed_rows = []
    for record in records:
        parsed = parse_model_response(record.get("raw_response", ""))

        parsed_dict = asdict(parsed)

        source_present = bool(parsed.source_authors or parsed.source_title or parsed.source_year)
        quote_present = bool(parsed.quote_text)

        if parsed.status == "answered" and (not source_present or not quote_present):
            parsed_dict["status"] = "partial"

        row = {
            "timestamp": record.get("timestamp", ""),
            "run_id": record.get("run_id", ""),
            "question_id": record.get("question_id", ""),
            "discipline": record.get("discipline", ""),
            "question": record.get("question", ""),
            "source_ref": record.get("source_ref", ""),
            "model": record.get("model", ""),
            "provider": record.get("provider", ""),
            "model_name": record.get("model_name", ""),
            "prompt_type": record.get("prompt_type", ""),
            "raw_status": record.get("raw_status", ""),
            "error": record.get("error", ""),
            **parsed_dict,
            "locator_present": bool(parsed.source_locator),
            "source_present": source_present,
            "quote_present": quote_present,
            "raw_response": record.get("raw_response", ""),
        }

        parsed_rows.append(row)

    df = pd.DataFrame(parsed_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )

    print(f"Parsed records: {len(df)}")
    print(f"Saved to: {output_path}")

    if "parse_error" in df.columns:
        errors = df[df["parse_error"].astype(str).str.len() > 0]
        print(f"Parse errors: {len(errors)}")

        if len(errors) > 0:
            print(
                errors[["question_id", "prompt_type", "parse_error"]]
                .head(10)
                .to_string(index=False)
            )


if __name__ == "__main__":
    main()
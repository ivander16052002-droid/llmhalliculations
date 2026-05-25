from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

from src.benchmark import load_benchmark
from src.config import PROJECT_ROOT
from src.model_clients.ollama_client import OllamaClient
from src.prompts import load_prompts, build_prompt
from src.utils import append_jsonl


def get_field(item: Any, name: str, default: str = "") -> str:
    """
    Safely get field from BenchmarkQuestion object or dict.
    """
    if isinstance(item, dict):
        return item.get(name, default)

    return getattr(item, name, default)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark-file",
        type=str,
        default="data/benchmark/benchmark_questions.csv",
        help="Path to benchmark CSV file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of benchmark questions.",
    )
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        default=["baseline", "provoking", "conservative"],
        help="Prompt types to run.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run id for output tracking.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="qwen2.5:7b",
        help="Ollama model name, e.g. qwen2.5:7b.",
    )

    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "qwen_quote_full_%Y%m%d_%H%M%S"
    )

    output_path = PROJECT_ROOT / "data" / "raw_responses" / f"{run_id}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    benchmark = load_benchmark(args.benchmark_file)
    prompts = load_prompts()
    client = OllamaClient(model_name=args.model_name)

    if args.limit is not None:
        benchmark = benchmark[: args.limit]

    total = len(benchmark) * len(args.prompt_types)
    counter = 0

    print(f"Run id: {run_id}")
    print(f"Model name: {args.model_name}")
    print(f"Questions: {len(benchmark)}")
    print(f"Prompt types: {args.prompt_types}")
    print(f"Total requests: {total}")
    print(f"Output: {output_path}")

    for item_index, item in enumerate(benchmark, start=1):
        question = get_field(item, "question")

        question_id = (
            get_field(item, "question_id")
            or get_field(item, "id")
            or get_field(item, "n")
            or get_field(item, "number")
            or str(item_index)
        )

        discipline = get_field(item, "discipline")
        source_ref = get_field(item, "source_ref") or get_field(item, "source") or ""

        for prompt_type in args.prompt_types:
            counter += 1

            prompt = build_prompt(
                question=question,
                prompt_type=prompt_type,
                prompts=prompts,
            )

            print(
                f"[{counter}/{total}] "
                f"question_id={question_id} "
                f"prompt_type={prompt_type}"
            )

            response = client.generate(prompt)

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "question_id": question_id,
                "discipline": discipline,
                "question": question,
                "source_ref": source_ref,
                "model": "qwen",
                "provider": "ollama",
                "model_name": args.model_name,
                "prompt_type": prompt_type,
                "prompt": prompt,
                "raw_response": response.text,
                "raw_status": response.raw_status,
                "raw_metadata": response.raw_metadata,
                "error": "" if response.raw_status == "FINAL" else response.raw_status,
            }

            append_jsonl(output_path, record)

    print("Done.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
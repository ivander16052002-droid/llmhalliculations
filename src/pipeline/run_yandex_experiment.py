from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from src.benchmark import load_benchmark
from src.config import PROJECT_ROOT, load_env, load_yaml
from src.model_clients.yandex_client import YandexClient
from src.prompts import build_prompt, load_prompts
from src.utils import append_jsonl, make_run_id, utc_now_iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3, help="How many questions to run")
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        default=["baseline"],
        help="Prompt types to run: baseline provoking conservative",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output jsonl file. If not provided, a file with run_id will be created.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id. If not provided, it will be generated automatically.",
    )
    args = parser.parse_args()

    run_id = args.run_id or make_run_id("yandex_test")

    if args.output_file is None:
        args.output_file = f"data/raw_responses/{run_id}.jsonl"

    load_env()

    experiment_config = load_yaml("configs/experiment.yaml")
    models_config = load_yaml("configs/models.yaml")
    prompts = load_prompts("configs/prompts.yaml")

    questions = load_benchmark(experiment_config["benchmark_path"])
    questions = questions[: args.limit]

    yandex_model_config = models_config["models"]["yandex"]
    client = YandexClient(model_name=yandex_model_config["model_name"])

    generation_config = experiment_config["generation"]
    max_tokens = int(generation_config["max_tokens"])
    temperature = float(generation_config["temperature"])

    output_path = PROJECT_ROOT / args.output_file

    total = len(questions) * len(args.prompt_types)
    progress = tqdm(total=total, desc="Yandex test run")

    for question in questions:
        for prompt_type in args.prompt_types:
            prompt = build_prompt(
                question=question.question,
                prompt_type=prompt_type,
                prompts=prompts,
            )

            try:
                result = client.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                record = {
                    "run_id": run_id,
                    "timestamp": utc_now_iso(),
                    "question_id": question.id,
                    "discipline": question.discipline,
                    "question": question.question,
                    "source_ref": question.source_ref,
                    "model": "yandex",
                    "provider": result.provider,
                    "model_name": result.model_name,
                    "prompt_type": prompt_type,
                    "prompt": prompt,
                    "raw_response": result.text,
                    "raw_status": result.raw_status,
                    "raw_metadata": result.raw_metadata,
                    "error": "",
                }

            except Exception as exc:
                record = {
                    "run_id": run_id,
                    "timestamp": utc_now_iso(),
                    "question_id": question.id,
                    "discipline": question.discipline,
                    "question": question.question,
                    "source_ref": question.source_ref,
                    "model": "yandex",
                    "provider": "yandex",
                    "model_name": yandex_model_config["model_name"],
                    "prompt_type": prompt_type,
                    "prompt": prompt,
                    "raw_response": "",
                    "raw_status": "",
                    "raw_metadata": {},
                    "error": repr(exc),
                }

            append_jsonl(output_path, record)
            progress.update(1)

    progress.close()
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
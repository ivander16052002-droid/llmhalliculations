from __future__ import annotations

import argparse
import json

from src.benchmark import get_question_by_id, load_benchmark
from src.config import ensure_dir, load_env, load_yaml, PROJECT_ROOT
from src.model_clients.yandex_client import YandexClient
from src.prompts import build_prompt, load_prompts
from src.utils import append_jsonl, utc_now_iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-id", type=int, default=1)
    parser.add_argument("--prompt-type", type=str, default="baseline")
    parser.add_argument("--output-file", type=str, default="data/raw_responses/yandex_smoke_test.jsonl")
    args = parser.parse_args()

    load_env()

    experiment_config = load_yaml("configs/experiment.yaml")
    models_config = load_yaml("configs/models.yaml")

    benchmark_path = experiment_config["benchmark_path"]
    questions = load_benchmark(benchmark_path)
    question = get_question_by_id(questions, args.question_id)

    prompts = load_prompts("configs/prompts.yaml")
    prompt = build_prompt(
        question=question.question,
        prompt_type=args.prompt_type,
        prompts=prompts,
    )

    yandex_model_config = models_config["models"]["yandex"]
    client = YandexClient(model_name=yandex_model_config["model_name"])

    generation_config = experiment_config["generation"]
    result = client.generate(
        prompt=prompt,
        max_tokens=int(generation_config["max_tokens"]),
        temperature=float(generation_config["temperature"]),
    )

    record = {
        "timestamp": utc_now_iso(),
        "question_id": question.id,
        "discipline": question.discipline,
        "question": question.question,
        "source_ref": question.source_ref,
        "model": "yandex",
        "provider": result.provider,
        "model_name": result.model_name,
        "prompt_type": args.prompt_type,
        "prompt": prompt,
        "raw_response": result.text,
        "raw_status": result.raw_status,
        "raw_metadata": result.raw_metadata,
    }

    output_path = PROJECT_ROOT / args.output_file
    append_jsonl(output_path, record)

    print("\n=== QUESTION ===")
    print(question.question)

    print("\n=== PROMPT TYPE ===")
    print(args.prompt_type)

    print("\n=== RAW RESPONSE ===")
    print(result.text)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
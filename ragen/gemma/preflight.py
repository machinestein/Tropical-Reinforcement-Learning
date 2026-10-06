"""Fail before allocating workers if the separate Gemma runtime is incomplete."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path


def check(model):
    expected = {"transformers": "5.6.0", "vllm": "0.19.1", "torch": "2.10.0"}
    for package, required in expected.items():
        actual = version(package).split("+")[0]
        if actual != required:
            raise RuntimeError(
                f"Gemma runtime requires {package}=={required}; found {actual}. Activate gemma_venv."
            )
    path = Path(model)
    if not (path / "ragen_gemma_export.json").is_file():
        raise RuntimeError(
            "Prepare Gemma's text checkpoint first: python -m ragen.gemma.prepare"
        )
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "gemma4_text":
        raise ValueError("Expected the prepared Gemma 4 text checkpoint")
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        Gemma4TextConfig,
    )
    from ragen.llm_agent.gemma import is_gemma_tokenizer

    if not is_gemma_tokenizer(
        AutoTokenizer.from_pretrained(path, local_files_only=True)
    ):
        raise ValueError("Gemma 4 turn tokens are missing from the tokenizer")
    if Gemma4TextConfig not in AutoModelForTokenClassification._model_mapping:
        raise RuntimeError(
            "Gemma critic is not registered; rerun scripts/setup_gemma_venv.sh"
        )
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout  # noqa: F401
    from ragen.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker  # noqa: F401

    print(f"Gemma runtime/model preflight passed: {path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=".cache/gemma/gemma-4-E4B-it")
    check(parser.parse_args().model)


if __name__ == "__main__":
    main()

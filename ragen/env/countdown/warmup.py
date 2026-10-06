"""Shared atomic-action warmup for the opt-in matched Countdown comparison.

Fine-tune on independently generated two-input arithmetic, then export a full
HF checkpoint so every RAGEN method starts from identical weights.
This is an interface warmup, not a reproduction of the paper's released adapter.
"""

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .tropic_env import OPERATORS, CountdownStepEnv


def atomic_config():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[3] / "config"), version_base=None):
        return compose(config_name="_4_countdown_atomic")


def interface_fingerprint(config):
    payload = {
        "version": 1,
        "instruction": config.custom_envs.Countdown.env_instruction,
        "max_tokens": config.custom_envs.Countdown.max_tokens,
        "agent": OmegaConf.to_container(config.agent_proxy, resolve=True),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def atomic_corpus(seed=2026):
    problems = defaultdict(list)
    for a in range(1, 21):
        for b in range(1, 21):
            for op, operation in OPERATORS.items():
                target = operation(Fraction(a), Fraction(b))
                if target is not None and target.denominator == 1 and 1 <= target <= 100:
                    problems[(tuple(sorted((a, b))), int(target))].append(f"{a} {op} {b}")
    keys = sorted(problems)
    rng = random.Random(seed)
    rng.shuffle(keys)
    split = max(1, len(keys) // 10)

    def examples(selected):
        rows = [{"nums": list(nums), "target": target, "action": action}
                for nums, target in selected for action in sorted(set(problems[(nums, target)]))]
        rng.shuffle(rows)
        return rows

    return examples(keys[split:]), examples(keys[:split])


def encode_example(context, row, depth=0):
    from .config import CountdownEnvConfig

    env = CountdownStepEnv.__new__(CountdownStepEnv)
    env.config = CountdownEnvConfig(max_steps=5)
    env.target = row["target"]
    env.remaining = sorted(Fraction(v) for v in row["nums"])
    history = [{} for _ in range(depth)] + [{"state": env.render(), "actions_left": 5 - depth}]
    batch = context.get_lm_inputs([{"env_id": 0, "group_id": 0, "history": history}], prepare_for_update=False)
    prompt = batch.batch["input_ids"][0][batch.batch["attention_mask"][0].bool()].tolist()
    completion = context.tokenizer.encode(row["action"] + "</answer>", add_special_tokens=False)
    completion.append(context.tokenizer.eos_token_id)
    return {"input_ids": prompt + completion, "attention_mask": [1] * (len(prompt) + len(completion)),
            "labels": [-100] * len(prompt) + completion}


def check_checkpoint(model_path, base_model, config=None):
    directory = Path(model_path)
    metadata = directory.parent / "warmup_manifest.json"
    if not (directory.parent / "COMPLETED").is_file() or not metadata.is_file():
        raise ValueError("Warmup checkpoint is incomplete; use a new EXPERIMENT, not an unfinished warmup")
    manifest = json.loads(metadata.read_text())
    if manifest["base_model"] != base_model or manifest["interface"] != interface_fingerprint(config or atomic_config()):
        raise ValueError("Warmup base model or Countdown interface differs from this run")
    if not (directory / "config.json").is_file() or not list(directory.glob("*.safetensors")):
        raise ValueError("Warmup must contain a full HF model, not only an adapter")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-model", type=Path)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    config = atomic_config()
    if args.check_model:
        check_checkpoint(args.check_model, args.model, config)
        print(f"Shared Countdown warmup checkpoint verified: {args.check_model}")
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.output is None or args.steps < 1 or args.micro_batch_size < 1:
        parser.error("--output and positive training steps/microbatch are required")
    if 128 % (world_size * args.micro_batch_size):
        parser.error("Warmup global batch 128 must divide into GPU microbatches")
    if args.output.exists():
        parser.error("Refusing to overwrite a warmup directory; reuse its completed model or use a new EXPERIMENT")

    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Trainer, TrainerCallback, TrainingArguments, set_seed)
    from ragen.llm_agent.ctx_manager import ContextManager

    if not torch.cuda.is_available():
        raise RuntimeError("Countdown warmup requires CUDA")
    training_args = TrainingArguments(
        output_dir=str(args.output), max_steps=args.steps,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=128 // (world_size * args.micro_batch_size),
        learning_rate=1e-5, lr_scheduler_type="cosine", warmup_steps=1, optim="adamw_torch_fused",
        weight_decay=0.01, max_grad_norm=1.0, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, save_strategy="no", logging_steps=1,
        seed=args.seed, data_seed=args.seed, report_to=[] if os.environ.get("WANDB_MODE") == "disabled" else ["wandb"],
        run_name=args.output.name, label_names=["labels"],
    )
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    context = ContextManager(config, tokenizer)
    train, val = atomic_corpus(args.seed)
    train_data = Dataset.from_list([encode_example(context, row, i % 5) for i, row in enumerate(train)])
    val_data = Dataset.from_list([encode_example(context, row, i % 5) for i, row in enumerate(val)])
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False

    class JsonMetrics(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero:
                with (Path(args.output_dir) / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": state.global_step, "data": logs}) + "\n")

    trainer = Trainer(model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data,
                      processing_class=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer),
                      callbacks=[JsonMetrics()])
    trainer.train()
    validation = trainer.evaluate()
    if trainer.is_world_process_zero():
        trained = trainer.accelerator.unwrap_model(trainer.model)
        trained.config.use_cache = True
        trained.to(torch.bfloat16).save_pretrained(args.output / "model", safe_serialization=True)
        tokenizer.save_pretrained(args.output / "model")
        manifest = {"base_model": args.model, "interface": interface_fingerprint(config),
                    "steps": args.steps, "seed": args.seed, "global_batch": 128,
                    "lr": 1e-5, "training": "full_weight", "compute_dtype": "bfloat16",
                    "train_rows": len(train), "validation_rows": len(val), "validation": validation}
        (args.output / "warmup_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (args.output / "COMPLETED").write_text("Shared full warmup checkpoint saved.\n")
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()

"""Export Google's text weights unchanged for native HF actor/critic and vLLM.

Usage: python -m ragen.gemma.prepare --output .cache/gemma/gemma-4-E4B-it
"""

import argparse
import json
from pathlib import Path
import re
import shutil

from huggingface_hub import snapshot_download, split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Gemma4TextConfig

MODEL = "google/gemma-4-E4B-it"
REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"


def export_text_checkpoint(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite {output}; choose a new output path"
        )
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "gemma4":
        raise ValueError("Expected the original Gemma 4 multimodal checkpoint")
    text_config = Gemma4TextConfig(**config["text_config"])
    text_config.architectures = ["Gemma4ForCausalLM"]
    text_config.eos_token_id = config.get("eos_token_id", text_config.eos_token_id)
    weights = {}
    ignored = []
    for file in sorted(source.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                if key.startswith("model.language_model."):
                    name = "model." + key.removeprefix("model.language_model.")
                elif key == "lm_head.weight":
                    name = key
                else:
                    ignored.append(key)
                    continue
                if name in weights:
                    raise ValueError(f"Duplicate tensor {name}")
                weights[name] = tensors.get_tensor(key)
    if not weights or "model.embed_tokens.weight" not in weights:
        raise ValueError("No Gemma text backbone found in the source checkpoint")

    import torch
    from transformers import Gemma4ForCausalLM

    with torch.device("meta"):
        reference = Gemma4ForCausalLM(text_config)
    expected = reference.state_dict()
    missing = set(expected) - set(weights)
    if text_config.tie_word_embeddings:
        missing.discard("lm_head.weight")
    unexpected = set(weights) - set(expected)
    unused_kv = {
        name
        for name in unexpected
        if any(
            re.search(pattern, name)
            for pattern in reference._keys_to_ignore_on_load_unexpected
        )
    }
    unexpected -= unused_kv
    if missing or unexpected:
        raise ValueError(
            f"Text checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for name, tensor in weights.items():
        if name in unused_kv:
            continue
        if tensor.shape != expected[name].shape:
            raise ValueError(
                f"Wrong shape for {name}: {tensor.shape} vs {expected[name].shape}"
            )

    output.mkdir(parents=True)
    text_config.save_pretrained(output)
    shard_info = split_torch_state_dict_into_shards(weights, max_shard_size="2GB")
    for filename, names in shard_info.filename_to_tensors.items():
        save_file(
            {name: weights[name].contiguous() for name in names},
            output / filename,
            metadata={"format": "pt"},
        )
    if shard_info.is_sharded:
        (output / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": shard_info.metadata,
                    "weight_map": shard_info.tensor_to_filename,
                },
                indent=2,
            )
        )
    metadata_files = {
        file
        for pattern in (
            "tokenizer*",
            "*token*.json",
            "*.jinja",
            "generation_config.json",
        )
        for file in source.glob(pattern)
        if file.is_file()
    }
    for file in metadata_files:
        shutil.copyfile(file, output / file.name)
    tokenizer_config = output / "tokenizer_config.json"
    if tokenizer_config.exists():
        tokenizer_data = json.loads(tokenizer_config.read_text())
        tokenizer_data.pop("processor_class", None)
        tokenizer_config.write_text(json.dumps(tokenizer_data, indent=2))
    (output / "ragen_gemma_export.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "model": MODEL,
                "revision": REVISION,
                "text_tensors": len(weights),
                "omitted_non_text_tensors": len(ignored),
                "shared_kv_tensors_retained_for_vllm": sorted(unused_kv),
                "weight_transformation": "prefix rename only; no quantization or numerical changes",
            },
            indent=2,
        )
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="Use an already downloaded local original checkpoint",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(".cache/gemma/gemma-4-E4B-it")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/gemma/hub"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; no files changed")
    source = args.source or snapshot_download(
        MODEL,
        revision=REVISION,
        cache_dir=str(args.cache_dir),
        allow_patterns=["*.safetensors", "*.json", "*.jinja", "tokenizer.model"],
    )
    print(
        f"Prepared text checkpoint: {export_text_checkpoint(source, args.output).resolve()}"
    )


if __name__ == "__main__":
    main()

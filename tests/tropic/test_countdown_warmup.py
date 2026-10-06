"""CPU-only contracts for the matched Countdown interface warmup."""

from fractions import Fraction
import json

import pytest

from conftest import CharacterTokenizer
from ragen.env.countdown.warmup import atomic_config, atomic_corpus, check_checkpoint, encode_example, interface_fingerprint
from ragen.env.countdown.tropic_env import OPERATORS
from ragen.llm_agent.ctx_manager import ContextManager


def test_atomic_corpus_is_deterministic_correct_and_semantically_disjoint():
    train, val = atomic_corpus()
    assert (train, val) == atomic_corpus()
    key = lambda row: (tuple(sorted(row["nums"])), row["target"])
    assert {key(row) for row in train}.isdisjoint(key(row) for row in val)
    assert all(len(row["nums"]) == 2 for row in train + val)
    for row in train + val:
        a, op, b = row["action"].split()
        assert sorted([int(a), int(b)]) == sorted(row["nums"])
        assert OPERATORS[op](Fraction(a), Fraction(b)) == row["target"]
    assert {row["action"].split()[1] for row in train} == set(OPERATORS)


@pytest.mark.parametrize("depth", range(5))
def test_warmup_supervises_exact_generated_suffix_with_prompt_masked(depth):
    config = atomic_config()
    tokenizer = CharacterTokenizer()
    context = ContextManager(config, tokenizer)
    row = {"nums": [5, 3], "target": 2, "action": "5 - 3"}
    encoded = encode_example(context, row, depth)
    boundary = next(i for i, label in enumerate(encoded["labels"]) if label != -100)
    prompt = tokenizer.decode(encoded["input_ids"][:boundary])
    assert prompt.endswith("<answer>")
    assert "Numbers: [3, 5]" in prompt
    assert f"Turn {depth + 1}" in prompt and f"{5 - depth} actions left" in prompt
    assert all(label == -100 for label in encoded["labels"][:boundary])
    assert encoded["labels"][boundary:] == encoded["input_ids"][boundary:]
    assert encoded["labels"][-1] == tokenizer.eos_token_id
    response = "<answer>" + tokenizer.decode(encoded["labels"][boundary:], skip_special_tokens=True)
    assert context._parse_response(response)[1] == [row["action"]]


def test_reuse_requires_complete_compatible_full_checkpoint(tmp_path):
    config = atomic_config()
    output = tmp_path / "warmup"
    model = output / "model"
    model.mkdir(parents=True)
    with pytest.raises(ValueError, match="incomplete"):
        check_checkpoint(model, "qwen-test", config)
    metadata = {"base_model": "qwen-test", "interface": interface_fingerprint(config)}
    (output / "warmup_manifest.json").write_text(json.dumps(metadata))
    (output / "COMPLETED").touch()
    with pytest.raises(ValueError, match="full HF model"):
        check_checkpoint(model, "qwen-test", config)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").touch()
    assert check_checkpoint(model, "qwen-test", config) == metadata
    with pytest.raises(ValueError, match="base model"):
        check_checkpoint(model, "different-model", config)
    config.custom_envs.Countdown.env_instruction += " changed"
    with pytest.raises(ValueError, match="interface"):
        check_checkpoint(model, "qwen-test", config)


def test_atomic_tokens_can_train_and_export_a_full_model_on_cpu(tmp_path):
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (AutoModelForCausalLM, DataCollatorForSeq2Seq, PreTrainedTokenizerFast,
                              Qwen2Config, Qwen2ForCausalLM, Trainer, TrainingArguments)

    vocabulary = {word: i for i, word in enumerate(
        ["[PAD]", "[UNK]", "<|im_start|>", "<|im_end|>", "<answer>", "</answer>", "2", "3", "5", "+", "-"])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                       eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>", "<answer>", "</answer>"])
    tokenizer.name_or_path = "qwen-test"
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    context = ContextManager(atomic_config(), tokenizer)
    data = Dataset.from_list([encode_example(context, row) for row in [
        {"nums": [2, 3], "target": 5, "action": "2 + 3"},
        {"nums": [5, 3], "target": 2, "action": "5 - 3"},
    ]])
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                       max_position_embeddings=1024, eos_token_id=tokenizer.eos_token_id))
    trainer = Trainer(model=model, train_dataset=data, processing_class=tokenizer,
                      data_collator=DataCollatorForSeq2Seq(tokenizer),
                      args=TrainingArguments(output_dir=str(tmp_path / "train"), use_cpu=True, max_steps=1,
                                             per_device_train_batch_size=2, report_to=[], save_strategy="no",
                                             label_names=["labels"], disable_tqdm=True))
    result = trainer.train()
    assert torch.isfinite(torch.tensor(result.training_loss))
    trained = trainer.accelerator.unwrap_model(trainer.model)
    trained.to(torch.bfloat16).save_pretrained(tmp_path / "model")
    tokenizer.save_pretrained(tmp_path / "model")
    restored = AutoModelForCausalLM.from_pretrained(tmp_path / "model", local_files_only=True)
    assert not any("lora_" in key for key in restored.state_dict())

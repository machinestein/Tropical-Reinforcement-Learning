"""Gemma chat masks with an offline tokenizer and optional Google's tokenizer."""

import os

import pytest
import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ragen.llm_agent.ctx_manager import ContextManager, get_masks_and_scores
from ragen.llm_agent.gemma import GEMMA_SPECIAL_TOKENS, is_gemma_tokenizer
from ragen.llm_agent.phi import is_response_end


@pytest.fixture(params=["offline", "google"])
def tokenizer(request):
    if request.param == "google":
        path = os.environ.get("RAGEN_TEST_GEMMA_TOKENIZER")
        if not path:
            pytest.skip("Set RAGEN_TEST_GEMMA_TOKENIZER for Google's tokenizer")
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    special = [*GEMMA_SPECIAL_TOKENS, "<pad>", "<|tool_response>"]
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["Rules State Plan Up Down model user system\n"],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=special,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    result = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        additional_special_tokens=special,
    )
    result.chat_template = (
        "{{ bos_token }}{% for m in messages %}{{ '<|turn>' + ('model' if m.role == 'assistant' else m.role)"
        " + '\n' + m.content + '<turn|>\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|turn>model\n' }}{% endif %}"
    )
    return result


def context(tokenizer, mode="full", response_mask=True):
    return ContextManager(
        OmegaConf.create(
            {
                "agent_proxy": {
                    "context_window_mode": mode,
                    "max_context_window": -1,
                    "enable_think": True,
                    "action_sep": "||",
                    "max_actions_per_turn": 1,
                },
                "enable_response_mask": response_mask,
                "es_manager": {
                    "train": {
                        "env_configs": {"n_groups": [2], "tags": ["sokoban"]},
                        "group_size": 1,
                    }
                },
                "custom_envs": {
                    "sokoban": {"env_type": "sokoban", "max_actions_per_traj": 5}
                },
                "actor_rollout_ref": {
                    "rollout": {"response_length": 32, "max_model_len": 1024}
                },
            }
        ),
        tokenizer,
    )


def turns(responses):
    messages = [{"role": "system", "content": "Rules"}]
    for response in responses:
        messages.extend(
            [
                {"role": "user", "content": "State"},
                {"role": "assistant", "content": response},
            ]
        )
    return messages


@pytest.mark.parametrize("mode", ["full", "single_turn", "limited_multi_turn"])
def test_masks_exclude_headers_padding_and_template_newlines(tokenizer, mode):
    ctx = context(tokenizer, mode)
    rows = [["Plan", "Up"], ["Down"]]
    batch = tokenizer(
        [ctx._chat_template(turns(row), False) for row in rows],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    loss, response = ctx._compute_loss_mask(batch.input_ids, batch.attention_mask, mode)
    assert torch.equal(loss, response)
    for i, row in enumerate(rows):
        expected = row if mode == "full" else row[-1:]
        assert tokenizer.decode(batch.input_ids[i, 1:][response[i].bool()]) == "".join(
            t + "<turn|>" for t in expected
        )
    assert not (response.bool() & ~batch.attention_mask[:, 1:].bool()).any()


@pytest.mark.parametrize("per_turn", [False, True])
def test_rewards_align_to_generated_turn_end(tokenizer, per_turn):
    ctx = context(tokenizer)
    batch = tokenizer(
        [ctx._chat_template(turns(row), False) for row in [["Plan", "Up"], ["Down"]]],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    scores, loss, _ = get_masks_and_scores(
        batch.input_ids,
        tokenizer,
        [[0.2, 0.8], [-0.5]],
        per_turn,
        True,
        batch.attention_mask,
    )
    torch.testing.assert_close(scores.sum(-1), torch.tensor([1.0, -0.5]))
    assert not (scores.ne(0) & ~loss.bool()).any()
    assert torch.all(
        batch.input_ids[:, 1:][scores.ne(0)]
        == tokenizer.convert_tokens_to_ids("<turn|>")
    )


def test_gemma_template_and_stop_tokens(tokenizer):
    assert is_gemma_tokenizer(tokenizer)
    ctx = context(tokenizer)
    text = ctx._render_messages(
        turns([]) + [{"role": "user", "content": "State"}], True
    )
    assert text.endswith("<|turn>model\n<think>")
    assert "<|think|>" not in text
    for token in ("<eos>", "<turn|>", "<|tool_response>"):
        assert is_response_end(tokenizer, tokenizer.convert_tokens_to_ids(token))
    assert not is_response_end(tokenizer, tokenizer.pad_token_id)


def test_unmasked_context_still_excludes_system_and_padding(tokenizer):
    ctx = context(tokenizer, response_mask=False)
    tokens = tokenizer(
        ctx._chat_template(turns(["Up"]), False),
        return_tensors="pt",
        add_special_tokens=False,
    )
    loss, response = ctx._compute_loss_mask(
        tokens.input_ids, tokens.attention_mask, "full"
    )
    text = tokenizer.decode(tokens.input_ids[0, 1:][loss[0].bool()])
    assert "Rules" not in text and "State" in text and "Up" in text
    assert response.sum() < loss.sum()


def test_eight_k_context_trims_old_turns_and_preserves_current_page(tokenizer):
    ctx = context(tokenizer)
    ctx.config.actor_rollout_ref.rollout.max_model_len = 8192
    ctx.config.actor_rollout_ref.rollout.response_length = 400
    system = {"role": "system", "content": "Follow the shopping instruction."}
    current = {"role": "user", "content": "Current page: requested blue shoes. Available action: click[buy now]"}
    messages = [system]
    for turn in range(10):
        messages.extend([
            {"role": "user", "content": f"Old page {turn}: " + "product description " * 1500},
            {"role": "assistant", "content": "<think>Inspect the product.</think><answer>click[features]</answer>"},
        ])
    messages.append(current)
    original = [dict(message) for message in messages]
    rendered = ctx._render_messages(messages, True)
    assert len(tokenizer(rendered, add_special_tokens=False)["input_ids"]) > 8192
    trimmed = ctx._apply_max_length(messages, add_generation_prompt=True)
    assert messages == original
    assert trimmed[0] == system and trimmed[-1] == current
    assert len(trimmed) < len(messages)
    prompt_tokens = len(tokenizer(ctx._render_messages(trimmed, True), add_special_tokens=False)["input_ids"])
    assert prompt_tokens + 400 <= 8192
    completed = trimmed + [{"role": "assistant", "content": "<answer>click[buy now]</answer>"}]
    training = ctx._apply_max_length(completed, add_generation_prompt=False)
    assert len(tokenizer(ctx._render_messages(training, False), add_special_tokens=False)["input_ids"]) <= 8192

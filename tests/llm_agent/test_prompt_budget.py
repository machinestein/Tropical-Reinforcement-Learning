"""Prompt truncation must leave room for the response, or vLLM asserts mid-generation."""

import copy
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from ragen.llm_agent.ctx_manager import ContextManager


class WordTokenizer:
    """One token per whitespace-separated word; enough to exercise length accounting."""
    name_or_path = "qwen"

    def __init__(self):
        self.vocab = {}

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        text = "\n".join(f"<|im_start|> {m['role']}: {m['content']} <|im_end|>" for m in messages)
        return text + ("\n<|im_start|> assistant:\n" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            return {"input_ids": self.encode(text)}
        rows = [self.encode(item) for item in text]
        width = max(map(len, rows))
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, -len(row):] = torch.tensor(row)
            mask[i, -len(row):] = 1
        return SimpleNamespace(input_ids=ids, attention_mask=mask)

    def encode(self, text):
        return [self.vocab.setdefault(word, len(self.vocab) + 1) for word in text.split()]


def make_ctx(max_model_len, response_length, env_max_tokens=None):
    cfg = OmegaConf.create({
        "agent_proxy": {"max_context_window": -1, "enable_think": False, "use_turn_scores": False,
                        "action_sep": "||", "reward_normalization": {"grouping": "batch", "method": "identity"}},
        "enable_response_mask": False,
        "es_manager": {"train": {"env_configs": {"n_groups": [1], "tags": ["sokoban"]}, "group_size": 1}},
        "custom_envs": {"sokoban": {"env_type": "sokoban", "max_actions_per_traj": 10}},
        "actor_rollout_ref": {"rollout": {"response_length": response_length, "max_model_len": max_model_len}},
    })
    ctx = ContextManager(config=cfg, tokenizer=WordTokenizer(), mode="train")
    ctx.env_config_lookup = {0: {"max_tokens": env_max_tokens or response_length}}
    return ctx


def long_conversation(turns, words_per_turn):
    messages = [{"role": "system", "content": "sys"}]
    for t in range(turns):
        messages.append({"role": "user", "content": " ".join(f"u{t}w{i}" for i in range(words_per_turn))})
        messages.append({"role": "assistant", "content": f"a{t}"})
        messages.append({"role": "user", "content": "Reward: 0"})
    messages.append({"role": "user", "content": "final turn"})
    return messages


def token_count(ctx, messages, add_generation_prompt):
    text = ctx.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)
    if add_generation_prompt:
        text += "<think>" if ctx.config.agent_proxy.enable_think else "<answer>"
    return len(ctx.tokenizer(text)["input_ids"])


def test_budget_reserves_response_tokens():
    assert make_ctx(8096, 512)._prompt_token_budget() == 8096 - 512
    assert make_ctx(8096, 128, env_max_tokens=512)._prompt_token_budget() == 8096 - 128
    assert make_ctx(None, 512)._prompt_token_budget() is None


def test_budget_rejects_context_smaller_than_response():
    with pytest.raises(ValueError):
        make_ctx(256, 512)._prompt_token_budget()


def test_truncated_prompt_plus_response_fits_model_context():
    ctx = make_ctx(max_model_len=200, response_length=60)
    messages = long_conversation(turns=12, words_per_turn=20)
    original = copy.deepcopy(messages)
    assert token_count(ctx, messages, True) > 200
    truncated = ctx._apply_max_length(messages, add_generation_prompt=True)
    assert truncated[0]["role"] == "system"
    assert truncated[-1]["content"] == "final turn"
    assert token_count(ctx, truncated, True) + 60 <= 200
    assert messages == original


def test_short_prompts_are_untouched():
    ctx = make_ctx(max_model_len=2000, response_length=60)
    messages = long_conversation(turns=2, words_per_turn=5)
    assert ctx._apply_max_length(messages, add_generation_prompt=True) == messages


@pytest.mark.parametrize("enable_think", [False, True])
def test_generation_prefix_counts_at_exact_boundary(enable_think):
    ctx = make_ctx(200, 60)
    ctx.config.agent_proxy.enable_think = enable_think
    messages = long_conversation(1, 5)
    size = token_count(ctx, messages, True)
    ctx.config.actor_rollout_ref.rollout.max_model_len = size + 60
    assert ctx._apply_max_length(messages, True) == messages
    ctx.config.actor_rollout_ref.rollout.max_model_len -= 1
    truncated = ctx._apply_max_length(messages, True)
    assert truncated != messages
    assert token_count(ctx, truncated, True) <= size - 1


def test_completed_samples_do_not_reserve_another_response():
    ctx = make_ctx(200, 60)
    messages = long_conversation(2, 10)
    messages.append({"role": "assistant", "content": "finished response"})
    size = token_count(ctx, messages, False)
    ctx.config.actor_rollout_ref.rollout.max_model_len = size
    assert ctx._apply_max_length(messages, False) == messages


@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_reward_merged_with_current_observation_is_preserved(add_generation_prompt):
    ctx = make_ctx(200, 60)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old state " * 100},
        {"role": "assistant", "content": "old response"},
        {"role": "user", "content": "Reward: 0\nTurn 2:\nState: current proof state"},
    ]
    if not add_generation_prompt:
        messages.append({"role": "assistant", "content": "current response"})
    truncated = ctx._apply_max_length(messages, add_generation_prompt)
    assert truncated == [messages[0]] + messages[3:]


def test_oversized_current_observation_fails_before_generation():
    ctx = make_ctx(200, 60)
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "proof state " * 200}]
    original = copy.deepcopy(messages)
    with pytest.raises(ValueError, match="current state was not truncated"):
        ctx._apply_max_length(messages, True)
    assert messages == original


def test_lean_long_observation_fits_explicit_context_increase_without_truncation():
    ctx = make_ctx(8096, 512)
    messages = [{"role": "system", "content": "Lean tactics"},
                {"role": "user", "content": "proof state"}]
    padding = 7777 - token_count(ctx, messages, True)
    messages[-1]["content"] += " goal" * padding
    original = copy.deepcopy(messages)
    assert token_count(ctx, messages, True) == 7777
    with pytest.raises(ValueError, match="require 7777 tokens, exceeding the prompt budget of 7584"):
        ctx._apply_max_length(messages, True)
    ctx.config.actor_rollout_ref.rollout.max_model_len = 16384
    assert ctx._apply_max_length(messages, True) == original
    assert token_count(ctx, messages, True) + 512 <= 16384
    assert messages == original


@pytest.mark.parametrize("context_window_mode", ["full", "limited_multi_turn", "single_turn"])
@pytest.mark.parametrize("enable_think", [False, True])
def test_actual_inference_tokens_fit_budget(context_window_mode, enable_think):
    ctx = make_ctx(200, 60)
    ctx.config.agent_proxy.context_window_mode = context_window_mode
    ctx.config.agent_proxy.enable_think = enable_think
    ctx.prefix_lookup = {0: "system instruction"}
    history = [{"state": f"old{i} " * 50, "llm_response": "<answer> Up </answer>",
                "reward": 0, "actions_left": 10 - i} for i in range(6)]
    history.append({"state": "current proof state", "actions_left": 4})
    env_outputs = [{"env_id": 0, "group_id": 0, "history": history}]
    original = copy.deepcopy(env_outputs)

    result = ctx.get_lm_inputs(env_outputs, prepare_for_update=False)

    assert int(result.batch["attention_mask"].sum()) + 60 <= 200
    assert "current proof state" in str(result.non_tensor_batch["messages_list"])
    assert env_outputs == original


def test_single_turn_training_uses_full_context_budget():
    ctx = make_ctx(200, 60)
    ctx.prefix_lookup = {0: "system instruction"}
    history = [{"state": f"state{i}", "llm_response": "<answer> Up </answer>",
                "reward": 0, "actions_left": 10 - i} for i in range(2)]
    env_output = {"env_id": 0, "group_id": 0, "history": history}
    kwargs = dict(env_output=env_output, history=history, turn_idx=1, history_start=0,
                  turn_offset=0, include_warning=False, include_assistant=True)
    messages = ctx._build_single_turn_messages(**kwargs)
    ctx.config.actor_rollout_ref.rollout.max_model_len = token_count(ctx, messages, False)
    assert ctx._fit_single_turn_history_start_to_max_len(**kwargs, add_generation_prompt=False) == 0


@pytest.fixture(scope="module")
def cached_qwen_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer is not cached; this regression test never downloads it")


@pytest.mark.parametrize("context_window_mode", ["full", "limited_multi_turn", "single_turn"])
@pytest.mark.parametrize("enable_think", [False, True])
def test_qwen_long_history_with_lean_context_limit(cached_qwen_tokenizer, context_window_mode, enable_think):
    ctx = make_ctx(8096, 512)
    ctx.tokenizer = cached_qwen_tokenizer
    ctx.config.agent_proxy.context_window_mode = context_window_mode
    ctx.config.agent_proxy.enable_think = enable_think
    ctx.prefix_lookup = {0: "Prove the theorem using Lean tactics."}
    history = [{"state": "theorem example_goal : 1 + 1 = 2 := by\n" + "-- tactic feedback\n" * 200,
                "llm_response": "<think> simplify </think><answer> norm_num </answer>",
                "reward": 0, "actions_left": 30 - i} for i in range(15)]
    history.append({"state": "CURRENT_GOAL: 1 + 1 = 2", "actions_left": 15})
    result = ctx.get_lm_inputs([{"env_id": 0, "group_id": 0, "history": history}], False)
    prompt_ids = result.batch["input_ids"][0][result.batch["attention_mask"][0].bool()].tolist()
    rendered = ctx._render_messages(result.non_tensor_batch["messages_list"][0], True)

    assert len(prompt_ids) + 512 <= 8096
    assert "CURRENT_GOAL" in rendered
    assert prompt_ids == ctx.tokenizer(rendered, add_special_tokens=False)["input_ids"]

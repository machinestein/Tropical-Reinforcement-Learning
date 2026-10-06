"""Phi boundaries, reward alignment and unchanged legacy Qwen behavior.

Set RAGEN_TEST_PHI_TOKENIZER to a local Phi tokenizer snapshot to repeat the
format tests with Microsoft's actual tokenizer; normal CI stays offline.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ragen.llm_agent.ctx_manager import ContextManager, get_masks_and_scores
from ragen.llm_agent.phi import (
    PHI_SPECIAL_TOKENS, is_phi_model_path, is_phi_tokenizer, is_response_end, phi_masks,
)
from ragen.tropic.batch import make_batch
from verl.utils.torch_functional import get_response_mask


@pytest.fixture(scope="module", params=["synthetic", "real"])
def phi_tokenizer(request):
    if request.param == "real":
        path = os.environ.get("RAGEN_TEST_PHI_TOKENIZER")
        if not path:
            pytest.skip("Set RAGEN_TEST_PHI_TOKENIZER for local real-tokenizer checks")
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(["Rules State Plan Up Down Left Right"], trainers.BpeTrainer(
        vocab_size=300, special_tokens=list(PHI_SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    result = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, eos_token="<|endoftext|>", pad_token="<|endoftext|>",
        additional_special_tokens=list(PHI_SPECIAL_TOKENS[:-1]))
    result.name_or_path = "/renamed/local/checkpoint"
    result.chat_template = (
        "{% for message in messages %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %}")
    return result


def context(tokenizer, mode="full", response_mask=True):
    cfg = OmegaConf.create({
        "agent_proxy": {
            "context_window_mode": mode, "max_context_window": -1, "enable_think": True,
            "use_turn_scores": False, "action_sep": "||", "max_actions_per_turn": 1,
            "reward_normalization": {"grouping": "batch", "method": "identity"},
        },
        "enable_response_mask": response_mask,
        "es_manager": {"train": {"env_configs": {"n_groups": [2], "tags": ["sokoban"]}, "group_size": 1}},
        "custom_envs": {"sokoban": {"env_type": "sokoban", "max_actions_per_traj": 5}},
        "actor_rollout_ref": {"rollout": {"response_length": 128, "max_model_len": 10000}},
    })
    return ContextManager(cfg, tokenizer)


def messages(responses):
    result = [{"role": "system", "content": "Rules"}]
    for text in responses:
        result.extend([{"role": "user", "content": "State"}, {"role": "assistant", "content": text}])
    return result


def encoded(ctx, rows):
    return ctx.tokenizer([ctx._chat_template(messages(row), False) for row in rows],
                         return_tensors="pt", padding=True, padding_side="left")


@pytest.mark.parametrize("mode", ["full", "single_turn", "limited_multi_turn"])
def test_only_intended_assistant_tokens_are_selected(phi_tokenizer, mode):
    ctx = context(phi_tokenizer, mode)
    rows = [["Plan", "Up"], ["Right"]]
    tokens = encoded(ctx, rows)
    loss, response = ctx._compute_loss_mask(tokens.input_ids, tokens.attention_mask, mode)
    assert torch.equal(loss, response)
    for i, row in enumerate(rows):
        expected = row if mode == "full" else row[-1:]
        text = phi_tokenizer.decode(tokens.input_ids[i, 1:][response[i].bool()])
        assert text == "".join(item + "<|end|>" for item in expected)
    assert not (loss.bool() & ~tokens.attention_mask[:, 1:].bool()).any()


@pytest.mark.parametrize("turn_scores", [False, True])
def test_rewards_land_on_selected_end_tokens_and_do_not_leak_between_episodes(phi_tokenizer, turn_scores):
    ctx = context(phi_tokenizer)
    tokens = encoded(ctx, [["Plan", "Up"], ["Right"]])
    scores, loss, _ = get_masks_and_scores(
        tokens.input_ids, phi_tokenizer, [[.25, .75], [-.5]], turn_scores, True, tokens.attention_mask)
    assert scores.sum(-1).tolist() == [1., -.5]
    assert not (scores.ne(0) & ~loss.bool()).any()
    assert torch.all(tokens.input_ids[:, 1:][scores.ne(0)] == phi_tokenizer.convert_tokens_to_ids("<|end|>"))
    assert scores[0][scores[0].ne(0)].tolist() == ([.25, .75] if turn_scores else [1.])
    assert scores[1][scores[1].ne(0)].tolist() == [-.5]


def test_truncated_history_uses_retained_reward_suffix(phi_tokenizer):
    ctx = context(phi_tokenizer)
    tokens = encoded(ctx, [["Up"]])
    scores, _, _ = get_masks_and_scores(tokens.input_ids, phi_tokenizer,
                                       [[.25, .75]], True, True, tokens.attention_mask)
    assert scores.sum().item() == .75


def test_unmasked_mode_includes_observations_but_never_system_or_padding(phi_tokenizer):
    ctx = context(phi_tokenizer)
    tokens = encoded(ctx, [["Plan", "Up"], ["Right"]])
    loss, response = phi_masks(tokens.input_ids, phi_tokenizer, tokens.attention_mask,
                               enable_response_mask=False)
    for i in range(2):
        decoded = phi_tokenizer.decode(tokens.input_ids[i, 1:][loss[i].bool()])
        assert decoded.startswith("<|user|>State<|end|><|assistant|>")
        assert "Rules" not in decoded and "<|endoftext|>" not in decoded
    assert torch.all(loss >= response)


def test_padding_equal_to_eos_does_not_erase_a_real_generated_eos(phi_tokenizer):
    t = phi_tokenizer
    end = t.convert_tokens_to_ids("<|end|>")
    eos = t.eos_token_id
    raw = torch.tensor([[123, end, eos, eos], [123, eos, eos, eos]])
    attention = get_response_mask(raw, eos_token=[end, eos])
    assert attention.tolist() == [[1, 1, 0, 0], [1, 1, 0, 0]]
    rows = [SimpleNamespace(prompt_ids=(12, 13), completion_ids=tuple(raw[i][attention[i].bool()].tolist()))
            for i in range(2)]
    batch = make_batch(rows, eos).batch
    for i in range(2):
        assert batch['responses'][i][batch['response_mask'][i].bool()].tolist() == list(rows[i].completion_ids)
    assert is_response_end(t, end) and is_response_end(t, eos)
    assert not is_response_end(t, 123)


@pytest.mark.parametrize("mode", ["full", "single_turn", "limited_multi_turn"])
def test_context_builders_produce_trainable_batches_and_valid_actions(phi_tokenizer, mode):
    ctx = context(phi_tokenizer, mode)
    response = "<think>Plan</think><answer>Up</answer>"
    episodes = [dict(env_id=i, group_id=i, history=[
        dict(state="State" * (i + 1), actions_left=5, llm_response=response, reward=.25),
        dict(state="Next", actions_left=4, llm_response=response, reward=.75),
        dict(state="Done", actions_left=3),
    ]) for i in range(2)]
    batch = ctx.get_lm_inputs(episodes, prepare_for_update=True).batch
    assert batch['input_ids'].shape[0] == (2 if mode == "full" else 4)
    end = phi_tokenizer.convert_tokens_to_ids("<|end|>")
    assert (batch['responses'][:, -1] == end).all()
    assert (batch['loss_mask'][:, -1] == 1).all()
    assert (batch['rm_scores'].sum(-1) == 1).all()
    assert not (batch['loss_mask'].bool() & ~batch['attention_mask'][:, 1:].bool()).any()
    parsed, actions = ctx._parse_response(response)
    assert actions == ["Up"] and parsed == response
    infer = ctx.get_lm_inputs(episodes, prepare_for_update=False)
    for ids, valid in zip(infer.batch['input_ids'], infer.batch['attention_mask']):
        assert phi_tokenizer.decode(ids[valid.bool()]).endswith("<|assistant|><think>")


def test_phi_detection_works_after_renaming_checkpoint(phi_tokenizer, tmp_path):
    assert is_phi_tokenizer(phi_tokenizer)
    assert is_phi_model_path("microsoft/Phi-4-mini-instruct")
    assert not is_phi_model_path("Qwen/Qwen2.5-3B-Instruct")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "phi3"}))
    assert is_phi_model_path(tmp_path)


def test_qwen_masks_and_reward_placement_match_legacy_golden():
    tokenizer = SimpleNamespace(name_or_path="Qwen/Qwen2.5-3B-Instruct",
                                 encode=lambda text: [1 if text == "<|im_start|>" else 2])
    ids = torch.tensor([[1, 10, 2, 9, 1, 20, 2, 9, 1, 30, 2, 9, 1, 20, 2, 9, 1, 30, 2, 9]])
    scores, loss, response = get_masks_and_scores(ids, tokenizer, [[.25, .75]], True, True)
    assert loss.tolist() == [[0.] * 8 + [1.] * 4 + [0.] * 4 + [1.] * 3]
    assert torch.equal(loss, response)
    assert scores.nonzero().tolist() == [[0, 10], [0, 18]]
    assert scores[0, [10, 18]].tolist() == [.25, .75]
    assert not is_phi_tokenizer(tokenizer)


@pytest.mark.parametrize("config_name", [
    "_2_sokoban", "_3_frozen_lake", "_4_countdown", "_6_webshop", "_7_lean_light",
    "_2_sokoban_tropic", "_3_frozen_lake_tropic", "_4_countdown_tropic", "_6_webshop_tropic",
    "_2_sokoban_maxrl", "_2_sokoban_tstar",
])
def test_phi_configs_pass_entrypoint_validation(config_name):
    from train import add_dependency_and_validate_config
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / "config"), version_base=None):
        cfg = compose(config_name=config_name, overrides=["model_path=microsoft/Phi-4-mini-instruct"])
    assert add_dependency_and_validate_config(cfg).data.train_batch_size > 0


def test_native_phi_actor_and_critic_support_forward_backward():
    from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, Phi3Config
    from ragen.tropic.batch import verified_path_loss
    config = Phi3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                        original_max_position_embeddings=64, pad_token_id=0, bos_token_id=1,
                        eos_token_id=2, num_labels=1, partial_rotary_factor=.75)
    for cls in (AutoModelForCausalLM, AutoModelForTokenClassification):
        model = cls.from_config(config, attn_implementation="sdpa")
        inputs = torch.tensor([[0, 1, 3, 4, 5], [1, 3, 4, 5, 6]])
        attention = inputs.ne(0).long()
        positions = (attention.cumsum(-1) - 1).clamp(min=0)
        outputs = model(input_ids=inputs, attention_mask=attention, position_ids=positions)
        if cls == AutoModelForCausalLM:
            log_probs = outputs.logits[:, :-1].log_softmax(-1).gather(-1, inputs[:, 1:, None]).squeeze(-1)
            loss = verified_path_loss(log_probs, attention[:, 1:], torch.tensor([.1, .2]))
        else:
            assert outputs.logits.shape == (2, 5, 1)
            loss = outputs.logits.square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)

"""Gemma-only native model and checkpoint tests; run inside gemma_venv."""

import json
import pytest
import torch
import transformers

if not hasattr(transformers, "Gemma4TextConfig"):
    pytest.skip("Requires gemma_venv", allow_module_level=True)

from safetensors.torch import save_file, load_file
from transformers import (
    Gemma4TextConfig,
    Gemma4ForCausalLM,
    AutoModelForTokenClassification,
)
from ragen.gemma.model import Gemma4ForTokenClassification
from ragen.gemma.prepare import export_text_checkpoint


@pytest.fixture(params=["E2B", "E4B"])
def variant(request):
    return request.param


def tiny_config(variant="E4B"):
    return Gemma4TextConfig(
        vocab_size=64,
        vocab_size_per_layer_input=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1 if variant == "E2B" else 2,
        final_logit_softcapping=30.0,
        head_dim=16,
        global_head_dim=32,
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=2,
        use_double_wide_mlp=variant == "E2B",
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=16,
        max_position_embeddings=128,
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        num_labels=1,
    )


@pytest.mark.parametrize("cls", [Gemma4ForCausalLM, Gemma4ForTokenClassification])
def test_shared_kv_checkpointed_backward_and_save_reload(cls, variant, tmp_path):
    config = tiny_config(variant)
    config._attn_implementation = "sdpa"
    model = cls(config)
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy

    assert get_fsdp_wrap_policy(model) is not None
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    inputs = torch.tensor([[0, 2, 3, 4], [2, 4, 5, 6]])
    attention = inputs.ne(0).long()
    positions = (attention.cumsum(-1) - 1).clamp(min=0)
    output = model(
        input_ids=inputs,
        attention_mask=attention,
        position_ids=positions,
        use_cache=False,
    )
    output.logits.square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(grad).all() for grad in gradients)
    torch.optim.AdamW(model.parameters(), lr=1e-4).step()
    model.save_pretrained(tmp_path)
    factory = (
        transformers.AutoModelForCausalLM
        if cls == Gemma4ForCausalLM
        else AutoModelForTokenClassification
    )
    restored = factory.from_pretrained(
        tmp_path, local_files_only=True, attn_implementation="sdpa"
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_export_preserves_text_weights_and_rejects_missing_weights(variant, tmp_path):
    model = Gemma4ForCausalLM(tiny_config(variant))
    source = tmp_path / "original"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "gemma4", "text_config": model.config.to_dict()})
    )
    weights = {
        name.replace("model.", "model.language_model.", 1): value.clone()
        for name, value in model.state_dict().items()
        if name != "lm_head.weight"
    }
    weights["model.vision_tower.example.weight"] = torch.zeros(2)
    extra = "model.language_model.layers.2.self_attn.k_norm.weight"
    weights[extra] = torch.ones(16)
    save_file(weights, source / "model.safetensors")
    (source / "tokenizer_config.json").write_text(
        '{"processor_class":"Gemma4Processor"}'
    )
    exported = export_text_checkpoint(source, tmp_path / "text")
    actual = load_file(exported / "model.safetensors")
    assert "model.layers.2.self_attn.k_norm.weight" in actual
    assert all("vision_tower" not in name for name in actual)
    for name, value in model.state_dict().items():
        if name != "lm_head.weight":
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
    restored = Gemma4ForCausalLM.from_pretrained(exported, local_files_only=True)
    torch.testing.assert_close(
        restored.lm_head.weight, model.lm_head.weight, rtol=0, atol=0
    )
    assert "processor_class" not in json.loads(
        (exported / "tokenizer_config.json").read_text()
    )
    with pytest.raises(FileExistsError):
        export_text_checkpoint(source, exported)
    del weights["model.language_model.layers.0.self_attn.q_proj.weight"]
    save_file(weights, source / "model.safetensors")
    with pytest.raises(ValueError, match="missing="):
        export_text_checkpoint(source, tmp_path / "incomplete")
    assert not (tmp_path / "incomplete").exists()


def test_entropy_recomputation_preserves_native_gemma_training_gradients():
    from ragen.workers.actor.entropy import entropy_from_logits_recomputed

    torch.manual_seed(17)
    config = tiny_config()
    config._attn_implementation = "sdpa"
    model = Gemma4ForCausalLM(config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    inputs = torch.tensor([[0, 2, 3, 4, 5], [2, 4, 5, 6, 7]])
    attention = inputs.ne(0).long()
    positions = (attention.cumsum(-1) - 1).clamp(min=0)
    gradients = []
    for recompute in (False, True):
        model.zero_grad(set_to_none=True)
        output = model(input_ids=inputs, attention_mask=attention, position_ids=positions, use_cache=False)
        logits = output.logits[:, :-1] / 0.5
        log_probs = logits.log_softmax(-1).gather(-1, inputs[:, 1:, None]).squeeze(-1)
        values = logits.float()
        entropy = (entropy_from_logits_recomputed(logits.reshape(-1, 64), 3).reshape(2, 4)
                   if recompute else values.logsumexp(-1) - (values.softmax(-1) * values).sum(-1))
        loss = -log_probs.mean() - 0.001 * entropy.mean()
        loss.backward()
        gradients.append({name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None})
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=1e-4, atol=1e-6, msg=name)

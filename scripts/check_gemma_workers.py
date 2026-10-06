"""Single-GPU Gemma worker lifecycle check with temporary tiny weights.

Activate gemma_venv, then: CUDA_VISIBLE_DEVICES=0 python scripts/check_gemma_workers.py
"""

import os
import shutil
import socket
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from transformers import AutoTokenizer, Gemma4ForCausalLM, Gemma4TextConfig
from safetensors.torch import load_file, save_file


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This integration check requires one CUDA GPU and gemma_venv"
        )
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("VLLM_CACHE_ROOT", str(root / ".cache/gemma/vllm"))
    with tempfile.TemporaryDirectory(prefix="ragen-gemma-worker-") as folder:
        folder = Path(folder)
        model_path = folder / "gemma-4-tiny"
        cfg = Gemma4TextConfig(
            vocab_size=262144,
            vocab_size_per_layer_input=262144,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            final_logit_softcapping=30.0,
            head_dim=16,
            global_head_dim=32,
            hidden_size_per_layer_input=8,
            num_kv_shared_layers=2,
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
            eos_token_id=[1, 106, 50],
            bos_token_id=2,
        )
        model = Gemma4ForCausalLM(cfg).to(torch.bfloat16)
        model.save_pretrained(model_path)
        weights = load_file(model_path / "model.safetensors")
        for i in (2, 3):
            weights[f"model.layers.{i}.self_attn.k_norm.weight"] = torch.ones(
                16 if i == 2 else 32, dtype=torch.bfloat16
            )
        save_file(weights, model_path / "model.safetensors", metadata={"format": "pt"})
        del model, weights
        prepared = root / ".cache/gemma/gemma-4-E4B-it"
        for pattern in ("tokenizer*", "*.jinja", "generation_config.json"):
            for file in prepared.glob(pattern):
                shutil.copyfile(file, model_path / file.name)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        with initialize_config_dir(config_dir=str(root / "config"), version_base=None):
            conf = compose(
                config_name="_2_sokoban", overrides=[f"model_path={model_path}"]
            )
        from train import add_dependency_and_validate_config

        add_dependency_and_validate_config(conf)
        with open_dict(conf):
            conf.actor_rollout_ref.rollout.max_model_len = 128
            conf.actor_rollout_ref.rollout.max_num_batched_tokens = 128
            conf.actor_rollout_ref.rollout.max_num_seqs = 2
            conf.actor_rollout_ref.rollout.response_length = 4
            conf.actor_rollout_ref.rollout.gpu_memory_utilization = 0.08
            conf.actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes = (
                16 * 1024 * 1024
            )
            conf.actor_rollout_ref.actor.ppo_mini_batch_size = 1
            conf.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
            conf.actor_rollout_ref.actor.optim.total_training_steps = 2
            conf.critic.ppo_mini_batch_size = 1
            conf.critic.ppo_micro_batch_size_per_gpu = 1
            conf.critic.model.tokenizer_path = str(model_path)
            conf.critic.optim.total_training_steps = 2
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ.update(
            RANK="0",
            WORLD_SIZE="1",
            LOCAL_RANK="0",
            LOCAL_WORLD_SIZE="1",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
        )
        torch.cuda.set_device(0)
        from ragen.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl import DataProto
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
        from vllm import SamplingParams

        critic_config = omega_conf_to_dataclass(conf.critic)

        inputs = tokenizer("Test", return_tensors="pt", add_special_tokens=False)
        reference_ids = []
        original_release = vLLMRollout.release

        async def capture_before_sleep(rollout):
            if not reference_ids:
                output = rollout.inference_engine.generate(
                    [{"prompt_token_ids": inputs.input_ids[0].tolist()}],
                    SamplingParams(temperature=0, max_tokens=4, stop_token_ids=[1, 106, 50]),
                    use_tqdm=False,
                )
                reference_ids.extend(output[0].outputs[0].token_ids)
            await original_release(rollout)

        worker = ActorRolloutRefWorker(conf.actor_rollout_ref, role="actor_rollout")
        with patch.object(vLLMRollout, "release", capture_before_sleep):
            worker.init_model()
        print("GEMMA_WORKER_INIT_OK", flush=True)
        assert worker.rollout.sleep_level == 1
        positions = torch.arange(inputs.input_ids.shape[1]).unsqueeze(0)
        batch = DataProto.from_dict(
            tensors={
                "input_ids": inputs.input_ids,
                "attention_mask": inputs.attention_mask,
                "position_ids": positions,
            },
            meta_info={
                "eos_token_id": [1, 106, 50],
                "pad_token_id": 0,
                "validate": True,
                "do_sample": False,
            },
        )
        generated = worker.generate_sequences(batch)
        assert reference_ids
        assert generated.batch["responses"][0, :len(reference_ids)].tolist() == reference_ids
        repeated = worker.generate_sequences(batch)
        torch.testing.assert_close(repeated.batch["responses"], generated.batch["responses"])
        print("GEMMA_SLEEP_REFRESH_PARITY_OK", flush=True)
        print(
            "GEMMA_WORKER_GENERATE_OK", generated.batch["responses"].shape, flush=True
        )
        ids = torch.tensor([[2, 3, 4, 5]], device="cuda")
        mb = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "position_ids": torch.arange(4, device="cuda")[None, :],
            "responses": ids[:, 1:],
        }
        worker.actor.actor_optimizer.zero_grad()
        entropy, logp = worker.actor._forward_micro_batch(
            mb, 1.0, calculate_entropy=True
        )
        assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()
        (-logp.mean() - 0.001 * entropy.mean()).backward()
        worker.actor.actor_optimizer.step()
        print("GEMMA_ACTOR_UPDATE_OK", flush=True)
        worker.save_checkpoint(str(folder / "actor_checkpoint"), global_step=1)
        print("GEMMA_ACTOR_SAVE_OK", flush=True)
        worker.generate_sequences(batch)
        print("GEMMA_WEIGHT_REFRESH_OK", flush=True)
        critic = CriticWorker(critic_config)
        critic.init_model()
        critic.critic_optimizer.zero_grad()
        values = critic.critic._forward_micro_batch(mb)
        values.square().mean().backward()
        critic.critic_optimizer.step()
        critic.save_checkpoint(str(folder / "critic_checkpoint"), global_step=1)
        print("GEMMA_CRITIC_UPDATE_SAVE_OK", flush=True)
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

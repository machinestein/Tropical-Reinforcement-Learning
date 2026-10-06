# Tropical Reinforcement Learning

Official code for the paper
**[Tropical Reinforcement Learning](https://arxiv.org/abs/2610.02478)**
(arXiv:2610.02478, 2026).

Arip Asadulaev, Aladin Djuhera, Karim Salta, Holger Boche, Fakhri Karray, Martin Takac
(MBZUAI, TUM)

This repository implements **TROPIC**, the training algorithm proposed in the
paper, together with all baselines, ablations, and environments used in the
experiments. It is built on [RAGEN-2](https://github.com/mll-lab-nu/RAGEN) and
[VERL](https://github.com/volcengine/verl).

## Contents

- [Summary](#summary)
- [Method](#method)
- [Results](#results)
- [What is included](#what-is-included)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproducing the paper](#reproducing-the-paper)
- [Configuration reference](#configuration-reference)
- [Additional experiments](#additional-experiments)
- [Outputs](#outputs)
- [Testing](#testing)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Attribution and license](#attribution-and-license)

## Summary

Reinforcement learning for language models usually maximizes expected return,
which **adds up** the probabilities of all successful trajectories. The paper
argues that this sum is a poor fit for compositional reasoning, where a solution
must be assembled from steps that the model produces in separate, often failed,
attempts but rarely produces together:

- A sum reports *how often* the policy succeeds, not *which* solution worked, so
  fragments from different rollouts cannot be combined.
- Because probabilities sum to one, reinforcing one solution can lower another
  that was never shown to be wrong.

**Tropical RL** changes one operation: alternative trajectories are combined by
their **maximum** instead of their sum, while log-probabilities are still added
along each trajectory. This is the max-plus, or *tropical*, semiring. The value
of a state becomes the log-probability of its most likely verified solution,
together with an explicit path that can be replayed. Because addition
distributes over the maximum, the best prefix and the best suffix meeting at a
shared state can be joined even when they come from different rollouts.

Main theoretical results:

- **Forcing result (Theorem 2).** The maximum is the only aggregation whose
  value always names a verified path (certification) and never decreases when
  new verified paths are found (retention).
- **Compositional policy improvement (Theorem 4).** For a per-state softmax
  actor, a gradient step on the best verified path never lowers the tropical
  value of any state in the graph, and states whose best path shares a piece of
  it gain at least as much as that piece.
- **pass@k lower bound (Eq. 10).** Raising the tropical value of a problem
  raises a lower bound on its pass@1 and pass@k.

## Method

TROPIC targets deterministic, resettable environments with verifiable outcomes.
Its objective is to make the best verified solution of each problem as likely
as possible. Each iteration:

1. **Memory.** For every problem, keep a graph of all valid steps seen so far,
   including steps from failed attempts. Complete paths that replay from the
   start and pass the verifier form the archive. The actor is frozen and every
   stored step is rescored under it.
2. **Prefix and suffix values.** Run the tropical Bellman recursion backward
   from verified endings (how easily the actor can finish from a state) and
   forward from the start (how easily it can reach a state). Both passes are
   linear in the size of the graph.
3. **Sampling.** Each problem receives G groups of n attempts. The first group
   starts from the initial state; later groups start from *frontier* states,
   which the graph can reach but not yet finish from. Frontier states are chosen
   at the least-explored depth by reachability minus a visit-count penalty
   (weight kappa).
4. **Composition.** Wherever a state has a known way in and a known way out,
   join the L best prefixes with the L best suffixes, then replay and verify
   each joined path. Accepted joins enter the archive.
5. **Training.** Select up to K archived paths per problem (the best path plus
   paths that cover rarely trained steps) and take a maximum-likelihood step on
   their per-token log-likelihood. There are no reward weights, no baseline,
   no critic, and no term that pushes any path down.

The actor is prompted with the problem and the current environment state only,
and the step count is part of the state, so suffix scores depend only on where
they start and the state graph is acyclic.

At evaluation time TROPIC uses **no** archive, restarts, or composition. The
trained policy must solve each instance in a single attempt, exactly like the
baselines.

## Results

Success rate (%) on 512 held-out instances per task, mean and standard
deviation over three training seeds (paper, Table 2). Higher is better.

| Task | Method | Qwen2.5-3B-Instruct | Gemma 4 E4B-it |
| --- | --- | --- | --- |
| Sokoban | Base model | 12.89 | 45.50 |
| | PPO | 21.90 +/- 2.14 | 71.68 +/- 1.71 |
| | GRPO | 27.60 +/- 2.47 | 66.53 +/- 2.32 |
| | DAPO | 36.55 +/- 2.11 | 72.65 +/- 1.35 |
| | SNR-Aware | 42.20 +/- 1.96 | 76.55 +/- 1.94 |
| | **TROPIC** | **58.40 +/- 1.83** | **83.10 +/- 1.88** |
| Countdown | Base model | 13.23 | 42.18 |
| | PPO | 26.71 +/- 1.46 | 66.53 +/- 1.80 |
| | GRPO | 21.97 +/- 1.72 | 67.65 +/- 1.10 |
| | DAPO | 24.75 +/- 1.51 | 63.10 +/- 2.25 |
| | SNR-Aware | 28.93 +/- 1.38 | 58.20 +/- 1.95 |
| | **TROPIC** | **34.79 +/- 1.62** | **73.63 +/- 1.48** |
| Frozen Lake | Base model | 33.34 | 71.30 |
| | PPO | 76.08 +/- 1.65 | 84.40 +/- 1.33 |
| | GRPO | 59.57 +/- 2.28 | 83.90 +/- 1.85 |
| | DAPO | 76.88 +/- 1.57 | 85.10 +/- 2.12 |
| | SNR-Aware | 77.24 +/- 1.74 | 79.50 +/- 2.45 |
| | **TROPIC** | **86.35 +/- 1.29** | **92.30 +/- 1.10** |
| WebShop | Base model | 22.75 | 55.27 |
| | PPO | 42.57 +/- 2.11 | 68.42 +/- 1.94 |
| | GRPO | 56.68 +/- 2.34 | 73.16 +/- 1.81 |
| | DAPO | 57.39 +/- 2.19 | 74.58 +/- 1.69 |
| | SNR-Aware | 58.79 +/- 2.51 | 76.84 +/- 1.73 |
| | **TROPIC** | **65.72 +/- 1.96** | **81.40 +/- 1.55** |

TROPIC is best on every task for both models, improving over the strongest
baseline by up to 16.2 points (Sokoban, Qwen). It generates a comparable or
smaller number of tokens per episode and trains faster in all but one setting.
Token counts and wall-clock times are reported in the paper.

### Ablations (Sokoban, Qwen2.5-3B-Instruct)

From Table 3 of the paper. First-Solve is the fraction of solved training
problems whose first verified solution came from cross-rollout composition,
before any complete successful trajectory was sampled.

| Method | Success rate (%) | First-Solve (%) |
| --- | --- | --- |
| MaxRL | 51.30 +/- 2.31 | - |
| GiGPO | 47.60 +/- 2.18 | - |
| GiGPO + T-STAR | 53.10 +/- 2.18 | - |
| TROPIC, no composition | 40.80 +/- 2.26 | 0.0 |
| TROPIC, root-only sampling | 51.80 +/- 2.12 | 24.7 +/- 3.2 |
| TROPIC, successful fragments only | 53.30 +/- 1.97 | 0.0 |
| **Full TROPIC** | **58.40 +/- 1.83** | **31.6 +/- 3.5** |

Composition is the main source of improvement: removing it drops success to
roughly the level of the strongest on-policy baseline, while fragments from
failed rollouts and frontier restarts each add further gains.

## What is included

| Category | Contents |
| --- | --- |
| Tasks (paper) | Sokoban, Countdown, Frozen Lake, WebShop |
| Tasks (extra, not in the paper) | Lean (theorem proving), Sudoku |
| Methods | TROPIC, PPO, GRPO, DAPO (RAGEN-2 implementation), SNR-Aware filtering |
| TROPIC ablations | No composition, successful fragments only, root-only sampling |
| Extra Sokoban comparisons | MaxRL and T-STAR adaptations |
| Models | Qwen2.5-3B-Instruct, Gemma-4-E4B-it (paper); Phi-4-mini-instruct (extra) |

Bundled in the source tree: the VERL runtime (v0.6.1) and the WebShop
simulator with its small catalog and search indexes. No Git submodules are
required. Model weights, generated datasets, trained checkpoints, and
experiment outputs are not included.

## Requirements

| Component | Requirement |
| --- | --- |
| OS | Linux |
| Python | 3.12 (enforced by `setup.sh`) |
| GPU | NVIDIA with CUDA and enough memory for full finetuning. The paper uses 8 NVIDIA A100 GPUs. |
| Build tools | `nvcc` and a C++ compiler, unless a prebuilt FlashAttention wheel is supplied |
| WebShop only | Java 21+ |
| Lean only | Docker (for the Kimina Lean verifier) |

Pinned stack for Qwen and Phi: PyTorch 2.8.0, vLLM 0.11.0, Transformers 4.56.1,
FlashAttention 2.8.1. Gemma uses a separate pinned runtime. Training uses FSDP
with BF16 mixed precision, vLLM for rollouts, and Ray for distributed
execution.

## Installation

There are two separate environments. Do not mix them.

### Qwen and Phi

```bash
git clone https://github.com/machinestein/Tropical-Reinforcement-Learning.git
cd Tropical-Reinforcement-Learning
bash setup.sh
source venv/bin/activate
```

`setup.sh` accepts the following environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BASE_PYTHON` | `python3` | Interpreter used to create the venv (must be 3.12) |
| `VENV_DIR` | `venv` | Location of the virtual environment |
| `MAX_JOBS` | `8` | Parallel jobs when compiling FlashAttention |
| `FLASH_ATTN_WHEEL` | unset | Path to a compatible FlashAttention 2.8.1 wheel to skip the source build |

### Gemma

```bash
bash scripts/setup_gemma_venv.sh
source gemma_venv/bin/activate
python -m ragen.gemma.prepare     # writes a text-only checkpoint to .cache/gemma/gemma-4-E4B-it
python -m ragen.gemma.preflight   # sanity-checks the runtime
```

Accept the model's access terms on Hugging Face and authenticate
(`huggingface-cli login`) before running `prepare`.

### WebShop extras

If WebShop dependencies or the spaCy model are missing:

```bash
bash scripts/setup_webshop_venv.sh
```

Set `JAVA_HOME` if Java 21+ is not on the default path.

## Quick start

Inspect what a launcher will do without loading a model:

```bash
bash scripts/runs/run_sokoban_light.sh --dry-run
```

Train only TROPIC on Sokoban on a single GPU:

```bash
RUNS=TROPIC GPU=0 bash scripts/runs/run_sokoban_light.sh
```

Every launcher supports `--help`, which lists its settings and defaults.

## Reproducing the paper

### Main results (Table 2)

From the repository root with the appropriate environment activated:

```bash
export GPU=0,1,2,3,4,5,6,7
export MODEL=Qwen/Qwen2.5-3B-Instruct
export MICRO_BATCH_SIZE=1
export EVAL_PROBLEMS=512 EVAL_ATTEMPTS=1
export SEED=10000 EVAL_SEED=123
export WANDB_MODE=disabled

RUNS=EVAL,PPO,GRPO,SNR,DAPO,TROPIC \
  bash scripts/runs/run_sokoban_light.sh

RUNS=EVAL,PPO,GRPO,SNR,DAPO,TROPIC PROTOCOL=original DATA=tropic \
  bash scripts/runs/run_countdown_light.sh

RUNS=EVAL,PPO,GRPO,SNR,DAPO,TROPIC TROPIC_COVERAGE=false \
  bash scripts/runs/run_frozen_lake_light.sh

RUNS=EVAL,PPO,GRPO,SNR,DAPO,TROPIC TROPIC_COVERAGE=true STEPS=100 \
  bash scripts/runs/run_webshop_light.sh
```

Each command runs the selected methods sequentially; `RUNS=EVAL` evaluates the
base model. The paper trains for at most 200 iterations on Sokoban, Countdown,
and Frozen Lake and at most 100 on WebShop, validates every 25 iterations, and
reports the mean over three training seeds. Repeat each command with three
different `SEED` values and distinct `EXPERIMENT` names to reproduce this.

Evaluation matches the paper: 512 held-out instances, temperature 0.5, single
attempt (pass@1). Set `EVAL_ATTEMPTS=4` to also estimate pass@2 and pass@4.

Task data:

- **Sokoban** (6x6 grid, one box) and **Frozen Lake** (4x4 grid): generated
  from seeds.
- **Countdown:** `DATA=tropic` generates deterministic, disjoint training and
  validation sets automatically.
- **WebShop:** uses the RAGEN-2 webshop-minimal catalog and search indexes,
  which are bundled. The launcher checks simulator and replay compatibility
  before training.

### Gemma

Activate `gemma_venv` and use the wrapper, which selects the prepared model and
defaults to 100 steps on WebShop:

```bash
unset MODEL
RUNS=EVAL,PPO,GRPO,SNR,DAPO,TROPIC GPU=0,1,2,3,4,5,6,7 \
  bash scripts/runs/run_gemma_light.sh sokoban
```

Supported tasks: `sokoban`, `countdown`, `frozen_lake`, `webshop`, `lean`,
`sudoku`, `sokoban_ablations`, `countdown_ablations`, `sokoban_maxrl`,
`sokoban_tstar`.

For long WebShop sequences, critic offloading trades speed for GPU memory:

```bash
GEMMA_WEBSHOP_CPU_OFFLOAD=true GEMMA_WEBSHOP_MAX_MODEL_LEN=15000 \
RUNS=EVAL,PPO,GRPO,SNR,DAPO,TROPIC GPU=0,1,2,3,4,5,6,7 \
EXPERIMENT=webshop_gemma_offload_seed10000 \
  bash scripts/runs/run_gemma_light.sh webshop
```

Offloading applies to PPO, SNR, and DAPO; GRPO and TROPIC have no critic.
Changing the context limit changes the experimental setting and the amount of
history available to the agent.

### Ablations and comparisons (Table 3)

```bash
bash scripts/runs/sokoban_ablations.sh     # no composition, root-only, successful-only
bash scripts/runs/run_sokoban_maxrl.sh     # MaxRL
bash scripts/runs/run_sokoban_tstar.sh     # T-STAR
bash scripts/runs/countdown_ablations.sh   # Countdown ablations plus a full-TROPIC control
```

Use the main TROPIC run as the control for the Sokoban ablations.

### Hyperparameters

TROPIC settings from the paper (Table 4), shared across tasks and models, and
their names in `config/*_tropic.yaml`:

| Symbol | Description | Value | Config key |
| --- | --- | --- | --- |
| G | Attempt groups per problem per iteration | 2 | `tropic.waves_per_iteration` |
| n | Attempts per group | 8 | |
| kappa | Frontier exploration weight | 1.0 | `tropic.frontier_eta` |
| L | Prefixes and suffixes per shared state | 4 each | `tropic.top_l` |
| K | Maximum training paths per problem | 2 | `tropic.basis_size` |
| | Composition candidates per problem per refresh | 64 | `tropic.max_candidates_per_refresh` |
| | Maximum retained edges per problem | 1024 | `tropic.max_edges_per_problem` |
| | Maximum archived solutions per problem | 64 | `tropic.max_solutions_per_problem` |
| | Edge-rescoring batch size | 64 | `tropic.rescore_batch_size` |
| | Coverage-based path selection | enabled | `tropic.coverage_enabled` |
| | Training-rollout temperature | 1.0 | |
| | Update epochs per iteration | 2 | |
| | Recurring training-problem pool | 128 | |

Shared optimization settings: AdamW, actor learning rate 1e-6, betas
(0.9, 0.999), weight decay 0.01, mini-batch of 32 sequences. PPO and SNR-Aware
use a critic with learning rate 1e-5 and GAE with gamma = lambda = 1. Baselines
use clip range [0.8, 1.28], entropy coefficient 0.001, and no KL penalty.
SNR-Aware uses the RAGEN-2 rollout filter with threshold p = 0.9.

### Notes on the comparison

- Baselines keep their original RAGEN-2 interfaces (full-history prompts).
  TROPIC uses one action per decision and state-conditioned prompts.
- The nominal budget is 16 rollout starts per problem for all methods (for
  TROPIC, two groups of eight). This does not imply equal generated-token
  counts or verification costs. Replaying a composed path generates no tokens.
- TROPIC stops early when no training problem yields a new update; baselines
  follow RAGEN-2's stopping rules. All methods share the same iteration cap.
- Frontier restarts require a deterministic, resettable environment, which is
  the setting TROPIC is designed for.

## Configuration reference

Common launcher variables (see each script's `--help` for the full list):

| Variable | Default | Description |
| --- | --- | --- |
| `RUNS` | `EVAL,PPO,GRPO,SNR,TROPIC` | Comma-separated methods: `EVAL`, `PPO`, `GRPO`, `SNR`, `DAPO`, `TROPIC` |
| `MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Hugging Face model id |
| `GPU` | `0` | Comma-separated GPU ids |
| `MICRO_BATCH_SIZE` | `1` | Sequences per GPU per microbatch |
| `STEPS` | `200` | Training iterations |
| `TEST_FREQ` | `25` | Validation interval (iterations) |
| `SAVE_FREQ` | `100` | Checkpoint interval (iterations) |
| `EVAL_PROBLEMS` | task-dependent (512 for most) | Number of validation problems |
| `EVAL_ATTEMPTS` | task-dependent | Samples per validation problem (pass@k) |
| `SEED`, `EVAL_SEED` | `10000`, task-dependent | Training and evaluation seeds |
| `EXPERIMENT` | `<task>_seed<SEED>` | Run name; must be new for every run |
| `SAVE_DIR` | `saves/` | Output root |
| `WANDB_MODE` | `disabled` | Set to `online` to enable Weights & Biases |
| `WANDB_PROJECT` | `RAGEN2` | W&B project name |
| `TROPIC_COVERAGE` | task-dependent | Coverage-based path selection (Frozen Lake, WebShop) |
| `PROTOCOL`, `DATA` | `original`, `tropic` | Countdown protocol and dataset source |

`DAPO` is not in the default `RUNS` list for the standard launchers; include it
explicitly as shown above. Existing run folders are never overwritten, so use a
fresh `EXPERIMENT` for each new run.

## Additional experiments

These are supported by the code but are not part of the paper.

### Phi

Use the standard launchers with `MODEL=microsoft/Phi-4-mini-instruct` and a
distinct `EXPERIMENT` name.

### Lean and Sudoku

Lean requires a Kimina verifier. To run one locally:

```bash
docker run -d --name tropical-lean -p 127.0.0.1:8000:8000 \
  -e LEAN_SERVER_HOST=0.0.0.0 -e LEAN_SERVER_PORT=8000 \
  -e LEAN_SERVER_MAX_REPLS=32 -e LEAN_SERVER_MAX_REPL_USES=-1 \
  -e LEAN_SERVER_MAX_REPL_MEM=8G -e LEAN_SERVER_MAX_WAIT=60 \
  -e LEAN_SERVER_ENVIRONMENT=prod -e LEAN_SERVER_LOG_LEVEL=WARNING \
  projectnumina/kimina-lean-server:2.0.0
```

Then prepare the data and launch:

```bash
python scripts/download_data.py
LEAN_SERVER_URL=http://127.0.0.1:8000 bash scripts/runs/run_lean_light.sh
bash scripts/runs/run_sudoku_light.sh
```

These two launchers support `EVAL`, `PPO`, `GRPO`, `SNR`, and `TROPIC`
(no `DAPO`).

## Outputs

Each run writes to `saves/<run name>/`:

- `metrics.jsonl` with training and validation metrics
- validation trajectories
- checkpoints
- the resolved configuration and the launch command

Metrics are written locally even with W&B disabled. To enable online tracking,
run `wandb login`, set `WANDB_MODE=online`, and optionally `WANDB_ENTITY`.

## Testing

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest tests/tropic tests/sokoban_baselines -q
```

The core graph and launcher tests run without GPUs. Some integration tests
require optional data, tokenizers, Java, or a Lean server. Passing unit tests
does not guarantee that a full multi-GPU job fits a given machine.

## Repository layout

```
train.py                    Training entry point (Hydra, config/base.yaml)
config/                     Hydra configs per task and method
ragen/
  tropic/                   TROPIC state graph, collectors, path batching
  trainer/                  Agent and TROPIC trainers, core algorithms, rollout filter
  workers/                  FSDP workers, actors (including tropic_actor.py), critic
  llm_agent/                Agent proxy, context and environment-state managers
  env/                      Task environments
  sokoban_baselines/        MaxRL and T-STAR adaptations
  gemma/                    Gemma model preparation and preflight checks
scripts/
  runs/                     Experiment launchers
  setup_*_venv.sh           Environment setup for Gemma and WebShop
  download_data.py          Lean and Sudoku dataset preparation
verl/                       Vendored VERL v0.6.1 runtime
external/webshop-minimal/   Vendored WebShop simulator and indexes
requirements/               Pinned lock files for the Qwen/Phi and Gemma stacks
tests/                      Unit and integration tests
```

## Troubleshooting

- **`setup.sh` reports "Use Python 3.12":** point `BASE_PYTHON` at a 3.12
  interpreter.
- **FlashAttention build fails:** install the CUDA toolkit and a C++ compiler,
  or supply `FLASH_ATTN_WHEEL`. Lower `MAX_JOBS` if the build runs out of RAM.
- **Launcher refuses to start because the run folder exists:** choose a new
  `EXPERIMENT` name.
- **WebShop fails to start:** check Java 21+ and `JAVA_HOME`, then run
  `scripts/setup_webshop_venv.sh`.
- **Out of GPU memory on WebShop with Gemma:** enable
  `GEMMA_WEBSHOP_CPU_OFFLOAD=true`.

## Citation

```bibtex
@article{asadulaev2026tropical,
  title   = {Tropical Reinforcement Learning},
  author  = {Asadulaev, Arip and Djuhera, Aladin and Salta, Karim and
             Boche, Holger and Karray, Fakhri and Takac, Martin},
  journal = {arXiv preprint arXiv:2610.02478},
  year    = {2026}
}
```

## Attribution and license

This project adapts:

- [RAGEN-2](https://github.com/mll-lab-nu/RAGEN) (MIT, see `LICENSE`)
- [VERL](https://github.com/volcengine/verl) v0.6.1 (Apache-2.0, see `verl/LICENSE`)
- [webshop-minimal](https://github.com/ZihanWang314/webshop-minimal), derived
  from [WebShop](https://github.com/princeton-nlp/WebShop)

Full details are in `THIRD_PARTY_NOTICES`. Model weights and external datasets
are downloaded separately and remain subject to their own terms.

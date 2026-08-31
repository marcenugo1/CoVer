# CoVer: Co-trained Coder and Verifier

**Information-Gain Rewards over Diversity-Pruned Tests: GT-Anchored Verifier Co-Training for Reliable Code Generation**

CoVer is a single-policy GRPO framework that co-trains one language model to act as both **coder** and **verifier** (unit-test author). It fixes two failure modes of standard self-play code RL:

- **Permissiveness collapse** — pass-rate/binary-acceptance rewards are maximised by trivial, non-discriminative tests that carry zero information about correctness.
- **Concentration bias** — i.i.d.-sampled tests cluster on modal inputs, producing correlated pass/fail columns and an inflated-variance reward estimate.

It addresses both with two coupled mechanisms:

1. **Information-Gain (IG) verifier reward.** Each self-generated test is scored by the mutual information between its pass/fail column (across the *m* candidate solutions) and a **graded, ground-truth-anchored correctness signal** `y ∈ [0,1]ᵐ`, gated by the sign of their covariance so **only positively discriminative tests are rewarded**. The graded `y` keeps the signal alive from the first training step, where a binary signal would vanish (`H(y) ≈ 0`).
2. **Three-stage diversity-aware selection.** A candidate test pool is pruned to a behaviourally non-redundant `k`-test suite via (i) invalidity filtering, (ii) input-string deduplication, and (iii) execution-profile deduplication — provably lowering the variance of the IG estimator at a fixed execution budget.

Both rewards are optimised in a single GRPO update so gradients flow back to the same shared policy under both roles.

## Pipeline

```
Program spec ─► Shared policy π_θ ─► Trajectory execution ─► CoVer verifier signal ─► GRPO update
                 (coder + verifier)   (run codes × tests,      1. diversity-aware       (coder reward y,
                                        graded GT signal y)        selection → S_τ        verifier IG reward
                                                                2. covariance-gated        over S_τ)
                                                                   IG reward on S_τ
```

This maps directly onto the `optimization/` modules, run in a loop by [run.py](run.py):

| Stage | Module | What it does |
|---|---|---|
| Sampling | [optimization/sample.py](optimization/sample.py) | Generate `m` candidate codes and `K` candidate tests per task with vLLM |
| Execution | [optimization/execute.py](optimization/execute.py) | Run codes × tests in a sandbox; build the pass/fail table; diversity-aware pruning to `k` tests |
| Reward | [optimization/reward.py](optimization/reward.py) | Compute the signed plug-in MI (IG) verifier reward and the graded coder reward |
| Training | [optimization/train.py](optimization/train.py) | Single-policy GRPO update on the shared model |

## Repository layout

```
optimization/        training pipeline (run.py + sample/execute/reward/train + optimization_config.py)
evaluation/          benchmark evaluation (eval.py + evaluation_config.py)
data/                dataset download (download_data.py)
requirements.txt     dependencies
```

## Installation

```bash
conda create --name coVer python=3.12
source activate coVer
pip install -r requirements.txt
```

Do not install `torch` separately — `vllm` pins it, so let pip resolve the two together.

[FlashAttention](https://github.com/Dao-AILab/flash-attention) is **required**, not optional
(`optimization/train_utils/models.py` imports it at module level), and must match your
torch/CUDA build, so install it from a matching wheel:

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

Our runs used Python 3.12, torch 2.9.1, vllm 0.16.0 and transformers 4.57.6; `requirements.txt`
pins those exactly. One exception: our training env carried numpy 1.26.4, but vllm 0.16.0's own
dependencies require numpy >= 2, so a clean install resolves to 2.2.x; `requirements.txt` follows
what is installable.

Install `requirements.txt` **before** the FlashAttention wheel, and only if it succeeded — the
wheel is built for torch 2.9/CUDA 12, and installing it into an env without the torch pin already
applied will silently pull a newer torch and a mismatched CUDA stack.

## Download data

Datasets are provided in Stdio format on the HuggingFace hub. Download what you need into `data/`:

```bash
cd data
python download_data.py --dataset CodeContests_train   # training data
python download_data.py --dataset LiveBench             # an evaluation set
```

Available: `CodeContests_train`, `LiveBench`, `LiveCodeBench`, `CodeContests`, `CodeForces`,
`MBPP-ReasonFlux`. (`MBPP-ReasonFlux` is the 221-task Stdio-converted MBPP subset reported as
"MBPP" in the results table; it is the file `evaluation/run_table1_7B.py` expects.)

## Training

Set your configuration in [optimization/optimization_config.py](optimization/optimization_config.py), then:

```bash
python run.py
```

Run it from the repository root — `run.py` launches each stage as a subprocess with
`cwd='optimization'` / `cwd='evaluation'`, so starting it from elsewhere breaks those paths.

Key knobs (defaults reflect the paper):

| Config | Default | Meaning |
|---|---|---|
| `pretrained_model` | `Qwen/Qwen2.5-14B-Instruct` | backbone to train (also `Qwen/Qwen2.5-7B-Instruct`) |
| `k_code` / `k_case` | `16` / `32` | candidate codes (`m`) and tests (`K`) sampled per task |
| `k_case_keep` | `16` | tests kept after diversity-aware selection (`k`; `None` disables pruning) |
| `cover_y_binary` | `False` | ablation: use binary `y` instead of graded `y` |
| `total_steps` | `350` | GRPO steps |
| `actor_learning_rate` / `kl_loss_coef` | `1e-6` / `0.01` | learning rate, KL coefficient (β) |
| `gpu_groups` / `total_num_nodes` | `[[0,1]]` / `2` | vLLM engine layout / training GPUs |
| `eval_interval` / `save_interval` | `20` / `50` | steps between evals / checkpoints |

To resume after a stop, set `start_from_scratch = False` and `resume_step = <last completed step>` near the top of [run.py](run.py). Training metrics log to Weights & Biases when `use_wandb = True`. Rollouts and results are written to `optimization/temp_data/` and `optimization/results/`; checkpoints to `optimization/ckpt/`.

## Evaluation

Configure [evaluation/evaluation_config.py](evaluation/evaluation_config.py), then:

```bash
cd evaluation
python eval.py
```

Supports one-shot coding accuracy, unit-test generation, and Best-of-N — via vLLM or an external API. The train and eval prompts are kept byte-identical (`MUST_MIRROR` blocks) so the model sees the same task at both stages. All options are documented inline in `evaluation/evaluation_config.py`.

## Results

One-shot pass@1 (%) on five benchmarks, macro-averaged. CoVer achieves the best macro-average at both scales over the Qwen2.5-Instruct backbone.

| Model | LiveBench | MBPP | LiveCodeBench | CodeContests | CodeForces | Avg. |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 31.1 | 66.3 | 26.9 | 21.2 | 5.4 | 30.18 |
| Qwen2.5-7B-Coder-Instruct | 35.0 | 68.0 | 29.8 | 22.8 | 6.7 | 32.46 |
| **CoVer-7B** | **40.0** | **72.4** | **32.8** | **26.0** | **8.5** | **35.94** |
| Qwen2.5-14B-Instruct | 36.4 | 76.3 | 33.5 | 25.6 | 7.3 | 35.82 |
| **CoVer-14B** | **47.75** | **79.44** | **41.49** | **32.92** | **12.88** | **42.90** |

## Acknowledgement

The training infrastructure builds on [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and [(Wang et al., 2026)](https://huggingface.co/datasets/Gen-Verse/MBPP-ReasonFlux), who also released the benchmark data in Stdio format.

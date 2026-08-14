<div align="center">

# TA-OPD: Tail-Aware Top-k On-Policy Distillation

<p><em>Preserving tail probability for stable and effective on-policy distillation.</em></p>

[![GitHub](https://img.shields.io/badge/GitHub-TA--OPD-181717?logo=github)](https://github.com/HuipengHuang/TA-OPD)
[![Framework](https://img.shields.io/badge/Built%20with-verl-2563eb)](https://github.com/verl-project/verl)

</div>

---

## Overview

- [TA-OPD Objective](#ta-opd-objective)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgement](#acknowledgement)


## TA-OPD Objective

Let $p_t$ and $q_t$ be the student and teacher distributions, and let
$S_t^k = \mathrm{TopK}(q_t, k)$. Their tail probabilities are

$$
p_t^{\mathrm{tail}} = 1 - \sum_{v \in S_t^k} p_t(v),
\qquad
q_t^{\mathrm{tail}} = 1 - \sum_{v \in S_t^k} q_t(v).
$$

TA-OPD adds a tail token and computes reverse KL over the top-k tokens and their aggregated tail:

$$
\ell_t^{\mathrm{TA}} = \sum_{v \in S_t^k} p_t(v) \log \frac{p_t(v)}{q_t(v)} + p_t^{\mathrm{tail}} \log \frac{p_t^{\mathrm{tail}}}{q_t^{\mathrm{tail}}}.
$$

Let $S_t^+ = S_t^k \cup \{v_{\mathrm{tail}}\}$. For $\hat{y}_t \sim p_t$, sample-corrected TA-OPD is

$$
\begin{aligned}
\ell_t^{\mathrm{SC-TA}}
= {} & \sum_{v \in S_t^+} p_t(v)\,\mathrm{sg}\left(\log \frac{p_t(v)}{q_t(v)}\right) \\
& + 1[\hat{y}_t \notin S_t^k]
\frac{p_t(\hat{y}_t)}{\mathrm{sg}(p_t(\hat{y}_t))}\,
\mathrm{sg}\left(
\log \frac{p_t(\hat{y}_t)/p_t^{\mathrm{tail}}}
{q_t(\hat{y}_t)/q_t^{\mathrm{tail}}}
\right).
\end{aligned}
$$

Here $\mathrm{sg}$ denotes stop-gradient.

## Quick Start

### Installation

TA-OPD is implemented on top of [verl](https://github.com/verl-project/verl) and uses vLLM for student rollout and teacher inference.
You can install TA-OPD dependencies by running the following commands:

```bash
conda create -n taopd python=3.12 -y
conda activate taopd

cd TA-OPD
python -m pip install   --ignore-requires-python   --no-deps   --force-reinstall   -r ~/requirements.txt

cd verl
pip3 install --no-deps -e .
```


## Training
The repository already contains the processed DAPO-MATH-17K training set and MATH-500 validation set under `training_data/`.

You can run the following command to run TA-OPD:
```bash
bash train.sh
```

This runs TA-OPD with the default student–teacher pair. Three options are exposed on the command line:

```bash
# Choose the distillation objective
bash train.sh --loss_mode normalized_reverse_kl_topk   # Normalized top-k OPD
bash train.sh --loss_mode taopd_reverse_kl_topk        # TA-OPD (default)
bash train.sh --loss_mode sample_corrected_ta_opd      # Sample-corrected TA-OPD

# Choose the models (HuggingFace ID or local path)
bash train.sh --student Qwen/Qwen2.5-7B-Instruct --teacher open-thoughts/OpenThinker3-7B
```

| Option | Default | Description |
|---|---|---|
| `--student` | `Qwen/Qwen3-1.7B` | Student model ID or path |
| `--teacher` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Teacher model ID or path |
| `--loss_mode` | `taopd_reverse_kl_topk` | Distillation objective |

## Evaluation

The `evaluation/` directory provides vLLM generation and answer verification.

`MODEL_BEAUTIFUL_NAME` is a short, user-defined name used only to identify the model in the output filename. It does not need to match the Hugging Face model ID. For example, use `taopd_qwen3_1.7b` for a Qwen3-1.7B model trained with TA-OPD.

`valid_without_mmlu_pro.parquet` contains 2,762 examples from the following benchmarks:

- MATH-500
- AMC
- OlympiadBench
- AIME 2024
- AIME 2025
- Minerva Math
- ARC-Challenge (ARC-c)

You can do evaluation by running the following commands:

```bash
cd evaluation
mkdir -p llm_output

python generate_vllm.py \
  --model_path /path/to/model \
  --model_beautiful_name MODEL_BEAUTFUL_NAME \
  --input_file ./evaluation_data/valid_without_mmlu_pro.parquet \
  --n 8 \
  --temperature 0.7 \
  --top_p 0.95 \
  --tensor_parallel_size 4
```


## Acknowledgement

TA-OPD is built on [verl](https://github.com/verl-project/verl) and [vLLM](https://github.com/vllm-project/vllm). The training recipe uses [DAPO-MATH-17K](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k), and the evaluation code uses [Math-Verify](https://github.com/huggingface/Math-Verify). We thank the open-source community for these projects and resources.

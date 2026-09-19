# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def sample_corrected_ta_opd(
    taopd_with_sg: torch.Tensor,
    student_logp_y: torch.Tensor,
    teacher_logp_y: torch.Tensor,
    student_log_p_tail: torch.Tensor,
    teacher_log_q_tail: torch.Tensor,
    sampled_in_topk: torch.Tensor,
    behavior_logp_y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Add the sample correction from sample-corrected TA-OPD.

    ``taopd_with_sg`` is TA-OPD with a detached log-ratio. For a sampled token
    outside the teacher top-k set, the correction has value

        p(y) / sg(p(y)) * sg(log((p(y)/p_tail) / (q(y)/q_tail))).

    When ``behavior_logp_y`` is provided, the sampled correction uses the
    importance ratio ``p_theta(y) / p_behavior(y)``. This is needed when the
    same rollout batch is reused after an optimizer step.
    """
    if not (
        taopd_with_sg.shape
        == student_logp_y.shape
        == teacher_logp_y.shape
        == student_log_p_tail.shape
        == teacher_log_q_tail.shape
        == sampled_in_topk.shape
        and (behavior_logp_y is None or behavior_logp_y.shape == student_logp_y.shape)
    ):
        raise ValueError("All sample-corrected TA-OPD inputs must have the same shape")

    student_logp_y = student_logp_y.float()
    teacher_logp_y = teacher_logp_y.detach().float()
    student_log_p_tail = student_log_p_tail.detach().float()
    teacher_log_q_tail = teacher_log_q_tail.detach().float()
    if behavior_logp_y is not None:
        behavior_logp_y = behavior_logp_y.detach().float()
    sampled_outside_topk = (~sampled_in_topk.bool()).to(student_logp_y.dtype)

    # Tail log-probabilities come directly from log1mexp(logsumexp(top-k)),
    # so the correction never forms 1 - sum(exp(top-k log-probs)).
    conditional_log_ratio = (
        student_logp_y
        - student_log_p_tail
        - teacher_logp_y
        + teacher_log_q_tail
    ).detach()

    if behavior_logp_y is None:
        # On-policy p(y) / sg(p(y)); this avoids exp(log p) underflow and 0/0.
        sample_weight = torch.exp(student_logp_y - student_logp_y.detach())
    else:
        # Off-policy correction for samples reused after the behavior-policy
        # snapshot: p_theta(y) / p_behavior(y). Do not clip this ratio here:
        # clipping would turn the exact importance-sampling estimator into a
        # biased one. If variance control is needed, it should be exposed as
        # an explicit, documented estimator option rather than silently
        # changing the SC-TA objective.
        sample_weight = torch.exp(student_logp_y - behavior_logp_y)
    sample_correction = sampled_outside_topk * sample_weight * conditional_log_ratio
    return taopd_with_sg + sample_correction.to(dtype=taopd_with_sg.dtype)





def compute_reverse_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute reverse KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    # if loss_config.log_prob_min_clamp is not None:
    #     student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    #     teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_p=student_topk_log_probs, log_q=teacher_topk_log_probs)

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }




def compute_mopd_reverse_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """
    For student p, teacher q, and teacher top-k support S:

        L_MOPD = sum_{v in S} p(v) log[p(v) / q(v)]
                 + p_tail - q_tail.

    The tail correction is evaluated as ``teacher_mass - student_mass``,
    which is algebraically identical and avoids subtracting values near 1.
    """
    del config, data_format

    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)

    assert (
        teacher_topk_log_probs.shape[:2]
        == teacher_topk_ids.shape[:2]
        == student_logits.shape[:2]
    )

    # Compute only the student log probabilities needed on the teacher support.
    student_log_z = torch.logsumexp(student_logits, dim=-1).float()
    student_topk_logits = torch.gather(student_logits, dim=-1, index=teacher_topk_ids).float()
    student_topk_log_probs = student_topk_logits - student_log_z.unsqueeze(-1)
    teacher_topk_log_probs = teacher_topk_log_probs.float()

    student_topk_probs = student_topk_log_probs.exp()
    teacher_topk_probs = teacher_topk_log_probs.exp()
    student_mass = student_topk_probs.sum(dim=-1)
    teacher_mass = teacher_topk_probs.sum(dim=-1)

    support_kl = (
        student_topk_probs * (student_topk_log_probs - teacher_topk_log_probs)
    ).sum(dim=-1)
    distillation_losses = support_kl + teacher_mass - student_mass

    with torch.no_grad():
        student_topk_ids = torch.topk(
            student_logits,
            k=teacher_topk_ids.shape[-1],
            dim=-1,
        ).indices
        overlap_mask = (
            teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)
        ).any(dim=-1)
        overlap_count = overlap_mask.sum(dim=-1)

        token_kl = teacher_topk_probs * (
            teacher_topk_log_probs - student_topk_log_probs
        )
        overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
        overlap_token_advantage = torch.where(
            overlap_count > 0,
            overlap_token_advantage_sum / overlap_count.clamp_min(1),
            torch.zeros_like(overlap_token_advantage_sum),
        )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass.detach(),
        "teacher_mass": teacher_mass.detach(),
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }


def compute_normalized_reverse_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute normalized reverse KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd".

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)

    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise normalized KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    student_topk_ids = torch.topk(
        student_log_probs,
        k=teacher_topk_ids.shape[-1],
        dim=-1,
    ).indices

    # student probs on teacher top-k support
    student_topk_log_probs = torch.gather(
        student_log_probs,
        dim=-1,
        index=teacher_topk_ids,
    )


    # original masses before normalization, used for metrics
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

    # Normalize both teacher and student distributions on teacher top-k support.
    student_log_mass = torch.logsumexp(student_topk_log_probs, dim=-1, keepdim=True)
    teacher_log_mass = torch.logsumexp(teacher_topk_log_probs, dim=-1, keepdim=True)

    student_topk_log_probs = student_topk_log_probs - student_log_mass
    teacher_topk_log_probs = teacher_topk_log_probs - teacher_log_mass

    distillation_losses = kl_divergence(
        log_p=student_topk_log_probs,
        log_q=teacher_topk_log_probs,
    )

    # Diagnostics for tracking teacher/student top-k overlap in OPD.
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)

    token_kl = teacher_topk_log_probs.exp() * (
        teacher_topk_log_probs - student_topk_log_probs
    )

    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0,
        overlap_token_advantage,
        torch.zeros_like(overlap_token_advantage),
    )

    with torch.no_grad():
        log_student = student_topk_log_probs.float()  # log p_i
        log_teacher = teacher_topk_log_probs.float()  # log q_i
        student_prob = log_student.exp()  # p_i

        # E_{I~p}[p_I]
        e_student = (student_prob * student_prob).sum(dim=-1)

        # E_{I~p}[log p_I], E_{I~p}[log q_I]
        e_log_student = (student_prob * log_student).sum(dim=-1)
        e_log_teacher = (student_prob * log_teacher).sum(dim=-1)

        # E_{I~p}[p_I log p_I], E_{I~p}[p_I log q_I]
        e_student_log_student = (student_prob * student_prob * log_student).sum(dim=-1)
        e_student_log_teacher = (student_prob * student_prob * log_teacher).sum(dim=-1)

        # Cov_{I~p}(p_I, log p_I)
        cov_student_log_student = e_student_log_student - e_student * e_log_student

        # Cov_{I~p}(p_I, log q_I)
        cov_student_log_teacher = e_student_log_teacher - e_student * e_log_teacher

        # Cov_{I~p}(p_I, log p_I - log q_I)
        cov_student_minus_teacher_residual = (
                cov_student_log_student - cov_student_log_teacher
        )

        tail_weighted_cov = (
                student_mass * (1.0 - student_mass) * cov_student_minus_teacher_residual
        )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
        "cov_student_log_student": cov_student_log_student,
        "cov_student_log_teacher": cov_student_log_teacher,
        "cov_student_minus_teacher_residual": cov_student_minus_teacher_residual,
        "tail_weighted_cov": tail_weighted_cov,
    }



def compute_unnormalized_reverse_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute unnormalized reverse KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd".

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)

    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise normalized KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    student_topk_ids = torch.topk(
        student_log_probs,
        k=teacher_topk_ids.shape[-1],
        dim=-1,
    ).indices

    # student probs on teacher top-k support
    student_topk_log_probs = torch.gather(
        student_log_probs,
        dim=-1,
        index=teacher_topk_ids,
    )


    # original masses before normalization, used for metrics
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)

    distillation_losses = kl_divergence(
        log_p=student_topk_log_probs,
        log_q=teacher_topk_log_probs,
    )

    # Diagnostics for tracking teacher/student top-k overlap in OPD.
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)

    token_kl = teacher_topk_log_probs.exp() * (
        teacher_topk_log_probs - student_topk_log_probs
    )

    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0,
        overlap_token_advantage,
        torch.zeros_like(overlap_token_advantage),
    )

    with torch.no_grad():
        log_student = student_topk_log_probs.float()  # log p_i
        log_teacher = teacher_topk_log_probs.float()  # log q_i
        student_prob = log_student.exp()  # p_i

        # E_{I~p}[p_I]
        e_student = (student_prob * student_prob).sum(dim=-1)

        # E_{I~p}[log p_I], E_{I~p}[log q_I]
        e_log_student = (student_prob * log_student).sum(dim=-1)
        e_log_teacher = (student_prob * log_teacher).sum(dim=-1)

        # E_{I~p}[p_I log p_I], E_{I~p}[p_I log q_I]
        e_student_log_student = (student_prob * student_prob * log_student).sum(dim=-1)
        e_student_log_teacher = (student_prob * student_prob * log_teacher).sum(dim=-1)

        # Cov_{I~p}(p_I, log p_I)
        cov_student_log_student = e_student_log_student - e_student * e_log_student

        # Cov_{I~p}(p_I, log q_I)
        cov_student_log_teacher = e_student_log_teacher - e_student * e_log_teacher

        # Cov_{I~p}(p_I, log p_I - log q_I)
        cov_student_minus_teacher_residual = (
                cov_student_log_student - cov_student_log_teacher
        )

        tail_weighted_cov = (
                student_mass * (1.0 - student_mass) * cov_student_minus_teacher_residual
        )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
        "cov_student_log_student": cov_student_log_student,
        "cov_student_log_teacher": cov_student_log_teacher,
        "cov_student_minus_teacher_residual": cov_student_minus_teacher_residual,
        "tail_weighted_cov": tail_weighted_cov,
    }






import math
import torch
import torch.nn.functional as F

_LOG2 = math.log(2.0)


def log1mexp(log_x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    log_x = log_x.clamp_max(-eps)
    cond = log_x > -_LOG2
    a_in = torch.where(cond, log_x, torch.full_like(log_x, -_LOG2))
    b_in = torch.where(cond, torch.full_like(log_x, -_LOG2), log_x)
    out_a = torch.log(-torch.expm1(a_in))   # (-log2, 0]
    out_b = torch.log1p(-torch.exp(b_in))   # (-inf, -log2]
    return torch.where(cond, out_a, out_b)


def compute_taopd_with_sg(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Compute TA-OPD with a stop-gradient log-ratio.

    The forward value is reverse KL on the teacher top-k tokens plus one
    collapsed tail token. Its log-ratio is detached exactly as required by
    ``sum_v p(v) sg(log(p(v) / q(v)))``.
    """
    del config, data_format

    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)

    assert (
        teacher_topk_log_probs.shape[:2]
        == teacher_topk_ids.shape[:2]
        == student_logits.shape[:2]
    )

    student_log_z = torch.logsumexp(student_logits, dim=-1).float()
    student_topk_logits = torch.gather(
        student_logits, dim=-1, index=teacher_topk_ids
    ).float()
    student_topk_log_probs = student_topk_logits - student_log_z.unsqueeze(-1)
    teacher_topk_log_probs = teacher_topk_log_probs.float()

    student_log_mass = torch.logsumexp(student_topk_log_probs, dim=-1)
    teacher_log_mass = torch.logsumexp(teacher_topk_log_probs, dim=-1)
    student_log_p_tail = log1mexp(student_log_mass)
    teacher_log_q_tail = log1mexp(teacher_log_mass)

    student_augmented_log_probs = torch.cat(
        [student_topk_log_probs, student_log_p_tail.unsqueeze(-1)], dim=-1
    )
    teacher_augmented_log_probs = torch.cat(
        [teacher_topk_log_probs, teacher_log_q_tail.unsqueeze(-1)], dim=-1
    )
    log_ratio = (student_augmented_log_probs - teacher_augmented_log_probs).detach()
    distillation_losses = (student_augmented_log_probs.exp() * log_ratio).sum(dim=-1)

    with torch.no_grad():
        student_mass = student_log_mass.exp()
        teacher_mass = teacher_log_mass.exp()
        student_topk_ids = torch.topk(
            student_logits, k=teacher_topk_ids.shape[-1], dim=-1
        ).indices
        overlap_mask = (
            teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)
        ).any(dim=-1)
        overlap_count = overlap_mask.sum(dim=-1)
        token_kl = teacher_topk_log_probs.exp() * (
            teacher_topk_log_probs - student_topk_log_probs
        )
        overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
        overlap_token_advantage = torch.where(
            overlap_count > 0,
            overlap_token_advantage_sum / overlap_count.clamp_min(1),
            torch.zeros_like(overlap_token_advantage_sum),
        )

    return {
        "taopd_with_sg": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "student_log_p_tail": student_log_p_tail.detach(),
        "teacher_log_q_tail": teacher_log_q_tail.detach(),
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }



def compute_taopd_reverse_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """TA-OPD per-token loss: reverse KL on the teacher's top-k tokens plus a tail token.

    Computes KL(p || q) over the augmented token set S+ = TopK(q, k) U {v_tail}, where the
    tail token carries the probability mass outside the teacher's top-k tokens:
        loss = sum_{v in S} p(v) log(p(v)/q(v)) + p_tail * log(p_tail / q_tail).

    Everything is done in log space: the tail probabilities are never formed in probability
    space, which would be unstable when the tail mass is near zero (see Appendix, numerically
    stable computation of TA-OPD).

    Args:
        student_logits: (1, nnz, vocab) student logits over the full vocabulary.
        teacher_topk_log_probs: nested (bsz, seqlen, topk) teacher log q(v) on its top-k tokens.
        teacher_topk_ids: nested (bsz, seqlen, topk) vocabulary ids of the teacher's top-k tokens.

    Returns:
        dict with the per-token loss and no-grad training diagnostics.
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)

    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # log-normalizer of the student's full-vocabulary softmax, used to turn logits into log-probs.
    student_logsumexp = torch.logsumexp(student_logits, dim=-1).float()  # (1, nnz)

    student_topk_logits = torch.gather(
        student_logits,
        dim=-1,
        index=teacher_topk_ids,
    ).float()  # (1, nnz, topk) — upcast only after slicing to k columns
    student_topk_log_probs = student_topk_logits - student_logsumexp.unsqueeze(-1)

    teacher_topk_log_probs = teacher_topk_log_probs.float()

    # log of the total probability on the top-k tokens: log(1 - tail). logsumexp is stable
    # and cannot overflow, unlike exponentiating and summing the top-k probabilities.
    teacher_log_mass = torch.logsumexp(teacher_topk_log_probs, dim=-1)  # (1, nnz)
    student_log_mass = torch.logsumexp(student_topk_log_probs, dim=-1)

    # Tail log-probability log(tail) = log(1 - exp(log(1 - tail)))
    teacher_tail_log_prob = log1mexp(teacher_log_mass).unsqueeze(-1)
    student_tail_log_prob = log1mexp(student_log_mass).unsqueeze(-1)

    # Append the tail token to build the augmented set S+
    teacher_augmented_log_probs = torch.cat(
        [teacher_topk_log_probs, teacher_tail_log_prob],
        dim=-1,
    )
    student_augmented_log_probs = torch.cat(
        [student_topk_log_probs, student_tail_log_prob],
        dim=-1,
    )

    # KL(student || teacher)
    distillation_losses = kl_divergence(
        log_p=student_augmented_log_probs,
        log_q=teacher_augmented_log_probs,
    )

    # ============================================================
    # Training metrics. kept entirely out of the autograd graph
    # ============================================================
    with torch.no_grad():
        # Probability the two models place on the teacher's top-k tokens; 1 - mass is the
        # tail probability plotted in the training-dynamics figures.
        teacher_mass = teacher_log_mass.exp()  # (bsz, seqlen/sp_size)
        student_mass = student_log_mass.exp()  # (bsz, seqlen/sp_size)

        # Student's own top-k tokens, used only to measure student-teacher top-k overlap.
        student_topk_ids = torch.topk(
            student_logits,
            k=teacher_topk_ids.shape[-1],
            dim=-1,
        ).indices

        # For each teacher top-k token, whether it also appears in the student's top-k set.
        # Broadcasting compares every teacher id against every student id.
        overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
        overlap_count = overlap_mask.sum(dim=-1)

        # Per-token forward-KL contribution q(v) * log(q(v)/p(v)), a diagnostic only.
        # only use real teacher top-k tokens here, not the collapsed other token
        token_kl = teacher_topk_log_probs.exp() * (
            teacher_topk_log_probs - student_topk_log_probs
        )

        # Average of -token_kl over the overlapping tokens; positions with no overlap get 0.
        overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
        overlap_token_advantage = torch.where(
            overlap_count > 0,
            overlap_token_advantage_sum / overlap_count.clamp_min(1),
            torch.zeros_like(overlap_token_advantage_sum),
        )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }

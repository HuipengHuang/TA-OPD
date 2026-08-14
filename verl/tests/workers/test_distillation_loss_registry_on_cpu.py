# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from types import SimpleNamespace

import pytest
import torch

from verl.trainer.distillation import losses as distillation_losses
from verl.workers.config.distillation import DistillationLossConfig


@pytest.mark.parametrize("loss_mode", ["mopd_topk", "mopd_reverse_kl_topk"])
def test_mopd_modes_are_registered(loss_mode):
    config = DistillationLossConfig(loss_mode=loss_mode)

    assert config.loss_settings.use_topk
    assert not config.loss_settings.use_sampled_logprob


@pytest.mark.parametrize(
    "loss_mode",
    [
        "sample_corrected_ta_opd",
        "sc_ta_opd",
        "sample_corrected_reverse_kl_topk",
    ],
)
def test_sample_corrected_ta_opd_modes_are_registered(loss_mode):
    config = DistillationLossConfig(loss_mode=loss_mode)

    assert config.loss_settings.use_topk
    assert config.loss_settings.use_sampled_logprob


def test_sample_corrected_ta_opd_reports_overlap_metrics(monkeypatch):
    monkeypatch.setattr(distillation_losses, "no_padding_2_padding", lambda tensor, data: tensor)

    model_output = {
        "taopd_with_sg": torch.tensor([[0.2, 0.3]], requires_grad=True),
        "student_mass": torch.tensor([[0.7, 0.8]]),
        "teacher_mass": torch.tensor([[0.6, 0.9]]),
        "student_log_p_tail": torch.tensor([[-1.2, -1.6]]),
        "teacher_log_q_tail": torch.tensor([[-1.0, -1.8]]),
        "overlap_count": torch.tensor([[2, 1]]),
        "overlap_token_advantage": torch.tensor([[0.4, 0.2]]),
        "log_probs": torch.tensor([[-2.0, -2.5]], requires_grad=True),
    }
    data = {
        "teacher_sampled_logprob": torch.tensor([[[-2.2], [-2.7]]]),
        "teacher_ids": torch.tensor([[[3, 4], [5, 6]]]),
        "responses": torch.tensor([[3, 7]]),
        "response_mask": torch.tensor([[True, True]]),
    }
    config = SimpleNamespace(distillation_loss=SimpleNamespace(topk=2))

    _, metrics = distillation_losses.compute_sample_corrected_ta_opd_loss(
        config=None,
        distillation_config=config,
        model_output=model_output,
        data=data,
    )

    assert metrics["distillation/overlap_ratio"] == pytest.approx(0.75)
    assert metrics["distillation/overlap_token_advantage"] == pytest.approx(0.3)

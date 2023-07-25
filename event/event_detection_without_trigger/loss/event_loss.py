#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
损失函数

Authors: panxu(panxu@baidu.com)
Date:    2020/06/12 17:07:00
"""
import torch
from torch.nn import MSELoss

from easytext.loss import Loss
from easytext.model import ModelOutputs


class EventLoss(Loss):
    """
    事件识别模型的 Loss 计算
    """

    def __init__(self):
        super().__init__()
        self._loss = MSELoss(reduction="mean")

    def __call__(self, model_outputs: ModelOutputs, golden_label: torch.Tensor) -> torch.Tensor:

        # 计算loss, 注意，这里的loss，后续 follow paper 要修改成带有 beta 的loss.
        logits = model_outputs.logits

        assert logits.dim() == 1, f"logits dim: {logits.dim()} != 1, 应该是 (B,)"
        assert logits.shape, golden_label.shape

        loss = self._loss(logits, golden_label.float())

        return loss





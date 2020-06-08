#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
acc 指标

Authors: panxu(panxu@baidu.com)
Date:    2020/05/30 07:36:00
"""
from typing import Dict

import torch

from .metric import Metric


class AccMetric(Metric):
    """
    Acc Metric
    """

    ACC = "acc"

    def __init__(self):
        super().__init__()
        self._num_true = 0
        self._num_total = 0

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: torch.LongTensor) -> Dict:

        if mask is not None:
            raise RuntimeError("对于 Acc 来说, mask 应该为 None")

        predictions, gold_labels = predictions.detach().cpu(), gold_labels.detach().cpu()

        predict_labels = torch.argmax(predictions, dim=-1)

        num_true = (predict_labels == gold_labels).sum().item()
        num_total = gold_labels.size(0)

        self._num_true += num_true
        self._num_total += num_total

        acc = AccMetric._compute(num_true=num_true, num_total=num_total)
        return {AccMetric.ACC: acc}

    @staticmethod
    def _compute(num_true: int, num_total: int):
        return num_true / (float(num_total) + 1e-10)

    @property
    def metric(self) -> Dict:
        acc = AccMetric._compute(self._num_true, self._num_total)

        return {AccMetric.ACC: acc}

    def reset(self):
        self._num_true = 0
        self._num_total = 0
        return self


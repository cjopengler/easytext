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

from easytext.metrics import Metric


class _AccMetricData:

    def __init__(self):
        self.num_true = 0
        self.num_total = 0

    def to_tensor(self) -> torch.LongTensor:
        return torch.tensor([self._num_true, self._num_total], dtype=torch.long)

    def update_from_tensor(self, values: torch.LongTensor) -> None:
        self.num_true = values[0]
        self.num_total = values[1]


class AccMetric(Metric):
    """
    Acc Metric
    """

    ACC = "acc"

    def __init__(self, is_distributed: bool = False):
        super().__init__(is_distributed=is_distributed)

        self._num_true = 0
        self._num_total = 0
        self._data = _AccMetricData()

    def __call__(self,
                 prediction_labels: torch.Tensor,
                 gold_labels: torch.Tensor, mask: torch.LongTensor) -> Dict:
        """
        Acc metric 计算
        :param prediction_labels: 预测的标签
        :param gold_labels: gold 标签
        :param mask:
        :return:
        """

        if mask is not None:
            raise RuntimeError("对于 Acc 来说, mask 应该为 None")

        prediction_labels, gold_labels = prediction_labels.detach().cpu(), gold_labels.detach().cpu()

        num_true = (prediction_labels == gold_labels).sum().item()
        num_total = gold_labels.size(0)

        self._data.num_true += num_true
        self._data.num_total += num_total

        acc = AccMetric._compute(num_true=num_true, num_total=num_total)
        return {AccMetric.ACC: acc}

    @staticmethod
    def _compute(num_true: int, num_total: int):
        return num_true / (float(num_total) + 1e-10)

    def _distributed_data(self):
        distributed_tensor = self._data.to_tensor()

        if torch.distributed.get_backend() == "nccl":
            distributed_tensor.to(torch.distributed.get_rank())

        torch.distributed.all_reduce(tensor=distributed_tensor)

        self._data.update_from_tensor(distributed_tensor)

    @property
    def metric(self) -> Dict:

        if self.is_distributed:
            self._distributed_data()

        acc = AccMetric._compute(self._data.num_true,
                                 self._data.num_total)

        return {AccMetric.ACC: acc}

    def reset(self):
        self._data = _AccMetricData()
        return self


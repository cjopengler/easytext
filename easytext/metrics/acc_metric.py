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
from typing import Dict, Union, List, Tuple

import torch
from torch import Tensor
from torch.distributed import ReduceOp

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

    def to_synchronized_data(self) -> Tuple[Union[Dict[Union[str, int], Tensor], List[Tensor], Tensor], ReduceOp]:
        sync_data = torch.tensor([self._num_true, self._num_total], dtype=torch.long)
        return sync_data, ReduceOp.SUM

    def from_synchronized_data(self, sync_data: Union[Dict[Union[str, int], Tensor], List[Tensor], Tensor],
                               reduce_op: ReduceOp) -> None:

        self._num_true = sync_data[0].item()
        self._num_total = sync_data[1].item()




#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
f1 metric 基类，具体的 f1 需要子类实现

Authors: panxu(panxu@baidu.com)
Date:    2020/06/16 09:08:00
"""

from typing import Dict, Set, List
from collections import OrderedDict

import torch

from easytext.metrics import Metric


class _F1MetricData:

    def __init__(self, labels: List[str]):
        """
        初始化 F1MetricData
        :param labels: 最后结果中，需要查看的 F1 值的 label 列表
        """
        self.labels = labels
        # 下面之所以是字典，是为了计算多个 label 的 f1
        self.true_positives: Dict[str, int] = OrderedDict()
        self.false_positives: Dict[str, int] = OrderedDict()
        self.false_negatives: Dict[str, int] = OrderedDict()

        for label in labels:
            self.true_positives[label] = 0
            self.false_negatives[label] = 0
            self.false_negatives[label] = 0

    def to_tensor(self) -> torch.LongTensor:
        """
        转换成 tensor
        :return: 转换后的 tensor value
        """
        true_positive_values = [value for _, value in self.true_positives.items()]
        false_positive_values = [value for _, value in self.false_positives.items()]
        false_negative_values = [value for _, value in self.false_negatives]

        return torch.tensor([true_positive_values, false_positive_values, false_negative_values],
                            dtype=torch.long)

    def update_from_tensor(self, values: torch.LongTensor) -> None:
        """
        使用 tensor value 填充
        :param values: to_tensor 生成 value
        :return: self
        """
        for i, label in enumerate(self.labels):
            self.true_positives[label] = values[0][i].item()
            self.false_positives[label] = values[1][i].item()
            self.false_negatives[label] = values[2][i].item()


class F1Metric(Metric):
    """
    f1 metric 基类，具体的 f1 需要子类实现，来完成
    _true_positives, _false_positives, _false_negatives
    三个字典的填充, 但是注意: 对于 *-overall 是会自动计算的，不需要
    指定计算。
    """

    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"

    PRECISION_OVERALL = f"{PRECISION}-overall"
    RECALL_OVERALL = f"{RECALL}-overall"
    F1_OVERALL = f"{F1}-overall"

    def __init__(self, labels: List[str], is_distributed: bool) -> None:
        """
        初始化
        :param labels: 最终输出的 F1 的 label
        :param is_distributed: True: 分布式 metric; False: 非分布式 metric
        """
        super().__init__(is_distributed=is_distributed)
        self._labels = labels
        self._data = _F1MetricData(labels=labels)

    def __call__(self,
                 prediction_labels: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: torch.LongTensor) -> Dict:
        """
        计算 metric. 返回的是 F1 字典:

        {"precision_[tag]": [value],
         "recall_[tag]" : [value],
         "f1-measure_[tag]": [value],
         "precision-overall": [value],
         "recall-overall": [value],
         "f1-measure-overall": [value]}

         其中的 [tag] 是 实际返回的tag

        :param prediction_labels: 预测的结果, shape: (B, SeqLen)
        :param gold_labels: 实际的结果, shape: (B, SeqLen)
        :param mask: 对 predictions 和 gold label 的 mask, shape: (B, SeqLen)
        :return: 当前的 metric 计算字典结果.
        """

        raise NotImplementedError()

    def _metric(self,
                true_positives: Dict[str, int],
                false_positives: Dict[str, int],
                false_negatives: Dict[str, int]) -> Dict:
        """
        计算 metric, 注意是 输入是的字典
        :param true_positives:
        :param false_positives:
        :param false_negatives:
        :return:
        """

        all_tags: Set[str] = set()
        all_tags.update(true_positives.keys())
        all_tags.update(false_positives.keys())
        all_tags.update(false_negatives.keys())
        all_metrics = {}

        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(true_positives[tag],
                                                                  false_positives[tag],
                                                                  false_negatives[tag])
            precision_key = f"{F1Metric.PRECISION}-{tag}"
            recall_key = f"{F1Metric.RECALL}-{tag}"
            f1_key = f"{F1Metric.F1}-{tag}"
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for overall
        precision, recall, f1_measure = self._compute_metrics(sum(true_positives.values()),
                                                              sum(false_positives.values()),
                                                              sum(false_negatives.values()))
        all_metrics[F1Metric.PRECISION_OVERALL] = precision
        all_metrics[F1Metric.RECALL_OVERALL] = recall
        all_metrics[F1Metric.F1_OVERALL] = f1_measure

        return all_metrics

    def _distributed_data(self):
        distributed_tensor = self._data.to_tensor()

        if torch.distributed.get_backend() == "nccl":
            distributed_tensor.to(torch.distributed.get_rank())

        torch.distributed.all_reduce(tensor=distributed_tensor)

        self._data.update_from_tensor(distributed_tensor)

    @property
    def metric(self) -> Dict:
        """
        :return

        一个包含所有metric的字典:
        {
            precision-[tag] : float
            recall-[tag] : float
            f1-[tag] : float
            ...
        }

        另外注意 precision-overall, recall-overall, f1-overall 是所有的综合指标, 这是非常有必要的
        因为有多个tag，那么模型的指标衡量需要一个综合指标来衡量。
        """

        if self.is_distributed:
            self._distributed_data()

        return self._metric(true_positives=self._data.true_positives,
                            false_positives=self._data.false_positives,
                            false_negatives=self._data.false_negatives)

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        """
        计算 f1 metric
        """
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        """
        将所有的状态reset, f1 重新计算。
        """
        self._data = _F1MetricData(labels=self._data.labels)
        return self


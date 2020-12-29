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

from typing import Dict, Set, List, Union, Tuple
from collections import OrderedDict

import torch
from torch import Tensor
from torch.distributed import ReduceOp
from torch import distributed as TorchDist


from easytext.metrics import Metric


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

    def __init__(self, labels: List[str]) -> None:
        """
        初始化
        :param labels: 最终输出的 F1 的 label
        :param is_distributed: True: 分布式 metric; False: 非分布式 metric
        """
        super().__init__()
        self._labels = labels

        # 下面之所以是字典，是为了计算多个 label 的 f1
        self._true_positives: Dict[str, int] = OrderedDict()
        self._false_positives: Dict[str, int] = OrderedDict()
        self._false_negatives: Dict[str, int] = OrderedDict()

        for label in labels:
            self._true_positives[label] = 0
            self._false_positives[label] = 0
            self._false_negatives[label] = 0

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

        return self._metric(true_positives=self._true_positives,
                            false_positives=self._false_positives,
                            false_negatives=self._false_negatives)

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        """
        计算 f1 metric
        """   
        precision = float(true_positives) / (float(true_positives + false_positives) + 1e-13)
        recall = float(true_positives) / (float(true_positives + false_negatives) + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        """
        将所有的状态reset, f1 重新计算。
        """
        for label in self._labels:
            self._true_positives[label] = 0
            self._false_positives[label] = 0
            self._false_negatives[label] = 0
        return self

    def to_synchronized_data(self) -> Tuple[Union[Dict[Union[str, int], Tensor], List[Tensor], Tensor], ReduceOp]:
        true_positives = torch.tensor([v for _, v in self._true_positives.items()], dtype=torch.long)

        false_positives = torch.tensor([v for _, v in self._false_positives.items()], dtype=torch.long)
        false_negatives = torch.tensor([v for _, v in self._false_negatives.items()], dtype=torch.long)

        sync_data = {"true_positives": true_positives,
                     "false_positives": false_positives,
                     "false_negatives": false_negatives}
        return sync_data, ReduceOp.SUM

    def from_synchronized_data(self, sync_data: Union[Dict[Union[str, int], Tensor], List[Tensor], Tensor],
                               reduce_op: ReduceOp) -> None:

        true_positives: Tensor = sync_data["true_positives"]
        false_positives = sync_data["false_positives"]
        false_negatives = sync_data["false_negatives"]

        assert true_positives.size(0) == len(self._true_positives), \
            f"true_positives length: {true_positives.size(0)} sync data " \
            f"与 self.true_positives length: {len(self._true_positives)} 不一致"

        assert false_positives.size(0) == len(self._false_positives), \
            f"true_positives length: {false_positives.size(0)} sync data " \
            f"与 self.true_positives length: {len(self._false_positives)} 不一致"

        assert false_negatives.size(0) == len(self._false_negatives), \
            f"true_positives length: {false_negatives.size(0)} sync data " \
            f"与 self.true_positives length: {len(self._false_negatives)} 不一致"

        for kv, sync_value in zip(self._true_positives.items(), true_positives):
            k, _ = kv
            self._true_positives[k] = sync_value

        for kv, sync_value in zip(self._false_positives.items(), false_positives):
            k, _ = kv
            self._false_positives[k] = sync_value

        for kv, sync_value in zip(self._false_negatives.items(), false_negatives):
            k, _ = kv
            self._false_negatives[k] = sync_value



#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
f1 measure

Authors: panxu(panxu@baidu.com)
Date:    2020/02/06 14:51:00
"""
import logging
from typing import Union, Tuple, Dict, List, Optional

import torch
from allennlp.training.metrics import Metric


class EventDetectionWithoutKeywordF1Measure(Metric):
    """
    Event Detection Without Keyword F1 Measure
    """

    def __init__(self, event_type: str):
        super().__init__()

        self._true_positive = 0

        self._pred_positive = 0

        self._ori_positive = 0
        self._event_type = event_type

    @property
    def event_type(self):
        return self._event_type

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        计算f1。 predictions 和 gold_labels 看起来是 [1, 1, 0, 0]，通过设置不同的mask，来计算不同的事件类型的F1.
        :param predictions: 预测结果向量. Shape: B (BatchSize). 每一个元素 属于 {0, 1}.
        :param gold_labels: gold labels. Shape: B. 每一个元素 属于 {0, 1}
        :param mask: 将某些事件类型屏蔽掉，比如 negative 的事件类型屏蔽掉. 如果单独计算某个事件类型，
        那么就是，用这个事件类型做出mask即可。
        :return:
        """

        if mask is None:
            mask = torch.ones(predictions.size(0), dtype=torch.long)
            logging.debug(f"mask dtype: {mask.dtype}")

        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions,
                                                                gold_labels,
                                                                mask)

        # 将mask的保留，其他的设置为 0
        logging.debug(f"mask dtype: {mask.dtype}, predictions dtype: {predictions.dtype}")
        predictions = predictions * mask
        gold_labels = gold_labels * mask

        # predictions 和 gold_labels 看起来是 [1, 1, 0, 0].
        # 要计算的是 predictions 中是: 1 或者 gold_labels是: 1 的部分.
        # true_positive: 只有2者都是1，才是, 所以用 element wise乘法，在求和，实际就是向量 点乘。
        self._true_positive += torch.dot(predictions, gold_labels)

        self._pred_positive += torch.sum(predictions).item()
        self._ori_positive += torch.sum(gold_labels).item()

    def precision_key(self):
        return f"precision_{self._event_type}"

    def recall_key(self):
        return f"recall_{self._event_type}"

    def f1_key(self):
        return f"f1_{self._event_type}"

    def get_metric(self, reset: bool) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        precision = float(self._true_positive) / (float(self._pred_positive) + 1e-13)
        recall = float(self._true_positive) / (float(self._ori_positive) + 1e-13)
        f1 = float(2 * precision * recall / (precision + recall + 1e-13))

        if reset:
            self.reset()

        return {
            self.precision_key(): precision,
            self.recall_key(): recall,
            self.f1_key(): f1
        }

    def reset(self) -> None:
        self._true_positive = 0
        self._pred_positive = 0
        self._ori_positive = 0


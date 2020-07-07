#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2020/05/16 00:59:00
"""

from typing import List, Dict, Tuple

import torch
from torch import Tensor

from easytext.model import ModelOutputs


class Metric:
    """
    Metrics
    """

    def __call__(self, prediction_labels: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: torch.LongTensor) -> Dict:
        """
        返回metric的结果, 依然来说，因为 model_outputs 是一个bath产生的结果，
        所以这里返回的是一个 batch 的 metric 结果。
        :param prediction_labels: 预测的label 结果，一般是 label index，这与 gold_labels 是一致的
        :param gold_labels: 正确的结果
        :param mask: prediction 的 mask. 这里的 mask 类型是 Long, 不是 bool.
        :return:
        """
        raise NotImplementedError()

    @property
    def metric(self) -> Dict:
        """
        返回全部计算的 metric, 也就是将每一个 batch 计算的metric合并到一起的，总的 metric
        :return:
        """
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class ModelTargetMetric:
    """
    模型的目标 metric 数据，作为模型唯一度量的metric， 因为 metric 本身会返回多个metric,
    但是模型需要一个唯一的 metric 作为评估以及 early stopping 等。
    """

    def __init__(self, metric_name: str, metric_value: float):
        self._metric_name = metric_name
        self._metric_value = metric_value

    @property
    def name(self):
        return self._metric_name

    @property
    def value(self):
        return self._metric_value


class ModelMetricAdapter:
    """
    模型的 metric 在 __call__ 的属于与 metric 不同,
    """

    def __call__(self,
                 model_outputs: ModelOutputs,
                 golden_labels: Tensor) -> Tuple[Dict, ModelTargetMetric]:
        """
        在每一个 batch 中 计算metric
        :param model_outputs:
        :param golden_labels:
        :return: 当前batch下的 metric 字典值, 以及 Model Target Metric.
        Model Target Metric 的含义是用来对该模型的唯一评判指标
        """
        raise NotImplementedError()

    @property
    def metric(self) -> Tuple[Dict, ModelTargetMetric]:
        """
        将所有 batch 的 metric 结果整合在一起计算的 metric.
        一般会在一个 epoch 结束调用这个函数来，得到这个 epoch 的 metric结果。
        如果每一个 batch 结束后调用，得到的是前面所有 batch 到现在的综合结果。
        :return: 1. Metric Dict: 作为模型最详细的metric，包括每一个细分的 metric, 例如 sequence label, 返回每一个label
        的 metric
                 2. Model Target Metric: 作为模型的唯一评估指标，训练时候会被用在 early stopping 中。
        """
        raise NotImplementedError()

    def reset(self) -> "ModelMetricAdapter":
        """
        所有数据重置, 一般是在每一个 epoch 的开始 做这个操作，将所有数据清零。
        :return: self
        """
        raise NotImplementedError()



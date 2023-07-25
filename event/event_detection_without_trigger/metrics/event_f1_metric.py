#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
计算 event detection f1 metric

Authors: panxu(panxu@baidu.com)
Date:    2020/06/15 19:17:00
"""
from typing import Tuple, Dict

import torch
from torch import Tensor

from easytext.metrics import Metric, ModelMetricAdapter, ModelTargetMetric
from easytext.metrics import LabelF1Metric, F1Metric
from easytext.model import ModelOutputs
from easytext.data import Vocabulary

from event.event_detection_without_tirgger.models import EventModelOutputs


class EventF1MetricAdapter(ModelMetricAdapter):
    """
    Event F1 Metric Adapter
    F1 计算，因为每一个样本是 pair <event_type, label>。其中:
    event_type: 所有的事件类型 排除了 Negative 类型，(Negative 对应  event_type_vocabulary.unk)
    label: label 在 {0, 1}

    所以 F1 的计算是计算所有 event type 下 label == 1 的 F1 值。

    返回的结果是字典:
    {
        "precision-[event_type]": [value],
        "recall-[event_type]": [value],
        "f1-[event_type]": [value],
        ...
        "precision-overall": [value],
        "recall-overall": [value],
        "f1-overall": [value],
    }
    """

    __OVERALL = "overall"

    def __init__(self, event_type_vocabulary: Vocabulary):
        """
        初始化
        :param event_type_vocabulary: event type vocabulary
        """
        super().__init__()

        self._event_type_f1: Dict[str, LabelF1Metric] = dict()

        for index in range(0, event_type_vocabulary.size):
            event_type = event_type_vocabulary.token(index)

            if event_type != event_type_vocabulary.unk:
                self._event_type_f1[event_type] = LabelF1Metric(labels=[1],
                                                                label_vocabulary=None)

        self._event_type_f1[EventF1MetricAdapter.__OVERALL] = LabelF1Metric(labels=[1], label_vocabulary=None)
        self._event_type_vocabulary = event_type_vocabulary

    def __call__(self,
                 model_outputs: EventModelOutputs,
                 golden_labels: Tensor) -> Tuple[Dict, ModelTargetMetric]:

        metrics = dict()
        logits = model_outputs.logits.detach().cpu()
        event_type_indices = model_outputs.event_type.detach().cpu()
        golden_labels = golden_labels.detach().cpu()

        assert logits.dim() == 1, f"logits shape 应该是 (B,), 现在 dim 是: {logits.dim()}"
        assert event_type_indices.dim() == 1, f"event_type shape 应该是 (B,), 现在 dim 是: {logits.dim()}"
        assert golden_labels.dim() == 1, f"golden_labels shape 应该是 (B,), 现在 dim 是: {golden_labels.dim()}"

        predictions = (logits > 0.5).long()

        for event_type, f1_metric in self._event_type_f1.items():

            if event_type == EventF1MetricAdapter.__OVERALL:
                negative_event_type_index = self._event_type_vocabulary.index(self._event_type_vocabulary.unk)
                mask = (event_type_indices != negative_event_type_index).long()
            else:
                event_type_index = self._event_type_vocabulary.index(event_type)
                mask = (event_type_indices == event_type_index).long()
            event_type_metric = f1_metric(prediction_labels=predictions, gold_labels=golden_labels, mask=mask)

            event_type_metric = EventF1MetricAdapter._event_metric_from(event_type,
                                                                        event_type_metric)
            metrics.update(event_type_metric)

        target_metric = ModelTargetMetric(metric_name=F1Metric.F1_OVERALL,
                                          metric_value=metrics[F1Metric.F1_OVERALL])

        return metrics, target_metric

    @staticmethod
    def _event_metric_from(event_type: str, label_f1_metric: Dict[str, float]):
        """
        label_f1_metric 的结果是 以为 label 为 1 作为字典的 key, 例如: precision-1.
        那么，需要 key替换成 precision-[event_type] 这种形式。要替换的是三个:
        precision-1, recall-1, f1-1
        :param event_type: 事件类型
        :param label_f1_metric: label f1 的计算结果, 该结果是以 label 为key的，在这里 是 1，
        那么需要将 1 替换成 event type.
        """

        metrics: Dict[str, float] = dict()
        for k, v in label_f1_metric.items():

            if k.startswith(F1Metric.PRECISION):
                if event_type == EventF1MetricAdapter.__OVERALL:
                    metrics[F1Metric.PRECISION_OVERALL] = v
                else:
                    metrics[f"{F1Metric.PRECISION}-{event_type}"] = v
            elif k.startswith(F1Metric.RECALL):
                if event_type == EventF1MetricAdapter.__OVERALL:
                    metrics[F1Metric.RECALL_OVERALL] = v
                else:
                    metrics[f"{F1Metric.RECALL}-{event_type}"] = v
            elif k.startswith(F1Metric.F1):
                if event_type == EventF1MetricAdapter.__OVERALL:
                    metrics[F1Metric.F1_OVERALL] = v
                else:
                    metrics[f"{F1Metric.F1}-{event_type}"] = v
            else:
                raise RuntimeError(f"Invalidate metric key: {k}")

        return metrics

    @property
    def metric(self) -> Tuple[Dict, ModelTargetMetric]:
        metrics: Dict[str, float] = dict()
        for event_type, f1_metric in self._event_type_f1.items():
            event_metric = EventF1MetricAdapter._event_metric_from(event_type, f1_metric.metric)
            metrics.update(event_metric)

        target_metric = ModelTargetMetric(metric_name=F1Metric.F1_OVERALL,
                                          metric_value=metrics[F1Metric.F1_OVERALL])
        return metrics, target_metric

    def reset(self) -> "EventF1MetricAdapter":
        for _, f1_metric in self._event_type_f1.items():
            f1_metric.reset()
        return self

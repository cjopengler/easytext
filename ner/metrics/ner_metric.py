#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
ner metric

Authors: PanXu
Date:    2020/06/27 20:48:00
"""
from typing import Tuple, Dict

from torch import Tensor

from easytext.metrics import ModelMetricAdapter, ModelTargetMetric
from easytext.metrics import SpanF1Metric
from easytext.model import ModelOutputs
from easytext.data import LabelVocabulary
from easytext.label_decoder import ModelLabelDecoder

from ner.models import NerModelOutputs


class NerModelMetricAdapter(ModelMetricAdapter):
    """
    Ner Model Metric Adapter
    计算 Ner Model Metric
    """

    def __init__(self,
                 label_vocabulary: LabelVocabulary,
                 model_label_decoder: ModelLabelDecoder):
        self.span_f1_metric = SpanF1Metric(label_vocabulary)
        self.model_label_decoder = model_label_decoder

    def __call__(self, model_outputs: ModelOutputs, golden_labels: Tensor) -> Tuple[Dict, ModelTargetMetric]:
        model_outputs: NerModelOutputs = model_outputs

        prediction_labels = self.model_label_decoder.decode_label_index(model_outputs=model_outputs)

        metric_dict = self.span_f1_metric(prediction_labels=prediction_labels,
                            gold_labels=golden_labels,
                            mask=model_outputs.mask)

        target_metric = ModelTargetMetric(metric_name=SpanF1Metric.F1_OVERALL,
                                          metric_value=metric_dict[SpanF1Metric.F1_OVERALL])

        return metric_dict, target_metric

    @property
    def metric(self) -> Tuple[Dict, ModelTargetMetric]:

        target_metric = ModelTargetMetric(metric_name=SpanF1Metric.F1_OVERALL,
                                          metric_value=self.span_f1_metric.metric[SpanF1Metric.F1_OVERALL])
        return self.span_f1_metric.metric, target_metric

    def reset(self) -> "NerModelMetricAdapter":
        self.span_f1_metric.reset()

        return self


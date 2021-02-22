#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
acsa model metric

Authors: PanXu
Date:    2020/07/18 18:12:00
"""
from typing import Tuple, Dict

from torch import Tensor

from easytext.metrics import ModelMetricAdapter, ModelTargetMetric
from easytext.metrics import AccMetric
from easytext.label_decoder import ModelLabelDecoder
from easytext.component.register import ComponentRegister
from acsa.models import ACSAModelOutputs


@ComponentRegister.register(name_space="acsa")
class ACSAModelMetric(ModelMetricAdapter):
    """
    ACSA model Metric
    """

    def __init__(self, label_decoder: ModelLabelDecoder):
        self._metric = AccMetric()
        self._label_decoder = label_decoder

    def __call__(self, model_outputs: ACSAModelOutputs, golden_labels: Tensor) -> Tuple[Dict, ModelTargetMetric]:

        logits = model_outputs.logits.detach().cpu()
        cpu_model_outputs = ACSAModelOutputs(logits=logits)

        golden_labels = golden_labels.detach().cpu()

        prediction_labels = self._label_decoder.decode_label_index(model_outputs=cpu_model_outputs)
        metric_dict = self._metric(prediction_labels=prediction_labels,
                                   gold_labels=golden_labels,
                                   mask=None)

        target_metric = ModelTargetMetric(metric_name=AccMetric.ACC,
                                          metric_value=metric_dict[AccMetric.ACC])

        return metric_dict, target_metric

    @property
    def metric(self) -> Tuple[Dict, ModelTargetMetric]:

        metric_dict = self._metric.metric

        target_metric = ModelTargetMetric(metric_name=AccMetric.ACC,
                                          metric_value=metric_dict[AccMetric.ACC])

        return metric_dict, target_metric

    def reset(self) -> "ACSAModelMetric":
        self._metric.reset()

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
from typing import Tuple, Dict, Union, List

import torch
from torch import Tensor
from torch.distributed import ReduceOp

from easytext.metrics import ModelMetricAdapter, ModelTargetMetric
from easytext.metrics import SpanF1Metric
from easytext.component.register import ComponentRegister

from mrc.label_decoder import MRCModelLabelDecoder
from mrc.models import MRCNerOutput
from mrc.metric import MRCF1Metric


@ComponentRegister.register(name_space="mrc")
class MrcModelMetricAdapter(ModelMetricAdapter):
    """
    Ner Model Metric Adapter
    计算 Ner Model Metric
    """

    def __init__(self):
        self.model_label_decoder = MRCModelLabelDecoder()
        self.mrc_f1_metric = MRCF1Metric()

    def __call__(self, model_outputs: MRCNerOutput, golden_label_dict: Dict[str, Tensor]) -> Tuple[Dict, ModelTargetMetric]:
        """
        计算 metric
        :param model_outputs:
        :param golden_label_dict: start_position_labels, end_position_labels, batch_match_positions
        :return:
        """
        model_outputs: MRCNerOutput = model_outputs

        match_prediction_labels = self.model_label_decoder.decode_label_index(model_outputs=model_outputs)

        match_golend_labels = golden_label_dict["match_position_labels"]

        # 计算 overall f1
        mask = model_outputs.mask.detach()

        metric_dict = self.mrc_f1_metric(prediction_match_labels=match_prediction_labels,
                                         gold_match_labels=match_golend_labels,
                                         mask=mask)

        target_metric = ModelTargetMetric(metric_name=MRCF1Metric.F1_OVERALL,
                                          metric_value=metric_dict[MRCF1Metric.F1_OVERALL])

        return metric_dict, target_metric

    @property
    def metric(self) -> Tuple[Dict, ModelTargetMetric]:
        f1_metric = self.mrc_f1_metric.metric
        target_metric = ModelTargetMetric(metric_name=MRCF1Metric.F1_OVERALL,
                                          metric_value=f1_metric[MRCF1Metric.F1_OVERALL])
        return f1_metric, target_metric

    def reset(self) -> "MrcModelMetricAdapter":
        self.mrc_f1_metric.reset()

        return self
    
    def to_synchronized_data(self) -> Tuple[Union[Dict[Union[str, int], Tensor], List[Tensor], Tensor], ReduceOp]:
        return self.mrc_f1_metric.to_synchronized_data()

    def from_synchronized_data(self, sync_data: Union[Dict[Union[str, int], Tensor], List[Tensor], Tensor],
                               reduce_op: ReduceOp) -> None:
        self.mrc_f1_metric.from_synchronized_data(sync_data=sync_data, reduce_op=reduce_op)


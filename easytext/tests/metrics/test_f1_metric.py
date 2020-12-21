#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 f1 metric

Authors: PanXu
Date:    2020/12/21 07:24:00
"""
from typing import Dict, Tuple, Union, List

import torch
from torch import Tensor
from torch.distributed import ReduceOp

from easytext.metrics import F1Metric

from easytext.tests import ASSERT


class _DemoF1Metric(F1Metric):

    def __init__(self):
        labels = ["a", "b", "c"]
        super().__init__(labels=labels)

        for i, label in enumerate(labels):

            self._true_positives[label] = i + 1
            self._false_negatives[label] = i + 2
            self._false_positives[label] = i + 3

    @property
    def true_positives(self):
        return self._true_positives

    @property
    def false_negatives(self):
        return self._false_negatives

    @property
    def false_positives(self):
        return self._false_positives

    def __call__(self, prediction_labels: torch.Tensor, gold_labels: torch.Tensor, mask: torch.LongTensor) -> Dict:
        pass


def test_synchronized_data():
    """
    测试 to_synchronized_data 和 from_synchronized_data
    :return:
    """

    demo_metric = _DemoF1Metric()

    sync_data, op = demo_metric.to_synchronized_data()

    true_positives = sync_data["true_positives"]
    false_positives = sync_data["false_positives"]
    false_negatives = sync_data["false_negatives"]

    expect_values = [v for _, v in demo_metric.true_positives.items()]
    ASSERT.assertListEqual(expect_values, true_positives.tolist())

    expect_values = [v for _, v in demo_metric.false_positives.items()]
    ASSERT.assertListEqual(expect_values, false_positives.tolist())

    expect_values = [v for _, v in demo_metric._false_negatives.items()]
    ASSERT.assertListEqual(expect_values, false_negatives.tolist())

    expect_true_positives = dict(demo_metric.true_positives)
    expect_false_positives = dict(demo_metric.false_positives)
    expect_false_negatives = dict(demo_metric.false_negatives)

    demo_metric.from_synchronized_data(sync_data=sync_data, reduce_op=op)

    ASSERT.assertDictEqual(expect_true_positives, demo_metric.true_positives)
    ASSERT.assertDictEqual(expect_false_positives, demo_metric.false_positives)
    ASSERT.assertDictEqual(expect_false_negatives, demo_metric.false_negatives)




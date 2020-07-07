#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 label 的 f1 metric

Authors: panxu(panxu@baidu.com)
Date:    2020/06/16 10:29:00
"""
import logging
import torch

from easytext.utils import log_util
from easytext.utils.json_util import json2str
from easytext.metrics import LabelF1Metric

from easytext.tests import ASSERT


# 配置log
log_util.config()


def test_label_f1_metric():
    """
    测试 label f1 metric
    """

    predictions = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    gold_labels = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4, 4, 1])

    labels = [0, 1, 2, 3, 4]
    f1_metric = LabelF1Metric(labels=labels, label_vocabulary=None)

    metrics = f1_metric(prediction_labels=predictions, gold_labels=gold_labels, mask=None)

    logging.debug(json2str(metrics))

    ASSERT.assertEqual((len(labels) + 1) * 3, len(metrics))

    precision_0 = metrics[f"{LabelF1Metric.PRECISION}-0"]
    recall_0 = metrics[f"{LabelF1Metric.RECALL}-0"]
    f1_0 = metrics[f"{LabelF1Metric.F1}-0"]

    expect_precision_0 = 1. / 2.
    ASSERT.assertAlmostEqual(expect_precision_0, precision_0)
    expect_recall_0 = 1. / 1.
    ASSERT.assertAlmostEqual(expect_recall_0, recall_0)

    expect_f1_0 = 2. * expect_precision_0 * expect_recall_0 / (expect_precision_0 + expect_recall_0)
    ASSERT.assertAlmostEqual(expect_f1_0, f1_0)

    expect_precision_overall = 5. / 10.
    expect_recall_overall = 5. / 10
    precision_overall = metrics[LabelF1Metric.PRECISION_OVERALL]
    recall_overall = metrics[LabelF1Metric.RECALL_OVERALL]

    ASSERT.assertAlmostEqual(expect_precision_overall, precision_overall)
    ASSERT.assertAlmostEqual(expect_recall_overall, recall_overall)

    predictions = torch.tensor([0, 2])
    gold_labels = torch.tensor([0, 1])

    f1_metric(prediction_labels=predictions, gold_labels=gold_labels, mask=None)

    precision_0 = f1_metric.metric[f"{LabelF1Metric.PRECISION}-0"]
    recall_0 = f1_metric.metric[f"{LabelF1Metric.RECALL}-0"]
    f1_0 = f1_metric.metric[f"{LabelF1Metric.F1}-0"]

    expect_precision_0 = (1. + 1.) / (2. + 1.)
    ASSERT.assertAlmostEqual(expect_precision_0, precision_0)
    expect_recall_0 = (1. + 1.) / (1. + 1.)
    ASSERT.assertAlmostEqual(expect_recall_0, recall_0)
    expect_f1_0 = 2. * expect_precision_0 * expect_recall_0 / (expect_precision_0 + expect_recall_0)
    ASSERT.assertAlmostEqual(expect_f1_0, f1_0)


def test_label_f1_metric_with_mask():
    """
    测试 label f1 metric
    """

    predictions = torch.tensor([0, 1, 2, 3])
    gold_labels = torch.tensor([0, 0, 0, 2])
    mask = torch.tensor([1, 1, 1, 0], dtype=torch.long)

    labels = [0, 1, 2, 3]
    f1_metric = LabelF1Metric(labels=labels, label_vocabulary=None)

    metrics = f1_metric(prediction_labels=predictions, gold_labels=gold_labels, mask=mask)

    logging.debug(json2str(metrics))

    ASSERT.assertEqual((len(labels) + 1) * 3, len(metrics))

    precision_0 = metrics[f"{LabelF1Metric.PRECISION}-0"]
    recall_0 = metrics[f"{LabelF1Metric.RECALL}-0"]
    f1_0 = metrics[f"{LabelF1Metric.F1}-0"]

    expect_precision_0 = 1. / 1.
    ASSERT.assertAlmostEqual(expect_precision_0, precision_0)
    expect_recall_0 = 1. / 3.
    ASSERT.assertAlmostEqual(expect_recall_0, recall_0)

    expect_f1_0 = 2. * expect_precision_0 * expect_recall_0 / (expect_precision_0 + expect_recall_0)
    ASSERT.assertAlmostEqual(expect_f1_0, f1_0)

    expect_precision_overall = 1. / 3.
    expect_recall_overall = 1. / 3.
    precision_overall = metrics[LabelF1Metric.PRECISION_OVERALL]
    recall_overall = metrics[LabelF1Metric.RECALL_OVERALL]

    ASSERT.assertAlmostEqual(expect_precision_overall, precision_overall)
    ASSERT.assertAlmostEqual(expect_recall_overall, recall_overall)


#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 metric

Authors: panxu(panxu@baidu.com)
Date:    2020/06/17 09:35:00
"""
import logging
import pytest

import torch

from easytext.tests import ASSERT
from easytext.data import Vocabulary
from easytext.utils.json_util import json2str
from easytext.utils import log_util
from easytext.metrics import F1Metric

from event.event_detection_without_tirgger.models import EventModelOutputs
from event.event_detection_without_tirgger.metrics import EventF1MetricAdapter

log_util.config()


@pytest.fixture(scope="class")
def event_type_vocabulary():
    event_types = [["A", "B", "C"], ["A", "B"], ["A"]]

    vocabulary = Vocabulary(tokens=event_types, padding="", unk="Negative", special_first=True)

    ASSERT.assertEqual(4, vocabulary.size)
    ASSERT.assertEqual(0, vocabulary.index(vocabulary.unk))
    ASSERT.assertEqual(1, vocabulary.index("A"))
    ASSERT.assertEqual(2, vocabulary.index("B"))
    ASSERT.assertEqual(3, vocabulary.index("C"))

    return vocabulary


def test_event_f1_metric(event_type_vocabulary):
    """
    测试 event f1 metric
    """
    f1_metric = EventF1MetricAdapter(event_type_vocabulary=event_type_vocabulary)

    # label: [1, 0, 1, 1, 0, 0, 1, 1, 0]
    logits = torch.tensor([0.6, 0.2, 0.7, 0.8, 0.1, 0.2, 0.8, 0.9, 0.3], dtype=torch.float)

    #                            [1, 0, 1, 1, 0, 0, 1, 1, 0]
    golden_labels = torch.tensor([1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long)

    event_type = torch.tensor([event_type_vocabulary.index("A"),
                               event_type_vocabulary.index("A"),
                               event_type_vocabulary.index("A"),
                               event_type_vocabulary.index("B"),
                               event_type_vocabulary.index("B"),
                               event_type_vocabulary.index("C"),
                               event_type_vocabulary.index("C"),
                               event_type_vocabulary.index(event_type_vocabulary.unk),
                               event_type_vocabulary.index(event_type_vocabulary.unk)],
                              dtype=torch.long)

    model_outputs = EventModelOutputs(logits=logits,
                                      event_type=event_type)

    metric, target_metric = f1_metric(model_outputs=model_outputs, golden_labels=golden_labels)

    expect_precision_A_1 = 1 / 2
    expect_recall_A_1 = 1 / 1
    expect_f1_A_1 = 2 * expect_precision_A_1 * expect_recall_A_1 / (expect_precision_A_1 + expect_recall_A_1)

    ASSERT.assertAlmostEqual(expect_precision_A_1, metric[f"{F1Metric.PRECISION}-A"])
    ASSERT.assertAlmostEqual(expect_recall_A_1, metric[f"{F1Metric.RECALL}-A"])
    ASSERT.assertAlmostEqual(expect_f1_A_1, metric[f"{F1Metric.F1}-A"])

    expect_precision_overall = 2 / 4
    expect_recall_overall = 2 / 3

    expect_f1_overall = 2 * expect_precision_overall * expect_recall_overall / (expect_precision_overall +
                                                                                expect_recall_overall)

    ASSERT.assertAlmostEqual(expect_precision_overall, metric[F1Metric.PRECISION_OVERALL])
    ASSERT.assertAlmostEqual(expect_recall_overall, metric[F1Metric.RECALL_OVERALL])
    ASSERT.assertAlmostEqual(expect_f1_overall, metric[F1Metric.F1_OVERALL])

    # 在增加一个数据，因为实际是多个batch的
    # label: [1, 1, 0]
    logits = torch.tensor([0.6, 0.8, 0.2], dtype=torch.float)
    golden_labels = torch.tensor([1, 1, 1], dtype=torch.long)

    event_type = torch.tensor([event_type_vocabulary.index("A"),
                               event_type_vocabulary.index("A"),
                               event_type_vocabulary.index("A")],
                              dtype=torch.long)

    model_outputs = EventModelOutputs(logits=logits,
                                      event_type=event_type)

    metric, target_metric = f1_metric(model_outputs=model_outputs, golden_labels=golden_labels)

    expect_final_precision_A_1 = (1 + 2) / (2 + 2)
    expect_final_recall_A_1 = (1 + 2) / (1 + 3)
    expect_final_f1_A_1 = 2 * expect_final_precision_A_1 * expect_final_recall_A_1 / (expect_final_precision_A_1
                                                                                      + expect_final_recall_A_1)

    ASSERT.assertAlmostEqual(expect_final_precision_A_1, f1_metric.metric[0][f"{F1Metric.PRECISION}-A"])
    ASSERT.assertAlmostEqual(expect_final_recall_A_1, f1_metric.metric[0][f"{F1Metric.RECALL}-A"])
    ASSERT.assertAlmostEqual(expect_final_f1_A_1, f1_metric.metric[0][f"{F1Metric.F1}-A"])

    expect_final_precision_overall = (2 + 2) / (4 + 2)
    expect_final_recall_overall = (2 + 2) / (3 + 3)
    expect_final_f1_overall = 2 * expect_final_precision_overall * expect_final_recall_overall / (
        expect_final_precision_overall + expect_final_recall_overall
    )
    ASSERT.assertAlmostEqual(expect_final_precision_overall, f1_metric.metric[0][F1Metric.PRECISION_OVERALL])
    ASSERT.assertAlmostEqual(expect_final_recall_overall, f1_metric.metric[0][F1Metric.RECALL_OVERALL])
    ASSERT.assertAlmostEqual(expect_final_f1_overall, f1_metric.metric[0][F1Metric.F1_OVERALL])

    ASSERT.assertEqual(F1Metric.F1_OVERALL, f1_metric.metric[1].name)
    ASSERT.assertAlmostEqual(f1_metric.metric[1].value, expect_final_f1_overall)




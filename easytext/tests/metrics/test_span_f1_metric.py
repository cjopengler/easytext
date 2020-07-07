#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
测试 span f1 measure

Authors: panxu(panxu@baidu.com)
Date:    2020/05/21 19:21:00
"""
import json
import torch

from easytext.data.vocabulary import LabelVocabulary
from easytext.tests import ASSERT
from easytext.metrics import ModelTargetMetric
from easytext.metrics.span_f1_metric import SpanF1Metric

VOCAB = LabelVocabulary([["B-T", "B-T", "B-T", "I-T", "I-T", "O"]],
                                 padding=LabelVocabulary.PADDING)

b_index = VOCAB.index("B-T")
ASSERT.assertEqual(0, b_index)
i_index = VOCAB.index("I-T")
ASSERT.assertEqual(1, i_index)
o_index = VOCAB.index("O")
ASSERT.assertEqual(2, o_index)


def test_span_f1_measure():

    # [[O, B, I], [B, B, I], [B, I, I], [B, I, O]]
    batch_sequence_logits = torch.tensor([[[0.2, 0.3, 0.4], [0.7, 0.2, 0.3], [0.2, 0.3, 0.1]],
                                          [[0.8, 0.3, 0.4], [0.7, 0.2, 0.3], [0.2, 0.3, 0.1]],
                                          [[0.8, 0.3, 0.4], [0.1, 0.7, 0.3], [0.2, 0.3, 0.1]],
                                          [[0.8, 0.3, 0.4], [0.1, 0.7, 0.3], [0.2, 0.3, 0.5]]],
                                         dtype=torch.float)

    batch_sequence_labels = [["O", "B-T", "I-T"],
                             ["B-T", "B-T", "I-T"],
                             ["B-T", "I-T", "I-T"],
                             ["B-T", "I-T", "O"]]
    sequence_label_indices = list()

    for sequence_label in batch_sequence_labels:
        sequence_label_indices.append([VOCAB.index(label) for label in sequence_label])

    sequence_label_indices = torch.tensor(sequence_label_indices, dtype=torch.long)

    gold = torch.tensor([
        [2, 0, 1],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 2]
    ])

    f1 = SpanF1Metric(label_vocabulary=VOCAB)

    f1(prediction_labels=sequence_label_indices, gold_labels=gold, mask=None)

    metrics = f1.metric

    print(f"metrics: {json.dumps(metrics)}")

    expect = {f"{SpanF1Metric.PRECISION}-T": 1., f"{SpanF1Metric.RECALL}-T": 1., f"{SpanF1Metric.F1}-T": 1.,
              SpanF1Metric.PRECISION_OVERALL: 1., SpanF1Metric.RECALL_OVERALL: 1., SpanF1Metric.F1_OVERALL: 1.}

    for key, _ in expect.items():
        ASSERT.assertAlmostEqual(expect[key], metrics[key])


def test_span_f1_measure_with_mask():

    # [[O, B, I], [B, B, I], [B, I, I], [B, I, O]]
    batch_sequence_logits = torch.tensor([[[0.2, 0.3, 0.4], [0.7, 0.2, 0.3], [0.2, 0.3, 0.1]],
                                          [[0.8, 0.3, 0.4], [0.7, 0.2, 0.3], [0.2, 0.3, 0.1]],
                                          [[0.8, 0.3, 0.4], [0.1, 0.7, 0.3], [0.2, 0.3, 0.1]],
                                          [[0.8, 0.3, 0.4], [0.1, 0.7, 0.3], [0.2, 0.3, 0.5]]],
                                         dtype=torch.float)

    batch_sequence_labels = [["O", "B-T", "I-T"],
                             ["B-T", "B-T", "I-T"],
                             ["B-T", "I-T", "I-T"],
                             ["B-T", "I-T", "O"]]
    sequence_label_indices = list()

    for sequence_label in batch_sequence_labels:
        sequence_label_indices.append([VOCAB.index(label) for label in sequence_label])

    sequence_label_indices = torch.tensor(sequence_label_indices, dtype=torch.long)

    gold = torch.tensor([
        [2, 0, 1],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 2]
    ])

    f1 = SpanF1Metric(label_vocabulary=VOCAB)

    mask = torch.tensor([
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1]
    ], dtype=torch.long)

    f1(prediction_labels=sequence_label_indices, gold_labels=gold, mask=mask)

    metrics = f1.metric

    print(f"metrics: {json.dumps(metrics)}")

    expect = {f"{SpanF1Metric.PRECISION}-T": 1., f"{SpanF1Metric.RECALL}-T": 1., f"{SpanF1Metric.F1}-T": 1.,
              f"{SpanF1Metric.PRECISION_OVERALL}": 1., f"{SpanF1Metric.RECALL_OVERALL}": 1., f"{SpanF1Metric.F1_OVERALL}": 1.}

    for key, _ in expect.items():
        ASSERT.assertAlmostEqual(expect[key], metrics[key])


def test_span_f1_measure_part_match():

    # [[O, B, I], [B, B, I], [B, I, I], [B, I, O]]
    batch_sequence_logits = torch.tensor([[[0.2, 0.3, 0.4], [0.7, 0.2, 0.3], [0.2, 0.3, 0.1]],
                                          [[0.8, 0.3, 0.4], [0.7, 0.2, 0.3], [0.2, 0.3, 0.1]],
                                          [[0.8, 0.3, 0.4], [0.1, 0.7, 0.3], [0.2, 0.3, 0.1]],
                                          [[0.8, 0.3, 0.4], [0.1, 0.7, 0.3], [0.2, 0.3, 0.5]]],
                                         dtype=torch.float)

    batch_sequence_labels = [["O", "B-T", "I-T"],
                             ["B-T", "B-T", "I-T"],
                             ["B-T", "I-T", "I-T"],
                             ["B-T", "I-T", "O"]]
    sequence_label_indices = list()

    for sequence_label in batch_sequence_labels:
        sequence_label_indices.append([VOCAB.index(label) for label in sequence_label])

    sequence_label_indices = torch.tensor(sequence_label_indices, dtype=torch.long)

    gold = torch.tensor([
        [2, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 2]
    ])

    f1 = SpanF1Metric(label_vocabulary=VOCAB)

    f1(prediction_labels=sequence_label_indices, gold_labels=gold, mask=None)

    metrics = f1.metric

    print(f"metrics: {json.dumps(metrics)}")

    expect = {f"{SpanF1Metric.PRECISION}-T": 0.8, f"{SpanF1Metric.RECALL}-T": 2/3,
              f"{SpanF1Metric.PRECISION_OVERALL}": 0.8, f"{SpanF1Metric.RECALL_OVERALL}": 2/3}

    for key, _ in expect.items():
        ASSERT.assertAlmostEqual(expect[key], metrics[key])


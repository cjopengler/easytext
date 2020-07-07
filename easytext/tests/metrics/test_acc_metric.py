#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
测试 acc metric

Authors: panxu(panxu@baidu.com)
Date:    2020/05/30 14:56:00
"""
import torch
from easytext.metrics import AccMetric

from easytext.tests import ASSERT


def test_acc_metric():

    # 对应的label是 [1, 1, 0, 1]
    logits = torch.tensor([[1., 2.], [3., 4.], [5, 4.], [3., 7.]], dtype=torch.float)
    prediction_labels = torch.argmax(logits, dim=-1)

    golden_labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    acc_metric = AccMetric()

    expect = 1/4

    acc = acc_metric(prediction_labels=prediction_labels,
                     gold_labels=golden_labels,
                     mask=None)

    ASSERT.assertAlmostEqual(expect, acc[acc_metric.ACC])
    ASSERT.assertAlmostEqual(expect, acc_metric.metric[acc_metric.ACC])

    # 对应的label是 [0, 1, 0, 1]
    logits = torch.tensor([[3., 2.], [4., 6.], [5, 4.], [3., 7.]], dtype=torch.float)
    prediction_labels = torch.argmax(logits, dim=-1)

    golden_labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    acc = acc_metric(prediction_labels=prediction_labels, gold_labels=golden_labels, mask=None)

    expect = 2 / 4
    ASSERT.assertAlmostEqual(expect, acc[acc_metric.ACC])

    # 下面的会计算将两次结果综合起来
    expect = (1+2)/(4+4)
    ASSERT.assertAlmostEqual(expect, acc_metric.metric[acc_metric.ACC])







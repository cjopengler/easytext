#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
测试 mrc metric

Authors: PanXu
Date:    2021/11/09 08:26:00
"""
import logging
import torch

from easytext.utils.json_util import json2str

from mrc.models import MRCNerOutput
from mrc.metric import MrcModelMetricAdapter
from mrc.metric import MRCF1Metric

from mrc.tests import ASSERT


def test_mrc_metric():
    start_logits = torch.tensor([[1, 1,  -1,  1, -1,  1],
                                 [1, -1, -1, -1, 1, -1]])

    end_logits = torch.tensor([[1, 1, 1, 1, -1, 1],
                               [1, 1, 1, -1, 1, 1]])

    # (1, 1), (1, 2), (1, 3), (1, 5)
    # (3, 3), (3, 5)
    # (5, 5)
    ################################
    # (4, 4), (4, 5)
    match_logits = torch.tensor([
        [
            [1, 1, 1, 1, 1, -1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, -1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]
        ],
        [
            [1, -1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, -1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]
        ]
    ])

    mask = torch.tensor([
        [False, True, True, True, True, True],
        [False, True, True, True, True, True]
    ])

    model_outputs = MRCNerOutput(start_logits=start_logits,
                                 end_logits=end_logits,
                                 match_logits=match_logits,
                                 mask=mask)

    # (1, 1), (1, 2), (1, 3), (1, 5)
    # (2, 2), (2, 5)
    # (3, 3), (3, 4), (3, 5)
    # (4, 4), (4, 5)
    # (5, 5)
    ###################
    # (1, 1), (1, 3), (1, 4)
    # (2, 2), (2, 3,) (2, 4) (2, 5)
    # (3, 3), (3, 5)
    # (4, 4), (4, 5)
    # (5, 5)
    golden_match_logits = torch.tensor([
        [
            [1, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]
        ],
        [
            [1, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1]
        ]
    ])


    golden_label_dict = {"match_position_labels": golden_match_logits}
    mrc_metric = MrcModelMetricAdapter()

    metric_dict, target_metric = mrc_metric(model_outputs=model_outputs, golden_label_dict=golden_label_dict)

    logging.info(f"metric dict: {json2str(metric_dict)}\ntarget metric: {json2str(target_metric)}")

    expect_precision = 9/9
    expect_recall = 9/24
    ASSERT.assertAlmostEqual(expect_precision, metric_dict[MRCF1Metric.PRECISION_OVERALL])
    ASSERT.assertAlmostEqual(expect_recall, metric_dict[MRCF1Metric.RECALL_OVERALL])

    ASSERT.assertEqual(MRCF1Metric.F1_OVERALL, target_metric.name)
    ASSERT.assertAlmostEqual(metric_dict[MRCF1Metric.F1_OVERALL], target_metric.value)


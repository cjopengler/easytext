#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""

Authors: PanXu
Date:    2020/07/18 16:08:00
"""
import numpy as np
import torch

from easytext.utils.nn.nn_util import masked_softmax


from easytext.tests import ASSERT


def test_masked_softmax():
    """
    测试 masked softmax
    :return:
    """

    vector = torch.FloatTensor(
        [
            [1., 2., 3.],
            [4., 5., 6.]
        ]
    )

    mask = torch.ByteTensor(
        [
            [1, 1, 0],
            [1, 1, 1]
        ]
    )

    result = masked_softmax(vector=vector, mask=mask)

    expect1 = np.exp(np.array([1., 2.]))

    expect1 = expect1 / np.sum(expect1)
    expect1 = np.concatenate([expect1, np.array([0.])], axis=-1).tolist()

    result1 = result[0].tolist()

    ASSERT.assertEqual(len(expect1), len(result1))

    for expect_data, result_data in zip(expect1, result1):
        ASSERT.assertAlmostEqual(expect_data, result_data)

    expect2 = np.exp(np.array([4., 5., 6.]))
    expect2 = expect2 / np.sum(expect2)
    expect2 = expect2.tolist()

    result2 = result[1].tolist()

    ASSERT.assertEqual(len(expect2), len(result2))

    for expect_data, result_data in zip(expect2, result2):
        ASSERT.assertAlmostEqual(expect_data, result_data)

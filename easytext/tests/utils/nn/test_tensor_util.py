#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
测试 tensor util

Authors: PanXu
Date:    2021/02/15 12:40:00
"""

import torch

from easytext.utils.nn import tensor_util
from easytext.tests import ASSERT


def test_is_tensor_equal():
    """
    测试两个 tensor 是否相等
    :return:
    """

    x = torch.tensor([1, 2, 3])
    y = torch.tensor([1, 2, 3])

    equal = tensor_util.is_tensor_equal(tensor1=x, tensor2=y, epsilon=0)

    ASSERT.assertTrue(equal)

    x = torch.tensor([1, 2, 3])
    y = torch.tensor([2, 2, 3])

    equal = tensor_util.is_tensor_equal(tensor1=x, tensor2=y, epsilon=0)

    ASSERT.assertFalse(equal)

    x = torch.tensor([1.0001, 2.0001, 3.0001])
    y = torch.tensor([1., 2., 3.])

    equal = tensor_util.is_tensor_equal(tensor1=x, tensor2=y, epsilon=1e-3)

    ASSERT.assertTrue(equal)

    equal = tensor_util.is_tensor_equal(tensor1=x, tensor2=y, epsilon=1e-4)

    ASSERT.assertFalse(equal)



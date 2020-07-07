#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 max label index decoder

Authors: PanXu
Date:    2020/07/06 10:23:00
"""
import torch

from easytext.tests import ASSERT
from easytext.label_decoder import MaxLabelIndexDecoder


def test_max_label_index_decoder():
    """
    测试 max label index
    :return:
    """
    decoder = MaxLabelIndexDecoder()

    logits = torch.tensor([[0.1, 0.9], [0.3, 0.7], [0.8, 0.2]])

    label_indices = decoder(logits=logits)

    expect = [1, 1, 0]

    ASSERT.assertListEqual(expect, label_indices.tolist())

#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 conll2003 dataset

Authors: panxu(panxu@baidu.com)
Date:    2020/06/26 22:07:00
"""
import os
import pytest

from ner.tests import ASSERT


def test_msra_dataset(msra_dataset):
    """
    测试 msra 数据集
    :param conll2003_dataset: 数据集
    :return: None
    """

    ASSERT.assertEqual(5, len(msra_dataset))

    instance2 = msra_dataset[2]

    ASSERT.assertEqual(22, len(instance2["tokens"]))

    expect_labels = ["O"] * 22
    expect_labels[6] = "B-LOC"
    expect_labels[7] = "I-LOC"

    ASSERT.assertListEqual(expect_labels, instance2["sequence_label"])

    instance4 = msra_dataset[4]
    expect_labels = ["O", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-PER", "I-PER", "B-LOC"]
    ASSERT.assertListEqual(expect_labels, instance4["sequence_label"])

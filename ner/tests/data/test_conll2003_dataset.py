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


def test_conll2003_dataset(conll2003_dataset):
    """
    测试 conll2003 数据集
    :param conll2003_dataset: 数据集
    :return: None
    """

    ASSERT.assertEqual(2, len(conll2003_dataset))

    instance0 = conll2003_dataset[0]

    ASSERT.assertEqual(11, len(instance0["tokens"]))

    instance1 = conll2003_dataset[1]

    expect_labels = ["B-LOC", "O"]

    ASSERT.assertListEqual(expect_labels, instance1["sequence_label"])


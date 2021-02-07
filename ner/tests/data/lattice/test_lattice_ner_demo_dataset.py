#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
测试 lattice ner demo dataset

Authors: PanXu
Date:    2021/02/07 09:06:00
"""
from typing import List

from easytext.data.tokenizer import Token

from ner.tests import ASSERT


def test_lattice_ner_demo_dataset(lattice_ner_demo_dataset):
    """
    测试 lattice ner demo dataset
    :param lattice_ner_demo_dataset: lattice ner demo dataset
    :return:
    """

    instance = lattice_ner_demo_dataset[0]

    tokens: List[Token] = instance["tokens"]

    sentence = "".join([t.text for t in tokens])
    expect_sentence = "陈元呼吁加强国际合作推动世界经济发展"
    expect_sequence_label = ["B-PER", "I-PER"] + ["O"] * 16

    ASSERT.assertEqual(expect_sentence, sentence)
    ASSERT.assertListEqual(expect_sequence_label, instance["sequence_label"])


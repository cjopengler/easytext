#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
测试 bio schema 文件

Authors: PanXu
Date:    2021/02/03 10:02:00
"""

from easytext.utils import bio_schema

from easytext.tests import ASSERT


def test_ibo1_to_bio():
    """
    测试 ibo1 转换到 bio
    :return:
    """
    ibo1 = ["I-L1", "I-L1", "O",
            "I-L1", "I-L2", "O",
            "I-L1", "I-L1", "I-L1", "B-L1", "I-L1", "O",
            "B-L1", "I-L1", "O"]

    expect_bio = ["B-L1", "I-L1", "O",
                  "B-L1", "B-L2", "O",
                  "B-L1", "I-L1", "I-L1", "B-L1", "I-L1", "O",
                  "B-L1", "I-L1", "O"]

    bio_sequence = bio_schema.ibo1_to_bio(ibo1)

    ASSERT.assertListEqual(expect_bio, bio_sequence)


def test_bmes_to_bio():
    """
    测试 BMES schema 转换成 bio
    :return:
    """
    bmes = ["B-T", "M-T", "E-T", "O", "S-T", "B-T", "E-T"]
    expect_bio = ["B-T", "I-T", "I-T", "O", "B-T", "B-T", "I-T"]

    bio_sequence_label = bio_schema.bmes_to_bio(bmes)

    ASSERT.assertListEqual(expect_bio, bio_sequence_label)

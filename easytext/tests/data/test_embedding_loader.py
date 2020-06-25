#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 embedding loader

Authors: panxu(panxu@baidu.com)
Date:    2020/06/25 09:56:00
"""
import os
from easytext.tests import ROOT_PATH
from easytext.tests import ASSERT

from easytext.data import GloveLoader


def test_glove_loader():
    pretrained_file_path = "data/easytext/tests/pretrained/word_embedding_sample.3d.txt"
    pretrained_file_path = os.path.join(ROOT_PATH, pretrained_file_path)

    glove_loader = GloveLoader(embedding_dim=3,
                               pretrained_file_path=pretrained_file_path)

    embedding_dict = glove_loader.load()
    expect_embedding_dict = {"a": [1.0, 2.0, 3.0],
                             "b": [4.0, 5.0, 6.0],
                             "美丽": [7.0, 8.0, 9.0]}

    ASSERT.assertDictEqual(expect_embedding_dict, embedding_dict)
    ASSERT.assertEqual(glove_loader.embedding_dim, 3)

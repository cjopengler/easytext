#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 pretrained vocabulary

Authors: panxu(panxu@baidu.com)
Date:    2020/06/25 12:00:00
"""
import os
import pytest

from easytext.tests import ROOT_PATH
from easytext.tests import ASSERT

from easytext.data import PretrainedVocabulary
from easytext.data import GloveLoader
from easytext.data import Vocabulary


@pytest.fixture(scope="class")
def embedding_loader():
    pretrained_file_path = "data/easytext/tests/pretrained/word_embedding_sample.3d.txt"
    pretrained_file_path = os.path.join(ROOT_PATH, pretrained_file_path)

    glove_loader = GloveLoader(embedding_dim=3,
                               pretrained_file_path=pretrained_file_path)
    return glove_loader


def test_pretrained_vocabulary(embedding_loader):
    tokens = [["我"], ["美丽"]]

    vocab = Vocabulary(tokens=tokens,
                       padding=Vocabulary.PADDING,
                       unk=Vocabulary.UNK,
                       special_first=True)

    pretrained_vocab = PretrainedVocabulary(vocabulary=vocab,
                                            pretrained_word_embedding_loader=embedding_loader)

    ASSERT.assertEqual(4, pretrained_vocab.size)
    ASSERT.assertEqual(4, len(pretrained_vocab))
    ASSERT.assertEqual(2, pretrained_vocab.index("我"))
    ASSERT.assertEqual(3, pretrained_vocab.index("美丽"))

    ASSERT.assertEqual((pretrained_vocab.size, embedding_loader.embedding_dim),
                        pretrained_vocab.embedding_matrix.size())

    expect_embedding_dict = {"a": [1.0, 2.0, 3.0],
                             "b": [4.0, 5.0, 6.0],
                             "美丽": [7.0, 8.0, 9.0]}

    ASSERT.assertListEqual(expect_embedding_dict["美丽"],
                       pretrained_vocab.embedding_matrix[
                           pretrained_vocab.index("美丽")
                       ].tolist())

    zero_vec = [0.] * embedding_loader.embedding_dim

    for index in [pretrained_vocab.index("我"),
                  pretrained_vocab.padding_index,
                  pretrained_vocab.index(pretrained_vocab.unk)]:
        ASSERT.assertListEqual(zero_vec, pretrained_vocab.embedding_matrix[index].tolist())







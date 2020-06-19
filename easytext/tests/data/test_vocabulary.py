#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
测试 Vocabulary

Authors: panxu(panxu@baidu.com)
Date:    2020/05/20 14:43:00
"""
import os
from unittest import TestCase

from easytext.data import Vocabulary, LabelVocabulary
from easytext.tests import ROOT_PATH, ASSERT


def test_vocabulary():
    """

    :return:
    """

    batch_tokens = [["我", "和", "你"], ["在", "我"]]
    vocabulary = Vocabulary(batch_tokens,
                            padding="",
                            unk="",
                            special_first=True,
                            min_frequency=1,
                            max_size=None)

    ASSERT.assertEqual(vocabulary.size, 4)

    ASSERT.assertTrue(not vocabulary.padding)
    ASSERT.assertTrue(not vocabulary.unk)

    ASSERT.assertEqual(vocabulary.index("我"), 0)
    ASSERT.assertEqual(vocabulary.index("和"), 1)


def test_vocabulary_speical_first():
    """
    测试 vocabulary speical first
    :return:
    """
    batch_tokens = [["我", "和", "你"], ["在", "我"]]
    vocabulary = Vocabulary(batch_tokens,
                            padding=Vocabulary.PADDING,
                            unk=Vocabulary.UNK,
                            special_first=True,
                            min_frequency=1,
                            max_size=None)

    ASSERT.assertEqual(vocabulary.size, 6)

    ASSERT.assertEqual(vocabulary.padding, vocabulary.PADDING)
    ASSERT.assertEqual(vocabulary.unk, vocabulary.UNK)
    ASSERT.assertEqual(vocabulary.index(vocabulary.padding), 0)
    ASSERT.assertEqual(vocabulary.index(vocabulary.unk), 1)


def test_other_speical_tokens():
    """
    测试 vocabulary speical first
    :return:
    """
    batch_tokens = [["我", "和", "你"], ["在", "我"]]
    vocabulary = Vocabulary(batch_tokens,
                            padding=Vocabulary.PADDING,
                            unk=Vocabulary.UNK,
                            special_first=True,
                            other_special_tokens=["<Start>", "<End>"],
                            min_frequency=1,
                            max_size=None)

    ASSERT.assertEqual(vocabulary.size, 8)

    ASSERT.assertEqual(vocabulary.padding, vocabulary.PADDING)
    ASSERT.assertEqual(vocabulary.unk, vocabulary.UNK)
    ASSERT.assertEqual(vocabulary.index(vocabulary.padding), 0)
    ASSERT.assertEqual(vocabulary.index(vocabulary.unk), 1)
    ASSERT.assertEqual(vocabulary.index("<Start>"), 2)
    ASSERT.assertEqual(vocabulary.index("<End>"), 3)


def test_speical_last():
    batch_tokens = [["我", "和", "你"], ["在", "我"]]
    vocabulary = Vocabulary(batch_tokens,
                            padding=Vocabulary.PADDING,
                            unk=Vocabulary.UNK,
                            special_first=False,
                            other_special_tokens=["<Start>", "<End>"],
                            min_frequency=1,
                            max_size=None)

    ASSERT.assertEqual(vocabulary.size, 8)

    ASSERT.assertEqual(vocabulary.padding, vocabulary.PADDING)
    ASSERT.assertEqual(vocabulary.unk, vocabulary.UNK)
    ASSERT.assertEqual(vocabulary.index(vocabulary.padding), 3 + 1)
    ASSERT.assertEqual(vocabulary.index(vocabulary.unk), 3 + 2)
    ASSERT.assertEqual(vocabulary.index("<Start>"), 3 + 3)
    ASSERT.assertEqual(vocabulary.index("<End>"), 3 + 4)


def test_label_vocabulary():
    """
    测试 label vocabulary
    :return:
    """
    vocabulary = LabelVocabulary([["A", "B", "C"], ["D", "E"]], padding="")
    ASSERT.assertEqual(vocabulary.size, 5)

    vocabulary = LabelVocabulary([["A", "B", "C"], ["D", "E"]], padding=LabelVocabulary.PADDING)
    ASSERT.assertEqual(vocabulary.size, 6)
    ASSERT.assertEqual(vocabulary.label_size, 5)

    ASSERT.assertEqual(vocabulary.index(vocabulary.padding), 5)

    for index, w in enumerate(["A", "B", "C", "D", "E"]):
        ASSERT.assertEqual(vocabulary.index(w), index)


def test_save_and_load():
    """
    测试存储和载入 vocabulary
    :return:
    """
    batch_tokens = [["我", "和", "你"], ["在", "我"], ["newline\nnewline"]]
    vocabulary = Vocabulary(batch_tokens,
                            padding=Vocabulary.PADDING,
                            unk=Vocabulary.UNK,
                            special_first=True,
                            other_special_tokens=["<Start>", "<End>"],
                            min_frequency=1,
                            max_size=None)

    ASSERT.assertEqual(vocabulary.size, 9)

    ASSERT.assertEqual(vocabulary.padding, vocabulary.PADDING)
    ASSERT.assertEqual(vocabulary.unk, vocabulary.UNK)
    ASSERT.assertEqual(vocabulary.index(vocabulary.padding), 0)
    ASSERT.assertEqual(vocabulary.index(vocabulary.unk), 1)
    ASSERT.assertEqual(vocabulary.index("<Start>"), 2)
    ASSERT.assertEqual(vocabulary.index("<End>"), 3)
    ASSERT.assertEqual(vocabulary.index("我"), 4)
    ASSERT.assertEqual(vocabulary.index("newline\nnewline"), 8)
    ASSERT.assertEqual(vocabulary.index("哈哈"), vocabulary.index(vocabulary.unk))

    vocab_dir = os.path.join(ROOT_PATH, "data/easytext/tests")

    if not os.path.isdir(vocab_dir):
        os.makedirs(vocab_dir, exist_ok=True)

    vocabulary.save_to_file(vocab_dir)

    loaded_vocab = Vocabulary.from_file(directory=vocab_dir)

    ASSERT.assertEqual(vocabulary.size, 9)

    ASSERT.assertEqual(loaded_vocab.padding, vocabulary.PADDING)
    ASSERT.assertEqual(loaded_vocab.unk, vocabulary.UNK)
    ASSERT.assertEqual(loaded_vocab.index(vocabulary.padding), 0)
    ASSERT.assertEqual(loaded_vocab.index(vocabulary.unk), 1)
    ASSERT.assertEqual(loaded_vocab.index("<Start>"), 2)
    ASSERT.assertEqual(loaded_vocab.index("<End>"), 3)
    ASSERT.assertEqual(vocabulary.index("我"), 4)
    ASSERT.assertEqual(vocabulary.index("newline\nnewline"), 8)
    ASSERT.assertEqual(vocabulary.index("哈哈"), vocabulary.index(vocabulary.unk))







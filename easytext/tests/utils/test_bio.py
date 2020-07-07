#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
测试 Bio

Authors: panxu(panxu@baidu.com)
Date:    2020/05/15 10:50:00
"""
import torch
from unittest import TestCase
from easytext.data import LabelVocabulary
from easytext.utils import bio as BIO

ASSERT = TestCase()


def test_fill():
    """
    测试 bio
    :return:
    """

    pairs = [(1, 2), (2, 4)]

    for begin, end in pairs:
        sl = ["O"] * 10
        tag = "Test"
        BIO.fill(sequence_label=sl, begin_index=begin, end_index=end, tag=tag)

        for i in range(begin, end):

            if i == begin:
                ASSERT.assertEqual(sl[i], f"B-{tag}")
            else:
                ASSERT.assertEqual(sl[i], f"I-{tag}")


def test_decode_one_sequence_logits_to_label():
    """
    测试 decode sequence label
    :return:
    """

    sequence_logits_list = list()
    expect_list = list()

    sequence_logits = torch.tensor([[0.2, 0.3, 0.4], [0.7, 0.2, 0.3], [0.2, 0.3, 0.1]],
                                   dtype=torch.float)  # O B I 正常
    expect = ["O", "B-T", "I-T"]
    sequence_logits_list.append(sequence_logits)
    expect_list.append(expect)

    sequence_logits = torch.tensor([[0.9, 0.3, 0.4], [0.2, 0.8, 0.3], [0.2, 0.3, 0.1]],
                                   dtype=torch.float)
    expect = ["B-T", "I-T", "I-T"]

    sequence_logits_list.append(sequence_logits)
    expect_list.append(expect)

    sequence_logits = torch.tensor([[0.9, 0.3, 0.4], [0.2, 0.8, 0.3], [0.2, 0.3, 0.9]],
                                   dtype=torch.float)
    expect = ["B-T", "I-T", "O"]
    sequence_logits_list.append(sequence_logits)
    expect_list.append(expect)

    vocabulary = LabelVocabulary([["B-T", "B-T", "B-T", "I-T", "I-T", "O"]],
                                 padding=LabelVocabulary.PADDING)

    b_index = vocabulary.index("B-T")
    ASSERT.assertEqual(0, b_index)
    i_index = vocabulary.index("I-T")
    ASSERT.assertEqual(1, i_index)
    o_index = vocabulary.index("O")
    ASSERT.assertEqual(2, o_index)

    for sequence_logits, expect in zip(sequence_logits_list, expect_list):
        sequence_label, sequence_label_indices = BIO.decode_one_sequence_logits_to_label(
            sequence_logits=sequence_logits,
            vocabulary=vocabulary)

        ASSERT.assertListEqual(sequence_label, expect)
        expect_indices = [vocabulary.index(label) for label in expect]
        ASSERT.assertListEqual(sequence_label_indices, expect_indices)


def test_decode_one_sequence_logits_to_label_abnormal():
    """
    测试异常case
    :return:
    """

    # [0.2, 0.5, 0.4] argmax 解码是 I 这是异常的 case, 整个序列是: I B O
    # 而 decode_sequence_lable_bio 会将 概率值 是 0.4 的 也就是 O 作为标签输出 来修订这个个错误
    sequence_logits = torch.tensor([[0.2, 0.5, 0.4], [0.7, 0.2, 0.3], [0.2, 0.3, 0.1]],
                                   dtype=torch.float)

    vocabulary = LabelVocabulary([["B-T", "B-T", "B-T", "I-T", "I-T", "O"]],
                                 padding=LabelVocabulary.PADDING)

    b_index = vocabulary.index("B-T")
    ASSERT.assertEqual(0, b_index)
    i_index = vocabulary.index("I-T")
    ASSERT.assertEqual(1, i_index)
    o_index = vocabulary.index("O")
    ASSERT.assertEqual(2, o_index)

    sequence_label, sequence_label_indices = BIO.decode_one_sequence_logits_to_label(sequence_logits=sequence_logits,
                                                             vocabulary=vocabulary)

    expect = ["O", "B-T", "I-T"]
    expect_indices = [vocabulary.index(label) for label in expect]
    ASSERT.assertListEqual(expect, sequence_label)
    ASSERT.assertListEqual(expect_indices, sequence_label_indices)

    # argmax 解码是 I I I 经过修订后是: O O B
    sequence_logits = torch.tensor([[0.2, 0.5, 0.4], [0.2, 0.9, 0.3], [0.2, 0.3, 0.1]],
                                   dtype=torch.float)

    sequence_label, sequence_label_indices = BIO.decode_one_sequence_logits_to_label(sequence_logits=sequence_logits,
                                                             vocabulary=vocabulary)
    expect = ["O", "O", "B-T"]
    expect_indices = [vocabulary.index(label) for label in expect]
    ASSERT.assertListEqual(expect, sequence_label)
    ASSERT.assertListEqual(expect_indices, sequence_label_indices)


def test_decode_one_sequence_label_to_span():
    """
    测试对 sequence label 解码成 span 字典
    :return:
    """

    sequence_label_list = list()
    expect_list = list()

    sequence_label = ["B-T", "I-T", "O-T"]
    expect = [{"label": "T", "begin": 0, "end": 2}]

    sequence_label_list.append(sequence_label)
    expect_list.append(expect)

    sequence_label = ["B-T", "I-T", "I-T"]
    expect = [{"label": "T", "begin": 0, "end": 3}]

    sequence_label_list.append(sequence_label)
    expect_list.append(expect)

    sequence_label = ["B-T", "I-T", "I-T", "B-T"]
    expect = [{"label": "T", "begin": 0, "end": 3},
              {"label": "T", "begin": 3, "end": 4}]

    sequence_label_list.append(sequence_label)
    expect_list.append(expect)

    for expect, sequence_label in zip(expect_list, sequence_label_list):
        span = BIO.decode_one_sequence_label_to_span(sequence_label)

        ASSERT.assertListEqual(expect, span)


def test_decode():
    """
    测试 模型输出的 batch logits 解码
    :return:
    """

    # [[O, B, I], [B, B, I], [B, I, I], [B, I, O]]
    batch_sequence_logits = torch.tensor([[[0.2, 0.3, 0.4], [0.7, 0.2, 0.3], [0.2, 0.3, 0.1]],
                                          [[0.8, 0.3, 0.4], [0.7, 0.2, 0.3], [0.2, 0.3, 0.1]],
                                          [[0.8, 0.3, 0.4], [0.1, 0.7, 0.3], [0.2, 0.3, 0.1]],
                                          [[0.8, 0.3, 0.4], [0.1, 0.7, 0.3], [0.2, 0.3, 0.5]]],
                                         dtype=torch.float)

    expect = [[{"label": "T", "begin": 1, "end": 3}],
              [{"label": "T", "begin": 0, "end": 1}, {"label": "T", "begin": 1, "end": 3}],
              [{"label": "T", "begin": 0, "end": 3}],
              [{"label": "T", "begin": 0, "end": 2}]]

    vocabulary = LabelVocabulary([["B-T", "B-T", "B-T", "I-T", "I-T", "O"]],
                                 padding=LabelVocabulary.PADDING)

    b_index = vocabulary.index("B-T")
    ASSERT.assertEqual(0, b_index)
    i_index = vocabulary.index("I-T")
    ASSERT.assertEqual(1, i_index)
    o_index = vocabulary.index("O")
    ASSERT.assertEqual(2, o_index)

    spans = BIO.decode(batch_sequence_logits=batch_sequence_logits,
                       mask=None,
                       vocabulary=vocabulary)

    ASSERT.assertListEqual(expect, spans)


def test_decode_decode_label_index_to_span():
    """
    测试解码 golden label index
    :return:
    """

    vocabulary = LabelVocabulary([["B-T", "B-T", "B-T", "I-T", "I-T", "O"]],
                                 padding=LabelVocabulary.PADDING)

    b_index = vocabulary.index("B-T")
    ASSERT.assertEqual(0, b_index)
    i_index = vocabulary.index("I-T")
    ASSERT.assertEqual(1, i_index)
    o_index = vocabulary.index("O")
    ASSERT.assertEqual(2, o_index)

    golden_labels = torch.tensor([[0, 1, 2, 0],
                                  [2, 0, 1, 1]])

    expect = [[{"label": "T", "begin": 0, "end": 2}, {"label": "T", "begin": 3, "end": 4}],
              [{"label": "T", "begin": 1, "end": 4}]]

    spans = BIO.decode_label_index_to_span(batch_sequence_label_index=golden_labels,
                                           mask=None,
                                           vocabulary=vocabulary)

    ASSERT.assertListEqual(expect, spans)


def test_span_intersection():
    span_list1 = [{"label": "T", "begin": 0, "end": 2}, {"label": "T", "begin": 3, "end": 4}]
    span_list2 = [{"label": "T", "begin": 0, "end": 2}]

    intersetction = BIO.span_intersection(span_list1=span_list1,
                                          span_list2=span_list2)

    expect = [{"label": "T", "begin": 0, "end": 2}]
    ASSERT.assertListEqual(expect, intersetction)

    span_list1 = [{"label": "T", "begin": 0, "end": 2}, {"label": "T", "begin": 3, "end": 4}]
    span_list2 = [{"label": "T", "begin": 9, "end": 10}]

    intersetction = BIO.span_intersection(span_list1=span_list1,
                                          span_list2=span_list2)

    expect = []
    ASSERT.assertListEqual(expect, intersetction)

    span_list1 = [{"label": "T", "begin": 0, "end": 2}, {"label": "T", "begin": 3, "end": 4}]
    span_list2 = []

    intersetction = BIO.span_intersection(span_list1=span_list1,
                                          span_list2=span_list2)

    expect = []
    ASSERT.assertListEqual(expect, intersetction)


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

    bio_sequnce = BIO.ibo1_to_bio(ibo1)

    ASSERT.assertListEqual(expect_bio, bio_sequnce)


def test_allowed_transitions():
    """
    测试允许转移mask pair
    :return:
    """

    label_vocabulary = LabelVocabulary(labels=[["B-L1", "I-L1", "B-L2", "I-L2", "O"]],
                                       padding=LabelVocabulary.PADDING)

    allowed_pairs = BIO.allowed_transitions(label_vocabulary=label_vocabulary)

    for from_idx, to_idx in allowed_pairs:

        if from_idx == label_vocabulary.label_size:
            from_label = "START"
        else:
            from_label = label_vocabulary.token(from_idx)

        if to_idx == label_vocabulary.label_size + 1:
            to_label = "STOP"
        else:
            to_label = label_vocabulary.token(to_idx)
        print(f"(\"{from_label}\", \"{to_label}\"),")

    expect_trainsition_labels = [
        ("B-L1", "B-L1"), ("B-L1", "I-L1"), ("B-L1", "B-L2"), ("B-L1", "O"), ("B-L1", "STOP"),
        ("I-L1", "B-L1"), ("I-L1", "I-L1"), ("I-L1", "B-L2"), ("I-L1", "O"), ("I-L1", "STOP"),
        ("B-L2", "B-L1"), ("B-L2", "B-L2"), ("B-L2", "I-L2"), ("B-L2", "O"), ("B-L2", "STOP"),
        ("I-L2", "B-L1"), ("I-L2", "B-L2"), ("I-L2", "I-L2"), ("I-L2", "O"), ("I-L2", "STOP"),
        ("O", "B-L1"), ("O", "B-L2"), ("O", "O"), ("O", "STOP"),
        ("START", "B-L1"), ("START", "B-L2"), ("START", "O")]


    expect = list()

    for from_label, to_label in expect_trainsition_labels:
        if from_label == "START":
            from_idx = label_vocabulary.label_size
        else:
            from_idx = label_vocabulary.index(from_label)

        if to_label == "STOP":
            to_idx = label_vocabulary.label_size + 1
        else:
            to_idx = label_vocabulary.index(to_label)

        expect.append((from_idx, to_idx))

    ASSERT.assertSetEqual(set(expect), set(allowed_pairs))

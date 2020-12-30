#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 Bert Model Collate

Authors: PanXu
Date:    2020/09/10 15:23:00
"""

import logging
from torch.utils.data import DataLoader

from easytext.data import ModelInputs
from easytext.utils.json_util import json2str
from easytext.utils import log_util

from ner.tests import ASSERT
from ner.data import BertModelCollate

log_util.config()


def test_bert_model_collate_with_special_token(msra_dataset, msra_vocabulary, bert_tokenizer):
    """
    测试带有 CLS 和 SEP 的 bert model collate
    :param msra_dataset: msra 数据集
    :param msra_vocabulary: 在 conftest.py 中的 msra_vocabulary 返回结果
    :return: None
    """

    label_vocab = msra_vocabulary["label_vocabulary"]

    sequence_max_len = 13
    model_collate = BertModelCollate(tokenizer=bert_tokenizer,
                                     sequence_label_vocab=label_vocab,
                                     add_special_token=True,
                                     sequence_max_len=sequence_max_len)
    batch_size = 5
    data_loader = DataLoader(dataset=msra_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=model_collate)

    for model_inputs in data_loader:
        model_inputs: ModelInputs = model_inputs

        logging.info(f"model inputs: {json2str(model_inputs)}")

        ASSERT.assertEqual(batch_size, model_inputs.batch_size)

        ASSERT.assertEqual((batch_size, sequence_max_len), model_inputs.labels.size())

        input_ids = model_inputs.model_inputs["input_ids"]
        ASSERT.assertEqual((batch_size, sequence_max_len), input_ids.size())

        sequence_mask = model_inputs.model_inputs["sequence_mask"]
        ASSERT.assertEqual((batch_size, sequence_max_len), sequence_mask.size())

        ASSERT.assertEqual((batch_size, sequence_max_len), model_inputs.labels.size())

        sequence_mask0 = sequence_mask[0].tolist()
        expect_sequence_mask0 = [0] + [1] * (sequence_max_len - 2) + [0]
        ASSERT.assertEqual(expect_sequence_mask0, sequence_mask0)

        sequence_mask4 = sequence_mask[4].tolist()
        expect_sequence_mask0 = [0] + [1] * 8 + [0] * (sequence_max_len - 8 - 1)
        ASSERT.assertEqual(expect_sequence_mask0, sequence_mask4)

        sequence_label0 = model_inputs.labels[0].tolist()
        sequence_label0_str = model_inputs.model_inputs["metadata"][0]["labels"][0:sequence_max_len - 2]
        expect_sequence_label0 = [label_vocab.padding_index] \
                                 + [label_vocab.index(label) for label in sequence_label0_str] \
                                 + [label_vocab.padding_index]
        ASSERT.assertEqual(sequence_label0, expect_sequence_label0)

        sequence_label4 = model_inputs.labels[4].tolist()
        sequence_label4_str = model_inputs.model_inputs["metadata"][4]["labels"]

        expect_sequence_label4 = [label_vocab.padding_index] \
                                 + [label_vocab.index(label) for label in sequence_label4_str] \
                                 + [label_vocab.padding_index] * (sequence_max_len - 1 - len(sequence_label4_str))
        ASSERT.assertEqual(sequence_label4, expect_sequence_label4)


def test_bert_model_collate_without_special_token(msra_dataset, msra_vocabulary, bert_tokenizer):
    """
    测试没有 CLS 和 SEP 的 bert model collate
    :param msra_dataset: msra 数据集
    :param msra_vocabulary: 在 conftest.py 中的 msra_vocabulary 返回结果
    :return: None
    """

    label_vocab = msra_vocabulary["label_vocabulary"]

    sequence_max_len = 13
    model_collate = BertModelCollate(tokenizer=bert_tokenizer,
                                     sequence_label_vocab=label_vocab,
                                     add_special_token=False,
                                     sequence_max_len=sequence_max_len)
    batch_size = 5
    data_loader = DataLoader(dataset=msra_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=model_collate)

    for model_inputs in data_loader:
        model_inputs: ModelInputs = model_inputs

        logging.info(f"model inputs: {json2str(model_inputs)}")

        ASSERT.assertEqual(batch_size, model_inputs.batch_size)

        ASSERT.assertEqual((batch_size, sequence_max_len), model_inputs.labels.size())

        input_ids = model_inputs.model_inputs["input_ids"]
        ASSERT.assertEqual((batch_size, sequence_max_len), input_ids.size())

        sequence_mask = model_inputs.model_inputs["sequence_mask"]
        ASSERT.assertEqual((batch_size, sequence_max_len), sequence_mask.size())

        ASSERT.assertEqual((batch_size, sequence_max_len), model_inputs.labels.size())

        sequence_mask0 = sequence_mask[0].tolist()
        expect_sequence_mask0 = [1] * sequence_max_len
        ASSERT.assertEqual(expect_sequence_mask0, sequence_mask0)

        sequence_mask4 = sequence_mask[4].tolist()
        expect_sequence_mask0 = [1] * 8 + [0] * (sequence_max_len - 8)
        ASSERT.assertEqual(expect_sequence_mask0, sequence_mask4)

        sequence_label0 = model_inputs.labels[0].tolist()
        sequence_label0_str = model_inputs.model_inputs["metadata"][0]["labels"][0:sequence_max_len]
        expect_sequence_label0 = [label_vocab.index(label) for label in sequence_label0_str]

        ASSERT.assertEqual(sequence_label0, expect_sequence_label0)

        sequence_label4 = model_inputs.labels[4].tolist()
        sequence_label4_str = model_inputs.model_inputs["metadata"][4]["labels"]

        expect_sequence_label4 = [label_vocab.index(label) for label in sequence_label4_str] \
                                 + [label_vocab.padding_index] * (sequence_max_len - len(sequence_label4_str))
        ASSERT.assertEqual(sequence_label4, expect_sequence_label4)

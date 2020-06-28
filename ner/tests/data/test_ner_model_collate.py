#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 model collate

Authors: PanXu
Date:    2020/06/27 12:25:00
"""
import logging
from torch.utils.data import DataLoader

from easytext.data import ModelInputs
from easytext.utils.json_util import json2str
from easytext.utils import log_util

from ner.tests import ASSERT
from ner.data import NerModelCollate


log_util.config()


def test_ner_model_collate(conll2003_dataset, vocabulary):
    """
    测试 ner model collate
    :param conll2003_dataset: conll2003 数据集
    :param vocabulary: 在 conftest.py 中的 vocabulary 返回结果, 字典
    :return: None
    """

    token_vocab = vocabulary["token_vocabulary"]
    label_vocab = vocabulary["label_vocabulary"]

    sequence_max_len = 5
    model_collate = NerModelCollate(token_vocab=token_vocab,
                                    sequence_label_vocab=label_vocab,
                                    sequence_max_len=sequence_max_len)

    data_loader = DataLoader(dataset=conll2003_dataset,
                             batch_size=2,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=model_collate)

    for model_inputs in data_loader:

        model_inputs: ModelInputs = model_inputs

        logging.info(f"model inputs: {json2str(model_inputs)}")

        ASSERT.assertEqual(2, model_inputs.batch_size)

        ASSERT.assertEqual((2, sequence_max_len), model_inputs.labels.size())

        tokens = model_inputs.model_inputs["tokens"]

        expect_tokens = [[2, 3, 4, 5, 6], [13, 14, 0, 0, 0]]
        ASSERT.assertListEqual(expect_tokens, tokens.tolist())

        mask = (tokens != token_vocab.padding_index).long()

        sequence_lengths = mask.sum(dim=-1).tolist()

        ASSERT.assertListEqual([5, 2], sequence_lengths)



#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
测试 gaz vocabulary collate

Authors: PanXu
Date:    2021/02/02 10:11:00
"""

import os
import logging

from torch.utils.data import DataLoader

from easytext.data import Vocabulary, PretrainedVocabulary, LabelVocabulary
from easytext.data import GeneralPretrainedWordEmbeddingLoader
from easytext.utils.json_util import json2str

from ner.data.lattice import GazVocabularyCollate
from ner.data.lattice import Gazetteer

from ner.tests import ASSERT
from ner.data.lattice import LatticeModelCollate
from ner.data import VocabularyCollate


def test_gaz_model_collate(lattice_ner_demo_dataset, gaz_pretrained_embedding_loader):
    # 仅仅取前两个作为测试
    batch_instances = lattice_ner_demo_dataset[0:2]

    vocabulary_collate = VocabularyCollate()

    collate_result = vocabulary_collate(batch_instances)

    tokens = collate_result["tokens"]
    sequence_label = collate_result["sequence_labels"]

    token_vocabulary = Vocabulary(tokens=tokens,
                                  padding=Vocabulary.PADDING,
                                  unk=Vocabulary.UNK,
                                  special_first=True)

    label_vocabulary = LabelVocabulary(labels=sequence_label, padding=LabelVocabulary.PADDING)

    gazetter = Gazetteer(gaz_pretrained_word_embedding=gaz_pretrained_embedding_loader)

    gaz_vocabulary_collate = GazVocabularyCollate(gazetteer=gazetter)

    gaz_words = gaz_vocabulary_collate(batch_instances)

    gaz_vocabulary = Vocabulary(tokens=gaz_words,
                                padding=Vocabulary.PADDING,
                                unk=Vocabulary.UNK,
                                special_first=True)

    gaz_vocabulary = PretrainedVocabulary(vocabulary=gaz_vocabulary,
                                          pretrained_word_embedding_loader=gaz_pretrained_embedding_loader)

    lattice_model_collate = LatticeModelCollate(token_vocabulary=token_vocabulary,
                                                gazetter=gazetter,
                                                gaz_vocabulary=gaz_vocabulary,
                                                label_vocabulary=label_vocabulary)

    model_inputs = lattice_model_collate(batch_instances)

    logging.debug(json2str(model_inputs.model_inputs["metadata"]))

    metadata_0 = model_inputs.model_inputs["metadata"][0]

    # 陈元呼吁加强国际合作推动世界经济发展
    expect_gaz_words_0 = [["陈元"], [], ["呼吁"], ["吁加"], ["加强"], ["强国"], ["国际"], [],
                          ["合作"], [], ["推动"], [], ["世界"], [], ["经济"], [], ["发展"], []]

    gaz_words_0 = metadata_0["gaz_words"]

    ASSERT.assertListEqual(expect_gaz_words_0, gaz_words_0)

    gaz_list_0 = model_inputs.model_inputs["gaz_list"][0]

    expect_gaz_list_0 = list()

    for expect_gaz_word in expect_gaz_words_0:

        if len(expect_gaz_word) > 0:
            indices = [gaz_vocabulary.index(word) for word in expect_gaz_word]
            lengthes = [len(word) for word in expect_gaz_word]

            expect_gaz_list_0.append([indices, lengthes])

        else:
            expect_gaz_list_0.append([])

    logging.debug(f"expect_gaz_list_0: {json2str(expect_gaz_list_0)}\n gaz_list_0:{json2str(gaz_list_0)}")
    ASSERT.assertListEqual(expect_gaz_list_0, gaz_list_0)

    tokens_0 = model_inputs.model_inputs["tokens"]
    ASSERT.assertEqual((2, 19), tokens_0.size())
    sequence_label_0 = model_inputs.labels
    ASSERT.assertEqual((2, 19), sequence_label_0.size())

    # 新华社华盛顿４月2８日电（记者翟景升）
    expect_gaz_word_1 = [["新华社", "新华"],  # 新
                         ["华社"],  # 华
                         ["社华"],  # 社
                         ["华盛顿", "华盛"],  # 华
                         ["盛顿"],  # 盛
                         [],  # 顿
                         [],  # 4
                         [],  # 月
                         [],  # 2
                         [],  # 8
                         [],  # 日
                         [],  # 电
                         [],  # （
                         ["记者"],  # 记
                         [],  # 者
                         ["翟景升", "翟景"],  # 翟
                         ["景升"],  # 景
                         [],  # 升
                         []]  # ）

    metadata_1 = model_inputs.model_inputs["metadata"][1]
    gaz_words_1 = metadata_1["gaz_words"]

    ASSERT.assertListEqual(expect_gaz_word_1, gaz_words_1)

    expect_gaz_list_1 = list()

    for expect_gaz_word in expect_gaz_word_1:

        if len(expect_gaz_word) > 0:
            indices = [gaz_vocabulary.index(word) for word in expect_gaz_word]
            lengthes = [len(word) for word in expect_gaz_word]

            expect_gaz_list_1.append([indices, lengthes])

        else:
            expect_gaz_list_1.append([])

    gaz_list_1 = model_inputs.model_inputs["gaz_list"][1]
    ASSERT.assertListEqual(expect_gaz_list_1, gaz_list_1)

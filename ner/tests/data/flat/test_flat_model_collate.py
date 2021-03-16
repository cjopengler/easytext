#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
测试 flat model collate

Authors: PanXu
Date:    2021/02/24 10:14:00
"""
import os
import logging

import torch
import pytest

from easytext.utils.json_util import json2str
from easytext.utils.nn import tensor_util
from easytext.data import PretrainedWordEmbeddingLoader, GeneralPretrainedWordEmbeddingLoader
from easytext.data import Vocabulary, LabelVocabulary, PretrainedVocabulary
from easytext.utils.nn.tensor_util import is_tensor_equal

from ner import ROOT_PATH
from ner.tests.data.lattice.conftest import lattice_ner_demo_dataset, gaz_pretrained_embedding_loader
from ner.data.flat import FLATModelCollate, FlatPretrainedVocabulary
from ner.data import VocabularyCollate
from ner.data import FLATModelCollate
from ner.data.lattice import Gazetteer, GazVocabularyCollate

from ner.tests import ASSERT


@pytest.fixture(scope="session")
def character_pretrained_embedding_loader() -> PretrainedWordEmbeddingLoader:
    """
    生成 char 的 pretrained embedding loader
    """
    char_pretrained_embedding_file_path = "data/ner/flat/sample.gigaword_chn.all.a2b.bi.ite50.vec"
    char_pretrained_embedding_file_path = os.path.join(ROOT_PATH, char_pretrained_embedding_file_path)

    return GeneralPretrainedWordEmbeddingLoader(embedding_dim=50,
                                                pretrained_file_path=char_pretrained_embedding_file_path)


def test_flat_model_collate(lattice_ner_demo_dataset,
                            character_pretrained_embedding_loader,
                            gaz_pretrained_embedding_loader):
    """
    测试 flat model collate
    :return:
    """
    # 仅仅取前两个作为测试
    batch_instances = lattice_ner_demo_dataset[0:2]

    vocabulary_collate = VocabularyCollate()

    collate_result = vocabulary_collate(batch_instances)

    characters = collate_result["tokens"]
    sequence_label = collate_result["sequence_labels"]

    character_vocabulary = Vocabulary(tokens=characters,
                                      padding=Vocabulary.PADDING,
                                      unk=Vocabulary.UNK,
                                      special_first=True)
    character_vocabulary = PretrainedVocabulary(vocabulary=character_vocabulary,
                                                pretrained_word_embedding_loader=character_pretrained_embedding_loader)

    label_vocabulary = LabelVocabulary(labels=sequence_label, padding=LabelVocabulary.PADDING)

    gazetter = Gazetteer(gaz_pretrained_word_embedding_loader=gaz_pretrained_embedding_loader)

    gaz_vocabulary_collate = GazVocabularyCollate(gazetteer=gazetter)

    gaz_words = gaz_vocabulary_collate(batch_instances)

    gaz_vocabulary = Vocabulary(tokens=gaz_words,
                                padding=Vocabulary.PADDING,
                                unk=Vocabulary.UNK,
                                special_first=True)

    gaz_vocabulary = PretrainedVocabulary(vocabulary=gaz_vocabulary,
                                          pretrained_word_embedding_loader=gaz_pretrained_embedding_loader)

    flat_vocabulary = FlatPretrainedVocabulary(character_pretrained_vocabulary=character_vocabulary,
                                               gaz_word_pretrained_vocabulary=gaz_vocabulary)

    flat_model_collate = FLATModelCollate(token_vocabulary=flat_vocabulary,
                                          gazetter=gazetter,
                                          label_vocabulary=label_vocabulary)

    model_inputs = flat_model_collate(batch_instances)

    logging.debug(json2str(model_inputs.model_inputs["metadata"]))

    metadata_0 = model_inputs.model_inputs["metadata"][0]

    sentence = "陈元呼吁加强国际合作推动世界经济发展"

    # 陈元呼吁加强国际合作推动世界经济发展
    expect_squeeze_gaz_words_0 = ["陈元", "呼吁", "吁加", "加强", "强国", "国际", "合作", "推动", "世界", "经济", "发展"]

    squeeze_gaz_words_0 = metadata_0["squeeze_gaz_words"]

    ASSERT.assertListEqual(expect_squeeze_gaz_words_0, squeeze_gaz_words_0)

    expect_tokens = [character for character in sentence] + expect_squeeze_gaz_words_0

    tokens = metadata_0["tokens"]

    ASSERT.assertListEqual(expect_tokens, tokens)

    character_pos_begin = [index for index in range(len(sentence))]
    character_pos_end = [index for index in range(len(sentence))]

    squeeze_gaz_words_begin = list()
    squeeze_gaz_words_end = list()

    for squeeze_gaz_word in squeeze_gaz_words_0:
        index = sentence.find(squeeze_gaz_word)

        squeeze_gaz_words_begin.append(index)
        squeeze_gaz_words_end.append(index + len(squeeze_gaz_word) - 1)

    pos_begin = model_inputs.model_inputs["pos_begin"][0]
    pos_end = model_inputs.model_inputs["pos_end"][0]

    expect_pos_begin = character_pos_begin + squeeze_gaz_words_begin
    expect_pos_begin += [0] * (pos_begin.size(0) - len(expect_pos_begin))
    expect_pos_begin = torch.tensor(expect_pos_begin)

    expect_pos_end = character_pos_end + squeeze_gaz_words_end
    expect_pos_end += [0] * (pos_end.size(0) - len(expect_pos_end))
    expect_pos_end = torch.tensor(expect_pos_end)

    ASSERT.assertTrue(tensor_util.is_tensor_equal(expect_pos_begin, pos_begin))
    ASSERT.assertTrue(tensor_util.is_tensor_equal(expect_pos_end, pos_end))

    expect_character_length = len(sentence)
    expect_squeeze_gaz_word_length = len(expect_squeeze_gaz_words_0)

    character_length = model_inputs.model_inputs["sequence_length"][0]
    squeeze_word_length = model_inputs.model_inputs["squeeze_gaz_word_length"][0]

    ASSERT.assertEqual(expect_character_length, character_length.item())
    ASSERT.assertEqual(expect_squeeze_gaz_word_length, squeeze_word_length.item())











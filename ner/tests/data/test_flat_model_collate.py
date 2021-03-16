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

    gaz_words_indices = model_inputs.model_inputs["gaz_words"]

    ASSERT.assertEqual((2, 11), gaz_words_indices.size())

    metadata_0 = model_inputs.model_inputs["metadata"][0]

    # 陈元呼吁加强国际合作推动世界经济发展
    expect_squeeze_gaz_words_0 = ["陈元", "呼吁", "吁加", "加强", "强国", "国际", "合作", "推动", "世界", "经济", "发展"]

    sequeeze_gaz_words_0 = metadata_0["sequeeze_gaz_words"]

    ASSERT.assertListEqual(expect_squeeze_gaz_words_0, sequeeze_gaz_words_0)

    expect_squeeze_gaz_words_indices_0 = torch.tensor(
        [gaz_vocabulary.index(word) for word in expect_squeeze_gaz_words_0],
        dtype=torch.long)

    ASSERT.assertTrue(is_tensor_equal(expect_squeeze_gaz_words_indices_0,
                                      gaz_words_indices[0]))

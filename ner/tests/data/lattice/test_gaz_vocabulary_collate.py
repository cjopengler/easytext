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

from easytext.data.pretrained_word_embedding_loader import GeneralPretrainedWordEmbeddingLoader
from easytext.utils.json_util import json2str

from ner.data.lattice import GazVocabularyCollate
from ner.data.lattice import Gazetteer

from ner import ROOT_PATH
from ner.tests import ASSERT


def test_gaz_vocabulary_collate(lattice_ner_demo_dataset):

    pretrained_file_path = "data/ner/lattice_ner/ctb.50d.vec"
    pretrained_file_path = os.path.join(ROOT_PATH, pretrained_file_path)

    pretrained_embedding_loader = GeneralPretrainedWordEmbeddingLoader(embedding_dim=50,
                                                                       pretrained_file_path=pretrained_file_path)
    logging.debug(f"Begin load pretrained embedding...")
    pretrained_embedding_loader.load()
    logging.debug(f"End load pretrained embedding...")

    gazetter = Gazetteer(gaz_pretrained_word_embedding=pretrained_embedding_loader)

    gaz_vocabulary_collate = GazVocabularyCollate(gazetteer=gazetter)

    result = gaz_vocabulary_collate(lattice_ner_demo_dataset)

    logging.info(json2str(result))

    # 相应的句子是: "陈元呼吁加强国际合作推动世界经济发展", 得到的 gaz words 是
    expect_0 = [["陈元"], [], ["呼吁"], ["吁加"], ["加强"], ["强国"], ["国际"], [],
                ["合作"], [], ["推动"], [], ["世界"], [], ["经济"], [], ["发展"], []]
    gaz_words_0 = result["gaz_words"][0]
    ASSERT.assertListEqual(expect_0, gaz_words_0)

    # 新华社华盛顿４月2８日电（记者翟景升）
    expect_1 = [["新华社", "新华"],  # 新
                ["华社"],           # 华
                ["社华"],           # 社
                ["华盛顿", "华盛"],  # 华
                ["盛顿"],           # 盛
                [],                # 顿
                [],                # 4
                [],                # 月
                [],                # 2
                [],                # 8
                [],                # 日
                [],                # 电
                [],                # （
                ["记者"],           # 记
                [],                # 者
                ["翟景升", "翟景"],  # 翟
                ["景升"],           # 景
                [],                # 升
                []]                # ）
    gaz_words_1 = result["gaz_words"][1]
    ASSERT.assertListEqual(expect_1, gaz_words_1)



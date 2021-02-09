#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
产生 gaz 词汇表的 collate

Authors: PanXu
Date:    2021/02/02 09:55:00
"""

from typing import List, Dict

from easytext.data import Instance
from easytext.component.register import ComponentRegister

from ner.data.lattice import Gazetteer


class GazVocabularyCollate:
    """
    Gaz 词汇表构建的 Collate
    """

    def __init__(self, gazetteer: Gazetteer):
        self._gazetteer = gazetteer

    def __call__(self, instances: List[Instance]) -> List[List[str]]:
        """
        Gaz Vocabulary Collate
        :param instances: dataset 中的 instance
        :return: 二维数组，每一个 instance 对一个词 list. 例如:
                "陈元呼吁加强国际合作推动世界经济发展"(一个 instance 中对应的句子), 相应的结果
                ["陈元", "呼吁", "吁加", "加强", "强国", "国际", "合作", "推动", "世界", "经济", "发展"]
        """

        words = list()
        for instance in instances:
            flat_gaz_words = list()
            sentence = "".join([t.text for t in instance["tokens"]])

            for gaz_words in self._gazetteer.enumerate_match_list(sentence):
                for gaz_word in gaz_words:
                    flat_gaz_words.append(gaz_word)
            words.append(flat_gaz_words)

        return words


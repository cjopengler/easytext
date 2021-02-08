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

    def __call__(self, instances: List[Instance]) -> Dict[str, List[List[str]]]:

        words = list()
        for instance in instances:
            flat_gaz_words = list()
            sentence = "".join([t.text for t in instance["tokens"]])

            for gaz_words in self._gazetteer.enumerate_match_list(sentence):
                for gaz_word in gaz_words:
                    flat_gaz_words.append(gaz_word)
            words.append(flat_gaz_words)

        return words


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


@ComponentRegister.register("lattice")
class GazVocabularyCollate:
    """
    Gaz 词汇表构建的 Collate
    """

    def __init__(self, gazetteer: Gazetteer):
        self._gazetteer = gazetteer

    def __call__(self, instances: List[Instance]) -> Dict[str, List[List[str]]]:
        sentences = ["".join([t.text for t in instance["tokens"]]) for instance in instances]

        words = [self._gazetteer.enumerate_match_list(sentence) for sentence in sentences]

        return {"gaz_words": words}


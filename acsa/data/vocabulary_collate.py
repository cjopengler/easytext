#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
词汇表 collate

Authors: PanXu
Date:    2020/07/12 16:46:00
"""

from typing import List, Dict
from easytext.data import Instance

from easytext.data.tokenizer import EnTokenizer


class VocabularyCollate:
    """
    Vocabulary Collate
    """

    def __init__(self):
        # 这里使用基本的 EnTokenizer, 为了达到更好的效果，可以使用 Spacy.
        self._tokenizer = EnTokenizer(is_remove_invalidate_char=True)

    def __call__(self, instances: List[Instance]) -> Dict[str, List[str]]:
        """
        collate_fn
        :param instances: dataset 的 batch instances
        :return:
        """

        tokens = list()
        categories = list()
        labels = list()

        for instance in instances:
            sentence = instance["sentence"]

            sentence_tokens = self._tokenizer.tokenize(sentence)

            tokens.extend([t.text for t in sentence_tokens])

            categories.append(instance["category"])
            labels.append(instance["label"])

        return {"tokens": tokens,
                "categories": categories,
                "labels": labels}


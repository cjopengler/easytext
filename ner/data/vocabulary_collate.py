#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
词汇表的 collate

Authors: PanXu
Date:    2020/06/26 22:50:00
"""

from typing import List, Dict

from easytext.data import Instance


class VocabularyCollate:
    """
    词汇表的 collate 产生的结果用来构建词汇表
    """

    def __call__(self, instances: List[Instance]) -> Dict[str, List[List[str]]]:
        batch_tokens = [[t.text for t in instance["tokens"]] for instance in instances]
        batch_sequence_labels = [instance["sequence_label"] for instance in instances]

        collate_dict = {"tokens": batch_tokens,
                        "sequence_labels": batch_sequence_labels}

        return collate_dict

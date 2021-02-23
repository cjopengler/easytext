#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
FLAT model collate

Authors: PanXu
Date:    2021/02/23 11:13:00
"""

from typing import List, Tuple, Dict
import numpy as np

import torch

from easytext.data import ModelCollate
from easytext.data import Vocabulary
from easytext.data import LabelVocabulary
from easytext.data import Instance
from easytext.data import ModelInputs
from easytext.component.register import ComponentRegister

from ner.data.lattice import Gazetteer


@ComponentRegister.register(name_space="ner")
class FLATModelCollate(ModelCollate):
    """
    Lattice Model Collate
    """

    def __init__(self,
                 token_vocabulary: Vocabulary,
                 gazetter: Gazetteer,
                 label_vocabulary: LabelVocabulary):
        self._token_vocabulary = token_vocabulary
        self._gazetteer = gazetter
        self._label_vocabulary = label_vocabulary

    def __call__(self, instances: List[Instance]) -> ModelInputs:
        batch_size = 0

        batch_token_max_len = 0
        batch_character_max_len = 0
        batch_tokens = list()
        # 存放 character 以及 gaz word， 统一叫做 token, 其中[0: sequence_length] 是 character 序列，也就是实际的句子序列
        # [sequence_length: gaz_word_length] 是保存的 所有匹配上的 gaz word
        # 统一的 padding 我们使用 character_vocabulary padding 填充
        batch_token_indices = list()

        batch_sequence_length = list()
        batch_gaz_word_length = list()

        # 用来保存每一个 token 的开始和结束位置
        batch_pos_begin = list()
        batch_pos_end = list()

        batch_squeeze_gaz_words = list()
        batch_squeeze_gaz_word_indices = list()

        batch_gaz_words_list = list()

        batch_sequence_label_indices = None
        batch_metadatas = list()

        for instance in iter(instances):
            characters = [t.text for t in instance["tokens"]]

            if len(characters) > batch_character_max_len:
                batch_character_max_len = len(characters)

            batch_sequence_length.append(len(characters))

            # 产生 gaz words 已经相应的 index
            sentence = "".join([t.text for t in instance["tokens"]])

            gaz_words_list = self._gazetteer.enumerate_match_list(sentence)
            batch_gaz_words_list.append(gaz_words_list)

            token_len = len(characters) + len(squeeze_gaz_words)

            if token_len > batch_token_max_len:
                batch_token_max_len = token_len

            # 计算 pos
            # character 的 begin 和 end 是一样的
            pos_begin = [_ for _ in range(len(instance["tokens"]))]
            pos_end = [_ for _ in range(len(instance["tokens"]))]
            # 增加 gaz word 的 pos, 以及 gaz word
            squeeze_gaz_words = list()
            gaz_word_pos_begins = list()
            gaz_word_pos_ends = list()

            for index, gaz_words in enumerate(gaz_words_list):
                squeeze_gaz_words.extend(gaz_words)
                gaz_word_pos_begins.extend([index] * len(gaz_words))
                gaz_word_pos_ends.extend([(index + len(gaz_word) - 1) for gaz_word in gaz_words])

            batch_gaz_word_length.append(len(squeeze_gaz_words))

            pos_begin.extend(gaz_word_pos_begins)
            pos_end.extend(gaz_word_pos_ends)

            batch_pos_begin.append(pos_begin)
            batch_pos_end.append(pos_end)
            batch_squeeze_gaz_words.append(squeeze_gaz_words)

            batch_tokens.append(characters + squeeze_gaz_words)

        # index 转换
        for tokens in batch_tokens:
            token_indices = [self._token_vocabulary.index(t) for t in tokens]
            token_indices += [self._token_vocabulary.padding_index] * (batch_token_max_len - len(token_indices))
            batch_token_indices.append(token_indices)

        for pos_begin in batch_pos_begin:
            pos_begin += [0] * (batch_token_max_len - len(pos_begin))

        for pos_end in batch_pos_end:
            pos_end += [0] * (batch_token_max_len - len(pos_end))

        for instance in instances:

            if "sequence_label" in instance:
                sequence_label = list()
                sequence_label_indices = [self._label_vocabulary.padding_index] * batch_character_max_len

                for i, sl in enumerate(instance["sequence_label"][0: batch_token_max_len]):
                    sequence_label_indices[i] = self._label_vocabulary.index(sl)
                    sequence_label.append(sl)

                if batch_sequence_label_indices is None:
                    batch_sequence_label_indices = list()

                batch_sequence_label_indices.append(sequence_label_indices)

        batch_token_indices = torch.tensor(batch_token_indices, dtype=torch.long)

        batch_squeeze_gaz_word_indices = torch.tensor(batch_squeeze_gaz_word_indices, dtype=torch.long)

        if batch_sequence_label_indices is not None:
            batch_sequence_label_indices = torch.tensor(batch_sequence_label_indices, dtype=torch.long)

        return ModelInputs(batch_size=batch_size,
                           model_inputs={"tokens": batch_token_indices,
                                         "sequence_length": batch_squeeze_gaz_word_indices,
                                         "gaz_word_length": batch_gaz_word_length,
                                         "pos_begin": batch_pos_begin,
                                         "pos_end": batch_pos_end,
                                         "metadata": batch_metadatas},
                           labels=batch_sequence_label_indices)


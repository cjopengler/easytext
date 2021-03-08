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

        # 句子的字序列的最大长度
        character_max_len = 0

        # token: 存放 character 以及 squeeze gaz word， 统一叫做 token,
        # squeeze gaz word: 是打平之后的 gaz words
        # 其中[0: sequence_length] 是 character 序列，也就是实际的句子序列
        # [sequence_length: squeeze_gaz_word_length] 是保存的 所有匹配上的 squeeze gaz word
        # 统一的 padding 我们使用 character_vocabulary padding 填充
        token_max_len = 0  # token 的最大长度
        batch_tokens = list()
        batch_token_indices = list()

        batch_character_sequence_length = list()  # 字序列的实际长度

        batch_squeeze_gaz_words = list()
        batch_squeeze_gaz_word_length = list()  # gaz word 的实际长度

        # 用来保存每一个 token 的开始和结束位置
        batch_pos_begin = list()
        batch_pos_end = list()

        # label 序列
        batch_sequence_label_indices = None

        # metadatas
        batch_metadatas = [dict() for _ in range(len(instances))]
        batch_size = len(instances)

        for i, instance in enumerate(instances):

            metadata = batch_metadatas[i]

            characters = [t.text for t in instance["tokens"]]

            if len(characters) > character_max_len:
                character_max_len = len(characters)

            # 填充 character 序列长度
            batch_character_sequence_length.append(len(characters))

            # 计算 character 的 pos
            # character 的 begin 和 end 是一样的
            character_pos_begin = [_ for _ in range(len(instance["tokens"]))]
            character_pos_end = [_ for _ in range(len(instance["tokens"]))]

            # 增加 gaz word 的 pos, 以及 gaz word
            squeeze_gaz_words = list()
            metadata["squeeze_gaz_words"] = squeeze_gaz_words

            squeeze_gaz_word_pos_begin = list()
            squeeze_gaz_word_pos_end = list()

            # 产生 gaz words 已经相应的 index
            sentence = "".join([t.text for t in instance["tokens"]])

            gaz_words_list = self._gazetteer.enumerate_match_list(sentence)

            for index, gaz_words in enumerate(gaz_words_list):
                squeeze_gaz_words.extend(gaz_words)
                squeeze_gaz_word_pos_begin.extend([index] * len(gaz_words))
                squeeze_gaz_word_pos_end.extend([(index + len(gaz_word) - 1) for gaz_word in gaz_words])

            batch_squeeze_gaz_word_length.append(len(squeeze_gaz_words))

            pos_begin = character_pos_begin + squeeze_gaz_word_pos_begin
            pos_end = character_pos_end + squeeze_gaz_word_pos_end

            assert len(pos_begin) == len(pos_end), f"pos begin: {len(pos_begin)} 和 pos end: {len(pos_end)} 不相等"

            batch_pos_begin.append(pos_begin)
            batch_pos_end.append(pos_end)

            batch_squeeze_gaz_words.append(squeeze_gaz_words)

            tokens = characters + squeeze_gaz_words

            metadata["tokens"] = tokens

            batch_tokens.append(tokens)

            if len(tokens) > token_max_len:
                token_max_len = len(tokens)

        # index 转换
        for tokens in batch_tokens:
            token_indices = [self._token_vocabulary.index(t) for t in tokens]
            token_indices += [self._token_vocabulary.padding_index] * (token_max_len - len(token_indices))
            batch_token_indices.append(token_indices)

        for pos_begin in batch_pos_begin:
            pos_begin += [0] * (token_max_len - len(pos_begin))

        for pos_end in batch_pos_end:
            pos_end += [0] * (token_max_len - len(pos_end))

        for i, instance in enumerate(instances):

            metadata = batch_metadatas[i]

            if "sequence_label" in instance:
                sequence_label = list()
                sequence_label_indices = [self._label_vocabulary.padding_index] * character_max_len

                for i, sl in enumerate(instance["sequence_label"][0: character_max_len]):
                    sequence_label_indices[i] = self._label_vocabulary.index(sl)
                    sequence_label.append(sl)

                if batch_sequence_label_indices is None:
                    batch_sequence_label_indices = list()

                batch_sequence_label_indices.append(sequence_label_indices)

        batch_token_indices = torch.tensor(batch_token_indices, dtype=torch.long)
        batch_character_sequence_length = torch.tensor(batch_character_sequence_length, dtype=torch.long)
        batch_squeeze_gaz_word_length = torch.tensor(batch_squeeze_gaz_word_length, dtype=torch.long)
        batch_pos_begin = torch.tensor(batch_pos_begin, dtype=torch.long)
        batch_pos_end = torch.tensor(batch_pos_end, dtype=torch.long)

        if batch_sequence_label_indices is not None:
            batch_sequence_label_indices = torch.tensor(batch_sequence_label_indices, dtype=torch.long)

        return ModelInputs(batch_size=batch_size,
                           model_inputs={"tokens": batch_token_indices,
                                         "sequence_length": batch_character_sequence_length,
                                         "squeeze_gaz_word_length": batch_squeeze_gaz_word_length,
                                         "pos_begin": batch_pos_begin,
                                         "pos_end": batch_pos_end,
                                         "metadata": batch_metadatas},
                           labels=batch_sequence_label_indices)


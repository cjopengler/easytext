#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
预训练好的词汇表，带有预训练好的词向量

Authors: panxu(panxu@baidu.com)
Date:    2020/06/23 11:49:00
"""
from typing import Iterable, List
import torch
from .vocabulary import Vocabulary

from easytext.data.pretrained_word_embedding_loader import PretrainedWordEmbeddingLoader


class PretrainedVocabulary(Vocabulary):

    def __init__(self,
                 pretrained_word_embedding_loader: PretrainedWordEmbeddingLoader,
                 tokens: Iterable[List[str]],
                 padding: str,
                 unk: str,
                 special_first: bool,
                 other_special_tokens: List = None,
                 min_frequency: int = 1,
                 max_size: int = None
                 ):
        """
        初始化
        :param pretrained_word_embedding_loader: 预训练载入器
        :param tokens: (B, seq_len)
        :param padding: padding的字符串, 可以用 Vocabulary.PADDING, 如果为 None 或者 "", 表示不进行padding
        :param unk: unknown 的单词，可以用 Vocabulary.UNK, 如果为 None 或者 "", 表示不进行padding
        :param special_first: special 是指: padding, unk.
        True: 表示放在最前面, padding index=0, unk index=1; False: 表示放在最后面。
        这涉及到mask, 对于 token 来说，一般 padding_index = 0;
        而对于 label 来说, 如果需要对label,
        比如 sequence label 进行 padding 的时候, padding_index 和 unk_index 必须大于 label的数量，因为 小于 label 数量的是对应的
        label 分类。
        :param min_frequency: 小于 min_frequency 的会被过滤
        :param max_size: 词表的最大长度, 如果为None, 不限制词表大小
        """

        super().__init__(tokens=tokens,
                         padding=padding,
                         unk=unk,
                         special_first=special_first,
                         other_special_tokens=other_special_tokens,
                         min_frequency=min_frequency,
                         max_size=max_size)
        embedding_dict = pretrained_word_embedding_loader.load()

        embeddings = list()

        for index in range(len(self._index2token)):
            token = self._index2token[index]

            if token in embedding_dict:
                embeddings.append(embedding_dict[token])
            else:
                empty_vec = [0.] * pretrained_word_embedding_loader.dim()
                embeddings.append(empty_vec)

        self._embedding_matrix = torch.tensor(embeddings, dtype=torch.float)

    @property
    def embedding_matrix(self) -> torch.Tensor:
        """
        词向量 matrix
        """
        return self._embedding_matrix

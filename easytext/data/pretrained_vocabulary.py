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
import os
from typing import Iterable, List, Dict, Union
import torch
from .vocabulary import IVocabulary, Vocabulary

from easytext.data.pretrained_word_embedding_loader import PretrainedWordEmbeddingLoader


class PretrainedVocabulary(IVocabulary):
    """
    带有预训练词向量的词汇表。
    """

    EMBEDDING_MATRIX_FILE_NAME = "embedding_matrix.pt"

    def __init__(self,
                 vocabulary: Vocabulary,
                 pretrained_word_embedding_loader: PretrainedWordEmbeddingLoader,
                 ):
        """
        初始化
        :param vocabulary: 词汇表
        :param pretrained_word_embedding_loader: word embedding 载入器
        """

        self._vocabulary = vocabulary

        if pretrained_word_embedding_loader is not None:
            embedding_dict = pretrained_word_embedding_loader.load()

            embeddings = list()

            for index in range(self._vocabulary.size):
                token = self._vocabulary.token(index)

                if token in embedding_dict:
                    embeddings.append(embedding_dict[token])
                else:
                    empty_vec = [0.] * pretrained_word_embedding_loader.embedding_dim
                    embeddings.append(empty_vec)

            self._embedding_matrix = torch.tensor(embeddings, dtype=torch.float)
        else:
            self._embedding_matrix = None

    @property
    def embedding_matrix(self) -> torch.Tensor:
        """
        词向量 matrix
        """
        return self._embedding_matrix

    def save_to_file(self, directory: str) -> "PretrainedVocabulary":
        self._vocabulary.save_to_file(directory)

        # 将 embedding matrix 存起来
        embedding_matrix_file_path = os.path.join(directory, PretrainedVocabulary.EMBEDDING_MATRIX_FILE_NAME)
        torch.save(self._embedding_matrix, embedding_matrix_file_path)

        return self

    @classmethod
    def from_file(cls, directory: str) -> "PretrainedVocabulary":
        vocabulary = Vocabulary.from_file(directory)

        pretrianed_vocabulary = cls(pretrained_word_embedding_loader=None,
                                    vocabulary=vocabulary)
        embedding_matrix_file_path = os.path.join(directory, PretrainedVocabulary.EMBEDDING_MATRIX_FILE_NAME)
        pretrianed_vocabulary._embedding_matrix = torch.load(embedding_matrix_file_path)

        return pretrianed_vocabulary

    def __len__(self):
        return len(self._vocabulary)

    @property
    def unk(self):
        return self._vocabulary.unk

    @property
    def padding(self):
        return self._vocabulary.padding

    @property
    def other_special_tokens(self):
        return self._vocabulary.other_special_tokens

    @property
    def padding_index(self) -> Union[None, int]:
        """
        :return: 获取 padding 的 index, 如果 padding 没有设置，那么返回 None; 否则，返回实际的index.
        """
        return self._vocabulary.padding_index

    def index(self, token: str) -> int:
        """
        获取token的index
        :param token: 输入的token
        :return: token 的 index
        """

        return self._vocabulary.index(token)

    def token(self, index: int) -> str:
        """
        获取 index 的 token
        :param index: 指定 index
        :return: 当前 index 的 token
        """
        return self._vocabulary.token(index)

    @property
    def size(self):
        return self._vocabulary.size


#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
Glove 的词向量

Authors: panxu(panxu@baidu.com)
Date:    2020/06/25 09:57:00
"""

from .general_pretrained_word_embedding_loader import GeneralPretrainedWordEmbeddingLoader


class GloveLoader(GeneralPretrainedWordEmbeddingLoader):
    """
    Glove Word Embedding Loader.
    glove.6B.50d.txt
    glove.6B.100d.txt
    glove.840B.300d.txt
    """

    def __init__(self, embedding_dim: int, pretrained_file_path: str):
        """
        初始化 glove embedding loader
        :param embedding_dim: embedding 维度
        :param pretrained_file_path: 预训练的文件路径, 一般是指:
        glove.6B.50d.txt glove.6B.100d.txt glove.840B.300d.txt ...
        """
        super().__init__(embedding_dim=embedding_dim,
                         pretrained_file_path=pretrained_file_path,
                         encoding="utf-8",
                         separator=" ",
                         skip_num_line=0)

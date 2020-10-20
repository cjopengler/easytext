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

    def __init__(self, embedding_dim: int, pretrained_file_path: str, max_size: int = None):
        """
        初始化 glove embedding loader
        :param embedding_dim: embedding 维度
        :param pretrained_file_path: 预训练的文件路径, 一般是指:
        glove.6B.50d.txt glove.6B.100d.txt glove.840B.300d.txt ...
        :param max_size: 载入的最大向量数。因为某些词向量可能很大，比如 glove.840B.300d.txt 有 5G 大小，
        通过该参数进行限制，在比如测试阶段是非常有益的，或者一些内存小的机器上。
        """
        super().__init__(embedding_dim=embedding_dim,
                         pretrained_file_path=pretrained_file_path,
                         encoding="utf-8",
                         separator=" ",
                         skip_num_line=0,
                         max_size=max_size)

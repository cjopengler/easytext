#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
通用的 预训练词向量载入器

Authors: panxu(panxu@baidu.com)
Date:    2020/06/24 11:44:00
"""

from tqdm import tqdm
from typing import Dict, List

from .pretrained_word_embedding_loader import PretrainedWordEmbeddingLoader


class GeneralPretrainedWordEmbeddingLoader(PretrainedWordEmbeddingLoader):
    """
    一般的通用预训练好的词向量载入器。试用范围: 词向量是一个文本文件，每一行是一个词向量。格式是:

    [word] [value] [value] ...

    比如 Glove6B 这些
    """

    def __init__(self,
                 embedding_dim: int,
                 pretrained_file_path: str,
                 encoding="utf-8",
                 separator: str = " ",
                 skip_num_line: int = 0):
        """
        初始化
        :param embedding_dim: embedding dim 大小
        :param pretrained_file_path: 预训练好的词向量文件路径
        :param encoding: 文件编码格式，默认是 "utf-8"
        :param separator: 每一行词向量的分隔符，默认是 " " 空格。
        :param skip_num_line: 跳过的行数，因为有些词向量头部 不是词向量。
        """
        self._embedding_dim = embedding_dim
        self._pretrained_file_path = pretrained_file_path
        self._separator = separator
        self._encoding = encoding
        self._skip_num_line = skip_num_line

    def load(self) -> Dict[str, List[float]]:
        """
        预训练的载入方法
        return: embedding dict, key 是 token, value: List[float] 向量，例如:
        {"the": [1.0, 2.0, ...]}
        """
        embedding_dict = dict()
        with open(self._pretrained_file_path, encoding=self._encoding) as f:
            for line_no, line in tqdm(enumerate(f)):

                if line_no < self._skip_num_line:
                    continue

                items = line.split(sep=self._separator)
                token = items[0]
                vec = [float(v) for v in items[1:]]

                assert len(vec) == self._embedding_dim, f"读取的向量维度: {len(vec)} != 设置的维度: {self._embedding_dim}"

                embedding_dict[token] = vec
        return embedding_dict

    @property
    def embedding_dim(self) -> int:
        """
        词向量维度
        return: 词向量维度
        """
        return self._embedding_dim


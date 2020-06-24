#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
预训练好的 词向量 载入其

Authors: panxu(panxu@baidu.com)
Date:    2020/06/24 11:35:00
"""

from typing import List, Dict


class PretrainedWordEmbeddingLoader:
    """
    预训练词向量 loader
    """

    def load(self) -> Dict[str, List[float]]:
        """
        载入预训练的词向量
        return: 返回词向量字典. 字典的 key 是 token, value 是 float List, 也就是向量内容;
        """
        raise NotImplementedError()

    def dim(self):
        """
        return: 词向量的维度
        """
        raise NotImplementedError
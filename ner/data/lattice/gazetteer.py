#!/usr/bin/env python 2.7
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#

from typing import List, Optional

from easytext.data.pretrained_word_embedding_loader import GeneralPretrainedWordEmbeddingLoader

from ner.data.lattice import Trie


class Gazetteer:
    """
    构建 gaz 的对应词汇表, 提供根据当前词获得词序列的能力
    """

    def __init__(self, gaz_pretrained_word_embedding: GeneralPretrainedWordEmbeddingLoader):
        """
        初始化
        :param gaz_pretrained_word_embedding: gaz pretrained word embedding
        """

        # 树状存储
        self.trie = Trie()

        # 分隔符是 空 的，那么在 trie 中匹配的就是词了
        self.space = ""

        self.__init_trie(gaz_pretrained_word_embedding=gaz_pretrained_word_embedding)

    def __init_trie(self, gaz_pretrained_word_embedding: GeneralPretrainedWordEmbeddingLoader):

        for word, _ in gaz_pretrained_word_embedding.embedding_table:
            self.trie.insert(word)

    def enumerate_match_list(self, word: str) -> List[str]:
        """
        枚举所有匹配的 词 以及 字序列。
        例如: word = "长江大桥", 相应的所有开始位置不变，结束位置逐渐推进的子序列是
        ["长江大", "长江"], 要查找的全集是["长江大桥", "长江大", "长江"], 将命中的序列，用 space 分隔开表示，
        返回的结果就是 ["长江大桥", "长江"], ("长江大" 没有命中)
        :param word: 词 或者 称之为 字的列表
        :return: 所有命中的字序列词列表
        """

        match_list = self.trie.enumerate_match(word, self.space)

        return match_list

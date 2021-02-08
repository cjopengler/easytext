#!/usr/bin/env python 2.7
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
import logging
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

        if gaz_pretrained_word_embedding.embedding_table is None:
            logging.info(f"Begin: gaz pretrained word embedding load...")
            gaz_pretrained_word_embedding.load()
            logging.info(f"End: gaz pretrained word embedding load.")

        for word, _ in gaz_pretrained_word_embedding.embedding_table.items():
            self.trie.insert(word)

    def enumerate_match_list(self, word: str) -> List[List[str]]:
        """
        枚举word中每一个字以及以该字为开始的所有匹配的词。
        例如: word = "中国长江大桥"
        步骤1:
        分割成子句，"中国长江大桥", "国长江大桥", "长江大桥", "江大桥", "大桥", "桥"
        步骤2:
        针对每一个子句，获取匹配的词。
        例如: "长江大桥", 要查找的全集是["长江大桥", "长江大", "长江"],
        将命中的词是 ["长江大桥", "长江"]，("长江大" 没有命中)
        最终返回的结果是每一个字对应的子序列:
        [[中国], [], [长江, 长江大桥], [江大桥], [大桥], []]
        :param word: 词 或者 称之为 字的列表, 或者称之为句子
        :return: 命中每一个位置的字的所有子序列,
        """

        match_list = list()
        for i in range(len(word)):
            matched_sub_words = self.trie.enumerate_match(word[i:], self.space)
            match_list.append(matched_sub_words)

        return match_list

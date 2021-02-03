#!/usr/bin/env python 2.7
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#

import collections


class TrieNode:
    """
    Trie Node
    """
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False


class Trie:
    """
    构建一个 Trie 用来进行查找，判断一个词是否在 Trie 里面, 与 Lattice 论文的源码一样
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        插入一个词到当前 trie 中
        :param word: 词
        :return:
        """
        
        current = self.root
        for letter in word:
            current = current.children[letter]
        current.is_word = True

    def search(self, word: str) -> bool:
        """
        搜索一个词是否在 trie 中
        :param word: 词
        :return: True: 存在该词; False: 不存在该词
        """

        current = self.root
        for letter in word:
            current = current.children.get(letter)

            if current is None:
                return False
        return current.is_word

    def starts_with(self, prefix: str) -> bool:
        """
        prefix 是否在 trie 中, 实际上该函数与 search是一样的
        :param prefix: 前缀
        :return: True: 存在该前缀; False: 不存在该前缀
        """
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True

    def enumerate_match(self, word: str, space: str = "_"):
        """
        枚举出所有命中 word 以及相应子序列。例如: word = "长江大桥", 相应的所有开始位置不变，结束位置逐渐推进的子序列是
        ["长江大", "长江"], 要查找的全集是["长江大桥", "长江大", "长江"], 将命中的序列，用 space 分隔开表示，
        返回的结果就是 ["长_江_大_桥", "长_江"], ("长江大" 没有命中)
        :param word: 要搜索的词
        :param space: 命中的序列中的字的分隔符
        :return:
        """
        matched = []

        while len(word) > 1:
            if self.search(word):
                matched.append(space.join(word[:]))
            del word[-1]
        return matched


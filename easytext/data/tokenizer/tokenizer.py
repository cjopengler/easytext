#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
tokenizer

Authors: panxu(panxu@baidu.com)
Date:    2020/05/13 16:17:00
"""

from typing import List
import unicodedata

from easytext.data.tokenizer.token import Token


class Tokenizer:
    """
    Tokenizer
    """

    def __init__(self, is_remove_invalidate_char: bool = False):
        """
        初始化
        :param is_remove_invalidate_char: 是否移除无效的字符。True: 移除; False: 不移除。
        无效字符包括: unicode: 0x0, 0xFFFD 以及 控制字符。一般来说，在分类任务中，应该设置为 True,
        但是在 序列标注任务中要小心，因为移除了会导致标注数据中的 token index 发生变化。
        这几个字符是非常特殊的字符，如果数据很干净，没有这些字符，那么，这个参数设置 True 还是 False
        都不会受到影响。
        """
        self._is_removed_invalidate_char = is_remove_invalidate_char

    def _is_whitespace(self, char):
        """
        Checks whether `chars` is a whitespace character.
        判断是否是 空白字符， 这包括 " ", "\t", "\n", and "\r".

        :param char:
        :return:
        """

        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    def _is_control(self, char):
        """
        检查是否是 控制字符。
        :param char: 字符
        :return:
        """
        """"""

        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def _clean_text(self, text):
        """
        将无效的字符以及空白符进行移除和替换
        :param text: 输入的文本
        :return:
        """
        output = []

        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or self._is_control(char):
                if self._is_removed_invalidate_char:
                    continue

            output.append(char)

        return "".join(output)

    def tokenize(self, text: str) -> List[Token]:
        """
        tokenize
        :param text: 输入文本
        :return:
        """
        return self._tokenize(self._clean_text(text))

    def _tokenize(self, text: str) -> List[Token]:
        """
        实际进行 tokenize 的方法
        :param text:
        :return:
        """
        raise NotImplementedError

    def batch_tokenize(self, batch_text: List[str]) -> List[List[Token]]:
        """
        批量 tokenize
        :param batch_text:
        :return:
        """
        raise NotImplementedError()

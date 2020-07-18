#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
英文tokenizer

Authors: panxu(panxu@baidu.com)
Date:    2020/06/09 11:18:00
"""
from typing import List

from .token import Token
from .tokenizer import Tokenizer


class EnTokenizer(Tokenizer):
    """
    英文tokenizer, 会对文本进行分隔成单词，再将单词全部转换成小写，作为token.
    默认的分隔符是 " "
    """

    def __init__(self, is_remove_invalidate_char: bool = False, separator: str = " "):
        """
        初始化
        :param is_remove_invalidate_char: 是否移除无效的字符，注意在实体识别的时候要小心，因为移除可能导致label无法对其。
        :param separator: 分隔符，用来切分英文文本成单词，默认是 " "
        """
        super().__init__(is_remove_invalidate_char=is_remove_invalidate_char)
        self._separator = separator

    def _tokenize(self, text: str) -> List[Token]:

        return [Token(word.lower()) for word in text.split(self._separator)]

    def batch_tokenize(self, batch_text: List[str]) -> List[List[Token]]:

        return [self._tokenize(text) for text in batch_text]
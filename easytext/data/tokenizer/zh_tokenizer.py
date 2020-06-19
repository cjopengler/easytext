#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2020/05/13 16:26:00
"""
from typing import List

from easytext.data.tokenizer.token import Token
from easytext.data.tokenizer.tokenizer import Tokenizer


class ZhTokenizer(Tokenizer):
    """
    中文 Tokenizer
    """

    def _tokenize(self, text: str) -> List[Token]:
        """
        Tokenize
        :param text: 文本
        :return:
        """
        return [Token(t) for t in text]

    def batch_tokenize(self, batch_text: List[str]) -> List[List[Token]]:
        """
        批量 tokenize
        :param batch_text: 文本
        :return:
        """
        return [self.tokenize(t) for t in batch_text]


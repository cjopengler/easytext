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

    def _tokenize(self, text: str) -> List[Token]:
        return [Token(char.lower()) for char in text]

    def batch_tokenize(self, batch_text: List[str]) -> List[List[Token]]:

        return [self._tokenize(text) for text in batch_text]
#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
brief

Authors: PanXu
Date:    2020/09/13 15:29:00
"""

from easytext.data.tokenizer import ZhTokenizer

def test_zh_tokenizer():

    tokenizer = ZhTokenizer(is_remove_invalidate_char=False)



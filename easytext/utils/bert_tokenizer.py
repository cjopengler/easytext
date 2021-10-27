#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
bert tokenizer component

Authors: PanXu
Date:    2020/11/03 18:11:00
"""

from transformers import BertTokenizer

from easytext.component.register import ComponentRegister
from easytext.component.component_builtin_key import ComponentBuiltinKey


@ComponentRegister.register(typename="BertTokenizer", name_space=ComponentBuiltinKey.EASYTEXT_NAME_SPACE)
def bert_tokenizer(bert_dir: str):
    return BertTokenizer.from_pretrained(bert_dir)

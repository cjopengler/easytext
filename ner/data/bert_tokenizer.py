#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
bert tokenizer

Authors: PanXu
Date:    2020/11/03 18:11:00
"""

from transformers import BertTokenizer

from easytext.component.register import ComponentRegister


@ComponentRegister.register(typename="BertTokenizer", name_space="ner")
def bert_tokenizer(bert_dir: str):
    return BertTokenizer.from_pretrained(bert_dir)

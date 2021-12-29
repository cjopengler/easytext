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
from typing import List, Tuple
from transformers import BertTokenizerFast

from easytext.component.register import ComponentRegister
from easytext.component.component_builtin_key import ComponentBuiltinKey


@ComponentRegister.register(typename="BertTokenizer", name_space=ComponentBuiltinKey.EASYTEXT_NAME_SPACE)
def bert_tokenizer(bert_dir: str):
    return BertTokenizerFast.from_pretrained(bert_dir)


def mapping_label(token_offset_mapping: List[Tuple[int, int]], labels: List) -> List:
    """
    bert tokenizer 会对 index 进行转化, 所以需要将其转换后的 index 与 label 对应起来。
    转换前 index = label index -> 转换后 index -> 转换后的 label index
    :param token_offset_mapping: bert tokenizer 返回的 offset_mapping, [(being, end), ...], 其 list index 就是
                                转换前的 token index
    :param labels: 标签列表
    :return:
    """


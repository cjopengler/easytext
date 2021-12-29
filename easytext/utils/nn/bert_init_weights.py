#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
使用 bert 的 init weights

Authors: PanXu
Date:    2021/11/08 08:44:00
"""

from torch.nn import Module
from torch import nn

from transformers import BertConfig


class BertInitWeights:
    """
    bert 初始化权重

    参考: BertPreTrainedModel._init_weights
    """

    def __init__(self, bert_config: BertConfig):
        self.config = bert_config

    def __call__(self, module: Module) -> None:
        """
        参考: BertPreTrainedModel._init_weights
        :param module: 模型
        :return: None
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()





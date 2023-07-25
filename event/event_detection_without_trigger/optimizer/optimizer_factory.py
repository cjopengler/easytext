#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
优化器

Authors: panxu(panxu@baidu.com)
Date:    2020/06/17 22:20:00
"""
from typing import Dict
import torch
from torch.optim import Optimizer
from torch.optim import SGD
from easytext.model import Model
from easytext.optimizer import OptimizerFactory


class EventOptimizerFactory(OptimizerFactory):
    """
    优化器工厂
    """

    def __init__(self, is_fine_tuning: bool = False):
        """
        初始化
        :param is_fine_tuning: 是否进行 fine tuning, 默认是不进行 fine tuning
        """
        self._is_fine_tuning = is_fine_tuning

    def create(self, model: Model) -> "Optimizer":
        """
        创建 optimizer
        :param model: 模型
        """

        sentence_embedding_param_name = "_sentence_embedder.weight"

        parameter_dict: Dict[str, torch.nn.Parameter] = \
            {name: parameter for name, parameter in model.named_parameters()}

        sentence_embedding_param = parameter_dict.pop(sentence_embedding_param_name)

        if self._is_fine_tuning:
            # 设置 requires_grad = True
            sentence_embedding_param.requires_grad = True

            # 分组设置 params 对于 微调的参数 设置 lr 要小一些
            params = [
                {"params": parameter_dict.values()},
                {"params": [sentence_embedding_param], "lr": 1e-3}
            ]

            optimizer = SGD(params=params,
                            lr=0.01)
        else:
            # 将不需要 fine tuning 参数设置成 不需要梯度更新
            # 同时也不需要将这个参数放在 optimizer 中
            sentence_embedding_param.requires_grad = False
            optimizer = SGD(params=parameter_dict.values(),
                            lr=0.01)
        return optimizer

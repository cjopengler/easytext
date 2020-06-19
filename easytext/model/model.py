#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
模型

Authors: panxu(panxu@baidu.com)
Date:    2020/05/18 10:08:00
"""

from typing import Any, Dict
import torch
from torch.nn import Module

from .model_outputs import ModelOutputs


class Model(Module):
    """
    模型
    """

    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        raise NotImplementedError()

    def forward(self, *input: Any, **kwargs: Any) -> ModelOutputs:
        pass



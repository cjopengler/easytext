#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
cuda 相关的工具

Authors: panxu(panxu@baidu.com)
Date:    2020/06/01 15:34:00
"""

import torch


def cuda(obj, cuda_device: torch.device):
    """
    将复杂对象全部移动到 cuda 中
    :param obj: obj
    :param cuda_device: cuda device
    :return:
    """
    if isinstance(obj, torch.Tensor):
        return obj.cuda(cuda_device)
    elif isinstance(obj, dict):
        return {key: cuda(value, cuda_device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [cuda(item, cuda_device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([cuda(item, cuda_device) for item in obj])
    else:
        return obj

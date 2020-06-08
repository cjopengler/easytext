#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2020/06/01 15:40:00
"""

import torch


def has_tensor(obj) -> bool:
    """
    检查 对象中是否包含 Tensor
    :param obj:
    :return: True: 包含 Tensor; False: 没有 Tensor
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False

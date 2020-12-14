#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
分布式训练用到的相关工具

Authors: PanXu
Date:    2020/11/09 18:08:00
"""
from typing import Union, Optional
from torch import distributed


class NotCall:
    pass


class DistributedFuncWrapper:
    """
    分布式训练函数包装器, 会根据指定的进程 rank 进行处理
    """

    def __init__(self, dst_rank: Optional[int] = 0):
        """
        初始化
        :param dst_rank: 需要调用函数的 dst rank, 如果是 None, 则表示直接返回不会进行分布式处理
        """
        self._dst_rank = dst_rank

    @property
    def dst_rank(self):
        return self._dst_rank

    def __call__(self, func, *args, **kwargs) -> Union[NotCall, object]:
        """
        在指定的 dst rank 进程运行该函数
        :param func: 函数对象
        :param args: 函数的参数
        :param kwargs: 函数的参数
        :return: 如果没有被运行, 返回 NotCall 对象; 否则, 返回 函数的返回值。
        """

        if self._dst_rank is not None:
            if distributed.get_rank() == self._dst_rank:
                return func(*args, **kwargs)
            return NotCall()
        else:
            return func(*args, **kwargs)

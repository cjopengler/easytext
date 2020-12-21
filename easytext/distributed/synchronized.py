#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
需要同步的接口

Authors: PanXu
Date:    2020/12/17 19:31:00
"""
from typing import List, Dict, Union, Tuple
from torch import Tensor
from torch.distributed import ReduceOp


class Synchronized:
    """
    需要同步数据的接口, 进行多 GPU 训练时候，训练中需要同步的数据，需要继承该接口
    """

    def to_synchronized_data(self) -> Tuple[Union[Dict[Union[str, int], Tensor], List[Tensor], Tensor], ReduceOp]:
        """
        将需要同步的数据转换成 tensor.
        :return: 需要同步的数据, 以及 同步的操作
        """

        raise NotImplementedError()

    def from_synchronized_data(self,
                               sync_data: Union[Dict[Union[str, int], Tensor], List[Tensor], Tensor],
                               reduce_op: ReduceOp) -> None:
        """
        从 tensor 数据恢复
        :param sync_data: 进行恢复的 tensor 数据
        :param reduce_op: 同步的操作，与 to_synchronized_data 的一致
        :return: None
        """
        raise NotImplementedError()

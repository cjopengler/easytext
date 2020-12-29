#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
同步数据

Authors: PanXu
Date:    2020/12/16 16:51:00
"""

from typing import Dict, Union

import torch
from torch import distributed as TorchDist
from torch.distributed import ReduceOp
from typing import List, Dict, Union, Tuple
from torch import Tensor


class Sync:
    """
    同步数据
    """

    @staticmethod
    def sync(value: Union[Tensor, int, float, Dict[str, Tensor]],
             device: torch.device,
             op: ReduceOp = ReduceOp.SUM) -> Union[Tensor, int, float, Dict[str, Tensor]]:
        """
        同步数据, in place 操作
        :param value: 需要同步的数据
        :param device: 当前 device
        :param op: op
        :return: 同步后的数据
        """
        if isinstance(value, int) or isinstance(value, float):
            return Sync.sync_value(value=value, device=device, op=op)
        elif isinstance(value, Tensor):
            return Sync.sync_tensor(value=value, device=device, op=op)
        elif isinstance(value, Dict):
            return Sync.sync_tensor_dict(value=value, device=device, op=op)
        else:
            raise RuntimeError(f"value 类型: {type(value)} 非法! 必须是 Tensor, int, float, Dict[str, Tensor] 中类型")


    @staticmethod
    def sync_tensor(value: Tensor,
                    device: torch.device,
                    op: ReduceOp = ReduceOp.SUM) -> Tensor:
        """
        同步 tensor
        :param value: tensor
        :param device: tensor 会放到指定 device 上进行同步, 这不是 tensor 本身带的 device
        :param op: reduce op
        :return: 返回同步后的 value, value 的 device 也会赋值成先前的
        """

        assert isinstance(value, Tensor), f"value: {type(value)} 类型必须是 Tensor"
        value_device = value.device

        if TorchDist.get_backend() == "nccl":
            assert device.type == "cuda", f"backend nccl, device 需要 cuda"

        sync_value = value.to(device)

        TorchDist.all_reduce(tensor=sync_value, op=op)

        return sync_value.to(value_device)

    @staticmethod
    def sync_value(value: Union[int, float],
                   device: torch.device,
                   op: ReduceOp = ReduceOp.SUM) -> Union[int, float]:
        """
        同步一个数值, 会将该数值先转换成 tensor 再进行同步
        :param value: 需要同步的数值
        :param device: 会放到指定 device 上进行同步
        :param op: reduce op
        :return: 同步后的数值，类型保持不变
        """

        if isinstance(value, float):
            sync_value_tensor = torch.tensor(value, dtype=torch.float, device=device)
        elif isinstance(value, int):
            sync_value_tensor = torch.tensor(value, dtype=torch.long, device=device)
        else:
            raise RuntimeError(f"value 字典，值必须是 float 或者 int 类型")

        sync_value_tensor = Sync.sync_tensor(value=sync_value_tensor, device=device, op=op)

        return sync_value_tensor.item()

    @staticmethod
    def sync_tensor_dict(value: Dict[str, Tensor],
                         device: torch.device,
                         op: ReduceOp = ReduceOp.SUM) -> Dict[str, Tensor]:
        """
        同步一个数值, 会将该数值先转换成 tensor 再进行同步
        :param value: 需要同步的数值
        :param device: 会放到指定 device 上进行同步
        :param op: reduce op
        :return: 同步后的数值，类型保持不变
        """
        for k, tensor in value.items():
            tensor = Sync.sync_tensor(value=tensor, device=device, op=op)

            value[k] = tensor

        return value

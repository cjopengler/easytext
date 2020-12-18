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
    def sync_tensor(value: Tensor,
                   device: torch.device,
                   op: ReduceOp = ReduceOp.SUM) -> Tensor:

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

        if isinstance(value, float):
            sync_value_tensor = torch.tensor(value, dtype=torch.float, device=device)
        elif isinstance(value, int):
            sync_value_tensor = torch.tensor(value, dtype=torch.long, device=device)
        else:
            raise RuntimeError(f"value 字典，值必须是 float 或者 int 类型")

        if TorchDist.get_backend() == "nccl":
            assert device.type == "cuda", f"backend nccl, device 需要 cuda"

        TorchDist.all_reduce(tensor=sync_value_tensor, op=op)

        return sync_value_tensor.item()

    @staticmethod
    def sync_dict(value: Dict[str, Union[int, float]],
                  device: torch.device,
                  op: ReduceOp = ReduceOp.SUM) -> Dict[str, Union[int, float]]:
        """
        多 GPU 同步字典数据, in place 操作，修改后的数据会重新存放到 value 中。
        :param value: 字典，只能同步一层字典，而且值必须 int 或者 float
        :param device: 当前计算的 device
        :param op: 操作
        :return: value, in place
        """
        sync_values = list()

        for k, v in value.items():
            sync_values.append(v)

            if isinstance(v, float):
                sync_value_tensor = torch.tensor(sync_values, dtype=torch.float, device=device)
            elif isinstance(v, int):
                sync_value_tensor = torch.tensor(sync_values, dtype=torch.long, device=device)
            else:
                raise RuntimeError(f"value 字典，值必须是 float 或者 int 类型")

        if TorchDist.get_backend() == "nccl":
            assert device.type == "cuda", f"backend nccl, device 需要 cuda"

        TorchDist.all_reduce(tensor=sync_value_tensor, op=op)

        for k, _, v_tensor in zip(value.items(), sync_value_tensor):
            value[k] = v_tensor.item()

        return value

    @staticmethod
    def sync_list(value: Dict[str, Union[int, float]],
                  device: torch.device,
                  op: str = ReduceOp.SUM) -> Dict[str, Union[int, float]]:
        """
        多 GPU 同步字典数据, in place 操作，修改后的数据会重新存放到 value 中。
        :param value: 字典，只能同步一层字典，而且值必须 int 或者 float
        :param device: 当前计算的 device
        :param op: 操作
        :return: value, in place
        """
        sync_values = list()

        for k, v in value.items():
            sync_values.append(v)

            if isinstance(v, float):
                sync_value_tensor = torch.tensor(sync_values, dtype=torch.float, device=device)
            elif isinstance(v, int):
                sync_value_tensor = torch.tensor(sync_values, dtype=torch.long, device=device)
            else:
                raise RuntimeError(f"value 字典，值必须是 float 或者 int 类型")

        if TorchDist.get_backend() == "nccl":
            assert device.type == "cuda", f"backend nccl, device 需要 cuda"

        TorchDist.all_reduce(tensor=sync_value_tensor, op=op)

        for k, _, v_tensor in zip(value.items(), sync_value_tensor):
            value[k] = v_tensor.item()

        return value






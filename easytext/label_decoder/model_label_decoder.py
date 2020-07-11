#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
将模型输出转化成最终的label

Authors: PanXu
Date:    2020/07/05 10:20:00
"""
import torch
from typing import List, Tuple

from easytext.model import ModelOutputs
from easytext.data import LabelVocabulary


class ModelLabelDecoder:
    """
    模型 label 解码器，
    1. 在 predict 中直接使用, 将模型结果产生最终输出结果
    2. 在 metric 中使用 decode_label_index 将 logits 转化成 label index 用来进行 metric 计算
    """

    def decode_label_index(self, model_outputs: ModelOutputs) -> torch.LongTensor:
        """
        将模型输出解码成 index
        :param model_outputs: 模型输出
        :return:
        """
        raise NotImplementedError()

    def decode_label(self, model_outputs: ModelOutputs, label_indices: torch.LongTensor) -> List:
        """
        将 label index 解码成最终目标结果, 会使用到 label_vocabulary, 对 label index 进行实际的转换。
        对于分类来说, 假设 label index = [1, 5, 7], 返回结果是 labels=["军事", "政治", "经济"];
        对于其他任务来说也同样，返回的是用户能读懂的结果。
        :param model_outputs: 模型输出
        :param label_indices: decode_label_index 解码出的 label index
        :return: 转换后的最终结果
        """
        raise NotImplementedError()

    def __call__(self, model_outputs: ModelOutputs) -> List:
        """
        将模型输出解码成最终用户需要的输出
        :param model_outputs: 模型输出
        :return: 转化后的最终结果
        """
        label_indices = self.decode_label_index(model_outputs=model_outputs)
        return self.decode_label(model_outputs=model_outputs, label_indices=label_indices)

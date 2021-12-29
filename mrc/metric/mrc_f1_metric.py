#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
mrc f1 metric

Authors: PanXu
Date:    2021/11/03 09:56:00
"""
from typing import Union, Dict, List, Tuple

import torch


from easytext.metrics import F1Metric


class MRCF1Metric(F1Metric):
    """
    MRC F1 Metric
    """

    All = "all"

    def __init__(self, labels: List[str]) -> None:
        super().__init__(labels)

        self._true_positives[MRCF1Metric.All] = 0
        self._false_positives[MRCF1Metric.All] = 0
        self._false_negatives[MRCF1Metric.All] = 0

    def __call__(self, prediction_match_labels: torch.Tensor, gold_match_labels: torch.Tensor, mask: torch.LongTensor) -> Dict:
        """
        真正的计算 metric f1。
        目前仅仅计算了全部的 F1, 也就是 f1-overall, 而对于每一个实体的指标并没有计算。
        原始的论文中没有这一部分，所以暂时没有计算每一种类型实体的指标。
        ToDo:// 增加每一种实体类型的指标计算
        :param prediction_match_labels: 预测的 match label indices
        :param gold_match_labels: golden label indices
        :param mask: mask
        :return: metric dict
        """
        mask = mask.bool()
        batch_size, seq_length = mask.size()

        match_label_mask = (mask.unsqueeze(-1).expand(-1, -1, seq_length)
                            & mask.unsqueeze(1).expand(-1, seq_length, -1))

        # 取上三角
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less or equal to end

        # 判断多少个是一样的
        prediction_match_labels = prediction_match_labels & match_label_mask
        gold_match_labels = gold_match_labels & match_label_mask

        true_positive_value = (gold_match_labels & prediction_match_labels).long().sum()
        true_positives = {MRCF1Metric.All: true_positive_value}
        false_positive_value = (~gold_match_labels & prediction_match_labels).long().sum()
        false_positives = {MRCF1Metric.All: false_positive_value}
        false_negative_value = (gold_match_labels & ~prediction_match_labels).long().sum()
        false_negatives = {MRCF1Metric.All: false_negative_value}

        self._true_positives[MRCF1Metric.All] += true_positive_value
        self._false_positives[MRCF1Metric.All] += false_positive_value
        self._false_negatives[MRCF1Metric.All] += false_negative_value

        return self._metric(true_positives=true_positives,
                            false_positives=false_positives,
                            false_negatives=false_negatives)


    def reset(self):
        """
        将所有的状态reset, f1 重新计算。
        """
        self._true_positives[MRCF1Metric.All] = 0
        self._false_positives[MRCF1Metric.All] = 0
        self._false_negatives[MRCF1Metric.All] = 0
        
        return self
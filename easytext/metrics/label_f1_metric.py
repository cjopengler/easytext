#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
对 label 的 f1 metric 计算

Authors: panxu(panxu@baidu.com)
Date:    2020/06/15 19:24:00
"""
from typing import Dict, Union, List
from collections import defaultdict
import torch

from .f1_metric import F1Metric

from easytext.data.vocabulary import Vocabulary


class LabelF1Metric(F1Metric):
    """
    对 label 的 f1 metric 计算. 适用的情况是, label shape: (B,), 每一个 label 是 [0, num_class) 的值。
    f1 的计算是针对 某个 class 的 f1。
    """

    def __init__(self, labels: List[Union[str, int]], label_vocabulary: Vocabulary):
        """
        对 label 的 f1 metric 计算. 适用的情况是, label shape: (B,), 每一个 label 是 [0, num_class) 的值。
        f1 的计算是针对 某个 class 的 f1。

        :param labels: 需要计算 f1 的标签序列. 如果类型是 str, 那么, 需要设置 label_vocabulary，用作
        label 和 index 之间的转换; 如果类型是 int, 那么, 不会使用 label_ vocabulary 进行转换 （可以设置为 None)，
        认为是 label index.
        :param label_vocabulary: label vocabulary 用来将 str label 转换成 label index
        """
        super().__init__()

        if isinstance(labels[0], str) and label_vocabulary is None:
            raise RuntimeError("labels 是 str 类型, label_vocabulary 不能是 None, "
                               "因为需要进行 label 到 index转换")

        self._label_vocabulary = label_vocabulary
        self._labels = labels

        if isinstance(labels[0], str):
            self._label_indices = [label_vocabulary.index(label) for label in self._labels]
        else:
            self._label_indices = labels

    def __call__(self,
                 prediction_labels: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: torch.LongTensor) -> Dict[str, float]:
        """
        计算 metric.

        :param prediction_labels: 预测结果 shape: (B,), 这是模型解码成label的结果，注意是 label，不是 logits
        :param gold_labels: 实际结果 shape: (B,)
        :param mask: mask, shape: (B,)
        :return 每一个label的f1值。这包括 precision, recall, f1。具体结果类似:

        {"precision_[label]": [value],
         "recall_[label]" : [value],
         "f1-measure_[label]": [value],
         "precision-overall": [value],
         "recall-overall": [value],
         "f1-measure-overall": [value]}

         说明: "*-overall" 表示的是所有 命中 在初始化参数的 labels 的综合 metric 值。
         这是有必要的，作为一个综合的值作为统一衡量。
        """

        assert prediction_labels.dim() == 1, "predictions shape 是 (B,)"
        assert gold_labels.dim() == 1, "gold_labels shape 是 (B,)"

        if mask is not None:
            assert mask.dim() == 1, "mask shape 是 (B,)"

        # 转换到 cpu 进行计算
        prediction_labels, gold_labels = prediction_labels.detach().cpu(), gold_labels.detach().cpu()

        if mask is not None:
            bool_mask = mask.detach().cpu().bool()

            prediction_labels = prediction_labels.masked_select(bool_mask)
            gold_labels = gold_labels.masked_select(bool_mask)

        # 当前 batch 下的 true_positives
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)

        for label, label_index in zip(self._labels, self._label_indices):
            num_prediction = (prediction_labels == label_index).sum().item()
            num_golden = (gold_labels == label_index).sum().item()

            # 计算 true positives
            label_mask = (prediction_labels == label_index)
            label_predictions = prediction_labels.masked_select(label_mask)
            label_gold = gold_labels.masked_select(label_mask)

            true_positives[label] = (label_predictions == label_gold).sum().item()
            false_positives[label] = num_prediction - true_positives[label]
            false_negatives[label] = num_golden - true_positives[label]

        return self._metric(true_positives=true_positives,
                            false_positives=false_positives,
                            false_negatives=false_negatives)


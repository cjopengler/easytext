#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2020/05/19 18:36:00
"""
from typing import Dict, Set
import torch
from collections import defaultdict

from easytext.data import LabelVocabulary
from easytext.utils import bio as BIO
from .metric import Metric
from .f1_metric import F1Metric


class SpanF1Metric(F1Metric):
    """
    计算基于 BIO, BIOUL 形式的 f1
    """

    def __init__(self,
                 label_vocabulary: LabelVocabulary) -> None:
        """
        初始化
        :param label_vocabulary: label 的 vocabulary
        """
        super().__init__()

        self.label_vocabulary = label_vocabulary

        # 下面之所以是字典，是为了计算多个 tag 的 f1, 比如: B-Per, B-Loc 就是需要需要计算 Per 以及 Loc 的 F1
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    @property
    def schema(self):
        return "BIO"

    def __call__(self,
                 prediction_labels: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: torch.Tensor) -> Dict:
        """
        计算 metric. 返回的是 F1 字典:

        {"precision_[tag]": [value],
         "recall_[tag]" : [value],
         "f1-measure_[tag]": [value],
         "precision-overall": [value],
         "recall-overall": [value],
         "f1-measure-overall": [value]}

         其中的 [tag] 是 span 的 tag, 也就是 "B-[tag]" 中的 "[tag]"

        :param prediction_labels: 预测的结果, shape: (B, SeqLen)
        :param gold_labels: 实际的结果, shape: (B, SeqLen)
        :param mask: 对 predictions 和 gold label 的 mask, shape: (B, SeqLen)
        :return: 当前的 metric 计算字典结果.
        """

        if prediction_labels.dim() != 2:
            raise RuntimeError(f"prediction_labels shape 应该是: (B, SeqLen), 现在是:{prediction_labels.size()}")
        if gold_labels.dim() != 2:
            raise RuntimeError(f"gold_labels shape 应该是: (B, SeqLen), 现在是:{gold_labels.size()}")

        if mask is not None:
            if mask.dim() != 2:
                raise RuntimeError(f"mask shape 应该是: (B, SeqLen), 现在是:{mask.size()}")



        # 转换到 cpu 进行计算
        prediction_labels, gold_labels = prediction_labels.detach().cpu(), gold_labels.detach().cpu()

        if mask is not None:
            mask = mask.detach().cpu()
        else:
            mask = torch.ones(size=(prediction_labels.size(0), prediction_labels.size(1)),
                              dtype=torch.long).cpu()

        assert prediction_labels.size() == gold_labels.size(), \
            f"prediction_labels.size: {prediction_labels.size()} 与 gold_labels.size: {gold_labels.size()} 不匹配!"

        assert prediction_labels.size() == mask.size(), \
            f"prediction_labels.size: {prediction_labels.size()} 与 mask.size: {mask.size()} 不匹配!"

        bool_mask = (mask != 0)

        num_classes = self.label_vocabulary.label_size

        if (torch.masked_select(gold_labels, bool_mask) >= num_classes).any():
            raise RuntimeError(f"gold_labels 中存在比 num_classes 大的数值")

        # 将预测的结果 decode 成 span list
        prediction_spans_list = BIO.decode_label_index_to_span(batch_sequence_label_index=prediction_labels,
                                                               mask=mask,
                                                               vocabulary=self.label_vocabulary)

        # 将gold label index decode 成 span  list
        gold_spans_list = BIO.decode_label_index_to_span(batch_sequence_label_index=gold_labels,
                                                         mask=mask,
                                                         vocabulary=self.label_vocabulary)

        # 预测的 每个 label 的 span 数量字典
        num_prediction = defaultdict(int)

        # golden 每一个 label 的 span
        num_golden = defaultdict(int)

        # 当前 batch 下的 true_positives
        true_positives = defaultdict(int)
        false_positives = defaultdict(int)
        false_negatives = defaultdict(int)

        for prediction_spans, gold_spans in zip(prediction_spans_list, gold_spans_list):
            intersection = BIO.span_intersection(span_list1=prediction_spans,
                                                 span_list2=gold_spans)

            for span in intersection:
                # self._true_positives[span["label"]] += 1
                true_positives[span["label"]] += 1

            for span in prediction_spans:
                num_prediction[span["label"]] += 1

            for span in gold_spans:
                num_golden[span["label"]] += 1

        for label, num in num_prediction.items():
            false_positives[label] = num - true_positives[label]

        for label, num in num_golden.items():
            false_negatives[label] = num - true_positives[label]

        for k, v in true_positives.items():
            self._true_positives[k] += v

        for k, v in false_positives.items():
            self._false_positives[k] += v

        for k, v in false_negatives.items():
            self._false_negatives[k] += v

        return self._metric(true_positives=true_positives,
                            false_positives=false_positives,
                            false_negatives=false_negatives)


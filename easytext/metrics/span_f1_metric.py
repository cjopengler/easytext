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


class SpanF1Metric(Metric):
    """
    计算基于 BIO, BIOUL 形式的 f1
    """

    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"

    PRECISION_OVERALL = f"{PRECISION}-overall"
    RECALL_OVERALL = f"{RECALL}-overall"
    F1_OVERALL = f"{F1}-overall"

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
                 predictions: torch.Tensor,
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

        :param predictions: 预测的结果, shape: (B, SeqLen, num_label)
        :param gold_labels: 实际的结果, shape: (B, SeqLen)
        :param mask: 对 predictions 和 gold label 的 mask, shape: (B, SeqLen)
        :return: 当前的 metric 计算字典结果.
        """

        if predictions.dim() != 3:
            raise RuntimeError(f"predictions shape 应该是: (B, SeqLen, num_label), 现在是:{predictions.size()}")
        if gold_labels.dim() != 2:
            raise RuntimeError(f"gold_labels shape 应该是: (B, SeqLen), 现在是:{gold_labels.size()}")

        if mask is not None:
            if mask.dim() != 2:
                raise RuntimeError(f"mask shape 应该是: (B, SeqLen), 现在是:{mask.size()}")

        # 转换到 cpu 进行计算
        predictions, gold_labels = predictions.detach().cpu(), gold_labels.detach().cpu()

        if mask is not None:
            mask = mask.detach().cpu()
        else:
            mask = torch.ones(size=(predictions.size(0), predictions.size(1)),
                              dtype=torch.long).cpu()

        bool_mask = (mask != 0)

        num_classes = predictions.size(-1)

        if (torch.masked_select(gold_labels, bool_mask) >= num_classes).any():
            raise RuntimeError(f"gold_labels 中存在比 num_classes 大的数值")

        # 将预测的结果 decode 成 span list
        prediction_spans_list = BIO.decode(batch_sequence_logits=predictions,
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

    def _metric(self,
                true_positives: Dict[str, int],
                false_positives: Dict[str, int],
                false_negatives: Dict[str, int]) -> Dict:
        """
        计算 metric, 注意是 输入是的字典
        :param true_positives:
        :param false_positives:
        :param false_negatives:
        :return:
        """

        all_tags: Set[str] = set()
        all_tags.update(true_positives.keys())
        all_tags.update(false_positives.keys())
        all_tags.update(false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                                                  self._false_positives[tag],
                                                                  self._false_negatives[tag])
            precision_key = SpanF1Metric.PRECISION + "-" + tag
            recall_key = SpanF1Metric.RECALL + "-" + tag
            f1_key = SpanF1Metric.F1 + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values()),
                                                              sum(self._false_negatives.values()))
        all_metrics[SpanF1Metric.PRECISION_OVERALL] = precision
        all_metrics[SpanF1Metric.RECALL_OVERALL] = recall
        all_metrics[SpanF1Metric.F1_OVERALL] = f1_measure

        return all_metrics

    @property
    def metric(self) -> Dict:
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        return self._metric(true_positives=self._true_positives,
                            false_positives=self._false_positives,
                            false_negatives=self._false_negatives)

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
        return self

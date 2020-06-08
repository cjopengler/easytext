#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
event detection without trigger

Authors: panxu(panxu@baidu.com)
Date:    2020/01/31 09:11:00
"""
import json
import logging
from typing import Dict

import torch
from torch import LongTensor
from torch import Tensor
from torch.nn.modules.loss import MSELoss
import torch.nn.functional as F

from allennlp.data import Vocabulary

from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders import TokenEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.regularizers import RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import get_final_encoder_states
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import BooleanAccuracy, Metric

from zznlp.models.event_detection_without_tirgger.types_define import NEGATIVE_EVENT_TYPE
from zznlp.models.event_detection_without_tirgger.metrics import EventDetectionWithoutKeywordF1Measure
from zznlp.models.event_detection_without_tirgger.dataset_readers import EventDetectionWithoutTriggerDatasetReader


@Model.register("EventDetectionWithoutTriggerModel")
class EventDetectionWithoutTriggerModel(Model):
    """
    event detection without trigger

    ACL 2019 reference: https://www.aclweb.org/anthology/N19-1080/
    """

    def __init__(self,
                 sentence_embedder: TextFieldEmbedder,
                 entity_tag_embedder: TextFieldEmbedder,
                 event_type_embedder_1: TokenEmbedder,
                 event_type_embedder_2: TokenEmbedder,
                 sentence_encoder: Seq2SeqEncoder,
                 alpha: float,
                 vocab: Vocabulary,
                 activate_score: bool = True,
                 initializer: InitializerApplicator = None,
                 regularizer: RegularizerApplicator = None):
        super().__init__(vocab=vocab, regularizer=regularizer)

        self._sentence_embedder = sentence_embedder
        self._entity_tag_embedder = entity_tag_embedder

        # event_type_embedder 因为要做attention，所以维度是 sentence_embedder + entity_tag_embedder
        self._event_type_embedder_1 = event_type_embedder_1

        self._event_type_embedder_2 = event_type_embedder_2

        self._sentence_encoder = sentence_encoder

        self._alpha = alpha

        self._loss = MSELoss()

        self._metrics: Dict[str, Metric] = {"accuracy": BooleanAccuracy()}
        self._f1_metrics: Dict[str, EventDetectionWithoutKeywordF1Measure] = \
            {"f1_all": EventDetectionWithoutKeywordF1Measure("all")}

        self._activate_score = activate_score
        # for event_type in vocab.get_token_to_index_vocabulary(
        #         EventDetectionWithoutTriggerDatasetReader.EVENT_TYPE_NAMESPACE):
        #     logging.debug(f"event type: {event_type}")
        #     self._f1_metrics[f"f1_{event_type}"] = EventDetectionWithoutKeywordF1Measure(event_type)

        if initializer:
            initializer(self)

        # debug
        for name, parameter in self.named_parameters():
            print(f"parameter: {name}: {parameter}")

    def forward(self,
                sentence: Dict[str, LongTensor],
                entity_tag: Dict[str, LongTensor],
                event_type: LongTensor,
                metadata: Dict = None,
                label: LongTensor = None) -> Dict[str, torch.Tensor]:
        output_dict = dict()

        # sentence, entity_tag 使用的是同一个 mask
        mask = get_text_field_mask(sentence).float()

        # shape: B * SeqLen * InputSize1
        sentence_embedding = self._sentence_embedder(sentence)

        # shape: B * SeqLen * InputSize2
        entity_tag_embedding = self._entity_tag_embedder(entity_tag)

        # shape: B * SeqLen * InputSize, InputSize = InputSize1 + InputSize2
        sentence_embedding = torch.cat((sentence_embedding, entity_tag_embedding),
                                       dim=-1)
        # shape: B * SeqLen * InputSize
        sentence_encoding: Tensor = self._sentence_encoder(sentence_embedding, mask=mask)

        # shape: B * InputSize
        event_type_embedding_1: Tensor = self._event_type_embedder_1(event_type)

        # attention
        # shape: B * InputSize * 1
        event_type_embedding_1_tmp = event_type_embedding_1.unsqueeze(-1)

        # shape: (B * SeqLen * InputSize) bmm (B * InputSize * 1) = B * SeqLen * 1
        attention_logits = sentence_encoding.bmm(event_type_embedding_1_tmp)

        # shape: B * SeqLen
        attention_logits = attention_logits.squeeze(-1)

        # Shape: B * SeqLen
        tmp_attention_logits = torch.exp(attention_logits) * mask

        # Shape: B * Seqlen
        tmp_attenttion_logits_sum = torch.sum(tmp_attention_logits, dim=-1, keepdim=True)

        # Shape: B * SeqLen
        attention = tmp_attention_logits / tmp_attenttion_logits_sum

        # Score1 计算, Shape: B * 1
        score1 = torch.sum(attention_logits * attention, dim=-1, keepdim=True)

        # global score

        # 获取最后一个hidden, shape: B * INPUT_SIZE
        hidden_last = get_final_encoder_states(encoder_outputs=sentence_encoding,
                                               mask=mask,
                                               bidirectional=self._sentence_encoder.is_bidirectional())
        # event type 2, shape: B * INPUT_SIZE
        event_type_embedding_2: Tensor = self._event_type_embedder_2(event_type)

        # shape: B * INPUT_SIZE
        tmp = hidden_last * event_type_embedding_2

        # shape: B * 1
        score2 = torch.sum(tmp, dim=-1, keepdim=True)

        # 最终的score, B * 1
        score = score1 * self._alpha + score2 * (1 - self._alpha)
        if self._activate_score:  # 使用 sigmoid函数激活
            score = torch.sigmoid(score)

        # Shape: B
        y_pred = torch.gt(score.squeeze(-1), 0.5).long()

        output_dict["label"] = y_pred
        output_dict["score"] = score
        output_dict["metadata"] = metadata

        if label is not None:
            # 计算loss, 注意，这里的loss，后续 follow paper 要修改成带有 beta 的loss.
            loss_ok = self._loss(score.squeeze(-1), label.float())
            loss_mse_ok = F.mse_loss(score.squeeze(-1), label.float())

            # 下面的代码 因为维度不一致，会导致无法收敛, 这个问题需要查看
            loss_no = self._loss(score, label.float())
            loss_mse_no = F.mse_loss(score, label.float())

            loss = loss_ok

            logging.info(f"score: {score}\nlabel: {label}\n")
            logging.info(f"\nloss_ok: {loss_ok}, loss_mse_ok: {loss_mse_ok}\n"
                         f"loss_no: {loss_no}, loss_mse_no: {loss_mse_no}\n")

            output_dict["loss"] = loss

            for _, metric in self._metrics.items():
                metric(y_pred, label)

            y_pred_list = y_pred.tolist()
            label_list = label.tolist()
            event_type_list = event_type.tolist()

            logging.debug("-" * 80)
            for yy, ll, ee in zip(y_pred_list, label_list, event_type_list):
                if ll == 1:
                    ee = self.vocab.get_token_from_index(ee,
                                                         namespace=EventDetectionWithoutTriggerDatasetReader.EVENT_TYPE_NAMESPACE)
                    logging.debug(f"{ee}:[{yy},{ll}]")
            logging.debug("+" * 80)

            # 计算 f1 metric
            for _, f1_metric in self._f1_metrics.items():
                f1_mask = self.mask_for_f1(f1_metric.event_type, event_type)
                f1_metric(y_pred, label, f1_mask)

        return output_dict

    def mask_for_f1(self, target_event_type: str, event_types: LongTensor) -> LongTensor:
        """
        通过 target event type 来计算 mask
        :param target_event_type:
        :param event_types:
        :return: 某个 target event type的mask
        """
        if target_event_type == "all":
            negative_event_index = self.vocab.get_token_index(
                NEGATIVE_EVENT_TYPE,
                EventDetectionWithoutTriggerDatasetReader.EVENT_TYPE_NAMESPACE)

            negative = torch.full_like(event_types, negative_event_index)

            # 不是 negative 的保留, negative mask 成 0
            mask = torch.ne(negative, event_types)
        else:
            target_event_index = self.vocab.get_token_index(
                target_event_type,
                EventDetectionWithoutTriggerDatasetReader.EVENT_TYPE_NAMESPACE)
            target_event = torch.full_like(event_types, target_event_index)

            # target event mask, 与 target event type 一致的保留
            mask = torch.eq(target_event, event_types)
        return mask.long()

    def _f1_metric(self,
                   target_event_type: str,
                   predictions: LongTensor,
                   golden_labels: LongTensor,
                   event_types: LongTensor):
        """
        计算 target event type的f1
        :param target_event_type: 某个event type.  target_event_type="all"，说明是计算全部的f1.
        :param predictions: 预测结果
        :param golden_labels: golden labels
        :param event_types: 事件类型tensor
        :return:
        """
        mask = self.mask_for_f1(target_event_type=target_event_type,
                                event_types=event_types)

        f1_measure = self._f1_metrics[target_event_type]

        return f1_measure(predictions=predictions, gold_labels=golden_labels, mask=mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        获取metrics结果
        :param reset:
        :return:
        """

        metrics = {name: metric.get_metric(reset) for name, metric in self._metrics.items()}

        for name, f1_metric in self._f1_metrics.items():
            f1_value_dict: Dict = f1_metric.get_metric(reset)
            metrics.update(f1_value_dict)
        return metrics

    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """解码"""
        return output_dict

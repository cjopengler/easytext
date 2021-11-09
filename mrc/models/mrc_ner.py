#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
mrc ner model

Authors: PanXu
Date:    2021/10/27 11:22:00
"""

from typing import Dict
import torch
from torch.nn import Module
from torch.nn import Linear
from torch.nn import Dropout, GELU
from torch.nn import Embedding, LayerNorm

from transformers import BertModel, BertConfig, BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from easytext.utils.nn.bert_init_weights import BertInitWeights
from easytext.utils.seed_util import set_seed

from mrc.models import MRCNerOutput


class MultiNonLinearClassifier(Module):
    def __init__(self, hidden_size, num_label, dropout):
        super(MultiNonLinearClassifier, self).__init__()
        self.num_label = num_label
        self.classifier1 = Linear(hidden_size, hidden_size)
        self.classifier2 = Linear(hidden_size, num_label)
        self.dropout = Dropout(dropout)
        self.activate = GELU()

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        # features_output1 = F.relu(features_output1)
        features_output1 = self.activate(features_output1)
        features_output1 = self.dropout(features_output1)
        features_output2 = self.classifier2(features_output1)
        return features_output2


class MRCNer(Module):
    """
    基于 MRC 的 ner 模型
    """

    def __init__(self, bert_dir: str, dropout: float):

        super().__init__()
        self.bert = BertModel.from_pretrained(bert_dir)
        bert_config = self.bert.config

        self.start_classifier = Linear(bert_config.hidden_size, 1)
        self.end_classifier = Linear(bert_config.hidden_size, 1)
        self.match_classifier = MultiNonLinearClassifier(bert_config.hidden_size * 2, 1, dropout)

        self.init_weights = BertInitWeights(bert_config=bert_config)
        self.reset_parameters()

    def reset_parameters(self):
        self.start_classifier.apply(self.init_weights)
        self.end_classifier.apply(self.init_weights)
        self.match_classifier.apply(self.init_weights)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor,
                sequence_mask: torch.Tensor,
                metadata: Dict) -> MRCNerOutput:
        """
        模型前向计算
        :param input_ids:
        :param attention_mask:
        :param token_type_ids:
        :param sequence_mask:
        :param metadata:
        :return:
        """
        bert_output: BaseModelOutputWithPooling = self.bert(input_ids=input_ids,
                                                            attention_mask=attention_mask,
                                                            token_type_ids=token_type_ids,
                                                            return_dict=True)

        sequence_output = bert_output["last_hidden_state"]
        sequence_length = sequence_output.size(1)

        start_logits = self.start_classifier(sequence_output)
        # 最后一个维度去掉 (B, seq_len)
        start_logits = start_logits.squeeze(-1)

        assert len(start_logits.size()) == 2

        end_logits = self.end_classifier(sequence_output)

        # (B, seq_len)
        end_logits = end_logits.squeeze(-1)

        assert len(end_logits.size()) == 2

        # 将每一个 i 与 j 连接在一起， 所以是 N*N的拼接，使用了 expand, 进行 两个方向的扩展
        # 产生一个 match matrix
        # 对于每一个 i 都与 j concat 在一起
        # [B, seq_len, seq_len, hidden]
        start_extend = sequence_output.unsqueeze(2).expand(-1, -1, sequence_length, -1)

        # [B, seq_len, seq_len, hidden]
        end_extend = sequence_output.unsqueeze(1).expand(-1, sequence_length, -1, -1)

        # [B, seq_len, seq_len, hidden*2]
        match_matrix = torch.cat([start_extend, end_extend], 3)

        # (B, seq_len, seq_len)
        match_logits = self.match_classifier(match_matrix).squeeze(-1)

        assert len(match_logits.size()) == 3

        return MRCNerOutput(start_logits=start_logits,
                            end_logits=end_logits,
                            match_logits=match_logits,
                            mask=sequence_mask)



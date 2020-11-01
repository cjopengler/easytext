#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
rnn with crf

Authors: PanXu
Date:    2020/10/29 16:17:00
"""

import logging
from typing import Union, Dict

import torch
from torch import Tensor
from torch.nn import LSTM, GRU, Linear, Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from easytext.modules.seq2seq import RnnSeq2Seq
from easytext.modules import DynamicRnn
from easytext.modules import ConditionalRandomField
from easytext.model import Model, ModelOutputs
from easytext.data import Vocabulary, PretrainedVocabulary, LabelVocabulary
from easytext.utils import bio as BIO
from easytext.component.register import ComponentRegister

from ner.config.vocabulary_builder import VocabularyBuilder
from ner.models.ner_model_outputs import NerModelOutputs


@ComponentRegister.register_class(name="RnnWithCrf", name_space="model")
class RnnWithCrf(Model):
    """
    rnn + crf
    """

    def __init__(self,
                 vocabulary_builder: VocabularyBuilder,
                 word_embedding_dim: int,
                 rnn_type: str,
                 hidden_size: int,
                 num_layer: int,
                 dropout: float,
                 is_used_crf: bool):

        super().__init__()

        self.word_embedding_dim = word_embedding_dim
        self.token_vocabulary = vocabulary_builder.token_vocabulary
        self.label_vocabulary = vocabulary_builder.label_vocabulary
        self.is_used_crf = is_used_crf

        if isinstance(self.token_vocabulary, Vocabulary):
            self.embedding: Embedding = Embedding(num_embeddings=self.token_vocabulary.size,
                                                  embedding_dim=word_embedding_dim,
                                                  padding_idx=self.token_vocabulary.padding_index)

        elif isinstance(self.token_vocabulary, PretrainedVocabulary):
            self.embedding: Embedding = Embedding.from_pretrained(self.token_vocabulary.embedding_matrix,
                                                                  freeze=True,
                                                                  padding_idx=self.token_vocabulary.padding_index)

        self.hidden_size = hidden_size

        if rnn_type == DynamicRnn.LSTM:

            lstm = LSTM(input_size=word_embedding_dim,
                        hidden_size=hidden_size,
                        num_layers=num_layer,
                        bidirectional=True,
                        dropout=dropout)
            dynamic_rnn = DynamicRnn(rnn=lstm)
        elif rnn_type == DynamicRnn.GRU:
            gru = GRU(input_size=word_embedding_dim,
                      hidden_size=hidden_size,
                      num_layers=num_layer,
                      bidirectional=True,
                      dropout=dropout)
            dynamic_rnn = DynamicRnn(rnn=gru)
        else:
            raise RuntimeError(f"rnn_type: {rnn_type} 必须是 {DynamicRnn.LSTM} 或 {DynamicRnn.GRU} ")

        self.rnn_seq2seq = RnnSeq2Seq(dynamic_rnn=dynamic_rnn)

        self.liner = Linear(in_features=hidden_size * 2,
                            out_features=label_vocabulary.label_size)

        if self.is_used_crf:
            constraints = BIO.allowed_transitions(label_vocabulary=label_vocabulary)
            self.crf = ConditionalRandomField(num_tags=label_vocabulary.label_size,
                                              constraints=constraints)
        else:
            self.crf = None

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, tokens: Tensor, metadata: Dict) -> NerModelOutputs:
        """
        模型运行
        :param tokens: token 序列, Shape: (batch_size, seq_len)
        :return: NerModelOutputs
        """

        assert tokens.dim() == 2, f"tokens shape: {tokens.dim()} 与 (batch_size, seq_len) 不匹配"

        mask = (tokens != self.token_vocabulary.padding_index)

        token_embedding = self.embedding(tokens)

        encoding = self.rnn_seq2seq(token_embedding, mask)

        logits = self.liner(encoding)

        model_outputs = NerModelOutputs(logits=logits,
                                        mask=mask,
                                        crf=self.crf)

        return model_outputs

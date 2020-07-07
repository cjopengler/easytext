#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
使用 bilstm + crf

Authors: PanXu
Date:    2020/07/04 19:51:00
"""

from easytext.modules import ConditionalRandomField


import logging
from typing import Union, Dict

import torch
from torch import Tensor
from torch.nn import LSTM, Linear, Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from easytext.model import Model, ModelOutputs
from easytext.data import Vocabulary, PretrainedVocabulary, LabelVocabulary
from easytext.modules import ConditionalRandomField
from easytext.utils import bio as BIO

from .ner_model_outputs import NerModelOutputs


class NerV3(Model):
    """
    Ner v3 版本，glove6B.100d + bilstm + crf
    """

    DESTCRIPTION = "glove6B.100d + bilstm + crf"

    def __init__(self,
                 token_vocabulary: Union[Vocabulary, PretrainedVocabulary],
                 label_vocabulary: LabelVocabulary,
                 word_embedding_dim: int,
                 hidden_size: int,
                 num_layer: int,
                 dropout: float):

        super().__init__()

        self.word_embedding_dim = word_embedding_dim
        self.token_vocabulary = token_vocabulary
        self.label_vocabulary = label_vocabulary

        if isinstance(token_vocabulary, Vocabulary):
            self.embedding: Embedding = Embedding(num_embeddings=token_vocabulary.size,
                                                  embedding_dim=word_embedding_dim,
                                                  padding_idx=token_vocabulary.padding_index)

        elif isinstance(token_vocabulary, PretrainedVocabulary):
            self.embedding: Embedding = Embedding.from_pretrained(token_vocabulary.embedding_matrix,
                                                                  freeze=True,
                                                                  padding_idx=token_vocabulary.padding_index)

        self.hidden_size = hidden_size

        self.lstm = LSTM(input_size=word_embedding_dim,
                         hidden_size=hidden_size,
                         num_layers=num_layer,
                         bidirectional=True,
                         dropout=dropout)

        self.liner = Linear(in_features=hidden_size * 2,
                            out_features=label_vocabulary.label_size)

        constraints = BIO.allowed_transitions(label_vocabulary=label_vocabulary)
        self.crf = ConditionalRandomField(num_tags=label_vocabulary.label_size,
                                          constraints=constraints)

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

        batch_size = tokens.size(0)
        seq_len = tokens.size(1)

        mask = (tokens != self.token_vocabulary.padding_index)
        mask_long = mask.long()

        sequence_lengths = mask_long.sum(dim=-1)

        token_embedding = self.embedding(tokens)

        assert (batch_size, seq_len, self.word_embedding_dim) == token_embedding.size()

        pack = pack_padded_sequence(token_embedding,
                                    lengths=sequence_lengths,
                                    batch_first=True,
                                    enforce_sorted=False)

        packed_seqence_encoding, _ = self.lstm(pack)

        encoding, pad_seqence_length = pad_packed_sequence(packed_seqence_encoding,
                                                           batch_first=True,
                                                           padding_value=0.0)

        # 校验 length 一致性
        # 转换到相应的device
        pad_seqence_length = pad_seqence_length.to(sequence_lengths.device)
        assert (sequence_lengths == pad_seqence_length).long().sum() == batch_size

        assert (batch_size, seq_len, 2 * self.hidden_size) == encoding.size()

        logits = self.liner(encoding)

        assert (batch_size, seq_len, self.label_vocabulary.label_size) == logits.size()

        model_outputs = NerModelOutputs(logits=logits,
                                        mask=mask_long,
                                        crf=self.crf)

        return model_outputs

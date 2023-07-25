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
from torch.nn import Embedding
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from easytext.data import Vocabulary, LabelVocabulary, PretrainedVocabulary
from easytext.model import Model
from easytext.model import ModelOutputs
from easytext.utils.nn import nn_util


class EventModelOutputs(ModelOutputs):
    """
    Event Model 的输出数据
    """

    def __init__(self, logits: torch.Tensor, event_type: torch.LongTensor):
        super().__init__(logits)
        self.event_type = event_type


class EventModel(Model):
    """
    event detection without trigger

    ACL 2019 reference: https://www.aclweb.org/anthology/N19-1080/
    """

    def __init__(self,
                 alpha: float,
                 activate_score: bool,
                 sentence_vocab: PretrainedVocabulary,
                 sentence_embedding_dim: int,
                 entity_tag_vocab: Vocabulary,
                 entity_tag_embedding_dim: int,
                 event_type_vocab: Vocabulary,
                 event_type_embedding_dim: int,
                 lstm_hidden_size: int,
                 lstm_encoder_num_layer: int,
                 lstm_encoder_droupout: float):
        """
        初始化
        * 注意这里存在的约束: entity_tag_embedding_dim 与 lstm_hidden_size 要相等，因为要进行 attention 操作。

        :param alpha: 论文中 alpha 参数， 对两个 score 混合的参数
        :param sentence_vocab: 句子中 字/词 的词典, 也可以叫做 word vocabulary
        :param sentence_embedding_dim: sentence 中 word embedding dim.
        设置成300， 这样容易与主流词向量维度对齐。
        :param entity_tag_vocab: 实体标签词汇表
        :param entity_tag_embedding_dim: 实体标签的 embedding dim
        :param event_type_vocab: 事件类型词汇表
        :param event_type_embedding_dim: 事件类型 dim
        :param lstm_hidden_size: lstm 的隐层维度
        :param lstm_encoder_num_layer: lstm layer 数量
        :param lstm_encoder_droupout: lstm droupout 参数
        """

        super().__init__()

        assert event_type_embedding_dim == lstm_hidden_size, \
            f"event_type_embedding_dim 与lstm_hidden_size 不相等, " \
            f"当前: event_type_embedding_dim={event_type_embedding_dim} " \
            f"lstm_hidden_size={lstm_hidden_size}"

        self._alpha = alpha
        self._activate_score = activate_score
        self._sentence_vocab = sentence_vocab
        self._sentence_embedding_dim = sentence_embedding_dim

        if isinstance(self._sentence_vocab, Vocabulary):
            self._sentence_embedder = Embedding(self._sentence_vocab.size,
                                                embedding_dim=sentence_embedding_dim,
                                                padding_idx=self._sentence_vocab.padding_index)

        elif isinstance(self._sentence_vocab, PretrainedVocabulary):
            self._sentence_embedder = Embedding.from_pretrained(
                embeddings=sentence_vocab.embedding_matrix
            )
        else:
            raise RuntimeError(f"sentence_vocab 类型: {type(self._sentence_vocab)} 不是 Vocabulary 或者 PretrainedVocabulary")

        self._entity_tag_embedding_dim = entity_tag_embedding_dim
        self._entity_tag_vocab = entity_tag_vocab
        self._entity_tag_embedder = Embedding(self._entity_tag_vocab.size,
                                              embedding_dim=entity_tag_embedding_dim,
                                              padding_idx=self._entity_tag_vocab.padding_index)

        self._event_type_embedding_dim = event_type_embedding_dim
        self._event_type_vocab = event_type_vocab
        self._event_type_embedder_1 = Embedding(self._event_type_vocab.size,
                                                embedding_dim=event_type_embedding_dim)
        self._event_type_embedder_2 = Embedding(self._event_type_vocab.size,
                                                embedding_dim=event_type_embedding_dim)

        self._lstm_hidden_size = lstm_hidden_size
        # lstm 作为encoder
        self._lstm = LSTM(input_size=(sentence_embedding_dim + entity_tag_embedding_dim),
                          hidden_size=lstm_hidden_size,
                          num_layers=lstm_encoder_num_layer,
                          batch_first=True,
                          dropout=lstm_encoder_droupout,
                          bidirectional=False)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self,
                sentence: LongTensor,
                entity_tag: LongTensor,
                event_type: LongTensor,
                metadata: Dict = None) -> EventModelOutputs:
        """
        模型运行
        :param sentence: shape: (B, SeqLen), 句子的 index tensor
        :param entity_tag: shape: (B, SeqLen), 句子的 实体 index tensor
        :param event_type: shape: (B,), event type 的 tensor
        :param metadata: metadata 数据，不参与模型运算
        """

        assert sentence.dim() == 2, f"Sentence 的维度 {sentence.dim()} !=2, 应该是(B, seq_len)"
        assert entity_tag.dim() == 2, f"entity_tag 维度 {entity_tag.dim()} != 2, 应该是 (B, seq_len)"
        assert event_type.dim() == 1, f"event_type 维度 {event_type.dim()} != 1, 应该是 (B,)"

        batch_size = sentence.size(0)
        seq_len = sentence.size(1)

        # sentence, entity_tag 使用的是同一个 mask
        mask = nn_util.sequence_mask(sentence,
                                     self._sentence_vocab.index(self._sentence_vocab.padding))
        assert mask.shape == (batch_size, seq_len), f"mask 维度是: (B, seq_len)"

        # shape: B * SeqLen * sentence_embedding_dim
        sentence_embedding = self._sentence_embedder(sentence)

        assert sentence_embedding.shape == (batch_size, seq_len, self._sentence_embedding_dim)

        # shape: B * SeqLen * entity_tag_embedding_dim
        entity_tag_embedding = self._entity_tag_embedder(entity_tag)

        assert entity_tag_embedding.shape == (batch_size, seq_len, self._entity_tag_embedding_dim)

        # shape: B * SeqLen * InputSize, InputSize = sentence_embedding_dim + entity_tag_embedding_dim
        sentence_embedding = torch.cat((sentence_embedding, entity_tag_embedding),
                                       dim=-1)
        assert sentence_embedding.shape, (batch_size,
                                          seq_len,
                                          self._sentence_embedding_dim + self._entity_tag_embedding_dim)
        # 使用 mask 计算 sentence 实际长度, shape: (B,)
        sentence_length = mask.long().sum(dim=-1)

        assert sentence_length.shape == (batch_size,)

        # 使用 lstm sequence encoder 进行 encoder
        packed_sentence_embedding = pack_padded_sequence(input=sentence_embedding,
                                                         lengths=sentence_length,
                                                         batch_first=True,
                                                         enforce_sorted=False)

        packed_sequence, (h_n, c_n) = self._lstm(packed_sentence_embedding)

        # Tuple, sentence: shape: B * SeqLen * InputSize 和 sentence length
        (sentence_encoding, _) = pad_packed_sequence(packed_sequence, batch_first=True)

        assert sentence_encoding.shape == (batch_size, seq_len, self._lstm_hidden_size)

        # shape: B * InputSize
        event_type_embedding_1: Tensor = self._event_type_embedder_1(event_type)
        assert event_type_embedding_1.shape == (batch_size, self._event_type_embedding_dim)

        # attention
        # shape: B * InputSize * 1
        event_type_embedding_1_tmp = event_type_embedding_1.unsqueeze(-1)
        assert event_type_embedding_1_tmp.shape == (batch_size, self._event_type_embedding_dim, 1)

        # shape: (B * SeqLen * InputSize) bmm (B * InputSize * 1) = B * SeqLen * 1
        attention_logits = sentence_encoding.bmm(event_type_embedding_1_tmp)

        # shape: B * SeqLen
        attention_logits = attention_logits.squeeze(-1)

        assert attention_logits.shape == (batch_size, seq_len)

        # Shape: B * SeqLen
        tmp_attention_logits = torch.exp(attention_logits) * mask.float()

        # Shape: B * Seqlen
        tmp_attenttion_logits_sum = torch.sum(tmp_attention_logits, dim=-1, keepdim=True)

        # Shape: B * SeqLen
        attention = tmp_attention_logits / tmp_attenttion_logits_sum

        assert attention.shape == (batch_size, seq_len)

        # Score1 计算, Shape: B * 1
        score1 = torch.sum(attention_logits * attention, dim=-1, keepdim=True)

        assert score1.shape == (batch_size, 1)

        score1 = score1.squeeze(dim=-1)

        # global score

        # 获取最后一个hidden, shape: B * INPUT_SIZE
        hidden_last = h_n.squeeze(dim=0)
        assert hidden_last.shape == (batch_size, self._lstm_hidden_size)

        # event type 2, shape: B * INPUT_SIZE
        event_type_embedding_2: Tensor = self._event_type_embedder_2(event_type)

        assert event_type_embedding_2.shape == (batch_size, self._event_type_embedding_dim)

        # shape: B * INPUT_SIZE
        tmp = hidden_last * event_type_embedding_2

        # shape: B * 1
        score2 = torch.sum(tmp, dim=-1, keepdim=True)

        assert score2.shape == (batch_size, 1)

        score2 = score2.squeeze(dim=-1)

        # 最终的score, B
        score = score1 * self._alpha + score2 * (1 - self._alpha)
        assert score.shape == (batch_size,)

        if self._activate_score:  # 使用 sigmoid函数激活
            score = torch.sigmoid(score)

        return EventModelOutputs(logits=score, event_type=event_type)

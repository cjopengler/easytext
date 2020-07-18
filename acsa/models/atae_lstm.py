#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
AE LSTM

Authors: PanXu
Date:    2020/07/15 08:41:00
"""

from typing import Union, List, Dict

import torch
from torch import LongTensor, BoolTensor
from torch.nn import Embedding, LSTM, Linear
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from easytext.model import Model
from easytext.model import ModelOutputs
from easytext.data import Vocabulary, PretrainedVocabulary, LabelVocabulary
from easytext.utils.nn.nn_util import sequence_mask
from easytext.modules.seq2vec import AttentionSeq2Vec

from .acsa_model_outputs import ACSAModelOutputs


class ATAELstm(Model):
    """
    基于 ATAE-LSTM 模型
    相关论文: 2016-emnlp-Attention-based LSTM for Aspect-level Sentiment Classification
    在 docs/docs/acsa/相关文章及论文/2016-emnlp-Attention-based LSTM for Aspect-level Sentiment Classification.pdf
    """

    def __init__(self,
                 token_vocabulary: Union[Vocabulary, PretrainedVocabulary],
                 token_embedding_dim: int,
                 category_vocabulary: LabelVocabulary,
                 category_embedding_dim: int,
                 label_vocabulary: LabelVocabulary
                 ):
        super().__init__()

        self._token_vocabulary = token_vocabulary

        if isinstance(self._token_vocabulary, Vocabulary):
            self.token_embedding = Embedding(num_embeddings=self._token_vocabulary.size,
                                             embedding_dim=token_embedding_dim,
                                             padding_idx=self._token_vocabulary.padding_index)
        elif isinstance(self._token_vocabulary, PretrainedVocabulary):
            self.token_embedding = Embedding.from_pretrained(
                embeddings=self._token_vocabulary.embedding_matrix,
                padding_idx=self._token_vocabulary.padding_index,
                freeze=False
            )
        else:
            raise RuntimeError(
                f"token_vocabulary type: {type(token_vocabulary)} 不是 Vocabulary 或 PretrainedVocabulary")

        self._category_vocabulary = category_vocabulary
        self.category_embedding = Embedding(num_embeddings=self._category_vocabulary.label_size,
                                            embedding_dim=category_embedding_dim,
                                            padding_idx=self._category_vocabulary.padding_index)

        lstm_hidden_size = token_embedding_dim
        lstm_input_size = token_embedding_dim + category_embedding_dim
        self.lstm = LSTM(input_size=lstm_input_size,
                         hidden_size=lstm_hidden_size,
                         num_layers=1,
                         bidirectional=False,
                         batch_first=True,
                         dropout=0.4)

        attention_input_size = (category_embedding_dim + lstm_hidden_size)
        attetion_value_hidden_size = None
        self.attention_seq2vec = AttentionSeq2Vec(input_size=attention_input_size,
                                                  query_hidden_size=lstm_input_size,
                                                  value_hidden_size=attetion_value_hidden_size)

        attention_output_size = \
            attention_input_size if attetion_value_hidden_size is None else attetion_value_hidden_size

        fc_input_size = attention_output_size + lstm_hidden_size
        self.fc = Linear(in_features=fc_input_size,
                         out_features=label_vocabulary.label_size)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, sentence: LongTensor, category: LongTensor) -> ACSAModelOutputs:
        """
        模型运行
        :param sentence: 句子的 index tensor
        :param category: category index tensor
        :return:
        """

        assert sentence.dim() == 2, f"sentence dim: {sentence.dim()} 与 (batch_size, seq_len) 不匹配"
        assert category.dim() == 1, f"category dim: {category.dim()} 与 （batch_size,) 不匹配"

        bool_mask: BoolTensor = sequence_mask(sequence=sentence,
                                              padding_index=self._token_vocabulary.padding_index)
        long_mask = bool_mask.long()

        sentence_length = long_mask.sum(dim=-1)
        assert sentence_length.dim() == 1, f"sentence_length dim: {sentence_length.dim()} 与 (batch_size,) 不匹配"

        # sentence embedding, shape: (batch_size, seq_len, embedding_dim)
        sentence_embedding = self.token_embedding(sentence)

        assert sentence_embedding.dim() == 3, \
            f"sentence_embedding dim: {sentence_embedding.dim()} 与 (batch_size, seq_len, embedding_dim) 不匹配"

        # 对 category expand，(batch_size,) -> (batch_size, seq_len)
        # category.unsequeeze, (batch_size,) -> (batch_size, 1)
        category = category.unsqueeze(dim=1)
        # category.expand_as, (batch_size, 1) -> (batch_size, seq_len)
        category = category.expand_as(sentence)

        # category embedding, shape: (batch_size, seq_len, category_embedding_dim)
        category_embedding = self.category_embedding(category)
        assert category_embedding.dim() == 3, \
            f"category_embedding dim: {category_embedding.dim()} 与 (batch_size, seq_len, category_embedding_dim) 不匹配"

        # 将word embedding 与 category embedding 合并在一起
        input_embedding = torch.cat((category_embedding, sentence_embedding), dim=-1)

        # 使用 lstm sequence encoder 进行 encoder
        packed_sentence_embedding = pack_padded_sequence(input=input_embedding,
                                                         lengths=sentence_length,
                                                         batch_first=True,
                                                         enforce_sorted=False)

        packed_sequence, (h_n, c_n) = self.lstm(packed_sentence_embedding)

        # Tuple, sentence: shape: B * SeqLen * InputSize 和 sentence length
        (sentence_encoding, _) = pad_packed_sequence(packed_sequence, batch_first=True)

        # h_n shape (num_layers * num_directions, batch_size, hidden_size)
        h_n = torch.transpose(h_n, 0, 1)

        last_index = -2 if self.lstm.bidirectional else -1
        hidden_size = self.lstm.hidden_size * 2 if self.lstm.bidirectional else self.lstm.hidden_size

        # hn_last shape: (batch_size, hidden_size * (1 or 2))
        hn_last = h_n[:, last_index:, :].contiguous().view(-1, hidden_size)

        # 将 lstm 输出与 aspect embedding 合并在一起，准备做 attention
        # attention_inputs shape: (batch_size, seq_len, attention_dim = (lstm_hidden_size + category_dim))
        attention_inputs = torch.cat((sentence_encoding, category_embedding), dim=-1)

        # attention_seq_vec shape: (B,  lstm_hidden_size + category_dim)
        attention_seq_vec = self.attention_seq2vec(sequence=attention_inputs, mask=long_mask)

        # sentiment_vec shape: (B, lstm_hidden_size + category_dim + lstm_hidden_size)
        sentiment_vec = torch.cat((attention_seq_vec, hn_last), dim=-1)

        logits = self.fc(sentiment_vec)

        model_outputs = ACSAModelOutputs(logits=logits)

        return model_outputs


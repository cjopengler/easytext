#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
模型输入的 collate

Authors: PanXu
Date:    2020/06/27 00:24:00
"""
from typing import Iterable, Union

import torch

from easytext.data import ModelInputs, Instance
from easytext.data import ModelCollate
from easytext.data import Vocabulary, LabelVocabulary, PretrainedVocabulary


class NerModelCollate(ModelCollate):
    """
    ner 的 model collate
    """

    def __init__(self,
                 token_vocab: Union[Vocabulary, PretrainedVocabulary],
                 sequence_label_vocab: LabelVocabulary,
                 sequence_max_len: int = 512):
        self._token_vocab = token_vocab
        self._sequence_label_vocab = sequence_label_vocab
        self._max_len = sequence_max_len

    def __call__(self, instances: Iterable[Instance]) -> ModelInputs:

        batch_token_indices = list()
        batch_sequence_label_indices = list()
        batch_metadatas = list()

        batch_max_len = 0
        batch_size = 0

        for instance in iter(instances):
            tokens = instance["tokens"]

            if len(tokens) > batch_max_len:
                batch_max_len = len(tokens)

        batch_max_len = batch_max_len if batch_max_len < self._max_len else self._max_len

        for instance in iter(instances):
            batch_size += 1
            token_indices = [self._token_vocab.padding_index] * batch_max_len

            for i, token in enumerate(instance["tokens"][0: batch_max_len]):
                token_indices[i] = self._token_vocab.index(token.text)

            batch_token_indices.append(token_indices)

            sequence_label_indices = [self._sequence_label_vocab.padding_index] * batch_max_len

            for i, sl in enumerate(instance["sequence_label"][0: batch_max_len]):
                sequence_label_indices[i] = self._sequence_label_vocab.index(sl)

            batch_sequence_label_indices.append(sequence_label_indices)

            batch_metadatas.append(instance["metadata"])

        batch_token_indices = torch.tensor(batch_token_indices, dtype=torch.long)
        batch_sequence_label_indices = torch.tensor(batch_sequence_label_indices, dtype=torch.long)

        model_inputs = ModelInputs(batch_size=batch_size,
                                   model_inputs={
                                       "tokens": batch_token_indices,
                                       "metadata": batch_metadatas
                                   },
                                   labels=batch_sequence_label_indices)
        return model_inputs






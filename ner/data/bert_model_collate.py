#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
基于 bert 的 token indexer

Authors: PanXu
Date:    2020/09/09 20:06:00
"""
import json
from typing import Iterable, Union

import torch

from transformers import BertTokenizer

from easytext.data import ModelInputs, Instance
from easytext.data import ModelCollate
from easytext.data import LabelVocabulary
from easytext.component.register import ComponentRegister


@ComponentRegister.register(name="BertModelCollate", name_space="data")
class BertModelCollate(ModelCollate):
    """
    ner 的 bert model collate
    """

    def __init__(self,
                 tokenizer: BertTokenizer,
                 sequence_label_vocab: LabelVocabulary,
                 sequence_max_len: int = 508):
        self._tokenizer = tokenizer
        self._sequence_label_vocab = sequence_label_vocab
        self._max_len = sequence_max_len

    def __call__(self, instances: Iterable[Instance]) -> ModelInputs:

        batch_tokens = list()
        batch_sequence_labels = None
        batch_metadatas = list()

        batch_size = 0

        for instance in iter(instances):
            batch_size += 1

            text = "".join([token.text for token in instance["tokens"]])
            batch_tokens.append(text)

            if "sequence_label" in instance:
                if batch_sequence_labels is None:
                    batch_sequence_labels = list()
                batch_sequence_labels.append(instance["sequence_label"])

            batch_metadatas.append(instance["metadata"])

        batch_inputs = self._tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_tokens,
                                                         truncation=True,
                                                         padding=True,
                                                         max_length=self._max_len,
                                                         return_length=True,
                                                         return_special_tokens_mask=True,
                                                         return_tensors="pt")

        input_ids = batch_inputs["input_ids"]

        attention_mask = batch_inputs["attention_mask"]
        token_type_ids = batch_inputs["token_type_ids"]

        batch_special_tokens_mask = batch_inputs["special_tokens_mask"]

        # 将speical_tokens_mask 0->1, 1->0, 就变成了 seuquence 去掉 CLS 和 SEP 的 mask 了
        batch_sequence_mask: torch.Tensor = batch_special_tokens_mask == 0

        batch_sequence_label_indices = None

        if batch_sequence_labels is not None:
            batch_sequence_label_indices = list()

            for sequence_label, sequence_mask in zip(batch_sequence_labels, batch_sequence_mask):
                sequence_label_indices = [self._sequence_label_vocab.index(label) for label in sequence_label]
                sequence_label_indices = torch.tensor(sequence_label_indices, dtype=torch.long)

                full_padding = torch.full_like(sequence_mask,
                                               self._sequence_label_vocab.padding_index,
                                               dtype=torch.long)
                sequence_label_indices = full_padding.masked_scatter(mask=sequence_mask,
                                                                     source=sequence_label_indices)
                batch_sequence_label_indices.append(sequence_label_indices)

            batch_sequence_label_indices = torch.stack(batch_sequence_label_indices, dim=0)

        model_inputs = ModelInputs(batch_size=batch_size,
                                   model_inputs={
                                       "input_ids": input_ids,
                                       "attention_mask": attention_mask,
                                       "token_type_ids": token_type_ids,
                                       "sequence_mask": batch_sequence_mask,
                                       "metadata": batch_metadatas
                                   },
                                   labels=batch_sequence_label_indices)
        return model_inputs




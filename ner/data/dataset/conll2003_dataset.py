#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
conll2003 dataset

Authors: panxu(panxu@baidu.com)
Date:    2020/06/26 12:16:00
"""
import logging

from typing import List
import itertools
from torch.utils.data import Dataset

from easytext.data import Instance
from easytext.data.tokenizer import Token
from easytext.data.tokenizer import EnTokenizer
from easytext.utils import bio as BIO


class Conll2003Dataset(Dataset):
    """
    conll2003 数据集
    """

    def __init__(self, dataset_file_path: str):
        """
        初始化
        :param dataset_file_path: 数据集的文件路径
        """
        super().__init__()

        self._instances: List[Instance] = list()

        tokenizer = EnTokenizer(is_remove_invalidate_char=False)

        logging.info(f"Begin read conll2003 dataset: {dataset_file_path}")

        with open(dataset_file_path, encoding="utf-8") as data_file:

            # 两个 分隔行 之间的是一个样本
            for is_divider, lines in itertools.groupby(data_file, Conll2003Dataset._is_divider):

                if not is_divider:

                    lines = [_ for _ in lines]
                    fields = [line.strip().split() for line in lines]

                    fields = [list(field) for field in zip(*fields)]
                    tokens_, pos_tags, chunk_tags, ibo1_labels = fields

                    text = " ".join(tokens_)

                    logging.debug(f"text: {text}")
                    tokens = tokenizer.tokenize(text)

                    assert len(tokens) == len(ibo1_labels), \
                        f"token 长度: {len(tokens)} 与 标签长度: {len(ibo1_labels)} 不匹配"

                    bio_labels = BIO.ibo1_to_bio(ibo1_labels)

                    instance = Instance()
                    instance["metadata"] = {"text": text,
                                            "sequence_label": bio_labels}
                    instance["tokens"] = tokens
                    instance["sequence_label"] = bio_labels

                    self._instances.append(instance)

    @staticmethod
    def _is_divider(line: str) -> bool:
        """
        判断该行是否是 分隔行。包括两种情况: 1. 空行 2. "-DOCSTART-" 这两种否是分隔行
        :param line: 行的内容
        :return: True: 是分隔行; False: 不是分隔行
        """

        if line.strip() != "":
            first_token = line.split()[0]
            if first_token != "-DOCSTART-":
                return False

        return True

    def __getitem__(self, index: int) -> Instance:
        return self._instances[index]

    def __len__(self) -> int:
        return len(self._instances)



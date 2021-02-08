#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
lattice demo 数据的 dataset

Authors: PanXu
Date:    2021/02/07 08:53:00
"""
from typing import List
import logging
import itertools

from torch.utils.data import Dataset

from easytext.data import Instance
from easytext.data.tokenizer import ZhTokenizer
from easytext.utils.bio_schema import bmes_to_bio


class LatticeNerDemoDataset(Dataset):
    """
    Lattice Demo Dataset
    """

    def __init__(self, dataset_file_path: str):
        """
        初始化
        :param dataset_file_path: 数据集的文件路径
        """
        super().__init__()

        self._instances: List[Instance] = list()

        tokenizer = ZhTokenizer(is_remove_invalidate_char=False)

        logging.info(f"Begin read lattice ner demo dataset: {dataset_file_path}")

        with open(dataset_file_path, encoding="utf-8") as data_file:

            # 两个 分隔行 之间的是一个样本
            for is_divider, lines in itertools.groupby(data_file, LatticeNerDemoDataset._is_divider):

                if not is_divider:

                    lines = [_ for _ in lines]
                    fields = [line.strip().split() for line in lines]

                    fields = [list(field) for field in zip(*fields)]
                    tokens_, bmes_labels = fields

                    text = "".join(tokens_)

                    # logging.debug(f"text: {text}")
                    tokens = tokenizer.tokenize(text)

                    assert len(tokens) == len(bmes_labels), \
                        f"token 长度: {len(tokens)} 与 标签长度: {len(bmes_labels)} 不匹配"

                    bio_labels = bmes_to_bio(bmes_labels)

                    instance = Instance()
                    instance["metadata"] = {"text": text,
                                            "sequence_label": bio_labels}
                    instance["tokens"] = tokens
                    instance["sequence_label"] = bio_labels

                    self._instances.append(instance)

    @staticmethod
    def _is_divider(line: str) -> bool:
        """
        判断该行是否是 分隔行。空行，就是分隔符。
        :param line: 行的内容
        :return: True: 是分隔行; False: 不是分隔行
        """

        return line.strip() == ""

    def __getitem__(self, index: int) -> Instance:
        return self._instances[index]

    def __len__(self) -> int:
        return len(self._instances)



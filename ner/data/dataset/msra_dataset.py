#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
msra dataset

Authors: PanXu
Date:    2020/09/09 15:11:00
"""
from typing import List
from torch.utils.data import Dataset

from easytext.data import Instance
from easytext.data.tokenizer import ZhTokenizer


class MsraDataset(Dataset):
    """
    Msra Dataset 处理，相应说明请参考 docs/docs/ner/命名实体识别.md
    """
    NAME = "msra"

    PER = "nr"
    LOC = "ns"
    ORG = "nt"
    NONE = "o"

    def __init__(self, dataset_file_path: str):
        """
        初始化, 会将数据集转换成 instance
        :param dataset_file_path: 数据集文件路径
        """
        self._instances: List[Instance] = list()
        self._tokenizer = ZhTokenizer()

        with open(dataset_file_path, encoding="utf-8") as dataset_file:
            for line in dataset_file:
                line = line.strip()
                if len(line) == 0:
                    continue
                seg_tags = line.split()

                sentence = list()
                sequence_label = list()

                for seg_tag in seg_tags:
                    seg_tag_item = seg_tag.split("/")

                    assert len(seg_tag_item) == 2, f"{seg_tag} 没有被分成2部分: {line}"

                    seg, tag = seg_tag_item

                    sentence.extend(seg)

                    if tag == MsraDataset.NONE:

                        bio_tags = ["O"] * len(seg)

                    elif tag == MsraDataset.PER:
                        bio_tags = ["I-PER"] * len(seg)
                        bio_tags[0] = "B-PER"

                    elif tag == MsraDataset.LOC:
                        bio_tags = ["I-LOC"] * len(seg)
                        bio_tags[0] = "B-LOC"
                    elif tag == MsraDataset.ORG:
                        bio_tags = ["I-ORG"] * len(seg)
                        bio_tags[0] = "B-ORG"
                    else:
                        raise RuntimeError(f"tag: {tag} 是错误的，应该是 "
                                           f"{MsraDataset.NONE}, {MsraDataset.PER}, "
                                           f"{MsraDataset.LOC}, {MsraDataset.ORG}")
                    sequence_label.extend(bio_tags)

                sentence = "".join(sentence)
                instance = Instance()
                instance["tokens"] = self._tokenizer.tokenize(sentence)
                instance["sequence_label"] = sequence_label
                instance["metadata"] = {"text": sentence,
                                        "labels": sequence_label}

                self._instances.append(instance)

    def __getitem__(self, index: int) -> Instance:
        return self._instances[index]

    def __len__(self) -> int:
        return len(self._instances)


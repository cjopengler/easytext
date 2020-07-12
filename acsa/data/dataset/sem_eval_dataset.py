#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
sem eval dataset

Authors: PanXu
Date:    2020/07/12 12:31:00
"""

from typing import List

from bs4 import BeautifulSoup

from torch.utils.data import Dataset

from easytext.data import Instance


class SemEvalDataset(Dataset):
    """
    sem eval dataset, 包括:
    sentence, term, category
    """

    def __init__(self, dataset_file_path: str):
        self._instances: List[Instance] = list()

        # 读取文件
        with open(dataset_file_path, encoding="utf-8") as f:
            content = "".join([line for line in f])

        # 使用 BeautifulSoup 解析
        soup = BeautifulSoup(content, "lxml")

        sentence_tags = soup.find_all('sentence')

        for sentence_tag in sentence_tags:
            # 提取 sentence
            sentence = sentence_tag.text.strip()

            # 提取 aspect term
            aspect_term_tags = sentence_tag.find_all('aspectterm')
            aspect_terms = []
            for aspect_term_tag in aspect_term_tags:
                term = aspect_term_tag['term'].strip()
                polarity = aspect_term_tag['polarity']
                from_index = int(aspect_term_tag['from'])
                to_index = int(aspect_term_tag['to'])
                aspect_term = {"term": term,
                               "polarity": polarity,
                               "begin": from_index,
                               "end": to_index}

                aspect_terms.append(aspect_term)

            # 提取 aspect categories
            aspect_categories = []
            aspect_category_tags = sentence_tag.find_all('aspectcategory')

            for aspect_category_tag in aspect_category_tags:
                category = aspect_category_tag['category'].strip()
                polarity = aspect_category_tag['polarity'].strip()
                aspect_category = {"category": category,
                                   "polarity": polarity}
                aspect_categories.append(aspect_category)

            instance = Instance()
            instance["sentence"] = sentence
            instance["aspect_categories"] = aspect_categories
            instance["aspect_terms"] = aspect_terms

            self._instances.append(instance)

    def __getitem__(self, index: int) -> Instance:
        return self._instances[index]

    def __len__(self) -> int:
        return len(self._instances)





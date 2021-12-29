#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
mrc msra flat ner 数据集读入

Authors: PanXu
Date:    2021/10/18 17:43:00
"""
import json
from typing import Dict, Union, List

from easytext.data import Instance
from easytext.data.dataset import Dataset
from easytext.component.register import ComponentRegister


@ComponentRegister.register(name_space="mrc_ner")
class MSRAFlatNerDataset(Dataset):
    """
    MSRA flat ner dataset
    """

    def __init__(self, is_training: bool, dataset_file_path: str):
        super(MSRAFlatNerDataset, self).__init__(is_training=is_training)

        self._instances: List[Instance] = list()

        if self._is_training:
            with open(dataset_file_path, encoding="utf-8") as f:

                all_data = json.load(f)

                for item in all_data:
                    instance = self.create_instance(item)
                    self._instances.append(instance)

    def create_instance(self, input_data: Dict) -> Union[Instance, List[Instance]]:
        instance = Instance()

        instance["context"] = "".join(input_data["context"].split())
        instance["query"] = input_data["query"]

        instance["entity_label"] = input_data.get("entity_label", None)
        instance["impossible"] = input_data.get("impossible", None)

        instance["start_positions"] = input_data.get("start_position", None)
        instance["end_positions"] = input_data.get("end_position", None)

        return instance

    def __getitem__(self, index: int) -> Instance:
        return self._instances[index]

    def __len__(self) -> int:
        return len(self._instances)







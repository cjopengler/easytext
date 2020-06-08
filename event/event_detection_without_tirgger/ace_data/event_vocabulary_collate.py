#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
基于

Authors: panxu
Date:    2020/06/08 12:13:00
"""
from typing import Iterable

from data import Instance
from data.collate import ModelInputs

from easytext.data.collate import Collate


class EventVocabularyCollate(Collate):
    """
    用来计算 ACE Event
    """

    def __call__(self, instances: Iterable[Instance]) -> ModelInputs:
        pass
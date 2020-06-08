#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2020/05/15 10:50:00
"""
import os
from unittest import TestCase

ASSERT = TestCase()

# 设置 root， 放在 tests 中，是因为 easytext 是库，只有在 tests 的时候，涉及到数据存储
ROOT_PATH = os.path.join(os.path.dirname(__file__), "../..")

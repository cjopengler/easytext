#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
brief

Authors: PanXu
Date:    2021/08/14 11:54:00
"""

import logging
import os
from easytext.utils import log_util

# 设置 root, 用作数据路径访问，包括测试数据，训练数据等
ROOT_PATH = os.path.join(os.path.dirname(__file__), "../")

log_util.config(level=logging.DEBUG)

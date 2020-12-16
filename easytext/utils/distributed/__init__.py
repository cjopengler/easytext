#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
多 GPU 相关工具

Authors: PanXu
Date:    2020/12/16 16:51:00
"""

from .sync_util import Sync
from .distributed_util import DistributedFuncWrapper
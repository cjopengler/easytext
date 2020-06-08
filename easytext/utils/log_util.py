#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2020/05/25 17:04:00
"""

import os
import logging
import logging.handlers


def config(level: int = logging.DEBUG,
           log_file_path: str = None,
           log_mode: str = 'w') -> None:
    """
    this function is deprecated
    log配置
    :param level: log level
    :param log_file_path: log输出路径
    :param log_mode: w or a文件写入模式
    :return: None
    """
    if log_file_path is None:
        logging.basicConfig(
            level=level,
            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
            datefmt='%Y-%m-%D %H:%M:%S')
    else:
        logging.basicConfig(
            level=level,
            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
            datefmt='%Y-%m-%D %H:%M:%S',
            filename=log_file_path,
            filemode=log_mode)

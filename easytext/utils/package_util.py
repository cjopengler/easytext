#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
brief

Authors: PanXu
Date:    2020/11/01 18:00:00
"""
import sys
import importlib
import pkgutil


def import_submodules(package_name: str) -> None:
    """
    导入指定报名的所有包
    :param package_name: 当前包名下所有包
    :return: None
    """
    importlib.invalidate_caches()

    sys.path.append('.')

    module = importlib.import_module(package_name)
    path = getattr(module, '__path__', [])
    path_string = '' if not path else path[0]

    for module_finder, name, _ in pkgutil.walk_packages(path):
        if path_string and module_finder.path != path_string:
            continue
        subpackage = f"{package_name}.{name}"
        import_submodules(subpackage)
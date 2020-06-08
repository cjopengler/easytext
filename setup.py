#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
安装脚本

Authors: panxu
Date:    2020/06/08 15:44:00
"""

from setuptools import setup
from setuptools import find_packages

setup(
    name="easytext",
    version="0.0.0.1",
    author="Pan Xu",
    author_email="cjopengler@163.com",
    description="Make it easy to train and metric NLP model.",
    url="https://github.com/cjopengler/easytext",
    license="MIT",
    packages=find_packages(),   # 指定需要安装的模块
    install_requires=["torch>=1.5.0"],

    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        # Pick your license as you wish (should match "license" above)
        'License :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',

        "Operating System :: OS Independent",
    ],
    zip_safe=False
)
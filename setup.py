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
    description="NLP model tool",
    long_description="Make it easy to train andm metric NLP model.",
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=find_packages(),   # 指定需要安装的模块
    # packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    # py_modules=["six"], # 剔除不属于包的单文件Python模块
    # install_requires=['peppercorn'], # 指定项目最低限度需要运行的依赖项
    # python_requires='>=2.6, !=3.0.*, !=3.1.*, !=3.2.*, <4', # python的依赖关系
    # package_data={
    # 'sample': ['package_data.dat'],
    # }, # 包数据，通常是与软件包实现密切相关的数据
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
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.2',
        # 'Programming Language :: Python :: 3.3',
        # 'Programming Language :: Python :: 3.4',
        "Operating System :: OS Independent",
    ],
)

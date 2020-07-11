#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
词汇表

Authors: panxu(panxu@baidu.com)
Date:    2020/04/25 08:33:00
"""
import os
import logging
import json
from typing import List, Dict, Iterable, Union
from collections import Counter


class IVocabulary:
    """
    词汇表的接口类
    """

    # 常量定义
    PADDING = "@@PADDING@@"
    UNK = "@@UNK@@"

    _NEW_LINE = "@@NEWLINE@@"  # "\n" 的替代， 只有在保存字典的时候才会用到

    _VOCABULARY_FILE_NAME = "vocabulary.txt"
    _CONFIG_JSON_FILE_NAME = "config.json"

    def __len__(self):
        raise NotImplementedError()

    @property
    def padding(self) -> str:
        """
        padding属性
        :return: padding属性
        """
        raise NotImplementedError

    @property
    def unk(self):
        """
        unk 属性
        :return: unk 属性
        """
        raise NotImplementedError

    @property
    def padding_index(self) -> Union[None, int]:
        """
        :return: 获取 padding 的 index, 如果 padding 没有设置，那么返回 None; 否则，返回实际的index.
        """
        raise NotImplementedError()

    def index(self, token: str) -> int:
        """
        获取token的index
        :param token: 输入的token
        :return: token 的 index
        """
        raise NotImplementedError()

    def token(self, index: int) -> str:
        """
        获取 index 的 token
        :param index: 指定 index
        :return: 当前 index 的 token
        """
        raise NotImplementedError()

    @property
    def size(self):
        raise NotImplementedError()

    def save_to_file(self, directory: str) -> "Vocabulary":
        """
        保存到指定路径中，会在下面生成两个文件，分别是:
        * vocabulary.txt - 保留全部的字典信息
        * config.json - 保留一些配置信息，比如 padding, unk 等
        :param directory: 存放的路径。
        :return: self
        """

        raise NotImplementedError()

    @classmethod
    def from_file(cls, directory: str) -> "Vocabulary":
        """
        从文件中载入词典，
        保存到指定路径中，包含两个文件，分别是:
        * vocabulary.txt - 保留全部的字典信息
        * config.json - 保留一些配置信息，比如 padding, unk 等
        :param directory: 文件路径
        :return:
        """

        raise NotImplementedError


class Vocabulary(IVocabulary):

    def __init__(self,
                 tokens: Iterable[List[str]],
                 padding: str,
                 unk: str,
                 special_first: bool,
                 other_special_tokens: List = None,
                 min_frequency: int = 1,
                 max_size: int = None):
        """
        初始化
        :param tokens: (B, seq_len)
        :param padding: padding的字符串, 可以用 Vocabulary.PADDING, 如果为 None 或者 "", 表示不进行padding
        :param unk: unknown 的单词，可以用 Vocabulary.UNK, 如果为 None 或者 "", 表示不进行padding
        :param special_first: special 是指: padding, unk.
        True: 表示放在最前面, padding index=0, unk index=1; False: 表示放在最后面。
        这涉及到mask, 对于 token 来说，一般 padding_index = 0;
        而对于 label 来说, 如果需要对label,
        比如 sequence label 进行 padding 的时候, padding_index 和 unk_index 必须大于 label的数量，因为 小于 label 数量的是对应的
        label 分类。
        :param min_frequency: 小于 min_frequency 的会被过滤
        :param max_size: 词表的最大长度, 如果为None, 不限制词表大小
        """
        self._padding = padding
        self._unk = unk
        self.min_frequency = min_frequency
        self.max_size = max_size
        self._token2index: Dict = dict()
        self._index2token: Dict = dict()

        self.other_special_tokens = other_special_tokens or list()

        counter = Counter(t for tt in tokens for t in tt)

        special_tokens = list()

        if self.padding:
            special_tokens.append(self.padding)

        if self.unk:
            special_tokens.append(self.unk)

        special_tokens.extend(self.other_special_tokens)

        # 删除 special tokens
        for st in special_tokens:
            del counter[st]

        # 将counter按照词频从高到低排序
        token_freqency = sorted(counter.items(), key=lambda item: item[1], reverse=True)

        # 构建token index 和 index token 字典

        unique_tokens = list()

        if special_first:
            unique_tokens.extend(special_tokens)

        max_size = None if self.max_size is None else (self.max_size - len(special_tokens))

        for token, freqency in token_freqency:

            if freqency < self.min_frequency:
                continue

            unique_tokens.append(token)

            if max_size and len(unique_tokens) >= max_size:
                break

        if not special_first:
            unique_tokens.extend(special_tokens)

        for index, token in enumerate(unique_tokens):
            self._index2token[index] = token
            self._token2index[token] = index

    def __len__(self):
        return len(self._token2index)

    @property
    def padding(self) -> str:
        return self._padding

    @property
    def unk(self) -> str:
        return self._unk

    @property
    def padding_index(self) -> Union[None, int]:
        """
        :return: 获取 padding 的 index, 如果 padding 没有设置，那么返回 None; 否则，返回实际的index.
        """

        if self.padding is None or self.padding == "":
            return None
        return self.index(self.padding)

    def index(self, token: str) -> int:
        """
        获取token的index
        :param token: 输入的token
        :return: token 的 index
        """
        if token is None or token == "":
            raise RuntimeError(f"token:[{token}] 非法!")

        if not isinstance(token, str):
            raise RuntimeError(f"token 类型: {type(token)} 不是 str")

        if self.unk is None or self.unk == "":
            # 当 unk invalidate 的时候, 说明这里没有 unk, 那么直接返回；
            # 如果 token  有异常，那么，返回错误即可
            index = self._token2index[token]
        else:
            # 如果 unk 存在, 那么, 无效的 token 要返回 unk index.
            index = self._token2index.get(token, self._token2index[self.unk])
        return index

    def token(self, index: int) -> str:
        """
        获取 index 的 token
        :param index: 指定 index
        :return: 当前 index 的 token
        """
        return self._index2token[index]

    @property
    def size(self):
        return len(self._token2index)

    def save_to_file(self, directory: str) -> "Vocabulary":
        """
        保存到指定路径中，会在下面生成两个文件，分别是:
        * vocabulary.txt - 保留全部的字典信息
        * config.json - 保留一些配置信息，比如 padding, unk 等
        :param directory: 存放的路径。
        :return: self
        """

        if not os.path.isdir(directory):
            raise RuntimeError(f"{directory} 不存在!")

        vocabulary_file_path = os.path.join(directory, Vocabulary._VOCABULARY_FILE_NAME)

        if os.path.isfile(vocabulary_file_path):
            logging.warning(f"{vocabulary_file_path} 已经存在，会被覆盖!")

        config_file_path = os.path.join(directory, Vocabulary._CONFIG_JSON_FILE_NAME)

        if os.path.isfile(config_file_path):
            logging.warning(f"{config_file_path} 已经存在，会被覆盖!")

        # 写入 vocabulary
        with open(vocabulary_file_path, mode="w", encoding="utf-8") as f:

            for i in range(len(self._index2token)):
                token = self._index2token[i].replace('\n', Vocabulary._NEW_LINE)
                f.write(f"{token}\n")

        # 写入 config
        with open(config_file_path, mode="w", encoding="utf-8") as f:
            config = {"padding": self.padding,
                      "unk": self.unk,
                      "other_special_tokens": self.other_special_tokens}
            json.dump(config, f, ensure_ascii=False)

        return self

    @classmethod
    def from_file(cls, directory: str) -> "Vocabulary":
        """
        从文件中载入词典，
        保存到指定路径中，包含两个文件，分别是:
        * vocabulary.txt - 保留全部的字典信息
        * config.json - 保留一些配置信息，比如 padding, unk 等
        :param directory: 文件路径
        :return:
        """

        if not os.path.isdir(directory):
            raise RuntimeError(f"{directory} 不存在!")

        vocabulary_file_path = os.path.join(directory, Vocabulary._VOCABULARY_FILE_NAME)

        if not os.path.isfile(vocabulary_file_path):
            raise RuntimeError(f"{vocabulary_file_path} 不存在!")

        config_file_path = os.path.join(directory, Vocabulary._CONFIG_JSON_FILE_NAME)

        if not os.path.isfile(config_file_path):
            raise RuntimeError(f"{config_file_path} 不存在!")

        # 载入config

        with open(config_file_path, mode="r", encoding="utf-8") as f:
            config = json.load(f)

        if config is None:
            raise RuntimeError(f"config 从 {config_file_path} 载入失败!")

        padding = config["padding"]
        unk = config["unk"]
        other_special_tokens = config["other_special_tokens"]

        # 构造对象，实际的数据由后面来填充
        vocabulary = cls([[]],
                         padding=padding,
                         unk=unk,
                         other_special_tokens=other_special_tokens,
                         special_first=True)

        with open(vocabulary_file_path, mode="r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                # 将 \n 去掉，这里不能使用 strip， 因为 token 中可能会存在其他不可见字符
                # 同时将 之前被 _NEW_LINE 替换掉的 "\n" 替换回来
                token = line[:-1].replace(Vocabulary._NEW_LINE, '\n')
                vocabulary._index2token[index] = token
                vocabulary._token2index[token] = index

        return vocabulary


class LabelVocabulary(IVocabulary):
    """
    label 的 vocabulary
    """

    def __init__(self, labels: Iterable[List[str]], padding: str):
        """
        Label 词汇表
        :param labels: label 列表, 二维, 类似于 batch labels
        :param padding: label 需要填充的padding, 填充的 padding index 是不在 [0, label_num) 范围内的，
        也就是说不会影响 label的index. 如果不需要填充，请设置为 `""` 或者 `None`
        """
        self._vocabulary = Vocabulary(tokens=labels,
                                      padding=padding,
                                      unk="",
                                      special_first=False,
                                      min_frequency=1,
                                      max_size=None)

    def __len__(self):
        return len(self._vocabulary)

    @property
    def padding(self) -> str:
        return self._vocabulary.padding

    @property
    def unk(self):
        return self._vocabulary.unk

    @property
    def padding_index(self) -> Union[None, int]:
        return self._vocabulary.padding_index

    def index(self, token: str) -> int:
        return self._vocabulary.index(token)

    def token(self, index: int) -> str:
        return self._vocabulary.token(index)

    @property
    def size(self):
        return self._vocabulary.size

    def save_to_file(self, directory: str) -> "LabelVocabulary":
        self._vocabulary.save_to_file(directory=directory)
        return self

    @classmethod
    def from_file(cls, directory: str) -> "LabelVocabulary":

        label_vocabulary = cls(labels=[[]], padding="")
        label_vocabulary._vocabulary = Vocabulary.from_file(directory=directory)
        return label_vocabulary

    @property
    def label_size(self):
        """
        实际的 label 数量，不包含 padding. 如果需要获得包含 padding 的长度，请使用 size 或者 len(LabelVocabulary)
        :return: label 数量
        """

        if self.padding is None or self.padding == "":
            return len(self)

        else:
            # 将padding 减掉
            return len(self) - 1

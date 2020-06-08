# coding:utf-8
"""
json 工具
"""
import torch
import json
from datetime import datetime, date

from easytext.utils import dt_util


class _JsonEncoder(json.JSONEncoder):
    """
    全局的json encoder
    """

    def default(self, o):
        """
        json encoder执行的地方
        :param o: object
        :return:
        """

        if isinstance(o, datetime):
            return o.strftime(dt_util.FORMATE_1)
        elif isinstance(o, date):
            return o.strftime(dt_util.FORMATE_3)
        elif isinstance(o, torch.Tensor):
            if o.dim() == 0:
                return o.item()
            else:
                return o.tolist()
        else:
            pass

        return o.__dict__


def json2str(json_obj: json, cls: json.JSONEncoder = _JsonEncoder, indent: int = None) -> str:
    """
    obj转换成json字符串
    :param json_obj: 转换的对象
    :param cls: json encoder
    :param indent: 对json进行美化. index=4 表示按照4个空格进行缩进
    :return: 字符串
    """
    return json.dumps(json_obj, cls=cls, ensure_ascii=False, indent=indent)

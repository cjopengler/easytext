# coding:utf-8
"""
时间日期工具包
"""
import time

FORMATE_1 = '%Y-%m-%d %H:%M:%S'
FORMATE_2 = '%Y-%m-%d_%H:%M:%S'
FORMATE_3 = '%Y-%m-%d'
FORMATE_4 = '%H:%M:%S'
FORMATE_5 = '%Y%m%d'
FORMATE_6 = '%Y%m%d%H%M%S'
FORMATE_7 = '%Y%m%d_%H:%M:%S'
FORMATE_8 = '%Y%m%d%H0000'
FORMATE_9 = '%Y%m%d_%H%M%S'

def datetime2timestamp(cur_datetime):
    """
    日期转换成时间戳
    :param cur_datetime: data_time
    :return: 时间戳整数
    """

    return int(time.mktime(cur_datetime.timetuple()))

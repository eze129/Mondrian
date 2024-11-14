# coding:utf-8
"""
main module of mondrian
"""

# Implemented by Qiyuan Gong
# qiyuangong@gmail.com
# 2014-09-11

# @InProceedings{LeFevre2006,
#   Title = {Mondrian Multidimensional K-Anonymity},
#   Author = {LeFevre, Kristen and DeWitt, David J. and Ramakrishnan, Raghu},
#   Booktitle = {ICDE '06: Proceedings of the 22nd International Conference on Data Engineering},
#   Year = {2006},
#   Address = {Washington, DC, USA},
#   Pages = {25},
#   Publisher = {IEEE Computer Society},
#   Doi = {http://dx.doi.org/10.1109/ICDE.2006.101},
#   ISBN = {0-7695-2570-9},
# }

# !/usr/bin/env python
# coding=utf-8

import pdb
import time
from utils.utility import cmp_value, value, merge_qi_value
from functools import cmp_to_key

# warning all these variables should be re-inited, if
# you want to run mondrian with different parameters
__DEBUG = False
QI_LEN = 10
GL_K = 0
RESULT = []
QI_RANGE = []
QI_DICT = []
QI_ORDER = []


class Partition(object):

    """
    Class for Group (or EC), which is used to keep records
    self.member: records in group
    self.low: lower point, use index to avoid negative values
    self.high: higher point, use index to avoid negative values
    self.allow: show if partition can be split on this QI
    """

    def __init__(self, data, low, high):
        """
        split_tuple = (index, low, high)
        """
        self.low = list(low)
        self.high = list(high)
        self.member = data[:]
        self.allow = [1] * QI_LEN

    def add_record(self, record, dim):
        """
        add one record to member
        """
        self.member.append(record)

    def add_multiple_record(self, records, dim):
        """
        add multiple records (list) to partition
        """
        for record in records:
            self.add_record(record, dim)

    def __len__(self):
        """
        return number of records
        """
        return len(self.member)


def get_normalized_width(partition, index):
    """
    return Normalized width of partition
    similar to NCP
    """
    d_order = QI_ORDER[index]
    width = value(d_order[partition.high[index]]) - value(d_order[partition.low[index]])
    if width == QI_RANGE[index]:
        return 1
    return width * 1.0 / QI_RANGE[index]


def choose_dimension(partition):
    """
    choose dim with largest norm_width from all attributes.
    This function can be upgraded with other distance function.
    """
    max_width = -1
    max_dim = -1
    for dim in range(QI_LEN):
        if partition.allow[dim] == 0:
            continue
        norm_width = get_normalized_width(partition, dim)
        if norm_width > max_width:
            max_width = norm_width
            max_dim = dim
    if max_width > 1:
        pdb.set_trace()
    return max_dim


def frequency_set(partition, dim):
    """
    get the frequency_set of partition on dim
    """
    frequency = {}
    for record in partition.member:
        try:
            frequency[record[dim]] += 1
        except KeyError:
            frequency[record[dim]] = 1
    return frequency


def find_median(partition, dim):
    """
    find the middle of the partition, return split_val
    """
    # use frequency set to get median
    frequency = frequency_set(partition, dim)
    split_val = ''
    next_val = ''
    value_list = list(frequency.keys())
    value_list.sort(key=cmp_to_key(cmp_value))
    total = sum(frequency.values())
    middle = total // 2
    if middle < GL_K or len(value_list) <= 1:
        try:
            return '', '', value_list[0], value_list[-1]
        except IndexError:
            return '', '', '', ''
    index = 0
    split_index = 0
    for i, qi_value in enumerate(value_list):
        index += frequency[qi_value]
        if index >= middle:
            split_val = qi_value
            split_index = i
            break
    else:
        print("Error: cannot find split_val")
    try:
        next_val = value_list[split_index + 1]
    except IndexError:
        # there is a frequency value in partition
        # which can be handle by mid_set
        # e.g.[1, 2, 3, 4, 4, 4, 4]
        next_val = split_val
    return (split_val, next_val, value_list[0], value_list[-1])



def anonymize_strict(partition):
    """
    recursively partition groups until not allowable
    """
    allow_count = sum(partition.allow)
    # only run allow_count times
    if allow_count == 0:
        RESULT.append(partition)
        return
        
    for i in range(allow_count):
        dimension = choose_dimension(partition)
        if dimension == -1:
            raise ValueError("无法进行维度分割！")    
        
        # use frequency set to get median (*)
        
        (split_val,next_val,low,high) = find_median(partition,dimension)

        """
            if middle < GL_K or len(value_list) <= 1:
            try:
                return '', '', value_list[0], value_list[-1]
            except IndexError:
                return '', '', '', ''
        """
        #(分隔值，下一个值，最低，最高）
        # update parent low and high (*)

        if low != '':#indexError
            partition.low[dimension] = QI_DICT[dimension][low]
            partition.high[dimension] = QI_DICT[dimension][high]
        if split_val == '':
            #此时该维度不能分割，进行下一步
            partition.allow[dimension] = 0
            continue 
        # split the group from median (records equal to the mean are stored first) (*)

        #根据split_val
        mean_val = QI_DICT[dimension][split_val]#中间值
        lhs_high = partition.high.copy()
        rhs_low = partition.low.copy()
        lhs_high[dimension] = mean_val
        rhs_low[dimension] = QI_DICT[dimension][next_val]
        (lhs,rhs) = (Partition([],partition.low,lhs_high),Partition([],rhs_low,partition.high))
        #遍历原分区
        for record in partition.member:
            mid = QI_DICT[dimension][record[dimension]]
            if mid <= mean_val:#小于中间值的全部归于lhs
                lhs.add_record(record,dimension)
            else :
                rhs.add_record(record,dimension)

        # check is lhs and rhs satisfy k-anonymity*
        
        if len(lhs) < GL_K or len(rhs) < GL_K:
            partition.allow[dimension] = 0
            continue

        # anonymize sub-partition
        anonymize_strict(lhs)
        anonymize_strict(rhs)
        return
    RESULT.append(partition)


def anonymize_relaxed(partition):
    """
    recursively partition groups until not allowable
    """
    if sum(partition.allow) == 0:
        # can not split
        RESULT.append(partition)
        return
    # choose attribute from domain (*)

    #print(partition.member[-1])
    dimension = choose_dimension(partition)
    if dimension == -1:
        raise ValueError("无法进行维度分割！")    
    
    # use frequency set to get median (*)
    
    (split_val,next_val,low,high) = find_median(partition,dimension)

    """
        if middle < GL_K or len(value_list) <= 1:
        try:
            return '', '', value_list[0], value_list[-1]
        except IndexError:
            return '', '', '', ''
    """
    #(分隔值，下一个值，最低，最高）
    # update parent low and high (*)

    if low != '': #indexError
        partition.low[dimension] = QI_DICT[dimension][low]
        partition.high[dimension] = QI_DICT[dimension][high]
    if split_val == '':
        #此时该维度不能分割，进行下一步
        partition.allow[dimension] = 0
        anonymize_relaxed(partition)
        return 
    
    # split the group from median (records equal to the mean are stored first) (*)

    #根据split_val
    mean_val = QI_DICT[dimension][split_val]#中间值
    lhs_high = partition.high.copy()
    rhs_low = partition.low.copy()
    lhs_high[dimension] = mean_val
    rhs_low[dimension] = QI_DICT[dimension][next_val]
    (lhs,rhs) = (Partition([],partition.low,lhs_high),Partition([],rhs_low,partition.high))
    mid_count = 0#中间值的数量
    #遍历原分区
    for record in partition.member:
        mid = QI_DICT[dimension][record[dimension]]
        if mid < mean_val:
            lhs.add_record(record,dimension)
        elif mid == mean_val:
            mid_count += 1
        else :
            rhs.add_record(record,dimension)
    # handle records in the middle (*)
    # these records will be divided evenly between lhs and rhs,
    # such that |lhs| = |rhs| (+1 if total size is odd)
    
    #处理中间记录
    #对于中间值，采用第二种方法，将中间值平均分分到左右两侧
    for j in range(len(partition)//2-len(lhs)): 
        lhs.add_record(mean_val,dimension)
        mid_count -=1
    if mid_count > 0:
        rhs.low[dimension] = mean_val#rhs的最小值更新为中间值
        for i in range(mid_count):
            rhs.add_record(mean_val,dimension)

    # check is lhs and rhs satisfy k-anonymity (*)

    # anonymize sub-partition
    anonymize_relaxed(lhs)
    anonymize_relaxed(rhs)

def init(data, k, QI_num=-1):
    """
    reset global variables
    """
    global GL_K, RESULT, QI_LEN, QI_DICT, QI_RANGE, QI_ORDER
    if QI_num <= 0:
        QI_LEN = len(data[0]) - 1
    else:
        QI_LEN = QI_num
    GL_K = k
    RESULT = []
    # static values
    QI_DICT = []
    QI_ORDER = []
    QI_RANGE = []
    att_values = []
    for i in range(QI_LEN):
        att_values.append(set())
        QI_DICT.append(dict())
    for record in data:
        for i in range(QI_LEN):
            att_values[i].add(record[i])
    for i in range(QI_LEN):
        value_list = list(att_values[i])
        value_list.sort(key=cmp_to_key(cmp_value))
        QI_RANGE.append(value(value_list[-1]) - value(value_list[0]))
        QI_ORDER.append(list(value_list))
        for index, qi_value in enumerate(value_list):
            QI_DICT[i][qi_value] = index


def mondrian(data, k, relax=False, QI_num=-1):
    """
    Main function of mondrian, return result in tuple (result, (ncp, rtime)).
    data: dataset in 2-dimensional array.
    k: k parameter for k-anonymity
    QI_num: Default -1, which exclude the last column. Othewise, [0, 1,..., QI_num - 1]
            will be anonymized, [QI_num,...] will be excluded.
    relax: determine use strict or relaxed mondrian,
    Both mondrians split partition with binary split.
    In strict mondrian, lhs and rhs have not intersection.
    But in relaxed mondrian, lhs may be have intersection with rhs.
    """
    init(data, k, QI_num)
    result = []
    data_size = len(data)
    low = [0] * QI_LEN
    high = [(len(t) - 1) for t in QI_ORDER]
    whole_partition = Partition(data, low, high)
    # begin mondrian
    start_time = time.time()
    if relax:
        # relax model
        anonymize_relaxed(whole_partition)
    else:
        # strict model
        anonymize_strict(whole_partition)
    rtime = float(time.time() - start_time)
    # generalization result and
    # evaluation information loss
    ncp = 0.0
    dp = 0.0
    for partition in RESULT:
        rncp = 0.0
        for index in range(QI_LEN):
            rncp += get_normalized_width(partition, index)
        rncp *= len(partition)
        ncp += rncp
        dp += len(partition) ** 2
        for record in partition.member[:]:
            for index in range(QI_LEN):
                record[index] = merge_qi_value(QI_ORDER[index][partition.low[index]],
                                QI_ORDER[index][partition.high[index]])
            result.append(record)
    # If you want to get NCP values instead of percentage
    # please remove next three lines
    ncp /= QI_LEN
    ncp /= data_size
    ncp *= 100
    if __DEBUG:
        from decimal import Decimal
        print("Discernability Penalty=%.2E" % Decimal(str(dp)))
        print("size of partitions=%d" % len(RESULT))
        print("K=%d" % k)
        print("NCP = %.2f %%" % ncp)
    return (result, (ncp, rtime))

#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2020/02/07 08:10:00
"""

NEGATIVE_ENTITY_TYPE = "NEGATIVE"  # 无效实体类型

NEGATIVE_EVENT_TYPE = "Negative"  # 无效事件类型

# 事件类型全部集合
EVENT_TYPES = [NEGATIVE_EVENT_TYPE,
               "Justice:Arrest-Jail",
               "Conflict:Demonstrate",
               "Justice:Extradite",
               "Movement:Transport",
               "Justice:Charge-Indict",
               "Life:Die",
               "Transaction:Transfer-Money",
               "Business:Declare-Bankruptcy",
               "Justice:Release-Parole",
               "Justice:Trial-Hearing",
               "Business:Merge-Org",
               "Life:Marry",
               "Justice:Sentence",
               "Justice:Convict",
               "Business:End-Org",
               "Personnel:End-Position",
               "Justice:Appeal",
               "Justice:Execute",
               "Life:Divorce",
               "Conflict:Attack",
               "Contact:Phone-Write",
               "Life:Be-Born",
               "Justice:Pardon",
               "Personnel:Nominate",
               "Personnel:Elect",
               "Justice:Acquit",
               "Justice:Sue",
               "Contact:Meet",
               "Justice:Fine",
               "Life:Injure",
               "Personnel:Start-Position",
               "Transaction:Transfer-Ownership",
               "Business:Start-Org"]

assert isinstance(EVENT_TYPES, list)
assert len(EVENT_TYPES) == 34, f"target event type should be 34, now {len(EVENT_TYPES)}"
assert NEGATIVE_EVENT_TYPE in EVENT_TYPES, f"{NEGATIVE_EVENT_TYPE} not in event types file"

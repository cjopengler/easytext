# EasyText

## 应用指南

### data

创建 `data` 代码目录, 实现两个类，分别是 `Dataset` 和 `Collate`.

* `Dataset` - `from torch.utils.data import Dataset`, 读取数据，返回 `Instance`
* `Collate` - `from easytext.data.collate import Collate`, 处理数据返回模型需要的数据。

#### 具体样例

# 数据读取设计方案

数据读入以 `torch.utils.data.Dataset` 或者 `torch.utils.data.IterableDataset`
一次性读取。对数据依赖的处理，比如: `Vocabulary`, padding 或者对 token 进行
indexing 都是通过 `collate_fn` 来进行处理。 所以 `collate_fn` 是可以设置成类的，  
只要实现 `__call__` 函数即可。

## Collate

`easytext.data.collate.Collate` 是 `Collate` 基类，所有的 `Collate`
都要继承这个函数。输入是 `Instance`, 对于 `Dataset` 返回的就是 `Instance`，所以 Collate 做所有的数据预处理以及特征工程。

为什么选择这样的方式，因为 `DataLoader` 可以进行并发运算，在进行数据预处理和特征工程
这会让效率更高。

# Metric 设计

`Metric` 是所有 `Metric` 的基类，但是为了扩展，在模型中可能会出现需要计算 `Metric`
需要更多的数据，所以有了 `ModelMetricAdapter` 用来将模型的 `Outputs`
作为 `Metric` 的输入参数。

对于 `__call__` 是计算 `batch` 的 metric, 当一个 `epoch` 计算结束，需要调用
`metric` 来将 `batch` 计算的结果综合输出。


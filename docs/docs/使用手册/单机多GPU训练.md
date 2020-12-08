# 单机多GPU

主要实践在: 单机多卡。

分布式训练的思想是: 基于多进程，将数据拆分到不同的进程，进行训练。Pytorch 提供了在不同的进程中
对参数进行合并的操作。

基于上面的思想，在写分布式训练的时候，需要解决的一些组件包括:

1. `DataLoader` 如何运行?
2. `Model` 如何参数更新?
3. `Loss` 如何计算?
4. `Metric` 如何计算?

##  `DataLoader` 如何运行?

使用 `torch.utils.data.distributed.DistributedSampler` 根据当前进程对 dataset 进行采样。
类似下面这段代码:

```python

import torch

def create_dataloader(distributed: bool, dataset: torch.data.Dataset, batch_size: int):
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)
    return data_loader
```


##  `Model` 如何参数更新?
Model 使用 `torch.nn.parallel.DistributedDataParallel(model)` 设置，会自动多进程进行参数的更新。

##  `Loss` 如何计算?
Loss 会在 Trainer 中自动进行 分布式计算，所以正常写即可。
##  `Metric` 如何计算?

Metric 需要自己定义分布式该如何处理。

## 训练需要其他参数

由于使用了 pytorch 的分布式训练作为单机多GPU训练方案，那么，就需要对 `init_process_group` 
参数提供一个 `init_method` 参数, 该参数默认使用了 TCP 协议，所以，需要提供一个空闲的 PORT，
这是需要在训练的时候指定的。并没有使用 ENV 协议。

另外，说明一下关于使用设置本机环境变量。

1. 设置 `CUDA_VISIBLE_DEVICES`, 可以设置，如果设置了，要注意在配置训练的 GPU 时候，特别注意序号. 如果不配置就会正常使用训练时候配置的 gpu 参数。
2. 由于在 `init_process_group` 使用了 TCP 协议，所以不需要设置任何进程组相关环境变量，设置了也不会生效。

## 单机CPU多进程训练
这是否可行?

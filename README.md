# easytext
让自然语言模型训练更容易

## 安装

`pip install easytext-nlp`

注意: pip repository 中存在一个 easytext，不是本项目，不要安装错了。
也就是 `pip install easytext` 并非安装的是本项目。

## 构建训练
训练的具体构建流程如下:

![](./docs/docs/images/easytext.png)

# docs: 文档

`mkdocs serve` 启动文档服务。

其中:

* uml: 文件夹下是 uml 设计文档，使用 "Visual Paradigm" 工具打开
* 开发计划: 列出了已经开发出的功能和特性
* 相关模型说明以及论文文档

# ner: 命名实体识别

## 相关模型以及对应的配置文件

| 序号 | 模型描述 | 配置文件/`config_file_path` |
|-----|---------|---------|
| 1 | rnn + crf |  `data/ner/rnn_with_crf/config/config.json` |
| 2 | rnn + softmax | `data/ner/rnn_with_crf/config/config_without_crf.json`|
| 3 | bert + crf | `data/ner/bert_with_crf/config/config.json` |
| 4 | bert + softmax | `data/ner/bert_with_crf/config/config_without_crf.json` |
| 5 | bert + rnn + crf | `data/ner/bert_rnn_with_crf/config/config.json` |
| 6 | bert + rnn + softmax | `data/ner/bert_rnn_with_crf/config/config_without_crf.json` |

## 启动命令

`python -m ner.launcher --config {config_file_path}`

* config_file_path - 参考 "相关模型以及对应的配置文件" 中 "配置文件/`config_file_path`" 列内容。

# event: 事件识别

事件识别以及事件要素识别模型。

## event_detection_without_tirgger

ACL 2019 论文，忽略触发词的事件类型检测。
* [相关论文](https://www.aclweb.org/anthology/N19-1080/)
* [我的简书](https://www.jianshu.com/p/01deb5b22240)

更加详细说明，请参考: `docs/docs/event/event_detection_without_trigger.md`

# acsa

属性级情感分析，baseline 模型:

## ATAELstm

2016 emnlp 基于 Attention Lstm 的 属性级情感分析模型.

* 相关论文参考: `docs/docs/acsa/相关文章及论文/2016-emnlp-Attention-based LSTM for Aspect-level Sentiment Classification.pdf`

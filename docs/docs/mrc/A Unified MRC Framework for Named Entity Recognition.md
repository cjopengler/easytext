**A Unified MRC Framework for Named Entity Recognition**

<!-- vscode-markdown-toc -->
* 1. [NER](#ner)
* 2. [Nested NER](#nested-ner)
* 3. [MRC](#mrc)
* 4. [任务形式化定义](#任务形式化定义)
* 5. [问题生成](#问题生成)
* 6. [模型细节](#模型细节)
    * 6.1. [Span Selection](#span-selection)
* 7. [训练和测试](#训练和测试)
* 8. [MRC 的提升还是 BERT 的提升](#mrc-的提升还是-bert-的提升)
* 9. [如何构造 Query](#如何构造-query)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->


# 摘要

NER 任务被划分成两种，分别是 nested NER 和 flat NER。序列标注模型只能处理 flat Ner, 而无法处理 nested NER。

在这篇论文中，我们提出了 uified framework 对于 flat 和 nested NER 都能够处理。我们提出了基于机器阅读理解的范式(MRC) 来代替序列标注。 例如，抽取 PER 标签被形式化为抽取 "which person is mentioned in text" 问题的答案的 spans。该范式可以非常自然的处理实体的重叠问题: 两个带有不同标签的重叠的实体对应的问题不同。另外，由于 query 提供了先验知识,所以这个对性能也是有提升的。

我们在 nested 和 flat NER 数据集上都做了试验。试验结果表明前面提到的范式是有效的。我们比现有的 nested NER SOTA 有较大的提升。

# 介绍

MRC 范式有两个优点:

1. 能够处理 Neste Ner, 也就是处理重叠。
2. 利用了先验知识，比如 "ORG", 在标签角度来说，仅仅是 one-hot，是没有语义信息的，而我们现在增加先验信息进去，query 可以设计成: "find an organization such as company, agency and institution in the context"。另外，增加的先验描述，有助于区分相似的tag（设计query 是一个技术活?)

# 相关工作

##  1. <a name='ner'></a>NER

##  2. <a name='nested-ner'></a>Nested NER

##  3. <a name='mrc'></a>MRC

MRC 抽取给定问题的答案 spans。这个任务可以形式化两个多分类任务，分别预测答案的 start 和 end.

在过去的一两年，NLP 任务有种将问题转化成 MRC 的趋势。 例如, 将关系抽取转化成 QA 任务。例如: "EDUCATED-AT" 关系，能够转化成: q(x): "Where did x study?"。 摘要或情感分析也转化成 MRC 任务。对于摘要来说，Q: "What is the summary?"。NER 任务是收到关系抽取的 MRC 启发，在这篇 paper 中，更多客观事实知识，诸如同义词和示例被集成进入到 query 中，我们也会深入分析 query 构建策略的影响。

# NER as MRC

##  4. <a name='任务形式化定义'></a>任务形式化定义

给定输入序列 $X = \{x_1, x_2, ..., x_n\}$, 其中 $n$ 表示序列的长度，我们需要找到在 $X$ 中的每一个实体, 并且给他一个标签 $y \in Y$, 其中 $Y$ 是我们预先定义的所有 tag 类型，诸如 PER, LOC 等。

**数据集构造** 首先我们需要将标注好的 NER 数据集转化成 (QUESTION, ANSWER, CONTEXT) 三元组集合。对于每一个 tag 类型 $y \in Y$ 都有一个问题与之对应，这字问题定义为: $q_y = \{q_1, q_2, ..., q_m\}$, 其中 $m$ 表示产生的问题的序列长度。一个标注好的实体 $x_{start,end} = {x_{start}, x_{start+1}, ..., x_{end-1}, x_{end}}$ 是 $X$ 的子串，并且满足 $start \le end$. 每一个实体都有一个 golden label $y \in Y$. 通过生成一个基于 label $y$ 的问题 $q_y$, 我们能够获得三元组 $(q_y, x_{start, end}, X)$, 这就确切的表示了我们需要的 (QUESTION, ANSWER, CONTEXT) 三元组。注意我们使用下标 $start,end$ 来表示从 $start$ 到 $end$ 的连续 tokens.

##  5. <a name='问题生成'></a>问题生成

问题的生成过程是非常重要的，因为相对label的显然知识编码对最终的结果有着非常重要的影响。在关系抽取上，使用了一种基于模板的问题生成方法。在这篇论文汇总，我们使用标注的说明作为构建问题的参考。标注说明是有数据集的构建者提供。他们尽可能通用而又精准的描述了标签类型，一遍标注人员能够标注相关文本。表1，展示了实体类型的描述。

| Entity|Natural Language Question |
| ------| ------------------|
| Location| Find locations in the text, including non- geographical locations, mountain ranges and bodies of water.|
| Facility | Find facilities in the text, including buildings, airports, highways and bridges.|
| Organization | Find organizations in the text, including companies, agencies and institutions. |

##  6. <a name='模型细节'></a>模型细节

### 

给一个问题 $q_y$, 我们需要抽取出对应的 text span $x_{start, end}$, 类型就是 $y$。 我们使用 BERT 作为 backbone。为了使用 BERT，我们将问题 $q_y$ 与 文本 $X$ 连接在一起，形式化表示如下:

$$
\{[CLS], q_1, q_2, ..., q_m, [SEP], x_1, x_2, ..., x_n\}
$$

其中 $[CLS]$ 和 $[SEP]$ 是特殊 token. 然后 BERT 该组合字符串并且输出表示矩阵, $E \in \mathbb{R}^ {n \times d}$, 其中 $d$ 是 BERT 最后一层的向量维度, $n$ 是 $X$ 序列长度，我们会去掉问题的表示，所以就只剩下序列的长度了。

###  6.1. <a name='span-selection'></a>Span Selection

在 MRC 中, 有两种策略来进行 span 选择: 1. 有两个 n-class 的分类器，分别用来预测 start 和 end, 其中 n 是 context 的长度。这种分类方式，只能预测一个 span。另外一种方法是两个二分类预测每一个 token 是否是 start 或者 是否是 end. 这种方式允许输出多个 start 和 end，这些都是 $q_y$ 的潜在实体。我们使用第二种策略。

**Start Index Prediction**

给定一个 Bert 的输出 E, 模型首先预测每一个 token 是 start，使用如下公式:

$$
P_{start} = softmax_{each row}(E \cdot T_{start}) \in \mathbb{R}^{n \times 2}
$$

$T_{start} \in R^{d \times 2}$ 是学习的权重。$P_{start}$ 表示了一个实体在给定 query 在每一个 token index 上的 start 位置的概率分布。

**End Index Prediction**

End index 的预测过程与 start 是一样的，我们用另外一个矩阵 $T_{end}$ 来计算 $P_{end} \in \mathbb{R} ^ {n \times 2}$ 的概率矩阵。

**Start-End Matching**

在序列 X 中，我们会有个实体。这也就意味着，我们能够预测出多个 start index 以及 end index. 直接使用 start 与其 最近的 end 进行匹配是不 work 的。因为我们需要一个方法来预测 start 与 end 的匹配。

具体来说，我们应用 argmax 到每一行的 $P_{start}$ 和 $P_{end}$, 我们会得到 start 和 end 预测到的 index, 分别是: $\hat{I}_{start}, \hat{I}_{end}$.

$$
\begin{aligned}
&\hat{I}_{start} = {i | argmax(P^{i}_{start}) = 1, i = 1, ..., n} \\
&\hat{I}_{end} = {i | argmax(P^{i}_{end}) = 1, i = 1, ..., n}
\end{aligned}
$$

其中 $i$ 表示的序列中的位置。接下来使用一个二分类来预测，start 和 end 是否匹配:

$$
P_{istart,jend} = sigmod(m \cdot concat(E_{istart}, E_{jend}))
$$

其中 $m \in \mathbb{R^{1 \times 2d}}$ 是学习的参数。

##  7. <a name='训练和测试'></a>训练和测试

在训练时, X 与 ground truth 序列 $Y_{start}$ 和 $Y_{end}$ 组合在一起。 因此我们有两个 loss, 分别是:

$$
\begin{aligned}
&L_{start} = CE(P_{start}, Y_{start}) \\
&L_{end} = CE(P_{end}, Y_{end})
\end{aligned}
$$

使用 $Y_{start, end}$ 来表示 start 和 end 匹配的 golden label. 那么, start-end 匹配的 loss:

$$
L_{span} = CE(P_{start, end}, Y_{start, end})
$$

最终，总的 loss 如下:

$$
L = \alpha L_{start} + \beta L_{end} + \gamma L_{span}
$$

其中, $\alpha, \beta, \gamma \in [0, 1]$ 是控制训练时候的超参。三个 loss 被联合训练，共享 BERT。在测试阶段, start 和 end 被分离出来得到 $\hat{I}_{start}, \hat{I}_{end}$. 然后进入到 match 模型，来得到最终的实体。

# 实验

# Ablation studies

##  8. <a name='mrc-的提升还是-bert-的提升'></a>MRC 的提升还是 BERT 的提升
该方法，也可以去掉 Bert 直接使用也是有提升的。如果都使用 Bert 那么会有接近 1.95 百分点的提升。

##  9. <a name='如何构造-query'></a>如何构造 Query

* Position index of labels: 没太理解说的什么意思 ？
* Keyword: 使用关键词来标书
* 规则模板来填充: 使用一个规则模板，比如: "在文中那个组织机构被提到了?"
* wiki: wiki 上的解释
* 同义词: 使用字典上的同义词
* Keyword + 同义词
* 标注描述: 这是目前为止最好的





 



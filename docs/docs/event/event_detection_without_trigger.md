# Event Detection without Trigger

ACL 2019 论文，忽略触发词的事件类型检测。[相关论文](https://www.aclweb.org/anthology/N19-1080/)
和 [我的简书](https://www.jianshu.com/p/01deb5b22240)

## 摘要

以前的事件检测是要识别触发词以及触发词类型，但是，触发词对于事件检测来说是不必要的，而对于标注来说挑选出"most clearly"的词也是非常耗费时间的。昂贵的训练语料标注限制的现有方法的应用。为了减少人工，我们探索了无触发词的事件监测。在这项工作中，我们提出了一套先进的框架, 称之为 "Type-aware Bias Neural Network with Attention Mechanisms"(TBNNAM). 事件结果显示了这种方法的有效性，另外，请注意，我们推荐方法甚至超过了使用标注trigger的方法的state of the arts.

## 介绍

我们的工作是应对 event detection (ED), ED 的目标是识别预先定义好的事件以及类型。 例如: `In Baghdad, a cameraman died when an American tank fired on the Palestine Hotel.`， ED 应该识别出两个事件，分别是 `死亡`和`攻击`。

以前的工作都是基于ACE，必须识别出 *触发词* 。上面的句子必须识别出 `died` 和 `attack`。 以前的方法都是把这个任务当做一个 `word classification` 来处理。

可是，对于这个任务来说，事件触发词并非必须的。事件检测的目标是识别事件类型，而触发词是这个任务的中间结果。 进而， 触发词的标注也是非常耗费时间的。 为了减少这种人工成本，我们探索了 `无触发词的事件检测` 方法。 在这种方案下，只需要标注每一个句子的事件类型即可。还是前面的例子，只需要标注 `{死亡，攻击}` 即可，而以前的标注工作是 `{Death: died, Attack: fired}`。

没有了事件触发词，直观感觉模型的任务是文本分类。 可是，这里有两个挑战: 1. 多标签问题: 每一个句子包含了一些列事件，这就意味着可能有0或者多个事件类型标签。在机器学习中，这个问题被叫做: 多标签问题。 2. 触发词缺失问题： 之前的工作都表明触发词在事件检测中扮演着非常重要的角色。而现在的挑战是没有触发词。

为了解决第一个挑战，我们将多标签分类问题转换成多个二分类问题。 特别的, 句子 $s$ 以及 对应的事件类型 $t$ 作为一个实例，使用 $0$ 或 $1$ 来表示是否 $s$ 包含一个事件类型 $t$. 例如: 假设只有3个预先定义好的事件类型($t_1$, $t_2$, $t_3$)，那么，就转换成3个实例:

| instance | label |
|----------|-------|
| $<s, t_1>$ | 1 |
| $<s, t_2>$ | 0 |
| $<s, t_3>$ | 1 |

在这个图中，包含多个事件的句子变成了多个为1的pair, 因此多标签的问题被解决了。

**注: 是句子和事件类型 pair 作为一个样本来进行预测。**

而且, 每一个事件类型常常被一些列特定的词作为触发词。 例如: $Death$ 事件，常常由 "die", "passed away", "gone"等组成。因此，事件触发词对于这个任务来说是非常重要的。因为已存在的工作都表明标注的触发词能够直接作用于他们的模型。可是，在我们这个方式中，标注触发词是不可获得的。为了对这个信息建模，我们提出了简单但是有效的模型: "Type-aware Bias Neural Network with Attention Mechanisms"。 图1 展示了这个模型:


![ed_no_trigger_1.png](https://upload-images.jianshu.io/upload_images/1809271-d65e0f47d23055d8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


输入包含两部分: 以实体为token的句子 和 事件类型。 输出 $o$ 是 $1$ 如果该句子包含这个时间类型; 否则是0. 具体来说，给定一个句子这个模型首先将输入token进行embedding, 然后 应用 LSTM 层来计算每一个token的上下文依赖表示。然后基于目标事件类型, 它计算了一个attention向量 $\alpha$, 在这个过程希望触发词能够获得较高的分数。最终，通过 $\alpha$ 计算产生了 句子表达: $s_{att}$. 在这里, $s_{att}$ 被期望关注在触发词信息上，也就是 local information。 为了捕捉 global information, 最终的输出 $o$ 也会与 LSTM单元链接，(LSTM 将整个输入句子编码了)。此外, 为了强化正样本的影响，我们设计了一个有偏的目标函数。我们叫这个模型是"type-ware"， 是因为句子的表达, $s_att$ 是基于目标事件的类型来计算的。


我们使用了 被广泛使用的ACE2005作为数据集，作为实验比较。结果表明，我们的方法超越了比较的baseline, 甚至超过了哪些使用标注的触发词的方法。为了进一步学习，我们发布了我们[代码](https://github.com/liushulinle/event_detection_without_triggers)到NLP社区。

总结，我们工作的主要贡献:

1. 根据我们现在掌握的知识，这是第一个没有触发词的事件监测工作。相比于已经存在的方法，现在的方法需要更少的人工标注
2. 没有了触发词，这个任务会遭遇两大挑战: 多标签问题和触发词缺失问题。我们提出一种简单但是有效的模型，甚至达到了使用标注了触发词的方法。
3. 由于这是没有触发词的第一词事件检测的工作，我们实现了一些列baseline模型，并且系统的评估了他们。

## 背景

### 任务定义

Event detection 任务 需要在标注数据集中识别特定类型的事件。最常用的benchmark是ACE 2005语料集。 这份语料集包含了8个事件类型，33个子类型。我们简单的使用了33个分离的事件类型，忽略掉了类型的继承结构。考虑到下面的句子:

`In Bagh- dad, a cameraman died when an American tank fired on the Palestine Hotel`

一个理想的检测器应该能够识别出两个事件类型: $Die$ 和 $Attack$ 事件类型.

### 相关工作

事件检测是NLP中非常重要的一个topic. 很多方法都已经被实现来处理这个任务。几乎所有在ACE事件任务，都是使用的监督学习。我们进一步的设计了 feature-based 方法和 representation-based 方法。

在 feature-base方法中，很多方法被开发来产生特征向量。

最近一些年，representation-based方法开始主宰了这项研究。在这种范式下，候选事件的mention被embedding表示，然后被放到神经网络中。*Chen et al. (2015)* and *Nguyen and Grishman (2015)* 是第一个将这项工作使用这种范式的。他们的模型基于CNN。为了建模 trigger和argument之间的依赖, *Nguyen and Grishman (2016)* 提出了基于RNN的联合抽取方法。*Liu et al. (2017)* 提出了编码argument信息到event detection中，通过使用attention机制。最近,*Nguyen and Grishman (2018)* and *Sha et al. (2018)* 提出了利用语法信息来进行事件监测。

所有的已经存在的方法都依赖触发词的标注。训练数据昂贵的标注限制了这些方法的应用。为了减少人工成本，我们将这个任务设计为无触发词的方法。

## 方法论

为了解决多标签问题，我们将这个任务建模成多个二分类。给定一个句子，我们将该句子以及候选事件类型放入二分类中。我们增加了 *NA* label, 用来表示不包含任何事件类型的情况。为了补货隐含的触发词信息，我们使用了 "Type-aware Bias Neural Network with Attention Mechanisms"(TBNNAM)。 我们的模型是 "type-ware", 因为他计算句子的表示是基于目标事件类型。上图表明了 TBNNAM 框架。输入包含两部分: 基于实体的句子tokens和目标事件类型。输出$o$是1，如果给定的句子包含这种类型；否则是0.接下来，我们自底向上来描述这个模型架构。

### 输入tokens

给定一个句子，我们使用 Standford CoreNLP tool 将文本转换成tokens. ACE2005语料不仅仅标注了事件还标注了每个句子的实体。跟随前面的工作，我们使用标注的实体tag在我们的模型中 *(Li et al., 2013; Chen et al., 2015; Nguyen and Grishman, 2015, 2016; Liu et al., 2016b)*。

### Word/Entity Embeddings

Word Embeddings从大量的无label语料中学习。

在这个工作中，我们使用了 Skip-gram 来学习word embedding在NTY 语料上。 此外，我们随机初始化了一个实体embedding table为每一个实体tag. 所有的word token和entity tag将会被转换成低维向量通过查找embedding tables. 在这个工作中，我们使用 $d_w$来表示word embedding维度, $d_e$来表示entity embedding维度。

### Event Type Embeddings

如图1所示，事件类型被转换成两个embedding 向量: $t_1$ 和 $t_2$. $t_1$被设计成捕捉local 信息(触发词), 而 后一个$t_2$(红色的)被设计成捕捉全局信息。两个向量都是随机初始化。事件类型的维度是 $d_{evt}$。

### LSTM Layer

如图1所示， LSTM 是在word和entity concate之上的sequence的结果。

### Attention Layer

每一个事件类型常常是由一系列特定的词来触发的，这些词叫做触发词。例如, $Death$事件常常由 "die", "passed away", "gone"等词来处罚。因此，事件触发词是非常重要的任务。可是，这些信息对于我们的任务来说，是隐藏的，因为标注的触发词是不可获取的。为了建模隐藏的触发词，我们在我们的方法中使用attention机制。

如图1所示， attention vector $\alpha$ 基于事件类型 $t_1$和LSTM的隐藏状态 $h$来计算得出。 具体来说, 对于 $k$-th token的 attention的分数, 由如下计算得到:

$$
\alpha^k = \frac {exp(h_k \cdot t^{T}_{1})} {\sum_{i} {exp(h_i \cdot t^{T}_{1})}}
$$

在这个模型中，触发词希望得比其他词更高的分数。最后，句子的表示, $s_{att}$ 由下面的公式计算得到:

$$
s_{att} = \alpha^{T}H
$$

其中 $\alpha = [\alpha_{1}, ..., \alpha_{n}]$ 是attention vector, $H=[h_{1}, h_{2}, ..., h_{n}]$ 是一个矩阵， $h_k$是 LSTM的第 $k$-th token输出，而 $s_{att}$是给定句子的表示。

### Output Layer

如图1所示， 最终的 ouput $o$ 是由两部分组成: $v_att$ 和 $v_global$. 一方面, $v_att$ 是 $s_{att}$ 与 $t_1$的点乘结果，$v_att$ 被设计成捕获 local 特征（具体来说，就是隐藏的触发词的那些特征)。另一方面, LSTM层最后一个输出, $h_n$ 编码了整个句子的全局信息(global information), 因此 $v_{global} = h_{n} \cdot t^{T}_{2}$ 被期望来捕获句子的 global features. 最终，$o$ 被定义成 $v_att$ 与 $v_{global}$ 的权重和(weighted sum):

$$
o = \sigma( \lambda \cdot v_{att} + (1-\lambda) \cdot v_{global})
$$

### 运算推导(增加部分，论文没有)

上面虽然有一些公式和图，但是与最后的计算还有一点差距，同时，当看源代码的时候，作者进行了一点简化操作，所以看起来和论文描述的不太一样，下面的推导就是把这些澄清。

$$
h_t \in \mathbb{R}^{1 \times N},
H \in \mathbb{R}^{SeqLen \times N},
t_1 \in \mathbb{R}^{1 \times N}
\\
logits = H \times t^{T} = [h_1 \cdot t_1, h_2 \cdot t_1, ..., h_n \cdot t_1],logits \in \mathbb{R}^{SeqLen \times 1}
\\
\alpha_{k} = \frac {\exp(logits_k)} {\sum_{i=1}^{n} {\exp(logits_i)}} = \frac{\exp(h_k \cdot t_1)}{\sum_{i=1}^{n} {\exp(h_i \cdot t_1)}} , \alpha \in \mathbb{R}^{SeqLen \times 1}
\\
S_{att} = \sum_{i=1}^{n}{\alpha_i \times h_i} = \alpha^T \times H, S_{att} \in \mathbb{R}^{1 \times N}
\\
V_{att} = S_{att} \cdot t_1, V_{att} \in \mathbb{R}^{1 \times 1}
$$

将 $V_{att}$ 的计算过程整体来看，会发现一点技巧和简化，也就是论文的代码中实现的。注: $\alpha_i$ 是一个值，不是向量.

$$
V_{att} = S_{att} \cdot t_1 = (\sum_{i=1}^{n}{\alpha_i \times h_i}) \cdot t_1 = \sum_{i=1}^{n}{\alpha_i \times (h_i \cdot t_1)} = \sum_{i=1}^{n}{\alpha_i \times logits_i}
$$

$$
\therefore
\left\{\begin{matrix}
    V_{att} = \sum_{i=1}^{n}{\alpha_i \times logits_i}
    \\
    \alpha_{k} = \frac {\exp(logits_k)} {\sum_{i=1}^{n} {\exp(logits_i)}}
\end{matrix}\right.
$$

从上面的计算过程来看, $logits$ 第一次计算出来后，可以重复使用了。另外，通过 $V_{att}$的最后计算，知道当计算完 $\alpha$ 之后，就可以可以直接计算 $V_{att}$ 了，而不必将 $S_{att}$ 计算出来， 这也就是在实际的代码中，没有看到 $S_{att}$ 的计算，而是直接就计算了 $V_{att}$.



### Bias Loss Function (带偏置的损失函数)

我们设计了一个 "bias loss function" 来强化正样本的影响，因为如下原因:

1. 正样本比负样本要少很多。在我们的方法中，每一个训练样本是 `<句子, 事件类型>` 对， 标签是 $1$ 还是 $0$ 取决于该句子是否属于这个事件类型。 例如，我们一共定义了33中事件类型，如果一个句子仅仅包含一个事件，那么将会有32个负样本和1个正样本。大量的句子包含最多2个事件，因此负样本会比正样本多很多。
2. 正样本比负样本更具有信息性。一个正样本对 $<s, t>$ 意味着 $s$ 包含一个事件类型，而负样本对意味着 $s$ 不包含该事件类型。显然，前者更加具有信息性

给定所有的训练样本，数量 $T$, $(x^{(i)}, y^{(i)})$, loss function定义如下:

$$
J(\theta) = \frac{1}{T} \sum^T_{i=1}{(o(x^i) - y^{(i)})^2 (1 + y^{(i)}\cdot \beta) + \delta||\theta||}
$$

在这个公式中 $x$ 是包含一个句子和事件类型的pair, $y \in \{0, 1\}$, $\theta$ 是模型的参数， $\delta > 0$ 是 L2 normalization的权重. $1 + y^{(i)}\cdot \beta$ 是 偏置项 (bias term). 具体来说, 这一项的值对于负样本($y^{(i)}=0$) 来说是1, 而对于正样本来说($y^{(i)}=1$)是 $1+\beta$, 其中 $\beta \ge 0$。

**注: 这里并没有使用交叉熵损失, 而是使用欧拉距离损失**

### 训练

我们使用SGD在随机化的小批量上基于 Ada Rule来训练。 Regularization通过dropout和L2来实现。

给定实例 $x$, 模型预测的label $\tilde{y}$ 计算如下:

$$
\tilde{y} = \left\{\begin{matrix}
0 \qquad o(x) < 0.5
\\
1 \qquad otherwise
\end{matrix}\right.
$$

$x$ 是 pari $<s, t>$, $o(x)$ 是模型对$x$的输出，$\tilde{y}$ 是最终的预测结果。

## Baseline System

因为这是第一份没有trigger的事件检测工作，我们实现了一些列与baseline system的对比结果。这种对比分成两类: 基于这种方法的二分类与多分类。

### Binary Classifcation


就像我们提出的方法一样，baseline system 通过二分类来解决这个任务。图2显示了这种方法的framework. 这些模型将一个句子和事件类型作为输入。然后，所有的输入转换成embedding。这些模型有和我们提出的方法一样的 loss function. 这些模型的核心组件是句子encoder. 根据不同的 encoder 策略，我们实现了三个比较模型: $BC-CNN$, $BC-LSTM_{last}$, $BC-LSTM_{avg}$ （$BC$ 的意思是: "Binary Classifcation").

* $BC-CNN$ 使用了CNN来编码句子。
* $BC-LSTM_{last}$ 使用了 LSTM 模型，并使用了最后token的hidden state作为句子的表示。
* $BC-LSTM_{avg}$ 使用了 LSTM 模型, 但是使用了所有 hidden state 平均值，作为句子的表示。

![ed_no_trigger_2.png](https://upload-images.jianshu.io/upload_images/1809271-e04c69174a3af7d8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### Multi-class Classification

所有的现存方法建模事件监测（带有triggers)都是使用的 mulit-class classification. 给定一个句子，这些方法预测每一个token是否是事件触发词以及事件类型。我们也实现了多个 multi-class classification 用来比较。因为标注的trigger在我们的任务中是不可以获得的，所以句子就是我们模型的输入。图3显示了这些模型的framework. 依据现有的工作 （Chen
et al., 2015; Liu et al., 2017), 我们使用了 negative log-lokelihood loss 来作为softmax 分类器:

$$
J(\theta) = -\frac{1}{T} \sum^T_{i=1} log(p(y^{(i)}|x^{(i)}), \theta)
$$

其中 $(x^{(i)}, y^{(i)})$ 是一个训练样本  $y^{(i)}$ 是所有可获得label （所有需要预测事件类型以及 无类型的 NA), $T$ 是全部的训练实例数量，$\theta$ 是模型参数。 根据现有的编码句子的策略，我们实现了三个模型: $MC-CNN$, $MC-LSTM_{last}$, $MC-LSTM_{avg}$.

* $MC-CNN$ 使用了CNN来编码句子。
* $MC-LSTM_{last}$ 使用了 LSTM 模型，并使用了最后token的hidden state作为句子的表示。
* $MC-LSTM_{avg}$ 使用了 LSTM 模型, 但是使用了所有 hidden state 平均值，作为句子的表示。

![ed_no_trigger_3.png](https://upload-images.jianshu.io/upload_images/1809271-ed180e8561b421c3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## 试验结果

### 试验设置
在这个章节，我们介绍数据集，评估指标以及超参设置。

#### 数据集

我们试验的数据集是 ACE 2005数据集。根据先前的评估工作 (Li et al., 2013; Chen et al., 2015; Nguyen and Grishman, 2016; Liu et al., 2017), 我们随机从不同体裁选取30篇文章作为 development set, 然后使用独立的40篇新闻文档作为测试集。我们使用剩余的529篇作为训练集。

这个工作集中在没有触发词的情况下检测事件。因此，我们从语料中删除了触发词的标注。具体来说，我们使用 Standford CoreNLP Toolkit 将每一个文档切分成句子，然后根据ACE 2005语料的原始标注，给每一个句子设置label。如果一个句子不包含任何事件，我们会赋值给他一个特殊的label, $NA$. 如果一个句子包含多个相同类型的事件(少于3%在ACE语料集中)，我们仅仅保留一个标签。下表展示了语料中的样本。

| sentence | labels|
|----------|-------|
|They got married in 1985. | $\{Marry\}$ |
|They got married in 1985, and divorced 3 years latter. | $\{Marry, Divorce\}$ |
| They are very happy every day. | $\{NA\}$ |

*Table 2: 语料集中没有trigger的标注样本样例*

#### 评估指标

依据前面的工作 (Liao and Grishman, 2010; Li et al., 2013; Chen et al., 2015; Liu et al., 2017), 我们使用 percision (**P**), recall (**R**) 和 F1 measure (**F1**) 来评估结果。

* **Precision**: 正确预测的事件在所有预测的事件中的比例
* **Recall**: 正确预测的事件在数据集中所有事件的比例
* **F1-measure**: $\frac{2 \times P \times R}{P + R}$

#### 超参数

超参是通过grid search来调节的。在所有的试验中，我们设置 word embeddings 维度是200， 实体类型维度是50, batch size是100， L2 正则化 超参数是 $10^{-5}$, $\beta$ 是 $1.0$。 另外，我们也在开发集中调节了在等式3中的 $\lambda$。图4显示了使用不同的 $\lambda$ 设置的试验结果，最后我们设置 $\lambda$ 为 $0.25$. 在所有的 CNN-based 的 Baseline System中， filter window size 被设置为 $1$, $2$, $3$， feature map 是 $100$.

![ed_no_trigger_4.png](https://upload-images.jianshu.io/upload_images/1809271-325050899844ae27.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### Multi-class Classifcation vs. Binary Classification

Talbe 3 说明了这个试验结果，带有 **MC-\***是基于 multi-class 分类的结果，而带有 **BC-\***是基于 binary classfication的结果。根据不同的编码句子的方法，在Table 3 中被分组成3个部分来展示。从这个tabel中，我们能够看到以下的观察的结论:

* 在每一个分组中, binary classifcation 能够极大的超过 multi-class classfication. 原因是 **BC-\***能够解决多标签问题，但是 **MC-\***不能。并且，
**MC-\*** 的召回率也比 **BC-\*** 的低，因为他们为每一个句子仅仅预测一个事件。

![ed_no_trigger_table_3.png](https://upload-images.jianshu.io/upload_images/1809271-8fb1b6f5b20e97ef.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### 整体表现

![ed_no_trigger_table_4.png](https://upload-images.jianshu.io/upload_images/1809271-59d1ca25120ee924.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


在这个章节中，我们说明这个方法结果 (参考 Table 4). Baseline System的结果被列在第一组中. 第二组是我们提出的方法。他们都有相同的结构，参考 图1。在 $BC-LSTM_{att}$, $\lambda$ (参考 Equation 3) 被设置成了1.0, 这种方式用来设计成显示attention的策略结果。在 **TBNNAM**，$\lambda$ 被设置成 $0.25$，这种设计使用了local information (attention 机制) 和 global information (LSTM的最后token输出). 在最后一个group中是 ED 系统 在 ACE 2005 数据集上 state-of-the-art 结果. 我们给出他们一些剪短的介绍:

1. Nguyen’s CNN: CNN 模型 由 Nguyen and Grishman (2015) 提出
2. Chen’s DMCNN: the dynamic multi-pooling CNN model proposed by Chen et al. (2015)
3. Liu’s PSL: the soft probabilistic soft logic model proposed by Liu et al. (2016b)
4. DS-DMCNN : the DMCNN model augmented with automatic labeled data, proposed by Chen et al. (2017)

从这个表4来看，我们有如下的观察结果:

* $BC-LSTM_{att}$ 超过了所有 Baseline System，并且有着显著的收益，这表明了提出的attention机制是有效的。
* $TBNNAM$ 比 $BC-LSTM_{att}$ 更好的性能 ($69.9\%$ vs. $66.3\%$), 这表明LSTM 捕获的 last state global information 对这个任务来说是很重要的。 因此 global infromation 和 local information 是相互补充的。
* 所有的 state-of-the-art ED 系统都需要 trigger 标注。没有trigger标注的情况下，我们的方法达到了非常有竞争力的结果，甚至超过了他们。

### $\alpha$ 权重分析

图5展示了一些模型学到的 attention vector $\alpha$ 的例子。在第一个例子中, "die" 对于 *Death* 事件来说是最重要的词汇，而我们的模型成功的捕获了这个特征，并且赋予它较大的 attention 分数。同样的，在第二个例子中, "fired" 是 *Attack* 事件的关键线索, 而我们的模型也学习到了，并且给了较大的 attention 分数。 实际上, "died" 和 "fired" 是 *Death* 和 *Attack* 事件的触发词。 因此，我们可以说，尽管触发词没有被标注，但是我们的模型人就能够使用触发词的信息。而且，我们方法也能够对不同事件之间彼此的依赖建模，这已经被表明对这个任务来说是有用的 (Liao and Grishman, 2010; Liu et al., 2016b)。例如, *Attack* 事件常常伴随着 *Death* 事件。 在 Case1 和 Case2 (图5), 我们的方法模型关注在 "died" 和 "fired"单词上。 另外，第3个case是一个负样本, 里面没有任何关键词的线索。我们的模型赋予每一个token几乎一样的attention score.

### Bias Term 在 Loss Function 中的影响

在这个部分，我们说明了 bias term 在 Equation 4 中的影响。 Table 5 展示了试验结果. 名字带有 **\*\Bias** 没有使用 bias term. 从这个试验表格中，我们观察到带有 bias term 的 loss function 比没有 bias term 有较大的提升。这也证明了在 3.7节中正样本应该加强的分析的正确性。


![ed_no_trigger_table_5.png](https://upload-images.jianshu.io/upload_images/1809271-402bcb04016fc53e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


## 结论

现存的事件监测的方法需要标注触发词，由于这种昂贵的标注限制了这种方法的使用。为了减少人工成本，我们调研了没有事件触发词的实现方法。在这个背景下, ED 任务遇到两个挑战: 多标签问题 和 触发词缺失问题。 我们提出一种简单有效的模型来解决他们, 这种方法计算伴随事件类型的句子表示。事件结果表明这是有效的。出乎意料的是，这种方法甚至与那些带有触发词标注的 state-of-the-arts 方法达到了相同的结果。

## Reference

* David Ahn. 2006. The stages of event extraction. In Proceedings of the Workshop on Annotating and Reasoning about Time and Events, pages 1–8. Association for Computational Linguistics.
* Yoshua Bengio, Re ́jean Ducharme, Pascal Vincent, and Christian Janvin. 2003. A neural probabilistic language model. The Journal of Machine Learning Research, 3:1137–1155.
* Yubo Chen, Shulin Liu, Xiang Zhang, Kang Liu, and Jun Zhao. 2017. Automatically labeled data generation for large scale event extraction. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 409–419, Vancouver, Canada.
* Yubo Chen, Liheng Xu, Kang Liu, Daojian Zeng, and Jun Zhao. 2015. Event extraction via dynam- ic multi-pooling convolutional neural networks. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics, pages 167–176. Association for Computational Linguistics.
* Dumitru Erhan, Yoshua Bengio, Aaron Courville, Pierre-Antoine Manzagol, Pascal Vincent, and Samy Bengio. 2010. Why does unsupervised pre-training help deep learning? The Journal of Machine Learning Research, 11:625–660.
* Prashant Gupta and Heng Ji. 2009. Predicting unknown time arguments based on cross-event prop- agation. In Proceedings of the ACL-IJCNLP 2009 Conference Short Papers, pages 369–372. Associa- tion for Computational Linguistics.
* Yu Hong, Jianfeng Zhang, Bin Ma, Jianmin Yao, Guodong Zhou, and Qiaoming Zhu. 2011. Using cross-entity inference to improve event extraction. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 1127–1136. Associ- ation for Computational Linguistics.
Heng Ji and Ralph Grishman. 2008. Refining event extraction through cross-document inference. In Proceedings of ACL-08: HLT, pages 254–262. As- sociation for Computational Linguistics.
* Qi Li, Heng Ji, and Liang Huang. 2013. Joint event extraction via structured prediction with global features. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 73–82.
* Shasha Liao and Ralph Grishman. 2010. Using document level cross-event inference to improve event extraction. In Proceedings of the 48th Annual Meet- ing of the Association for Computational Linguistic- s, pages 789–797.
* Shulin Liu, Yubo Chen, Shizhu He, Kang Liu, and Jun Zhao. 2016a. Leveraging framenet to improve au- tomatic event detection. In Proceedings of the 54th Annual Meeting of the Association for Computation- al Linguistics, volume 1, pages 2134–2143. Associ- ation for Computational Linguistics.
* Shulin Liu, Yubo Chen, Kang Liu, and Jun Zhao. 2017.
Exploiting argument information to improve event detection via supervised attention mechanisms. In Proceedings of the 55th Annual Meeting of the Asso- ciation for Computational Linguistics, pages 1789– 1798, Vancouver, Canada. Association for Compu- tational Linguistics.
* Shulin Liu, Kang Liu, Shizhu He, and Jun Zhao. 2016b.
A probabilistic soft logic based approach to exploit- ing latent and global information in event classifi- cation. In Proceedings of the thirtieth AAAI Conference on Artificail Intelligence, pages 2993–2999. Association for Computational Linguistics.
* Christopher D. Manning, Mihai Surdeanu, John Bauer, Jenny Finkel, Steven J. Bethard, and David Mc- Closky. 2014. The Stanford CoreNLP natural lan- guage processing toolkit. In Association for Compu- tational Linguistics (ACL) System Demonstrations, pages 55–60.
* Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient estimation of word representations in vector space. arXiv preprint arX-iv:1301.3781.
* George Miller. 1998. WordNet: An electronic lexical database. MIT press.
* Huu Thien Nguyen and Ralph Grishman. 2015. Event detection and domain adaptation with convolutional neural networks. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics, pages 365–371. Association for Computational Linguistics.
* Huu Thien Nguyen and Ralph Grishman. 2016. Modeling skip-grams for event detection with convolutional neural networks. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language * Processing, pages 886–891. Association for Computational Linguistics.
* Thien Huu Nguyen and Ralph Grishman. 2018. Graph convolutional networks with argument-aware pooling for event detection.
* Lei Sha, Feng Qian, Baobao Chang, and Zhifang Sui. 2018. Jointly extracting event triggers and arguments by dependency-bridge rnn and tensor-based argument interaction. In AAAI.
* Richard Socher, Brody Huval, Christopher D Manning, and Andrew Y Ng. 2012. Semantic compositionality through recursive matrix-vector spaces. In Pro- ceedings of the 2012 joint conference on empirical methods in natural language processing, pages 1201–1211. Association for Computational Linguistics.
* David Yarowsky. 1995. Unsupervised word sense disambiguation rivaling supervised methods. In 33rd Annual Meeting of the Association for Computation- al Linguistics. Association for Computational Lin- guistics.
* Matthew D Zeiler. 2012. Adadelta: An adaptive learning rate method. arXiv preprint arXiv:1212.5701.
* Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, and Jun Zhao. 2014. Relation classification via convolutional deep neural network. In Proceedings of COLING 2014, pages 2335–2344.





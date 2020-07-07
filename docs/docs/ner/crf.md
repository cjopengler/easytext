# CRF

CRF 公式如下:

$$
p(y|x) = \frac {p(x,y)}{p(x)} = \frac {p(x,y)}{\sum_{i \in Path} p(x, y_i)}
$$

其中 $p(x,y)$ 得形式化定义如下:

$$
p(x,y) = \frac{exp(W\phi (x, y))} {R}
$$

$R$ 不需要知道实际表达，只是用来保证 $p(x,y) \in [0, 1]$，因为在 $p(x|y)$ 中分子和分母都有 $R$ 自然就消掉了。

$\phi(x,y)$ 表示的由 $x, y$ 构成的特征函数，从这个角度来看 $W\phi (x, y))$ 就是 线性计算，对每一个特征有一个权重，这就是非常熟悉的意思了。

所以对于 $\phi(x,y)$, 可以用常规方法来涉及特征函数，一共有多少个特征函数呢？

1. 每一个$<{x_i, y_j}>$ 作为一个特征函数，有 $vocabulary\_size \times label\_size$ - 被称作 发射矩阵 Emission matrix, 
2. $<{y_i, y_j}>$ 作为特征函数，有 $label\_size*label\_size$ 个 - 被称作转移矩阵 Transition matrix。

举个例子来说，比如词性标注。 

"我 喜欢 中国". $\phi(x,y)$ 可以设计成:

1. $\phi(我, N)$ = 0.3
2. $\phi(我, V)$ = 0.2
3. $\phi(喜欢, N)$ = 0.0
4. $\phi(喜欢, V)$ = 0.8
5. $\phi(中国, N)$ = 0.9
6. $\phi(中国, V)$ = 0.1
7. $\phi(N, V)$ = 0.8
8. $\phi(N, N)$ = 0.1
9. $\phi(V, N)$ = 0.7
10. $\phi(V, V)$ = 0.3

特征函数设计完就可以进行训练的。这些特征函数的值，可以通过训练预料中统计得到。从上面，也可以看出 $1$ 到 $6$ 是发射矩阵，$7$到$8$ 是转移矩阵。

## 基于深度学习来自动学习 $\phi(x,y)$ 

上面介绍的是传统的设计方式，人为定义和统计定义$\phi(x,y)$, 在 “Neural Architectures for Named Entity Recognition” paper中，基于深度学习，将 $\phi(x,y)$ 作为学习的对象，来求解。也就是 自动学习 Emission matrix 和 Transition matrix。

对于一条路径上的 $Score$ 其实也就是 $W\phi(x,y)$定义如下:

$$
W\phi(x,y) = Score(X, y) = \sum_{i=1}^{n}{E(x_i, y_i)} + \sum_{i=0}^{n}{T(y_{i}, y_{i+1})}
$$

其中 $X$ 是 输入序列, $y$ 是输出序列:

* $X = [x_1, x_2, ..., x_n]$
* $y = [y_1, y_2, ..., y_n]$

其中: $X_0=<BOS>, X_{n+1}=<EOS>$

这里特别说明 $<BOS>, <EOS>$，这两个特殊标志是为了方便计算"转移矩阵", 那么，这两个标志自动加进去参与"转移矩阵"计算即可，不需要在 $X$ 以及 $y$ 序列中添加。


所以

$$
p(x,y) = exp(S(X,y))
$$

那么:

$$
p(y|x) = \frac {p(x,y)}{p(x)} = \frac {p(x,y)}{\sum_{i \in Path} p(x, y_i)}
$$

从另外一个角度来看 $p(y|x)$, $p(x,y)$ 表示的是 gloden path 的概率, $\sum_{i \in Path} p(x, y_i)$ 所有路径相加的总和，$p(y|x)$就是求使得 gloden path 的在全部路径占比最大化的过程。

为了方便计算，两边取对数: 

$$
log(p(y|x)) = log(p(x,y)) - log(\sum_{i \in Path} p(x, y_i))
$$

所以 CRF loss 是最小化过程, 那么就是 
$$
Loss = -log(p(y|x)) = -(log(p(x,y)) - log(\sum_{i \in Path} p(x, y_i))) = -Score(x,y) + log(\sum_{i \in Path} p(x, y_i)))
$$

下面通过公式推导来逐个计算。

## 公式推导

![sequece label](../images/ner/ner_seq.png)

### $p(x,y)$ 联合概率计算 

先计算: $p(x,y) = Score(x,y)$

$$
Score(X, y) = \sum_{i=1}^{n}{E(w_i, y_i)} + \sum_{i=0}^{n}{T(y_{i}, y_{i+1})} = \sum_{i=0,j=index(y_i)}^{n}{x_{ij}} + \sum_{k=0, i=index(y_{ki}), j=index(y_{k+1j})}^{n}{t_{ij}}
$$

写成下面的递推式，方便在程序中实现:
$$
Score_{j} = Score_{i+1} = Score_i + E(w_{(i+1)}, y_{i+1}) + T(y_{i}, y_{i+1})
$$

### $p(x)$ 边缘概率计算

$$
p(x) = \sum_{y_i \in Path}(p(x,y_i))
$$

如上图所示，为了方便演示，$label$ 数量一共是2个， 令 $w_i$ 上的分数分别是 $S_{i1}$ 和 $S_{i2}$。那么，$Total_i$ 可以计算为:

$$
Total_i = log(e^{S_{i1}} + e^{S_{i2}})
$$

现在，令 $j=i+1$, 需要推导在 $w_j$ 上的 $Total_j$ 如何使用  $S_{i1}$ 和 $S_{i2}$ 进行表示。

$$
Total_j 
        \\ = log((e^{S_{i1} + x_{j1} + t_{11}} + 
                 e^{S_{i2} + x_{j1} + t_{21}}) +
                 (e^{S_{i1} + x_{j2} + t_{12}} +
                 e^{S_{i2} + x_{j2} + t_{22}}))
        \\ = log(e^{S_{j1}} + e^{S_{j2}})
$$

所以有:

$$
e^{S_{j1}} = e^{S_{i1} + x_{j1} + t_{11}} + e^{S_{i2} + x_{j1}+ t_{21}}
\\ \Rightarrow 
S_{j1} = log(e^{S_{i1} + x_{j1} + t_{11}} + e^{S_{i2} + x_{j1}+ t_{21}})
$$

同理:

$$
S_{j2} = log(e^{S_{i1} + x_{j2} + t_{12}} + e^{S_{i2} + x_{j2} + t_{22}})
$$

到这里，其实从数学角度已经说明完毕，因为已经有了:

$$
\begin{bmatrix}
S_{i1} \\ 
S_{i2}
\end{bmatrix}
\Rightarrow
\begin{bmatrix}
S_{j1} \\ 
S_{j2}
\end{bmatrix}
$$

这样的递推表达式，就可以从 $S_0$  一直推到到 $S_n$, 也就是有了 $Total_n$

上面的式子在计算的时候依然不太方便，变换下计算过程:

$$
T =
\begin{bmatrix}
S_{i1} & S_{i1}\\ 
S_{i2} & S_{i2}
\end{bmatrix}
+
\begin{bmatrix}
x_{j1} & x_{j2}\\ 
x_{j1} & x_{j2}
\end{bmatrix}
+
\begin{bmatrix}
t_{11} & t_{12}\\ 
t_{21} & t_{22}
\end{bmatrix}
=
\begin{bmatrix}
S_{i1} + x_{j1} + t_{11}& S_{i1} + x_{j2} + t_{12}\\ 
S_{i2} + x_{j1} + t_{21}& S_{i2} + x_{j2} + t_{22} 
\end{bmatrix}
$$

$$
\begin{bmatrix}
S_{j1} \\ 
S_{j2} 
\end{bmatrix}
= log(sum(exp(T), dim=0)) 
= \begin{bmatrix}
log(e^{S_{i1} + x_{j1} + t_{11}} + e^{S_{i2} + x_{j1}+ t_{21}}) \\ 
log(e^{S_{i1} + x_{j2} + t_{12}} + e^{S_{i2} + x_{j2} + t_{22}})
\end{bmatrix}
$$

此时的 

$$
Total_j = log(e^{S_{j1}} + e^{S_{j2}}) = 
    log((e^{S_{i1} + x_{j1} + t_{11}} + 
         e^{S_{i2} + x_{j1} + t_{21}}) +
        (e^{S_{i1} + x_{j2} + t_{12}} +
         e^{S_{i2} + x_{j2} + t_{22}}))
$$

所以计算 $T$ 的过程是很容易用代码矩阵运算的。

## 具体例子验证

当 $j=0$ 时:
$$
S_0 = \begin{bmatrix}
S_{01} \\ 
S_{02} 
\end{bmatrix}
=
\begin{bmatrix}
x_{01} \\ 
x_{02} 
\end{bmatrix}
$$

当 $i=0, j=1$ 时:
$$
S1 = 
\begin{bmatrix}
S_{11} \\ 
S_{12} 
\end{bmatrix}
= 
\begin{bmatrix}
log(e^{x_{01} + x_{11} + t_{11}} + e^{x_{02} + x_{11}+ t_{21}}) \\ 
log(e^{x_{01} + x_{12} + t_{12}} + e^{x_{02} + x_{12} + t_{22}})
\end{bmatrix}
$$

$$
S2 = 
\begin{bmatrix}
S_{21} \\ 
S_{22} 
\end{bmatrix} 
= 
\begin{bmatrix}
log(e^{S_{11} + x_{21} + t_{11}} + e^{S_{12} + x_{21}+ t_{21}}) \\ 
log(e^{S_{11} + x_{22} + t_{12}} + e^{S_{12} + x_{22} + t_{22}})
\end{bmatrix}
=
\begin{bmatrix}
log(exp(S_{11})\cdot exp(x_{21} + t_{11}) + exp(S_{12}) \cdot exp(x_{21}+ t_{21})) \\ 
log(exp(S_{11}) \cdot exp(x_{22} + t_{12}) + exp(S_{12}) \cdot exp(x_{22} + t_{22}))
\end{bmatrix}
\\ =
\begin{bmatrix}
log((e^{x_{01} + x_{11} + t_{11}} + e^{x_{02} + x_{11}+ t_{21}}) \cdot exp(x_{21} + t_{11}) 
+ (e^{x_{01} + x_{12} + t_{12}} + e^{x_{02} + x_{12} + t_{22}}) \cdot exp(x_{21}+ t_{21})) \\ 
log((e^{x_{01} + x_{11} + t_{11}} + e^{x_{02} + x_{11}+ t_{21}}) \cdot exp(x_{22} + t_{12}) 
+ (e^{x_{01} + x_{12} + t_{12}} + e^{x_{02} + x_{12} + t_{22}}) \cdot exp(x_{22} + t_{22}))
\end{bmatrix}
\\ =
\begin{bmatrix}
log(exp(x_{01} + x_{11} + t_{11} + x_{21} + t_{11}) 
  + exp(x_{02} + x_{11}+ t_{21} + x_{21} + t_{11})  
  + exp(x_{01} + x_{12} + t_{12} + x_{21}+ t_{21} 
  + exp(x_{02} + x_{12} + t_{22} + x_{21}+ t_{21}) \\
log(exp(x_{01} + x_{11} + t_{11} + x_{22} + t_{12}) 
  + exp(x_{02} + x_{11}+ t_{21} + x_{22} + t_{12}) 
  + exp(x_{01} + x_{12} + t_{12} + x_{22} + t_{22} 
  + exp(x_{02} + x_{12} + t_{22}) + x_{22} + t_{22})
\end{bmatrix}
$$

此时的

$$
Total_2 = log(exp(S_{21}) + exp(S_{22}))
= 
\\   exp(x_{01} + x_{11} + t_{11} + x_{21} + t_{11}) 
\\ + exp(x_{02} + x_{11}+ t_{21} + x_{21} + t_{11})  
\\ + exp(x_{01} + x_{12} + t_{12} + x_{21}+ t_{21} 
\\ + exp(x_{02} + x_{12} + t_{22} + x_{21}+ t_{21} 
\\ + exp(x_{01} + x_{11} + t_{11} + x_{22} + t_{12}) 
\\ + exp(x_{02} + x_{11}+ t_{21} + x_{22} + t_{12}) 
\\ + exp(x_{01} + x_{12} + t_{12} + x_{22} + t_{22} 
\\ + exp(x_{02} + x_{12} + t_{22}) + x_{22} + t_{22}
$$

正好是8条路径。

## Viterbi 解码

在使用 CRF 训练完模型，进行 inference 的时候，需要将 best path 挑选出来，也就是 $Score$ 最大的那条路径。这里使用了 Viterbi 算法。简单介绍 Viterbi 算法。

从 $w_i$ 到 $w_j = w_{i+1}$，$y$ 的最大路径，也就是 $Score$ 最大的路径。

依然向上面的推导一样使用只有两个 label 的 y 作为推导。现在推导在第 $i$ 步 与 第 $j$ 步的分数关系。

$$
max(S_j1) = max( max(S_{i1}) + x_{j1} + t_{11}, max(S_{i2} + x_{j1} + t_{21})) = max\_index, max\_value
$$

其中：

* $max\_index$ - 表示前驱结点的index, 要么是 1 要么是 2， 实际编程时候应该是 0 和 1.
* $max\_value$ - 就是计算的值，也会作为 max(S_j1) 的值。


同理，也可以计算出 $max(S_{j2})$，并将 $max\_index, max\_value$ 保存好。

将 $max\_index$ 保存起来，这实际是一个 2维矩阵 IndexMatrix，$shape = (n, 2)$.

那么，最终的 

$$
BestScore = max(max(S_j1), max(S_j2)) =  max\_index, max\_value $$

在 $BestScore$ 中返回的 $max\_index$ 从 IndexMatrix 中提取出来Best Path，也就是 $IndexMatrix[:, max\_index]$。

知乎这篇文章用具体的例子说的很清楚:

[Viterbi 例子](https://www.zhihu.com/question/20136144/answer/763021768)





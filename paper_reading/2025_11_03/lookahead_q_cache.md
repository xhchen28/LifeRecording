# Lookahead_Q_Cache(LAQ)
EMNLP 25
简单来说就是通过一些 pseudo lookahead queries 充当观察窗口

一切始于观察，作者观察到，P和D 阶段的不一致性，即观察选择不同position 的cache，计算对应的召回率。不仅仅是P阶段的Cache。还与D阶段的Cache 相关。

![alt text](./images/recall_figure.png)

从图中看到，召回率逐渐升高，随着index 向前推进。同时，生成初期的几个token，召回率相当高。
# Word2Vec 论文学习文档

## 1. 引言

自然语言处理（Natural Language Processing, NLP）的核心挑战之一是如何将离散的词汇符号转化为计算机可处理的数值表示。传统的独热编码（One-hot Encoding）将每个词表示为一个高维稀疏向量，维度等于词汇表大小 V。这种方式不仅造成维度灾难，而且无法捕捉词语之间的语义关系——任何两个不同词的独热向量都是正交的，其内积为零，无法表达“国王”与“王后”之间的类比关系。

2013 年，Tomas Mikolov 等人在论文《Efficient Estimation of Word Representations in Vector Space》中提出了 Word2Vec 模型，通过自监督学习将词汇映射到低维稠密的向量空间（通常 100–300 维），使得语义相似的词在向量空间中距离相近，并且词向量能够通过简单的向量运算揭示句法和语义规律（例如 `vec("国王") - vec("男人") + vec("女人") ≈ vec("王后")`）。该论文以及同年发布的后续工作《Distributed Representations of Words and Phrases and their Compositionality》共同奠定了现代词嵌入（Word Embedding）研究的基础。

本文档旨在为读者提供一份详细、结构化的 Word2Vec 论文学习指南。我们将深入剖析其提出的两种模型架构——连续词袋模型（CBOW）和跳元模型（Skip-gram），详解用于加速训练的层次 Softmax 和负采样技术，并结合论文实验分析其性质。全文以中文撰写，关键学术术语保留英文，所有数学符号采用 LaTeX 格式以保持严谨性。

---

## 2. 背景：语言模型与词表示

### 2.1 统计语言模型

在 Word2Vec 提出之前，统计语言模型（Statistical Language Model）已经为词的分布式表示奠定了基础。一个经典的目标是计算一个词序列的概率：

$$
P(w_1, w_2, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \dots, w_{t-1})
$$

早期的 N-gram 模型使用最大似然估计直接统计词频，但面临严重的数据稀疏问题。为了克服维度灾难，基于前馈神经网络的语言模型（Neural Probabilistic Language Model, 2003 年由 Bengio 等人提出）将每个词映射为一个低维稠密向量，并输入神经网络来估计条件概率：

$$
P(w_t \mid w_1, \dots, w_{t-1}) \approx f(w_t, C(w_{t-1}), \dots, C(w_{t-n+1}); \theta)
$$

其中 $C$ 是词向量查找表。该模型在训练语言模型的同时，顺带学到了高质量的词向量，这证明了词向量可以作为语言模型的“副产品”。然而，其训练代价巨大，主要瓶颈在于输出层需要在整个词汇表上计算 Softmax 归一化：

$$
P(w_t \mid \text{context}) = \frac{\exp(\text{score}(w_t, \text{context}))}{\sum_{w \in \mathcal{V}} \exp(\text{score}(w, \text{context}))}
$$

当词汇表 \(\mathcal{V}\) 达到数十万甚至百万级别时，分母的计算变得不可接受。

### 2.2 对高效训练的需求

Word2Vec 的动机正是为了解决上述计算瓶颈，同时保持甚至提升词向量的质量。Mikolov 等人意识到，学习词向量的过程不一定需要完整的概率语言模型框架，我们可以专注于训练词向量本身。因此，他们设计了一种简化的架构：通过去除隐藏层中的非线性激活函数，并采用特殊的输出层结构（层次 Softmax 或负采样）来大幅降低计算复杂度。这种简化使得模型可以在大规模语料（数十亿词）上以极快的速度完成训练。

---

## 3. Word2Vec 概览

Word2Vec 提供了两种架构来学习词向量，它们都基于一个简单的思想：用中心词预测上下文词，或用上下文词预测中心词。训练的目标是让词向量内积（或点积）最大化实际出现的词对概率，同时最小化未出现的词对概率。

论文提出的核心模型包括：

1. **Continuous Bag-of-Words (CBOW)**：根据上下文（目标词的周边词）预测目标词本身。
2. **Continuous Skip-gram (Skip-gram)**：根据目标词预测其上下文窗口内的各个词。

两者互为“镜像”，并在不同任务上各有优劣。CBOW 将多个上下文词向量投影到同一位置（求平均），训练速度更快，对小规模语料鲁棒；Skip-gram 对每个上下文词单独预测，能更好地学习罕见词的表示，在大规模语料上通常表现更优。

---

## 4. CBOW 模型

### 4.1 模型结构

CBOW（Continuous Bag-of-Words）模型的目标是利用一个大小为 \(m\) 的上下文窗口内的词来预测中心词。对于输入序列中的中心词 \(w_t\)，其上下文为：

$$
\text{context}(w_t) = \{ w_{t-m}, \dots, w_{t-1}, w_{t+1}, \dots, w_{t+m} \}
$$

模型结构极其简洁，仅包含三个层级（无隐藏层非线性激活）：

- **输入层**：上下文窗口中每个词 \(w_i\) 的独热向量，维度 \(V\)。
- **投影层（Projection Layer）**：将每个词的独热向量映射为对应的词向量（维度 \(d\)），然后对所有上下文的词向量**求平均**（或求和，论文使用平均）。该层是线性的。
- **输出层**：输出一个维度为 \(V\) 的概率分布，表示词汇表中每个词作为中心词的可能性。

数学上，对于上下文词 \(w_{t+i}\)，其独热向量为 \(\mathbf{x}_{t+i}\)，词向量矩阵为 \(\mathbf{W} \in \mathbb{R}^{V \times d}\)（输入矩阵），投影层向量 \(\mathbf{h}\) 为：

$$
\mathbf{h} = \frac{1}{2m} \sum_{-m \leq i \leq m, i \neq 0} \mathbf{W}^\top \mathbf{x}_{t+i} = \frac{1}{2m} \sum_{\text{context}} \mathbf{v}_{w_i}
$$

其中 \(\mathbf{v}_{w_i}\) 是词 \(w_i\) 的输入向量。输出层采用 Softmax 计算中心词 \(w_t\) 的概率：

$$
P(w_t \mid \text{context}) = \frac{\exp(\mathbf{u}_{w_t}^\top \mathbf{h})}{\sum_{w \in \mathcal{V}} \exp(\mathbf{u}_{w}^\top \mathbf{h})}
$$

这里 \(\mathbf{u}_{w}\) 是词 \(w\) 的输出向量，矩阵记为 \(\mathbf{W}' \in \mathbb{R}^{d \times V}\)。

### 4.2 训练目标

CBOW 的训练目标是在给定上下文下最大化正确中心词的对数似然：

$$
\mathcal{L} = \sum_{t=1}^{T} \log P(w_t \mid \text{context}(w_t))
$$

通过随机梯度下降（SGD）对 \(\mathbf{W}\) 和 \(\mathbf{W}'\) 进行优化。值得注意的是，每个词最终会有两套向量：输入向量 \(\mathbf{v}_w\) 和输出向量 \(\mathbf{u}_w\)。实践中通常取 \(\mathbf{v}_w\) 或两者之和/平均作为最终的词向量。

### 4.3 CBOW 的特点

- **连续词袋**：名称来源于上下文的词序完全被忽略（类似于词袋模型），仅将向量平均。这使得模型对于局部语序不敏感，倾向于学习词义的概括性表示。
- **训练效率高**：相比 Skip-gram，CBOW 一次只预测一个中心词，而 Skip-gram 在一次迭代中需要对窗口内每个上下文词分别进行预测。因此在同等硬件下 CBOW 训练更快。
- **对高频词更友好**：由于上下文平均会“平滑”掉低频词的影响，CBOW 对高频词的表示往往更好，但对罕见词的利用不够充分。

---

## 5. Skip-gram 模型

### 5.1 模型结构

与 CBOW 的“多对一”相反，Skip-gram 是“一对多”的模型：给定中心词 \(w_t\)，预测其上下文窗口内每个位置上的词 \(w_{t+j}\)（\( -m \le j \le m, j \neq 0\)）。输入是中心词的独热向量，投影层直接取出其输入向量 \(\mathbf{v}_{w_t}\)，然后分别对每个目标输出上下文词计算 Softmax。

中心词向量 \(\mathbf{h} = \mathbf{v}_{w_t}\)，对于每个目标上下文词 \(w_{t+j}\)，条件概率为：

$$
P(w_{t+j} \mid w_t) = \frac{\exp(\mathbf{u}_{w_{t+j}}^\top \mathbf{v}_{w_t})}{\sum_{w \in \mathcal{V}} \exp(\mathbf{u}_{w}^\top \mathbf{v}_{w_t})}
$$

### 5.2 训练目标

Skip-gram 的目标函数是独立地最大化每个上下文词的概率：

$$
\mathcal{L} = \sum_{t=1}^{T} \sum_{-m \le j \le m, j \neq 0} \log P(w_{t+j} \mid w_t)
$$

训练时，窗口内所有词对 \((w_t, w_{t+j})\) 都被当作正样本参与梯度更新。这使得 Skip-gram 对罕见词更敏感，因为每个中心词都会为其上下文的每个词产生训练信号，包括那些平时出现次数很少的词。

### 5.3 Skip-gram 的特点

- **细粒度学习**：每个上下文词独立预测，迫使向量捕捉更精细的语义和句法信息。实验表明，Skip-gram 在语义类比任务（如 king–man+woman）上通常优于 CBOW。
- **更适应大规模语料**：随着数据量增加，Skip-gram 能持续从每个词对的共现中学习，而 CBOW 的平均操作会损失部分信息。对于数十亿词级别的训练，Skip-gram 表现尤为突出。
- **计算成本更高**：最初版本的 Skip-gram 对于每个中心词需要计算 \(2m\) 次 Softmax，计算量约为 CBOW 的 \(2m\) 倍。因此论文后续引入了层次 Softmax 和负采样来大幅度降低开销。

---

## 6. 优化技术

普通的 Softmax 需要计算整个词汇表的得分，时间复杂为 \(O(V)\)，在 \(V\) 为 \(10^5 \sim 10^7\) 时不可行。论文提出了两种降低复杂度的方法：层次 Softmax（Hierarchical Softmax）和负采样（Negative Sampling）。

### 6.1 层次 Softmax

层次 Softmax 使用一棵二叉树（通常是哈夫曼树）来表示词汇表，每个词都是树上的叶子节点。从根节点到叶子节点的路径代表该词的概率。设路径长度为 \(L(w)\)，路径上的节点序列为 \(n(w,1), n(w,2), \dots, n(w, L(w))\)，其中 \(n(w,1)\) 是根节点，\(n(w, L(w))\) 是词 \(w\) 的叶子。每个内部节点拥有一个可学习的向量 \(\mathbf{v}_n'\)。

在 Skip-gram 中，以中心词向量 \(\mathbf{v}_w\) 为条件，预测上下文词 \(w_O\) 的概率为：

$$
P(w_O \mid w_I) = \prod_{j=1}^{L(w_O)-1} \sigma \left( [n(w_O, j+1) = \text{ch}(n(w_O, j))] \cdot \mathbf{v}_{n(w_O, j)}'^\top \mathbf{v}_{w_I} \right)
$$

其中 \(\text{ch}(n)\) 表示节点 \(n\) 的左孩子，\([ \cdot ]\) 为指示函数：如果路径下一步走向左孩子则为 \(+1\)，右孩子则为 \(-1\)。\(\sigma(x) = 1/(1+e^{-x})\)。这样，每一步只是一个二分类问题，计算复杂度从 \(O(V)\) 降低到 \(O(\log V)\)。

**哈夫曼树**的妙用：根据词频构建哈夫曼树，可以使高频词拥有短路径（靠近根），低频词路径较长。这进一步缩短了高频词的计算时间，使平均复杂度优于平衡树。

### 6.2 负采样（Negative Sampling）

负采样是更简单、更高效的一种方法，也是实际应用最广泛的策略。其核心思想是：不计算完整的 Softmax，而是将多分类问题转化为二分类问题——对某个上下文词对 \((w_I, w_O)\)，我们希望最大化其共现概率，同时从噪声分布中采样 \(K\) 个负样本（未出现的词），最小化它们的共现概率。

对于 Skip-gram，给定中心词 \(w_I\) 和上下文词 \(w_O\)，正样本的损失定义为：

$$
\log \sigma(\mathbf{u}_{w_O}^\top \mathbf{v}_{w_I})
$$

对于每个负样本 \(w_k \sim P_n(w)\)（噪声分布），损失为：

$$
\log \sigma(-\mathbf{u}_{w_k}^\top \mathbf{v}_{w_I})
$$

总目标函数为：

$$
\mathcal{L} = \sum_{(w_I, w_O) \in \mathcal{D}} \left( \log \sigma(\mathbf{u}_{w_O}^\top \mathbf{v}_{w_I}) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n} \left[ \log \sigma(-\mathbf{u}_{w_k}^\top \mathbf{v}_{w_I}) \right] \right)
$$

其中 \(\mathcal{D}\) 是语料中所有正样本窗口词对。

**噪声分布的选择**：论文经过实验发现，将分布平滑化为词频的 \(3/4\) 次方效果最佳：

$$
P_n(w) = \frac{\text{count}(w)^{3/4}}{\sum_{i} \text{count}(w_i)^{3/4}}
$$

这种平滑可以增大低频词被采样的概率，避免负样本过于集中高频词，从而提升词向量的区分性。

负采样的计算复杂度约为 \(O((K+1)d)\)，通常 \(K\) 取 5～20，对于大规模词汇表依然极快。

### 6.3 高频词二次采样（Subsampling）

论文中还提出了高频词二次采样技术，以平衡罕见词与高频词（如“the”、“a”、“is”）的影响。在训练时，序列中的每个词 \(w_i\) 以概率

$$
P(\text{drop}) = 1 - \sqrt{\frac{t}{f(w_i)}}
$$

被丢弃，其中 \(f(w_i)\) 是词频，\(t\) 为阈值（通常 \(10^{-5}\) 左右）。该公式使得词频大于 \(t\) 的高频词有一定概率被跳过，既加快了训练速度，又显著提高了罕见词向量的质量，同时改善了类比任务的准确率。

---

## 7. 训练细节

### 7.1 数据集与预处理

论文使用的训练数据包括：
- **Google News 语料**：约 1000 亿词，用于最大规模的模型。
- **Wikipedia 等**公共语料，作为对比实验。
- 文本经过简单分词（tokenization），词汇表截取频率最高的 \(10^5 \sim 10^6\) 词，低频词替换为 `<UNK>` 或直接忽略。

### 7.2 超参数设置

典型的超参数配置：
- 词向量维度 \(d\)：100–1000，常用 300。
- 上下文窗口大小 \(m\)：5–10（Skip-gram 通常 10，CBOW 5）。
- 负采样数 \(K\)：5～20（对于小数据集 5–10 即可，大数据集可适当增加）。
- 学习率：初始 0.025，线性衰减至接近 0。
- 训练轮次（epoch）：通常 3～15，大数据集 1 也可以。
- 二次采样阈值 \(t\)：\(10^{-5}\) 或 \(10^{-3}\)。

### 7.3 并行化与训练效率

Word2Vec 工具包采用多线程并行，将大规模语料分块，每个线程更新共享的向量矩阵（或通过 Hogwild! 风格的无锁异步更新）。加上层次 Softmax 或负采样，使得单机即可在几小时内完成 1000 亿词级别的训练，这在当时是革命性的。

---

## 8. 论文实验与结果

### 8.1 语义-句法词汇类比任务

论文设计了著名的类比任务测试集，包含 5 类语义类比（例如 `Athens Greece Baghdad Iraq`）和 9 类句法类比（例如 `dance dancing predict predicting`）。模型需要通过向量运算 `vec(a) - vec(b) + vec(c)` 找到最接近 `vec(d)` 的词。准确率定义为正确词语排在第一位的比例。

结果要点：
- **维度影响**：Skip-gram 模型在 300 维时准确率显著提升，继续增大维度收益递减。CBOW 在 300 维后趋向饱和。
- **训练数据量**：随着数据从数百万词增长到数十亿词，准确率持续提高，Skip-gram 尤其受益。
- **模型对比**：Skip-gram 在所有类比任务上全面超越 CBOW，尤其句法类比差距明显。
- **与 RNNLM 对比**：Word2Vec 在语义任务上远超当时最好的递归神经网络语言模型（RNNLM），而训练时间仅为其一小部分。

### 8.2 词向量相加的性质

论文发现，Skip-gram 学到的向量能捕捉单词之间的组合关系，例如 “German” + “airlines” 的结果向量最接近 “Lufthansa”。这提示词向量空间中存在某种线性平移不变性，也启发了后续 FastText、GloVe 等模型。

### 8.3 不同优化技术的比较

- **层次 Softmax vs. 负采样**：负采样训练的向量在类比任务上表现略优于层次 Softmax，且实现更简单，不需要构造哈夫曼树。但层次 Softmax 在计算资源极端受限时仍有优势。
- **二次采样**：使用 subsampling 后，模型对高频词的强化减少，类比准确率平均提升约 2%–4%，尤其句法任务提升显著。

---

## 9. 词向量的性质与语义关系

### 9.1 线性结构

Word2Vec 最吸引人的特性是词向量的线性平移性质。通过简单的向量加减：

$$
\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}
$$

这表明词向量空间中编码了某种“语义轴”。这种线性关系来源于训练目标：Skip-gram 的目标是最大化词对的内积，对于共现模式相似的词，其向量差异会趋向于一致。

### 9.2 向量偏移与类比推理

类比推理是评估词向量质量的核心任务。给定三词 a, b, c，寻找 d 使得 `a : b :: c : d`，即：

$$
d = \arg\max_{w} \cos(\vec{v}_b - \vec{v}_a + \vec{v}_c, \vec{v}_w)
$$

常用余弦相似度作为度量。论文证明，通过大规模自监督学习，这个简单的公式能准确解答大量类比问题，包括国家-首都、时态变化、形容词-副词等。

### 9.3 可视化

通过 PCA 或 t-SNE 将高维词向量降维至 2D/3D，可以直观看到语义类别（国家、动物、食物等）和句法模式（动词时态）形成有规律的聚集。这种可视化进一步证实了向量的语言学价值。

---

## 10. 后续工作与改进

Word2Vec 开创了词嵌入的新时代，随后涌现了大量衍生和优化模型：

### 10.1 GloVe (2014)

斯坦福大学提出的 GloVe（Global Vectors for Word Representation）结合了全局矩阵分解和局部上下文窗口方法的优点。它在全局词共现矩阵上进行分解，相比 Word2Vec 更好地利用了统计信息，生成的向量在类比任务上具有类似甚至更优的性能。

### 10.2 FastText (2016)

Facebook 的 FastText 在 Word2Vec 的词级别向量基础上，引入了子词（subword）信息，使用字符 n-gram 的向量和来表示单词。这解决了 Word2Vec 无法处理未登录词（OOV）和拼写变体的痛点，在形态丰富的语言和罕见词处理上表现出色。

### 10.3 ELMo、BERT 与上下文相关表示

Word2Vec 生成的是静态向量，即每个词只有一个固定嵌入，无法区分“bank”在“river bank”和“bank account”中的不同含义。ELMo（2018）引入双向 LSTM 产生上下文相关的词向量，而 BERT（2019）基于 Transformer 架构将预训练语言模型推向了新的高度。但 Word2Vec 的思想——“通过预测共现学习表示”——依然是这些模型的基础。

### 10.4 对工业界的影响

Word2Vec 的低成本、易部署使词向量成为 NLP 流水线的标准组件。从推荐系统到搜索广告，物品和查询的嵌入化都借鉴了 Word2Vec 的范式。它的负采样和分组 Softmax 思想也被推广到图嵌入（node2vec）、序列嵌入等众多领域。

---

## 11. 影响与总结

Word2Vec 论文之所以成为里程碑，不仅因为它提出了两种高效的词向量训练模型，更重要的是它证明了**简单的模型结构结合大规模数据可以学到令人惊叹的语义表示**。它的成功推动了以下认知的普及：

1. **表示学习**：从符号到稠密向量的转变打开了深度学习在 NLP 的应用大门。
2. **自监督预训练**：无需昂贵的人工标注，利用文本自身结构即可获得优质特征，这成为了现代预训练模型（如 GPT、BERT）的核心哲学。
3. **可解释性**：词向量的线性规则让人们看到，神经网络的内部表征亦可能具有语义可解释性，激发了后续大量的探针（probe）和分析研究。

**总结 Word2Vec 的核心贡献**：
- **架构贡献**：CBOW 和 Skip-gram，简单有效。
- **训练加速**：层次 Softmax 和负采样，使十亿级别的训练可行。
- **实证发现**：词向量加法揭示语义-句法规律，二次采样改善罕见词学习。

对于学习词向量或准备进入 NLP 领域的研究者，精读 Word2Vec 论文是必不可少的基础。它优雅地将统计语言模型、神经网络优化和大规模数据处理融为一体，至今仍启发着无数表示学习的工作。

---

## 参考文献

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. *arXiv preprint arXiv:1301.3781*.
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. *Advances in Neural Information Processing Systems*, 26.
3. Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A Neural Probabilistic Language Model. *Journal of Machine Learning Research*, 3(Feb), 1137–1155.
4. Goldberg, Y., & Levy, O. (2014). word2vec Explained: Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method. *arXiv preprint arXiv:1402.3722*.
5. Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for Word Representation. *EMNLP*.
6. Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching Word Vectors with Subword Information. *Transactions of the ACL*.

---

> **如何使用本文档**：您可以将以上内容复制并保存为 `word2vec_learning_notes.md` 文件，使用支持 Markdown 的编辑器（如 VS Code、Typora）阅读或导出为 PDF。本文全面覆盖了 Word2Vec 论文的核心理论、模型细节、优化技术和实验分析，适合作为研究入门和复习的参考资料。
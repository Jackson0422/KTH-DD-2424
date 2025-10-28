# KTH-DD-2424
Assignment 1 — Softmax Linear Classifier (CIFAR-10)

Goal. Train a one-layer softmax classifier with mini-batch GD, cross-entropy loss, and L2 regularization on W. Implement forward pass, accuracy, analytical gradients, and training. Visualize W as class templates; report curves and test accuracy for several (λ, η) settings. 
Deliverables. Single code file + brief PDF report with loss/cost curves, W visualizations, and test accuracy per setting. 
Bonus (optional). Use all data (smaller val set), data augmentation (horizontal flip), grid search over λ/η/batch, and LR decay. 
Your results (summary). Reported test accuracy examples: η=0.001, λ=0 → 39.48%; λ=0.1 → 39.44%; λ=1 → 37.3%; η=0.1 unstable; discussion on λ/η trade-off.

Assignment 2 — Two-Layer ReLU MLP + Cyclical Learning Rates

Goal. Extend to a 2-layer MLP (W1,b1→ReLU→W2,b2→Softmax). Implement forward/backward, verify gradients (e.g., via PyTorch on reduced dims), and train with CLR (triangular schedule with ηmin, ηmax, ns). Perform coarse-to-fine search over λ and scale up training. 
Deliverables. Code + brief report: replicate default CLR curves, list ranges and top settings for coarse/fine search, and final test performance using nearly all training data. 
Bonus (optional). Wider hidden layers, dropout & augmentation, alternative optimizers (e.g., Adam), longer runs; compare performances.

Assignment 3 — 3-Layer Net with Patchify Convolution (ViT-style front)

Goal. Build a 3-layer classifier where the first layer is a convolution with stride = patch size. Start with a slow reference implementation, then an efficient matrix/einsum version; add ReLU and two FC layers. Optional label smoothing and CLR with increasing step sizes for longer runs. 
Deliverables. Code + report: correctness checks using provided debug data, training curves, and comparisons across architectures (filter sizes/num filters). 
Bonus (optional). Wider models + augmentation; compare training speed vs. PyTorch conv2d; document accuracy and timing conclusions. 

Assignment 4 — Character-Level Vanilla RNN (Goblet of Fire)

Goal. Implement a tanh RNN (U,W,V,b,c) trained with Adam for next-character prediction; report smooth loss over long runs and show sampled text during training. Explore temperature and nucleus (top-p) sampling. Consider simple speedups (precomputations, sparse one-hot indexing). 
Deliverables. Code + brief report: gradient-check notes, smooth-loss plot (≥2 epochs), 200-char samples every 10k steps, and a 1000-char sample from the best model. 
Bonus (optional). Training order variants, batch size > 1, alternative sampling strategies and analyses.

作业一 — CIFAR-10 线性 Softmax 分类器

目标。 使用迷你批量 GD、交叉熵与 W 的 L2 正则训练一层 softmax 分类器；实现前向、精度、解析梯度与训练；可视化 W；在多组 (λ, η) 下画曲线并汇报测试准确率。
提交。 单文件代码 + 简短报告（训练/验证损失或代价曲线、W 可视化、各设置的测试精度）。
加分（可选）。 全量训练+更小验证集、水平翻转增强、对 λ/η/批大小做网格/随机搜索、学习率衰减等。
你的结果摘要。 例如 η=0.001, λ=0 → 39.48%；λ=0.1 → 39.44%；λ=1 → 37.3%；η=0.1 不稳定；并讨论 λ/η 折中。

作业二 — 两层 ReLU 网络 + 循环学习率（CLR）

目标。 扩展为两层 MLP（W1,b1→ReLU→W2,b2→Softmax），实现前/反向与 CLR（三角波，ηmin, ηmax, ns），并通过 粗到细 的 λ 搜索与更长训练得到更优结果；用小维度 PyTorch 校验梯度。
提交。 代码 + 简报：复现默认 CLR 曲线；给出粗/细搜索范围与前三名配置；在几乎全量训练集上的最终测试表现。
加分（可选）。 增大隐藏层、加入 dropout/数据增强、尝试 Adam、拉长训练并比较效果。

作业三 — 带 Patchify 卷积首层的三层网络

目标。 第一层为步幅=patch 尺寸的卷积；先写慢版对照，再写高效矩阵/einsum 版本；接 ReLU 与两层全连接；可选 Label Smoothing 与步长递增的 CLR 进行更久训练。
提交。 代码 + 报告：利用调试数据做正确性校验；给出不同架构（滤波器大小/数量）的曲线与对比。
加分（可选）。 拓宽模型+增强；与 PyTorch conv2d 的训练计时对比；记录准确率与速度差异并作结论。

作业四 — 字符级 Vanilla RNN（《哈利·波特4》）

目标。 实现 tanh RNN（参数 U,W,V,b,c）并用 Adam 训练字符级预测；记录平滑损失并在训练中展示采样文本；尝试 温度与核采样（top-p）；结合预计算/稀疏索引等做简单提速。
提交。 代码 + 报告：梯度校验说明，≥2 个 epoch 的平滑损失曲线；每 1 万步的 200 字样本；最佳模型的 1000 字样本。
加分（可选）。 改变训练顺序、批大小>1、不同采样策略等，并进行效果分析。

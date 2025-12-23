---
title: Diffusion Models
date: 2025-12-18 19:48:53
categories:
- research
tags:
- AI
- generative models
- computer vision
- diffusion model
---

生成式模型是如何从“加噪”与“去噪”的物理直觉，进化到“随机微分过程”与“概率流匹配”的数学精密推导的？
本文深入解析了 DDPM 的马尔可夫链基础与噪声回归本质，探讨了 DDIM 如何通过 ODE 视角实现采样加速，并进一步延伸至 Score Matching 在流形分布上的理论支撑，最后还剖析了 Flow Matching 如何利用最优传输理论将复杂的概率演化简化为高效的线性轨迹。通过对比 SDE 与 ODE 两种范式，希望探索生成模型在质量、速度与确定性之间取得平衡的数学底层逻辑，进而窥见扩散模型的数学本质。

<!--more-->

详细数学推导和介绍参考[我的笔记](https://kind-acai-b84.notion.site/diffusion-model-learning)~ (还在持续更新完善中!😀)

# DDPM
DDPM 是一种通过模拟非平衡热力学过程来生成的概率模型。它由两个对称的过程组成：前向扩散和反向去噪。
1. **前向扩散过程**是从真实数据向纯噪声演化的过程，具有以下特点：
- 马尔可夫链：通过 $T$ 步逐步向原始数据 $x_0$ 添加高斯噪声。
- 重参数化技巧：利用 $\alpha_t = 1 - \beta_t$ 的定义，可以在任意时间步 $t$ 直接采样出 $x_t$：
$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
- 固定过程：该过程不需要学习，由预定义的方差序列 $\beta_t$ 控制。
2. **反向去噪过程**是从高斯噪声中恢复数据样本的过程，是模型学习的核心：
- 贝叶斯后验：在已知 $x_0$ 的条件下，反向过程的真实分布 $q(x_{t-1}|x_t, x_0)$ 是高斯的，其均值 $\tilde{\mu}\_t$ 是 $x_t$ 与噪声 $\epsilon$ 的线性组合，我们使用神经网络 $p_\theta$ 来预测均值 $\mu_\theta$ 和方差 $\Sigma_\theta$。
- 学习目标：模型本质上是在预测“在时间步 $t$ 注入的噪声 $\epsilon$”。
3. **训练与损失函数** (Training & Loss)
- 简化损失项：虽然理论推导基于变分下界（VLB），但实证发现预测噪声的简单 MSE 损失效果更好：
$$L\_{simple} = \mathbb{E}\_{t, x\_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x\_t, t) \|^2 \right]$$
- 训练特点：训练时可以随机抽取时间步 $t$ 进行监督回归；而生成时必须从 $x_T$ 开始逐步迭代采样。

![](ddpm.png)

$$x\_t = \sqrt{\alpha_t}x\_{t-1} + \sqrt{1-\alpha_t}\epsilon = \sqrt{\overline{\alpha}\_t} x\_0 + \sqrt{1-\overline{\alpha}\_t} \epsilon$$
$$P(x\_{t-1}|x\_t)=\mathcal{N}(\frac{1}{\sqrt{\alpha}\_t}(x\_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}\_t}}\epsilon),\frac{(1-\alpha_t)(1-\overline{\alpha}\_{t-1})}{1-\overline{\alpha}\_t})$$

我的**简单总结**：
在训练阶段，由于原始样本 $x_0$ 可知，可以通过正向扩散过程构造带有真实噪声标签的中间状态 $z_t$，从而将扩散模型的学习转化为一个对噪声的有监督回归问题。而在生成阶段，由于不存在真实噪声标签，模型只能依据当前状态 $z_t$ 对噪声进行预测，并按照学习到的条件分布 $p\_\theta(z\_{t-1}\mid z\_t)$ 逐步反向采样。由于模型本身仅学习了局部时间步上的去噪映射，因此生成过程必须以多步迭代的方式进行；同时在每一步中引入适当的随机噪声以保持生成分布的随机性和多样性 (每一步再加噪声是为了保证采样仍然符合一个随机概率过程，而不是塌缩成一条确定路径)。


# DDIM
DDPM 的致命缺点是推理速度过慢，有无法避免的迭代过程，因为其本身是一个马尔科夫链的过程，无法进行跳跃预测。想要高质量的图片，就意味着 T 要取一个较大的值。而 DDIM 通过数学推理，打破了马尔科夫链的过程；且其无需重新训练 DDPM，只对采样器进行修改即可，修改后的采样器能够大幅度增加采样速度。

$$x\_{prev} = \underbrace{\sqrt{\bar{\alpha}\_{prev}} \left( \frac{x\_t - \sqrt{1-\bar{\alpha}\_t} \epsilon_\theta(x\_t)}{\sqrt{\bar{\alpha}\_t}} \right)}\_{\text{预测的 } x\_0} + \underbrace{\sqrt{1-\bar{\alpha}\_{prev} - \sigma^2} \epsilon_\theta(x\_t)}\_{\text{指向 } x\_t \text{ 的方向}} + \underbrace{\sigma_t \epsilon}\_{\text{随机噪声}}$$

其中，$\sigma$ 是控制随机性的超参数：当 $\sigma = \sigma_{DDPM}$ 时，DDIM 回退为 DDPM 的马尔可夫采样；当 $\sigma = 0$ 时，采样过程变为确定性的，此时扩散模型变成了一个 ODE（常微分方程）过程。


DDIM 的**三大优势**：
- 更少的步骤，更高的质量：在较小的采样步数下（如 20-50 步），DDIM 的样本质量显著优于相同步数的 DDPM。
- 生成一致性：由于采样过程可以是确定性的，给定同一个初始噪声 $x_T$，模型总能生成同一张图片，这对于图像编辑非常重要。
- 语义插值：由于确定性的映射关系，用户可以在初始潜变量 $x_T$ 空间进行线性插值，从而实现两张生成图片之间的平滑过渡。

# Score Matching

分数函数为 $\nabla_x \log p(x)$, Score Matching 损失函数为：$\frac{1}{2} \mathbb{E}\_{q\_\sigma(\tilde{x}|x)p\_{data}(x)}[||s\_\theta(\tilde{x})-\nabla\_{\tilde{x}}\log q\_\sigma(\tilde{x}|x)||^2\_2]$

损失函数（去噪分数匹配, DSM）：$$\mathcal{L} = \frac{1}{2}\mathbb{E}\_{p\_{data}(x)}\mathbb{E}\_{\tilde{x}\sim \mathcal{N}(x;\sigma^2\mathbf{I})}\left[ \left\| s\_\theta(\tilde{x},\sigma) + \frac{\tilde{x}-x}{\sigma^2} \right\|\_2^2 \right]$$ 这里的 $-\frac{\tilde{x}-x}{\sigma^2}$ 实际上就是高斯噪声扰动后的真实分数。神经网络 $s_\theta$ 的目标就是预测这个“指向高密度区域”的向量。

**Q: 为什么要加不同级别的噪声？**
- 打破流形假设（Manifold Hypothesis）: 现实数据分布在低维流形上（比如图像空间中只有极小一部分像人脸）。在流形之外，分数函数是没有定义的; 加噪声会让数据分布“铺满”整个全空间，使得神经网络在任何位置都能学到有效的指向, 从而解决低密度区域的估计问题：如果不加噪声，在数据稀疏的地方，分数估计会产生巨大的偏差。
- 加大噪声, 可以让模型在离流形很远的地方也能感知到大体的方向（全局概览）; 加小噪声, 当粒子接近流形时负责微调，确保细节精度（局部细节）。多级噪声可以确保采样过程既能快速定位又能精准收敛

# Flow Matching

Flow Matching 的本质是利用常微分方程 (ODE) 来描述样本随时间 $t$ 的演化。

前置知识：
- 连续归一化流 (CNF)：定义一个含时间参数的向量场 $v_t$，通过欧拉方法采样：$x\_{t+\delta t} = x\_t + v\_t(x\_t) \delta t$。
- 连续性方程 (Continuity Equation)：确保向量场 $v_t$ 与概率路径 $p_t$ 满足物理一致性：$\frac{d}{dt} p\_t(x) + \text{div}(p\_t(x)v\_t(x)) = 0$

由于直接获取目标分布的向量场 $u_t(x)$ 是不可行的，FM 借鉴了 DDPM 的思路，通过引入条件变量 $x_1$ 构造损失函数, 同时也证明了预测“条件向量场” $u_t(x|x_1)$ 的梯度下降方向与预测“边缘向量场” $u_t(x)$ 是一致的。损失函数：$$\mathcal{L}\_{CFM}(\theta) = \mathbb{E}\_{t, q(x\_1), p\_t(x|x\_1)} \| v\_\theta(x, t) - u\_t(x|x\_1) \|^2$$其中 $v_\theta$ 是神经网络预测的向量场，$u_t(x|x_1)$ 是基于已知数据点 $x_1$ 的真实条件向量场。

假设条件概率分布满足高斯分布 $p_t(x|x_1) = \mathcal{N}(x|\mu_t, \sigma_t^2 \mathbf{I})$，可以推导出条件向量场的一般形式：$$u_t(x|x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)}(x-\mu_t(x_1)) + \mu_t'(x_1)$$根据不同的 $\mu_t$ 和 $\sigma_t$ 设计，FM 可以统一多种模型路径：
- 最优传输路径 (Optimal Transport)：$\mu_t = t x_1$ 且 $\sigma_t = 1 - (1-\sigma_{min})t$。这是 FM 最具代表性的改进，由于其轨迹是直线，采样效率极高。
- 扩散路径：通过设置不同的均值和方差，FM 可以模拟 Variance Exploding (Score-matching) 或 Variance Preserving (DDPM) 的轨迹。

Flow Matching 的优势
- 采样速度快：通过最优传输路径（直线轨迹），FM 可以比弯曲轨迹的扩散模型在更少的步骤内达到高质量生成效果。
- 确定性与效率：相比于具有随机性的 SDE（随机微分方程），FM 这种基于 ODE 的模型更易于训练且推理更加稳健。
- 架构灵活：FM 可以直接应用在像素空间，也可以像 Stable Diffusion 一样在潜在变量空间运行。


# A Brief Summary
我的一些总结整理：

- DDPM：基于 SDE 的方法
- DDIM：允许跳步采样，并且在采样过程中消除了随机项（将其设置为零），是基于 ODE 的方法
- Score Matching：
    - Song 等人在 2020 年的工作中，将所有 Score-Based Generative Models (包括 SMLD 和 DDPM 的连续形式) 统一到了一个SDE 框架下。这个框架中，生成过程（逆时间 SDE）是一个随机过程。
    - 概率流 ODE (Probability Flow ODE)： SDE 框架还导出了一个等价的ODE，称为概率流 ODE (Probability Flow ODE)。这个 ODE 描述了与 SDE 相同的边际分布，但路径是确定性的。因此，Score Matching 训练出的模型可以同时用于 SDE 采样（随机）和 ODE 采样（确定性）。
    - 数学上可以证明，这个“概率流 ODE”所控制的概率密度 $p_t(\mathbf{x})$ 的演化，也恰好满足同一个 Fokker-Planck 方程，即：SDE 满足 FPE；概率流 ODE 同样满足 FPE
- Flow Matching：整个训练和生成过程都是在确定性的 ODE 路径上进行的

![](ODE_SDE.png)

P.S. 本文均为作者阅读后的一些总结与思考，如有错误，欢迎指出~
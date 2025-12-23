---
title: 'EDM: A Landmark Work in Diffusion Models'
date: 2025-12-19 15:02:02
categories:
- research
tags:
- AI
- generative models
- computer vision
- diffusion model
---

EDM 论文可以说是一篇扩散模型领域里程碑式的论文，其真正总结了当前出现的主流扩散模型并形成了一套通用框架；在此基础上，EDM 还在通用框架的基础上提出了一套较优的实现方案，并在各大主要数据集上达到了 SOTA 的性能。

<!--more-->

EDM 这篇论文的核心思想在于：把当时混乱、高度依赖数学公式的扩散模型研究，拆解成了若干个独立的、可优化的工程模块，起到了**解耦**的作用。

更多的数学推导请参考[我的笔记](https://kind-acai-b84.notion.site/Diffusion-Model-Differential-Equation-2c7ef27c0e5e80538cb7d94ff77018d9)~


扩散模型训练过程与常规 CV 模型完全不同，通常也是大家入门感到最困难的地方。扩散模型建模的不是噪声和图像的一一对应关系，而是通过反复不断地随机采样，**建模两个分布之间的关系**，比如咱们熟悉的标准高斯噪声分布到自然图像分布。为了建模两个分布之间的关系，扩散模型首先通过加噪公式，缓慢地将一个分布 “推向” 另外一个分布，相比直接建模两个分布的方法（VAE、GAN 等），通常有更好的图像生成效果，但以生成速度作为代价。

# 扩散模型训练
## **通用加噪公式**

加噪公式就是将一个分布变为另外一个分布的桥梁，实际上也就是大家耳熟能详的流（Flow）。加噪公式的存在是为训练过程服务的，只有建立这个桥梁，才能明确模型的训练目标，扩散模型的学习才有可能。EDM 给定的通用加噪公式为：

$$
p(\boldsymbol{x}_t|\boldsymbol{x}_0) = \mathcal{N}\left(\boldsymbol{x}_t; s(t)\boldsymbol{x}_0, s^2(t)\sigma^2(t)\boldsymbol{I}\right) \tag{1}
$$

其中，$s(t)$ 和 $\sigma(t)$ 分别为时间的函数，$\boldsymbol{x}_0$ 是原始图像。由于当前扩散模型的加流模型基本都可满足公式 (1)，扩散模型的研究在 EDM 论文出现后变成了超参数的挑选过程了。从数学角度上，公式 (1) 又一定可以转化为一个随机微分方程（SDE），形式如下：

$$
d\boldsymbol{x}_t = f(t)\boldsymbol{x}_t dt + g(t)d\boldsymbol{w}_t \tag{2}
$$

其中，f(t) 和 g(t) 输入和输出均为$\mathbb{R}^1$，$d\boldsymbol{w}_t$ 是维纳过程。注意，这个 SDE 的解 $\boldsymbol{x}_t$ 是能够与公式 (1) 获得的 $\boldsymbol{x}_t$ 在分布上保持一致（起点分布相同条件下）。

>**常见问题**：为什么 SDE 能和一个加噪公式构建的$\boldsymbol{x}_t$保持分布一致？
>
>**个人理解**：首先公式 (1) 经过一些变换可以变成公式 (2) 的形式，二者本身就是等价的。其次，每一个伊藤过程 SDE 都能够写出解的均值和方差，通过均值和方差也能够反写出一个形如公式 (1) 的高斯加噪公式。

通过伊藤过程 SDE 解 $\boldsymbol{x}_t$ 的均值和方差公式，联合公式 (1)，可以推导得到下面的关键结论：

$$
s(t) = e^{\int\_0^t f(r)dr} \tag{3}
$$

$$
\sigma^2(t) = \int_0^t \frac{g^2(r)}{s^2(r)} dr \tag{4}
$$

上式清晰地给出了加噪公式中 $s(t)、\sigma(t)$ 与 SDE 中 $f(t)$ 和 $g(t)$ 的相关关系，更加说明了扩散模型中公式 (1) 和 (2) 是一一对应的。特别的，EDM 框架设计的非常简单，它令 $s(t)=1、\sigma(t)=t$，保证了时间和噪声水平完全等价。

## 通用模型框架

EDM分析了几乎所有的扩散模型框架，总结出了如下通用模型框架：

$$
D_\theta(\tilde{\boldsymbol{x}}; \sigma) = C_{\text{skip}}(\sigma)\tilde{\boldsymbol{x}} + C_{\text{out}}(\sigma) F_\theta\left(C_{\text{in}}(\sigma)\tilde{\boldsymbol{x}}; C_{\text{noise}}(\sigma)\right) \tag{5}
$$

模型框架需要掌握的关键点有以下三个：

- $F_\theta$ 是真正扩散模型训练的神经网络（U-net、DIT等），$D_\theta$ 是单纯的去噪网络，也即任意扩散模型训练的神经网络都可以转化为一个单纯的去噪网络，该单纯去噪网络由原始输入$\tilde{\boldsymbol{x}}$和真正训练的神经网络$F_\theta$加权求和获得，权重系数分别为$C_{\text{skip}}(\sigma)$和$C_{\text{out}}(\sigma)$。
- $C_{\text{noise}}(\sigma)$是一个对$\sigma$的变换函数，EDM框架要求模型输入为$\sigma$，但是你训练的神经网络的条件输入不一定是噪声$\sigma$，其中最典型的案例就是DDPM、DDIM、Flow Matching等输入的条件为时间$t$，但时间 t 又和噪声水平$\sigma$相关，所以给一个变换函数$C_{\text{noise}}(\sigma)$统一把 t 转换成我们输入的条件就可以了。
- $C_{\text{in}}(\sigma)$的出现也是为了统一框架而进行设定，注意这里的输入是$\tilde{\boldsymbol{x}}$，表示EDM定义的标准化输出，像素值区间为 [-1, 1]。由于加噪公式的 s(t) 能够导致图像分布均值像素值区间变化，需要额外的 $C_{\text{in}}(\sigma)$ 将标准化输出$\tilde{\boldsymbol{x}}$的像素值区间变换到均值的像素值区间。

![](模型比较.png)

## 通用训练框架

训练框架的核心就是损失函数，在有了单纯去噪模型(5)后，直接根据单纯去噪损失：

$$
\mathcal{L}\_{\text{diff}} = \mathbb{E}\_{\sigma \sim p\_{\text{train}}, \mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}), \mathbf{y} \sim p\_{\text{data}}} \left[ \lambda(\sigma) \| D\_\theta(\mathbf{y} + \mathbf{n};\sigma) - \mathbf{y} \|\_2^2 \right]
$$

联合基本的初等运算，可以得到下面的损失函数形式：

$$
\mathcal{L}\_{\text{diff}} = \mathbb{E}\_{\sigma,\boldsymbol{n},\boldsymbol{y}} \left[ \lambda(\sigma) \left\| C\_{\text{skip}}(\sigma)(\boldsymbol{y} + \boldsymbol{n}) + C\_{\text{out}}(\sigma) F\_\theta\left( C\_{\text{in}}(\sigma)(\boldsymbol{y} + \boldsymbol{n}); C\_{\text{noise}}(\sigma) \right) - \boldsymbol{y} \right\|\_2^2 \right]$$
$$
\mathcal{L}\_{\text{diff}} = \mathbb{E}\_{\sigma,\boldsymbol{n},\boldsymbol{y}} \left[ \underbrace{\lambda(\sigma) C\_{\text{out}}^2(\sigma)}\_{\text{损失权重}w(\sigma)} \left\| \underbrace{F\_\theta\left( C\_{\text{in}}(\sigma)(\boldsymbol{y} + \boldsymbol{n}); C\_{\text{noise}}(\sigma) \right)}\_{\text{模型输出}} - \underbrace{\frac{1}{C\_{\text{out}}(\sigma)} \left( \boldsymbol{y} - C\_{\text{skip}}(\sigma)(\boldsymbol{y} + \boldsymbol{n}) \right)}\_{\text{训练目标}F\_{\text{target}}(\boldsymbol{y},\boldsymbol{n},\sigma)} \right\|\_2^2 \right]
$$

其中，$\tilde{\boldsymbol{x}} = \boldsymbol{y} + \boldsymbol{n}$，$\boldsymbol{y}$表示原始干净数据，满足$\boldsymbol{y}\sim p_{\text{data}}$，$\boldsymbol{n}$表示噪声，满足$\boldsymbol{n}\sim \mathcal{N}(0,\sigma^2\boldsymbol{I})$，$\tilde{\boldsymbol{x}}$表示带噪声的数据，满足$\tilde{\boldsymbol{x}}\sim \mathcal{N}(\boldsymbol{y},\sigma^2\boldsymbol{I})$。

很明显，公式(9)包含了很多还不知道怎么确定的超参数$C_{\text{in}}(\sigma)、C_{\text{out}}(\sigma)、C_{\text{skip}}(\sigma)、\lambda(\sigma)$，EDM为了保证训练稳定，设定了下面三个“规矩”：

1. 神经网络的输入保持单位方差 $\Rightarrow C_{in}(\sigma) = \frac{1}{\sigma^2 + \sigma^2_{data}}$
2. 训练目标保持单位方差 $\Rightarrow C_{skip}(\sigma)=\frac{\sigma^2_{data}}{\sigma^2+\sigma^2_{data}} \quad C_{out} = \frac{\sigma\cdot \sigma_{data}}{\sqrt{\sigma^2+\sigma^2_{data}}}$
3. 等价对待所有的噪声水平损失函数，也即 $w(\sigma)=1$ $\Rightarrow \lambda(\sigma)= \frac{\sigma^2+\sigma^2_{data}}{(\sigma\cdot \sigma_{data})^2}$

最后，EDM通过实验发现损失函数在噪声水平很低和很高的时候，无论怎么训练，损失函数都难以下降，因此提出一种非均匀的 $\sigma$ 的分布 $p_{\text{train}}(\sigma)$，也即在训练过程中，不同 $\sigma$ 的出现不再是等概率采样出现，而是满足：

$$
\ln(\sigma)\sim \mathcal{N}(P_{mean},P_{std}^2) \quad P_{mean}=-1.2,\ P_{std}=1.2
$$

# 扩散模型推理

## 通用概率流常微分方程

对于任意一个扩散模型加噪过程SDE（公式(2)），通过福克普朗克方程，可进一步推导出一个常微分方程（ODE），也称为概率流常微分方程（PFODE）。这个PFODE在确定起点$\boldsymbol{x}_0$（前向）或$\boldsymbol{x}_N$（逆向）的前提下，解的分布与加噪过程SDE求得的解的分布是完全相同的。这个PFODE的形式为：

$$
\mathrm{d}\boldsymbol{x}_t = \left[ f(t)\boldsymbol{x}_t - \frac{1}{2}g^2(t)\nabla{\boldsymbol{x}_t}\log p_t(\boldsymbol{x}_t) \right] \mathrm{d}t
$$

进一步的，结合加噪公式中 $s(t)、\sigma(t)$ 与SDE中 $f(t)$ 和 $g(t)$ 的相关关系（公式(3)和(4)），并将分数项中的边缘概率密度表示为已知分布形式，可得：

$$
\mathrm{d}\boldsymbol{x}_t = \left[ \frac{\dot{s}(t)}{s(t)}\boldsymbol{x}t - s^2(t)\sigma(t)\dot{\sigma}(t)\nabla{\boldsymbol{x}_t}\log p\left( \frac{\boldsymbol{x}_t}{s(t)}; \sigma(t) \right) \right] \mathrm{d}t
$$

其中 $p(\boldsymbol{x}; \sigma(t)) = \left[ p\_{\text{data}} * \mathcal{N}(0, \sigma^2(t)\boldsymbol{I}) \right] (x\_t) = \mathcal{N}\left( \boldsymbol{x}\_t; \boldsymbol{x}, \sigma^2(t)\boldsymbol{I} \right) = p(\boldsymbol{x}\_t|\boldsymbol{x})$。

>**常见问题**：既然PFODE加个符号就能前向转逆向或者逆向转前向，能否用PFODE实现图像加噪？
>
>**个人理解**：绝对不可以！！这是因为PF-ODE在给定起点$\boldsymbol{x}_0$的时候，它的路径就完全确定了，一定会到达某个终点$\boldsymbol{x}_N$，这样就形成了样本的一一匹配。然而，扩散模型建模的是两个分布之间的关系，必须通过随机采样配对实现，因此绝对不能用PFODE实现图像加噪。


## 通用确定性采样

确定性采样基于PFODE和一个训练好的单纯去噪神经网络 $D_\theta(\boldsymbol{x};\sigma)$。因为PFODE中存在分数项，首先推导获得了单纯去噪神经网络与分数的关系如下：

$$
\nabla_{\boldsymbol{x}\_t} \log p\left( \frac{\boldsymbol{x}\_t}{s(t)}; \sigma(t) \right) = \frac{D\_\theta(\hat{\boldsymbol{x}}\_t; \sigma) - \hat{\boldsymbol{x}}\_t}{s(t)\sigma^2(t)}
$$

再将公式(18)代入公式(17)的PFODE中，可以得到：

$$
\mathrm{d}\boldsymbol{x}\_t = \left[ \left( \frac{\dot{s}(t)}{s(t)} + \frac{\dot{\sigma}(t)}{\sigma(t)} \right) \boldsymbol{x}\_t - \frac{s(t)\dot{\sigma}(t)}{\sigma(t)} D\_\theta\left( \frac{\boldsymbol{x}\_t}{s(t)}; \sigma \right) \right] \mathrm{d}t
$$

根据公式(19)就能使用一种ODE求解器（一阶Euler、二阶Heun），在给定起点的情况下，逐步采样获得生成图像（或别的数据）。

基于通用确定性采样的算法流程如下：

![](确定性采样.png)

此外，采样过程的噪声水平（时间点）序列的安排也对采样结果有影响，EDM根据实验结果确定了下面的噪声水平序列：

$$
\sigma\_{i<N} = \left( {\sigma\_{\text{max}}}^{\frac{1}{\rho}} + \frac{i}{N-1} ( {\sigma\_{\text{min}}}^{\frac{1}{\rho}} - {\sigma\_{\text{max}}}^{\frac{1}{\rho}} ) \right)^\rho \quad \text{and} \quad \sigma\_N = 0 $$

## 通用随机微分方程

通用随机微分方程的推导结合了热方程偏微分方程（heat equation PDE）和福克普朗克方程，形式如下：

$$
\mathrm{d}\mathbf{x}\_{\pm} = \underbrace{-\dot{\sigma}(t)\sigma(t)\nabla\_{\mathbf{x}}\log p(\mathbf{x};\sigma(t))\mathrm{d}t}\_{\text{PFODE}} \pm \underbrace{\beta(t)\sigma^2(t)\nabla\_{\mathbf{x}}\log p(\mathbf{x};\sigma(t))\mathrm{d}t + \sqrt{2\beta(t)\sigma(t)}\mathrm{d}\mathbf{w}\_t}\_{\text{Langevin diffusion SDE}}
$$

其中，$\mathrm{d}\boldsymbol{x}\_+$表示前向SDE，$\mathrm{d}\boldsymbol{x}_-$表示逆向SDE，去噪随机性采样过程使用逆向SDE形式。通用随机微分方程包含两大部分，一个部分就是PFODE，这个形式与前面公式是一样，只是令 $s(t)=1$ 而已；另一个部分是郎之万扩散SDE (Langevin SDE)，这个部分又包含了确定性噪声衰减和噪声注入两项。

>$\sqrt{2\beta(t)\sigma(t)}\mathrm{d}\mathbf{w}\_t$ 这一项是噪声注入，给当前的图片注入一点点新鲜的随机高斯噪声，目的是增加系统的熵，让样本有机会跳出当前的微小数值误差，探索更广阔的分布空间。

>$\beta(t)\sigma^2(t)\nabla\_{\mathbf{x}}\log p(\mathbf{x};\sigma(t))\mathrm{d}t$ 这一项是确定性噪声衰减，利用分数函数 (Score Function) 把刚刚注入的那部分噪声“消解”掉，这就像是 Langevin dynamics 通过“加噪-去噪”的循环，迫使样本不断贴近真实的概率流形。

## 随机性采样

细心的同学能够发现，只有这里的标题没有加上“通用”二字，这是因为随机性采样过程方法众多，甚至和逆向SDE公式本身“关系不大”。EDM论文也表示它设计的随机性采样过程**不是一种通用的SDE求解器**，而是一种面向扩散模型问题的垂类SDE求解器。

EDM设计的随机性采样过程非常简单，其核心就是在确定性采样的基础上增加了“回退”操作，也即先对样本额外加噪，再采用ODE求解器采样获得下一个时间点的图像。这种回退操作可以有效修正前面迭代步骤产生的误差，所以通常相比PFODE的生成效果更好，但同时也要花费更多的采样步数。

EDM 提出的 SDE 采样器（求解器）基本算法流程如下所示：

![](随机性采样.png)

可以发现，随机性采样算法中多了许多超参数，比如 $\gamma$ 和 $S_{\text{noise}}$，在 $\gamma$ 中又包含了$S_{\text{churn}}$、$S_{\text{tmin}}$和$S_{\text{tmax}}$ 三个超参数，EDM 论文通过实验表明这些超参数对最终图像生成质量均有影响，并通过实验选择了一组最优的超参数。设置这些超参数的原因在于：噪声的多次随机注入会影响解 $\boldsymbol{x}_t$ 的图像分布，使得图像出现过饱和、失真等现象，调整超参数的值可以缓解 “回退” 操作带来的负面影响。

# EDM 论文的核心贡献

- 采样策略的改良：
    - 高阶求解器：引入了二阶的 Runge-Kutta 方法，可以用更少的步数达到更高的精度。
    - 非线性采样步长：发现采样时不应该匀速减少噪声，而应该在噪声较小时走得更慢。
    - 随机性分析：讨论了在采样过程中加入多少“随机扰动”最合适，发现适度的随机性有助于修正采样过程中的累积误差。
- 训练动力学的优化
    - 标准缩放：设计了一套公式，确保无论在哪个噪声水平（$\sigma$）下，神经网络的输入、输出以及目标值的方差都保持在 1 左右。这大大降低了训练难度。
    - 损失函数加权：提出了更合理的噪声分布采样方式（Log-normal 分布），让模型重点学习那些对视觉质量影响最大的噪声区间。
- 设计空间的全面梳理
    - 将 DDPM、NCSN 等前人的模型全部纳入了自己的统一框架下，并明确了哪些参数是互相关的，哪些是可以独立调优的。
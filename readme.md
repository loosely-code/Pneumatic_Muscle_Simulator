# Pneumatic Muscle Simulator

这是一个气动人工肌肉(Pneumatic Artificial Muscle)的模拟器, 基于 Three Element  Phenomenological model.

## 项目架构

- package: muscle_sim
  - moudle: muscle_env: 构建气动肌肉的仿真环境
    - class: 3Element_env
      - method: init
        - 初始化类,定义采样时间,初始载荷
      - method: reset
        - 初始化全局时间,外载荷,系统状态向量,收缩状态
      - method: step
        - 按采样时间为步长计算系统状态, 输入$k$时刻气压,外载荷,输出$k+1$时刻系统状态向量
      - method: calculate_coe
        - 根据输入气压和系统状态,计算气动肌肉模型参数,输入气压与收缩状态,输出 $K$,$B$,$F$
  - moudle: signal_gen: 产生各类参考信号
    - class signal_sin: 正弦信号
      - method:init: 设定信号幅值,中心,周期
      - method:step: 以采样时间为步长更新输出
      - method:search: 输出全局时间,计算此时的信号值
- package:controller_sim:各式控制器
  - moudle:controller_pid
    - class:PID_increment:增量PID控制器
  - moudle:reinforement_learning
    - class:DDPG
    - class:TD3
- script_dir: demos
  - script: trajectory_pid: PID 控制实现肌肉位置控制
  - script: trajectory_ddpg: ddpg 控制实现肌肉位置控制

## 模型数学推导

气动肌肉三元素模型:

$$
M \ddot{x} + B \dot{x} + K x = F - L \\
F = F(p) \\
B = B(p) \\
K = K(p) \\
$$

系统离散化:

$$
\begin{align}
\left\{\begin{matrix}
\frac{\mathrm{d}x}{\mathrm{d}t} =&  \frac{x(k+1)-x(k)}{T}\\
\frac{\mathrm{d^2}x}{\mathrm{d}t^2} =&  \frac{x(k+1)-2x(k)+x(k-1)}{T^2}\\
\end{matrix}\right.
\end{align}
$$

离散后系统模型:

$$
x(k+1) = \frac{T^2(F-L)-(KT^2-2M-BT)x(k)-Mx(k-1)}{M+BT}
$$

离散化系统状态向量:

$$
\mathbf{x}(k) = \vec{x}(k) =
\begin{bmatrix}
x(k)\\
x(k-1)
\end{bmatrix}
$$

模型参数:

$$
\begin{align}
K = \left\{\begin{matrix}
& 32.7 − 0.0321P \ ,&& 150 ≤ P ≤ 314 kPa \\
& 17 + 0.0179P \ ,&& 314 ≤ P ≤ 550 kPa
\end{matrix}\right.\\
\end{align}
$$

$$
\begin{align}
B =\left\{\begin{matrix}
& 2.90 \ , && 150 ≤ P ≤ 550 kPa \ (\text{contraction})\\
& 1.57 \ ,&& 150 ≤ P ≤ 372 kPa \ (\text{relaxation}) \\
& 0.311 + 0.00338P \ ,&& 372 ≤ P ≤ 550 kPa \ (\text{relaxation})
\end{matrix}\right.\\
\end{align}
$$

$$
F= 2.91P+44.6
$$

量纲:

- Force: $N$
- Time: $S$
- length: $mm$
- Pressure: $kPa$
- Mass: $10^3kg$

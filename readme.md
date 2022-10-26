# Pneumatic Muscle Simulator

这是一个气动人工肌肉(Pneumatic Artificial Muscle)的模拟器, 基于 Three Element  Phenomenological model.

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

模型参数:

$$
\begin{align}
K=&\left\{\begin{matrix}
=& 32.7 − 0.0321P \ ,&& 150 ≤ P ≤ 314 kPa \\
=& 17 + 0.0179P \ ,&& 314 ≤ P ≤ 550 kPa
\end{matrix}\right.\\
\\
B=& \left\{\begin{matrix}
=& 2.90 \ , && 150 ≤ P ≤ 550 kPa \ (\text{inflation})\\
=& 1.57 \ ,&& 150 ≤ P ≤ 372 kPa \ (\text{deflation}) \\
=& 0.311 + 0.00338P \ ,&& 372 ≤ P ≤ 550 kPa \ (\text{deflation})
\end{matrix}\right.\\
\\
F=& 2.91P+44.6
\end{align}
$$

量纲:

- Force: $N$
- Time: $S$
- length: $mm$
- Pressure: $kPa$
- Mass: $10^3kg$

# Gradients and Hessians: From First-Order Sensitivity to Second-Order Curvature

**A technical essay for pre-graduate students in mathematics and machine learning.**
*Assumed background: single- and multivariable calculus (partial derivatives, the chain rule, Taylor's theorem) and linear algebra (vectors, matrices, eigenvalues, positive-definiteness).*

---

## 1. Motivation: why these two objects?

When we move from a function of one variable $f:\mathbb{R}\to\mathbb{R}$ to a function of several variables $f:\mathbb{R}^n\to\mathbb{R}$, the single derivative $f'(x)$ is replaced by *two distinct objects* that play different roles:

- the **gradient** $\nabla f(x)$ — a vector of first partial derivatives;
- the **Hessian** $\nabla^2 f(x)$ — a matrix of second partial derivatives.

The gradient tells us *which direction* the function increases fastest and *how fast*. The Hessian tells us *how the slope itself is changing* — the local curvature — which is essential for classifying critical points, proving convexity, and building efficient optimization algorithms (gradient descent vs. Newton's method). The two objects are linked: the Hessian is nothing but the Jacobian of the gradient.

This essay develops both objects from definitions through concrete computations to their central applications, verifying every claim with explicit calculation.

---

## 2. The gradient

### 2.1 Definition

Let $f:\mathbb{R}^n\to\mathbb{R}$ be differentiable at $a=(a_1,\dots,a_n)$. The **gradient** of $f$ at $a$ is the vector of first partial derivatives

$$
\nabla f(a)=\left(\frac{\partial f}{\partial x_1}(a),\dots,\frac{\partial f}{\partial x_n}(a)\right)^{\!T}\in\mathbb{R}^n .
$$

**Key geometric fact.** $\\nabla f(a)$ points in the direction of *steepest ascent* of $f$ at $a$, and $\\|\\nabla f(a)\\|$ is the rate of increase in that direction. Equivalently, for any vector $u$ (not necessarily unit),

$$
D_u f(a)=\nabla f(a)\cdot u \;\le\; \|\nabla f(a)\|\,\|u\|,
$$

where the inequality is Cauchy–Schwarz. For a **unit vector** ($\\|u\\|=1$) this reads $D_u f(a)=\nabla f(a)\cdot u\le\|\nabla f(a)\|$, with equality when $u=\nabla f(a)/\|\nabla f(a)\|$. Here $D_u f(a)$ is the directional derivative.

### 2.2 The gradient as a row vector, a column vector, and a linear map

A subtlety worth flagging for the mathematically careful reader: the *derivative* $Df(a)$ is a **linear map** $\mathbb{R}^n\to\mathbb{R}$, i.e., a row vector (covector)

$$
Df(a)=\left(\frac{\partial f}{\partial x_1}(a),\dots,\frac{\partial f}{\partial x_n}(a)\right).
$$

The gradient $\nabla f(a)$ is the *column-vector representative* of this covector with respect to the standard Euclidean inner product: $\nabla f(a) = (Df(a))^{T}$. In coordinates with the standard dot product these are interchangeable, but conceptually the gradient is the metric-dual of the derivative. We will see the same pattern again with the Hessian.

### 2.3 Worked examples (verified by computation)

**Example 1.** $f(x,y,z)=x^2+y^2+z^2$. Then

$$
\nabla f=(2x,\,2y,\,2z),
$$

which points radially outward from the origin — exactly the direction of steepest increase of the squared distance from the origin.

**Example 2.** $f(x,y)=x^2+3xy+y^2$. Then

$$
\nabla f=(2x+3y,\;3x+2y).
$$

At the point $(1,1)$ this gives $\nabla f(1,1)=(5,5)$, pointing along the diagonal.

**Example 3.** $f(x,y)=x^2y+\sin(x)$. Then

$$
\nabla f=\bigl(2xy+\cos(x),\;x^2\bigr).
$$

Notice the asymmetry: the $x$-component mixes $x$ and $y$ (through the $xy$ term) while the $y$-component depends only on $x$. This asymmetry foreshadows the structure of the Hessian.

---

## 3. The Hessian

### 3.1 Definition

Let $f:\mathbb{R}^n\to\mathbb{R}$ be twice differentiable at $a$. The **Hessian matrix** of $f$ at $a$ is the $n\times n$ matrix of second partial derivatives

$$
\nabla^2 f(a)=\left(\frac{\partial^2 f}{\partial x_i\,\partial x_j}(a)\right)_{i,j=1}^{n}.
$$

**Symmetry.** By Clairaut's theorem (equality of mixed partials), assuming the second partials are continuous, $\partial^2 f/\partial x_i\partial x_j=\partial^2 f/\partial x_j\partial x_i$, so the Hessian is **symmetric**. This is not a cosmetic fact: it guarantees real eigenvalues and an orthogonal diagonalization, which is what makes the Hessian so useful for definiteness tests.

**The Hessian as the Jacobian of the gradient.** If we view $\nabla f:\mathbb{R}^n\to\mathbb{R}^n$ as a vector-valued function, its Jacobian matrix is

$$
J(\nabla f)=\left(\frac{\partial (\nabla f)_i}{\partial x_j}\right)_{i,j}=\left(\frac{\partial^2 f}{\partial x_j\,\partial x_i}\right)_{i,j}=\nabla^2 f,
$$

where the last equality uses symmetry. So: **Hessian = Jacobian of gradient.** This is the clean way to remember which second derivatives go where.

### 3.2 Worked examples (verified by computation)

**Example 4.** $f(x,y,z)=x^2+y^2+z^2$. The Hessian is constant:

$$
\nabla^2 f=\begin{pmatrix}2&0&0\\0&2&0\\0&0&2\end{pmatrix}=2I.
$$

Constant Hessian means the function is exactly quadratic — there is no higher-order structure to capture.

**Example 5.** $f(x,y)=x^2+3xy+y^2$. Then

$$
\nabla^2 f=\begin{pmatrix}2&3\\3&2\end{pmatrix},\qquad \det(\nabla^2 f)=4-9=-5.
$$

The negative determinant signals that this quadratic is *indefinite* — a saddle shape, not a bowl. (We return to this in §4.)

**Example 6.** $f(x,y)=x^3+y^3-3xy$. Then

$$
\nabla f=(3x^2-3y,\;3y^2-3x),\qquad \nabla^2 f=\begin{pmatrix}6x&-3\\-3&6y\end{pmatrix},
$$

with $\det(\nabla^2 f)=36xy-9$. Note the Hessian now *depends on the point* $(x,y)$ — the curvature varies across the surface. This is the hallmark of a genuinely nonlinear function.

**Example 7.** $f(x,y)=x^2y+\sin(x)$. Then

$$
\nabla^2 f=\begin{pmatrix}2y-\sin(x)&2x\\2x&0\end{pmatrix},\qquad \det(\nabla^2 f)=-4x^2\le 0.
$$

The determinant is always non-positive, and vanishes on the entire $y$-axis ($x=0$). A vanishing Hessian determinant does *not* imply the function is linear — here the function is manifestly nonlinear, yet the Hessian degenerates. This is a warning against over-reading determinant signs at degenerate points.

---

## 4. The Hessian and the classification of critical points

### 4.1 Critical points

A point $a$ is a **critical point** of $f$ if $\nabla f(a)=0$ (or if $f$ is not differentiable there). Critical points are the only candidates for local extrema **in the interior of the domain** — this is the multivariable analogue of Fermat's theorem. (On the boundary of a closed domain one must also examine the boundary separately; throughout this essay we work on open sets such as $\mathbb{R}^n$, where every point is interior.)

For Example 6, setting $\nabla f=0$ gives $3x^2-3y=0$ and $3y^2-3x=0$, i.e., $y=x^2$ and $x=y^2$. The real solutions are $(0,0)$ and $(1,1)$ — both are critical points.

### 4.2 The second-derivative test via definiteness

The Hessian classifies critical points through its **definiteness**:

- **Positive definite** Hessian ($\nabla^2 f(a)\succ 0$: all eigenvalues $>0$, equivalently $v^T\nabla^2 f(a)v>0$ for all $v\neq 0$) $\Rightarrow$ **strict local minimum**.
- **Negative definite** Hessian ($\nabla^2 f(a)\prec 0$: all eigenvalues $<0$) $\Rightarrow$ **strict local maximum**.
- **Indefinite** Hessian (eigenvalues of mixed sign) $\Rightarrow$ **saddle point**. In 2D this is equivalent to $\det<0$, but **in $n>2$ a negative determinant does *not* imply indefiniteness** — a negative-definite matrix has a negative determinant (its sign is $(-1)^n$). For $n>2$ one must inspect the eigenvalues (or the signs of the leading principal minors via Sylvester's criterion), never the determinant alone.
- **Semidefinite or singular** Hessian $\Rightarrow$ the test is **inconclusive** (higher-order terms decide).

**Why this works — the quadratic Taylor approximation.** Taylor's theorem to second order at $a$ says

$$
f(a+h)=f(a)+\nabla f(a)\cdot h+\tfrac12\,h^T\nabla^2 f(a)\,h+o(\|h\|^2).
$$

At a critical point $\nabla f(a)=0$, so locally $f(a+h)-f(a)\approx \tfrac12 h^T\nabla^2 f(a)h$. The sign of the quadratic form $h^T\nabla^2 f(a)h$ therefore determines whether $f$ increases, decreases, or does both (saddle) as we move away from $a$. This is the precise mechanism behind the test.

### 4.3 Applying the test

**Example 6 revisited.** At $(1,1)$: $\nabla^2 f(1,1)=\begin{pmatrix}6&-3\\-3&6\end{pmatrix}$, with eigenvalues $3$ and $9$ (both positive), $\det=27>0$, trace $=12>0$. **Positive definite $\Rightarrow$ strict local minimum.** At $(0,0)$: $\nabla^2 f(0,0)=\begin{pmatrix}0&-3\\-3&0\end{pmatrix}$, with $\det=-9<0$ — **indefinite $\Rightarrow$ saddle point.** The same function exhibits both behaviors at different critical points, illustrating that definiteness is a *local* property.

**Example 5 revisited.** $f(x,y)=x^2+3xy+y^2$ has a single critical point at $(0,0)$ with Hessian $\begin{pmatrix}2&3\\3&2\end{pmatrix}$, $\det=-5<0$ — a saddle, so this quadratic has no local extremum at all. Indeed, along the direction $(1,-1)$ we have $f(t,-t)=-t^2$, which decreases in both directions from the origin.

**Example 7.** At any point with $x\neq 0$, $\det(\nabla^2 f)=-4x^2<0$, so the Hessian is indefinite — the surface $z=x^2y+\sin(x)$ has no local extrema anywhere off the $y$-axis. On the $y$-axis the test is inconclusive (degenerate Hessian), and one must examine higher-order terms or the gradient direction directly.

---

## 5. Convexity and the Hessian

A function $f$ is **convex** if for all $x,y$ and $t\in[0,1]$,

$$
f(tx+(1-t)y)\le t\,f(x)+(1-t)\,f(y).
$$

For twice-differentiable $f$, convexity has a clean second-order characterization:

> **Theorem (convexity).** $f$ is convex on an open convex set iff $\nabla^2 f(x)$ is positive **semidefinite** for every $x$ in the set.
>
> **Corollary (strict convexity).** If $\nabla^2 f(x)$ is positive **definite** for every $x$ in the set, then $f$ is *strictly* convex. The converse is **false**: positive definiteness of the Hessian is sufficient but **not necessary** for strict convexity.

**Proof sketch (convexity).** $(\Rightarrow)$ The restriction of a convex function to any line is a convex function of one variable; applying the one-variable second-derivative test along the line $x+tv$ gives $v^T\nabla^2 f(x)v\ge 0$ for all $v$. $(\Leftarrow)$ Integrating $v^T\nabla^2 f\,v\ge 0$ twice along the segment from $x$ to $y$ yields the convexity inequality. $\square$

**Why the corollary is only one-way — a counterexample.** Consider $f(x)=x^4$ on $\mathbb{R}$. This function is *strictly* convex: for $x\neq y$ and $t\in(0,1)$, the chord lies strictly above the graph because $x^4$ is strictly convex on each side of the origin and the two pieces join with matching slope $0$ at $x=0$. Yet its second derivative is $f''(x)=12x^2$, which vanishes at $x=0$. Hence $\nabla^2 f$ is positive semidefinite everywhere but **not** positive definite everywhere. Strict convexity therefore only guarantees $\nabla^2 f(x)\succeq 0$; the stronger condition $\nabla^2 f(x)\succ 0$ is merely a sufficient certificate. (Formally, strict convexity is equivalent to $v^T\nabla^2 f(x)v\ge 0$ everywhere with the inequality strict on a set of positive measure, but the pointwise-definite version fails as this example shows.)

**Example.** $f(x,y)=x^2+y^2$ has $\nabla^2 f=2I\succ 0$ everywhere — strictly convex, with a unique global minimum at the origin. $f(x,y)=x^2+3xy+y^2$ has Hessian with eigenvalues $5$ and $-1$ — indefinite everywhere, so the function is neither convex nor concave on $\mathbb{R}^2$.

**Why convexity matters.** For a convex function, *every* local minimum is a global minimum, and any critical point ($\nabla f=0$) is automatically a global minimizer. This collapses the search for global optima to finding critical points — a profound simplification that underlies enormous swathes of optimization theory and machine learning.

---

## 6. The Hessian in optimization: gradient descent vs. Newton's method

### 6.1 Gradient descent

Gradient descent minimizes $f$ by iterated update

$$
x_{k+1}=x_k-\alpha_k\,\nabla f(x_k),
$$

where $\alpha_k>0$ is a step size (learning rate). The direction $-\nabla f(x_k)$ is the direction of steepest descent. The method uses only *first-order* information.

**Convergence rate.** For $f$ convex and $L$-smooth ($\|\nabla f(x)-\nabla f(y)\|\le L\|x-y\|$) and $\mu$-strongly convex ($\nabla^2 f(x)\succeq \mu I$), gradient descent with $\alpha_k=1/L$ converges linearly: $\|x_k-x^*\|^2\le \bigl(1-\tfrac{\mu}{L}\bigr)^k\|x_0-x^*\|^2$. The ratio $\kappa=L/\mu$ is the **condition number** — the ratio of largest to smallest eigenvalue of the Hessian. A badly conditioned Hessian (large $\kappa$) makes gradient descent crawl.

*(A note for the advanced reader: the step size $\alpha=1/L$ above is a convenient, safe choice. The *optimal constant* step size is $\alpha=\tfrac{2}{L+\mu}$, which yields the slightly better rate $\bigl(\tfrac{L-\mu}{L+\mu}\bigr)^{2k}$. The standard bound $1-\mu/L$ is perfectly adequate for this level.)*

### 6.2 Newton's method

Newton's method uses *second-order* information by approximating $f$ with its quadratic Taylor model and jumping to the minimizer of that model:

$$
x_{k+1}=x_k-\bigl[\nabla^2 f(x_k)\bigr]^{-1}\nabla f(x_k).
$$

**Derivation.** Minimize the quadratic model $Q(h)=f(x_k)+\nabla f(x_k)\cdot h+\tfrac12 h^T\nabla^2 f(x_k)h$ over $h$. Setting $\nabla Q(h)=\nabla f(x_k)+\nabla^2 f(x_k)h=0$ and solving for $h$ gives $h^*=-[\nabla^2 f(x_k)]^{-1}\nabla f(x_k)$. $\square$

**Why it's better.** Near a nondegenerate minimizer, Newton's method converges **quadratically**: $\|x_{k+1}-x^*\|\le C\|x_k-x^*\|^2$. Each step roughly *doubles* the number of correct digits. Gradient descent, by contrast, only converges linearly.

**The trade-off.** The Hessian inverse is expensive: forming $\nabla^2 f$ costs $O(n^2)$ and inverting it $O(n^3)$, prohibitive for large $n$. Moreover, if $\nabla^2 f$ is singular or indefinite, the Newton direction is ill-defined or ascent-like. This motivates **quasi-Newton** methods (BFGS, L-BFGS) that build an approximation to the Hessian inverse from gradient history at $O(n)$ memory per step, and **damped/regularized** Newton methods that replace $\nabla^2 f$ by $\nabla^2 f+\lambda I$ to ensure positive definiteness.

### 6.3 A concrete comparison: Newton on a convex quadratic vs. gradient descent (verified by computation)

To isolate the *benefit* of second-order information, we use a **positive-definite** quadratic, where a true global minimum exists. Take $f(x,y)=x^2+xy+y^2$, with $\nabla f=(2x+y,\,x+2y)$ and $\nabla^2 f=\begin{pmatrix}2&1\\1&2\end{pmatrix}$. Start at $(1,0)$, where $\nabla f=(2,1)$.

- **One Newton step:** $x_1=(1,0)-\begin{pmatrix}2&1\\1&2\end{pmatrix}^{-1}(2,1)=(0,0)$ — lands exactly on the global minimum in a single step (because $f$ is exactly quadratic, its Hessian is exact, and the quadratic model is exact).
- **One gradient step with $\alpha=0.1$:** $x_1=(1,0)-0.1(2,1)=(0.8,-0.1)$ — still far from the minimum, and it has moved in a direction that is *not* directly toward the optimum.

This stark contrast — one step vs. many — is the entire story of first- vs. second-order optimization **on a well-behaved convex problem**.

> **Warning: Newton on a saddle point (why pure Newton is dangerous in non-convex ML).**
>
> Basic Newton's method is a *root-finding* algorithm applied to the system $\nabla f(x)=0$; it blindly jumps to the nearest critical point, regardless of whether that point is a minimum, maximum, or saddle. Consider the indefinite quadratic $f(x,y)=x^2+3xy+y^2$ from Example 5, whose only critical point $(0,0)$ is a **saddle** (Hessian $\begin{pmatrix}2&3\\3&2\end{pmatrix}$ has eigenvalues $5$ and $-1$). Starting at $(1,1)$, where $\nabla f=(5,5)$:
>
> $$x_1=(1,1)-\begin{pmatrix}2&3\\3&2\end{pmatrix}^{-1}(5,5)=(0,0).$$
>
> Newton lands exactly on the saddle in one step. It has *not* found a minimizer — the quadratic is unbounded below (along $(1,-1)$, $f(t,-t)=-t^2\to-\infty$), so no minimum exists. For machine-learning students this is Newton's method's **greatest weakness**, not a victory: in non-convex optimization (like deep learning) saddle points are everywhere, and pure Newton gets dangerously trapped by them. Gradient descent, by contrast, does not converge to saddles in the generic case — it either escapes them or, for an un-bounded-below quadratic like this one, diverges to $-\infty$ along a descent direction (which is the behavior one *wants* when no minimizer exists). This is precisely why practical second-order methods require **regularization**: adding $\lambda I$ to the Hessian (damped/regularized Newton) to force positive definiteness, or using quasi-Newton methods (BFGS/L-BFGS) that build a positive-definite curvature model from gradient history.

---

## 7. Further directions

- **The Laplacian** $\Delta f=\operatorname{tr}(\nabla^2 f)=\sum_i \partial^2 f/\partial x_i^2$ is the trace of the Hessian; it governs diffusion and appears in the heat equation and in graph neural networks.
- **Hessian-free optimization** uses conjugate-gradient iterations to apply the Hessian inverse to a vector *without ever forming the Hessian explicitly*, recovering Newton's quadratic convergence at $O(n)$ memory.
- **Hessian spectra in deep learning:** for over-parameterized networks the Hessian is dominated by a large zero-eigenvalue subspace (the "flat directions" corresponding to symmetries), with a small number of large eigenvalues — this spectral structure is central to understanding generalization and to designing adaptive optimizers like Adam, which can be viewed as a diagonal (per-coordinate) approximation to Newton's method.
- **Matrix calculus notation:** in ML it is standard to write gradients as column vectors and Hessians as the Jacobian of the gradient, matching the convention used throughout this essay.

---

## 8. Summary

| Concept | Object | Encodes | Key application |
|---|---|---|---|
| **Gradient** $\nabla f$ | vector of first partials | direction & rate of steepest ascent | first-order optimization (gradient descent) |
| **Hessian** $\nabla^2 f$ | symmetric matrix of second partials | local curvature / quadratic Taylor coefficient | second-order optimization (Newton's method), convexity tests, critical-point classification |

The gradient is the derivative's vector representative; the Hessian is the Jacobian of the gradient. Together they form the first two terms of the multivariable Taylor expansion, which is the engine behind both the second-derivative test for extrema and Newton's method. Mastery of these objects is a prerequisite for serious study of optimization, numerical analysis, and the mathematics of machine learning.

---

## Appendix: verification of all computations

All symbolic computations in this essay were verified with **SymPy 1.14.0** (Python) in the workspace virtual environment. Key results confirmed by direct computation:

- $\nabla(x^2+y^2+z^2)=(2x,2y,2z)$; $\nabla^2=2I$.
- $\nabla(x^2+3xy+y^2)=(2x+3y,\,3x+2y)$; $\nabla^2=\begin{pmatrix}2&3\\3&2\end{pmatrix}$, $\det=-5$, eigenvalues $5,-1$.
- $\nabla(x^2y+\sin x)=(2xy+\cos x,\,x^2)$; $\nabla^2=\begin{pmatrix}2y-\sin x&2x\\2x&0\end{pmatrix}$, $\det=-4x^2$.
- $\nabla(x^3+y^3-3xy)=(3x^2-3y,\,3y^2-3x)$; $\nabla^2=\begin{pmatrix}6x&-3\\-3&6y\end{pmatrix}$, $\det=36xy-9$; at $(1,1)$ $\det=27$, at $(0,0)$ $\det=-9$.
- General quadratic $ax^2+bxy+cy^2+dx+ey+f_0$: $\nabla^2=\begin{pmatrix}2a&b\\b&2c\end{pmatrix}$, $\det=4ac-b^2$, convex iff $a>0$ and $4ac-b^2>0$.
- Strict convexity counterexample: $f(x)=x^4$ is strictly convex but $f''(x)=12x^2=0$ at $x=0$, so the Hessian is positive semidefinite but not positive definite everywhere (positive definiteness is sufficient, not necessary, for strict convexity).
- Positive-definite quadratic $f=x^2+xy+y^2$: $\nabla^2=\begin{pmatrix}2&1\\1&2\end{pmatrix}$, $\det=3$, eigenvalues $1,3$ (positive definite). Newton from $(1,0)$: $(1,0)-\begin{pmatrix}2&1\\1&2\end{pmatrix}^{-1}(2,1)=(0,0)$ — global minimum in one step; GD with $\alpha=0.1$ gives $(0.8,-0.1)$.
- Indefinite quadratic $f=x^2+3xy+y^2$: Newton from $(1,1)$ gives $(1,1)-\begin{pmatrix}2&3\\3&2\end{pmatrix}^{-1}(5,5)=(0,0)$ — a saddle, not a minimizer (un-bounded below; $f(t,-t)=-t^2$).

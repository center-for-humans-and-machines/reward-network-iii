# Task Families, Coverage Payoffs, Batch Core Extraction, and Cultural Transmission

## 1. Overview

We model cultural adaptation in environments where tasks have latent structure (*task families*) and strategies are binary action patterns (bitstrings). A strategy earns reward only if it *covers* the task (bitwise). Task rewards increase exponentially with task complexity, while strategies incur costs increasing in the number of activated bits.

The all-zero strategy $000\ldots 0$ is always available, cost-free, and may emerge endogenously as the population default.

Agents operate in two phases:

1. **Learning phase:** agents explore and collect experience. At the end of learning, they extract a discrete **core strategy** from successful experiences (core extraction by counting).
2. **Evaluation phase:** agents execute a strategy drawn from their individual learning experience. Only evaluation performance determines social selection and transmission.

Machines differ from humans only in having longer learning phases. Evaluation time is fixed for all agents.

Transmission is stochastic and depends on the compressibility of the transmitted core strategy, operationalized via run-length encoding (RLE).

---

## 2. Spaces and Notation

- Ambient dimension: $K \in \mathbb{N}$.
- Tasks: $t \in \{0,1\}^K$.
- Strategies: $s \in \{0,1\}^K$.
- Bitwise AND: $(s \wedge t)_k = s_k t_k$.
- Number of active bits: $\|x\|_1 = \sum_{k=1}^K x_k$.

---

## 3. Task Families: Global and Individual

### 3.1 Global family

A population (or epoch) has a latent global family parameter
$$
p^{(G)} = (p^{(G)}_1,\dots,p^{(G)}_K), \quad p^{(G)}_k \in (0,1),
$$
sampled once as
$$
p^{(G)}_k \sim \mathrm{Beta}(a,b) \quad \text{i.i.d.}
$$

### 3.2 Individual family

Each individual $i$ experiences tasks from an individual family
$$
p^{(i)}_k = (1-\delta)\, p^{(G)}_k + \delta\, \tilde p^{(i)}_k,
$$
where
$$
\tilde p^{(i)}_k \sim \mathrm{Beta}(a,b) \quad \text{i.i.d.}, \qquad \delta \in [0,1].
$$

### 3.3 Task sampling

On each trial for individual $i$,
$$
t_k \sim \mathrm{Bernoulli}(p^{(i)}_k), \quad k=1,\dots,K.
$$

---

## 4. Success Condition and Payoffs

### 4.1 Coverage condition

A strategy $s$ solves a task $t$ if
$$
(s \wedge t) = t
\quad \Longleftrightarrow \quad
\forall k,\; t_k = 1 \Rightarrow s_k = 1.
$$

### 4.2 Task reward

Define task complexity as $m(t)=\|t\|_1$. The reward for solving task $t$ is
$$
R(t) = r\, \lambda^{m(t)}, \quad r>0,\; \lambda>1.
$$

### 4.3 Strategy cost

The cost of executing strategy $s$ is
$$
C(s) = c\,\|s\|_1, \quad c>0.
$$

### 4.4 Net payoff

$$
\pi(s,t) = \mathbf{1}\{(s \wedge t)=t\}\, R(t) - C(s).
$$

---

## 5. Strategy Sampling Parameters

Each agent $i$ is characterized by:
- a learning duration $T_{\mathrm{learn}}^{(i)}$,
- a common evaluation duration $T_{\mathrm{eval}}$,
- an **individual exploration baseline** $p_{\mathrm{ind}} \in (0,1)$,
- a **social weighting parameter** $\omega \in [0,1]$,
- a socially inherited representation $u_i \in \{0,1\}^K$, initially $u_i=\mathbf{0}$ unless inherited.

### 5.1 Individual exploration baseline

The individual baseline induces a flat exploration distribution:
$$
p^{\mathrm{ind}}_k = p_{\mathrm{ind}} \quad \forall k.
$$

### 5.2 Socially weighted exploration (updated)

Let $p^{\mathrm{learn}}_k = (u_i)_k$ denote the bitwise distribution induced by the inherited representation.  
The effective sampling probabilities during learning are
$$
q_{i,k} = \omega\, p^{\mathrm{learn}}_k + (1-\omega)\, p_{\mathrm{ind}},
\quad k=1,\dots,K.
$$

---

## 6. Learning Phase

For each learning trial $\tau = 1,\dots,T_{\mathrm{learn}}^{(i)}$:
1. Sample a task $t^{(\tau)} \sim \mathcal{D}(p^{(i)})$.
2. Sample a strategy $s^{(\tau)}$ bitwise:
   $$
   s^{(\tau)}_k \sim \mathrm{Bernoulli}(q_{i,k}), \quad k=1,\dots,K.
   $$
3. Observe payoff $\pi^{(\tau)} = \pi(s^{(\tau)}, t^{(\tau)})$.

The agent stores the set of experienced strategies and associated payoffs.

---

## 7. Core Extraction

### 7.1 Successful set

Define
$$
S^+ = \{\tau : \pi^{(\tau)} > 0\}.
$$

### 7.2 Feature frequencies

If $|S^+|>0$, compute
$$
\hat p_k = \frac{1}{|S^+|}\sum_{\tau \in S^+} t^{(\tau)}_k.
$$

### 7.3 Thresholding

Fix $\theta \in (0,1)$. Define
$$
(u_i^\star)_k = \mathbf{1}\{\hat p_k \ge \theta\}.
$$

If $|S^+|=0$, set $u_i^\star = \mathbf{0}$.

The extracted core $u_i^\star$ is the agent’s transmissible cultural representation.

---

## 8. Evaluation Phase

Evaluation is based on the extracted core strategy from the agent’s own learning phase.

For each evaluation trial $\tau = 1,\dots,T_{\mathrm{eval}}$:
1. Sample a task $t^{(\tau)} \sim \mathcal{D}(p^{(i)})$.
2. Execute the extracted strategy $s = u_i^\star$.
3. Record payoff $\pi(u_i^\star, t^{(\tau)})$.

Define evaluation performance:
$$
\mathrm{Perf}_i = \frac{1}{T_{\mathrm{eval}}}
\sum_{\tau=1}^{T_{\mathrm{eval}}} \pi(u_i^\star, t^{(\tau)}).
$$`

---

## 9. Cultural Transmission

### 9.1 Teacher selection

Each learner selects a teacher $j$ with probability
$$
\Pr(j) \propto \exp(\beta\, \mathrm{Perf}_j),
$$
with $\beta \ge 0$.

### 9.2 Transmission as RLE-dependent noisy copying

Transmission is modeled as noisy copying of the teacher’s extracted core $u_j^\star$.  
The learner receives a corrupted version $\tilde u_i \in \{0,1\}^K$ via an independent bit-flip channel:
$$
\tilde u_{i,k} =
\begin{cases}
u_{j,k}^\star & \text{w.p. } 1-\varepsilon(u_j^\star),\\
1-u_{j,k}^\star & \text{w.p. } \varepsilon(u_j^\star),
\end{cases}
\quad k=1,\dots,K.
$$

The flip probability increases with representational complexity:
$$
\varepsilon(u) = \sigma\!\left(\gamma\bigl(\mathrm{RLE}(u)-\alpha\bigr)\right),
\qquad
\sigma(x)=\frac{1}{1+e^{-x}}.
$$

The received vector $\tilde u_i$ biases exploration during the learner’s subsequent learning phase.


### 9.3 Transmission of exploration parameters

The parameters $(p_{\mathrm{ind}}, \mathrm{si})$ may be fixed across agents or inherited with mutation:
$$
p_{\mathrm{ind},i} = \mathrm{clip}(p_{\mathrm{ind},j} + \xi_{\mathrm{ind}}, \epsilon, 1-\epsilon),
$$
$$
\mathrm{si}_i = \mathrm{clip}(\mathrm{si}_j + \xi_{\mathrm{si}}, 0, 1),
$$
with small zero-mean noise.

---

## 10. Humans and Machines

Humans and machines share the same task environment, payoff function, learning procedure, evaluation protocol, and transmission dynamics.

They differ only in learning duration:
- Humans: $T_{\mathrm{learn}}^{(H)}$.
- Machines: $T_{\mathrm{learn}}^{(M)} \gg T_{\mathrm{learn}}^{(H)}$.

Evaluation duration $T_{\mathrm{eval}}$ is identical for all agents.

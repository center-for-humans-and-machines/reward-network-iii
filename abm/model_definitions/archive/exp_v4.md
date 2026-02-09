# Simulation Experiments

This section describes a **set of simulation experiments** designed to replace and supplement the original agent-based model. The experiments isolate how **finite-time innovation supply**, **heterogeneity in exploration horizons**, and **selective cultural transmission** jointly determine population-level cultural outcomes.

All agents are identical on a per-trial basis. Differences in outcomes arise exclusively from differences in the **number of learning trials available** and from social transmission dynamics.

---

## Common Model Components

### Solutions and Payoffs
Agents may end learning with one of three mutually exclusive outcomes:
- $O$ (optimal, high payoff)
- $M$ (moderate, lower payoff)
- $\varnothing$ (no solution)

Payoffs satisfy:
$$
R_O > R_M > R_\varnothing \, .
$$

### Individual Learning
Each agent $i$ has an exploration horizon $T_i$ (number of learning trials).

On each trial, the agent independently discovers:
- $O$ with probability $\pi_O$,
- $M$ with probability $\pi_M$,
- nothing otherwise,

with $\pi_O + \pi_M \leq 1$.

After $T_i$ trials, the agent retains the best discovered solution. The resulting probabilities are:
$$
p_O(T_i) = 1 - (1-\pi_O)^{T_i}
$$
$$
p_M(T_i) = (1-\pi_O)^{T_i}\left[1-(1-\pi_M)^{T_i}\right]
$$
$$
p_\varnothing(T_i) = (1-\pi_O-\pi_M)^{T_i}
$$

### Social Learning
- Population size: $N=8$ (lab-mimic); robustness at larger $N$
- Generations: $G=5$
- Teacher slate size: $K=5$
- Payoff bias strength: $\alpha$
- Transmission fidelity: $\tau$

Each learner samples $K$ teachers uniformly from the previous generation and chooses among them with probability:
$$
P(i \leftarrow j) = 
\frac{\exp(\alpha R_{s_j})}{\sum_{k \in \mathcal{C}_i} \exp(\alpha R_{s_k})} \, .
$$

The chosen teacher’s solution is transmitted with probability $\tau$; otherwise the learner receives $\varnothing$.

---

## Experiment 1 — Dominance of the Exploration-Horizon Tail

### Goal
Test whether cultural outcomes depend primarily on the **right tail of the exploration-horizon distribution**, rather than on the population-average exploration capacity.

### Design
- Population: $N=8$, $G=5$, $K=5$
- Mean exploration horizon fixed at $\bar{T}$
- Per-trial discovery probabilities $(\pi_O,\pi_M)$ fixed
- No agent differs in per-trial behavior or cognition

### Manipulation: Distribution of $T_i$
We compare populations that differ only in the **tail weight** of the $T$ distribution, while holding the mean constant.

#### Two-point mixture (main text)
$
T_i =
\begin{cases}
T_L & \text{with probability } 1-p \\
T_H & \text{with probability } p
\end{cases}
\quad \text{with} \quad
(1-p)T_L + pT_H = \bar{T}
$

- $T_H \gg \bar{T}$ fixed
- $p \in \{0, 0.05, 0.1\}$ varied
- $T_L$ adjusted to keep $\mathbb{E}[T_i]=\bar{T}$

Interpretation:
- $p=0$: homogeneous population
- $p>0$: rare high-throughput agents (interpretable as machines or extreme access to exploration capacity)

#### Lognormal distribution (SI robustness)
$$
T_i \sim \text{LogNormal}(\mu(\sigma),\sigma)
\quad \text{with} \quad
\mathbb{E}[T_i]=\bar{T}
$$
where $\mu(\sigma)=\ln(\bar{T})-\tfrac{1}{2}\sigma^2$ and $\sigma$ controls tail heaviness.

---

### Outcomes
- Probability that $O$ is introduced:
$$
P(O_{\text{intro}})
$$
- Probability that $O$ is present in the final generation:
$$
P(O_{\text{final}})
$$
- Probability that the introducer of $O$ lies in the top decile of $T_i$

---

### Key Result
Holding average exploration capacity fixed, increasing the mass of the right tail of the $T$ distribution sharply increases both the probability of innovation introduction and the thereby the average reward of successive generation. Cultural outcomes are disproportionately determined by rare high-$T$ agents.

---

## Experiment 2 — Diffusion Gate: Payoff Bias and Transmission Fidelity

### Goal
Isolate how cultural transmission parameters determine whether introduced innovations persist.

### Design
- Innovation introduction ensured by seeding a single $O$ agent in generation 0
- Homogeneous $T_i=\bar{T}$ for all agents
- $N=8$, $G=5$, $K=5$

### Manipulations
- Payoff bias: $\alpha \in \{0, 0.5, 1, 2, 4\}$
- Transmission fidelity: $\tau \in \{0.2, 0.4, 0.6, 0.8, 1.0\}$

### Outcomes
- Probability of persistence conditional on introduction:
$$
P(O_{\text{final}} \mid O_{\text{intro}})
$$
- Distribution of final population states

### Interpretation
Having shown that exploration capacity governs **whether** innovations appear, we now show that payoff bias and transmission fidelity govern **whether they persist**.

---

## Experiment 3 — Bounded Uplift Across Task Regimes

### Goal
Explain why the cultural impact of fast exploration is strongest in intermediate task regimes.

### Design
- Compare two conditions:
  - Baseline: homogeneous population with $T_i=\bar{T}$
  - Tail condition: rare high-$T$ agents (as in Experiment 1)
- $N=8$, $G=5$, $K=5$

### Manipulations
- Discovery difficulty: $\pi_O$ (scanned over a grid)
- Transmission fidelity: $\tau$ (scanned)

### Outcomes
- Difference in final mean payoff:
$$
\Delta \Pi = \mathbb{E}[\Pi_{\text{final}}^{\text{tail}}] - \mathbb{E}[\Pi_{\text{final}}^{\text{baseline}}]
$$
- Difference in $P(O_{\text{final}})$

### Interpretation
Fast explorers generate the largest cultural uplift when optimal solutions are rare but discoverable and when social transmission is sufficiently reliable.

---

## Robustness Analyses (Supplementary Information)

- **Distributional robustness**: lognormal vs two-point mixtures with matched means
- **Teacher access**: fixed $K=5$ vs scaled $K=\lceil 0.1N \rceil$ vs global access $K=N$
- **Population size**: $N \in \{8,16,32,100\}$

---

## Summary
Across experiments, cultural dynamics are governed by:
1. Finite learning horizons,
2. Heterogeneity in exploration capacity (especially the right tail),
3. Selective but imperfect cultural transmission.

These results clarify how a small number of high-throughput agents can induce persistent cultural change within finite time windows, without assuming per-trial cognitive differences.

# Model: Finite-Time Innovation Supply and Cultural Transmission

## Overview
We introduce a minimal generative model of cultural evolution designed to clarify how **heterogeneity in exploration time horizons**—rather than per-trial cognitive superiority—can shape population-level cultural outcomes. The model formalizes three ingredients:

1. **Finite-time individual learning**: agents explore an environment for a bounded number of trials.
2. **Innovation rarity**: high-payoff solutions are difficult to discover within limited horizons.
3. **Selective cultural transmission**: once discovered, solutions spread via payoff-biased social learning with imperfect transmission.

Crucially, **all agents are identical on a per-trial basis**. Differences in outcomes arise solely from heterogeneity in the **number of trials available** to agents.

---

## Environment and Solutions
The task environment contains three mutually exclusive solution states:

- $O$: optimal solution (high payoff, hard to discover)
- $M$: moderate solution (lower payoff, easier to discover)
- $\varnothing$: no solution discovered

Payoffs satisfy:
$$
R_O > R_M > R_\varnothing \, .
$$

---

## Individual Learning (Finite Horizon)

Each agent $i$ is endowed with an **exploration horizon** $T_i \in \mathbb{N}$, representing the number of learning trials they can execute.

On each trial, the agent independently discovers:
- $O$ with probability $\pi_O$,
- $M$ with probability $\pi_M$,
- nothing otherwise,

with $\pi_O + \pi_M \leq 1$.

All agents share identical per-trial probabilities $(\pi_O, \pi_M)$.

After $T_i$ trials, the agent retains the **best solution encountered**:
- $O$ if at least one $O$ occurred,
- else $M$ if at least one $M$ occurred,
- else $\varnothing$.

Thus, the probability that an agent with horizon $T$ ends learning with solution:
$$
p_O(T) = 1 - (1-\pi_O)^T
$$
$$
p_M(T) = (1-\pi_O)^T \cdot \left[1-(1-\pi_M)^T\right]
$$
$$
p_\varnothing(T) = (1-\pi_O-\pi_M)^T
$$

---

## Heterogeneity in Exploration Horizons

Populations may be homogeneous or heterogeneous in $T$.

- **Homogeneous case**: all agents have $T_i = T_H$.
- **Heterogeneous case**: $T_i$ is drawn from a distribution $D(T \mid \phi)$.

We focus on distributions whose **mean exploration horizon is fixed**:
$$
\mathbb{E}[T_i] = \bar{T}
$$
but whose **tail weight** (controlled by parameter $\phi$) varies.

We consider:
- Main Manuscript: Two-point mixtures:
$$
T_i = 
\begin{cases}
T_L & \text{with probability } 1-p \\
T_H & \text{with probability } p
\end{cases}
$$
with $T_L$ chosen such that $\mathbb{E}[T_i]=\bar{T}$.
- Robustness in SI: Lognormal distributions with fixed mean and increasing variance.

This allows us to isolate the effect of **rare high-$T$ agents** independent of average resources.

---

## Cultural Transmission

After learning, agents enter a social learning phase.

### Teacher Sampling
Each learner samples a **candidate set** of $K$ teachers uniformly at random from the previous generation.  
This captures attention or interface bottlenecks (e.g., $K=5$ as in the experiment).

### Payoff-Biased Choice
From the candidate set, learner $i$ selects teacher $j$ with probability:
$$
P(i \leftarrow j) = 
\frac{\exp(\alpha R_{s_j})}{\sum_{k \in \mathcal{C}_i} \exp(\alpha R_{s_k})}
$$
where:
- $s_j \in \{O,M,\varnothing\}$ is teacher $j$’s solution,
- $\alpha \ge 0$ controls the strength of payoff bias.

### Transmission Fidelity
The learner acquires the teacher’s solution with probability $\tau$:
$$
s_i^{\text{soc}} =
\begin{cases}
s_j & \text{with probability } \tau \\
\varnothing & \text{with probability } 1-\tau
\end{cases}
$$

The socially acquired solution is used for demonstration and payoff in the next generation.

---

## Generational Dynamics
The population evolves over discrete generations $g=1,\dots,G$:

1. **Learning**: each agent explores independently for $T_i$ trials.
2. **Demonstration**: agents demonstrate their best discovered (or socially acquired) solution.
3. **Social learning**: the next generation acquires solutions via payoff-biased copying.

---

## Modeling Machines
Machines are modeled as agents that differ **only** in exploration horizon:
$$
T_M \gg T_H \, .
$$

They share identical per-trial discovery probabilities $(\pi_O,\pi_M)$ and participate in the same social learning process.

Typically, machines are introduced only in early generations (e.g., generation 0) and removed thereafter, allowing us to study **persistence after introduction**.

---

## Key Outcome Measures
We track:
- Probability of persistence:
$$
P(O_{\text{final}})
$$
- Population mean payoff at final generation.

---

## Interpretation
This model deliberately excludes:
- Differences in per-trial intelligence or strategy,
- Agent-specific exploration preferences,
- Psychological biases.

All population-level effects arise from:
1. **Finite learning horizons**,
2. **Heterogeneity in exploration time**,
3. **Selective but noisy cultural transmission**.

It therefore isolates how **the maximum exploration capacity present in a population**, rather than the average capacity, can disproportionately shape cultural trajectories within finite time windows.

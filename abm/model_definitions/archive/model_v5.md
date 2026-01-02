# Model: Exploration Throughput and Cultural Transmission in Finite Time

## Overview
We present a minimal model intended to clarify how **differences in exploration throughput**—rather than per-trial cognitive superiority—can drive population-level cultural change. The model formalizes three ingredients:

1. **Finite-time individual learning**: agents explore for a bounded number of trials.
2. **Rare innovation**: high-payoff solutions are unlikely to be discovered under limited exploration.
3. **Selective cultural transmission**: once discovered, solutions spread via payoff-biased social learning with imperfect transmission.

The central object is the system’s **total exploration capacity**, which aggregates exploration across individuals, populations, and generations.

---

## Environment and Solutions
Agents may end learning with one of three mutually exclusive outcomes:
- $O$: optimal solution (high payoff, hard to discover)
- $M$: moderate solution (lower payoff, easier to discover)
- $\varnothing$: no solution

Payoffs satisfy:
$$
R_O > R_M > R_\varnothing \, .
$$

---

## Individual Learning (Finite Horizon)
Each agent $i$ has an exploration horizon $T_i \in \mathbb{N}$, the number of learning trials they execute.

On each learning trial, the agent independently discovers:
- $O$ with probability $\pi_O$,
- $M$ with probability $\pi_M$,
- nothing otherwise,

with $\pi_O + \pi_M \le 1$.

All agents share identical per-trial discovery probabilities $(\pi_O,\pi_M)$. Differences between humans and machines (if modeled) are represented **only** through $T$.

After $T_i$ trials, the agent retains the best solution encountered. Thus:
$$
p_O(T_i) = 1 - (1-\pi_O)^{T_i}
$$
$$
p_M(T_i) = (1-\pi_O)^{T_i}\left[1-(1-\pi_M)^{T_i}\right]
$$
$$
p_\varnothing(T_i) = (1-\pi_O-\pi_M)^{T_i} \, .
$$

---
### Total Exploration Capacity: An Analytical Upper Bound on Innovation Supply

To isolate the role of finite-time exploration in generating innovation, we define the system’s **total exploration capacity** as the total number of independent learning trials executed over a bounded time horizon.

For a population with $N$ individuals and possibly heterogeneous exploration horizons $\{T_i\}_{i=1}^N$, the total number of trials executed within a single generation is:
$$
T_{\text{gen}} \;=\; \sum_{i=1}^{N} T_i \, .
$$

Across $G$ generations the total exploration capacity is:
$$
T_{\text{total}} \;=\; \sum_{g=1}^{G}\sum_{i=1}^{N_g} T_{i,g} \, .
$$

In the simplest homogeneous case, with constant individual horizons $T_{\text{ind}}$ and population size $N_{\text{pop}}$ across generations, this reduces to:
$$
T_{\text{total}} \;=\; T \,N\,G \, .
$$

---

**Key implication (innovation supply).**  
If each trial independently yields the optimal solution $O$ with probability $\pi_O$, then the probability that at least one instance of $O$ is discovered until generation G is:
$$
P(O_{\text{intro by G}}) = 1 - (1-\pi_O)^{T_{\text{total}}} \, .
$$

---

**Interpretation.**  
Innovation supply therefore scales primarily with the total number of trials executed by the system. This capacity can be increased by extending individual learning horizons ($T$), increasing population size ($N$), increasing the number of generations ($G$), or—most critically in the case of machines—by introducing agents capable of executing substantially more trials within the same time window. 

---

## Cultural Transmission
After learning, each generation produces demonstrations and the next generation learns socially.

### Teacher sampling (attention/interface bottleneck)
Each learner samples a candidate set $\mathcal{C}_i$ of size $K$ uniformly at random from the previous generation (default $K=5$ to mirror a constrained demonstration slate).

### Payoff-biased choice
Learner $i$ selects teacher $j\in\mathcal{C}_i$ with probability:
$$
P(i \leftarrow j)=
\frac{\exp(\alpha R_{s_j})}{\sum_{k\in\mathcal{C}_i}\exp(\alpha R_{s_k})} \, ,
$$
where $s_j\in\{O,M,\varnothing\}$ and $\alpha\ge 0$ controls selectivity.

### Transmission fidelity
The chosen teacher’s solution is transmitted with probability $\tau$:
$$
s_i^{\text{soc}} =
\begin{cases}
s_j & \text{with probability } \tau \\
\varnothing & \text{with probability } 1-\tau
\end{cases}
$$

Learners then demonstrate their acquired solution in the next generation.

---

## Humans and Machines
Humans and machines are represented as agents with identical per-trial discovery rates $(\pi_O,\pi_M)$ but different horizons:
- Humans: $T_H$
- Machines: $T_M \gg T_H$

---

## Primary Outcomes
We track:
- $P(O_{\text{intro}})$: probability that $O$ appears (at least one discovery event)
- $P(O_{\text{final}})$: probability that $O$ is present in the final generation
- Final mean payoff and distribution of population states

We also report the decomposition:
$$
P(O_{\text{final}})=P(O_{\text{intro}})\cdot P(O_{\text{final}}\mid O_{\text{intro}}),
$$
where throughput largely governs the first term, and social learning/transmission govern the second.

---

# Simulation Experiments

## Common Parameters (Defaults)
- Payoffs: $R_O > R_M > R_\varnothing$ (e.g., $R_O=10, R_M=6, R_\varnothing=0$)
- Per-trial discovery: $(\pi_O,\pi_M)$ fixed across agent types
- Teacher slate: $K=5$ (lab-mimic); robustness in SI with alternative $K$
- Selectivity: $\alpha$ (default 1 unless varied)
- Fidelity: $\tau$ (default 0.6 unless varied)
- Replicates per condition: $\ge 200$ (more for small $N$)
- Two timescales:
  - Lab-mimic: $N=8$, $G=5$
  - Robustness: larger $N$ and/or larger $G$

---

## Experiment 1 — Few High-Throughput Agents vs Broad Increases (Holding Total Trials Comparable)

### Goal
Show that **one need not increase everyone’s $T$** to increase innovation supply: a small number of high-throughput agents (machines) can contribute comparable exploration mass.

### Manipulations (keep explicit)
- Human horizon: $T_H \in \{5,10,20\}$
- Machine horizon: $T_M \in \{50,100,200\}$
- Machine count: $N_M \in \{0,1,2,4\}$
- Total population: $N = N_H + N_M$ (default $N=8$), with $N_H = N - N_M$

### Design
- Machines present in generation 0 only (optional variant: present for all generations)
- Social learning: $K=5$, $\alpha=1$, $\tau=0.6$
- Generations: $G=5$

### Key derived quantity
For each condition compute:
$$
T_{\text{gen}} = N_H T_H + N_M T_M
\qquad\text{and}\qquad
T_{\text{total}} = \sum_{g=1}^{G} T_{\text{gen}}^{(g)} \, .
$$

### Outcomes
- $P(O_{\text{intro}})$ and $P(O_{\text{final}})$
- Plot outcomes against $T_{\text{gen}}$ (or $T_{\text{total}}$), with markers indicating $(N_M,T_M)$.

### Interpretation
If curves collapse primarily onto $T_{\text{gen}}$ (or $T_{\text{total}}$), this demonstrates that **total exploration capacity** is the key driver, while also showing that a few machines can supply a large share of that capacity.

---

## Experiment 2 — Generational Scaling of Exploration ($T_{\text{total}} = T_{\text{ind}}N_{\text{pop}}G$)

### Goal
Quantify how increasing generations $G$ (time for cultural accumulation) interacts with throughput and diffusion. This addresses the fact that cultural evolution experiments compress time, and makes explicit how cumulative exploration scales.

### Manipulations
- Number of generations: $G \in \{5,10,100, 1000\}$
- Compare at least two throughput regimes:
  - Human-only: $N_M=0$, $T_H$ fixed
  - Mixed: small $N_M$ with large $T_M$

### Design
- Keep $N$ fixed (e.g., 8 for lab-mimic; replicate at larger $N$ in SI)
- Keep $(\pi_O,\pi_M,\alpha,\tau,K)$ fixed

### Outcomes
- Over longer time horizion, (more generations, humans eventually discover the optimal strategy. ) as the total innovation capacity scales linear with time. 
- Machines enable to shorten this time period. 

### Interpretation
This experiment makes explicit that increasing $G$ increases $T_{\text{total}}$ and thus innovation supply.

---

## Experiment 3 — Diffusion Gate: Selectivity $\alpha$ and Fidelity $\tau$

### Goal
Isolate how diffusion parameters govern persistence conditional on introduction (addressing “isn’t this just payoff-biased copying?”).

### Design
- Ensure $O$ is introduced in generation 0 (e.g., seed one $O$ demonstrator)
- Use homogeneous horizons for simplicity

### Manipulations
- $\alpha \in \{0,0.5,1,2,4\}$
- $\tau \in \{0.2,0.4,0.6,0.8,1.0\}$

### Outcomes
- $P(O_{\text{final}}\mid O_{\text{intro}})$
- Frequency of loss vs persistence across runs

### Interpretation
Separates throughput-driven introduction from transmission-driven persistence.

---

## Experiment 4 — Bounded Uplift Across Task Regimes (Difficulty $\pi_O$ × Learnability $\tau$)

### Goal
Explain why the benefit of high-throughput agents is strongest in intermediate regimes.

### Manipulations
- $\pi_O$ scanned over a grid (difficulty)
- $\tau$ scanned over a grid (learnability)

### Design
- Compare human-only vs mixed-throughput populations
- Keep $T_H$ fixed; implement machines with larger $T_M$ and small $N_M$
- Use $K=5$ in main; alternative $K$ in SI

### Outcomes
- Uplift in final mean payoff:
$$
\Delta \Pi = \mathbb{E}[\Pi_{\text{final}}^{\text{mixed}}]-\mathbb{E}[\Pi_{\text{final}}^{\text{human-only}}]
$$
- Differences in $P(O_{\text{final}})$

### Interpretation
Uplift peaks when $O$ is rare enough that humans seldom introduce it within finite horizons, yet discoverable with added throughput, and when $\tau$ is high enough for diffusion.

---

## Supplementary Robustness (SI)
- Teacher access: $K=5$ vs $K=\lceil 0.1N\rceil$ vs $K=N$
- Population size: $N \in \{8,16,32,100\}$
- Machines present only in generation 0 vs present for $D$ generations

---

## Summary
These experiments makes $T_H$ and $T_M$ explicit while demonstrating that the key explanatory variable is **total exploration capacity**, scaling as:
$$
T_{\text{total}} = \bar{T}\,N\,G \;=\; \sum_{g=1}^{G}\sum_{i=1}^{N_g} T_{i,g} \, .
$$

A small number of high-throughput agents can contribute disproportionate exploration mass, thereby increasing the likelihood that rare high-payoff solutions appear within finite time windows, after which selective and imperfect cultural transmission determines persistence.

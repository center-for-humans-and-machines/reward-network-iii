# Simulation Experiments

This section summarizes the simulation experiments used to characterize the cultural dynamics of exploration propensity and solution transmission under selective social learning. All experiments use the learning-phase formulation in which discovery probabilities are analytically determined by the number of learning trials $T$ and per-trial success probabilities $\pi_M,\pi_O$.

---

## Shared model components (all experiments)

### Fixed structure
- Solution classes: $M$ (easy), $O$ (hard), $\varnothing$ (none)
- Learning horizon: $T$ learning trials per agent
- Per-trial discovery probabilities:
  $$
  \pi_M > \pi_O
  $$
- Discovery probabilities over $T$ trials:
  $$
  p_O(\theta,T)=1-(1-\theta\pi_O)^T,\quad
  p_M(\theta,T)=1-\big(1-(1-\theta)\pi_M\big)^T
  $$
- Demonstration payoffs:
  $$
  R_O > R_M > R_\varnothing
  $$
- Learning phase: no explicit costs; risk arises via the probability of ending with $\varnothing$
- Demonstration phase: agent exploits the best available solution $s^\star$
- Teacher selection (selective social learning):
  $$
  \Pr(j)\propto \exp\!\big(\alpha\,\Pi_{\text{demo}}(s_j^\star)\big)
  $$
- Trait transmission: $\theta$ copied in logit space with Gaussian noise
- Solution transmission: demonstrated solution transmitted with probability $\tau$
- Replicates per condition: 30

### Suggested default parameters
- Payoffs: $R_O=10,\ R_M=6,\ R_\varnothing=0$
- Selection strength: $\alpha=1.0$ (unless otherwise stated)
- Trait noise: $\sigma=0.3$
- Solution transmission: $\tau=0.6$

---

## Experiment 1 — Humans-only: convergence of exploration propensity

### Goal
Establish that, under short learning horizons, selective social learning drives the exploration propensity $\theta$ toward low values, suppressing risky exploration.

### Fixed parameters
- Single agent type (humans only)
- Learning horizon: $T=T_H=5$
- Per-trial discovery probabilities:
  $$
  \pi_M=0.25,\quad \pi_O=0.05
  $$
- Selective social learning: $\alpha=1.0$
- Initial $\theta$: $\theta_0\sim\mathrm{Uniform}(0.1,0.9)$
- Population size: $N=100$
- Generations: $G=300$

### Manipulated parameters
- None (baseline confirmation)
- Optional robustness: $\sigma\in\{0.1,0.3,0.6\}$

### Analysis
Plot the mean exploration propensity $\mathbb{E}[\theta_t]$ over generations with confidence intervals across replicates. Verify convergence to a low stationary value. Additionally, plot the final-generation distribution of $\theta$ and the fraction of agents demonstrating $M$, $O$, and $\varnothing$.

---

## Experiment 2 — Learning horizon scan: critical transition in $\theta$

### Goal
Identify a critical learning horizon at which selection on $\theta$ switches from favoring risk avoidance to favoring exploration.

### Fixed parameters
- Single agent type
- Per-trial discovery probabilities:
  $$
  \pi_M=0.25,\quad \pi_O=0.05
  $$
- Selective social learning: $\alpha=1.0$
- Initial $\theta_0=0.5$
- Population size: $N=100$
- Generations: $G=300$

### Manipulated parameters
- Learning horizon:
  $$
  T\in\{1,2,\dots,25\}
  $$

### Analysis
For each $T$, compute the stationary mean $\bar{\theta}$ (averaged over the final 50 generations). Plot $\bar{\theta}$ as a function of $T$ and identify a threshold $T^\star$ at which $\bar{\theta}$ increases sharply. Optionally, plot the stationary probability of demonstrating $O$ as a function of $T$.

---

## Experiment 3 — Humans and machines: transmission of optimal solutions

### Goal
Test when the optimal solution $O$ can spread among humans via social transmission from machines, and whether selective social learning is required for persistence.

### Fixed parameters
- Two agent types:
  - Humans: fixed $\theta_H=0.05$
  - Machines: fixed $\theta_M=0.5$
- Learning horizon:
  $$
  T_H=T_M=10
  $$
- Per-trial discovery probabilities:
    $$
    \pi_M=0.25,\quad \pi_O\in\{0.01,\dots,0.15\}
    $$
- Trait transmission: disabled (fixed $\theta$)
- Population size: $N=8$ (robustness $N=100$)
- Generations: $G=10$ (robustness $G=300$)
- Population: 
    - humans only
    - humans and machines


### Manipulated parameters
1. Discovery difficulty:
   $$
   \pi_O \text{ (grid)}
   $$
2. Solution transmission probability:
   $$
   \tau\in\{0.0,0.1,\dots,1.0\}
   $$
3. Social learning regime:
   - Selective: $\alpha=1.0$
   - Non-selective: $\alpha=0$
4. Population composition
   - humans only
   - humans and machines

### Procedure

Main: Introduce machines at 3 Machines for the first generation of the simulation. Then humans only.

Robustness: Burn-in for 50 generations. Then 50 generations with machines. Then 200 generations without machines. 


### Analysis
Measure the fraction of humans demonstrating $O$ over time. Summarize outcomes by the mean fraction of $O$ demonstrations in the final 50 generations. Plot heatmaps over $(\pi_O^H,\tau)$ separately for selective and non-selective social learning, showing that $O$ spreads and persists only when transmission is sufficiently strong and social learning is selective.

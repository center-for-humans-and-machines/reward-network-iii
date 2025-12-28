# Supplementary Information: Formal Model

## S1. Overview

This section specifies a minimal generative model of cultural exploration and transmission with **structured tasks**.  
The model is designed to isolate a single mechanism: how relaxing *discovery constraints* (e.g., via machines with larger trial budgets) can shift human cultural equilibria, even when human learning and transmission capacities remain unchanged.

The model extends a standard exploration–diffusion framework by introducing **task–strategy structure** while deliberately avoiding compositional strategy building, task inference, or rich cognition. All assumptions are explicit and intentionally conservative.

---

## S2. Task Space

### S2.1 Task representation

A task is represented as a binary string
$$
z \in \{0,1\}^K,
$$
where $K$ is fixed and denotes the ambient dimensionality of the task space.  
Each bit corresponds to a latent task feature or constraint.

Leading zeros are permitted in the task representation; tasks always live in the full $K$-dimensional space.

---

### S2.2 Task distribution

Tasks are drawn independently from a fixed distribution $\mathcal{D}$ over $\{0,1\}^K$.

A simple and flexible specification assumes independent Bernoulli features:
$$
z_k \sim \mathrm{Bernoulli}(p_k), \quad k = 1,\dots,K,
$$
where the vector $p=(p_1,\dots,p_K)$ controls environmental bias.  
Biasing $p$ allows the environment (or experimenter) to systematically over-represent certain task features.

The task distribution is exogenous and does not evolve.

---

## S3. Strategy Space

### S3.1 Strategy representation

A strategy is a binary string of variable length
$$
s \in \{0,1\}^{\ell}, \quad 0 \le \ell \le K,
$$
where $\ell(s)$ denotes the **effective length** (or complexity) of the strategy.

Leading zeros are ignored: the strategy is defined only by its canonical bitstring up to its most significant 1.  
Intuitively, a strategy specifies responses only for the first $\ell(s)$ task dimensions and is agnostic about the remaining $K-\ell(s)$ dimensions.

> *Computational note:* For implementation, strategies (and tasks) may be encoded as integers, with length defined as the position of the most significant 1. All theoretical definitions are stated in terms of bitstrings.

---

### S3.2 Length-limited overlap

The overlap between a strategy $s$ and a task $z$ is computed **only over the strategy’s effective length**:
$$
O(s,z) = \sum_{k=1}^{\ell(s)} s_k\, z_k.
$$

Only task features explicitly represented in the strategy can contribute to performance.  
Leading zeros in either representation play no role.

---

## S4. Payoffs

The payoff obtained by applying strategy $s$ to task $z$ is
$$
\pi(s,z) = r \cdot O(s,z) - c \cdot \ell(s),
$$
where:
- $r > 0$ is the reward per overlapping feature,
- $c > 0$ is the cost per unit strategy length.

This payoff structure induces a **generality–specificity tradeoff**:
- short strategies are cheap and robust but have limited upside,
- long strategies can yield high payoffs when well matched but incur higher costs and fail frequently when mismatched.

---

### S4.1 Safe strategy

There is a distinguished safe strategy $s_{\mathrm{safe}}$ with
$$
\ell(s_{\mathrm{safe}})=0, \qquad \pi(s_{\mathrm{safe}},z)=r_{\mathrm{safe}}>0
$$
for all tasks $z$.

The safe strategy is always available and provides a guaranteed baseline payoff.

---

## S5. Individual Learning

Each agent $i$ is characterized by:
- a trial budget $T_i$,
- a repertoire $K_i$ of previously discovered strategies.

### S5.1 Action selection

On each trial:
- with probability $1-\varepsilon$, the agent uses the safe strategy;
- with probability $\varepsilon$:
  - with probability $\phi$, the agent exploits by selecting a strategy uniformly from $K_i$ (if nonempty);
  - with probability $1-\phi$, the agent explores by sampling a new strategy (see below).

A task $z \sim \mathcal{D}$ is drawn independently on each trial.

---

### S5.2 Exploration: two-stage sampling

Exploration proceeds in two stages to make length control explicit.

**Stage 1 (length sampling):**
$$
\ell \sim \mathrm{Geom}(\rho),
$$
supported on $\{0,1,2,\dots\}$ with
$$
\Pr(\ell=k) = (1-\rho)^k \rho,
$$
and truncated to $\ell \le K$.

**Stage 2 (content sampling):**
Given $\ell$,
- if $\ell=0$, the sampled strategy is $s=0$;
- if $\ell>0$, sample $s$ uniformly from $\{0,1\}^{\ell}$ with the most significant bit equal to 1.

This procedure yields predominantly short exploration attempts with a controlled tail of longer strategies.  
An **effective exploration depth** emerges endogenously from $\rho$.

---

### S5.3 Retention

After observing payoff $\pi(s,z)$, a strategy $s$ is added to the agent’s repertoire iff
$$
\pi(s,z) > 0.
$$

Strategies that are costly and rarely overlap with tasks are therefore unlikely to be retained under limited trial budgets.

---

## S6. Social Learning

### S6.1 Teacher selection

Learners select a teacher $j$ with probability
$$
\Pr(j) \propto \exp\!\bigl(\beta \cdot \mathrm{Perf}_j\bigr),
$$
where $\mathrm{Perf}_j$ is the teacher’s average realized payoff.

---

### S6.2 Transmission and learnability

Each strategy $s \in K_j$ is adopted independently with probability
$$
q(s) = \sigma\!\left(\alpha - \gamma \cdot \mathrm{RLE}(s)\right),
$$
where:
- $\sigma(\cdot)$ is the logistic function,
- $\mathrm{RLE}(s)$ is the run-length encoding complexity of the **canonical bitstring of $s$** (computed without leading zeros).

Learnability thus depends on compressibility, independent of task structure and discoverability.

---

## S7. Machines

Machines differ from humans **only** in trial budget:
$$
T_M \gg T_H.
$$

Machines:
- follow the same exploration, payoff, and retention rules,
- may be present transiently,
- seed strategies through the same social-learning channel.

Machines have no privileged access to task features or payoffs.

---

## S8. Evolution of Exploration Profile (Calibration)

To avoid treating human exploration behavior as a free parameter, we include an auxiliary evolutionary procedure used solely for calibration. Exploration is governed by a geometric length-sampling distribution with parameter $\rho \in (0,1)$, where smaller $\rho$ implies heavier-tailed exploration.

At generation 0, agents are initialized with exploration parameters $\rho_i$ drawn from a **starting distribution** $P_0(\rho)$, chosen to be broad and uninformative (e.g., uniform over a bounded interval) unless otherwise stated. Initial repertoires are empty, and only the safe strategy is available.

In a human-only population, agents engage in $T_H$ trials using the learning and exploration rules described above, and their average realized payoff $\mathrm{Perf}_i$ is recorded. New agents select teachers via payoff-biased cultural transmission,
$$
\Pr(j) \propto \exp(\beta \cdot \mathrm{Perf}_j),
$$
inherit the teacher’s $\rho$, and mutate it slightly with small probability. This process is iterated until the population distribution of $\rho$ stabilizes.

Each value of $\rho$ induces an effective exploration depth (e.g., via the expected or quantile explored length under the geometric distribution). Once convergence is reached, we fix $\rho$ at the population mean value $\rho^\star$. All subsequent simulations—including those involving machines—use this fixed exploration profile, ensuring that observed differences across conditions reflect changes in discovery capacity rather than endogenous shifts in human exploration preferences.


---

## S9. Interpretation and Scope

The model captures:
- discovery bottlenecks arising from bounded trial budgets,
- specialization induced by task–strategy mismatch costs,
- discontinuous cultural shifts triggered by rare discoveries,
- a separation between discoverability and learnability.

The model does **not** represent:
- compositional or stepwise refinement of strategies,
- inference of task structure,
- or bandwidth-limited pedagogical transmission.

Accordingly, “cumulative culture” is understood here as **accumulation and diffusion of a repertoire**, not compositional ratcheting.
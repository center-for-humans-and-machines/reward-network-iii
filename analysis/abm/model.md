## Model (refined formulation with explicit payoffs)

We model cultural exploration as search in a large behavioral space, from which a small set of reusable strategies can be discovered, transmitted, and applied across tasks.

---

### Behavioral space

Let
$$
\mathcal{X} = \bigcup_{\ell=0}^{\ell_{\max}} \{0,1\}^{\ell}
$$
denote the space of all binary prefixes up to maximum length $\ell_{\max}$.  
Elements of $\mathcal{X}$ represent concrete action sequences that agents may attempt through exploration. Most elements of $\mathcal{X}$ are unstructured and do not correspond to reusable strategies.

---

### Latent strategy set

There exists a distinguished subset
$$
\mathcal{S} \subset \mathcal{X},
$$
whose elements are structured prefixes that constitute culturally meaningful strategies. Only elements of $\mathcal{S}$ are retained in memory, socially transmitted, and reused across tasks.

Strategies are ordered by complexity. For $s \in \mathcal{S}$, let $\ell(s)$ denote its length. The empty prefix
$$
s_0 = ""
$$
is included in $\mathcal{S}$ and represents a safe default strategy.

---

### Tasks and applicability

Agents encounter tasks $p \in \{1,\dots,P\}$.  
Each task specifies which strategies are effective:
$$
\mathcal{W}_p \subseteq \mathcal{S}.
$$

Strategies in $\mathcal{W}_p$ unlock a high-payoff region of task $p$; strategies outside $\mathcal{W}_p$ do not, even if they are globally meaningful. Different tasks thus correspond to different subsets of applicable strategies drawn from a shared strategic repertoire.

---

### Payoff structure

Payoffs are defined over the full behavioral space $\mathcal{X}$:
$$
R_p : \mathcal{X} \rightarrow \mathbb{R}.
$$

Let $\ell(x)$ denote the length of prefix $x$. Payoffs take the form
$$
R_p(x) =
\begin{cases}
r_{\mathrm{safe}}, & x = s_0, \\[6pt]
- c \, \ell(x), & x \notin \mathcal{S}, \\[6pt]
- c \, \ell(s), & x = s \in \mathcal{S} \setminus \mathcal{W}_p, \\[6pt]
- c \, \ell(s) + r \, \lambda^{\ell(s)}, & x = s \in \mathcal{W}_p,
\end{cases}
$$
where:
- $c > 0$ is a per-step exploration cost,
- $r > 0$ sets the reward scale,
- $\lambda > 1$ controls the superlinear growth of payoffs for applicable strategies,
- $r_{\mathrm{safe}} > 0$ is the payoff of the default strategy.

This structure ensures that deeper prefixes are increasingly costly to attempt, while applicable deep strategies yield disproportionately large returns. Non-strategic exploration and misapplied strategies incur only costs.

---

### Learnability

Each strategy $s \in \mathcal{S}$ has an intrinsic learnability determined by its run-length encoding (RLE) complexity:
$$
\mathrm{RLE}(s) =
\begin{cases}
0, & s = "", \\[4pt]
1 + \sum_{m=2}^{\ell(s)} \mathbf{1}[s_m \neq s_{m-1}], & \text{otherwise}.
\end{cases}
$$

The probability that a socially demonstrated strategy $s$ is successfully learned is
$$
q(s) = \sigma\!\left(\alpha - \gamma\, \mathrm{RLE}(s)\right),
$$
where $\sigma(\cdot)$ is the logistic function. Learnability is independent of discovery difficulty and task applicability.

---

### Agents

Each agent $i$ is characterized by:
- a fixed lifetime trial budget $T_i$;
- an exploration depth $d_i$, defining the maximum prefix length sampled during exploration;
- a personal repertoire $\mathcal{K}_i \subseteq \mathcal{S}$, initially containing at least $s_0$.

For each task $p$, the agent tracks the best-performing strategy discovered so far, indexed by $b_{i,p}$.

---

### Individual learning dynamics

For each trial $t = 1,\dots,T_i$:

1. **Task sampling.**  
   The agent samples a task $p$ uniformly from $\{1,\dots,P\}$.

2. **Action selection.**  
   With probability $1-\varepsilon$, the agent exploits the best-known strategy $s_{b_{i,p}}$.  
   With probability $\varepsilon$, the agent explores by sampling a random prefix
   $$
   x \sim \mathrm{Unif}\{\,x \in \mathcal{X} : \ell(x) \le d_i\,\}.
   $$

3. **Strategy recognition.**  
   If the sampled prefix $x$ satisfies $x \in \mathcal{S}$ and yields a payoff exceeding the current best for task $p$, it is added to the repertoire:
   $$
   \mathcal{K}_i \leftarrow \mathcal{K}_i \cup \{x\}, \qquad b_{i,p} \leftarrow x.
   $$

Through this process, agents test known strategies, occasionally discover new strategies by chance, and reallocate effort across tasks under a single finite budget.

---

### Social transmission

After completing their trials, agents act as teachers for the next generation.

Learners select a teacher $i$ with probability
$$
\Pr(i) \propto \exp\!\bigl(\beta\, \mathrm{Perf}_i\bigr),
$$
where $\mathrm{Perf}_i$ is the total payoff accumulated by agent $i$.

A teacher transmits a subset of its repertoire (e.g., the highest-performing strategies).  
Each transmitted strategy $s$ is adopted independently with probability $q(s)$ and added to the learner’s repertoire.

---

### Cultural evolution

For human agents, exploration depth $d$ evolves across generations via payoff-biased copying:
$$
\Pr(\text{parent}=i) \propto \exp\!\bigl(\beta_{\mathrm{evo}}\, \mathrm{Perf}_i\bigr),
$$
with offspring exploration depth
$$
d' = \mathrm{clip}(d + \Delta), \qquad \Delta \in \{-1,0,+1\}.
$$

Machines differ only in having larger trial budgets and may be introduced transiently to seed exploration.

---

## Experimental conditions and parameter regimes

The same model generates all reported phenomena by fixing parameters as follows.

### 1. Human-only societies and optimal exploration depth
- Set $T_i = T_H$ for all agents.
- Fix $\varepsilon$, $c$, $r$, and $\lambda$.
- Vary $T_H$ across simulations.
- Tasks differ in the minimum strategy length $\ell(s)$ required for inclusion in $\mathcal{W}_p$.
  
Outcome: populations converge to an exploration depth $d_{T_H}$ that maximizes individual payoff; strategies deeper than this remain undiscovered, limiting cumulative cultural exploration.

---

### 2. Fixed barrier ($B=3$) and binary machine-induced shift
- Define $\mathcal{W}_p$ such that only strategies with $\ell(s) \ge B$ are applicable.
- Choose $c$ and $r_{\mathrm{safe}}$ so that safe dominates for humans under $T_H$.
- Introduce machines with $T_M \gg T_H$ in the first generation only.

Outcome: without machines, deep strategies are not discovered and culture remains at the safe default; with machines, discovery occurs and spreads rapidly through social transmission, producing an abrupt cultural shift.

---

### 3. Distributed task difficulty
- Draw tasks with heterogeneous applicability thresholds (different $\mathcal{W}_p$).
- Introduce machines only in early generations.

Outcome: machines disproportionately seed deep strategies, shifting the long-run human repertoire toward higher-complexity strategies even after machines are removed.

---

### 4. Discovery difficulty independent of learnability
- Construct $\mathcal{S}$ such that some deep strategies have low RLE complexity.
- Keep discovery difficulty high via depth-dependent costs and sparse $\mathcal{W}_p$.
- Control learnability exclusively via $\gamma$.

Outcome: machines introduce strategies that are rare and costly to discover but easy to transmit, leading to persistent cultural change without corresponding increases in individual exploration depth.

---

### Summary

Across all conditions, differences in computational capacity affect which strategies enter culture, while social transmission and learnability determine which discoveries persist. Population size alone increases parallel search but does not alter the individually optimal depth of exploration.

## Model

We model cultural exploration as search in a large behavioral space, from which a small set of reusable strategies can be discovered, transmitted, and applied across tasks.

---

### Behavioral space

Let
$$
\mathcal{X} = \{""\} \cup \bigcup_{\ell=1}^{\ell_{\max}} \{1\} \times \{0,1\}^{\ell-1}
$$
denote the space of binary prefixes up to maximum length $\ell_{\max}$, where non-empty prefixes must start with 1 and the empty prefix $""$ is included as a special case.  
Elements of $\mathcal{X}$ represent concrete action sequences that agents may attempt through exploration. The space maps bijectively to integers $0, \ldots, 2^{\ell_{\max}} - 1$ (with $0 \mapsto ""$ and $n > 0$ mapping to the binary representation of $n$). Most elements of $\mathcal{X}$ do not correspond to usable strategies.

---

### Latent strategy set

There exists a distinguished subset
$$
\mathcal{S} \subset \mathcal{X},
$$
whose elements are prefixes that constitute culturally meaningful strategies. Only elements of $\mathcal{S}$ can yield positive payoffs when applicable to a task (see payoff structure below). The repertoire and applicability are represented over the full behavioral space $\mathcal{X}$ for computational convenience, but only elements of $\mathcal{S}$ have meaningful strategic content.

Strategies are ordered by complexity. For $s \in \mathcal{S}$, let $\ell(s)$ denote its length. The empty prefix
$$
s_0 = ""
$$
is included in $\mathcal{S}$ and represents a safe default strategy.

---

### Tasks and applicability

Agents encounter tasks drawn uniformly at random.  
Applicability is probabilistic: when a strategy $x \in \mathcal{S}$ is applied, it succeeds with probability $p_{\mathrm{applicable}} \in [0,1]$, independently of the specific strategy or task. Only elements of $\mathcal{S}$ can be applicable; non-strategic elements of $\mathcal{X}$ never succeed.

Strategies that succeed unlock high payoffs; strategies that fail incur only costs. The safe strategy $s_0$ always succeeds.

---

### Payoff structure

Payoffs are defined over the full behavioral space $\mathcal{X}$:
$$
R : \mathcal{X} \rightarrow \mathbb{R}.
$$

Let $\ell(x)$ denote the length of prefix $x$. When a strategy is applied, it succeeds with probability $p_{\mathrm{applicable}}$ (or deterministically for the safe strategy). However, only strategies in $\mathcal{S}$ can yield bonuses; strategies outside $\mathcal{S}$ have zero bonus regardless of success. Payoffs take the form
$$
R(x) =
\begin{cases}
r_{\mathrm{safe}}, & x = s_0, \\[6pt]
- c \, \ell(x), & x \notin \mathcal{S}, \\[6pt]
- c \, \ell(x), & x \in \mathcal{S}, \text{ with probability } 1 - p_{\mathrm{applicable}}, \\[6pt]
- c \, \ell(x) + r \, \lambda^{\ell(x)}, & x \in \mathcal{S}, \text{ with probability } p_{\mathrm{applicable}},
\end{cases}
$$
where:
- $c > 0$ is a per-step exploration cost,
- $r > 0$ sets the reward scale,
- $\lambda > 1$ controls the superlinear growth of payoffs for applicable strategies,
- $r_{\mathrm{safe}} > 0$ is the payoff of the default strategy,
- $p_{\mathrm{applicable}} \in [0,1]$ is the probability that a strategy in $\mathcal{S}$ succeeds when applied.

This structure ensures that deeper prefixes are increasingly costly to attempt, while applicable deep strategies yield disproportionately large returns. Non-strategic exploration (elements of $\mathcal{X} \setminus \mathcal{S}$) and unsuccessful applications of strategies in $\mathcal{S}$ incur only costs.

---

### Learnability

Each strategy $s \in \mathcal{S}$ has an intrinsic learnability determined by its run-length encoding (RLE) complexity. RLE is computed on the bitstring representation of the strategy, ignoring any leading zeros:
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
- a personal repertoire $K_i : \mathcal{X} \rightarrow \{0, 1\}$, represented as a boolean function over the full behavioral space, initially all zeros. $K_i(x) = 1$ indicates that the agent has discovered and retained strategy $x$. Only strategies in $\mathcal{S}$ that yield positive rewards are retained (see update rule below). [TODO: Clarify in the future, and potentially remove safe strategy] The safe strategy $s_0$ is always available for selection during action selection (see below) but is only added to the repertoire when it yields a positive reward.

---

### Individual learning dynamics (Option B: unified sampling)

For each trial $t = 1,\dots,T_i$, agent $i$ proceeds as follows:

1. **Action selection.**  
   The agent samples a prefix $x \in \mathcal{X}$ from a single mixture distribution,
   $$
   x \sim (1-\varepsilon)\,\delta_{\emptyset}
   \;+\;
   \varepsilon\Big[
      \phi\,\mathrm{Unif}(\{x : K_i(x) = 1\})
      + (1-\phi)\,\mathrm{Unif}(\mathcal{X}_{\le d_i})
   \Big],
   $$
   where $\emptyset$ denotes the empty (safe) prefix, the agent samples uniformly from strategies where $K_i(x) = 1$ for the exploit option, and $\mathcal{X}_{\le d_i}$ is the set of all prefixes mapped to indices $[0, 2^{d_i})$.  
   With probability $1-\varepsilon$, the agent selects the safe prefix. With probability $\varepsilon\phi$, it reuses a previously discovered strategy by sampling uniformly from strategies where $K_i(x) = 1$. With probability $\varepsilon(1-\phi)$, it engages in genuine exploration by sampling uniformly from indices in $[0, 2^{d_i})$, which (due to the bijection between indices and binary prefixes) assigns exponentially greater probability mass to longer prefixes, since there are $2^{\ell-1}$ prefixes of length $\ell$ for $\ell > 0$.

2. **Evaluation and update.**  
   The sampled prefix $x$ is applied and yields a payoff determined by the task environment. If the payoff is positive, the prefix is added to the agent's repertoire,
   $$
   K_i(x) \leftarrow 1.
   $$

Agents do not maintain task–strategy associations. Instead, strategies are treated as portable cultural objects whose reuse and further dissemination emerge endogenously from the sampling process under a fixed lifetime budget $T_i$.


---

### Social transmission

After completing their trials, agents act as teachers for the next generation.

Learners select a teacher $i$ with probability
$$
\Pr(i) \propto \exp\!\bigl(\beta\, \mathrm{Perf}_i\bigr),
$$
where $\mathrm{Perf}_i$ is the total payoff accumulated by agent $i$.

A teacher transmits its entire repertoire (all strategies $x$ where $K_i(x) = 1$) to the learner.  
Each transmitted strategy $x$ is adopted independently with probability $q(x)$ and added to the learner's repertoire (i.e., $K_j(x) \leftarrow 1$).

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
- Fix $\varepsilon$, $c$, $r$, $\lambda$, and $p_{\mathrm{applicable}}$.
- Vary $T_H$ across simulations.
  
Outcome: populations converge to an exploration depth $d_{T_H}$ that maximizes individual payoff; strategies deeper than this remain undiscovered, limiting cumulative cultural exploration.

---

### 2. Fixed barrier ($B=3$) and binary machine-induced shift
- Construct $\mathcal{S}$ such that only strategies with $\ell(s) \ge B$ are in the latent set (or set parameters so that only strategies with $\ell(s) \ge B$ have positive expected payoffs).
- Choose $c$, $r_{\mathrm{safe}}$, and $p_{\mathrm{applicable}}$ so that safe dominates for humans under $T_H$, while deeper strategies in $\mathcal{S}$ have positive expected value when they succeed.
- Introduce machines with $T_M \gg T_H$ in the first generation only.

Outcome: without machines, deep strategies are not discovered and culture remains at the safe default; with machines, discovery occurs and spreads rapidly through social transmission, producing an abrupt cultural shift.

---

### 3. Distributed task difficulty
- Construct $\mathcal{S}$ to include strategies across a range of complexities, with $p_{\mathrm{applicable}}$ set such that expected payoffs vary heterogeneously across strategy lengths (e.g., some intermediate lengths may be more favorable than very deep ones).
- Introduce machines only in early generations.

Outcome: machines disproportionately seed deep strategies, shifting the long-run human repertoire toward higher-complexity strategies even after machines are removed.

---

### 4. Discovery difficulty independent of learnability
- Construct $\mathcal{S}$ such that some deep strategies have low RLE complexity (and thus high learnability $q(s)$).
- Keep discovery difficulty high via depth-dependent costs and low $p_{\mathrm{applicable}}$ (sparse success probability).
- Control learnability exclusively via $\gamma$.

Outcome: machines introduce strategies that are rare and costly to discover (due to low $p_{\mathrm{applicable}}$ and high costs) but easy to transmit (due to low RLE complexity), leading to persistent cultural change without corresponding increases in individual exploration depth.

---

### Summary

Across all conditions, differences in computational capacity affect which strategies enter culture, while social transmission and learnability determine which discoveries persist. Population size alone increases parallel search but does not alter the individually optimal depth of exploration.

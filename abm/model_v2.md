## Model

### Environment and Solutions
Agents face a task with two solution classes:
- $M$: a myopic solution that is easy to discover and yields reliable short-run payoff.
- $O$: an optimal solution that yields higher payoff but is harder to discover.

Each agent has a learning horizon $T$. The probability of discovering each solution class when it is attempted is:
$$
p_M(T), \quad p_O(T),
$$
with $p_M(T) \ge p_O(T)$ for short horizons.

Payoffs are:
- $R_M$ for successful discovery of $M$,
- $R_O$ for successful discovery of $O$,
- $R_0$ for discovering neither solution,
with $R_O > R_M \ge R_0$.

---

### Inherited Traits
Each agent carries two heritable traits transmitted **vertically** (parent to offspring):

1. **Exploration propensity** $\theta \in (0,1)$  
   Interpreted as a latent risk-seeking or exploration trait that determines how learning effort is allocated.
2. **Solution token** $s \in \{M, O, \varnothing\}$  
   Representing the solution transmitted from the parent (if any).

---

### Vertical Transmission
Let $\theta_p$ and $s_p$ denote the parent’s traits.

#### Exploration Propensity
Transmission occurs with noise in logit space:
$$
\beta_p = \log\frac{\theta_p}{1-\theta_p}, \qquad
\beta_c = \beta_p + \epsilon, \quad \epsilon \sim \mathcal{N}(0,\sigma^2),
$$
$$
\theta_c = \frac{1}{1+e^{-\beta_c}}.
$$

#### Solution Transmission
The child inherits the parent’s solution with probability $1-\mu_s$:
$$
s_c =
\begin{cases}
s_p, & \text{with probability } 1-\mu_s, \\
\varnothing, & \text{with probability } \mu_s.
\end{cases}
$$

---

### Phase 1: Learning (Private Exploration)
An agent with exploration propensity $\theta$ allocates learning attempts as:
$$
\Pr(\text{attempt } O) = \theta, \qquad \Pr(\text{attempt } M) = 1-\theta.
$$

When attempting $O$ or $M$, discovery occurs with probabilities $p_O(T)$ and $p_M(T)$, respectively.

At the end of learning, the agent retains the **best discovered solution**:
$$
s^\star =
\begin{cases}
O, & \text{if } O \text{ is discovered}, \\
M, & \text{else if } M \text{ is discovered}, \\
\varnothing, & \text{otherwise}.
\end{cases}
$$

---

### Phase 2: Demonstration (Public Exploitation)
Agents exploit $s^\star$ during a demonstration or evaluation window. Demonstration payoff is:
$$
\Pi_{\text{demo}}(s^\star) =
\begin{cases}
R_O, & s^\star = O, \\
R_M, & s^\star = M, \\
R_0, & s^\star = \varnothing.
\end{cases}
$$

---

### Reproduction and Cultural Selection
Agents reproduce (or persist as cultural lineages) in proportion to their demonstration payoff. The expected number of offspring is:
$$
w = \exp\!\big(\alpha \, \Pi_{\text{demo}}(s^\star)\big),
$$
where $\alpha \ge 0$ controls the strength of selection.

Each offspring inherits $\theta$ and $s$ from its parent according to the transmission rules above.

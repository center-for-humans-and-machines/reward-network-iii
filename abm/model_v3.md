## Selective Social Learning of Exploration Propensity with Risky Learning and Imperfect Solution Transmission

### Environment and Solutions
Agents face a task with two solution classes:
- $M$: an easy-to-discover solution with moderate payoff.
- $O$: a hard-to-discover solution with higher payoff.

Each agent has a learning horizon $T\in\mathbb{N}$ corresponding to the number of learning trials. In each learning trial, the agent chooses which solution class to attempt and may discover that solution. Demonstration payoffs are
$$
R_O > R_M > R_\varnothing,
$$
where $R_\varnothing$ is the payoff if the agent enters demonstration with no usable solution. There is **no explicit cost** during learning; risk arises because failure to acquire a solution implies low demonstration payoff and thus low reproductive/cultural success.

We parameterize *per-attempt* discovery success by constants
$$
\pi_M \in (0,1),\qquad \pi_O \in (0,1),
$$
with typically $\pi_O < \pi_M$ in human-like regimes.

---

### Heritable Cultural Trait (Exploration Propensity)
Each agent carries an exploration propensity $\theta \in (0,1)$, interpreted as a tacit risk-seeking exploration trait that determines how learning attempts are allocated:
$$
\Pr(\text{attempt } O) = \theta, \qquad \Pr(\text{attempt } M)=1-\theta.
$$

---
### Learning Phase (Private Exploration)
Each agent has a learning horizon of $T$ trials. On each trial, the agent chooses which solution class to attempt and may discover it.

#### Attempt allocation
On every learning trial, the agent:
$$
\Pr(\text{attempt } O) = \theta, \qquad \Pr(\text{attempt } M) = 1-\theta.
$$

#### Discovery
If solution class $X \in \{M,O\}$ is attempted on a given trial, discovery succeeds with a fixed per-trial probability $\pi_X$, independently across trials:
$$
\pi_O < \pi_M.
$$

#### Discovery probabilities over $T$ trials
Because each trial independently attempts $O$ with probability $\theta$ and succeeds with probability $\pi_O$, the probability that $O$ is **never** discovered over $T$ trials is $(1-\theta\pi_O)^T$. Thus, the probability that $O$ is discovered at least once is
$$
p_O(\theta,T) = 1 - (1-\theta\pi_O)^T.
$$

Similarly, the probability that $M$ is discovered at least once is
$$
p_M(\theta,T) = 1 - \big(1-(1-\theta)\pi_M\big)^T.
$$

The probability that the agent ends the learning phase with **no solution** is
$$
p_\varnothing(\theta,T) = \big(1-\theta\pi_O-(1-\theta)\pi_M\big)^T.
$$

#### Learning outcome
At the end of learning, the agent has a discovered-solution set
$$
D \subseteq \{M,O\},
$$
where $X \in D$ if solution class $X$ was discovered at least once during the $T$ trials.

---

### Demonstration Phase (Public Exploitation)
Before demonstration, the agent may additionally possess a socially learned solution token $s^{\text{soc}} \in \{M,O,\varnothing\}$ (defined below). The agent demonstrates the best available solution:
$$
s^\star =
\begin{cases}
O, & \text{if } O \in D \text{ or } s^{\text{soc}}=O,\\
M, & \text{else if } M \in D \text{ or } s^{\text{soc}}=M,\\
\varnothing, & \text{otherwise.}
\end{cases}
$$

Demonstration payoff is
$$
\Pi_{\text{demo}}(s^\star)=
\begin{cases}
R_O, & s^\star=O,\\
R_M, & s^\star=M,\\
R_\varnothing, & s^\star=\varnothing.
\end{cases}
$$

---

### Selection and Reproduction Success
Agents are selected as cultural parents/teachers (or reproduce) with probability increasing in demonstration payoff. A convenient choice is softmax selection:
$$
\Pr(i \text{ is chosen}) \propto \exp\!\big(\alpha \, \Pi_{\text{demo}}(s_i^\star)\big),
$$
where $\alpha \ge 0$ controls selection strength.

---

### Selective Social Learning of the Trait $\theta$ (Oblique Transmission)
Each new agent (learner) chooses a teacher $j$ according to the selection rule above and acquires the exploration propensity $\theta$ via noisy copying in logit space:
$$
\beta_j=\log\frac{\theta_j}{1-\theta_j}, \qquad
\beta_{\text{new}}=\beta_j+\epsilon, \quad \epsilon \sim \mathcal{N}(0,\sigma^2),
$$
$$
\theta_{\text{new}}=\frac{1}{1+e^{-\beta_{\text{new}}}}.
$$

---

### Imperfect Transmission of Discovered Solutions
In addition to trait transmission, teachers may transmit a solution token derived from their demonstrated best solution $s_j^\star$.

Let $\tau \in [0,1]$ denote the probability that the teacher’s demonstrated solution is successfully transmitted to the learner. Then:
$$
s^{\text{soc}}=
\begin{cases}
s_j^\star, & \text{with probability } \tau,\\
\varnothing, & \text{with probability } 1-\tau.
\end{cases}
$$

(Optionally, transmission can be made solution-dependent, e.g., $\tau_O < \tau_M$ if $O$ is harder to communicate; the baseline model uses a single $\tau$.)

---

### Generational Dynamics
For each generation:
1. Each new learner selects a teacher using the payoff-based rule $\Pr(i)\propto \exp(\alpha \Pi_{\text{demo}}(s_i^\star))$.
2. The learner copies the teacher’s exploration propensity $\theta$ via noisy logit copying.
3. The learner receives a socially transmitted solution token $s^{\text{soc}}$ with probability $\tau$ (otherwise none).
4. The learner enters the learning phase, discovering solutions according to $\theta$ and $p_O(T), p_M(T)$.
5. The learner enters demonstration and exploits $s^\star$ (best among discovered and socially learned solutions), yielding $\Pi_{\text{demo}}(s^\star)$.
6. Selection operates on $\Pi_{\text{demo}}$ to determine teachers for the next generation.

Only the exploration propensity $\theta$ and solution token $s^{\text{soc}}$ are transmitted; learning trajectories and payoffs are not directly copied.

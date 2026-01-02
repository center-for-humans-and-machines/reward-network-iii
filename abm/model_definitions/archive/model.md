## S1. Model: Cultural Transmission of Exploration Propensity via Bayesian Inference

### Environment and Solutions
Agents repeatedly face a task with two solution classes:
- $M$: a myopic solution that is easy to discover and yields reliable short-run payoffs.
- $O$: an optimal solution that yields higher payoff but is harder to discover or incurs early costs.

Let $T$ denote the learning horizon. The probability of successfully discovering each solution class is:
$$
p_M(T), \quad p_O(T),
$$
with $p_M(T) \ge p_O(T)$ for short horizons.

Payoffs are:
- $R_M$ for successful discovery of $M$,
- $R_O$ for successful discovery of $O$,
- $R_0$ for failure to discover either solution.

Attempting $O$ may incur an additional cost $c_O \ge 0$.

---

### Individual Learning and Action Selection
Each agent holds a belief $\theta \in (0,1)$ representing the probability that attempting $O$ is worthwhile. Conditional on $\theta$, the agent samples which solution to attempt:
$$
\Pr(\text{attempt } O) = \theta, \qquad \Pr(\text{attempt } M) = 1-\theta.
$$

Given the attempted solution, success is realized probabilistically via $p_O(T)$ or $p_M(T)$, and payoffs are obtained accordingly.

---

### Cultural Transmission via Teacher Observation
Cultural transmission operates through observation of teachers’ exploration choices. Agents do not observe outcomes, payoffs, or solutions.

Each learner observes $X$ independent demonstrations from a single teacher. Each demonstration records only whether the teacher attempted $O$ or $M$.

Let:
- $n_O$ denote the number of observed attempts of $O$,
- $n_M = X - n_O$ denote the number of observed attempts of $M$.

---

### Bayesian Updating
Learners assume the teacher has a stable latent propensity $\theta$ to attempt $O$.

#### Prior
All learners begin with an unbiased prior:
$$
\theta \sim \mathrm{Beta}(1,1).
$$

#### Posterior
After observing $n_O$ attempts of $O$ in $X$ trials, the posterior distribution is:
$$
\theta \mid \text{data} \sim \mathrm{Beta}(1+n_O,\; 1+n_M).
$$

#### Behavioral Update
The learner’s own exploration probability is set equal to the posterior mean:
$$
\theta_{\text{learner}} = \mathbb{E}[\theta \mid \text{data}] = \frac{1+n_O}{2+X}.
$$

---

### Generational Dynamics
The process iterates across generations:

1. Agents in generation $t$ act according to their belief $\theta_t$.
2. Payoffs are realized based on the attempted solution and learning horizon $T$.
3. Teachers for generation $t+1$ are selected probabilistically as a function of realized payoff.
4. Learners in generation $t+1$ observe $X$ demonstrations from their selected teacher and update $\theta$ as described above.

Only exploration choices are culturally transmitted; no strategies, outcomes, or payoffs are copied.

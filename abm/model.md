## Agent based Model

We derived a parsimonic model of our multi-generational cultural experimental design. For each generation, we apply: (i) individual learning, (ii) reward-based teacher selection, (iii) social transmission, (iv) strategy integration, (v) demonstration, and (vi) reward updating.

Each simulation consists of $M=50$ independent replications, $G = 20$ generations, and $N_{\text{gen}} = 20$ agents per generation. Depending on condition, a subset $N_{\text{m}} = 5$ of agents are machines, with distinct exploration budgets $K_{\text{m}}$ and optimality biases $q_{\text{m}}$ which we maniplute.

---

### Individual Learning

Each agent performs $k_{\text{h}} = 10$ trials, allocated between exploring optimal and myopic strategies governed by a exploration bias $q_{\text{h}} = 0.01$:

$$
k_{\text{opt}} \sim \text{Binomial}(k, q_{\text{opt}}), \quad
k_{\text{myo}} = k - k_{\text{opt}}.
$$

Given per-trial discovery rates $d_{\text{opt}}$, which we manipulate, and $d_{\text{myo}} = 0.5$, cumulative discovery probabilities are:

$$
D_{\text{opt}} = 1-(1-d_{\text{opt}})^{k_{\text{opt}}}, \quad
D_{\text{myo}} = 1-(1-d_{\text{myo}})^{k_{\text{myo}}}.
$$

Agents stochastically discover an optimal solution ($s=1$), a myopic solution ($s=0$), or nothing ($s=-1$), with priority given to optimal discovery.

---

### Demonstration and Rewards

Successful agents produce $K_{\text{demo}} = 10$ demonstrations of their discovered strategy. Payoffs combine demonstrated outputs and Gaussian noise:

$$
R = n_{\text{opt}}R_{\text{opt}} + n_{\text{myo}}R_{\text{myo}} + \epsilon,
\quad
\epsilon \sim \mathcal{N}(0,\sigma^2).
$$

with $R_{\text{opt}} = 1$,  $R_{\text{myo}} = 0.5$ and $\epsilon = 0.5$. Rewards provide the basis for social selection.

---

### Teacher Selection and Social Learning

In each generation $g>0$, agents sample $k$ candidate teachers uniformly at random from the previous generation. Teachers are selected via softmax weighting over past rewards:

$$
p_i \propto \exp(r_i/\tau),
$$

where $\tau$ controls selection intensity.

Learners copy the selected teacher’s optimality state with probability $\lambda$. Individual and social learning outcomes are combined as:

$$
s_g = \max(s_{\text{ind}}, s_{\text{social}}),
$$

ensuring that socially or individually acquired optimal solutions dominate myopic or null states. In the first generation, learning is purely individual.

---

### Experimental Conditions

We model a scenario in which human agents are constrained by limited exploration capacity ($k_{\text{h}} = 10$), an intermediate “myopic” solution is easy to discover ($d_{\text{myo}} = 0.5$), and agents exhibit a strong bias toward this solution ($q_{\text{h}} = 0.01$). We compare populations with no machines, with machines present only in the first generation, and with machines present in every generation. We examine machine agents with increasing exploration capacity ($K_{\text{m}}$), and compare agents with biases matching human agents ($q_{\text{m}} = 0.01$) to those with neutral bias ($q_{\text{m}} = 0.5$). We systematically scan over learnability ($1-\lambda$) and discoverability ($-\log_{10} d_{\text{opt}}$).

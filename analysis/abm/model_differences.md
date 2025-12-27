# Model Implementation Differences

This document details all differences between the model specification in `model.md` and the actual implementation in this folder.

---

## 1. Action Selection Logic (Critical)

**Model Specification:**
- Action selection uses the mixture distribution:
  $$
  x \sim (1-\varepsilon)\,\delta_{\emptyset}
  \;+\;
  \varepsilon\Big[
     \phi\,\mathrm{Unif}(\mathcal{K}_i)
     + (1-\phi)\,\mathrm{Unif}(\mathcal{X}_{\le d_i})
  \Big]
  $$
- With probability $1-\varepsilon$, the agent selects the safe prefix (empty strategy).
- With probability $\varepsilon\phi$, it samples uniformly from the repertoire $\mathcal{K}_i$.
- With probability $\varepsilon(1-\phi)$, it explores by sampling uniformly from $\mathcal{X}_{\le d_i}$.

**Implementation:**
- In `agents.py:act()`, the logic is inverted:
  ```python
  go_save = (self.rng.random(size=(self.R, self.N)) < self.eps_i[:, self.g, :])
  go_exploit = (self.rng.random(size=(self.R, self.N)) < self.phi_i[:, self.g, :])
  ```
- This means:
  - With probability $\varepsilon$, the agent selects the safe prefix (WRONG - should be $1-\varepsilon$).
  - Otherwise, with probability $\phi$, it exploits from repertoire.
  - Otherwise, it explores.

**Impact:** This fundamentally changes the exploration-exploitation trade-off. Agents will explore/exploit with probability $1-\varepsilon$ instead of $\varepsilon$, making them much more exploratory than the model specifies.

---

## 2. Strategy Length Calculation (Critical)

**Model Specification:**
- For a prefix $x$, $\ell(x)$ denotes the **length of the prefix** (i.e., the number of bits in the binary representation).

**Implementation:**
- In `task.py:step()` (line 119) and `task.py:__post_init__()` (line 73):
  ```python
  x_len = np.log2(x_idx + 1).astype(np.int32)
  ```
- This computes the binary logarithm of (index + 1), which approximates the bit length but is not exact.

**Impact:** This causes incorrect length calculations for many strategies. For example:
- Index 2 (binary "10") has length 2, but `log2(2+1) ≈ 1.58` → 1 (incorrect).
- Index 4 (binary "100") has length 3, but `log2(4+1) ≈ 2.32` → 2 (incorrect).
- Index 5 (binary "101") has length 3, but `log2(5+1) ≈ 2.58` → 2 (incorrect).
- Index 7 (binary "111") has length 3, and `log2(7+1) = 3` (correct by coincidence).

This affects both cost calculations ($-c \ell(x)$) and reward calculations ($r \lambda^{\ell(s)}$), leading to systematically incorrect payoffs. Strategies with lengths that are not powers of 2 minus 1 will have incorrect length calculations.

**Correct Implementation Should Be:**
- For index 0: length = 0 (empty prefix)
- For index n > 0: length = `n.bit_length()` or `len(bin(n)) - 2` (the actual number of bits in the binary representation)

---

## 3. Strategy Discovery Condition

**Model Specification:**
- "If $x \in \mathcal{S}$, the prefix is added to the agent's repertoire" (line 124-127 of model.md).

**Implementation:**
- In `agents.py:update()` (line 168):
  ```python
  discovered = (reward > 0) & alive
  self.K[r_idx, self.g, n_idx, x_idx] |= discovered
  ```
- Strategies are added to the repertoire if the reward is positive (and the agent is alive).

**Impact:** 
- These conditions are different: a strategy $s \in \mathcal{S} \setminus \mathcal{W}_p$ (in the strategy set but not applicable to the task) has payoff $-c \ell(s) < 0$ according to the model, so it would NOT be added under the implementation's condition.
- However, if the strategy is applicable ($s \in \mathcal{W}_p$), it gets reward $r \lambda^{\ell(s)} - c \ell(s)$, which could be positive, so it would be added.
- The model specifies that ALL strategies in $\mathcal{S}$ should be added when encountered, regardless of whether they're applicable to the current task.

**Note:** This difference means non-applicable strategies from $\mathcal{S}$ are not retained in the repertoire, contradicting the model's specification that strategies are "portable cultural objects" independent of task applicability.

---

## 4. Exploration Space Distribution

**Model Specification:**
- The model states: "sampling uniformly from the combinatorially large space $\mathcal{X}_{\le d_i}$, which assigns exponentially greater probability mass to longer prefixes" (line 121 of model.md).
- However, the mathematical definition says "$\mathrm{Unif}(\mathcal{X}_{\le d_i})$", which should be uniform over all prefixes of length $\leq d_i$.

**Implementation:**
- In `agents.py:explore()` (line 122):
  ```python
  max_idx = np.minimum(self.pow2[self.d_i[:, self.g, :]], self.X).astype(np.int32)
  x_idx = self.rng.integers(0, max_idx, size=(self.R, self.N), dtype=np.int32)
  ```
- This samples uniformly from indices $[0, 2^{d_i})$.

**Impact:** 
- Since indices map bijectively to prefixes, and there are $2^{\ell-1}$ prefixes of length $\ell$ (for $\ell > 0$), sampling uniformly from $[0, 2^{d_i})$ does assign exponentially greater probability to longer prefixes.
- However, the comment in the model says "uniformly from the combinatorially large space" which is ambiguous. The implementation matches the mathematical notation if we interpret $\mathcal{X}_{\le d_i}$ as the set of all indices up to $2^{d_i}-1$.

**Clarification Needed:** The model's text says "uniformly" but then notes it assigns "exponentially greater probability mass to longer prefixes", which suggests the distribution is NOT uniform over prefixes (but uniform over indices). The implementation is consistent with uniform sampling over indices, which matches the comment but may not match the intended interpretation of "uniform over prefixes".

---

## 5. Initial Repertoire

**Model Specification:**
- "a personal repertoire $\mathcal{K}_i \subseteq \mathcal{S}$, initially containing at least $s_0$" (line 97 of model.md).

**Implementation:**
- In `agents.py:__post_init__()` (line 94):
  ```python
  self.K = np.zeros((self.R, self.G, self.N, self.X), dtype=bool)
  ```
- The repertoire is initialized to all `False`.
- The safe strategy (index 0) is only set to `True` in `agents.py:learn()` (line 210) when learning from a teacher, but NOT at initialization.

**Impact:** For generation 0, agents start with an empty repertoire (no safe strategy), which violates the model specification. They must discover the safe strategy through exploration or they won't have it available.

---

## 6. Task-Strategy Associations

**Model Specification:**
- "Agents do not maintain task–strategy associations. Instead, strategies are treated as portable cultural objects..." (line 129 of model.md).
- However, earlier it states: "For each task $p$, the agent tracks the best-performing strategy discovered so far, indexed by $b_{i,p}$" (line 99 of model.md).

**Implementation:**
- In `agents.py` (lines 32-33), the class docstring mentions:
  ```python
  # - b: best-known strategy index per task                                            # [R,G,N,P]
  # - best_r: best observed payoff per task                                            # [R,G,N,P]
  ```
- However, these arrays are **never initialized or used** in the code.

**Impact:** The implementation correctly does not maintain task-strategy associations (matching the "portable cultural objects" description), but the docstring is misleading. The earlier mention of $b_{i,p}$ in the model appears to be unused or represents a conceptual tracking that isn't implemented.

---

## 7. Cultural Evolution Parameter

**Model Specification:**
- Cultural evolution uses payoff-biased copying with parameter $\beta_{\mathrm{evo}}$:
  $$
  \Pr(\text{parent}=i) \propto \exp\!\bigl(\beta_{\mathrm{evo}}\, \mathrm{Perf}_i\bigr)
  $$

**Implementation:**
- In `agents.py`, teacher selection uses `beta_teacher`:
  ```python
  teach_probs = softmax(self.agent_config.beta_teacher * self.perf[:, self.g, :], axis=1)
  ```
- There is no separate `beta_evo` parameter.

**Impact:** The implementation uses the same parameter for both teacher selection (social transmission) and cultural evolution (depth inheritance). The model suggests these could be different, allowing independent control of social learning strength vs. evolutionary selection pressure.

---

## 8. Safe Strategy Payoff

**Model Specification:**
- "$r_{\mathrm{safe}} > 0$ is the payoff of the default strategy" (line 67 of model.md).

**Implementation:**
- In `config.yml` (line 36):
  ```yaml
  r_safe: 0.0
  ```
- The safe strategy payoff is set to 0.0, not a positive value.

**Impact:** This violates the model specification that $r_{\mathrm{safe}} > 0$. 

Additionally, if `r_safe` is `None`, the safe strategy (index 0) gets the regular reward calculation:
- Length calculation: `log2(0+1) = 0`
- Cost: `c * 0 = 0`
- If applicable: bonus = `r_scale * (lam ** 0) = r_scale`
- Reward = bonus - cost = `r_scale`

This means the safe strategy gets reward `r_scale` when applicable, rather than the specified `r_safe`. When `r_safe` is explicitly set to 0.0, the safe strategy always gets reward 0.0 (regardless of applicability), which also doesn't match the model's specification that it should be a positive constant.

---

## 9. Repertoire Representation

**Model Specification:**
- The repertoire $\mathcal{K}_i \subseteq \mathcal{S}$ is a subset of the latent strategy set.

**Implementation:**
- The repertoire `K` is stored as a boolean array of shape `[R, G, N, X]`, where `X = 2^L` is the full behavioral space (not just $\mathcal{S}$).

**Impact:** The implementation stores membership information for the entire behavioral space $\mathcal{X}$, not just $\mathcal{S}$. This is more memory-intensive but allows tracking which elements of $\mathcal{X}$ have been discovered. This difference is mostly a representational choice and doesn't fundamentally change the logic, as long as only elements of $\mathcal{S}$ can yield positive rewards (which they can't, due to issue #3).

---

## 10. Applicability Matrix Dimensions

**Model Specification:**
- Applicability is defined as $\mathcal{W}_p \subseteq \mathcal{S}$ for each task $p$.

**Implementation:**
- In `task.py:build_applicability()` (line 97):
  ```python
  self.W = np.zeros((self.R, self.P, self.X), dtype=bool)
  ```
- The applicability matrix has shape `[R, P, X]`, covering the full behavioral space, not just $\mathcal{S}$.

**Impact:** Similar to issue #9, this is a representational choice. The matrix stores applicability for all of $\mathcal{X}$, but only elements of $\mathcal{S}$ should have meaningful applicability (non-applicable strategies get negative payoffs anyway). This doesn't fundamentally change behavior but is inconsistent with the mathematical notation.

---

## 11. Strategy Distribution Generation

**Model Specification:**
- The model does not specify how the latent strategy set $\mathcal{S}$ is generated or distributed.

**Implementation:**
- In `task.py:strategies_from_distribution()` (lines 85-88):
  ```python
  for l in range(1, self.L + 1):
      bits = self.rng.integers(0, 2, size=l, dtype=np.int8)
      strategy_str = ''.join(str(bit) for bit in bits)
      strategies.append(strategy_str)
  ```
- When using "uniform" distribution, the implementation generates exactly one random strategy per length from 1 to L.

**Impact:** This is an implementation detail not specified in the model. The "uniform" name might be misleading since it generates a fixed set of strategies rather than sampling from a uniform distribution over all possible strategies.

---

## 12. Empty Prefix Representation

**Model Specification:**
- The empty prefix `""` is represented as index 0 in the bijection: "with $0 \mapsto ""$" (line 14 of model.md).

**Implementation:**
- The code consistently uses index 0 for the safe/empty strategy (e.g., `task.py:98`, `task.py:125`, `agents.py:210`).

**Status:** ✓ **Correct** - This matches the model specification.

---

## 13. RLE Computation

**Model Specification:**
- RLE is computed as:
  $$
  \mathrm{RLE}(s) =
  \begin{cases}
  0, & s = "", \\[4pt]
  1 + \sum_{m=2}^{\ell(s)} \mathbf{1}[s_m \neq s_{m-1}], & \text{otherwise}.
  \end{cases}
  $$

**Implementation:**
- In `utils.py:compute_rle()` (lines 27-30):
  ```python
  def compute_rle(value: int) -> int:
      if value == 0:
          return 0
      return (value ^ (value >> 1)).bit_count()
  ```
- This uses a Gray code transition count trick.

**Status:** ✓ **Correct** - Verified through testing that this implementation correctly computes the RLE as specified in the model.

---

## 14. Learnability Function

**Model Specification:**
- Learnability is $q(s) = \sigma(\alpha - \gamma \mathrm{RLE}(s))$ where $\sigma$ is the logistic function.

**Implementation:**
- In `task.py:__post_init__()` (line 75):
  ```python
  self.x_q[idx] = sigmoid(self.task_config.alpha - self.task_config.gamma * rle)
  ```
- The `sigmoid` function in `utils.py` is the logistic function: $1/(1 + e^{-x})$.

**Status:** ✓ **Correct** - Matches the model specification.

---

## 15. Social Transmission

**Model Specification:**
- Teacher selection: $\Pr(i) \propto \exp(\beta \mathrm{Perf}_i)$
- Each transmitted strategy $s$ is adopted with probability $q(s)$.

**Implementation:**
- Teacher selection in `agents.py:select_teacher()` (line 183) uses softmax with `beta_teacher`.
- Strategy transmission in `task.py:transmit()` (lines 141-143) uses independent Bernoulli trials with probability `x_q`.

**Status:** ✓ **Correct** - Matches the model specification (except for the parameter name `beta_teacher` vs $\beta$).

---

## 16. Depth Evolution

**Model Specification:**
- Offspring exploration depth: $d' = \mathrm{clip}(d + \Delta)$ where $\Delta \in \{-1, 0, +1\}$.

**Implementation:**
- In `task.py:transmit()` (lines 148-149):
  ```python
  d_mutation = self.rng.integers(-1, 2, size=(self.R, self.N), dtype=np.int32)
  student_d = np.clip(teacher_d + d_mutation, min_d, max_d)
  ```

**Status:** ✓ **Correct** - Matches the model specification.

---

## Summary of Critical Issues

The following issues are **critical** and fundamentally change the model's behavior:

1. **Action Selection Logic (Issue #1)**: Probability inversion changes exploration rates.
2. **Strategy Length Calculation (Issue #2)**: Incorrect length affects all payoff calculations.
3. **Strategy Discovery Condition (Issue #3)**: Non-applicable strategies from $\mathcal{S}$ are not retained.
4. **Initial Repertoire (Issue #5)**: Agents don't start with the safe strategy in generation 0.

The following are **minor** or **documentation** issues:

5. Task-strategy associations (documented but unused)
6. Cultural evolution parameter naming
7. Safe strategy payoff being 0.0 (should be > 0)
8. Repertoire/applicability matrix dimensions (representational choices)


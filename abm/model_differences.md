# Model Implementation Differences

This document details remaining differences between the model specification in `model.md` and the actual implementation in this folder.

---

## 1. Safe Strategy Payoff

**Model Specification:**
- "$r_{\mathrm{safe}} > 0$ is the payoff of the default strategy" (line 67 of model.md).

**Implementation:**
- In `config.yml` (line 36):
  ```yaml
  r_safe: 0.0
  ```
- The safe strategy payoff is set to 0.0, not a positive value.

**Impact:** This violates the model specification that $r_{\mathrm{safe}} > 0$. When `r_safe` is explicitly set to 0.0, the safe strategy always gets reward 0.0 (regardless of applicability), which doesn't match the model's specification that it should be a positive constant.

---

## 2. Strategy Distribution Generation

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

## Summary

The following issues have been **resolved**:

1. ✅ **Action Selection Logic**: Fixed - code now correctly uses $(1-\varepsilon)$ for safe strategy selection
2. ✅ **Strategy Length Calculation**: Fixed - code now uses `bit_length()` instead of `log2()`
3. ✅ **Strategy Discovery Condition**: Fixed in model - now specifies "if payoff is positive"
4. ✅ **Initial Repertoire**: Fixed in model - now specifies repertoire starts empty and safe strategy is always available
5. ✅ **Task-Strategy Associations**: Fixed - removed from docstring and model
6. ✅ **Exploration Space Distribution**: Clarified in model
7. ✅ **Repertoire/Applicability Representation**: Updated in model to reflect full behavioral space representation
8. ✅ **Cultural Evolution Parameter**: Fixed - added `beta_evo` parameter for separate control of depth inheritance

The remaining differences are minor implementation details or configuration choices that don't fundamentally affect the model's behavior.

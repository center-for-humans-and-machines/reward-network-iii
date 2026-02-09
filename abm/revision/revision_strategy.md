# Revision Strategy

## Overall Goal
The revision aims to address the editor’s and reviewers’ concerns about **interpretation and causal attribution** without expanding the empirical dataset. The central move is to **reframe the contribution** away from claims about human vs. machine cognitive superiority and toward a **finite-time, system-level account of exploration throughput** and its interaction with **cultural transmission**.

---

## Core Reframing
We explicitly distinguish between:
- **Innovation supply**: whether a high-payoff solution appears within a finite time window.
- **Cultural diffusion and persistence**: whether an introduced solution spreads and survives.

The revision makes clear that:
- Innovation supply scales with **total exploration capacity**,
  $$
  T_{\text{total}} = \sum_{g=1}^{G}\sum_{i=1}^{N_g} T_{i,g},
  $$
  rather than with per-trial cognitive differences.
- Machines matter primarily because they contribute disproportionate exploration mass within the experimental time window.
- Humans could in principle discover the same solutions given sufficient exploration, but this asymptotic question is orthogonal to the finite-time cultural dynamics studied here.

---

## Addressing Reviewer 1 (Human–Machine Confound, “Could Humans Do This with More Time?”)
**Reviewer concern:** The design does not disentangle human cognitive limitations from unequal learning opportunities.

**Revision response:**
- We explicitly concede that the experiment does not test asymptotic human capability.
- We reframe the contribution: the paper studies **what happens when exploration capacity is unevenly distributed in time**, not whether humans are incapable.
- The new model shows that innovation probability depends on total exploration capacity,
  $$
  P(O_{\text{intro}}) = 1-(1-\pi_O)^{T_{\text{total}}},
  $$
  and that this capacity can be supplied by a small number of high-throughput agents.
- Simulation experiments demonstrate that similar outcomes arise when total exploration is matched but concentrated in few agents versus distributed broadly.

**Key message to reviewer:** The question “would humans find it with enough time?” is acknowledged but shown to be beside the point for finite-time cultural evolution.

---

## Addressing Reviewer 4 (Payoff Bias, “Isn’t This Just Copying Good Solutions?”)
**Reviewer concern:** The results may simply reflect payoff-biased social learning once a good solution is available.

**Revision response:**
- We agree and make this mechanism explicit.
- The revised analysis decomposes outcomes as:
  $$
  P(O_{\text{final}}) = P(O_{\text{intro}})\cdot P(O_{\text{final}}\mid O_{\text{intro}}),
  $$
  showing that:
  - Throughput governs **introduction**.
  - Payoff bias ($\alpha$) and transmission fidelity ($\tau$) govern **persistence**.
- New simulation experiments vary $\alpha$ and $\tau$ to show when introduced innovations persist or are lost.

**Key message to reviewer:** Yes, payoff-biased copying explains persistence; the contribution is showing how throughput asymmetries determine which solutions enter the cultural pool in the first place.

---

## Clarifying the Role of Machines
- Machines are modeled as agents with larger exploration horizons ($T_M \gg T_H$) but identical per-trial discovery rates.
- We avoid any notion of a **exploration bias**. Some solutions are just harder to discover for humans and machines alike. 
- This avoids claims of intrinsic machine intelligence and aligns the model with the empirical design.
- We emphasize that machines instantiate an extreme but realistic form of **concentrated exploration capacity**.

---

## Why the New Model Is Not Trivial
Although the dependence of discovery probability on total trials is mathematically simple, its implications are non-trivial for cultural evolution:
- Cultural systems operate under finite time and attention constraints.
- Exploration capacity is unevenly allocated in practice.
- Selective cultural transmission amplifies rare innovations once they appear.

The model’s simplicity is a feature: it isolates the minimal conditions under which high-throughput agents can reshape cultural trajectories.

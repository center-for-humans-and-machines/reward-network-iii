# Vectorized ABM implementation (NumPy) — class-centric design

## Design goals implemented

- Core logic lives in two classes:
  - TaskEnv: task generation + reward computation
  - AgentPop: action/strategy sampling + within-lifetime updates + transmission
- Only a single instance of each class is created at runtime; all state is stored in NumPy arrays.
- Vectorization over dimensions: replication/condition (R) × generation (G) × agent (N) × task (P) × strategy (S).
- Matrix dimension annotations at end of lines using abbreviations: [R,G,N,P,S,L,T,K].
- np.einsum used for non-trivial multiplications.

## Notation
- X: all prefixes up to length L (implicit; not enumerated)
- S: latent strategy set size
- W: applicability W[r,p,s] ∈ {0,1} for strategy s on task p



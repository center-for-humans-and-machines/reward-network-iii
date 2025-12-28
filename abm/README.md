# Agent-Based Model (ABM) for Cultural Exploration

Agent-based model simulating cultural exploration where agents discover, transmit, and apply reusable strategies across tasks. Supports both human and machine agents with configurable exploration, social learning, and cultural evolution.

## Documentation

- **`model.md`**: Complete mathematical specification
- **`design.md`**: Implementation design notes (vectorized NumPy approach)
- **`model_differences.md`**: Differences between specification and implementation

## Running the Model

```bash
python simulation.py -c example_config.yml -o experiments/example 
```

**Arguments:**
- `-c, --config`: Path to configuration YAML file (required)
- `-o, --output`: Path to output directory (required)

**Output:**
- `K.parquet`: Agent repertoires (known strategies)
- `perf.parquet`: Performance (cumulative payoffs)

See `config.yml` for parameter configuration.

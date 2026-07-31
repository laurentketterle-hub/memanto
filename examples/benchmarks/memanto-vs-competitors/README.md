# Memanto vs Competitors Benchmark

A comprehensive, self-contained benchmark comparing **Memanto** against leading memory systems:

- **Memanto** — Active companion memory agent (Moorcheh backend)
- **Mem0** — Local/cloud memory layer with LLM extraction
- **Zep** — Graph-based memory with temporal understanding
- **Letta** — Stateful agent platform (formerly MemGPT)
- **LangMem** — LangChain ecosystem memory management

## Quick Start

```bash
cd examples/benchmarks/memanto-vs-competitors

# Run self-test (validates benchmark integrity, no API keys needed)
python run_benchmark.py --test

# Run all adapters through all scenarios (uses in-memory mode without API keys)
python run_benchmark.py

# Compare specific systems
python run_benchmark.py --adapters memanto mem0 zep

# Run specific scenario
python run_benchmark.py --scenarios temporal

# Save results
python run_benchmark.py --output results.json

# List available options
python run_benchmark.py --list
```

## Scenarios

Four synthetic scenarios test different memory capabilities:

| Scenario | Description | Key Metric |
|----------|------------|------------|
| **Basic Recall** | Simple fact storage and retrieval | Accuracy |
| **Temporal Reasoning** | Tracking preference changes over time | Staleness Avoidance |
| **Contradiction Handling** | Resolving conflicting information | Precision |
| **Multi-Session Persistence** | Cross-session memory recall | Recall@k |

## Metrics

- **Recall@k**: Fraction of evidence turns found in top-k results
- **Precision@k**: Fraction of retrieved items that are evidence
- **LLM Judge Scoring** (3 dimensions, 0-5 each):
  - Accuracy: Did the system recall correct information?
  - Staleness Avoidance: Did it avoid outdated/stale information?
  - Precision: Was the answer concise and on-topic?
- **Latency**: p50 and p95 response time percentiles

## Architecture

```
memanto-vs-competitors/
├── adapters/           # Memory system adapters
│   ├── base.py         #   Protocol, dataclasses, scoring
│   ├── memanto_adapter.py
│   ├── mem0_adapter.py
│   ├── zep_adapter.py
│   ├── letta_adapter.py
│   └── langmem_adapter.py
├── datasets/           # Benchmark scenarios
│   └── synthetic.py    #   Synthetic test data
├── benchmark.py        # Orchestrator
├── evaluator.py        # LLM judge
├── run_benchmark.py    # CLI
└── tests/              # Unit tests
    └── test_benchmark.py
```

## Running with Live APIs

For production-grade comparisons against real backends, set the appropriate
environment variables before running:

```bash
# Memanto (Moorcheh backend)
export MOORCHEH_API_KEY="your-key"

# Mem0
export MEM0_API_KEY="your-key"

# Zep Cloud
export ZEP_API_KEY="your-key"

# Letta (local server)
# Install and start: pip install letta letta-client && letta server
# The adapter connects to http://localhost:8283 by default

# LLM Judge (for semantic scoring)
export OPENAI_API_KEY="your-key"
python run_benchmark.py --judge-model gpt-4o-mini
```

When API keys are not provided, all adapters fall back to a local in-memory
store with keyword-based retrieval, enabling consistent CI testing and
benchmark validation without external dependencies.

## CI Integration

The benchmark test suite runs as part of the Memanto CI pipeline:

```yaml
# Run benchmark self-test (no API keys needed)
python run_benchmark.py --test
```

This validates:
- All 5 adapters implement the MemorySystem protocol correctly
- All 4 scenarios produce valid, well-formed data
- Scoring computations are mathematically correct
- The full benchmark pipeline executes without errors

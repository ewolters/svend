# Svend Roadmap

## Current State (January 2026)

### Training In Progress
- Language model pretraining: **48% complete** (48K/100K steps)
- Loss: 3.6993, PPL: 40.4
- ETA: ~5 hours remaining

### Infrastructure Complete

| Component | Status | Location |
|-----------|--------|----------|
| **Transformer Architecture** | Done | `src/models/` |
| **Tool System** | Done | `src/tools/` (21 tools) |
| **Training Pipeline** | Done | `src/training/`, `src/pipeline/` |
| **Data Pipeline** | Done | `src/data/` |
| **Evaluation** | Done | `src/evaluation/` |
| **Safety** | Done | `src/safety/` |
| **Server** | Done | `src/server/` |

---

## Tool Architecture

### The Distinction: Tools vs Specialists

**Tools** = Verified operations that eliminate error classes
- Execute deterministic computations
- Return provably correct results OR explicit uncertainty
- Model should NEVER improvise what a tool can compute

**Specialists** = Epistemic scaffolding for domain reasoning
- Encode how experts think about problems
- Provide ontologies, frameworks, and patterns
- Structure the reasoning process, not just compute

---

## Current Tool Inventory (21 tools)

### Tier 1: Core Reasoning (Verified Computation)

| Tool | Purpose | Operations |
|------|---------|------------|
| `calculator` | Exact arithmetic | expressions, arbitrary precision, fractions |
| `execute_python` | Sandboxed code | safe execution with restricted imports |
| `symbolic_math` | SymPy algebra | simplify, solve, differentiate, integrate |
| `logic_solver` | Z3 SAT/SMT | satisfiability, proofs, constraints |
| `unit_converter` | Dimensional analysis | 170+ units, 16 categories |

### Tier 2: Domain Computation

| Tool | Purpose | Operations |
|------|---------|------------|
| `combinatorics` | Counting | P(n,r), C(n,r), Stirling, Bell, Catalan, partitions |
| `graph` | Graph algorithms | BFS, DFS, Dijkstra, MST, cycles, topological sort |
| `geometry` | Coordinate geometry | points, lines, circles, triangles, polygons |
| `sequence` | Pattern recognition | arithmetic, geometric, polynomial, Fibonacci |
| `finance` | Financial math | TVM, NPV/IRR, bonds, amortization |
| `numerical` | NumPy/SciPy | linear algebra, optimization, ODEs |
| `statistics` | Statistical analysis | distributions, tests, regression |
| `chemistry` | Chemistry | molecular weight, stoichiometry, pH |
| `physics` | Physics | kinematics, thermodynamics, circuits |
| `plotter` | Visualization | function plots, scatter, histograms |

### Tier 3: Epistemic Honesty (V0.2 - Error Class Elimination)

These tools delete entire categories of bullshit:

| Tool | Error Class Eliminated | Key Output |
|------|----------------------|------------|
| `state_machine` | "What happens next" hallucinations | Simulated state traces |
| `constraint` | "Here's a plan" for impossible plans | SAT/UNSAT + conflicts |
| `enumerate` | "Seems unlikely" without checking | VERIFIED: ALL/NONE/count |
| `counterfactual` | Single-story thinking | ROBUST to X, SENSITIVE to Y |

### Tier 4: External Data (API-dependent)

| Tool | Purpose | Status |
|------|---------|--------|
| `wolfram` | WolframAlpha queries | Stub (needs API key) |
| `pubchem` | Chemical compound data | Stub |
| `web_search` | Current information | Stub |

### Tier 5: Utilities

| Tool | Purpose |
|------|---------|
| `latex_render` | Mathematical typesetting |

---

## Tooling Roadmap

### Phase 1: Complete (V0.1 + V0.2)
- [x] Core reasoning tools
- [x] Domain computation tools
- [x] Epistemic honesty tools (state_machine, constraint, enumerate, counterfactual)
- [x] ToolStatus epistemic values (CANNOT_VERIFY, UNCERTAIN, PARTIAL)

### Phase 2: Semantic Type System (V0.3)
Extend unit_converter with semantic dimensional analysis:

```
Quantity Types:
- PROBABILITY vs ODDS vs LIKELIHOOD
- RATE vs TOTAL vs RATIO
- MONEY vs MONEY_PER_TIME vs MONEY_PER_UNIT
- DURATION vs TIMESTAMP vs FREQUENCY

Operations:
- Type-check calculations
- Refuse invalid operations (probability + odds)
- Convert between related types
```

### Phase 3: Domain Specialists (V1.0)
Epistemic scaffolding for professional domains:

| Specialist | What It Encodes |
|------------|-----------------|
| **Systems Engineering** | Requirements decomposition, interface matrices, V-model, FMEA |
| **Manufacturing** | Process capability (Cp/Cpk), tolerance stackup, OEE, lean metrics |
| **Controls** | Transfer functions, stability margins, Bode/Nyquist, PID tuning |
| **Reliability** | Weibull analysis, MTBF, fault trees, redundancy |
| **Project Management** | Critical path, resource leveling, earned value |

Each specialist provides:
1. **Ontology**: Domain vocabulary and relationships
2. **Templates**: Standard analyses and frameworks
3. **Validation**: "Is this design sane?" checks
4. **Patterns**: How experts approach problems

### Phase 4: Meta-Reasoning (V1.1)
- **Epistemic Tracking**: GIVEN/DERIVED/VERIFIED/ASSUMED tags
- **Confidence Propagation**: Track how uncertainty flows through reasoning
- **Refusal as First-Class**: Deterministic "Cannot verify. Here's why."

---

## Error Classes and Their Tools

The GPT framework for what we're building:

| Error Class | Tool Solution | Status |
|-------------|---------------|--------|
| Floating point embarrassment | `calculator` | Done |
| Symbolic algebra mistakes | `symbolic_math` | Done |
| Logic errors | `logic_solver` | Done |
| State/temporal confusion | `state_machine` | Done |
| Impossible plan proposals | `constraint` | Done |
| "Seems unlikely" without proof | `enumerate` | Done |
| Single-story thinking | `counterfactual` | Done |
| Dimensional nonsense | `unit_converter` + V0.3 | Partial |
| Confident about assumptions | Epistemic tracking | V1.1 |
| Rhetorical certainty | Refusal as outcome | V1.1 |

---

## Training Plan

### Phase 1: Language Model Pretraining (CURRENT)
- 374M parameters, 100K steps
- Learning basic language modeling
- Status: ~50% complete, PPL ~39

### Phase 2: Reasoning Specialist Training
**Scale Options (all fit A100 80GB):**
| Model | Params | Training Memory | Notes |
|-------|--------|-----------------|-------|
| reasoning-500m | 336M | 9 GB | Fast iteration |
| reasoning-1b | 803M | 16 GB | Quick experiments |
| **reasoning-1.5b** | 1.37B | 27 GB | Sweet spot |
| **reasoning-2b** | 1.73B | 33 GB | Recommended |
| reasoning-3b | 2.37B | 45 GB | Maximum capability |

**Training Data Generation:**
```bash
python scripts/generate_tool_data.py --num-examples 15000 --output data/tool_traces.jsonl
```
Distribution: ~15K examples across 16 domains including V0.2 epistemic tools

**Fine-tuning Process:**
1. Generate synthetic tool-calling data (15K examples)
2. Fine-tune reasoning specialist on tool traces
3. Evaluate on GSM8K, tool correctness, epistemic honesty

### Phase 3: Verifier Training
- 250M-500M parameter verifier model
- Trained on (question, reasoning, correctness) tuples
- Used for self-verification during inference

### Phase 4: Specialist Integration
1. Add domain specialist prompts (systems engineering, manufacturing, etc.)
2. Fine-tune on domain-specific reasoning
3. Evaluate on professional benchmarks

---

## Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| GSM8K accuracy | >50% | - |
| Tool call correctness | >90% | - |
| Safety refusal correctness | >95% | - |
| Latency p90 | <5s | - |
| Epistemic honesty (new) | Model uses CANNOT_VERIFY appropriately | - |

---

## File Structure

```
src/tools/
├── __init__.py           # Registry factory
├── registry.py           # Tool, ToolResult, ToolStatus
├── executor.py           # Execution management
├── orchestrator.py       # Multi-step reasoning
│
├── # Tier 1: Core
├── calculator.py
├── code_sandbox.py
├── math_engine.py        # SymPy + Z3
├── unit_converter.py
│
├── # Tier 2: Domain
├── graph_tools.py        # combinatorics + graph
├── geometry.py
├── sequence_analyzer.py
├── finance.py
├── numerical.py
├── statistics_tool.py
├── chemistry.py
├── physics.py
├── plotter.py
│
├── # Tier 3: Epistemic
├── state_machine.py
├── constraint_solver.py
├── enumerator.py
├── counterfactual.py
│
├── # Tier 4: External
├── external_apis.py
│
└── # Tier 5: Utilities
    └── latex_render.py
```

---

## Hardware & Resources

| Resource | Spec | Usage |
|----------|------|-------|
| GPU | A100 80GB | Training |
| Storage | 300GB | Checkpoints |
| Budget | $500/30 days | Compute costs |

## Notes

- Landing page: svend.ai
- Target launch: May 2026
- Core principle: **Make bullshit structurally impossible, domain by domain**

## Quick Commands

```bash
# Generate training data
python scripts/generate_tool_data.py --num-examples 15000 --output data/tool_traces.jsonl

# Test V0.2 epistemic tools
python scripts/test_v02_tools.py

# Test V0.1 domain tools
python scripts/test_new_tools.py

# View model configs
python -c "from src.models.config import print_config_comparison; print_config_comparison()"
```

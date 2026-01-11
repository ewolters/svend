# Svend Production Architecture & Training Plan

**Budget**: $500 over 30 days
**Hardware**: A100 80GB (Colab Pro+, $10/month)
**Domain**: svend.ai
**Goal**: Production reasoning system with safety, tool use, and verification

---

## System Architecture

Svend is a **multi-model reasoning system**, not a single monolith. Here's the production architecture:

```
                                    +------------------+
                                    |   Safety Gate    |
                                    |   (Classifier)   |
                                    +--------+---------+
                                             |
                                             v
+------------+     +-------------------+     +------------------+
|   User     | --> |   API Gateway     | --> |  Input Router    |
|   Query    |     |   (FastAPI)       |     |  (Complexity)    |
+------------+     +-------------------+     +--------+---------+
                                                      |
                           +-------------+------------+-------------+
                           |             |                          |
                           v             v                          v
                   +-------+-------+  +--+---------------+  +-------+-------+
                   | Quick Path    |  | Standard Path   |  | Deep Path     |
                   | (Cache/Simple)|  | (13B Reasoner)  |  | (13B + Search)|
                   +---------------+  +--------+--------+  +-------+-------+
                                               |                    |
                                               v                    v
                                      +--------+--------+  +--------+--------+
                                      |  Tool Engine    |  |  Tree Search    |
                                      |  (Orchestrator) |  |  (Multi-path)   |
                                      +--------+--------+  +--------+--------+
                                               |                    |
                                               v                    v
                                      +--------+--------+  +--------+--------+
                                      |    Verifier     |  |    Verifier     |
                                      |    (3B Model)   |  |    (3B Model)   |
                                      +--------+--------+  +--------+--------+
                                               |                    |
                                               v                    v
                                      +--------+--------+  +--------+--------+
                                      |  Safety Check   |  |  Safety Check   |
                                      |    (Output)     |  |    (Output)     |
                                      +-----------------+  +-----------------+
```

---

## Component Breakdown

### 1. **Reasoning Core** (13B)
- Primary model for multi-step reasoning
- Tool calling capabilities (math, code, logic, chemistry, physics)
- 8K context (extendable to 32K)
- **Training**: Fine-tune from Qwen2.5-14B or Mistral-7B-v0.3 (better base than from-scratch)

### 2. **Verifier** (3B)
- Checks reasoning chains for logical errors
- Validates tool call results
- Provides confidence scores
- **Training**: Trained on (reasoning_chain, is_correct) pairs

### 3. **Safety Classifier** (Separate)
- Input classification: safe / needs_review / refuse
- Output classification: safe / needs_editing / block
- Lightweight (DistilBERT-based or small transformer)
- **Training**: Mix of standard safety datasets + domain-specific red-teaming

### 4. **Tool Engine**
Tools already implemented:
- `execute_python` - Sandboxed Python (code_sandbox.py)
- `symbolic_math` - SymPy algebra/calculus (math_engine.py)
- `logic_solver` - Z3 SAT/SMT (math_engine.py)

Tools to add:
- `chemistry` - RDKit molecular + reaction tools
- `physics` - Pint units + physics calculations
- `unit_convert` - Physical unit conversions
- `plot` - Matplotlib visualization (return base64)

---

## Training Strategy

### Phase 0: Base Model Selection (Day 1)
**Decision**: Fine-tune vs. train from scratch

| Approach | Pros | Cons |
|----------|------|------|
| **Fine-tune Qwen2.5-14B** | Proven foundation, faster, better initial capabilities | Less control, may inherit biases |
| **Fine-tune Mistral-7B-v0.3** | Excellent reasoning base, Apache 2.0, easier to serve | Smaller capacity |
| **Train from scratch (13B)** | Full control, no inherited behaviors | Much more compute, riskier |

**Recommendation**: Fine-tune Qwen2.5-14B-Instruct as reasoner, train 3B verifier from scratch, use DistilBERT for safety.

### Phase 1: Infrastructure Validation (Days 1-2)
**Goal**: Verify everything works before burning compute.

1. Run `scripts/test_pipeline.py` locally (6GB GPU or CPU)
2. Test Colab A100 connection and Drive mounting
3. Verify tool execution works (math, code, logic)
4. Set up WandB project for experiment tracking

**Estimated time**: 2-4 hours

### Phase 2: Safety Classifier (Days 2-4)
**Goal**: Build the safety layer FIRST (critical for production).

**Architecture**:
```python
# Lightweight binary + multi-class classifier
SafetyClassifier:
    - Encoder: DistilBERT or MiniLM (efficient)
    - Input head: [safe, review, refuse]
    - Output head: [safe, edit, block]
    - Harm category: [violence, illegal, self-harm, ...]
```

**Training Data**:
- Anthropic HH-RLHF (harmless/helpful splits)
- OpenAI Moderation dataset
- WildChat toxicity annotations
- Custom red-team examples (chemistry/physics specific)

**Training**: ~2-4 hours on A100

### Phase 3: Tool Extensions (Days 3-5)
**Goal**: Add chemistry and physics tools.

**Chemistry Tools** (RDKit):
```python
- molecular_structure(smiles) -> properties, 2D/3D coords
- reaction_balance(equation) -> balanced equation
- stoichiometry(reaction, amounts) -> product amounts
- functional_groups(smiles) -> identified groups
- molecular_weight(formula) -> MW
```

**Physics Tools** (Pint + SciPy):
```python
- unit_convert(value, from_unit, to_unit) -> converted
- kinematics(known_vars) -> solved vars
- thermodynamics(system, process) -> results
- circuits(components, topology) -> analysis
```

**Implementation**: Extend `src/tools/` with new modules

### Phase 4: Synthetic Data Generation (Days 5-8)
**Goal**: Generate high-quality reasoning traces for all domains.

**Data Mix**:
| Category | Count | Source |
|----------|-------|--------|
| Math reasoning | 20K | OpenMathInstruct-2 + synthetic |
| Logic problems | 10K | Synthetic (Claude) |
| Code reasoning | 10K | CodeAlpaca + synthetic |
| Chemistry | 5K | Synthetic (Claude) with tool calls |
| Physics | 5K | Synthetic (Claude) with tool calls |
| General reasoning | 10K | SlimOrca + CAMEL |
| **Total** | 60K | |

**Tool-augmented format**:
```json
{
  "instruction": "What is the pH of a 0.01M HCl solution?",
  "reasoning": [
    {"step": 1, "thought": "HCl is a strong acid, so it dissociates completely"},
    {"step": 2, "thought": "For strong acids, [H+] = concentration of acid"},
    {"step": 3, "tool_call": {"name": "chemistry", "args": {"op": "ph", "concentration": 0.01, "acid_type": "strong"}}},
    {"step": 4, "tool_result": {"pH": 2.0, "pOH": 12.0}},
    {"step": 5, "thought": "The pH is 2.0, which makes sense for a moderately dilute strong acid"}
  ],
  "answer": "The pH of 0.01M HCl is 2.0"
}
```

**Budget**: ~$100-150 in API calls (Claude Sonnet)

### Phase 5: Reasoner Training (Days 8-15)
**Goal**: Train the main 13B reasoning model.

**Stage 5a: Base Reasoning (No Tools)**
- Model: Qwen2.5-14B-Instruct (or 13B from scratch)
- Data: 40K examples (math, logic, code, general)
- Duration: ~24-36 hours on A100
- Learning rate: 2e-5 (fine-tune) or 1e-4 (from scratch)
- Effective batch: 32-64

**Stage 5b: Tool Integration**
- Resume from 5a checkpoint
- Data: 20K tool-augmented examples
- Duration: ~12-18 hours
- Focus: Learning tool calling patterns

### Phase 6: Verifier Training (Days 12-16)
**Goal**: Train 3B model to verify reasoning chains.

**Training Data Generation**:
1. Generate reasoning chains with the trained reasoner
2. Label chains as correct/incorrect (automated + manual)
3. Include failure modes: math errors, logic gaps, wrong tool usage

**Model**: 3B from scratch (your existing config)
**Duration**: ~18-24 hours

### Phase 7: Integration & Evaluation (Days 16-22)
**Goal**: Wire everything together and benchmark.

**Benchmarks**:
- GSM8K (target: >60% accuracy)
- MATH (target: >25% accuracy)
- HumanEval (target: >40% pass@1)
- Custom tool-use eval (target: >85% correct tool selection)
- Safety eval (target: >95% correct refusal on harmful prompts)

**Integration Tests**:
- Full pipeline: query -> safety -> reasoning -> tools -> verify -> safety -> response
- Latency testing (target: <5s for standard queries)
- Load testing (target: 10 concurrent requests)

### Phase 8: Deployment Prep (Days 22-28)
**Goal**: Production-ready deployment.

**Hosting Options**:
| Provider | Setup | Cost Estimate |
|----------|-------|---------------|
| RunPod | Serverless A100/A10 | ~$0.50-2/hr active |
| Modal | Pay-per-second | ~$1-3/hr active |
| Lambda Labs | Reserved A100 | ~$1.10/hr |
| Replicate | Managed inference | Per-prediction |

**Recommendation**: Start with Modal (auto-scaling, pay-per-second) then move to RunPod for steady traffic.

**Deployment Stack**:
- FastAPI server (already built)
- Model loading with vLLM for inference speed
- Redis for caching frequent queries
- Prometheus + Grafana for monitoring

### Phase 9: Launch & Iterate (Days 28-30)
**Goal**: Soft launch on svend.ai.

- Deploy to production
- Set up error monitoring
- Enable feedback collection
- Plan iteration cycle

---

## Budget Breakdown

| Item | Cost |
|------|------|
| Colab Pro+ (A100 access) | $50/month |
| Synthetic data (Claude API) | $100-150 |
| RunPod/Modal testing | $50-100 |
| Domain + hosting | $50 |
| Buffer/contingency | $100-150 |
| **Total** | ~$400-500 |

---

## Colab Training Sessions

A100 80GB on Colab gives you ~12-24 hour sessions. Plan your training to checkpoint frequently.

**Session 1** (4h): Safety classifier training
**Session 2** (24h): Reasoner Stage 5a (base reasoning)
**Session 3** (18h): Reasoner Stage 5b (tool integration)
**Session 4** (24h): Verifier training
**Session 5** (8h): Evaluation and benchmarking

**Critical**: Mount Google Drive and checkpoint every 500-1000 steps.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Colab session timeout | Checkpoint every 500 steps to Drive |
| Model doesn't learn tools | More synthetic tool data, curriculum learning |
| Safety gaps | Red-team before launch, start with conservative refusals |
| Latency too high | Quantize to int8, use vLLM, add caching |
| Budget overrun | Start with Mistral-7B (cheaper), scale up if working |

---

## Files to Create/Modify

### New Files Needed:
```
src/safety/
    __init__.py
    classifier.py        # Safety classifier model
    filters.py           # Rule-based safety filters
    training.py          # Safety model training

src/tools/
    chemistry.py         # RDKit chemistry tools
    physics.py           # Physics calculations

scripts/
    train_safety.py      # Safety classifier training
    generate_science_data.py  # Chemistry/physics synthetic data
    deploy.py            # Deployment automation

notebooks/
    train_reasoner.ipynb # Colab notebook for reasoner
    train_verifier.ipynb # Colab notebook for verifier
    train_safety.ipynb   # Colab notebook for safety
```

### Modifications:
- `src/models/config.py` - Add safety classifier config
- `src/tools/registry.py` - Register new tools
- `requirements.txt` - Add RDKit, pint, etc.

---

## Success Criteria (Before Production)

- [ ] Safety classifier >95% accuracy on eval set
- [ ] GSM8K accuracy >50%
- [ ] Tool calls execute correctly >90%
- [ ] Verifier catches obvious errors >80%
- [ ] End-to-end latency <5s for 90th percentile
- [ ] Red-team: refuses harmful chemistry/physics requests
- [ ] Memory fits on A100 40GB for inference (enables cheaper hosting)

---

## Next Immediate Steps

1. **Today**: Run `test_pipeline.py` to validate existing infrastructure
2. **Today**: Set up WandB project "svend-reasoning"
3. **Tomorrow**: Start building safety classifier (`src/safety/`)
4. **Day 3**: Add chemistry/physics tools
5. **Day 4-5**: Generate synthetic data
6. **Day 6+**: Begin training runs

---

## Decision Points Needed

1. **Base model**: Qwen2.5-14B-Instruct vs Mistral-7B vs train from scratch?
2. **Safety architecture**: Single classifier vs input/output pair?
3. **Chemistry scope**: Organic focus? Inorganic? Biochemistry?
4. **Hosting priority**: Cost optimization vs latency optimization?

Let me know your preferences and we can refine the plan.

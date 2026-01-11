# Tool-Trace Generation Plan

## Overview

Generate synthetic training data that teaches the model to call tools during reasoning. Uses Claude API to create high-quality examples with proper tool-call formatting.

---

## Target Output

Each example is a JSON object:

```json
{
  "question": "What is the derivative of x^3 + 2x at x=5?",
  "domain": "math",
  "reasoning": [
    {
      "step": 1,
      "content": "I need to find d/dx of x^3 + 2x, then evaluate at x=5."
    },
    {
      "step": 2,
      "content": "Let me use symbolic math to compute the derivative.",
      "tool_call": {
        "name": "symbolic_math",
        "args": {"operation": "differentiate", "expression": "x**3 + 2*x", "variable": "x"}
      }
    },
    {
      "step": 3,
      "content": "The derivative is 3x² + 2. Now I'll evaluate at x=5.",
      "tool_result": "3*x**2 + 2"
    },
    {
      "step": 4,
      "content": "Substituting x=5: 3(25) + 2 = 77.",
      "tool_call": {
        "name": "symbolic_math",
        "args": {"operation": "evaluate", "expression": "3*x**2 + 2", "substitutions": {"x": 5}}
      }
    },
    {
      "step": 5,
      "content": "Confirmed: the derivative at x=5 is 77.",
      "tool_result": "77"
    }
  ],
  "answer": "77",
  "tools_used": ["symbolic_math"]
}
```

---

## Tool Definitions

### 1. symbolic_math (SymPy)
```python
operations = [
    "simplify",        # simplify expressions
    "solve",           # solve equations
    "differentiate",   # d/dx
    "integrate",       # ∫ (definite and indefinite)
    "evaluate",        # substitute values
    "limit",           # compute limits (including at infinity)
    "series",          # Taylor/Maclaurin series
    "factor",          # factor polynomials
    "expand",          # expand polynomials
    "matrix",          # determinant, inverse, eigenvalues, rref, multiply
]

# Example: Compute limit
{
    "name": "symbolic_math",
    "args": {"operation": "limit", "expression": "sin(x)/x", "point": "0"}
}

# Example: Matrix determinant
{
    "name": "symbolic_math",
    "args": {"operation": "matrix", "matrix": "[[1,2],[3,4]]", "values": "determinant"}
}
```

### 2. execute_python
```python
# For numerical computation, verification, or complex logic
# Sandboxed execution with math, statistics, itertools, etc.
{
    "name": "execute_python",
    "args": {
        "code": "import math\nresult = sum(1/math.factorial(n) for n in range(10))\nprint(f'e ≈ {result}')"
    }
}
```

### 3. logic_solver (Z3)
```python
operations = [
    "check_sat",          # is formula satisfiable?
    "prove",              # prove theorem (shows unsatisfiability of negation)
]

# Example: Prove logical statement
{
    "name": "logic_solver",
    "args": {
        "operation": "prove",
        "constraints": "[\"x > 0\", \"y > 0\"]",
        "variables": "{\"x\": \"real\", \"y\": \"real\"}",
        "conclusion": "x + y > 0"
    }
}
```

### 4. chemistry
```python
operations = [
    "molecular_weight",    # compute molar mass from formula
    "parse_formula",       # parse formula to element counts
    "balance_equation",    # balance chemical equations
    "stoichiometry",       # mole ratios and yields
    "ph",                  # pH from concentration (strong/weak acids/bases with Ka/Kb)
    "molar_conversion",    # grams ↔ moles ↔ particles
    "dilution",            # C1V1 = C2V2
    "concentration",       # molarity and molality
    "percent_composition", # percent by mass of each element
    "limiting_reagent",    # find limiting reagent and theoretical yield
]

# Example: Dilution calculation
{
    "name": "chemistry",
    "args": {"operation": "dilution", "C1": 2.0, "V1": 50, "C2": 0.5}
}

# Example: Weak acid pH
{
    "name": "chemistry",
    "args": {"operation": "ph", "concentration": 0.1, "type": "acid", "strong": false, "Ka": 1.8e-5}
}
```

### 5. physics
```python
operations = [
    "unit_convert",        # convert between physical units
    "kinematics",          # v = v0 + at, x = v0t + ½at², etc.
    "ideal_gas",           # PV = nRT
    "waves",               # v = fλ, E = hf, Doppler
    "constant",            # get physical constants (c, G, h, e, k_B, etc.)
    "energy",              # KE = ½mv², PE = mgh, W = Fd, P = W/t
    "electricity",         # Ohm's law, power, series/parallel, Coulomb's law
    "optics",              # thin lens equation, magnification, Snell's law
    "projectile",          # range, max height, time of flight
    "shm",                 # simple harmonic motion (spring, pendulum)
]

# Example: Electric circuit
{
    "name": "physics",
    "args": {"operation": "electricity", "operation": "ohms_law", "params": {"V": 12, "R": 4}}
}

# Example: Projectile motion
{
    "name": "physics",
    "args": {"operation": "projectile", "v0": 20, "theta": 45}
}

# Example: Optics
{
    "name": "physics",
    "args": {"operation": "optics", "operation": "thin_lens", "params": {"f": 0.1, "do": 0.3}}
}
```

---

## Generation Categories

| Category | Tool Focus | Example Count | Estimated Cost |
|----------|------------|---------------|----------------|
| Calculus | symbolic_math (differentiate, integrate, limit, series) | 2,000 | $8 |
| Algebra | symbolic_math (solve, factor, expand, simplify) | 1,500 | $6 |
| Linear Algebra | symbolic_math (matrix operations) | 500 | $2 |
| Physics - Mechanics | physics (kinematics, energy, projectile, shm) | 1,500 | $8 |
| Physics - E&M/Optics | physics (electricity, optics, waves) | 1,000 | $5 |
| Chemistry - Fundamentals | chemistry (molecular_weight, stoichiometry, balancing) | 1,000 | $5 |
| Chemistry - Solutions | chemistry (ph, dilution, concentration) | 1,000 | $5 |
| Logic/proofs | logic_solver | 1,000 | $5 |
| Numerical/code | execute_python | 1,500 | $8 |
| Multi-tool | mixed (problems requiring 2+ tools) | 1,000 | $8 |
| **Total** | | **12,000** | **~$60** |

---

## Generation Prompts

### System Prompt for Claude

```
You are generating training data for a reasoning model that uses tools.

Given a problem, generate a detailed reasoning trace that:
1. Shows step-by-step thinking
2. Calls tools when computation is needed
3. Incorporates tool results into reasoning
4. Arrives at a final answer

Format tool calls as:
{"name": "tool_name", "args": {...}}

The model should learn WHEN to call tools (non-trivial computation) and when to reason directly (simple steps, conceptual understanding).

Available tools:
- symbolic_math: differentiate, integrate, solve, simplify, expand, factor, evaluate, limit, series
- execute_python: run arbitrary Python code
- logic_solver: satisfiable, prove, solve_constraints
- chemistry_calc: balance_equation, molar_mass, stoichiometry, dilution, ideal_gas
- physics_calc: kinematics, energy, force, electricity, optics
```

### Example Generation Prompt

```
Generate a training example for this problem:

Problem: {problem}
Domain: {domain}
Difficulty: {difficulty}
Required tools: {tools}

Output JSON with: question, domain, reasoning (array of steps with optional tool_call/tool_result), answer, tools_used
```

---

## Implementation Steps

### Step 1: Problem Generation
- Source problems from existing datasets (OpenMathInstruct, CAMEL, etc.)
- Filter for problems that benefit from tool use
- Add synthetic problems for underrepresented tool types

### Step 2: Trace Generation (Claude API)
- Batch problems by domain
- Generate traces with Claude
- Validate JSON format
- Verify tool calls are syntactically correct

### Step 3: Tool Execution & Validation
- Actually execute the tool calls
- Compare tool results with expected results in trace
- Flag/fix discrepancies

### Step 4: Quality Filtering
- Remove traces with invalid tool calls
- Remove traces where tool results don't match
- Remove traces that are too short or don't show reasoning

### Step 5: Format for Training
- Convert to training format (instruction/output pairs)
- Add tool call special tokens
- Split train/val

---

## Script Structure

```
scripts/generate_tool_data.py
├── problem_sources/
│   ├── math_problems()      # Sample from OpenMathInstruct
│   ├── physics_problems()   # Sample from CAMEL physics
│   ├── chemistry_problems() # Sample from CAMEL chemistry
│   └── synthetic_problems() # Generate new problems
├── generation/
│   ├── generate_trace()     # Call Claude API
│   ├── validate_trace()     # Check JSON, tool calls
│   └── execute_tools()      # Run tools, verify results
├── output/
│   ├── save_jsonl()         # Save to data/tool_traces.jsonl
│   └── generate_report()    # Stats on generation
└── main()
    ├── --num-examples 12000
    ├── --domain [math|physics|chemistry|all]
    ├── --output data/tool_traces.jsonl
    └── --api-key $ANTHROPIC_API_KEY
```

---

## Cost Optimization

1. **Batch similar problems** - reduces context overhead
2. **Use Haiku for validation** - cheaper for JSON checking
3. **Cache tool executions** - many problems share similar tool calls
4. **Generate in stages** - start with 1K, evaluate quality, then scale

---

## Quality Metrics

Track during generation:
- Valid JSON rate
- Tool call syntax correctness
- Tool result match rate
- Average reasoning steps
- Tool calls per example
- Domain distribution

---

## Timeline

| Task | Estimated Time |
|------|----------------|
| Script skeleton | 1 hour |
| Problem sourcing | 1 hour |
| Claude API integration | 1 hour |
| Tool execution/validation | 2 hours |
| Generate 1K pilot batch | 2 hours |
| Review & iterate | 1 hour |
| Generate full 12K | 4 hours |
| **Total** | **~12 hours** |

---

## Next Steps

1. Create `scripts/generate_tool_data.py` skeleton
2. Implement problem sourcing from existing datasets
3. Set up Claude API client with retry logic
4. Implement tool execution for validation
5. Run pilot batch of 100 examples
6. Review quality, adjust prompts
7. Scale to full generation

---

*Last updated: January 2025*

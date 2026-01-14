# Svend V0 Tool Specification

## Overview

This document defines the complete toolkit for Svend V0 (3B friends & family) through V1 (18-20B production).

The model learns to call tools by generating structured tokens. Tools do the heavy lifting - verified computation, external data, visualization - while the model handles:
1. Problem understanding
2. Tool selection
3. Call formatting
4. Result interpretation

## Token Format

```
<|tool_call|>
<|tool_name|>tool_name<|tool_args|>{"arg1": "value1", "arg2": "value2"}
<|/tool_call|>

<|tool_result|>
result data here
<|/tool_result|>
```

## V0 Toolkit (Locked for Training)

### Tier 1: Core Reasoning (Must Have)

| Tool | Purpose | Status |
|------|---------|--------|
| `calculator` | Verified arithmetic, no floating point errors | **NEW** |
| `execute_python` | Sandboxed code execution | EXISTS |
| `symbolic_math` | SymPy - algebra, calculus, equation solving | EXISTS |
| `logic_solver` | Z3 - SAT/SMT, constraints, proofs | EXISTS |
| `unit_converter` | Dimensional analysis, all unit types | **NEW** |

### Tier 2: Domain Specialists (High Value)

| Tool | Purpose | Status |
|------|---------|--------|
| `chemistry` | Molecular weight, stoichiometry, balancing, pH | EXISTS |
| `physics` | Kinematics, thermodynamics, circuits, optics | EXISTS |
| `numerical` | NumPy/SciPy - linear algebra, optimization, ODEs | **NEW** |
| `plotter` | Matplotlib - function plots, data visualization | **NEW** |
| `statistics` | Distributions, hypothesis tests, regression | **NEW** |

### Tier 3: External Data (Production)

| Tool | Purpose | Status |
|------|---------|--------|
| `wolfram` | WolframAlpha API - verified facts, data | **NEW** |
| `web_search` | Current information, fact checking | **NEW** |
| `pubchem` | Chemical properties, structures, safety | **NEW** |

### Tier 4: Utilities

| Tool | Purpose | Status |
|------|---------|--------|
| `latex_render` | Format mathematical output | **NEW** |
| `code_test` | Run tests on generated code | EXISTS (in sandbox) |

---

## Tool Specifications

### 1. Calculator (NEW)

**Purpose**: Guaranteed-correct arithmetic. Avoids floating point embarrassments.

```python
Tool: calculator
Description: Perform exact arithmetic calculations. Use this for any numerical computation to avoid floating point errors.

Operations:
- basic: +, -, *, /, //, %, **
- functions: sqrt, abs, factorial, gcd, lcm
- precision: Set decimal places for results

Parameters:
- expression (string, required): Mathematical expression
- precision (int, optional): Decimal places (default: 10)

Examples:
<|tool_call|><|tool_name|>calculator<|tool_args|>{"expression": "7 * 8"}<|/tool_call|>
<|tool_result|>56<|/tool_result|>

<|tool_call|><|tool_name|>calculator<|tool_args|>{"expression": "sqrt(2)", "precision": 15}<|/tool_call|>
<|tool_result|>1.414213562373095<|/tool_result|>
```

### 2. Unit Converter (NEW - Standalone)

**Purpose**: Comprehensive unit conversion with dimensional analysis verification.

```python
Tool: unit_converter
Description: Convert between units with dimensional analysis. Catches unit mismatches.

Categories:
- length: m, km, cm, mm, mi, ft, in, yd, nm, au, ly
- mass: kg, g, mg, lb, oz, ton
- time: s, ms, min, hr, day, yr
- temperature: K, C, F
- force: N, kN, dyn, lbf
- energy: J, kJ, cal, kcal, eV, kWh, BTU
- power: W, kW, MW, hp
- pressure: Pa, kPa, bar, atm, psi, mmHg
- speed: m/s, km/h, mph, kn
- volume: L, mL, m³, gal, qt, pt
- area: m², km², ha, acre, ft²
- angle: rad, deg, rev

Parameters:
- value (number, required): Value to convert
- from_unit (string, required): Source unit
- to_unit (string, required): Target unit

Example:
<|tool_call|><|tool_name|>unit_converter<|tool_args|>{"value": 100, "from_unit": "km/h", "to_unit": "m/s"}<|/tool_call|>
<|tool_result|>{"value": 27.778, "unit": "m/s"}<|/tool_result|>
```

### 3. Numerical (NEW)

**Purpose**: NumPy/SciPy for when symbolic math isn't enough.

```python
Tool: numerical
Description: Numerical computation using NumPy/SciPy. Use for linear algebra, optimization, differential equations, interpolation.

Operations:
- matrix: determinant, inverse, eigenvalues, SVD, solve_linear
- optimize: minimize, root_find, curve_fit
- ode: solve initial value problems
- interpolate: 1D/2D interpolation
- fft: Fourier transform
- integrate_numeric: Numerical integration (quad, trapz)
- stats: distributions, sampling

Parameters:
- operation (string, required): Operation type
- data (object, required): Input data (matrices, functions, etc.)
- options (object, optional): Solver options

Example - Solve linear system:
<|tool_call|><|tool_name|>numerical<|tool_args|>{
  "operation": "solve_linear",
  "data": {
    "A": [[3, 1], [1, 2]],
    "b": [9, 8]
  }
}<|/tool_call|>
<|tool_result|>{"x": [2.0, 3.0], "method": "LU decomposition"}<|/tool_result|>

Example - Find root:
<|tool_call|><|tool_name|>numerical<|tool_args|>{
  "operation": "root_find",
  "data": {"function": "x**3 - x - 2", "bracket": [1, 2]}
}<|/tool_call|>
<|tool_result|>{"root": 1.5214, "converged": true}<|/tool_result|>
```

### 4. Plotter (NEW)

**Purpose**: Generate visualizations for functions and data.

```python
Tool: plotter
Description: Create plots and visualizations. Returns base64-encoded image or plot description.

Plot Types:
- function: Plot mathematical functions
- scatter: Scatter plot of data points
- line: Line plot
- bar: Bar chart
- histogram: Distribution histogram
- contour: 2D contour plot
- vector_field: Vector field visualization

Parameters:
- plot_type (string, required): Type of plot
- data (object, required): Data or function to plot
- options (object, optional): Styling, labels, range

Example - Plot function:
<|tool_call|><|tool_name|>plotter<|tool_args|>{
  "plot_type": "function",
  "data": {"functions": ["sin(x)", "cos(x)"], "x_range": [-3.14, 3.14]},
  "options": {"title": "Trig Functions", "grid": true}
}<|/tool_call|>
<|tool_result|>{"image": "base64...", "description": "Plot of sin(x) and cos(x) from -π to π"}<|/tool_result|>
```

### 5. Statistics (NEW)

**Purpose**: Statistical analysis and hypothesis testing.

```python
Tool: statistics
Description: Statistical calculations, hypothesis tests, and regression analysis.

Operations:
- descriptive: mean, median, std, var, quartiles, skew, kurtosis
- distribution: PDF, CDF, inverse CDF, sampling (normal, t, chi2, f, binomial, poisson)
- hypothesis: t_test, chi_square, anova, mann_whitney
- regression: linear, polynomial, logistic
- correlation: pearson, spearman, kendall
- confidence_interval: For means, proportions

Parameters:
- operation (string, required): Statistical operation
- data (array or object, required): Input data
- options (object, optional): Confidence level, alternative hypothesis, etc.

Example - t-test:
<|tool_call|><|tool_name|>statistics<|tool_args|>{
  "operation": "t_test",
  "data": {
    "sample1": [23, 25, 28, 24, 26],
    "sample2": [30, 32, 29, 31, 33]
  },
  "options": {"alternative": "two-sided", "alpha": 0.05}
}<|/tool_call|>
<|tool_result|>{"t_statistic": -5.12, "p_value": 0.0009, "significant": true, "conclusion": "Reject null hypothesis"}<|/tool_result|>
```

### 6. Wolfram (NEW)

**Purpose**: Verified factual data from WolframAlpha.

```python
Tool: wolfram
Description: Query WolframAlpha for verified factual information, data, and computations.

Query Types:
- compute: Mathematical computations (backup for edge cases)
- data: Factual data (populations, distances, dates, etc.)
- convert: Unit conversions with context
- properties: Physical/chemical properties

Parameters:
- query (string, required): Natural language query
- format (string, optional): "short" | "full" | "steps"

Example:
<|tool_call|><|tool_name|>wolfram<|tool_args|>{"query": "distance from Earth to Mars"}<|/tool_call|>
<|tool_result|>{"value": "225 million km", "note": "average distance, varies from 55M to 400M km"}<|/tool_result|>
```

### 7. Web Search (NEW)

**Purpose**: Current information beyond training cutoff.

```python
Tool: web_search
Description: Search the web for current information. Use for recent data, current events, or fact-checking.

Parameters:
- query (string, required): Search query
- max_results (int, optional): Number of results (default: 5)
- site (string, optional): Restrict to specific site

Example:
<|tool_call|><|tool_name|>web_search<|tool_args|>{"query": "latest JWST discoveries 2025"}<|/tool_call|>
<|tool_result|>{"results": [{"title": "...", "snippet": "...", "url": "..."}]}<|/tool_result|>
```

### 8. PubChem (NEW)

**Purpose**: Chemical data and properties.

```python
Tool: pubchem
Description: Query PubChem database for chemical information.

Operations:
- compound: Get compound by name, formula, or CID
- properties: Physical/chemical properties
- structure: 2D/3D structure, SMILES, InChI
- safety: GHS hazards, handling info
- synonyms: Alternative names

Parameters:
- operation (string, required): Query type
- identifier (string, required): Compound name, formula, or CID
- properties (array, optional): Specific properties to fetch

Example:
<|tool_call|><|tool_name|>pubchem<|tool_args|>{
  "operation": "properties",
  "identifier": "aspirin",
  "properties": ["MolecularWeight", "MeltingPoint", "SMILES"]
}<|/tool_call|>
<|tool_result|>{
  "name": "Aspirin",
  "formula": "C9H8O4",
  "molecular_weight": 180.16,
  "melting_point": "135-136°C",
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"
}<|/tool_result|>
```

---

## Tool Trace Format (For Training Data)

Training data should include complete tool traces showing:
1. Problem analysis
2. Tool selection reasoning
3. Tool call
4. Result interpretation
5. Final answer

### Example Trace

```
User: What's the pH of a 0.01M HCl solution?

<|think|>
This is a chemistry problem about pH calculation. HCl is a strong acid, so it completely dissociates.
For strong acids: [H+] = concentration = 0.01M
pH = -log10([H+])

Let me calculate this using the chemistry tool.
<|/think|>

<|tool_call|>
<|tool_name|>chemistry<|tool_args|>{"operation": "ph", "concentration": 0.01, "type": "acid", "strong": true}
<|/tool_call|>

<|tool_result|>
{"success": true, "pH": 2.0, "pOH": 12.0, "H+_concentration": 0.01}
<|/tool_result|>

<|think|>
The tool confirms pH = 2.0, which matches the expected -log10(0.01) = 2.

Let me verify: 10^(-2) = 0.01, which equals our [H+] concentration. ✓
<|/think|>

<|answer|>
The pH of a 0.01M HCl solution is **2.0**.

Since HCl is a strong acid, it completely dissociates:
- [H+] = 0.01 M
- pH = -log₁₀(0.01) = 2.0
<|/answer|>
```

---

## Implementation Priority

### Phase 1: V0 Training (Now)
1. ✅ execute_python (sandbox)
2. ✅ symbolic_math (SymPy)
3. ✅ logic_solver (Z3)
4. ✅ chemistry
5. ✅ physics
6. 🔨 calculator
7. 🔨 unit_converter (standalone)
8. 🔨 numerical (NumPy/SciPy)
9. 🔨 statistics
10. 🔨 plotter

### Phase 2: V0.5 Enhancement
11. 🔜 wolfram
12. 🔜 pubchem

### Phase 3: V1 Production
13. 🔜 web_search
14. 🔜 latex_render
15. 🔜 Additional domain tools as needed

---

## Notes for Training Data Generation

1. **Tool diversity**: Each problem should use appropriate tools, not always the same one
2. **Multi-tool chains**: Complex problems should demonstrate calling multiple tools
3. **Error handling**: Include examples where tools return errors and model recovers
4. **Verification**: Show model verifying tool results (e.g., differentiating an integral)
5. **Tool selection reasoning**: `<|think|>` blocks should explain WHY a tool is chosen
6. **Norwegian style**: Direct, concise reasoning - no filler

---

## Appendix: Special Tokens

```python
# Reasoning
"<|think|>", "<|/think|>"
"<|answer|>", "<|/answer|>"

# Tool calling
"<|tool_call|>", "<|/tool_call|>"
"<|tool_name|>", "<|tool_args|>"
"<|tool_result|>", "<|/tool_result|>"

# Meta-cognitive
"<|clarification|>", "<|/clarification|>"  # Ask for clarification
"<|cannot_solve|>", "<|/cannot_solve|>"    # Admit limitations

# Per-tool tokens (for faster routing)
"<|tool:calculator|>"
"<|tool:symbolic_math|>"
"<|tool:logic_solver|>"
"<|tool:chemistry|>"
"<|tool:physics|>"
"<|tool:numerical|>"
"<|tool:plotter|>"
"<|tool:statistics|>"
"<|tool:wolfram|>"
"<|tool:web_search|>"
"<|tool:pubchem|>"
```

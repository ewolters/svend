# Svend System Architecture

## Overview

Svend is a multi-model reasoning ensemble deployed as a hosted API service.

```
                                    SVEND SYSTEM ARCHITECTURE

    User Request                                                        Response
         |                                                                  ^
         v                                                                  |
    +----------+     +-------------+     +------------------+     +---------+
    |   API    |---->|   Router    |---->|   Specialists    |---->| Output  |
    | Gateway  |     |   (125M)    |     |                  |     | Format  |
    +----------+     +-------------+     +------------------+     +---------+
         |                |                      |                      |
         v                v                      v                      v
    +----------+     +----------+     +-------------------+     +----------+
    |  Auth    |     | Language |     |    Reasoning      |     | Verifier |
    |  Layer   |     |  (336M)  |     |     (336M)        |     |  (191M)  |
    +----------+     +----------+     +-------------------+     +----------+
                                              |
                                              v
                                      +---------------+
                                      |    Tools      |
                                      +---------------+
                                      | - SymPy       |
                                      | - Z3 Solver   |
                                      | - Python      |
                                      | - Physics     |
                                      | - Chemistry   |
                                      +---------------+
```

## Components

### 1. API Gateway
- **Technology**: FastAPI
- **Port**: 8000 (internal), 443 (external via reverse proxy)
- **Endpoints**:
  - `POST /v1/chat/completions` - OpenAI-compatible chat
  - `POST /v1/completions` - Text completion
  - `GET /v1/models` - List available models
  - `GET /health` - Health check

### 2. Authentication Layer
- **Method**: API keys (for beta), OAuth 2.0 (for production)
- **Storage**: Hashed keys in PostgreSQL
- **Rate Limiting**: Per-key token bucket

### 3. Router Model (125M)
- **Purpose**: Classify intent, route to appropriate specialist(s)
- **Latency Target**: <50ms
- **Output**: `{specialist: "reasoning"|"language", confidence: 0.95}`

### 4. Specialist Models

| Model | Size | Purpose | Context | Tools |
|-------|------|---------|---------|-------|
| Language | 336M | Prompt interpretation, synthesis | 4K | No |
| Reasoning | 336M | Math, logic, chain-of-thought | 8K | Yes |
| Verifier | 191M | Validate answers, catch errors | 4K | No |

### 5. Tool System
- **Sandboxed Execution**: All code runs in isolated containers
- **Timeout**: 30 seconds per tool call
- **Available Tools**:
  - `symbolic_math`: SymPy for calculus, algebra, linear algebra
  - `logic_solver`: Z3 for SAT/SMT solving
  - `execute_python`: Sandboxed Python for numerical computation
  - `physics`: Kinematics, E&M, optics, thermodynamics
  - `chemistry`: Stoichiometry, pH, molecular weight

---

## Data Flow

### Request Processing

```
1. Request arrives at API Gateway
2. Authentication check (API key validation)
3. Rate limit check
4. Request logged (request_id, timestamp, user_id - NO content logged)
5. Router model classifies intent
6. Appropriate specialist(s) invoked
7. If reasoning: tool calls executed in sandbox
8. Verifier checks output (optional, based on confidence)
9. Response formatted and returned
10. Usage metrics recorded (tokens, latency)
```

### What We Store

| Data | Stored? | Duration | Purpose |
|------|---------|----------|---------|
| User prompts | NO | - | Privacy |
| Model outputs | NO | - | Privacy |
| Request metadata | YES | 90 days | Debugging, abuse detection |
| Token counts | YES | Indefinite | Billing |
| Error logs | YES | 30 days | Debugging |
| API keys (hashed) | YES | Until revoked | Authentication |

---

## Infrastructure

### Friends & Family Beta

```
Single Server Deployment (HP Tower or Cloud VM)

+------------------------------------------+
|              Docker Compose               |
|                                          |
|  +----------+  +----------+  +--------+  |
|  |  Nginx   |  |  Svend   |  | Redis  |  |
|  | (proxy)  |  |   API    |  | (cache)|  |
|  +----------+  +----------+  +--------+  |
|                     |                    |
|              +------+------+             |
|              |  PostgreSQL |             |
|              |   (users)   |             |
|              +-------------+             |
+------------------------------------------+
```

**Specs**:
- 32GB RAM minimum
- GPU optional (CPU inference acceptable for beta)
- 100GB SSD

### Production (May 2026)

```
Multi-Node Deployment

                    Load Balancer
                         |
         +---------------+---------------+
         |               |               |
    +----+----+     +----+----+     +----+----+
    |  API 1  |     |  API 2  |     |  API 3  |
    +---------+     +---------+     +---------+
         |               |               |
         +-------+-------+-------+-------+
                 |               |
           +-----+-----+   +----+----+
           |  Model    |   |  Model  |
           | Server 1  |   | Server 2|
           | (GPU)     |   | (GPU)   |
           +-----------+   +---------+
```

---

## Security Considerations

### Input Validation
- Max prompt length: 8192 tokens
- Content filtering before processing
- Injection prevention for tool calls

### Output Safety
- Safety classifier on all outputs
- Tool output sanitization
- No executable code in responses (unless explicitly requested)

### Network Security
- HTTPS only (TLS 1.3)
- API keys transmitted via header only
- No sensitive data in URLs

---

## Monitoring

### Metrics to Track
- Request latency (p50, p95, p99)
- Token throughput
- Error rates by type
- Model inference time per specialist
- Tool execution time
- Cache hit rate

### Alerts
- Error rate > 5%
- Latency p95 > 5s
- Memory usage > 80%
- Disk usage > 90%

---

## Cost Model (Production Target)

| Component | Cost/month |
|-----------|------------|
| GPU Instance (A10G) | $300-500 |
| Storage | $20 |
| Bandwidth | $50-100 |
| **Total** | **~$400-600** |

**Pricing Target**: $5/user/month or $3/1M tokens

Break-even: ~100 users at $5/month

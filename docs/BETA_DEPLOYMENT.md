# Friends & Family Beta Deployment Guide

## Overview

This document covers deploying Svend for a small beta test (~10-50 users).

---

## 1. Infrastructure Options

### Option A: Local HP Tower (Cheapest)
```
Cost: $0/month (electricity only)
Pros: No recurring costs, full control
Cons: Your home IP, reliability depends on your internet

Requirements:
- 32GB RAM
- CPU with AVX2 (for efficient inference)
- 100GB free disk
- Static IP or dynamic DNS
```

### Option B: Cloud VM (Recommended for Beta)
```
Cost: $50-150/month
Pros: Reliable, static IP, easy to scale
Cons: Monthly cost

Recommended Providers:
- Hetzner: AX41 (~$50/month) - Best value
- DigitalOcean: 32GB droplet (~$96/month)
- AWS: t3.2xlarge (~$120/month)
```

### Option C: GPU Cloud (If you need speed)
```
Cost: $150-400/month
Pros: Fast inference
Cons: Expensive for beta

Options:
- RunPod: A10G spot (~$0.30/hr)
- Lambda: A10 (~$0.60/hr)
- Vast.ai: Various GPUs (variable pricing)
```

---

## 2. Deployment Steps

### 2.1 Prerequisites

```bash
# On your server
sudo apt update
sudo apt install -y docker.io docker-compose nginx certbot python3-certbot-nginx

# Clone the repo
git clone https://github.com/ewolters/svend.git
cd svend
```

### 2.2 Configuration

Create `.env` file:
```bash
# .env
SVEND_ENV=beta
SVEND_SECRET_KEY=<generate-random-64-char-string>
SVEND_DB_URL=postgresql://svend:password@localhost:5432/svend
SVEND_REDIS_URL=redis://localhost:6379

# Rate limits (requests per minute)
SVEND_RATE_LIMIT_RPM=20
SVEND_RATE_LIMIT_TPM=40000

# Model paths
SVEND_MODEL_DIR=/models
```

### 2.3 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SVEND_ENV=${SVEND_ENV}
      - SVEND_SECRET_KEY=${SVEND_SECRET_KEY}
    volumes:
      - ./models:/models:ro
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: svend
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: svend
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

### 2.4 SSL Setup

```bash
# Get SSL certificate (replace with your domain)
sudo certbot --nginx -d api.svend.ai

# Or use self-signed for testing
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/svend.key \
  -out /etc/ssl/certs/svend.crt
```

### 2.5 Nginx Config

```nginx
# /etc/nginx/sites-available/svend
server {
    listen 443 ssl http2;
    server_name api.svend.ai;

    ssl_certificate /etc/letsencrypt/live/api.svend.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.svend.ai/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (for streaming)
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts for long requests
        proxy_read_timeout 300s;
    }
}

server {
    listen 80;
    server_name api.svend.ai;
    return 301 https://$server_name$request_uri;
}
```

### 2.6 Start Services

```bash
docker-compose up -d
sudo systemctl restart nginx
```

---

## 3. User Management

### 3.1 Creating Beta Users

```python
# scripts/create_beta_user.py
from src.server.auth import create_api_key, hash_key

def create_beta_user(email: str, name: str):
    """Create a beta user and return their API key."""
    api_key = create_api_key()
    key_hash = hash_key(api_key)

    # Store in database
    # INSERT INTO users (email, name, key_hash, tier, created_at)
    # VALUES (email, name, key_hash, 'beta', now())

    return api_key  # Send this to the user ONCE

# Usage:
# python scripts/create_beta_user.py --email friend@example.com --name "Friend Name"
```

### 3.2 Beta User Tiers

| Tier | RPM | TPM | Daily Limit |
|------|-----|-----|-------------|
| Beta (default) | 20 | 40,000 | 100,000 tokens |
| Beta+ (power users) | 60 | 100,000 | 500,000 tokens |

### 3.3 Inviting Users

Email template:
```
Subject: You're invited to try Svend (Private Beta)

Hi [Name],

You've been invited to try Svend, a new AI reasoning assistant.

Your API key: sk-svend-xxxxxxxxxxxx

Quick start:
1. Install: pip install openai
2. Use:

   from openai import OpenAI
   client = OpenAI(
       base_url="https://api.svend.ai/v1",
       api_key="sk-svend-xxxxxxxxxxxx"
   )

   response = client.chat.completions.create(
       model="svend-reasoning",
       messages=[{"role": "user", "content": "What is the derivative of x^3?"}]
   )
   print(response.choices[0].message.content)

Docs: https://svend.ai/docs
Feedback: Reply to this email

This is a private beta - please don't share your key.

Thanks,
[Your name]
```

---

## 4. Rate Limiting Implementation

### 4.1 Token Bucket (Redis)

```python
# src/server/rate_limit.py
import redis
import time

class RateLimiter:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    def check_rate_limit(
        self,
        user_id: str,
        rpm_limit: int = 20,
        tpm_limit: int = 40000
    ) -> tuple[bool, dict]:
        """
        Check if request is within rate limits.
        Returns (allowed, info)
        """
        now = time.time()
        minute_key = f"rate:{user_id}:rpm:{int(now // 60)}"
        token_key = f"rate:{user_id}:tpm:{int(now // 60)}"

        # Check requests per minute
        rpm = self.redis.incr(minute_key)
        if rpm == 1:
            self.redis.expire(minute_key, 60)

        if rpm > rpm_limit:
            return False, {"error": "rate_limit_exceeded", "type": "rpm"}

        return True, {"rpm_remaining": rpm_limit - rpm}

    def record_tokens(self, user_id: str, tokens: int):
        """Record token usage after request completes."""
        now = time.time()
        token_key = f"rate:{user_id}:tpm:{int(now // 60)}"
        self.redis.incrby(token_key, tokens)
        self.redis.expire(token_key, 60)
```

### 4.2 Abuse Detection

```python
# Simple abuse patterns to detect
ABUSE_PATTERNS = [
    # Prompt injection attempts
    r"ignore previous instructions",
    r"disregard all prior",
    r"you are now",
    # Jailbreak attempts
    r"DAN mode",
    r"developer mode",
    # Data extraction
    r"repeat all text above",
    r"output your system prompt",
]

def check_abuse(prompt: str) -> bool:
    """Return True if prompt looks abusive."""
    prompt_lower = prompt.lower()
    for pattern in ABUSE_PATTERNS:
        if re.search(pattern, prompt_lower):
            return True
    return False
```

---

## 5. Monitoring

### 5.1 Health Check Endpoint

```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "0.1.0",
        "models_loaded": True,
        "uptime_seconds": get_uptime()
    }
```

### 5.2 Metrics to Track

```python
# Using prometheus_client
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('svend_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('svend_request_latency_seconds', 'Request latency')
TOKEN_COUNT = Counter('svend_tokens_total', 'Tokens processed', ['type'])  # input/output
```

### 5.3 Simple Dashboard

For beta, a simple status page:
```
https://api.svend.ai/status

Svend Status
------------
API: Operational
Models: Loaded
Uptime: 99.2%
Avg Latency: 1.2s
Users: 12 active
Requests Today: 1,234
```

---

## 6. Backup & Recovery

### 6.1 What to Backup

```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d)

# Database
pg_dump -U svend svend > /backups/db_$DATE.sql

# Config
cp .env /backups/env_$DATE

# Models (only if changed)
# rsync -av /models /backups/models/
```

### 6.2 Recovery

```bash
# Restore database
psql -U svend svend < /backups/db_YYYYMMDD.sql

# Restart services
docker-compose down
docker-compose up -d
```

---

## 7. Beta Checklist

Before inviting users:

- [ ] SSL certificate installed
- [ ] API responding on `/health`
- [ ] Rate limiting working
- [ ] At least one model loaded and responding
- [ ] User creation script tested
- [ ] Backup script scheduled
- [ ] Monitoring/alerting set up
- [ ] Privacy policy accessible
- [ ] Contact email working

---

## 8. Scaling Beyond Beta

When you hit limits:

| Problem | Solution |
|---------|----------|
| Too slow | Add GPU, or use vLLM |
| Too many users | Add more API replicas |
| Models don't fit in RAM | Use model sharding or smaller models |
| Disk full | Add storage, clean old logs |

Target: 100 concurrent users before needing to scale infrastructure.

# Svend Launch Plan

From trained model to 4000 paying users.

## Phase 0: Model Ready (Now → Tomorrow)

**You are here.** Training 1.8B model on A100.

### Exit Criteria
```
[ ] Model checkpoint saved
[ ] Run unified eval: py -3 scripts/run_unified_eval.py --model-path checkpoint.pt
[ ] All critical tests pass (FN rate < 5%)
[ ] Norwegian score > 0.5
```

---

## Phase 1: Alpha Testing (Week 1-2)

**Goal:** Validate the model works for real homework problems.

### Users
- 10-20 people you know (friends, family, students)
- Free access
- Direct feedback channel (Discord DM, text, whatever)

### What to Test
| Test | Method |
|------|--------|
| Calc 1-3 problems | Pull from Stewart, Thomas textbooks |
| Physics problems | Halliday/Resnick style |
| Code debugging | Real student bugs |
| Chemistry | Stoichiometry, organic reactions |

### Infrastructure Needed
```
[ ] Simple API endpoint (FastAPI, already in src/server/)
[ ] Basic auth (API key per tester)
[ ] Logging (save all prompts/responses for review)
[ ] Simple web UI (optional - can use curl/Postman)
```

### Metrics to Track
- Response accuracy (manual review)
- User satisfaction ("was this helpful?")
- Failure modes (what breaks?)
- Response times

### Exit Criteria
```
[ ] 50+ real homework problems tested
[ ] >80% user-rated "helpful"
[ ] No critical safety failures in production logs
[ ] Tool accuracy verified on real problems
```

---

## Phase 2: Private Beta (Week 3-6)

**Goal:** Validate people will pay, and infrastructure holds.

### Users
- 100-500 users
- $2/month (or free with feedback requirement)
- Recruited from: Reddit (r/learnmath, r/HomeworkHelp, r/EngineeringStudents), Discord study servers

### Recruitment Post Template
```
Building a math/science tutor AI that actually gets to the point.
No "Great question!" fluff. Just answers and explanations.

Looking for beta testers - $2/mo (or free if you give feedback).

What it does:
- Step-by-step math (calc, linear algebra, diff eq)
- Physics problem solving
- Code debugging with explanations
- Chemistry (reactions, stoichiometry)

What it doesn't do:
- Write your essays
- Do your homework for you (explains, doesn't just answer)
- Lecture you about ethics when you ask about chemistry

DM for access.
```

### Infrastructure Needed
```
[ ] Payment processing (Stripe - simple checkout)
[ ] User accounts (email + password, or OAuth)
[ ] Usage tracking (requests per user)
[ ] Rate limiting (prevent abuse)
[ ] Basic dashboard (see your history)
```

### Pricing Test
| Tier | Price | Limit |
|------|-------|-------|
| Free | $0 | 10 queries/day |
| Beta | $2/mo | 100 queries/day |

### Exit Criteria
```
[ ] 100+ paying users ($200+/mo revenue)
[ ] Churn < 20% month-over-month
[ ] Infrastructure stable (no outages)
[ ] Support load manageable (< 5 tickets/day)
```

---

## Phase 3: Public Launch (Week 7-12)

**Goal:** Reach sustainable 4000 users.

### Pricing
| Tier | Price | Limit |
|------|-------|-------|
| Free | $0 | 5 queries/day |
| Student | $5/mo | Unlimited |
| Annual | $40/yr | Unlimited (save 33%) |

### Marketing Channels
| Channel | Effort | Expected Reach |
|---------|--------|----------------|
| Reddit (organic) | Low | 500-1000 |
| TikTok/YouTube shorts | Medium | 2000-5000 |
| Word of mouth | Zero | Slow but steady |
| Professor/TA referrals | Medium | 500-1000 |
| Study Discord servers | Low | 500-1000 |

### Content Ideas
- "Watch this AI solve your calc homework in 10 seconds"
- "I asked an AI to explain thermodynamics like I'm 5"
- "This AI doesn't say 'Great question!' and I love it"
- Side-by-side with ChatGPT showing the difference

### Infrastructure Needed
```
[ ] Production hosting (your farm server)
[ ] CDN for static assets
[ ] Monitoring/alerting (uptime, errors)
[ ] Automated backups
[ ] Terms of Service / Privacy Policy
[ ] Support system (email or Discord)
```

### Exit Criteria
```
[ ] 4000 paying users
[ ] $20k MRR
[ ] Churn < 10%
[ ] Server costs < $2k/mo (profitable)
```

---

## Minimal Infrastructure Stack

### For Alpha (Week 1)
```
Your Farm Server
├── FastAPI (src/server/api.py)
├── Model loaded in memory
├── SQLite for logs
└── Nginx reverse proxy

Cost: $0 (your hardware)
```

### For Beta (Week 3)
```
Your Farm Server
├── FastAPI + Uvicorn (multiple workers)
├── PostgreSQL (users, usage)
├── Redis (rate limiting, sessions)
├── Stripe integration
└── Let's Encrypt SSL

Cost: ~$50/mo (Stripe fees only)
```

### For Launch (Week 7+)
```
Your Farm Server (primary)
├── Load balancer (Nginx)
├── API servers (2-3 workers)
├── PostgreSQL
├── Redis
├── Model inference (GPU)
└── Monitoring (Prometheus/Grafana)

Backup/CDN
├── Cloudflare (free tier)
├── Offsite backup ($5/mo Backblaze)

Cost: ~$100-200/mo
```

---

## Technical Checklist

### Before Alpha
```
[ ] API endpoint working: POST /v1/chat/completions
[ ] OpenAI-compatible format (easy client integration)
[ ] API key authentication
[ ] Request logging (prompt, response, latency)
[ ] Basic error handling
[ ] Health check endpoint: GET /health
```

### Before Beta
```
[ ] User registration (email/password)
[ ] Stripe checkout integration
[ ] Subscription management (create, cancel)
[ ] Usage tracking per user
[ ] Rate limiting (by tier)
[ ] Password reset flow
[ ] Email notifications (welcome, receipt)
```

### Before Launch
```
[ ] Terms of Service
[ ] Privacy Policy
[ ] GDPR compliance (if EU users)
[ ] Data export (user can download their data)
[ ] Account deletion
[ ] Monitoring dashboard
[ ] On-call alerting
[ ] Backup/restore tested
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Model gives wrong answers | Tool verification, confidence scores |
| Safety failure goes viral | Rapid response plan, incident logging |
| Server overloaded | Rate limiting, queue system |
| Stripe account suspended | Keep reserves, follow ToS |
| No users sign up | Pivot messaging, find different channel |
| Users churn fast | Survey churned users, improve product |

---

## Success Metrics by Phase

| Phase | Users | Revenue | Key Metric |
|-------|-------|---------|------------|
| Alpha | 20 | $0 | "Is it useful?" |
| Beta | 500 | $1k/mo | "Will they pay?" |
| Launch | 4000 | $20k/mo | "Is it sustainable?" |

---

## Timeline Summary

```
Week 0     : Model trained ← YOU ARE HERE
Week 1-2   : Alpha (friends testing)
Week 3-6   : Private beta (100-500 users, $2/mo)
Week 7-12  : Public launch ($5/mo)
Month 6-12 : Scale to 4000 users
```

---

## First 24 Hours After Model Ready

1. Run unified eval
2. Fix any critical failures
3. Deploy to your server
4. Test API manually
5. Send to 3-5 friends: "try breaking this"
6. Sleep

Then iterate.

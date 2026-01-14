# Svend Style Guide

## Brand Identity

**Svend** is a reasoning system that shows its work. The visual identity reflects:
- **Deliberate**: Thoughtful, unhurried, precise
- **Natural**: Forest-at-night aesthetic, organic warmth
- **Transparent**: Shows its reasoning, nothing hidden

---

## Typography

### Primary Font: Inter

```css
font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
```

**Google Fonts:**
```html
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
```

### Font Weights

| Weight | Use |
|--------|-----|
| 300 (Light) | Body text, descriptions |
| 400 (Regular) | Default, highlighted text |
| 500 (Medium) | Buttons, labels, emphasis |
| 600 (Semibold) | Headings (sparingly) |

### Logo Treatment

```css
.logo {
    font-family: 'Inter', sans-serif;
    font-weight: 300;
    font-size: 2.5rem;
    letter-spacing: 0.4em;  /* Wide spacing is key */
    text-transform: uppercase;
}
```

**Example:** `S V E N D` (not `SVEND`)

### Type Scale

| Element | Size | Weight | Line Height |
|---------|------|--------|-------------|
| Logo | 2.5rem | 300 | 1 |
| H1 | 1.5rem | 400 | 1.3 |
| H2 | 1.25rem | 500 | 1.4 |
| H3 (labels) | 0.7rem | 500 | 1.2 |
| Body | 0.95rem | 300 | 1.7 |
| Small | 0.8rem | 300 | 1.5 |
| Code | 0.875rem | 400 | 1.5 |

---

## Color Palette

### Core Colors

```css
:root {
    /* === BACKGROUNDS === */
    --bg-primary: #0a0f0a;      /* Deep forest - main background */
    --bg-secondary: #0d120d;    /* Slightly lighter - cards, code blocks */
    --bg-tertiary: #121a12;     /* Elevated surfaces */
    --bg-hover: #1a261a;        /* Hover states */

    /* === PRIMARY ACCENT (Aurora Green) === */
    --accent-primary: #4a9f6e;  /* Main brand color */
    --accent-primary-dim: rgba(74, 159, 110, 0.15);  /* Backgrounds */
    --accent-primary-border: rgba(74, 159, 110, 0.3); /* Borders */

    /* === SECONDARY ACCENTS === */
    --accent-blue: #3a7f8f;     /* Info, links, secondary actions */
    --accent-purple: #5a4f7f;   /* Tertiary, special states */
    --accent-gold: #e8c547;     /* Highlights, important info */
    --accent-orange: #e89547;   /* Warnings */

    /* === TEXT === */
    --text-primary: #e8efe8;    /* Main text - slight green tint */
    --text-secondary: #9aaa9a;  /* Body text, descriptions */
    --text-dim: #5a6a5a;        /* Labels, placeholders, muted */
    --text-inverse: #0a0f0a;    /* Text on light backgrounds */

    /* === SEMANTIC === */
    --success: #4a9f6e;         /* Same as primary accent */
    --warning: #e89547;         /* Orange */
    --error: #9f4a4a;           /* Muted red */
    --info: #3a7f8f;            /* Blue */

    /* === TOOL INDICATORS === */
    --tool-python: #4a9f6e;     /* Green - code execution */
    --tool-sympy: #3a7f8f;      /* Blue - symbolic math */
    --tool-z3: #5a4f7f;         /* Purple - logic/SAT */
    --tool-chemistry: #e89547;  /* Orange - chemistry */
    --tool-physics: #e8c547;    /* Gold - physics */
}
```

### Color Usage

| Element | Background | Border | Text |
|---------|------------|--------|------|
| Page | `--bg-primary` | - | `--text-primary` |
| Card | `--bg-secondary` | `--accent-primary-border` | `--text-primary` |
| Button (primary) | `--accent-primary-dim` | `--accent-primary-border` | `--accent-primary` |
| Button (hover) | `rgba(74, 159, 110, 0.25)` | `rgba(74, 159, 110, 0.5)` | `--accent-primary` |
| Input | `rgba(255, 255, 255, 0.03)` | `--accent-primary-border` | `--text-primary` |
| Input (focus) | `rgba(255, 255, 255, 0.05)` | `rgba(74, 159, 110, 0.4)` | `--text-primary` |
| Code block | `--bg-secondary` | `--bg-hover` | `--text-primary` |
| Label | - | - | `--text-dim` |

---

## Components

### Buttons

```css
/* Primary Button */
.btn-primary {
    padding: 0.875rem 1.5rem;
    background: rgba(74, 159, 110, 0.15);
    border: 1px solid rgba(74, 159, 110, 0.3);
    border-radius: 3px;
    color: #4a9f6e;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
}

.btn-primary:hover {
    background: rgba(74, 159, 110, 0.25);
    border-color: rgba(74, 159, 110, 0.5);
}

/* Secondary Button */
.btn-secondary {
    padding: 0.875rem 1.5rem;
    background: transparent;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    color: #9aaa9a;
    font-size: 0.85rem;
    font-weight: 500;
}

.btn-secondary:hover {
    border-color: rgba(255, 255, 255, 0.2);
    color: #e8efe8;
}
```

### Cards

```css
.card {
    background: rgba(10, 15, 10, 0.85);
    border: 1px solid rgba(74, 159, 110, 0.15);
    border-radius: 4px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    box-shadow:
        0 4px 30px rgba(0, 0, 0, 0.5),
        inset 0 1px 0 rgba(255, 255, 255, 0.03);
}
```

### Inputs

```css
.input {
    padding: 0.875rem 1rem;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(74, 159, 110, 0.2);
    border-radius: 3px;
    color: #e8efe8;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    outline: none;
    transition: border-color 0.2s, background 0.2s;
}

.input::placeholder {
    color: #5a6a5a;
}

.input:focus {
    border-color: rgba(74, 159, 110, 0.4);
    background: rgba(255, 255, 255, 0.05);
}
```

### Status Indicators

```css
/* Pulsing status dot */
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #4a9f6e;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 1; }
}

/* Status colors */
.status-active { background: #4a9f6e; }   /* Green */
.status-warning { background: #e89547; }  /* Orange */
.status-error { background: #9f4a4a; }    /* Red */
.status-pending { background: #5a6a5a; }  /* Gray */
```

### Tool Call Badges

```css
/* For showing which tool is being used */
.tool-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.tool-badge--python {
    background: rgba(74, 159, 110, 0.15);
    color: #4a9f6e;
    border: 1px solid rgba(74, 159, 110, 0.3);
}

.tool-badge--sympy {
    background: rgba(58, 127, 143, 0.15);
    color: #3a7f8f;
    border: 1px solid rgba(58, 127, 143, 0.3);
}

.tool-badge--z3 {
    background: rgba(90, 79, 127, 0.15);
    color: #7a6f9f;
    border: 1px solid rgba(90, 79, 127, 0.3);
}

.tool-badge--chemistry {
    background: rgba(232, 149, 71, 0.15);
    color: #e89547;
    border: 1px solid rgba(232, 149, 71, 0.3);
}
```

---

## Reasoning Display

### Step-by-Step Reasoning

```css
.reasoning-step {
    padding: 1rem;
    border-left: 2px solid rgba(74, 159, 110, 0.3);
    margin-left: 0.5rem;
    margin-bottom: 1rem;
}

.reasoning-step--active {
    border-left-color: #4a9f6e;
    background: rgba(74, 159, 110, 0.05);
}

.reasoning-step__number {
    font-size: 0.7rem;
    font-weight: 500;
    color: #5a6a5a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

.reasoning-step__content {
    font-size: 0.9rem;
    color: #e8efe8;
    line-height: 1.6;
}
```

### Code Blocks

```css
.code-block {
    background: #0d120d;
    border: 1px solid #1a261a;
    border-radius: 4px;
    padding: 1rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.875rem;
    line-height: 1.5;
    overflow-x: auto;
}

.code-block__header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 1rem;
    background: rgba(255, 255, 255, 0.02);
    border-bottom: 1px solid #1a261a;
    font-size: 0.75rem;
    color: #5a6a5a;
}
```

---

## Animation Guidelines

### Timing

| Type | Duration | Easing |
|------|----------|--------|
| Hover | 0.2s | ease |
| Expand/Collapse | 0.3s | ease-out |
| Page transitions | 0.4s | ease-in-out |
| Ambient (fireflies) | 3-10s | ease-in-out |

### Principles

1. **Subtle**: Animations should feel natural, not flashy
2. **Purposeful**: Every animation communicates something
3. **Ambient**: Background animations (fireflies, aurora) create atmosphere without distraction
4. **Responsive**: UI feedback should be immediate (< 0.2s)

---

## Spacing Scale

```css
:root {
    --space-xs: 0.25rem;   /* 4px */
    --space-sm: 0.5rem;    /* 8px */
    --space-md: 1rem;      /* 16px */
    --space-lg: 1.5rem;    /* 24px */
    --space-xl: 2rem;      /* 32px */
    --space-2xl: 3rem;     /* 48px */
    --space-3xl: 4rem;     /* 64px */
}
```

---

## Iconography

Use simple, line-based icons. Recommended sets:
- **Lucide** (open source, consistent)
- **Heroicons** (Tailwind's icons)

Icon style:
- Stroke width: 1.5-2px
- Size: 16px (small), 20px (medium), 24px (large)
- Color: Inherit from text color

---

## Do's and Don'ts

### Do
- Use generous whitespace
- Keep text readable (high contrast)
- Show reasoning steps clearly
- Use subtle animations for ambient feel
- Maintain the forest/night theme

### Don't
- Use pure white (#fff) - it's too harsh
- Use pure black (#000) - use forest dark instead
- Overuse accent colors - green is primary
- Add unnecessary decorations
- Use flashy animations

---

## File Structure

```
site/
├── index.html          # Landing page
├── STYLE_GUIDE.md      # This file
├── styles/
│   ├── variables.css   # CSS custom properties
│   ├── base.css        # Reset, typography
│   ├── components.css  # Buttons, cards, inputs
│   └── utilities.css   # Helper classes
└── assets/
    ├── og-image.png    # Social share image
    └── favicon.svg     # Tree emoji favicon
```

---

## Quick Reference

```css
/* Copy-paste starter */
:root {
    --bg: #0a0f0a;
    --bg-card: #0d120d;
    --accent: #4a9f6e;
    --text: #e8efe8;
    --text-muted: #9aaa9a;
    --text-dim: #5a6a5a;
    --border: rgba(74, 159, 110, 0.2);
}

body {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
}
```

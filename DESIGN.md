# DESIGN.md — Streamlit Visual Identity

## Design Philosophy

Football data visualization has two failure modes: (1) generic "dashboard blue" that looks like every other analytics tool, and (2) over-designed theme parks drowning in animations. This system aims for a third path: **scoreboard minimalism** — the kind of visual restraint you see in broadcast overlays, where a small amount of deliberate typography carries weight.

**The rule:** spend visual boldness in exactly one place (the Turkey advance percentage hero metric). Everything else stays composed and functional.

---

## Colour Palette

| Token | Hex | Use |
|-------|-----|-----|
| `C_BG` | `#0a0f0d` | Page background — deep pitch black-green |
| `C_SURFACE` | `#111916` | Card/panel backgrounds |
| `C_BORDER` | `#1e2e25` | Subtle borders, dividers |
| `C_ACCENT` | `#00e676` | Primary electric green — goal flash, headings, model data |
| `C_ACCENT2` | `#ff6d00` | Amber — Turkey highlights only |
| `C_TEXT` | `#e8f5e9` | Near-white body text with green tint |
| `C_MUTED` | `#546e62` | Labels, secondary text, section headers |
| `C_HOME` | `#00bcd4` | Cyan — home win probability bars |
| `C_DRAW` | `#90a4ae` | Grey — draw probability bars |
| `C_AWAY` | `#ef5350` | Red — away win probability bars |

**Palette rationale:** Dark green base evokes a football pitch without being literal. The electric green accent is borrowed from stadium display boards (Wembley, Lusail). Amber is reserved exclusively for Turkey — one team, one colour.

---

## Typography

| Role | Font | Fallback |
|------|------|----------|
| Display (headings, scores, numbers) | Oswald 400/600/700 | system-ui sans-serif |
| Body (paragraphs, labels) | Inter 300/400/500 | system-ui sans-serif |
| Monospace (data, probabilities, match IDs) | JetBrains Mono | monospace |

**Typography rationale:** Oswald is a condensed grotesque — it looks like scoreboard typography without being a literal dot-matrix font. Inter is optimally readable at small sizes. JetBrains Mono disambiguates `0` from `O` in probability values.

---

## Layout Concept

- **Single-column content** within a `max-width: 1100px` container — prevents data tables from stretching across wide monitors uncomfortably.
- **Three tabs** as the primary navigation — no sidebar (clean mobile experience).
- **Hero metric first** on the Turkey tab — the advance percentage is the single most important number; it gets a full-width card at the top.
- **Section headings** in `C_MUTED` with a bottom border — functional dividers, not decorative flourishes.

---

## Signature Element

The **probability bars** component (`prob_bars()`) is the one design invention in this system. Instead of Streamlit's default `st.progress()` or plotly horizontal bars, it uses custom HTML: a three-row inline layout with team name (truncated), a thin bar track, and a monospace percentage. The bars are colored by outcome (cyan / grey / red), not by team.

This component appears in the Tournament tab (every match) and the Turkey tab (next match card) — it's the visual thread that ties the experience together.

---

## Responsiveness

- CSS uses `flex` for the probability bar rows → collapses cleanly on mobile.
- `@prefers-reduced-motion` is respected by default (no animations in Streamlit).
- Plotly charts use `use_container_width=True` for fluid sizing.

---

## What This Design Avoids

- No hero images or background photos (bloats load time, feels generic)
- No animated counters (distracting, no a11y benefit)
- No data gradient fills on bar charts (misleading visual weight)
- No emoji in headings or section titles (except the 🇹🇷 Turkey tab label, which is navigational)
- No dark/light mode toggle (single dark mode, consistent branding)

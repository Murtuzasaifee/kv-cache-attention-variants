# Archify Diagrams — Module 03

12 self-contained interactive HTML diagrams covering the full KV cache / attention variant curriculum.

## How to view

Three options, all working:

1. **MkDocs site** (recommended) — `mkdocs serve` and visit `/diagrams/` — diagrams are part of the docs site
2. **MkDocs landing page** — `site/diagrams/index.html` after `mkdocs build`
3. **Source-of-truth copies** — `docs/diagrams/*.html` (the same files MkDocs builds into the site)

Individual diagrams can be opened directly from `docs/diagrams/D01_*.html` through `docs/diagrams/D12_*.html`.

## What's interactive

Each diagram is a single HTML file with no external dependencies. The Archify Viewer (embedded in each HTML) provides:

- **Pan / zoom** — drag to pan, scroll to zoom
- **Theme switch** — light / dark (independent of presenter mode)
- **Search** — by node label, relationship, etc.
- **Focus mode** — single-component inspection
- **Relationship tracing** — hover arrows to highlight
- **Export** — to PNG, SVG, or JSON spec

## Source of truth

Source JSON specs live in `scripts/archify_D*.json`. Each was authored against the schema in `~/.commandcode/skills/archify/schemas/`, validated through all 9 showcase artifact checks before being frozen via `archify deliver`, and visually checked at 1440×900 / 1600×1000 / 1920×1080 / 2048×1320.

The series plan (rationale, file naming, validation gates, open questions) is in `scripts/archify_diagram_series_plan.md`.

## Diagram-to-session map

| Diagram | Type | Sessions served |
|---|---|---|
| D1 | architecture | 00–10 |
| D2 | sequence | 00, 01 |
| D3 | architecture | 02 |
| D4 | architecture | 03 |
| D5 | architecture | 04, 05, 07 |
| D6 | architecture | 05 |
| D7 | architecture | 07 |
| D8 | architecture | 06 |
| D9 | architecture | 06 |
| D10 | architecture | 08 |
| D11 | architecture | 09 |
| D12 | architecture | 10 |

Note: the curriculum dependency in D1 uses the **approved reordered sequence** (RoPE before MLA, MHA Recap before memory math). The session nav in `mkdocs.yml` is unchanged from the original module spec and may diverge in numbering.

## Re-delivering a diagram

If a JSON spec is edited, re-deliver with:

```bash
cd ~/.commandcode/skills/archify
TYPE=$(grep '"diagram_type"' ../../scripts/archify_D01_*.json | head -1 | cut -d'"' -f4)
node bin/archify.mjs deliver "$TYPE" \
  ../../scripts/archify_D01_*.json \
  ../../docs/diagrams/D01_*.html \
  --quality showcase
```

Or use the loop in `scripts/archify_diagram_series_plan.md` for batch re-delivery.

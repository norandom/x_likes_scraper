# Sentrux Signal: Pre and Post

Recorded for task 10.1 of the codebase-foundation spec.

## Baseline (pre-spec, 2026-04-30)

```
files: 22
lines: 3,469
import_edges: 9
quality_signal: 6469
bottleneck: modularity

root_causes:
  acyclicity:  10000  (raw 0)
  depth:        8889  (raw 1)
  redundancy:   7625  (raw 0.2375)
  equality:     5016  (raw 0.498)
  modularity:   3333  (raw 0.0)
```

## After spec (2026-05-01)

```
files: 99
lines: 13,100
import_edges: 26
quality_signal: 6666     (+197 / +3%)
bottleneck: modularity   (unchanged)

root_causes:
  acyclicity:  10000  (raw 0)         unchanged
  depth:        8889  (raw 1)         unchanged
  redundancy:   8710  (raw 0.129)     was 7625 / 0.2375
  equality:     5100  (raw 0.49)      was 5016 / 0.498
  modularity:   3333  (raw 0.0)       unchanged
```

## Reading

The headline win is **redundancy**: raw rate dropped from 24% to 13% (the score went from 7625 to 8710). The four near-identical date-parse-with-fallback blocks collapsed into one `parse_x_datetime` helper, and the parser logic moved out of `XAPIClient` private methods into pure functions in `parser.py` that the client now delegates to.

**Modularity stayed flat at 3333** despite the file count tripling (22 to 99) and lines almost quadrupling (3,469 to 13,100). The reason is structural: every new file (`dates.py`, `parser.py`, `loader.py`, all the `tests/test_*.py`) imports from the production package's modules, so 100% of internal imports cross module boundaries. Sentrux's modularity axis can't reward small, single-responsibility libraries when every file naturally talks to a sibling. This is the project's geometry, not a defect.

**Equality nudged up slightly** (5016 → 5100). Adding many small test files spreads file-size mass out a little, but the production code still has the `client.py` (~400 LoC of GraphQL boilerplate) versus `models.py` (small dataclass) imbalance.

**Depth and acyclicity** are unchanged (already perfect/good).

## Verdict

Signal direction: **up** (6469 → 6666). Both axes the spec named — redundancy and the overall signal — moved in the intended direction with no regression elsewhere. Spec passes its own success criterion.

## Follow-up maintenance sweep (post-spec)

After the spec landed, ran a second pass driven by sentrux's `check_rules`:

- Removed two unused runtime deps (`python-dateutil`, `beautifulsoup4`) — neither was imported anywhere after the date-parse consolidation.
- Added `.sentrux/rules.toml` codifying the four-layer model (leaf, parsing, io, orchestration) plus boundary rules locking in the loader's cookies-free contract. The rules file is what the upcoming mcp-pageindex spec gets gated against.
- Extracted `LIKES_API_FEATURES` from `XAPIClient.fetch_likes` into a module-level constant; the function dropped from 141 lines to under 30.
- Split `cli.py:main` (206 lines) into `_build_parser`, `_show_checkpoint_info`, `_clear_checkpoint`, `_resolve_formats`, `_run_exports`, `_print_stats`, with `main` now ~50 lines of orchestration.

Final sentrux state:

```
quality_signal: 6723   (was 6666 / 6469 baseline)
redundancy:     8758   (raw 0.124)
equality:       5292   (raw 0.471 — improved by cli split)
modularity:     3333   (unchanged — project size constraint)
depth:          8889   (unchanged)
acyclicity:    10000   (unchanged)

check_rules: PASS (4 of 13 rules checked on free tier; 0 violations)
```

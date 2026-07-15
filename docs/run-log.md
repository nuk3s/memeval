# Run log format

Every eval run appends events to `runs/<UTC>_<adapter>_<kind>.jsonl`, one JSON object per
line, in the order things happened. Same-second collisions get a `-2`, `-3`… suffix (an
`ab` run starts two legs inside one second). `memeval replay`, `memeval chart`, and
`memeval runs` consume this file; nothing else is persisted. Schema field:
`run_meta.schema` (currently 1).

| ev | fields |
|---|---|
| `run_meta` | `run_id`, `kind` (`rank`\|`adversarial`\|`ab-leg`), `adapter`, `adapter_config` (secrets masked), `schema`, `k`, `repeat`, `tool_version`, plus `corpus`/`gold`/`probe` paths and their `*_sha256` |
| `retain` | `n_items`, `ms`, `ok`, `error` (null on success) |
| `ready` | `polled_s`, `verified_n`, `sampled_n`, `missing` (ids that never became recallable) |
| `consolidate` | `ms`, `ok`, `error` (null on success) — emitted by adversarial runs around each consolidation attempt |
| `query` | `phase` (`rank`\|`adv_before`\|`adv_after`), `rep`, `query`, `rank` (null = miss), `ms`, `hits` (top-k `{text, score}`, text truncated to 300 chars), `error` |
| `verdict` | `phase`, `query`, `fabricated` (matched forbidden patterns), `expected_present`, `top3`, `error` (a check that errored is neither clean nor fabricated) |
| `summary` | rank: `n`, `k`, `hit@1`, `hit@3`, `hit@k`, `mrr`, `p50_ms`, `p95_ms`, `errors`, `per_rep?` — adversarial: `fabricated_before`, `fabricated_after`, `errors_before`, `errors_after`, `corrections_failed` |

Notes.

The `ready` event is the readiness poll that replaced a fixed sleep: after retain, memeval
polls recall on a sample of corpus docs (word-overlap match, ≥50% of the doc's content
words, because extracting stores rewrite text) until each appears or `--ready-timeout`
runs out. A doc in `missing` was accepted by the store but never became recallable —
silent write loss. The run continues; the event keeps the evidence.

Hit text is truncated in the log for size; fabrication verdicts are computed on the full
recalled text before truncation, so a `verdict` can reference a match you can't fully see
in the neighbouring `query` event.

Latency percentiles count only error-free queries; `errors` counts the rest. Error
messages are never empty strings — a bare exception records its class name.

# Bundled example dataset

125 facts about Northwind, a fictional commerce platform, plus 119 gold queries. It exists
so you can run a meaningful rank eval against your memory store without first writing a
corpus of your own.

## Design

- **One fact per doc.** Each corpus row is a single ops-knowledge-base entry: a port, a
  version, an incident, an owner, a policy. Retrieval either finds it or it doesn't;
  nothing hides in paragraph three.
- **Near-miss distractors.** Fourteen services on distinct ports, several databases with
  different versions and hosts, many dates, many retention windows. A "which port" or
  "how many days" query has real competition, so a sloppy retriever scores like one.
- **Stale pairs.** Six facts were superseded (session cache, search engine, object
  storage, CI system, monitoring, the anonymous rate limit). The old doc reads as
  historical and never states the new value; the new doc states the change and the current
  value. Each pair has a "what is it now" gold query whose answer matches only the new
  doc — a store that surfaces the stale fact first fails it.
- **Gold queries are paraphrases**, asked the way a person asks ("who gets paged first
  when the search cluster falls over?"), not substrings of the doc.

## Format

`corpus.jsonl`: `{"id": "nw-001", "content": "...", "tags": ["source:memeval"]}`

`gold.jsonl`: `{"query": "...", "answer": "<regex>"}` — **`answer` is a regex** that the
recalled text must match, case-insensitive. Use alternation for acceptable variants
(`"PostgreSQL 16|db-main"`); extracting stores rewrite text at retain time, so anchor on
tokens that survive rewriting (ports, versions, dates, names).

`adversarial.json` is separate and self-contained — it carries its own tiny corpus,
chosen to invite a bad consolidation merge.

## Extending it

Add corpus rows with fresh `nw-NNN` ids and matching gold rows, then run
`pytest tests/test_examples.py`. The tests enforce the invariants: every answer regex must
compile, match at least one doc, and match **exactly one** doc — unless your gold query is
one of the six stale-pair "what is it now" queries, in which case up to two is allowed (the
old doc and the new one). Any other regex that hits a second doc is too loose, not the test
too strict; anchor it on words distinctive to the target doc. A word-overlap floor check
catches gold queries that share too little vocabulary with their target doc to be
retrievable at all.

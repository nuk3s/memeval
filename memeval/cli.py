import argparse
import sys

from . import __version__
from .adapters import REGISTRY
from .adversarial import load_probe, run_adversarial
from .events import SCHEMA_VERSION, RunLog, file_sha256, read_events, redact, run_path
from .rank import load_jsonl, run_rank
from .render.text import format_ab, format_adversarial, format_summary
from .runner import Runner


def _load_input(path, loader, label):
    try:
        return loader(path)
    except (OSError, ValueError) as e:
        sys.exit(f"cannot load {label} file {path}: {e}")


def build_adapter(name):
    if name not in REGISTRY:
        sys.exit(f"unknown adapter '{name}'. known: {', '.join(sorted(REGISTRY))}")
    adapter = REGISTRY[name]()
    try:
        adapter.prepare()
    except Exception as e:
        print(f"warning: prepare() failed ({e}); continuing", file=sys.stderr)
    return adapter


def _start_run(args, adapter, kind, files):
    """Create the run log and emit run_meta. files: dict of label -> path."""
    path = run_path(args.runs_dir, adapter.name, kind)
    log = RunLog(path)
    meta = {"run_id": path.stem, "kind": kind, "adapter": adapter.name,
            "adapter_config": redact(adapter.config()), "schema": SCHEMA_VERSION,
            "k": getattr(args, "k", None), "repeat": getattr(args, "repeat", 1),
            "tool_version": __version__}
    for label, f in files.items():
        meta[label] = str(f)
        meta[f"{label}_sha256"] = file_sha256(f)
    log.emit("run_meta", **meta)
    runner = Runner(adapter, log, retries=args.retries,
                    ready_timeout=args.ready_timeout, echo=print)
    return path, log, runner


def _rank_one(args, name, kind):
    corpus = _load_input(args.corpus, load_jsonl, "corpus")
    gold = _load_input(args.gold, load_jsonl, "gold")
    adapter = build_adapter(name)
    path, log, runner = _start_run(args, adapter, kind,
                                   {"corpus": args.corpus, "gold": args.gold})
    try:
        s = run_rank(runner, corpus, gold, k=args.k, repeat=args.repeat)
    finally:
        log.close()
    if s is None:
        sys.exit(f"run aborted: retain failed (see {path})")
    print()
    print(format_summary(s, label=name))
    print(f"  run log: {path}")
    return s


def cmd_rank(args):
    _rank_one(args, args.adapter, "rank")


def cmd_ab(args):
    sa = _rank_one(args, args.a, "ab-leg")
    print()
    sb = _rank_one(args, args.b, "ab-leg")
    print()
    print(format_ab(args.a, sa, args.b, sb))


def cmd_adversarial(args):
    probe = _load_input(args.probe, load_probe, "probe")
    adapter = build_adapter(args.adapter)
    path, log, runner = _start_run(args, adapter, "adversarial", {"probe": args.probe})
    try:
        r = run_adversarial(runner, probe, consolidate=not args.no_consolidate)
    finally:
        log.close()
    if r is None:
        sys.exit(f"run aborted: retain failed (see {path})")
    print()
    print(format_adversarial(r))
    print(f"  run log: {path}")


def cmd_runs(args):
    from pathlib import Path
    d = Path(args.runs_dir)
    files = sorted(d.glob("*.jsonl")) if d.is_dir() else []
    if not files:
        print(f"no runs in {d}")
        return
    for f in files:
        try:
            evs = read_events(f)
        except OSError:
            continue
        meta = evs[0] if evs and evs[0].get("ev") == "run_meta" else {}
        summ = next((e for e in reversed(evs) if e.get("ev") == "summary"), {})
        head = ""
        if summ.get("kind") == "rank":
            head = (f"hit@1 {summ.get('hit@1', '?')}/{summ.get('n', '?')}  "
                    f"MRR {summ.get('mrr', '?')}  p50 {summ.get('p50_ms', '?')}ms")
        elif summ.get("kind") == "adversarial":
            head = (f"fabricated_before {summ.get('fabricated_before', '?')}  "
                    f"after {summ.get('fabricated_after', '?')}")
        print(f"{f.name:>60}  {meta.get('adapter', '?'):>10}  {meta.get('kind', '?'):>11}  {head}")


def cmd_adapters(args):
    for name, cls in sorted(REGISTRY.items()):
        doc = (cls.__doc__ or "").strip().splitlines()[0]
        print(f"{name:>14}  {cls.status:>8}  {doc}")


def cmd_doctor(args):
    import time as _time
    import uuid

    from .runner import _normalize_hits

    adapter = build_adapter(args.adapter)
    nonce = uuid.uuid4().hex[:8]
    sentinels = [("lantern", "crimson"), ("beacon", "viridian"), ("anchor", "cobalt")]
    docs = [{"id": f"doctor-{nonce}-{i}",
             "content": f"memeval doctor sentinel {nonce}: the {n} colour is {v}."}
            for i, (n, v) in enumerate(sentinels, 1)]
    report, ok, retain_ok = [], True, True

    try:
        adapter.retain(docs)
        report.append(("retain", "PASS", f"{len(docs)} sentinel docs"))
    except Exception as e:
        report.append(("retain", "FAIL", str(e) or e.__class__.__name__))
        ok = retain_ok = False

    if retain_ok:
        found, last_err = 0, None
        deadline = _time.monotonic() + args.ready_timeout
        while True:
            try:
                hits = _normalize_hits(
                    adapter.recall(f"memeval doctor sentinel {nonce} colour", 10))
            except Exception as e:
                # transient noise is normal against live stores — keep polling to deadline
                last_err = str(e) or e.__class__.__name__
                hits = []
            texts = " ".join(h["text"] for h in hits).lower()
            found = sum(1 for _, v in sentinels if v in texts)
            if found == len(sentinels) or _time.monotonic() >= deadline:
                break
            _time.sleep(min(1.0, max(0.0, deadline - _time.monotonic())))
        if found == len(sentinels):
            report.append(("recall roundtrip", "PASS",
                           f"{found}/{len(sentinels)} sentinels visible"))
        elif found:
            report.append(("recall roundtrip", "PARTIAL",
                           f"{found}/{len(sentinels)} sentinels visible — "
                           "silent write loss or slow indexing (try --ready-timeout)"))
            ok = False
        else:
            report.append(("recall roundtrip", "FAIL",
                           last_err or f"0/{len(sentinels)} sentinels visible"))
            ok = False

        try:
            adapter.consolidate()
            report.append(("consolidate", "OK", "callable (may be a no-op)"))
        except Exception as e:
            # consolidate has no NotImplementedError contract, so an exception is
            # ambiguous (missing endpoint vs real failure) — informational only
            report.append(("consolidate", "error", str(e) or e.__class__.__name__))
        try:
            adapter.supersede(docs[0]["id"], docs[0]["content"] + " (superseded)")
            report.append(("supersede", "OK", ""))
        except NotImplementedError as e:
            report.append(("supersede", "unsupported", str(e) or e.__class__.__name__))
        except Exception as e:
            report.append(("supersede", "FAIL", str(e) or e.__class__.__name__))
            ok = False  # adapter claims supersede but it errors — a real defect

    print(f"doctor [{adapter.name}] status={adapter.status}")
    for name, verdict, note in report:
        print(f"  {name:>18}  {verdict:<12} {note}")
    sys.exit(0 if ok else 1)


def cmd_replay(args):
    try:
        from .render.tui import ReplayApp
    except ImportError:
        sys.exit("replay needs textual — pip install 'memeval[tui]'")
    evs = read_events(args.run)
    if not evs:
        sys.exit(f"empty run log: {args.run}")
    ReplayApp(evs).run()


def cmd_chart(args):
    try:
        from .render.charts import render_ab_chart, render_chart
    except ImportError:
        sys.exit("chart needs matplotlib — pip install 'memeval[charts]'")
    try:
        if args.ab:
            out = render_ab_chart(args.ab[0], args.ab[1], out=args.out, png=args.png)
        elif len(args.run) == 1:
            out = render_chart(args.run[0], out=args.out, png=args.png)
        else:
            sys.exit("pass one run file, or --ab RUN_A RUN_B")
    except ValueError as e:
        sys.exit(str(e))
    print(f"wrote {out}")


def _add_common(p, gold=True):
    p.add_argument("--corpus", required=True)
    if gold:
        p.add_argument("--gold", required=True)
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--repeat", type=int, default=1)
    _add_run_opts(p)


def _add_run_opts(p):
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--ready-timeout", type=float, default=60.0)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="memeval",
        description="Evaluate an agent memory layer: rank scoring, adversarial "
                    "consolidation checks, recorded runs with replay and charts.")
    p.add_argument("--version", action="version", version=f"memeval {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("rank", help="load a corpus, run gold queries, score the ranks")
    r.add_argument("--adapter", required=True)
    _add_common(r)
    r.set_defaults(func=cmd_rank)

    ab = sub.add_parser("ab", help="same rank eval against two adapters, plus a delta")
    ab.add_argument("--a", required=True)
    ab.add_argument("--b", required=True)
    _add_common(ab)
    ab.set_defaults(func=cmd_ab)

    adv = sub.add_parser("adversarial", help="test consolidation for fabrication and self-heal")
    adv.add_argument("--adapter", required=True)
    adv.add_argument("--probe", required=True)
    adv.add_argument("--no-consolidate", action="store_true")
    _add_run_opts(adv)
    adv.set_defaults(func=cmd_adversarial)

    runs = sub.add_parser("runs", help="list recorded runs")
    runs.add_argument("--runs-dir", default="runs")
    runs.set_defaults(func=cmd_runs)

    ad = sub.add_parser("adapters", help="list adapters and their status")
    ad.set_defaults(func=cmd_adapters)

    doc = sub.add_parser("doctor", help="conformance-check an adapter before trusting it "
                                        "(writes sentinel docs — use a scratch namespace)")
    doc.add_argument("--adapter", required=True)
    doc.add_argument("--ready-timeout", type=float, default=60.0)
    doc.set_defaults(func=cmd_doctor)

    rp = sub.add_parser("replay", help="step through a recorded run in a TUI")
    rp.add_argument("run", help="path to a runs/*.jsonl file")
    rp.set_defaults(func=cmd_replay)

    ch = sub.add_parser("chart", help="render charts from a recorded run")
    ch.add_argument("run", nargs="*", help="one run file, or none with --ab")
    ch.add_argument("--ab", nargs=2, metavar=("RUN_A", "RUN_B"))
    ch.add_argument("-o", "--out", default=None)
    ch.add_argument("--png", action="store_true")
    ch.set_defaults(func=cmd_chart)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""rubric — score a local model against a frozen task file.

The shape of task this exists for: the answer is generated prose, several outputs
are acceptable, a constraint has to hold (answer stays in Korean), and the reference
is a rewrite rather than a label. `ollama-bench -decision` grades choices; this
grades answers.

Usage
  python rubric.py run --tasks tasks.jsonl --models gemma4:e2b,ene-v4
  python rubric.py run --tasks tasks.jsonl --models gemma4:e2b --limit 20 --show-failures 3
  python rubric.py run --tasks /path/to/test_set_v4.jsonl --fake gemma4:e2b
  python rubric.py list --tasks tasks.jsonl
  python rubric.py rescore --results results-v4.json --tasks tasks.jsonl --out results-v4.1.json

--fake runs the whole pipeline with synthetic answers so the scorer and the table
can be exercised without Ollama. It prints a banner, because those numbers are not
measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import scoring
from ollama_http import Client, FakeForTasks


# ------------------------------------------------------------------ helpers

def load_tasks(path: Path, limit: int = 0, category: str | None = None,
               ids: set[str] | None = None) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if category and item.get("category") != category:
                continue
            if ids and item.get("id") not in ids:
                continue
            rows.append(item)
    return rows[:limit] if limit else rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Normal approximation collapses at 0.98 and 208 items."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def pct(x: float) -> str:
    return f"{100 * max(0.0, min(1.0, x)):.1f}%"


def bar(frac: float, width: int = 20) -> str:
    filled = round(width * frac)
    return "#" * filled + "." * (width - filled)


def run_one(ask, item: dict, rubric_mode: str = "skip", judge_fn=None) -> dict:
    t0 = time.monotonic()
    try:
        answer, secs = ask(item)
        err = ""
    except Exception as e:  # a model error is a failed item, not a crashed run
        answer, secs, err = "", time.monotonic() - t0, f"{type(e).__name__}: {e}"
    res = scoring.score(item, answer, rubric_mode, judge_fn=judge_fn) if not err else {
        "id": item.get("id"), "category": item.get("category"),
        "passed": False, "needs_judge": False, "disagreement": None,
        "checks": [{"check": "run", "passed": False, "reason": err}],
    }
    res.update({
        "answer": answer,
        "seconds": round(secs, 3),
        "expected": item.get("fix", ""),
        "prompt": item.get("prompt", ""),
        "error": err,
    })
    return res


def make_judge(args, fake: bool = False):
    """Build (judge_fn, metadata) from CLI flags, or (None, None) when --judge is unset."""
    model = getattr(args, "judge", "") or ""
    if not model:
        return None, None
    judge_think_str = getattr(args, "judge_think", "false")
    judge_think = {"default": None, "true": True, "false": False}[judge_think_str]
    show_fix = bool(getattr(args, "judge_show_fix", False))
    if fake:
        def fake_judge(answer: str, _instruction: str, item: dict) -> tuple[bool, str]:
            h = hashlib.sha256((model + "|" + item.get("id", "") + "|" + answer).encode()).digest()
            passed = h[0] % 10 < 7
            return passed, f"fake judge ({'pass' if passed else 'fail'})"
        return fake_judge, {
            "model": model, "digest": None, "think": judge_think_str,
            "show_fix": show_fix, "fake": True,
        }

    jhost = getattr(args, "judge_host", "") or args.host
    jclient = Client(jhost, timeout=args.timeout)
    jenv = jclient.environment([model])
    if jenv["ollama_version"] is None:
        sys.exit(f"judge host unreachable: {jhost}")

    def real_judge(answer: str, instruction: str, item: dict) -> tuple[bool, str]:
        try:
            return scoring.judge(
                answer, instruction, item, client=jclient, model=model,
                think=judge_think, show_fix=show_fix)
        except Exception as e:
            return False, f"judge error: {type(e).__name__}: {e}"

    meta = {
        "model": model,
        "digest": jenv["model_digests"].get(model),
        "think": judge_think_str,
        "show_fix": show_fix,
    }
    return real_judge, meta


# -------------------------------------------------------------------- report

def report(model: str, results: list[dict], show_failures: int,
           show_disagreements: int = 0) -> None:
    n = len(results)
    k = sum(1 for r in results if r["passed"])
    lo, hi = wilson(k, n)
    lat = sorted(r["seconds"] for r in results if not r["error"])
    p50 = lat[len(lat) // 2] if lat else 0.0

    print(f"\n  {model}")
    print(f"  {'=' * 62}")
    print(f"  n={n}  pass={k}/{n}  {pct(k / n) if n else 'n/a'}"
          f"  95% CI {pct(lo)}–{pct(hi)}  p50 {p50:.2f}s")
    nj = sum(1 for r in results if r.get("needs_judge"))
    if nj:
        print(f"  {nj} of {n} items carry a rubric that no string check can grade"
              f" — the judge is the remaining work")

    judged = [r for r in results if any(c.get("check") == "rubric" for c in r.get("checks") or [])]
    disagreements = [r for r in judged if scoring.disagreement(r)]
    if judged:
        mp_jf = sum(1 for r in disagreements if scoring.disagreement(r) == "mech_pass_judge_fail")
        mf_jp = sum(1 for r in disagreements if scoring.disagreement(r) == "mech_fail_judge_pass")
        agree = len(judged) - len(disagreements)
        print(f"  judge: {len(judged)} items  mech/judge agree {agree}/{len(judged)} ({pct(agree / len(judged))})"
              f"  disagreements: {mp_jf} mech-pass/judge-fail, {mf_jp} mech-fail/judge-pass")

    cats: dict[str, list[dict]] = {}
    for r in results:
        cats.setdefault(r.get("category") or "-", []).append(r)
    if len(cats) > 1:
        print()
        for c in sorted(cats):
            rows = cats[c]
            kk = sum(1 for r in rows if r["passed"])
            print(f"  {c:<14} {kk:>3}/{len(rows):<4} {bar(kk / len(rows))}  {pct(kk / len(rows))}")

    if show_failures:
        fails = [r for r in results if not r["passed"]][:show_failures]
        if fails:
            print("\n  failures")
            for r in fails:
                failed = ", ".join(f"{c['check']}: {c['reason']}"
                                   for c in r["checks"] if not c["passed"])
                print(f"\n  [{r['id']}] {r.get('category') or ''}")
                print(f"    asked    {r['prompt'][:110]}")
                print(f"    expected {r['expected'][:110]}")
                print(f"    answered {r['answer'][:160].replace(chr(10), ' / ')}")
                print(f"    failed   {failed}")

    if show_disagreements and disagreements:
        print("\n  disagreements")
        for r in disagreements[:show_disagreements]:
            kind = scoring.disagreement(r)
            mech_parts = [f"{c['check']}={'ok' if c['passed'] else 'FAIL'} ({c['reason']})"
                          for c in r["checks"] if c["check"] != "rubric"]
            jcheck = next(c for c in r["checks"] if c["check"] == "rubric")
            print(f"\n  [{r['id']}] {kind}")
            print(f"    asked    {r['prompt'][:110]}")
            print(f"    answered {r['answer'][:160].replace(chr(10), ' / ')}")
            print(f"    mech     {'; '.join(mech_parts)}")
            print(f"    judge    {'ok' if jcheck['passed'] else 'FAIL'} ({jcheck['reason']})")


def report_comparison(out: dict[str, list[dict]]) -> None:
    """Markdown comparison table across models."""
    print("\n  comparison")
    print("  | model | pass | rate | 95% CI | correction | correction_ok | p50 | disagreements |")
    print("  | --- | --- | --- | --- | --- | --- | --- | --- |")
    for model, rows in out.items():
        n = len(rows)
        k = sum(1 for r in rows if r["passed"])
        lo, hi = wilson(k, n)
        lat = sorted(r["seconds"] for r in rows if not r.get("error"))
        p50 = lat[len(lat) // 2] if lat else 0.0
        corr = [r for r in rows if r.get("category") == "correction"]
        ok = [r for r in rows if r.get("category") == "correction_ok"]
        c_str = f"{sum(1 for r in corr if r['passed'])}/{len(corr)}" if corr else "-"
        ok_str = f"{sum(1 for r in ok if r['passed'])}/{len(ok)}" if ok else "-"
        dis = sum(1 for r in rows if scoring.disagreement(r))
        print(f"  | {model} | {k}/{n} | {pct(k / n) if n else 'n/a'} | {pct(lo)}-{pct(hi)} | "
              f"{c_str} | {ok_str} | {p50:.2f}s | {dis} |")


# ---------------------------------------------------------------------- main

def cmd_run(args) -> None:
    tasks = load_tasks(Path(args.tasks), args.limit, args.category,
                       set(args.id) if args.id else None)
    if not tasks:
        sys.exit("no tasks matched")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.fake:
        if not models:
            models = [args.fake]
        args.progress = False
        print("\n  FAKE MODE — synthetic answers. The plumbing is real, the numbers are not.",
              file=sys.stderr)

    judge_fn, judge_meta = make_judge(args, fake=bool(args.fake))
    rubric_mode = "judge" if judge_fn else args.rubric
    show_dis = args.show_disagreements if args.show_disagreements >= 0 else (5 if judge_fn else 0)

    system = Path(args.system).read_text(encoding="utf-8").strip() if args.system else None
    think = {"default": None, "true": True, "false": False}[args.think]
    settings = {"system": args.system or None, "think": args.think, "temperature": 0.0}
    environment = None
    if not args.fake:
        environment = Client(args.host, timeout=args.timeout).environment(models)
        digests = ", ".join(f"{m}@{(d or '?')[:12]}" for m, d in environment["model_digests"].items())
        print(f"  settings: system={settings['system']} think={args.think} temperature=0",
              file=sys.stderr)
        print(f"  ollama {environment['ollama_version'] or '?'}  models: {digests}", file=sys.stderr)
        if judge_meta:
            print(f"  judge: {judge_meta['model']}@{(judge_meta['digest'] or '?')[:12]} "
                  f"think={judge_meta['think']} show_fix={judge_meta['show_fix']}", file=sys.stderr)

    out = {}
    for model in models:
        if args.fake:
            fake = FakeForTasks(model, salt=args.fake_salt)
            ask = fake.answer
        else:
            client = Client(args.host, timeout=args.timeout)
            ask = lambda item, m=model: client.generate(
                m, item["prompt"], system=system, think=think)

        results = []
        if args.concurrency > 1:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                for i, res in enumerate(ex.map(
                        lambda it: run_one(ask, it, rubric_mode, judge_fn=judge_fn), tasks), 1):
                    results.append(res)
                    if args.progress and i % 20 == 0:
                        print(f"  {model}: {i}/{len(tasks)}", file=sys.stderr)
        else:
            for i, item in enumerate(tasks, 1):
                results.append(run_one(ask, item, rubric_mode, judge_fn=judge_fn))
                if args.progress and i % 20 == 0:
                    print(f"  {model}: {i}/{len(tasks)}", file=sys.stderr)

        out[model] = results
        report(model, results, args.show_failures, show_disagreements=show_dis)

    if len(out) >= 2:
        report_comparison(out)

    if args.out:
        payload = {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tasks": str(args.tasks), "fake": bool(args.fake), "settings": settings,
            "environment": environment, "results": out,
        }
        if judge_meta:
            payload["judge"] = judge_meta
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\n  wrote {args.out}")
    print()


def cmd_rescore(args) -> None:
    """Re-grade the answers stored in a --out file against a task file. No model calls for
    answers, so a grader, task-file, or judge change can be compared on the same answers."""
    src = json.loads(Path(args.results).read_text(encoding="utf-8"))
    all_tasks = {t["id"]: t for t in load_tasks(Path(args.tasks))}
    filtered = load_tasks(Path(args.tasks), args.limit, args.category,
                          set(args.id) if args.id else None)
    allowed_ids = {t["id"] for t in filtered}
    if not allowed_ids:
        sys.exit("no tasks matched")

    judge_fn, judge_meta = make_judge(args, fake=bool(src.get("fake")))
    rubric_mode = "judge" if judge_fn else args.rubric
    show_dis = args.show_disagreements if args.show_disagreements >= 0 else (5 if judge_fn else 0)
    if judge_meta and not judge_meta.get("fake"):
        print(f"  judge: {judge_meta['model']}@{(judge_meta['digest'] or '?')[:12]} "
              f"think={judge_meta['think']} show_fix={judge_meta['show_fix']}", file=sys.stderr)

    out = {}
    for model, rows in src["results"].items():
        selected_rows = []
        for old in rows:
            if old["id"] not in all_tasks:
                sys.exit(f"{old['id']} is in {args.results} but not in {args.tasks}")
            if old["id"] in allowed_ids:
                selected_rows.append(old)

        total_rubric = sum(
            1 for old in selected_rows
            if not old.get("error") and all_tasks[old["id"]].get("rubric")
        )
        judged_done = 0
        results = []
        for old in selected_rows:
            item = all_tasks[old["id"]]
            res = run_one(
                lambda _it, a=old["answer"], s=old["seconds"]: (a, s),
                item, rubric_mode, judge_fn=judge_fn,
            ) if not old.get("error") else dict(old)
            if item.get("rubric") and not old.get("error"):
                res["rubric"] = item["rubric"]
                if judge_fn:
                    judged_done += 1
                    if args.progress and (judged_done % 10 == 0 or judged_done == total_rubric):
                        print(f"  judge {model}: {judged_done}/{total_rubric}", file=sys.stderr)
            results.append(res)

        changed = sum(1 for a, b in zip(selected_rows, results) if a["passed"] != b["passed"])
        out[model] = results
        report(model, results, args.show_failures, show_disagreements=show_dis)
        print(f"  {changed} item(s) changed verdict against {args.results}")

    if len(out) >= 2:
        report_comparison(out)

    if args.out:
        payload = {
            **{k: v for k, v in src.items() if k != "results"},
            "rescored": {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "from": str(args.results),
                         "tasks": str(args.tasks)},
            "results": out,
        }
        if judge_meta:
            payload["judge"] = judge_meta
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"\n  wrote {args.out}")
    print()


# --------------------------------------------------------------- calibration

def _judged_records(src: dict, tasks: dict[str, dict] | None = None) -> list[dict]:
    """Collect every judged rubric row across models with mech and judge verdicts."""
    records = []
    for model, rows in (src.get("results") or {}).items():
        for r in rows:
            checks = r.get("checks") or []
            jcheck = next((c for c in checks if c.get("check") == "rubric"), None)
            if jcheck is None:
                continue
            mech_checks = [c for c in checks if c.get("check") != "rubric"]
            mech_ok = all(c.get("passed") for c in mech_checks) if mech_checks else True
            rubric_text = r.get("rubric") or ((tasks or {}).get(r["id"]) or {}).get("rubric", "")
            records.append({
                "model": model,
                "id": r["id"],
                "prompt": r.get("prompt", ""),
                "rubric": rubric_text,
                "answer": r.get("answer", ""),
                "mech_pass": mech_ok,
                "judge_pass": bool(jcheck.get("passed")),
                "judge_reason": jcheck.get("reason", ""),
                "disagreement": scoring.disagreement(r),
            })
    return records


def _pick_round_robin(buckets: dict[str, list[dict]], limit: int, picked_keys: set[tuple[str, str]]) -> list[dict]:
    out: list[dict] = []
    models = list(buckets.keys())
    idx = {m: 0 for m in models}
    while len(out) < limit:
        advanced = False
        for m in models:
            if len(out) >= limit:
                break
            rows = buckets[m]
            while idx[m] < len(rows):
                rec = rows[idx[m]]
                idx[m] += 1
                key = (rec["model"], rec["id"])
                if key not in picked_keys:
                    picked_keys.add(key)
                    out.append(rec)
                    advanced = True
                    break
        if not advanced:
            break
    return out


def select_calibration_sample(src: dict, n: int = 30, seed: int = 42,
                              tasks: dict[str, dict] | None = None) -> list[dict]:
    """Select n judged items across models: up to n//2 disagreements, then balanced both-pass
    and both-fail strata."""
    records = _judged_records(src, tasks=tasks)
    if not records or n <= 0:
        return []

    rng = random.Random(seed)
    models = list((src.get("results") or {}).keys())

    def group(pred):
        b = {m: [r for r in records if r["model"] == m and pred(r)] for m in models}
        for rows in b.values():
            rng.shuffle(rows)
        return b

    picked_keys: set[tuple[str, str]] = set()
    picked = _pick_round_robin(group(lambda r: r["disagreement"] is not None), n // 2, picked_keys)
    rem = n - len(picked)
    picked.extend(_pick_round_robin(
        group(lambda r: r["disagreement"] is None and r["mech_pass"] and r["judge_pass"]),
        rem // 2, picked_keys))
    picked.extend(_pick_round_robin(
        group(lambda r: r["disagreement"] is None and not r["mech_pass"] and not r["judge_pass"]),
        n - len(picked), picked_keys))
    if len(picked) < n:
        picked.extend(_pick_round_robin(group(lambda _r: True), n - len(picked), picked_keys))
    rng.shuffle(picked)
    return picked


def _load_task_map(src: dict, tasks_arg: str) -> dict[str, dict]:
    path = Path(tasks_arg) if tasks_arg else (Path(src["tasks"]) if src.get("tasks") else None)
    if path and path.exists():
        return {t["id"]: t for t in load_tasks(path)}
    return {}


def print_calibration_report(joined: list[tuple[bool, dict]]) -> None:
    """Print Human vs Judge and Human vs Mechanical 2x2 matrices and Cohen's kappa."""
    if not joined:
        print("\n  no matching labeled items to score")
        return
    hj = scoring.cohen_kappa([(hp, rec["judge_pass"]) for hp, rec in joined])
    hm = scoring.cohen_kappa([(hp, rec["mech_pass"]) for hp, rec in joined])
    print(f"\n  calibration ({hj['n']} labeled items)")
    for label, stats in [("human vs judge", hj), ("human vs mech ", hm)]:
        print(f"  {label}  agree {stats['agree']}/{stats['n']} ({pct(stats['agreement'])})"
              f"  kappa={stats['kappa']:.3f}  "
              f"[TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} TN={stats['tn']}]")


def cmd_calibrate(args) -> None:
    src = json.loads(Path(args.results).read_text(encoding="utf-8"))
    tasks = _load_task_map(src, args.tasks)
    records = _judged_records(src, tasks=tasks)
    if not records:
        sys.exit(f"no judged rubric items in {args.results} (run rescore --judge first)")
    lookup = {(r["model"], r["id"]): r for r in records}

    if args.labels:
        rows = [json.loads(line) for line in Path(args.labels).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        joined = [(bool(r["human_pass"]), lookup[(r["model"], r["id"])])
                  for r in rows if (r.get("model"), r.get("id")) in lookup and "human_pass" in r]
        print_calibration_report(joined)
        print()
        return

    sample = select_calibration_sample(src, n=args.sample, seed=args.seed, tasks=tasks)
    if args.interactive:
        out_path = Path(args.out) if args.out else Path("calibration.jsonl")
        existing: dict[tuple[str, str], dict] = {}
        if out_path.exists():
            for line in out_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    existing[(row["model"], row["id"])] = row
        todo = [rec for rec in sample if (rec["model"], rec["id"]) not in existing]
        print(f"\n  blind calibration: {len(sample)} sampled, {len(existing)} already labeled, "
              f"{len(todo)} remaining")
        print("  enter [p]ass, [f]ail, [s]kip, or [q]uit (optional note after space, e.g. 'f invented fix')")
        with out_path.open("a", encoding="utf-8") as fh:
            for idx, rec in enumerate(todo, 1):
                print(f"\n  [{idx}/{len(todo)}] {rec['model']} | {rec['id']}")
                print(f"  Prompt: {rec['prompt']}")
                print(f"  Rubric: {rec['rubric']}")
                print(f"  Answer:\n{rec['answer']}\n")
                while True:
                    try:
                        raw = input("  verdict [p/f/s/q]: ").strip()
                    except EOFError:
                        raw = "q"
                    head, _, note = raw.partition(" ")
                    cmd = head.lower()
                    if cmd in ("p", "pass", "f", "fail"):
                        hp = cmd in ("p", "pass")
                        entry = {"model": rec["model"], "id": rec["id"],
                                 "human_pass": hp, "note": note.strip()}
                        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        fh.flush()
                        existing[(rec["model"], rec["id"])] = entry
                        break
                    if cmd in ("s", "skip"):
                        break
                    if cmd in ("q", "quit"):
                        joined = [(bool(v["human_pass"]), lookup[k])
                                  for k, v in existing.items() if k in lookup and "human_pass" in v]
                        print_calibration_report(joined)
                        print(f"\n  saved labels to {out_path}\n")
                        return
        joined = [(bool(v["human_pass"]), lookup[k])
                  for k, v in existing.items() if k in lookup and "human_pass" in v]
        print_calibration_report(joined)
        print(f"\n  saved labels to {out_path}\n")
        return

    dis_n = sum(1 for r in sample if r["disagreement"] is not None)
    bp_n = sum(1 for r in sample if r["disagreement"] is None and r["mech_pass"] and r["judge_pass"])
    bf_n = sum(1 for r in sample if r["disagreement"] is None and not r["mech_pass"] and not r["judge_pass"])
    by_model: dict[str, int] = {}
    for r in sample:
        by_model[r["model"]] = by_model.get(r["model"], 0) + 1
    print(f"\n  calibration sample: {len(sample)} items (seed={args.seed})")
    print(f"  strata: disagreements={dis_n}  both_pass={bp_n}  both_fail={bf_n}")
    print(f"  models: {', '.join(f'{m}={c}' for m, c in by_model.items())}")
    if args.out:
        blind = [{"model": r["model"], "id": r["id"], "prompt": r["prompt"],
                  "rubric": r["rubric"], "answer": r["answer"]} for r in sample]
        Path(args.out).write_text(
            "\n".join(json.dumps(b, ensure_ascii=False) for b in blind) + "\n", encoding="utf-8")
        print(f"  wrote blind sample to {args.out}")
    print()


def cmd_list(args) -> None:
    tasks = load_tasks(Path(args.tasks), args.limit, args.category)
    cats: dict[str, int] = {}
    types: dict[str, int] = {}
    for t in tasks:
        cats[t.get("category") or "-"] = cats.get(t.get("category") or "-", 0) + 1
        ct = (t.get("check") or {}).get("type") or "none"
        types[ct] = types.get(ct, 0) + 1
    print(f"\n  {len(tasks)} tasks from {args.tasks}")
    for c, n in sorted(cats.items()):
        print(f"    {c:<16}{n}")
    print("  check types")
    for c, n in sorted(types.items()):
        print(f"    {c:<16}{n}")
    print(f"  with rubric field: {sum(1 for t in tasks if t.get('rubric'))}")
    print(f"  requiring Korean:  {sum(1 for t in tasks if t.get('require_korean'))}\n")


def _add_judge_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--judge", default="", metavar="MODEL",
                        help="Ollama model used to grade items carrying a rubric")
    parser.add_argument("--judge-host", default="",
                        help="Ollama host for the judge (defaults to --host)")
    parser.add_argument("--judge-think", default="false", choices=["default", "true", "false"],
                        help="think setting for the judge model (default: false)")
    parser.add_argument("--judge-show-fix", action="store_true",
                        help="include reference fix in judge prompt (off by default)")
    parser.add_argument("--show-disagreements", type=int, default=-1,
                        help="max disagreement rows to print (-1: 5 when --judge is set, else 0)")


def main() -> None:
    p = argparse.ArgumentParser(prog="rubric", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="score models against a task file")
    r.add_argument("--tasks", required=True)
    r.add_argument("--models", default="", help="comma separated Ollama model names")
    r.add_argument("--host", default="http://127.0.0.1:11434")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--category", default=None)
    r.add_argument("--id", action="append", default=None)
    r.add_argument("--concurrency", type=int, default=1)
    r.add_argument("--timeout", type=float, default=120.0)
    r.add_argument("--system", default="", metavar="FILE",
                   help="file whose text is sent as the system prompt on every item")
    r.add_argument("--think", default="default", choices=["default", "true", "false"],
                   help="value of Ollama's `think` field; default leaves it out. Recorded in --out.")
    r.add_argument("--show-failures", type=int, default=0)
    r.add_argument("--rubric", default="skip", choices=["skip", "fail"],
                   help="skip: rubric-only items are ungraded (default); fail: strict")
    _add_judge_args(r)
    r.add_argument("--out", default="")
    r.add_argument("--no-progress", dest="progress", action="store_false")
    r.add_argument("--fake", default="", metavar="MODEL",
                   help="run with synthetic answers, no server needed")
    r.add_argument("--fake-salt", default="")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("rescore", help="re-grade the answers in a --out file")
    s.add_argument("--results", required=True, help="a file written by `run --out`")
    s.add_argument("--tasks", required=True)
    s.add_argument("--host", default="http://127.0.0.1:11434")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--category", default=None)
    s.add_argument("--id", action="append", default=None)
    s.add_argument("--timeout", type=float, default=120.0)
    s.add_argument("--show-failures", type=int, default=0)
    s.add_argument("--rubric", default="skip", choices=["skip", "fail"])
    _add_judge_args(s)
    s.add_argument("--out", default="")
    s.add_argument("--no-progress", dest="progress", action="store_false")
    s.set_defaults(fn=cmd_rescore)

    c = sub.add_parser("calibrate", help="sample blind items for human labeling and compute kappa")
    c.add_argument("--results", required=True, help="a judged file written with --judge --out")
    c.add_argument("--tasks", default="", help="optional task JSONL if rubric text is not in results")
    c.add_argument("--sample", type=int, default=30)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--out", default="", help="JSONL file for blind sample or interactive labels")
    c.add_argument("--interactive", action="store_true")
    c.add_argument("--labels", default="", help="score an existing labeled JSONL file")
    c.set_defaults(fn=cmd_calibrate)

    l = sub.add_parser("list", help="describe a task file without running anything")
    l.add_argument("--tasks", required=True)
    l.add_argument("--limit", type=int, default=0)
    l.add_argument("--category", default=None)
    l.set_defaults(fn=cmd_list)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

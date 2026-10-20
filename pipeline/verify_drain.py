import argparse
import collections
import json
import re
import sys
from pathlib import Path

import drain_templates as d
from common import CONDITIONS, DERIVED_DIR, PipelineError, sha256_bytes

FORBIDDEN_RE = re.compile("|".join(re.escape(t) for t in
                                   ("stress", "experimental", "inject", "toxi", "chaos", "fault") + CONDITIONS))
LEAK_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|\b[0-9a-f]{12}\b|\d{10,}")
RUN_FIELDS = ["ts", "req", "service", "template_id"]
CHECKS = ("config", "files", "encoding", "lines", "alignment", "template_ids", "template_text", "unk_train",
          "state", "determinism", "manifest")


def check_template_text(view, templates):
    problems = []
    for t in templates:
        text = t["template"].lower()
        for term in FORBIDDEN_RE.findall(text):
            problems.append(("template_text", f"{view} template {t['id']} contains {term!r}"))
        if LEAK_RE.search(text):
            problems.append(("template_text", f"{view} template {t['id']} contains an id, req or ts"))
    return problems


def read_run_file(path):
    problems = []
    content = path.read_bytes()
    if b"\r" in content or (content and not content.endswith(b"\n")):
        problems.append(("encoding", f"{path.name}: CR or missing final LF"))
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return [("encoding", f"{path.name}: not UTF-8")], []
    return problems, [json.loads(line) for line in lines]


def check_manifest(base):
    listed = {}
    for line in (base / "manifest.sha256").read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        listed[name] = digest
    present = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() and p.name != "manifest.sha256"}
    problems = []
    if present != set(listed):
        problems.append(("manifest", f"manifest and files differ: {sorted(present ^ set(listed))[:5]}"))
    problems += [("manifest", f"sha256 differs: {name}") for name in sorted(present & set(listed))
                 if sha256_bytes((base / name).read_bytes()) != listed[name]]
    return problems


def verify_view(derived_dir, base, view, splits, config):
    problems = []
    templates = json.loads((base / view / "templates.json").read_text(encoding="utf-8"))
    known = {t["id"] for t in templates}
    problems += check_template_text(view, templates)
    runs_dir = base / view / "runs"
    names = {p.name for p in runs_dir.iterdir()}
    if names != {f"{run_id}.jsonl" for run_id in splits}:
        problems.append(("files", f"{view}: run files differ from splits.csv"))
    texts, saved = {}, {}
    lines, unk = collections.Counter(), collections.Counter()
    events = collections.defaultdict(collections.Counter)
    for run_id in sorted(splits):
        _, records = d.load_run(derived_dir, view, run_id)
        texts[run_id] = [d.render(view, r) for r in records]
        path = runs_dir / f"{run_id}.jsonl"
        if not path.exists():
            continue
        file_problems, rows = read_run_file(path)
        problems += file_problems
        if len(rows) != len(records):
            problems.append(("lines", f"{view}/{run_id[:8]}: {len(rows)} lines, sanitized has {len(records)}"))
        if any(list(row) != RUN_FIELDS for row in rows) or any(
                (row["ts"], row["req"], row["service"]) != (r["ts"], r["req"], r["service"])
                for row, r in zip(rows, records)):
            problems.append(("alignment", f"{view}/{run_id[:8]}: fields or order differ from the sanitized view"))
        saved[run_id] = [row.get("template_id") for row in rows]
        if set(saved[run_id]) - known - {d.UNK_ID}:
            problems.append(("template_ids", f"{view}/{run_id[:8]}: unknown template_id"))
        split = splits[run_id]
        lines[split] += len(rows)
        unk[split] += saved[run_id].count(d.UNK_ID)
        if split == "train":
            for tid, r in zip(saved[run_id], records):
                events[tid][r["event"]] += 1
    if unk["train"]:
        problems.append(("unk_train", f"{view}: {unk['train']} training lines without template"))

    expected = [(t["id"], t["template"], t["fit_lines"]) for t in templates]
    state = d.load_state((base / view / "state.bin").read_bytes(), config)
    if d.snapshot(state) != expected:
        problems.append(("state", f"{view}: reloaded state differs from templates.json"))
    train = {run_id: t for run_id, t in texts.items() if splits[run_id] == "train"}
    fits = [d.fit(train, config) for _ in range(2)]
    assignments = []
    for miner in fits:
        cache = {}
        assignments.append({run_id: d.assign(miner, texts[run_id], cache) for run_id in sorted(texts)})
    if d.snapshot(fits[0]) != d.snapshot(fits[1]) or assignments[0] != assignments[1]:
        problems.append(("determinism", f"{view}: two in-memory fits differ"))
    if d.snapshot(fits[0]) != expected or assignments[0] != saved:
        problems.append(("determinism", f"{view}: in-memory fit differs from the saved output"))

    mixed = [(t["id"], t["template"], dict(events[t["id"]])) for t in templates if len(events[t["id"]]) > 1]
    info = {"templates": templates, "wildcards": sum("<*>" in t["template"] for t in templates),
            "lines": lines, "unk": unk, "mixed": mixed}
    return problems, info


def verify(derived_dir, variant):
    base = derived_dir / "drain" / variant
    splits = d.read_splits(derived_dir)
    config = d.miner_config(variant)
    problems = []
    record = json.loads((base / "drain_config.json").read_text(encoding="utf-8"))
    if (record["variant"] != variant or record["official"] != (variant == d.OFFICIAL_VARIANT)
            or record["template_miner_config"] != json.loads(json.dumps(d.effective_config(config)))):
        problems.append(("config", "drain_config.json does not match the variant configuration"))
    if {p.name for p in base.iterdir()} != {"drain_config.json", "manifest.sha256", *d.VIEWS}:
        problems.append(("files", f"unexpected entries in {base}"))
    info = {"config": record}
    for view in d.VIEWS:
        view_problems, info[view] = verify_view(derived_dir, base, view, splits, config)
        problems += view_problems
    problems += check_manifest(base)
    return problems, info


def rate(n, total):
    return f"{n}/{total} ({100 * n / total:.3f}%)" if total else "0/0"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(d.VARIANTS), action="append")
    parser.add_argument("--derived", type=Path, default=DERIVED_DIR)
    parser.add_argument("--list", action="store_true", help="print every template")
    args = parser.parse_args(argv)
    failed = False
    for variant in args.variant or list(d.VARIANTS):
        try:
            problems, info = verify(args.derived, variant)
        except (PipelineError, OSError, KeyError, ValueError) as error:
            print(f"== {variant} ==\nERROR: {error}", file=sys.stderr)
            failed = True
            continue
        cfg = info["config"]
        print(f"== {variant}{' (official)' if cfg['official'] else ''}: drain3 {cfg['packages']['drain3']}, "
              f"sim_th={cfg['template_miner_config']['drain_sim_th']} ==")
        by_check = collections.defaultdict(list)
        for check, detail in problems:
            by_check[check].append(detail)
        for check in CHECKS:
            print(f"[{'FAIL' if by_check[check] else 'OK'}] {check}")
            for detail in by_check[check][:10]:
                print(f"    {detail}")
        for view in d.VIEWS:
            v = info[view]
            print(f"{view}: templates={len(v['templates'])} with_wildcard={v['wildcards']}")
            print("  unk: " + ", ".join(f"{s} {rate(v['unk'][s], v['lines'][s])}" for s in d.SPLITS))
            if args.list:
                for t in v["templates"]:
                    print(f"  [{t['id']}] train_lines={t['train_lines']} {t['template']}")
            print(f"  templates mixing events: {len(v['mixed'])}")
            for tid, text, events in v["mixed"]:
                print(f"    [{tid}] {text} <- {events}")
        print(f"result: {'FAIL' if problems else 'PASS'} ({len(problems)} problems)")
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

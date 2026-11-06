import collections
import hashlib
import json
import random
import unittest
import uuid

import common
import sanitize as v1
import sanitize_v2 as s
import split_runs_v2
import verify_sanitized_v2 as v
from test_pipeline import TempDirTest, quiet

ITEMS = [(c, r) for c in common.CONDITIONS_V2 for r in range(1, 16 if c == "baseline" else 6)]
RUNS_FIELDS = ["experiment_run_id", "condition", "repetition", "mode", "commit", "dirty", "jsonl_sha256", "status"]
ROUTES = (("GET", "/health", "health_checked", "Health request completed"),
          ("GET", "/products", "products_completed", "Product request completed"),
          ("POST", "/orders", "order_created", "Order request completed"))
FAILED = ("stresser_lookup_failed", "Dependency lookup failed")


def stamp(n, go_format=False):
    seconds, micro = divmod(n, 1000)
    base = f"2026-11-04T21:{seconds // 60:02d}:{seconds % 60:02d}"
    return f"{base}.{micro:03d}000000Z" if go_format else f"{base}.{micro:03d}000+00:00"


class Raw:
    """Raw lines in the shapes of official-v2: small lifecycle lines, final FastAPI lines and Go lines."""

    def __init__(self, run_id):
        self.run_id, self.lines, self.clock = run_id, [], 0

    def tick(self):
        self.clock += 1
        return self.clock

    def small(self, req, event, message, level="info", **extra):
        self.lines.append({"request_id": req, "experiment_run_id": self.run_id, "timestamp": stamp(self.tick()), "level": level,
                           "service": "application-api", "event": event, "message": message, **extra})

    def final(self, req, method, route, event, message, status, level="info", **extra):
        self.lines.append({"method": method, "route": route, "request_id": req, "experiment_run_id": self.run_id,
                           "status_code": status, "latency_ms": 160.125, "stress_level": 92,
                           "degradation_type": "none", "timestamp": stamp(self.tick()), "level": level,
                           "service": "application-api", "event": event, "message": message, **extra})

    def go(self, req, event="state_queried", message="state queried", level="info", error=None):
        line = {"timestamp": stamp(self.tick(), True), "level": level, "message": message, "service": "stresser-app",
                "event": event, "method": "GET", "route": "/api/state", "request_id": req, "experiment_run_id": self.run_id}
        if error:
            line["error"] = error
        else:
            line.update({"status_code": 200, "latency_ms": 62.129, "stress_level": 92, "cpu_workers": 1})
        self.lines.append(line)


def raw_events(run_id):
    raw = Raw(run_id)
    for number, (method, route, event, message) in enumerate(ROUTES):
        ok = f"ok-{number}"
        if number == 0:
            raw.small(ok, "dependency_connection_opened", "Dependency connection opened")
        raw.small(ok, "dependency_request_sent", "Dependency request sent")
        raw.go(ok)
        raw.small(ok, "dependency_response_received", "Dependency response received")
        raw.final(ok, method, route, event, message, 201 if route == "/orders" else 200)

        timed_out = f"timeout-{number}"
        raw.small(timed_out, "dependency_request_sent", "Dependency request sent")
        raw.small(timed_out, "dependency_response_failed", "Dependency response failed", "warning", error="ReadTimeout")
        raw.small(timed_out, "dependency_connection_closed", "Dependency connection closed")
        raw.final(timed_out, method, route, *FAILED, 503, "error", error_type="ReadTimeout")
        raw.go(timed_out)  # the Go logs after the FastAPI gave up: the canonical order must put it back in place

        deadline = f"deadline-{number}"
        raw.small(deadline, "dependency_request_sent", "Dependency request sent")
        raw.go(deadline, "state_request_deadline_exceeded", "State request deadline exceeded", "warn", "context deadline exceeded")
        raw.small(deadline, "dependency_response_received", "Dependency response received")
        raw.final(deadline, method, route, *FAILED, 503, "error", error_type="UnexpectedStatus")
    raw.small("cancel", "dependency_request_sent", "Dependency request sent")
    raw.small("cancel", "dependency_response_failed", "Dependency response failed", "warning", error="ReadTimeout")
    raw.small("cancel", "dependency_connection_closed", "Dependency connection closed")
    raw.final("cancel", "GET", "/products", *FAILED, 503, "error", error_type="ReadTimeout")
    raw.go("cancel", "state_request_canceled", "State request canceled", "warn", "context canceled")
    raw.final("r-404", "POST", "/orders", "product_not_found", "Product not found", 404, "warning")
    raw.final("r-422", "POST", "/orders", "request_rejected", "Request rejected", 422, "warning")
    return raw.lines


def raw_lines(run_id):
    return [json.dumps(e) for e in raw_events(run_id)]


def build_official(root, items=ITEMS, commit=common.SOURCE_COMMIT_V2, lines_of=raw_lines):
    rows = []
    for index, (condition, repetition) in enumerate(items, 1):
        run_id = str(uuid.UUID(int=index))
        content = "".join(line + "\n" for line in lines_of(run_id)).encode("utf-8")
        common.write_new(root / "raw" / f"{run_id}.jsonl", content)
        rows.append({"experiment_run_id": run_id, "condition": condition, "repetition": repetition, "mode": "official",
                     "commit": commit, "dirty": "False", "jsonl_sha256": common.sha256_bytes(content), "status": "valid"})
    rows.append({**rows[0], "experiment_run_id": str(uuid.uuid4()), "status": "invalid"})
    common.write_new(root / "control" / "runs.csv", common.csv_bytes(RUNS_FIELDS, rows))
    common.write_new(root / "control" / "manifest.sha256", b"fixture\n")
    return [r for r in rows if r["status"] == "valid"]


class SplitTest(TempDirTest):
    def test_counts_and_disjoint(self):
        assigned = split_runs_v2.assign_splits(build_official(self.official))
        ids = [a["experiment_run_id"] for a in assigned]
        self.assertEqual((len(ids), len(set(ids))), (50, 50))
        totals = collections.Counter(a["split"] for a in assigned)
        self.assertEqual(totals, {"train": 30, "validation": 10, "test": 10})
        for condition in common.CONDITIONS_V2:
            counts = [sum(a["condition"] == condition and a["split"] == split for a in assigned)
                      for split in split_runs_v2.SPLITS]
            self.assertEqual(counts, [9, 3, 3] if condition == "baseline" else [3, 1, 1])

    def test_deterministic_and_order_independent(self):
        runs = build_official(self.official)
        first = split_runs_v2.assign_splits(runs)
        reordered = list(runs)
        random.Random(7).shuffle(reordered)
        self.assertEqual(split_runs_v2.assign_splits(reordered), first)
        self.assertNotEqual(split_runs_v2.assign_splits(runs, seed=1), first)

    def test_seed_is_the_one_of_the_task(self):
        self.assertEqual(split_runs_v2.SPLIT_SEED, 20261104)
        record = split_runs_v2.split_record(split_runs_v2.assign_splits(build_official(self.official)))
        self.assertEqual(record["seed"], 20261104)
        self.assertEqual({k: len(x) for k, x in record["sets"].items()}, {"train": 30, "validation": 10, "test": 10})

    def test_rejects_v1_condition_or_missing_repetition(self):
        runs = build_official(self.official)
        with self.assertRaisesRegex(common.PipelineError, "cpu92"):
            split_runs_v2.assign_splits([r for r in runs if not (r["condition"] == "cpu92" and r["repetition"] == 3)])
        with self.assertRaisesRegex(common.PipelineError, "catalog"):
            split_runs_v2.assign_splits([{**r, "condition": "cpu80"} if r["condition"] == "cpu92" else r for r in runs])

    def test_v1_commit_is_refused(self):
        build_official(self.official, commit=common.SOURCE_COMMIT)
        self.assertEqual(quiet(split_runs_v2.main, self.args)[0], 1)

    def test_main_writes_and_refuses_overwrite(self):
        build_official(self.official)
        self.assertEqual(quiet(split_runs_v2.main, self.args)[0], 0)
        record = split_runs_v2.record_path(self.derived)
        before = record.read_bytes()
        code, err = quiet(split_runs_v2.main, self.args)
        self.assertEqual(code, 1)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual(record.read_bytes(), before)


class SanitizeTest(unittest.TestCase):
    RUN_ID = "3f1c2e0a-7b6d-4c5e-9f8a-1b2c3d4e5f60"

    def views(self):
        out = s.sanitize_run(raw_lines(self.RUN_ID), self.RUN_ID, "train")
        return {view: [json.loads(line) for line in content.decode("utf-8").splitlines()] for view, content in out.items()}

    def test_one_line_per_raw_line_with_text_and_meta_only(self):
        for records in self.views().values():
            self.assertEqual(len(records), len(raw_lines(self.RUN_ID)))
            for record in records:
                self.assertEqual(set(record), {"text", "meta"})
                self.assertEqual(tuple(record["meta"]), s.META_FIELDS)

    def test_controlled_text_is_exactly_the_c1_set(self):
        self.assertEqual({r["text"] for r in self.views()["controlada"]}, v.C1_TEXTS)

    def test_controlled_text_has_no_control_value(self):
        for record in self.views()["controlada"]:
            text = record["text"].lower()
            self.assertFalse(any(c.isdigit() for c in text), text)
            for word in ("stress", "cpu", "toxic", "latency", "status_code", "request_id", self.RUN_ID, "r-404", "ok-0"):
                self.assertNotIn(word, text)

    def test_warn_becomes_warning(self):
        events = {r["text"].split(" ")[2]: r["text"].split(" ")[1] for r in self.views()["controlada"]}
        self.assertEqual(events["state_request_canceled"], "warning")
        self.assertEqual(events["state_request_deadline_exceeded"], "warning")

    def test_renames(self):
        text = " ".join(r["text"] for r in self.views()["controlada"])
        self.assertIn("state-service", text)
        self.assertIn("dependency_lookup_failed", text)
        self.assertNotIn("stresser", text)

    def test_error_kept(self):
        event = {"service": "application-api", "level": "error", "event": FAILED[0], "message": FAILED[1], "method": "GET",
                 "route": "/products", "error_type": "UnexpectedStatus"}
        self.assertEqual(s.controlled_text(event), "application-api error dependency_lookup_failed Dependency lookup failed "
                                                   "GET /products error=UnexpectedStatus")

    def test_operational_is_controlled_plus_status_and_latency(self):
        views = self.views()
        with_values = 0
        for operational, controlled in zip(views["operacional"], views["controlada"]):
            base, value = v.split_operational(operational["text"])
            self.assertEqual(base, controlled["text"])
            self.assertEqual(operational["meta"], controlled["meta"])
            with_values += value is not None
        self.assertEqual(with_values, sum(e.get("status_code") is not None for e in raw_events(self.RUN_ID)))
        go = next(r["text"] for r in views["operacional"] if " state_queried " in r["text"])
        self.assertTrue(go.endswith(" GET /api/state status_code=200 latency_ms=62.129"), go)

    def test_operational_keeps_exact_latency_not_a_bucket(self):
        self.assertTrue(any(r["text"].endswith("latency_ms=160.125") for r in self.views()["operacional"]))

    def test_pseudonym_links_the_services_and_hides_the_id(self):
        expected = hashlib.sha256(f"{self.RUN_ID}:timeout-0".encode()).hexdigest()[:12]
        lines = [r for r in self.views()["controlada"] if r["meta"]["req"] == expected]
        self.assertEqual({r["meta"]["service"] for r in lines}, {"application-api", "state-service"})
        self.assertEqual(s.sanitize_run(raw_lines(self.RUN_ID), self.RUN_ID, "train")["controlada"],
                         s.sanitize_run(raw_lines(self.RUN_ID), self.RUN_ID, "train")["controlada"])

    def test_meta_has_set_and_run(self):
        out = s.sanitize_run(raw_lines(self.RUN_ID), self.RUN_ID, "test")
        record = json.loads(out["controlada"].decode("utf-8").splitlines()[0])
        self.assertEqual((record["meta"]["run_id"], record["meta"]["split"]), (self.RUN_ID, "test"))

    def test_canonical_order_undoes_the_log_inversion(self):
        req = v1.pseudonym(self.RUN_ID, "timeout-0")
        records = [r for r in self.views()["controlada"] if r["meta"]["req"] == req]
        self.assertEqual(records[-1]["meta"]["service"], "state-service")  # the file keeps the logged order
        by_order = sorted(records, key=lambda r: r["meta"]["event_order"])
        self.assertEqual([r["text"].split(" ")[2] for r in by_order],
                         ["dependency_request_sent", "state_queried", "dependency_response_failed",
                          "dependency_connection_closed", "dependency_lookup_failed"])
        self.assertEqual([r["meta"]["event_order"] for r in by_order], [0, 1, 2, 3, 4])

    def test_file_order_is_ts_with_ties_by_canonical_order(self):
        records = self.views()["controlada"]
        stamps = [r["meta"]["ts"] for r in records]
        self.assertEqual(stamps, sorted(stamps))
        same = "2026-11-04T21:00:00.000000+00:00"
        events = [{**e, "timestamp": same} for e in raw_events(self.RUN_ID) if e["request_id"] == "ok-1"][::-1]
        out = s.sanitize_run([json.dumps(e) for e in events], self.RUN_ID, "train")["controlada"].decode("utf-8")
        self.assertEqual([json.loads(line)["meta"]["event_order"] for line in out.splitlines()], [0, 1, 2, 3])

    def test_every_pair_has_a_rank(self):
        self.assertEqual({event for event, _ in s.SOURCE_PAIRS}, set(s.EVENT_RANKS))

    def test_repeated_event_in_a_request_stops(self):
        line = json.dumps(raw_events(self.RUN_ID)[1])
        with self.assertRaisesRegex(common.PipelineError, "share a position"):
            s.sanitize_run([line, line], self.RUN_ID, "train")

    def test_unmapped_pair_or_field_stops(self):
        event = {**raw_events(self.RUN_ID)[0], "event": "fault_injected", "message": "Fault injected"}
        with self.assertRaisesRegex(common.PipelineError, "unmapped pair"):
            s.sanitize_run([json.dumps(event)], self.RUN_ID, "train")
        with self.assertRaisesRegex(common.PipelineError, "unexpected fields"):
            s.inventory([json.dumps({**raw_events(self.RUN_ID)[0], "toxic": 1})], collections.Counter(), self.RUN_ID)

    def test_line_of_another_run_stops(self):
        with self.assertRaisesRegex(common.PipelineError, "another run"):
            s.sanitize_run(raw_lines(str(uuid.UUID(int=9))), self.RUN_ID, "train")


class PipelineTest(TempDirTest):
    def setUp(self):
        super().setUp()
        self.runs = build_official(self.official, ITEMS)
        self.lines = len(raw_lines(self.runs[0]["experiment_run_id"]))

    def run_pipeline(self):
        self.assertEqual(quiet(split_runs_v2.main, self.args)[0], 0)
        self.assertEqual(quiet(s.main, self.args)[0], 0)

    def problems(self):
        return {c for c, _ in v.verify(self.official, self.derived)[0]}

    def path(self, view="controlada", index=0):
        return v1.view_paths(self.derived, self.runs[index]["experiment_run_id"])[view]

    def tamper(self, transform, view="controlada", index=0):
        path = self.path(view, index)
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_bytes(("\n".join(transform(lines)) + "\n").encode("utf-8"))
        return self.problems()

    def edit_text(self, text):
        def transform(lines):
            record = json.loads(lines[0])
            return [json.dumps({**record, "text": text})] + lines[1:]
        return transform

    def test_requires_splits(self):
        code, err = quiet(s.main, self.args)
        self.assertEqual(code, 1)
        self.assertIn("split_runs_v2.py", err)

    def test_refuses_overwrite(self):
        self.run_pipeline()
        manifest = (self.derived / "control" / "manifest.sha256").read_bytes()
        code, err = quiet(s.main, self.args)
        self.assertEqual(code, 1)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual((self.derived / "control" / "manifest.sha256").read_bytes(), manifest)

    def test_outputs_and_metadata(self):
        self.run_pipeline()
        meta = json.loads((self.derived / "control" / "sanitization.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["source"]["commit"], common.SOURCE_COMMIT_V2)
        self.assertEqual(meta["split"]["seed"], 20261104)
        for view in s.VIEWS:
            self.assertEqual(meta["outputs"][view]["lines"], 50 * self.lines)
        names = [line.split("  ", 1)[1] for line in
                 (self.derived / "control" / "manifest.sha256").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(names), 3 + 100 + 1)
        self.assertIn("control/splits.json", names)

    def test_raw_files_are_not_changed(self):
        before = {p: p.read_bytes() for p in (self.official / "raw").iterdir()}
        self.run_pipeline()
        self.assertEqual({p: p.read_bytes() for p in (self.official / "raw").iterdir()}, before)

    def test_verifier_passes_on_clean_output(self):
        self.run_pipeline()
        problems, info = v.verify(self.official, self.derived)
        self.assertEqual(problems, [])
        self.assertEqual(info["raw_lines"], 50 * self.lines)
        self.assertEqual(sum(info["lines"]["controlada"].values()), 50 * self.lines)
        self.assertEqual(sum(info["lines"]["operacional"].values()), 50 * self.lines)
        self.assertEqual(len(info["texts"]), 19)
        for split, (matched, total) in info["correlation"].items():
            self.assertEqual(matched, total, split)

    def test_verifier_detects_forbidden_words_in_text(self):
        self.run_pipeline()
        for word in ("stresser-app", "CPU busy", "toxic", "net_timeout", "latency high", "baseline", "scenario", "label"):
            with self.subTest(word=word):
                self.assertIn("terms", self.tamper(self.edit_text(f"application-api info {word}"), index=1))

    def test_verifier_detects_digit_in_controlled_text(self):
        self.run_pipeline()
        self.assertIn("digits", self.tamper(self.edit_text("application-api info took seconds 12"), index=2))

    def test_verifier_detects_status_or_latency_in_controlled_text(self):
        self.run_pipeline()
        checks = self.tamper(self.edit_text("application-api info x status_code=200 latency_ms=62.1"), index=3)
        self.assertTrue({"terms", "digits"} <= checks)

    def test_operational_suffix_is_allowed_only_at_the_end(self):
        self.run_pipeline()
        self.assertEqual(v.split_operational("a b status_code=200 latency_ms=62.129"), ("a b", (200, "62.129")))
        self.assertIn("terms", self.tamper(self.edit_text("application-api info x latency_ms=62.1 y"), view="operacional", index=4))

    def test_verifier_detects_views_out_of_step(self):
        self.run_pipeline()
        self.assertIn("alignment", self.tamper(self.edit_text("application-api info other text"), view="operacional", index=5))

    def test_verifier_detects_changed_latency(self):
        self.run_pipeline()
        changed = self.tamper(lambda lines: [line.replace("latency_ms=62.129", "latency_ms=99.999") for line in lines],
                              view="operacional", index=6)
        self.assertIn("alignment", changed)

    def test_verifier_detects_ids_ip_and_port(self):
        self.run_pipeline()
        for text in (f"application-api info id {uuid.UUID(int=77)}", "application-api info from 172.18.0.5",
                     "application-api info at host:8080"):
            with self.subTest(text=text):
                checks = self.tamper(self.edit_text(text), index=7)
                self.assertIn("ids", checks)

    def test_verifier_detects_new_or_missing_text(self):
        self.run_pipeline()
        self.assertIn("texts", {c for c, _ in v.verify(self.official, self.derived, v.C1_TEXTS - {min(v.C1_TEXTS)})[0]})
        self.assertIn("texts", {c for c, _ in v.verify(self.official, self.derived, v.C1_TEXTS | {"other"})[0]})

    def test_verifier_detects_extra_meta_field(self):
        self.run_pipeline()

        def transform(lines):
            record = json.loads(lines[0])
            record["meta"]["condition"] = "x"
            return [json.dumps(record)] + lines[1:]
        self.assertIn("fields", self.tamper(transform, index=8))

    def test_verifier_detects_text_field_outside_text(self):
        self.run_pipeline()

        def transform(lines):
            return [json.dumps({**json.loads(lines[0]), "stress_level": 92})] + lines[1:]
        self.assertIn("fields", self.tamper(transform, index=9))

    def test_verifier_detects_wrong_set_in_meta(self):
        self.run_pipeline()

        def transform(lines):
            record = json.loads(lines[0])
            record["meta"]["split"] = "test" if record["meta"]["split"] == "train" else "train"
            return [json.dumps(record)] + lines[1:]
        self.assertIn("fields", self.tamper(transform, index=10))

    def test_verifier_detects_broken_correlation(self):
        self.run_pipeline()

        def transform(lines):
            out = []
            changed = False
            for line in lines:
                record = json.loads(line)
                if not changed and record["meta"]["service"] == "state-service":
                    record["meta"]["req"] = "0" * 12
                    changed = True
                out.append(json.dumps(record))
            return out
        self.assertIn("correlation", self.tamper(transform, index=11))

    def test_verifier_detects_missing_line_and_crlf(self):
        self.run_pipeline()
        self.assertTrue({"lines", "correlation", "alignment"} <= self.tamper(lambda lines: lines[:-1], index=12))
        path = self.path(index=13)
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        self.assertIn("encoding", self.problems())

    def test_verifier_detects_bad_event_order_and_ts(self):
        self.run_pipeline()

        def transform(lines):
            record = json.loads(lines[0])
            record["meta"]["event_order"] = 40
            return [json.dumps(record)] + lines[1:]
        self.assertIn("order", self.tamper(transform, index=14))
        self.assertIn("order", self.tamper(lambda lines: lines[::-1], index=15))

    def test_verifier_detects_condition_in_file_name(self):
        self.run_pipeline()
        path = self.path("operacional", 16)
        path.rename(path.with_name("cpu92-" + path.name))
        self.assertIn("names", self.problems())

    def test_verifier_detects_split_tampering(self):
        self.run_pipeline()
        gold_path, _ = split_runs_v2.split_paths(self.derived)
        text = gold_path.read_text(encoding="utf-8")
        gold_path.write_bytes(text.replace(",validation\n", ",train\n", 1).encode("utf-8"))
        self.assertIn("splits", self.problems())

    def test_verifier_detects_change_in_the_split_record(self):
        self.run_pipeline()
        path = split_runs_v2.record_path(self.derived)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["sets"]["test"].append(record["sets"]["train"][0])
        path.write_text(json.dumps(record), encoding="utf-8")
        self.assertIn("splits", self.problems())

    def test_verifier_detects_change_after_the_manifest(self):
        self.run_pipeline()
        path = self.derived / "control" / "sanitization.json"
        path.write_bytes(path.read_bytes() + b" ")
        self.assertIn("manifest", self.problems())


if __name__ == "__main__":
    unittest.main()

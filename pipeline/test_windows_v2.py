import json
import re
import unittest
import unittest.mock
import uuid

import build_windows_v2 as b
import common
import drain_templates as d1
import drain_templates_v2 as d
import sanitize
import sanitize_v2
import split_runs_v2
import test_pipeline_v2 as fixture
import verify_drain_v2 as vd
import verify_sanitized_v2 as vs
import verify_windows_v2 as v
from test_pipeline import TempDirTest, quiet

NS = sanitize.NS
T0 = 1_792_274_810_000_000_000
T0_ISO = "2026-11-04T21:00:00.000000000Z"
RARE_TEXT = "latency_ms=900.5"


def row(ts, req, order, template_id=1, service="application-api"):
    return {"ts": ts, "req": req, "service": service, "event_order": order, "template_id": template_id}


class PrepareTest(unittest.TestCase):
    BASE = "application-api info products_completed Product request completed GET /products"

    def test_controlled_text_is_not_changed(self):
        self.assertEqual(d.prepare("controlada", self.BASE), self.BASE)

    def test_operational_latency_becomes_the_v1_bucket(self):
        cases = {"0.3": "<10ms", "10": "10-50ms", "62.129": "50-100ms", "160.125": "100-500ms", "900.5": "500-1000ms",
                 "1000": "1000-1500ms", "1999.9": "1500-2000ms", "2000": ">=2000ms"}
        for value, label in cases.items():
            self.assertEqual(d.prepare("operacional", f"{self.BASE} status_code=200 latency_ms={value}"),
                             f"{self.BASE} status_code=200 latency_bucket={label}")

    def test_status_stays_and_lines_without_values_do_not_change(self):
        self.assertIn("status_code=503", d.prepare("operacional", f"{self.BASE} status_code=503 latency_ms=295.8"))
        self.assertEqual(d.prepare("operacional", self.BASE), self.BASE)


class DrainInferenceTest(unittest.TestCase):
    TRAIN = ["a b c d e f g h", "state c d e f g h i"]
    UNSEEN = "another line that never appeared here"

    def test_inference_does_not_change_the_tree_and_gives_unk(self):
        miner = d1.fit({"r": self.TRAIN}, d1.miner_config(d.VARIANT))
        before = d1.snapshot(miner)
        ids = d1.assign(miner, [self.UNSEEN, self.TRAIN[0], self.UNSEEN, "a b c d e f g x"], {})
        self.assertEqual(ids[0], d.UNK_ID)
        self.assertEqual(ids[2], d.UNK_ID)
        self.assertEqual(ids[3], d.UNK_ID)  # one token away is not a match in the exact configuration
        self.assertNotEqual(ids[1], d.UNK_ID)
        self.assertEqual(d1.snapshot(miner), before)

    def test_config_is_the_official_one(self):
        self.assertEqual(d.VARIANT, "exact")
        self.assertEqual(d1.miner_config(d.VARIANT).drain_sim_th, 0.95)


class AssignmentTest(unittest.TestCase):
    def test_line_goes_to_the_arrival_of_its_request(self):
        rows = [row(T0 + 29 * NS, "a", 0), row(T0 + 31 * NS, "a", 1), row(T0 + 32 * NS, "a", 2)]
        windows, outside = b.assign_windows(rows, T0)
        self.assertEqual([len(w) for w in windows[:2]], [3, 0])
        self.assertEqual(outside, [])

    def test_arrival_is_the_first_fastapi_line_in_canonical_order_not_the_earliest_time(self):
        rows = [row(T0 + 30 * NS, "a", 0), row(T0 + 29 * NS, "a", 1, service="state-service"), row(T0 + 31 * NS, "a", 2)]
        self.assertEqual(b.arrivals(rows), {"a": T0 + 30 * NS})
        windows, _ = b.assign_windows(rows, T0)
        self.assertEqual([len(w) for w in windows[:2]], [0, 3])

    def test_go_line_follows_the_request_across_the_boundary(self):
        rows = [row(T0 + 59 * NS, "a", 0), row(T0 + 61 * NS, "a", 1, 7, "state-service")]
        windows, _ = b.assign_windows(rows, T0)
        self.assertEqual([e[3] for e in windows[1]], [1, 7])
        self.assertEqual(windows[2], [])

    def test_line_without_request_goes_by_its_own_time(self):
        windows, _ = b.assign_windows([row(T0 + 95 * NS, "lonely", 0, 9, "state-service")], T0)
        self.assertEqual([e[3] for e in windows[3]], [9])

    def test_boundaries_and_lines_outside(self):
        rows = [row(T0, "a", 0), row(T0 + 30 * NS - 1, "b", 0), row(T0 + 30 * NS, "c", 0), row(T0 + 360 * NS - 1, "d", 0),
                row(T0 + 360 * NS, "e", 0), row(T0 - 1, "f", 0), row(T0 + 400 * NS, "e", 1)]
        windows, outside = b.assign_windows(rows, T0)
        self.assertEqual([sorted(e[1] for e in w) for w in (windows[0], windows[1], windows[11])], [["a", "b"], ["c"], ["d"]])
        self.assertEqual(sorted(e[1] for e in outside), ["e", "e", "f"])

    def test_twelve_windows_of_thirty_seconds(self):
        self.assertEqual((b.WINDOW_COUNT, b.WINDOW_SECONDS, b.WINDOW_NS), (12, 30, 30 * NS))


class OrderTest(unittest.TestCase):
    def sequence(self, rows):
        return [e[3] for e in b.assign_windows(rows, T0)[0][0]]

    def test_by_arrival_then_canonical_order_never_by_line_time(self):
        rows = [row(T0 + 1 * NS, "first", 0, 10), row(T0 + 2 * NS, "second", 0, 20),
                row(T0 + 3 * NS, "second", 1, 21), row(T0 + 9 * NS, "first", 2, 12),
                row(T0 + 8 * NS, "first", 1, 11, "state-service")]
        self.assertEqual(self.sequence(rows), [10, 11, 12, 20, 21])

    def test_requests_are_not_interleaved_on_equal_arrival(self):
        rows = [row(T0, "b", 0, 3), row(T0, "a", 0, 1), row(T0 + NS, "b", 1, 4), row(T0 + NS, "a", 1, 2)]
        self.assertEqual(self.sequence(rows), [1, 2, 3, 4])

    def test_file_order_does_not_matter(self):
        rows = [row(T0 + i * NS, "a", i, i) for i in range(5)]
        self.assertEqual(self.sequence(rows[::-1]), self.sequence(rows))


class ContentTest(unittest.TestCase):
    def test_content_has_only_ids_and_texts(self):
        windows, _ = b.assign_windows([row(T0 + 5 * NS, "abc123def456", 0, 1)], T0)
        line = b.content_line(windows[0], {"1": "application-api info x"})
        self.assertEqual(json.loads(line), {"template_ids": [1], "template_texts": ["application-api info x"]})
        for forbidden in (str(T0), str(T0 + 5 * NS), "abc123def456", '"k"', "run_id", "start", '"ts"'):
            self.assertNotIn(forbidden, line)


class PipelineFixture(TempDirTest):
    def setUp(self):
        super().setUp()
        items = fixture.ITEMS
        stub = [{"experiment_run_id": str(uuid.UUID(int=i)), "condition": c, "repetition": r} for i, (c, r) in enumerate(items, 1)]
        self.sets = {a["experiment_run_id"]: a["split"] for a in split_runs_v2.assign_splits(stub)}
        self.runs = fixture.build_official(self.official, items, lines_of=self.lines_of)
        for r in self.runs:
            common.write_new(self.official / "control" / f"{r['experiment_run_id']}-phases.json", json.dumps({
                "experiment_run_id": r["experiment_run_id"], "condition": r["condition"],
                "schedule": {"t0_utc": T0_ISO},
                "phase_boundaries_utc": {"intervention_start": "2026-11-04T21:02:00.000000000Z",
                                         "intervention_end": "2026-11-04T21:04:00.000000000Z"}}).encode("utf-8"))
        self.assertEqual(quiet(split_runs_v2.main, self.args)[0], 0)
        self.assertEqual(quiet(sanitize_v2.main, self.args)[0], 0)
        self.drain_args = ["--derived", str(self.derived)]

    def lines_of(self, run_id):
        """One final line per validation run has a latency seen nowhere in training: a UNK in the operational view only."""
        events = fixture.raw_events(run_id)
        if self.sets[run_id] == "validation":
            next(e for e in events if e["request_id"] == "r-404")["latency_ms"] = 900.5
        return [json.dumps(e) for e in events]

    def build_all(self):
        self.assertEqual(quiet(d.main, self.drain_args)[0], 0)
        self.assertEqual(quiet(b.main, self.args)[0], 0)

    def outputs(self, root):
        return {p.relative_to(root).as_posix(): p.read_bytes() for name in ("drain", "windows") for p in (root / name).rglob("*")
                if p.is_file()}


class PipelineTest(PipelineFixture):
    def test_verifiers_pass_and_c2a_manifest_still_holds(self):
        before = (self.derived / "control" / "manifest.sha256").read_bytes()
        self.build_all()
        self.assertEqual(vd.verify(self.derived)[0], [])
        problems, info = v.verify(self.official, self.derived)
        self.assertEqual(problems, [])
        self.assertEqual({s: len(info["metadata"][s]) for s in b.SPLITS}, {"train": 360, "validation": 120, "test": 120})
        self.assertEqual((self.derived / "control" / "manifest.sha256").read_bytes(), before)
        self.assertEqual(vs.verify(self.official, self.derived)[0], [])

    def test_templates_are_learned_only_from_training(self):
        self.build_all()
        for view, count in (("controlada", 19), ("operacional", None)):
            templates = json.loads((d.base_dir(self.derived) / view / "templates.json").read_text(encoding="utf-8"))
            self.assertFalse(any("500-1000ms" in t["template"] for t in templates))
            self.assertFalse(any("<*>" in t["template"] for t in templates))
            self.assertEqual(sum(t["train_lines"] for t in templates), 30 * len(fixture.raw_events(self.runs[0]["experiment_run_id"])))
            if count:
                self.assertEqual({t["template"] for t in templates}, vs.C1_TEXTS)

    def test_unk_counts_by_view(self):
        self.build_all()
        outputs = json.loads((d.base_dir(self.derived) / "drain_config.json").read_text(encoding="utf-8"))["outputs"]
        self.assertEqual(outputs["controlada"]["unk"], {"train": 0, "validation": 0, "test": 0})
        self.assertEqual(outputs["operacional"]["unk"], {"train": 0, "validation": 10, "test": 0})
        lines = (self.derived / "windows" / "operacional" / "validation.jsonl").read_text(encoding="utf-8")
        self.assertIn(d.UNK_TEXT, lines)
        self.assertNotIn(RARE_TEXT, lines)

    def test_operational_content_has_buckets_not_exact_latency(self):
        self.build_all()
        for split in b.SPLITS:
            text = (self.derived / "windows" / "operacional" / f"{split}.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("latency_ms", text)
            self.assertIn("latency_bucket=100-500ms", text)

    def test_content_has_no_time_run_or_index(self):
        self.build_all()
        for view in b.VIEWS:
            for split in b.SPLITS:
                text = (self.derived / "windows" / view / f"{split}.jsonl").read_text(encoding="utf-8")
                self.assertIsNone(re.search(r"[0-9a-f]{8}-[0-9a-f]{4}|\d{10,}|\d\.\d", text))
                for line in text.splitlines():
                    self.assertEqual(list(json.loads(line)), b.CONTENT_FIELDS)

    def test_metadata_rows_follow_the_content_lines(self):
        self.build_all()
        rows = v.read_metadata(self.derived / "windows", "train")
        content = v.read_content(self.derived / "windows" / "controlada" / "train.jsonl")
        self.assertEqual(len(rows), len(content))
        self.assertEqual(list(rows[0]), b.METADATA_COLUMNS)
        self.assertEqual([r["k"] for r in rows[:12]], [str(k) for k in range(12)])
        self.assertEqual((rows[1]["start_s"], rows[1]["end_s"]), ("30", "60"))
        for r, w in zip(rows, content):
            self.assertEqual(int(r["n_lines"]), len(w["template_ids"]))

    def test_generating_twice_gives_identical_bytes(self):
        self.build_all()
        other = self.derived.parent / "again"
        self.assertEqual(quiet(d.main, self.drain_args + ["--output", str(other)])[0], 0)
        self.assertEqual(quiet(b.main, self.args + ["--output", str(other)])[0], 0)
        self.assertEqual(self.outputs(other), self.outputs(self.derived))

    def test_refuses_overwrite(self):
        self.build_all()
        before = self.outputs(self.derived)
        for module, args in ((d, self.drain_args), (b, self.args)):
            code, err = quiet(module.main, args)
            self.assertEqual(code, 1)
            self.assertIn("refusing to overwrite", err)
        self.assertEqual(self.outputs(self.derived), before)

    def edit_content(self, view, transform):
        path = self.derived / "windows" / view / "train.jsonl"
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        path.write_bytes("".join(json.dumps(w) + "\n" for w in transform(lines)).encode("utf-8"))
        return {c for c, _ in v.verify(self.official, self.derived)[0]}

    def test_verifier_detects_time_in_content(self):
        self.build_all()

        def transform(lines):
            lines[0]["start_s"] = 0
            return lines
        self.assertIn("fields", self.edit_content("controlada", transform))

    def test_verifier_detects_forbidden_text_in_controlled_content(self):
        self.build_all()

        def transform(lines):
            lines[0]["template_texts"][0] = "application-api info latency_ms=62.1 cpu92"
            return lines
        checks = self.edit_content("controlada", transform)
        self.assertTrue({"forbidden", "templates", "manifest"} <= checks)

    def test_verifier_detects_changed_order(self):
        self.build_all()

        def transform(lines):
            first = lines[0]
            first["template_ids"] = first["template_ids"][::-1]
            first["template_texts"] = first["template_texts"][::-1]
            return lines
        self.assertIn("assignment", self.edit_content("operacional", transform))

    def test_verifier_detects_drain_template_outside_the_view_rules(self):
        self.assertEqual(quiet(d.main, self.drain_args)[0], 0)
        path = d.base_dir(self.derived) / "controlada" / "templates.json"
        templates = json.loads(path.read_text(encoding="utf-8"))
        templates[0]["template"] += " <*> 12"
        path.write_bytes(d1.json_bytes(templates))
        checks = {c for c, _ in vd.verify(self.derived)[0]}
        self.assertTrue({"view_rules", "one_to_one", "state", "manifest"} <= checks)

    def test_verifier_detects_exact_latency_in_operational_template(self):
        self.assertEqual(quiet(d.main, self.drain_args)[0], 0)
        path = d.base_dir(self.derived) / "operacional" / "templates.json"
        templates = json.loads(path.read_text(encoding="utf-8"))
        templates[0]["template"] = "application-api info x status_code=200 latency_ms=62.129"
        path.write_bytes(d1.json_bytes(templates))
        self.assertIn("view_rules", {c for c, _ in vd.verify(self.derived)[0]})


class AnalysisTest(unittest.TestCase):
    def info(self):
        def window(run_id, k, ids):
            return {"line": k, "run_id": run_id, "k": str(k)}, {"template_ids": ids}
        rows = [window("base", 0, [1, 1, 1, 2]), window("base", 1, [2, 2]), window("base", 2, [2, 2]), window("other", 0, [1, 2]),
                window("other", 1, [2, 3])]
        return {"metadata": {"train": [r for r, _ in rows], "validation": [], "test": []},
                "content": {"controlada": {"train": [w for _, w in rows], "validation": [], "test": []}},
                "mappings": {"controlada": {"1": "opened", "2": "sent", "3": "failed"}}}

    def test_baseline_diversity_counts_distinct_vectors_of_baseline_training_windows(self):
        self.assertEqual(v.baseline_diversity(self.info(), {"base": "baseline", "other": "cpu92"}), (2, 3))

    def test_position_flags_a_template_of_the_start(self):
        with unittest.mock.patch.object(v, "intervention", lambda official, run_id: None):
            report = {t["template"]: t for t in v.position_check(None, self.info(), {})}
        self.assertTrue(report["opened"]["flag"])
        self.assertIsNone(report["opened"]["ratio"])
        self.assertFalse(report["sent"]["flag"])
        self.assertEqual(report["opened"]["first_windows"], 2)

    def test_position_leaves_out_windows_that_touch_the_injection(self):
        with unittest.mock.patch.object(v, "intervention", lambda official, run_id: (30 * NS, 60 * NS)):
            report = {t["template"]: t for t in v.position_check(None, self.info(), {})}
        self.assertEqual(report["sent"]["other_windows"], 1)


if __name__ == "__main__":
    unittest.main()

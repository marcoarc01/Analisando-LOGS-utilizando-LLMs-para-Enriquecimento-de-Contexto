import json
import tempfile
import unittest
import unittest.mock
import uuid
from pathlib import Path

import build_windows_v2 as b
import common
import drain_templates_v2 as d
import operational_bins_v2 as o
import test_windows_v2 as tw
import verify_operational_v2 as v
import verify_sanitized_v2 as vs
import verify_windows_v2 as vw
from test_pipeline import quiet

API = "application-api info products_completed Product request completed GET /products"
GO = "state-service info state_queried state queried GET /api/state"
FAILED = "application-api error dependency_lookup_failed Dependency lookup failed GET /products error=ReadTimeout"
NOT_FOUND = "application-api warning product_not_found Product not found POST /orders"


def line(text, ts=1):
    return {"text": text, "meta": {"ts": ts, "req": "a" * 12, "run_id": "r", "split": "train", "service": text.split(" ")[0],
                                   "event_order": 0}}


class BinTest(unittest.TestCase):
    def test_labels_and_lower_limit_included(self):
        cuts = [150, 200, 292]
        self.assertEqual(o.bin_labels(cuts), ["<150ms", "150-200ms", "200-292ms", ">=292ms"])
        cases = {149.999: "<150ms", 150: "150-200ms", 199.9: "150-200ms", 200: "200-292ms", 291.999: "200-292ms",
                 292: ">=292ms", 2000: ">=292ms"}
        for ms, label in cases.items():
            self.assertEqual(o.bin_label(ms, cuts), label, ms)

    def test_cuts_are_rounded_sorted_and_without_repeats(self):
        self.assertEqual(o.cuts_of({"p50": 160.4, "p90": 160.49, "p99": 230.6}), [160, 231])
        self.assertEqual(o.cuts_of({"p50": 160.4, "p90": 200.0, "p99": 300.2}, (292,)), [160, 200, 292, 300])

    def test_percentiles_of_a_known_series(self):
        points = o.percentiles([float(x) for x in range(101)])
        self.assertEqual(points, {"p50": 50.0, "p90": 90.0, "p99": 99.0})

    def test_samples_take_only_concluded_requests_and_state_queried(self):
        records = [line(f"{API} status_code=200 latency_ms=160.5"), line(f"{GO} status_code=200 latency_ms=62.1"),
                   line(f"{FAILED} status_code=503 latency_ms=295.8"), line(f"{NOT_FOUND} status_code=404 latency_ms=0.3"),
                   line("application-api info dependency_request_sent Dependency request sent")]
        self.assertEqual(o.latency_samples(records), {"application-api": [160.5], "state-service": [62.1]})

    def test_each_service_uses_its_own_cuts_and_status_stays(self):
        prepare = o.preparer({"services": {"application-api": {"cuts": [150, 292]}, "state-service": {"cuts": [60, 100]}}})
        self.assertEqual(prepare("operacional", f"{API} status_code=200 latency_ms=160.5"),
                         f"{API} status_code=200 latency_bucket=150-292ms")
        self.assertEqual(prepare("operacional", f"{GO} status_code=200 latency_ms=160.5"),
                         f"{GO} status_code=200 latency_bucket=>=100ms")
        self.assertEqual(prepare("operacional", GO + " error=x"), GO + " error=x")


class BinSourceTest(unittest.TestCase):
    """Runs of every other kind have values that would move the cuts if they were read."""
    RUNS = (("baseline", "train", 100.0), ("baseline", "train", 200.0), ("baseline", "validation", 9000.0),
            ("baseline", "test", 9000.0), ("cpu92", "train", 9000.0), ("net_slow", "test", 9000.0))

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.derived = Path(tmp.name)
        gold = []
        self.ids = []
        for index, (condition, split, latency) in enumerate(self.RUNS, 1):
            run_id = str(uuid.UUID(int=index))
            self.ids.append(run_id)
            gold.append({"experiment_run_id": run_id, "condition": condition, "repetition": index, "split": split})
            self.write_run(run_id, latency)
        common.write_new(self.derived / "control" / "splits.csv",
                         common.csv_bytes(["experiment_run_id", "condition", "repetition", "split"], gold))
        common.write_new(self.derived / "splits.csv", common.csv_bytes(["experiment_run_id", "split"], [
            {k: g[k] for k in ("experiment_run_id", "split")} for g in gold]))

    def write_run(self, run_id, latency):
        rows = [{"text": f"{API} status_code=200 latency_ms={latency}", "meta": {"ts": 1, "run_id": run_id}},
                {"text": f"{GO} status_code=200 latency_ms={latency / 2}", "meta": {"ts": 2, "run_id": run_id}}]
        path = self.derived / "sanitized" / "operacional" / f"{run_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes("".join(json.dumps(r) + "\n" for r in rows).encode("utf-8"))

    def test_only_the_baseline_training_runs_are_read(self):
        bins = o.compute_bins(self.derived)
        self.assertEqual(bins["source"]["runs"], self.ids[:2])
        api, go = bins["services"]["application-api"], bins["services"]["state-service"]
        self.assertEqual((api["lines"], go["lines"]), (2, 2))
        self.assertLess(max(api["percentiles"].values()), 200.1)
        self.assertLess(max(go["percentiles"].values()), 100.1)

    def test_other_runs_do_not_change_the_cuts(self):
        before = o.compute_bins(self.derived)
        for run_id in self.ids[2:]:
            self.write_run(run_id, 12345.0)
        self.assertEqual(o.compute_bins(self.derived), before)

    def test_a_baseline_training_run_changes_the_cuts(self):
        before = o.compute_bins(self.derived)
        self.write_run(self.ids[0], 150.0)
        self.assertNotEqual(o.compute_bins(self.derived), before)

    def test_the_timeout_is_a_fastapi_cut_only(self):
        bins = o.compute_bins(self.derived)
        self.assertIn(292, bins["services"]["application-api"]["cuts"])
        self.assertNotIn(292, bins["services"]["state-service"]["cuts"])


class WarmupTest(unittest.TestCase):
    def rows(self):
        return [{"line": str(k), "run_id": "r", "split": "train", "k": str(k), "start_s": str(k * 30), "end_s": str(k * 30 + 30),
                 "n_requests": "1", "n_lines": "1"} for k in range(b.WINDOW_COUNT)]

    def test_only_the_first_window_is_marked_and_no_window_is_dropped(self):
        windows = {"r": [[(0, "a", 0, 1)] for _ in range(b.WINDOW_COUNT)]}
        rows = o.new_metadata(self.rows(), windows)
        self.assertEqual(len(rows), b.WINDOW_COUNT)
        self.assertEqual([r["warmup"] for r in rows], [1] + [0] * (b.WINDOW_COUNT - 1))

    def test_rows_that_disagree_with_the_windows_stop(self):
        with self.assertRaisesRegex(common.PipelineError, "differs"):
            o.new_metadata(self.rows(), {"r": [[] for _ in range(b.WINDOW_COUNT)]})

    def test_exclusion_is_documented_for_every_set(self):
        for word in ("train", "validation", "test", "C3"):
            self.assertIn(word, o.WARMUP_DOC["use"])


class SecondWindowTest(unittest.TestCase):
    def info(self):
        rows, content = [], []
        for run in ("x", "y"):
            for k, (requests, opened) in enumerate(((5, 9), (8, 1), (8, 1), (9, 2))):
                rows.append({"run_id": run, "k": str(k), "warmup": str(int(k == 0)), "n_requests": str(requests), "n_lines": str(requests)})
                content.append({"template_ids": [1] * opened + [2]})
        return {"metadata": {"train": rows, "validation": [], "test": []},
                "content": {"controlada": {"train": content, "validation": [], "test": []}},
                "mappings": {"controlada": {"1": "dependency_connection_opened", "2": "other"}}}

    def measures(self, bounds=None, **kwargs):
        with unittest.mock.patch.object(vw, "intervention", lambda official, run_id: bounds):
            return v.second_window_measures(None, self.info(), **kwargs)

    def test_warmup_windows_are_left_out(self):
        m = self.measures()
        self.assertEqual(m["requests"]["n"], (2, 4))
        self.assertEqual(m["requests"]["first"][:2], (8, 8.0))
        self.assertEqual(m["requests"]["others"][0], 8)

    def test_auc(self):
        self.assertEqual(v.auc([3], [1, 2]), 1.0)
        self.assertEqual(v.auc([1], [2, 3]), 0.0)
        self.assertEqual(v.auc([2], [2]), 0.5)

    def test_windows_that_touch_the_injection_are_left_out(self):
        self.assertEqual(self.measures(bounds=(90 * tw.NS, 120 * tw.NS))["requests"]["n"], (2, 2))


class PipelineTest(tw.PipelineFixture):
    def build_c2c(self):
        self.build_all()
        self.assertEqual(quiet(o.main, self.args)[0], 0)
        return self.derived / o.STAGE

    def files(self, root):
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    def test_verifier_passes_and_c2b_files_stay(self):
        self.build_all()
        before = self.outputs(self.derived)
        base = self.build_c2c_only()
        self.assertEqual(self.outputs(self.derived), before)
        problems, info = v.verify(self.official, self.derived)
        self.assertEqual(problems, [])
        self.assertTrue((base / "manifest.sha256").exists())
        self.assertEqual({s: len(info["metadata"][s]) for s in b.SPLITS}, {"train": 360, "validation": 120, "test": 120})
        self.assertEqual(vs.verify(self.official, self.derived)[0], [])

    def build_c2c_only(self):
        self.assertEqual(quiet(o.main, self.args)[0], 0)
        return self.derived / o.STAGE

    def test_metadata_marks_warmup_in_every_set(self):
        base = self.build_c2c()
        for split in b.SPLITS:
            rows = vw.read_metadata(base / "windows", split)
            self.assertEqual(list(rows[0]), o.METADATA_COLUMNS)
            self.assertEqual(sum(r["warmup"] == "1" for r in rows), len(rows) // 12)
            self.assertTrue(all(r["warmup"] == ("1" if r["k"] == "0" else "0") for r in rows))
        config = json.loads((base / "windows" / "windows_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["metadata"]["warmup"], o.WARMUP_DOC)

    def test_templates_use_the_new_bins_and_no_exact_latency(self):
        base = self.build_c2c()
        text = (base / "windows" / "operacional" / "train.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("latency_ms", text)
        self.assertNotIn("100-500ms", text)
        self.assertIn("latency_bucket=", text)

    def test_controlled_content_is_not_written(self):
        base = self.build_c2c()
        self.assertFalse(any("controlada" in name for name in self.files(base)))

    def test_generating_twice_gives_identical_bytes(self):
        base = self.build_c2c()
        other = self.derived.parent / "again"
        self.assertEqual(quiet(o.main, self.args + ["--output", str(other)])[0], 0)
        self.assertEqual(self.files(other), self.files(base))

    def test_refuses_overwrite(self):
        base = self.build_c2c()
        before = self.files(base)
        code, err = quiet(o.main, self.args)
        self.assertEqual(code, 1)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual(self.files(base), before)

    def tamper_rows(self, base, transform):
        path = base / "windows" / "metadata" / "train.csv"
        rows = vw.read_metadata(base / "windows", "train")
        path.write_bytes(common.csv_bytes(o.METADATA_COLUMNS, transform(rows)))
        return {c for c, _ in v.verify(self.official, self.derived)[0]}

    def test_verifier_detects_wrong_warmup_mark(self):
        base = self.build_c2c()

        def transform(rows):
            rows[1]["warmup"] = "1"
            return rows
        self.assertTrue({"warmup", "manifest"} <= self.tamper_rows(base, transform))

    def test_verifier_detects_a_dropped_window(self):
        base = self.build_c2c()
        self.assertIn("metadata", self.tamper_rows(base, lambda rows: rows[1:]))

    def test_verifier_detects_cuts_changed_by_hand(self):
        base = self.build_c2c()
        path = base / "bins.json"
        bins = json.loads(path.read_text(encoding="utf-8"))
        bins["services"]["application-api"]["cuts"].append(400)
        path.write_text(json.dumps(bins), encoding="utf-8")
        self.assertIn("bins", {c for c, _ in v.verify(self.official, self.derived)[0]})

    def test_verifier_detects_text_out_of_step_with_the_controlled_view(self):
        base = self.build_c2c()
        path = base / "windows" / "operacional" / "train.jsonl"
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        lines[0]["template_texts"][0] = "application-api info another text"
        path.write_bytes("".join(json.dumps(w) + "\n" for w in lines).encode("utf-8"))
        self.assertIn("views", {c for c, _ in v.verify(self.official, self.derived)[0]})


if __name__ == "__main__":
    unittest.main()

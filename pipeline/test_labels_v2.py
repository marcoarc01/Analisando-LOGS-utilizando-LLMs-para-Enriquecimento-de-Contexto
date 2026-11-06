"""Needs scipy: run with C:\\tcc-venv\\cenario1. In a venv without scipy (the Drain one) the module is skipped."""
import csv
import datetime as dt
import json
import random
import unittest

try:
    import scipy  # noqa: F401
except ImportError:
    raise unittest.SkipTest("needs scipy: run with the cenario1 venv")

import build_labels_v2 as l
import build_windows as bv1
import common
import verify_labels_v2 as v
from test_pipeline import TempDirTest, quiet

NS = 1_000_000_000
T0 = 1_792_274_810 * NS
API = "application-api"
GO = "state-service"
OK_TEXT = "application-api info products_completed Product request completed GET /products"
FAILED_TEXT = "application-api error dependency_lookup_failed Dependency lookup failed GET /products error=ReadTimeout"
RESPONSE_FAILED = "application-api warning dependency_response_failed Dependency response failed error=ReadTimeout"
SENT_TEXT = "application-api info dependency_request_sent Dependency request sent"
GO_TEXT = "state-service info state_queried state queried GET /api/state"
NOT_FOUND_TEXT = "application-api warning product_not_found Product not found POST /orders"
INJECTION = (130, 250)  # seconds from t0: windows k = 4..8 overlap it
RUNS = (("baseline", "train", "normal"), ("baseline", "train", "normal"), ("baseline", "validation", "normal"),
        ("baseline", "test", "normal"), ("cpu92", "train", "slow"), ("net_timeout", "train", "timeout"),
        ("cpu92", "test", "slow"), ("net_slow", "validation", "none"), ("net_timeout", "test", "timeout"),
        ("cpu92", "validation", "timeout"))


def iso(n):
    seconds, nanos = divmod(n, NS)
    return dt.datetime.fromtimestamp(seconds, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{nanos:09d}Z"


def row(**fields):
    base = {"warmup": False, "overlap": 0.5, "p_latency_holm": 1.0, "p_errors_holm": 1.0, "a12": 0.5, "failure_lines": 0, "split": "train"}
    return {**base, **fields}


class RuleTest(unittest.TestCase):
    def test_normal_without_overlap_whatever_the_tests_say(self):
        self.assertEqual(l.status_of(row(overlap=0.0, p_latency_holm=0.0, a12=1.0, p_errors_holm=0.0)), l.NORMAL)

    def test_anomalous_by_latency_needs_p_and_a12(self):
        self.assertEqual(l.status_of(row(p_latency_holm=0.0099, a12=0.64)), l.ANOMALOUS)
        self.assertEqual(l.status_of(row(p_latency_holm=0.0099, a12=0.6399)), l.EXCLUDED)
        self.assertEqual(l.status_of(row(p_latency_holm=0.01, a12=0.99)), l.EXCLUDED)

    def test_anomalous_by_5xx_alone(self):
        self.assertEqual(l.status_of(row(p_errors_holm=0.0099, a12=0.5)), l.ANOMALOUS)
        self.assertEqual(l.status_of(row(p_errors_holm=0.01, a12=0.5)), l.EXCLUDED)

    def test_excluded_when_nothing_is_confirmed(self):
        self.assertEqual(l.status_of(row()), l.EXCLUDED)
        self.assertEqual(l.status_of(row(a12=None)), l.EXCLUDED)

    def test_warmup_wins(self):
        self.assertEqual(l.status_of(row(warmup=True, overlap=0.0)), l.WARMUP)

    def test_a_tiny_overlap_counts(self):
        self.assertEqual(l.status_of(row(overlap=0.00001)), l.EXCLUDED)

    def test_other_limits(self):
        window = row(p_latency_holm=0.03, a12=0.6)
        self.assertEqual(l.status_of(window, 0.05, 0.56), l.ANOMALOUS)
        self.assertEqual(l.status_of(window, 0.01, 0.56), l.EXCLUDED)
        self.assertEqual(l.status_of(window, 0.05, 0.64), l.EXCLUDED)


class HolmTest(unittest.TestCase):
    def test_family_of_latency_and_errors_together(self):
        rows = [{"p_latency": 0.03, "p_errors": 0.01}, {"p_latency": 0.04, "p_errors": 0.02}]
        l.apply_holm(rows)
        self.assertEqual([round(r["p_latency_holm"], 10) for r in rows], [0.06, 0.06])
        self.assertEqual([round(r["p_errors_holm"], 10) for r in rows], [0.04, 0.06])

    def test_family_size_changes_the_result(self):
        small = [{"p_latency": 0.004, "p_errors": 1.0}]
        large = [dict(small[0]), {"p_latency": 1.0, "p_errors": 1.0}]
        l.apply_holm(small)
        l.apply_holm(large)
        self.assertLess(small[0]["p_latency_holm"], 0.01)
        self.assertGreater(large[0]["p_latency_holm"], 0.01)

    def test_verifier_correction_agrees(self):
        values = [0.03, 0.01, 0.04, 0.02]
        rows = [{"p_latency": a, "p_errors": b} for a, b in zip(values[:2], values[2:])]
        l.apply_holm(rows)
        self.assertEqual(v.own_holm(values), [rows[0]["p_latency_holm"], rows[1]["p_latency_holm"],
                                              rows[0]["p_errors_holm"], rows[1]["p_errors_holm"]])


class TypeAndSetTest(unittest.TestCase):
    def test_type_depends_on_the_failure_lines(self):
        self.assertEqual(l.anomaly_type(row(failure_lines=1)), l.VISIBLE)
        self.assertEqual(l.anomaly_type(row(failure_lines=0)), l.SLOW_ONLY)

    def test_failure_list_leaves_the_client_noise_out(self):
        texts = [NOT_FOUND_TEXT, "application-api warning request_rejected Request rejected POST /orders", GO_TEXT, OK_TEXT, "<UNK>"]
        self.assertEqual(l.content_failures(texts), {})
        for text in (FAILED_TEXT, RESPONSE_FAILED, "state-service warning state_request_canceled State request canceled GET /api/state error=context canceled",
                     "state-service warning state_request_deadline_exceeded State request deadline exceeded GET /api/state error=context deadline exceeded",
                     FAILED_TEXT.replace("ReadTimeout", "UnexpectedStatus")):
            self.assertEqual(sum(l.content_failures([text]).values()), 1, text)

    def test_sets(self):
        visible = {"status": l.ANOMALOUS, "anomaly_type": l.VISIBLE}
        slow = {"status": l.ANOMALOUS, "anomaly_type": l.SLOW_ONLY}
        for split in l.SPLITS:
            self.assertTrue(l.in_set({"status": l.NORMAL, "anomaly_type": "", "split": split}))
            self.assertTrue(l.in_set({**visible, "split": split}))
            self.assertFalse(l.in_set({"status": l.EXCLUDED, "anomaly_type": "", "split": split}))
            self.assertFalse(l.in_set({"status": l.WARMUP, "anomaly_type": "", "split": split}))
        self.assertFalse(l.in_set({**slow, "split": "train"}))
        self.assertFalse(l.in_set({**slow, "split": "validation"}))
        self.assertTrue(l.in_set({**slow, "split": "test"}))


class ReferenceTest(unittest.TestCase):
    def rows(self):
        return [{"condition": c, "split": s, "warmup": w, "run_id": r, "k": k} for c, s, w, r, k in (
            ("baseline", "train", False, "a", 1), ("baseline", "train", True, "a", 0), ("baseline", "validation", False, "b", 1),
            ("baseline", "test", False, "c", 1), ("cpu92", "train", False, "d", 1))]

    def test_only_baseline_training_windows_without_warmup(self):
        windows = {r: [{"latencies": [float(i)], "n5xx": 0} for i in range(2)] for r in "abcd"}
        self.assertEqual(sorted(l.build_reference(self.rows(), windows)), [("a", 1)])

    def test_a_window_of_the_reference_is_left_out_of_its_own_test(self):
        parts = {("a", 1): {"latencies": [1.0, 1.0], "n5xx": 0}, ("a", 2): {"latencies": [2.0, 2.0], "n5xx": 1}}
        outside = l.test_window({"latencies": [3.0], "n5xx": 1}, parts, ("x", 1))
        own = l.test_window({"latencies": [2.0, 2.0], "n5xx": 1}, parts, ("a", 2))
        self.assertEqual(own["a12"], 1.0)  # 2 against the two 1s only
        self.assertEqual(outside["a12"], 1.0)
        self.assertLess(own["p_errors"], 1.0)


class OldRuleAndDistanceTest(unittest.TestCase):
    def test_old_rule_table(self):
        parts = {("a", k): {"latencies": [10.0, 11.0, 12.0, 13.0]} for k in (1, 2)}
        rows = [{"warmup": False, "status": l.ANOMALOUS, "anomaly_type": l.SLOW_ONLY, "condition": "c", "rate_5xx": 0.0, "median_ms": 11.0},
                {"warmup": False, "status": l.ANOMALOUS, "anomaly_type": l.VISIBLE, "condition": "c", "rate_5xx": 0.5, "median_ms": 11.0},
                {"warmup": False, "status": l.NORMAL, "anomaly_type": "", "condition": "c", "rate_5xx": 0.0, "median_ms": 99.0},
                {"warmup": False, "status": l.EXCLUDED, "anomaly_type": "", "condition": "c", "rate_5xx": 0.0, "median_ms": 1.0},
                {"warmup": True, "status": l.WARMUP, "anomaly_type": "", "condition": "c", "rate_5xx": 1.0, "median_ms": 1.0}]
        out = l.old_rule(rows, parts)
        self.assertEqual(out["anomalous_lost"], 1)
        self.assertEqual(out["normal_marked_anomalous"], 1)
        self.assertEqual(out["agreement_normal_or_anomalous_windows"], 3)
        self.assertAlmostEqual(out["agreement_without_excluded"], 1 / 3)

    def test_nearest_boundary(self):
        interval = (100 * NS, 220 * NS)
        self.assertEqual(l.nearest_boundary(99 * NS + NS // 2, interval), (0.5, "before_start", "edge"))
        self.assertEqual(l.nearest_boundary(90 * NS, interval), (10.0, "before_start", "spontaneous"))
        self.assertEqual(l.nearest_boundary(224 * NS, interval), (4.0, "after_end", "edge"))
        self.assertEqual(l.nearest_boundary(300 * NS, interval), (80.0, "after_end", "spontaneous"))
        self.assertEqual(l.nearest_boundary(300 * NS, None), (None, None, "no_injection"))


def run_lines(run_id, kind, seed):
    """(raw lines, sanitized operational lines, controlled texts of each window), by the rule of each kind of run."""
    rng = random.Random(seed)
    raw, sanitized = [], []
    windows = [[] for _ in range(12)]
    for k in range(12):
        for i in range(30):
            arrival = T0 + (k * 30 * NS) + i * NS + 10_000
            inside = INJECTION[0] * NS <= arrival - T0 < INJECTION[1] * NS
            latency, status, failed = round(rng.uniform(100, 200), 3), 200, False
            if inside and kind == "slow":
                latency += 300
            if inside and kind == "timeout":
                latency, status, failed = 296.0, 503, True
            req = f"r{k}-{i}"
            pseudo = f"{(k * 100 + i):012x}"
            final = FAILED_TEXT if failed else OK_TEXT
            event = "stresser_lookup_failed" if failed else "products_completed"
            ts = arrival + round(latency * 1e6)
            sent = {"request_id": req, "timestamp": iso(arrival), "service": API, "event": "dependency_request_sent"}
            done = {"request_id": req, "timestamp": iso(ts), "service": API, "event": event, "status_code": status, "latency_ms": latency}
            raw += [sent, done]
            windows[k] += [SENT_TEXT, final]
            lines = [(SENT_TEXT, arrival, API, 0), (f"{final} status_code={status} latency_ms={latency}", ts, API, 1)]
            sanitized += [{"text": text, "meta": {"ts": t, "req": pseudo, "run_id": run_id, "split": "x", "service": service, "event_order": order}}
                          for text, t, service, order in lines]
    sanitized.sort(key=lambda r: r["meta"]["ts"])
    return raw, sanitized, windows


class PipelineTest(TempDirTest):
    def setUp(self):
        super().setUp()
        self.sets = {s: [] for s in l.SPLITS}
        self.ids = {}
        metadata, content = {s: [] for s in l.SPLITS}, {s: [] for s in l.SPLITS}
        for index, (condition, split, kind) in enumerate(RUNS, 1):
            run_id = f"00000000-0000-0000-0000-{index:012d}"
            self.ids[(condition, split)] = run_id
            self.sets[split].append({"experiment_run_id": run_id, "condition": condition, "repetition": index})
            raw, sanitized, windows = run_lines(run_id, kind, index)
            common.write_new(self.official / "raw" / f"{run_id}.jsonl", "".join(json.dumps(e) + "\n" for e in raw).encode("utf-8"))
            common.write_new(self.derived / "sanitized" / "operacional" / f"{run_id}.jsonl",
                             "".join(json.dumps(e) + "\n" for e in sanitized).encode("utf-8"))
            common.write_new(self.official / "control" / f"{run_id}-phases.json", json.dumps({
                "experiment_run_id": run_id, "condition": condition, "schedule": {"t0_utc": iso(T0)},
                "phase_boundaries_utc": {"intervention_start": iso(T0 + INJECTION[0] * NS), "intervention_end": iso(T0 + INJECTION[1] * NS)}}).encode("utf-8"))
            for k, texts in enumerate(windows):
                metadata[split].append({"line": len(metadata[split]), "run_id": run_id, "split": split, "k": k, "start_s": k * 30,
                                        "end_s": k * 30 + 30, "n_requests": 30, "n_lines": 60, "warmup": int(k == 0)})
                content[split].append({"template_ids": [1] * len(texts), "template_texts": texts})
        columns = list(metadata["train"][0])
        for split in l.SPLITS:
            common.write_new(self.derived / "c2c" / "windows" / "metadata" / f"{split}.csv", common.csv_bytes(columns, metadata[split]))
            for folder in (self.derived / "windows" / "controlada", self.derived / "c2c" / "windows" / "operacional"):
                common.write_new(folder / f"{split}.jsonl", "".join(json.dumps(w) + "\n" for w in content[split]).encode("utf-8"))
        common.write_new(self.derived / "control" / "splits.json", json.dumps({"sets": self.sets}).encode("utf-8"))
        for folder in ("windows", "c2c"):
            files = {p.relative_to(self.derived / folder).as_posix(): p.read_bytes() for p in (self.derived / folder).rglob("*") if p.is_file()}
            common.write_new(self.derived / folder / "manifest.sha256", bv1.manifest_bytes(files))
        common.write_new(self.derived / "control" / "manifest.sha256", b"")
        files = {p.relative_to(self.derived).as_posix(): p.read_bytes() for p in self.derived.rglob("*") if p.is_file()
                 and p.relative_to(self.derived).parts[0] == "sanitized" or p.name == "splits.json"}
        (self.derived / "control" / "manifest.sha256").write_bytes(bv1.manifest_bytes(files))

    def run_labels(self):
        self.assertEqual(quiet(l.main, self.args)[0], 0)
        return self.derived / "labels"

    def rows(self):
        return v.read_rows(self.derived / "labels")

    def test_real_flow_and_verifier(self):
        self.run_labels()
        problems, info = v.verify(self.official, self.derived)
        self.assertEqual(problems, [])
        self.assertEqual(sum(r["warmup"] == "True" for r in info["rows"]), len(RUNS))

    def status(self, condition, split, k):
        run_id = self.ids[(condition, split)]
        return next(r for r in self.rows() if r["run_id"] == run_id and r["k"] == str(k))

    def test_labels_by_kind_of_run(self):
        self.run_labels()
        slow = self.status("cpu92", "train", 6)
        self.assertEqual((slow["status"], slow["anomaly_type"], slow["in_set"]), ("anomalous", "slow_only", "0"))
        test = self.status("cpu92", "test", 6)
        self.assertEqual((test["status"], test["anomaly_type"], test["in_set"]), ("anomalous", "slow_only", "1"))
        timeout = self.status("net_timeout", "train", 6)
        self.assertEqual((timeout["status"], timeout["anomaly_type"], timeout["in_set"]), ("anomalous", "visible", "1"))
        absorbed = self.status("net_slow", "validation", 6)
        self.assertEqual((absorbed["status"], absorbed["in_set"]), ("excluded", "0"))
        self.assertEqual(self.status("baseline", "test", 6)["status"], "normal")
        self.assertEqual(self.status("net_slow", "validation", 2)["status"], "normal")

    def test_warmup_windows_are_in_no_set_and_were_not_tested(self):
        base = self.run_labels()
        for r in self.rows():
            if r["k"] == "0":
                self.assertEqual((r["status"], r["in_set"], r["p_latency"]), ("warmup", "0", ""))
        for split in l.SPLITS:
            with open(base / "sets" / f"{split}.csv", encoding="utf-8", newline="") as f:
                self.assertNotIn("0", {r["k"] for r in csv.DictReader(f)})

    def test_false_alarm_check_never_changes_a_label(self):
        self.run_labels()
        checked = [r for r in self.rows() if r["false_alarm_check"] != ""]
        self.assertTrue(checked)
        self.assertEqual({r["status"] for r in checked}, {"normal"})

    def test_holm_family_has_only_the_windows_with_overlap(self):
        base = self.run_labels()
        config = json.loads((base / "labels_config.json").read_text(encoding="utf-8"))
        with_overlap = sum(float(r["overlap"]) > 0 for r in self.rows())
        self.assertEqual(with_overlap, 5 * sum(kind != "normal" for _, _, kind in RUNS))
        self.assertEqual(config["rule"]["holm_family"]["tests"], 2 * with_overlap)

    def test_official_row_of_the_sensitivity_agrees_with_itself(self):
        base = self.run_labels()
        grid = json.loads((base / "sensitivity.json").read_text(encoding="utf-8"))["grid"]
        self.assertEqual(len(grid), 6)
        self.assertEqual([g["agreement"] for g in grid if g["official"]], [1.0])

    def test_files_have_no_time_of_generation_and_the_content_stays(self):
        before = (self.derived / "windows" / "controlada" / "train.jsonl").read_bytes()
        base = self.run_labels()
        self.assertEqual((self.derived / "windows" / "controlada" / "train.jsonl").read_bytes(), before)
        self.assertNotIn("created_utc", (base / "labels_config.json").read_text(encoding="utf-8"))

    def test_generating_twice_gives_identical_bytes(self):
        base = self.run_labels()
        other = self.derived.parent / "again"
        self.assertEqual(quiet(l.main, self.args + ["--output", str(other)])[0], 0)
        first = {p.relative_to(base): p.read_bytes() for p in base.rglob("*") if p.is_file()}
        self.assertEqual({p.relative_to(other): p.read_bytes() for p in other.rglob("*") if p.is_file()}, first)

    def test_refuses_overwrite(self):
        base = self.run_labels()
        before = (base / "labels.csv").read_bytes()
        code, err = quiet(l.main, self.args)
        self.assertEqual(code, 1)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual((base / "labels.csv").read_bytes(), before)

    def test_a_different_arrival_stops(self):
        path = self.derived / "c2c" / "windows" / "metadata" / "train.csv"
        path.write_text(path.read_text(encoding="utf-8").replace(",30,60,", ",31,60,", 1), encoding="utf-8")
        code, err = quiet(l.main, self.args)
        self.assertEqual(code, 1)
        self.assertIn("arrivals differ", err)

    def test_a_run_in_two_sets_stops(self):
        path = self.derived / "control" / "splits.json"
        sets = json.loads(path.read_text(encoding="utf-8"))
        sets["sets"]["test"].append(sets["sets"]["train"][0])
        path.write_text(json.dumps(sets), encoding="utf-8")
        code, err = quiet(l.main, self.args)
        self.assertEqual(code, 1)
        self.assertIn("more than one set", err)

    def tamper(self, column, value, match):
        base = self.derived / "labels"
        rows = self.rows()
        target = next(r for r in rows if match(r))
        target[column] = value
        (base / "labels.csv").write_bytes(common.csv_bytes(l.COLUMNS, rows))
        return {c for c, _ in v.verify(self.official, self.derived)[0]}

    def test_verifier_detects_a_changed_label(self):
        self.run_labels()
        self.assertTrue({"rule", "manifest"} <= self.tamper("status", "anomalous", lambda r: r["status"] == "normal"))

    def test_verifier_detects_a_changed_type(self):
        self.run_labels()
        self.assertIn("visible", self.tamper("anomaly_type", "visible", lambda r: r["anomaly_type"] == "slow_only"))

    def test_verifier_detects_a_changed_p_value(self):
        self.run_labels()
        self.assertTrue({"tests", "holm"} <= self.tamper("p_latency", "0.5", lambda r: r["status"] == "anomalous"))

    def test_verifier_detects_a_warmup_window_in_a_set(self):
        base = self.run_labels()
        path = base / "sets" / "train.csv"
        run_id = self.ids[("baseline", "train")]
        path.write_text(path.read_text(encoding="utf-8") + f"0,{run_id},0,0,\n", encoding="utf-8")
        self.assertIn("sets", {c for c, _ in v.verify(self.official, self.derived)[0]})

    def test_verifier_detects_a_label_in_the_content(self):
        self.run_labels()
        path = self.derived / "windows" / "controlada" / "train.jsonl"
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
        lines[0]["label"] = 1
        path.write_bytes("".join(json.dumps(w) + "\n" for w in lines).encode("utf-8"))
        self.assertTrue({"content", "inputs"} <= {c for c, _ in v.verify(self.official, self.derived)[0]})


if __name__ == "__main__":
    unittest.main()

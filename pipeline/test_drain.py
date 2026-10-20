import contextlib
import io
import json
import tempfile
import unittest
import uuid
from pathlib import Path

from drain3.template_miner_config import TemplateMinerConfig

import common
import drain_templates as d
import verify_drain as v

TS = 1792274810698987000
REQ = "4ef5a1b2c3d4"


def record(event="products_completed", message="Product request completed", status=200, bucket="<10ms",
           level="info", route="/products", service="application-api", ts=TS, req=REQ, error=None):
    r = {"ts": ts, "req": req, "service": service, "level": level, "event": event, "message": message,
         "method": "GET", "route": route, "status_code": status, "latency_bucket": bucket}
    if error:
        r["error_type"] = error
    return r


def controlled(r):
    return {k: val for k, val in r.items() if k not in ("status_code", "latency_bucket")}


TRAIN_LINES = [
    record(service="state-service", event="state_queried", message="state queried", route="/api/state"),
    record(),
    record(bucket="1000-1500ms"),
    record(event="internal_error", message="Internal server error", status=500, bucket="1500-2000ms", level="error"),
    record(event="health_checked", message="Health request completed", route="/health"),
]
UNSEEN = record(event="dependency_lookup_failed", message="Dependency lookup failed", status=503, bucket=">=2000ms",
                level="error", error="ReadTimeout")


def build_derived(root):
    runs = {str(uuid.UUID(int=i)): split for i, split in
            enumerate(["train", "train", "train", "validation", "test"], 1)}
    common.write_new(root / "splits.csv", common.csv_bytes(
        ["experiment_run_id", "split"], [{"experiment_run_id": k, "split": s} for k, s in sorted(runs.items())]))
    for n, (run_id, split) in enumerate(sorted(runs.items())):
        lines = [dict(r, ts=TS + i, req=f"{n:02d}{i:010x}") for i, r in enumerate(TRAIN_LINES)]
        if split != "train":
            lines.append(dict(UNSEEN, ts=TS + 99, req=f"{n:02d}{99:010x}"))
        for view, rows in (("operacional", lines), ("controlada", [controlled(r) for r in lines])):
            content = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows).encode("utf-8")
            common.write_new(root / "sanitized" / view / f"{run_id}.jsonl", content)
    return runs


def quiet(func, *args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
        return func(*args), err.getvalue()


class RenderTest(unittest.TestCase):
    def test_operational_text(self):
        self.assertEqual(d.render("operacional", record()),
                         "application-api info products_completed Product request completed GET /products "
                         "status=200 latency=<10ms")
        self.assertEqual(d.render("operacional", UNSEEN),
                         "application-api error dependency_lookup_failed Dependency lookup failed GET /products "
                         "status=503 latency=>=2000ms error=ReadTimeout")

    def test_controlled_text(self):
        self.assertEqual(d.render("controlada", controlled(record())),
                         "application-api info products_completed Product request completed GET /products")
        self.assertTrue(d.render("controlada", controlled(UNSEEN)).endswith("Dependency lookup failed GET /products "
                                                                           "error=ReadTimeout"))

    def test_ts_and_req_never_in_text(self):
        for view in d.VIEWS:
            text = d.render(view, record() if view == "operacional" else controlled(record()))
            self.assertNotIn(str(TS), text)
            self.assertNotIn(REQ, text)
            self.assertNotIn("ts", text.split())


class MinerTest(unittest.TestCase):
    def texts(self, view="operacional", lines=TRAIN_LINES):
        return [d.render(view, r if view == "operacional" else controlled(r)) for r in lines]

    def test_config(self):
        defaults = TemplateMinerConfig()
        exact, default = d.miner_config("exact"), d.miner_config("default")
        self.assertEqual(exact.drain_sim_th, 0.95)
        self.assertEqual(default.drain_sim_th, defaults.drain_sim_th)
        for config in (exact, default):
            self.assertFalse(config.parametrize_numeric_tokens)
            self.assertEqual(config.masking_instructions, [])
            self.assertEqual((config.drain_depth, config.drain_max_children),
                             (defaults.drain_depth, defaults.drain_max_children))

    def test_exact_keeps_categorical_tokens(self):
        miner = d.fit({"r": self.texts()}, d.miner_config("exact"))
        templates = [t for _, t, _ in d.snapshot(miner)]
        self.assertEqual(len(templates), len(set(self.texts())))
        self.assertFalse(any("<*>" in t for t in templates))
        merged = [t for _, t, _ in d.snapshot(d.fit({"r": self.texts()}, d.miner_config("default")))]
        self.assertTrue(any("<*>" in t for t in merged))

    def test_matching_does_not_learn_and_returns_unk(self):
        miner = d.fit({"r": self.texts()}, d.miner_config("exact"))
        before = d.snapshot(miner)
        ids = d.assign(miner, self.texts(lines=[UNSEEN, TRAIN_LINES[1], UNSEEN]), {})
        self.assertEqual(ids[0], d.UNK_ID)
        self.assertEqual(ids[2], d.UNK_ID)
        self.assertNotEqual(ids[1], d.UNK_ID)
        self.assertEqual(d.snapshot(miner), before)

    def test_fit_is_deterministic_and_state_round_trips(self):
        config = d.miner_config("exact")
        first, second = (d.fit({"b": self.texts(), "a": self.texts()[::-1]}, config) for _ in range(2))
        self.assertEqual(d.snapshot(first), d.snapshot(second))
        restored = d.load_state(d.save_state(first), config)
        self.assertEqual(d.snapshot(restored), d.snapshot(first))
        self.assertEqual(d.assign(restored, self.texts(), {}), d.assign(first, self.texts(), {}))


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.derived = Path(tmp.name) / "v1"
        self.runs = build_derived(self.derived)

    def run_variant(self, variant):
        return quiet(d.main, ["--variant", variant, "--derived", str(self.derived)])

    def test_outputs_verify_and_unk(self):
        for variant in d.VARIANTS:
            self.assertEqual(self.run_variant(variant)[0], 0)
            problems, info = v.verify(self.derived, variant)
            self.assertEqual(problems, [], variant)
            for view in d.VIEWS:
                self.assertEqual(info[view]["unk"]["train"], 0)
                self.assertEqual(info[view]["unk"]["validation"], 1)
                self.assertEqual(info[view]["unk"]["test"], 1)
        base = self.derived / "drain" / "exact"
        templates = json.loads((base / "operacional" / "templates.json").read_text(encoding="utf-8"))
        self.assertEqual(len(templates), len(TRAIN_LINES))
        self.assertEqual(sum(t["train_lines"] for t in templates), 3 * len(TRAIN_LINES))
        row = json.loads((base / "controlada" / "runs" / f"{sorted(self.runs)[0]}.jsonl").read_text().splitlines()[0])
        self.assertEqual(list(row), ["ts", "req", "service", "template_id"])
        config = json.loads((base / "drain_config.json").read_text(encoding="utf-8"))
        self.assertTrue(config["official"])
        self.assertEqual(config["packages"]["drain3"], "0.9.11")

    def test_refuses_overwrite(self):
        self.assertEqual(self.run_variant("exact")[0], 0)
        manifest = (self.derived / "drain" / "exact" / "manifest.sha256").read_bytes()
        code, err = self.run_variant("exact")
        self.assertEqual(code, 1)
        self.assertIn("refusing to overwrite", err)
        self.assertEqual((self.derived / "drain" / "exact" / "manifest.sha256").read_bytes(), manifest)

    def test_validation_and_test_runs_do_not_create_templates(self):
        self.assertEqual(self.run_variant("exact")[0], 0)
        templates = json.loads((self.derived / "drain" / "exact" / "operacional" / "templates.json").read_text())
        self.assertFalse(any("dependency_lookup_failed" in t["template"] for t in templates))

    def test_verifier_detects_forbidden_template_and_unk_in_train(self):
        self.assertEqual(self.run_variant("exact")[0], 0)
        path = self.derived / "drain" / "exact" / "controlada" / "templates.json"
        templates = json.loads(path.read_text(encoding="utf-8"))
        templates[0]["template"] += " stress cpu92"
        path.write_bytes(d.json_bytes(templates))
        train_run = next(r for r, s in sorted(self.runs.items()) if s == "train")
        run_path = self.derived / "drain" / "exact" / "operacional" / "runs" / f"{train_run}.jsonl"
        rows = [json.loads(line) for line in run_path.read_text().splitlines()]
        rows[0]["template_id"] = d.UNK_ID
        run_path.write_bytes("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows).encode())
        checks = {c for c, _ in v.verify(self.derived, "exact")[0]}
        self.assertTrue({"template_text", "unk_train", "manifest", "state", "determinism"} <= checks)


if __name__ == "__main__":
    unittest.main()

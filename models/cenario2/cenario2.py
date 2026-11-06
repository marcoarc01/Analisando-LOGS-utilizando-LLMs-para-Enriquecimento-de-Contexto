"""Scenario 2: fine-tuning of bert-base-uncased on the controlled view of dataset v2. Own implementation of the LUK fine-tuning
architecture: encoder, mean of the tokens under the attention mask, one linear layer with 2 outputs, cross entropy.
Stages, each a separate process: preregister, select (train and validation only), control (repetition and shuffled labels, validation
only) and evaluate (the test, once: a marker file is created before it is read and a second evaluation is refused).
Run from the repository root with C:\\tcc-venv\\bert: python -m models.cenario2.cenario2 <stage>."""
import argparse
import collections
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", r"C:\tcc-data\hf-cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from models.cenario1 import cenario1 as c1  # noqa: E402
from models.cenario1 import cenario1_v2 as c2  # noqa: E402
from models.common import data, data_v2, metrics, window_text as wt  # noqa: E402

VIEW = "controlada"
MODEL = "bert-base-uncased"
REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
LEARNING_RATES = (2e-5, 3e-5, 5e-5)
SEEDS = c1.SEEDS
EPOCHS = 20
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.1
MAX_GRAD_NORM = 1.0
NUM_LABELS = 2
THRESHOLD = 0.5
NO_DECAY = ("bias", "LayerNorm.weight")
RESULTS = data_v2.RESULTS_DIR / "cenario2"
CHECKPOINTS = Path(r"C:\tcc-data\checkpoints\cenario2")
CENARIO1 = data_v2.RESULTS_DIR / "cenario1"
PRE_REGISTER = "pre_registro.json"
TEST_MARKER = "test_access.json"
LATENCY_PASSES = 3
LATENCY_WARMUP = 5
PACKAGES = ("torch", "transformers", "huggingface_hub", "tokenizers", "safetensors", "numpy", "scipy", "scikit-learn", "psutil")
ERROR_COLUMNS = ["split", "run_id", "k", "condition", "label", "anomaly_type", "failure_lines", "weak_visible", "bert_positive_seeds",
                 "bert_wrong_seeds", "p_anomaly_mean", "rule", "rf_positive_seeds"]
REPRESENTATION_COLUMNS = ["rep_id", "in_train", "train_windows", "train_normal", "train_anomalous", "validation_windows", "test_windows",
                          "p_anomaly_mean", "p_anomaly_std", "text"]


def mean_pool(hidden, attention_mask):
    """Mean of the token vectors over the positions of the attention mask."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class WindowClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.hidden_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(mean_pool(hidden, attention_mask))


def pretrained_model():
    return WindowClassifier(AutoModel.from_pretrained(MODEL, revision=REVISION, local_files_only=True, add_pooling_layer=False))


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(derived_dir, split, templates, tokenizer):
    """Windows of a set of the C3, with their text, which must fit: nothing is ever truncated."""
    loaded = data_v2.read_set(derived_dir, VIEW, split, templates)
    loaded["text"] = [wt.window_text(w["templates"], templates) for w in loaded["windows"]]
    names = [w["window_id"] for w in loaded["windows"]]
    loaded["input_ids"] = tokenizer(loaded["text"], add_special_tokens=True, truncation=False, padding=False)["input_ids"]
    wt.check_limit([len(ids) for ids in loaded["input_ids"]], names)
    return loaded


def collate(sequences, pad_id, device):
    size = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), size), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), size), dtype=torch.long)
    for i, sequence in enumerate(sequences):
        ids[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[i, :len(sequence)] = 1
    return ids.to(device), mask.to(device)


def parameter_groups(model):
    """Weight decay on everything but biases and LayerNorm, as sentence_transformers does."""
    named = list(model.named_parameters())
    return [{"params": [p for n, p in named if not any(x in n for x in NO_DECAY)], "weight_decay": WEIGHT_DECAY},
            {"params": [p for n, p in named if any(x in n for x in NO_DECAY)], "weight_decay": 0.0}]


def predict(model, sequences, pad_id, device, y=None):
    """(probability of the anomalous class, mean loss per window)."""
    model.eval()
    probabilities, total = [], 0.0
    with torch.inference_mode():
        for start in range(0, len(sequences), BATCH_SIZE):
            ids, mask = collate(sequences[start:start + BATCH_SIZE], pad_id, device)
            logits = model(ids, mask)
            probabilities += torch.softmax(logits, dim=1)[:, 1].tolist()
            if y is not None:
                target = torch.tensor(y[start:start + BATCH_SIZE], device=device)
                total += nn.functional.cross_entropy(logits, target, reduction="sum").item()
    return probabilities, (total / len(sequences) if y is not None else None)


def better(candidate, best):
    """Best epoch: highest validation F1, then lowest validation loss, then the earliest."""
    return best is None or (candidate["validation_f1"], -candidate["validation_loss"]) > (best["validation_f1"], -best["validation_loss"])


def fit(train, validation, learning_rate, seed, epochs, build, pad_id, device):
    """Returns (history, best epoch, state of the best epoch, seconds)."""
    set_seed(seed)
    model = build().to(device)
    if next(model.parameters()).dtype != torch.float32:
        raise data.ModelError("the model must run in fp32")
    optimizer = torch.optim.AdamW(parameter_groups(model), lr=learning_rate)
    steps = epochs * math.ceil(len(train["y"]) / BATCH_SIZE)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_FRACTION * steps), steps)
    generator = torch.Generator().manual_seed(seed)
    history, best, best_state = [], None, None
    fit_seconds, eval_seconds = 0.0, 0.0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train["y"]), generator=generator).tolist()
        total = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            chosen = order[start:start + BATCH_SIZE]
            ids, mask = collate([train["input_ids"][i] for i in chosen], pad_id, device)
            target = torch.tensor([train["y"][i] for i in chosen], device=device)
            loss = nn.functional.cross_entropy(model(ids, mask), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item() * len(chosen)
        if device != "cpu":
            torch.cuda.synchronize()
        fit_seconds += time.perf_counter() - started
        started = time.perf_counter()
        probabilities, validation_loss = predict(model, validation["input_ids"], pad_id, device, validation["y"])
        eval_seconds += time.perf_counter() - started
        preds = [int(p >= THRESHOLD) for p in probabilities]
        row = {"epoch": epoch, "train_loss": total / len(order), "validation_loss": validation_loss,
               "validation_f1": float(f1_score(validation["y"], preds, zero_division=0))}
        history.append(row)
        if better(row, best):
            best = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return history, best, best_state, {"train": fit_seconds, "validation": eval_seconds, "steps": steps, "warmup_steps": int(WARMUP_FRACTION * steps)}


def run_name(learning_rate, seed):
    return f"lr{learning_rate:.0e}_seed{seed}"


def run_selection(derived_dir, checkpoints, learning_rates, seeds, epochs, build, tokenizer, device, log=print):
    """Every rate with every seed, on train and validation. A run with its files is not trained again."""
    templates = data_v2.read_templates(derived_dir, VIEW)
    train = load_split(derived_dir, "train", templates, tokenizer)
    validation = load_split(derived_dir, "validation", templates, tokenizer)
    data_v2.check_disjoint({"train": train["rows"], "validation": validation["rows"]})
    runs = []
    for learning_rate in learning_rates:
        for seed in seeds:
            folder = checkpoints / run_name(learning_rate, seed)
            if not (folder / "run.json").exists():
                history, best, state, seconds = fit(train, validation, learning_rate, seed, epochs, build, tokenizer.pad_token_id, device)
                folder.mkdir(parents=True, exist_ok=False)
                save_file(state, folder / "model.safetensors")
                record = {"learning_rate": learning_rate, "seed": seed, "best": best, "seconds": seconds, "history": history,
                          "checkpoint_sha256": data.sha256_bytes((folder / "model.safetensors").read_bytes())}
                (folder / "run.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
                log(f"{folder.name}: epoch {best['epoch']} F1={best['validation_f1']:.4f} loss={best['validation_loss']:.5f} "
                    f"train {seconds['train']:.1f}s")
            runs.append(json.loads((folder / "run.json").read_text(encoding="utf-8")))
    return runs


def choose_rate(runs):
    """Highest mean validation F1 of the best epochs; ties: lowest mean validation loss, then the order of the rates."""
    by_rate = collections.defaultdict(list)
    for run in runs:
        by_rate[run["learning_rate"]].append(run)
    table = [{"learning_rate": rate, "f1": metrics.mean_std([r["best"]["validation_f1"] for r in group]),
              "loss": metrics.mean_std([r["best"]["validation_loss"] for r in group]),
              "best_epochs": [r["best"]["epoch"] for r in group], "seeds": [r["seed"] for r in group],
              "train_seconds": sum(r["seconds"]["train"] for r in group)} for rate, group in by_rate.items()]
    chosen = min(table, key=lambda t: (-t["f1"]["mean"], t["loss"]["mean"]))
    return chosen, table


def environment():
    driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"python": sys.version.split()[0], "platform": platform.platform(), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": driver, "packages": {name: importlib.metadata.version(name) for name in PACKAGES}, "model": MODEL, "revision": REVISION}


def read_selection(base):
    return json.loads((base / "selection.json").read_text(encoding="utf-8"))


def preregister(base):
    """Written before any training; the clock of the machine is part of the record."""
    now = dt.datetime.now().astimezone()
    data.write_new(base / PRE_REGISTER, data.json_bytes({
        "created_local": now.isoformat(timespec="seconds"), "created_utc": now.astimezone(dt.timezone.utc).isoformat(timespec="seconds"),
        "written_before": "any training of Scenario 2",
        "view": VIEW, "representation": "distinct templates, no counts, fixed order by template id, joined by [SEP]",
        "expectations": {
            "so_lentidao": "recall 0 on the 4 slow-only test windows, or 73 false positives if the model marks that text as anomalous: "
                           "their text is identical to the text of 73 normal test windows",
            "todas": "F1 close to 0.931 over all anomalous windows"},
        "basis": {"cenario1_controlada_test_f1": 0.931, "regra_simples_test_f1": 0.931,
                  "windows_with_the_same_text": {"normal": 73, "slow_only": 4}, "source": "reports C4 and D1 (Part 2)"},
    }))
    return now


def agreement_tables(preds, rule, forest):
    per_seed = {}
    for seed, p in preds.items():
        per_seed[str(seed)] = {"rule": sum(a == b for a, b in zip(p, rule)), "rf": sum(a == b for a, b in zip(p, forest[seed]))}
    size = len(rule)
    return {"windows": size, "per_seed": per_seed,
            "rule_mean": statistics.fmean(s["rule"] for s in per_seed.values()), "rf_mean": statistics.fmean(s["rf"] for s in per_seed.values()),
            "windows_differing_from_rule_any_seed": sum(any(p[i] != rule[i] for p in preds.values()) for i in range(size))}


def read_predictions(base, name, windows):
    with open(base / "predictions" / VIEW / f"{name}.csv", encoding="utf-8", newline="") as f:
        by_id = {r["window_id"]: int(r["prediction"]) for r in data.csv.DictReader(f)}
    return [by_id[w["window_id"]] for w in windows]


def error_rows(split, loaded, details, preds, probabilities, rule, forest):
    rows = []
    seeds = list(preds)
    for i, row in enumerate(loaded["rows"]):
        y = loaded["y"][i]
        wrong = sum(preds[s][i] != y for s in seeds)
        if wrong or rule[i] != y:
            info = details[(split, int(row["line"]))]
            rows.append({"split": split, "run_id": row["run_id"], "k": row["k"], "condition": info["condition"], "label": y,
                         "anomaly_type": row["anomaly_type"], "failure_lines": info["failure_lines"],
                         "weak_visible": int(row["anomaly_type"] == c2.VISIBLE and int(info["failure_lines"]) <= c2.WEAK_LINES),
                         "bert_positive_seeds": sum(preds[s][i] for s in seeds), "bert_wrong_seeds": wrong,
                         "p_anomaly_mean": statistics.fmean(probabilities[s][i] for s in seeds), "rule": rule[i],
                         "rf_positive_seeds": sum(forest[s][i] for s in seeds)})
    return rows


def representation_rows(sets, probability_of):
    """One row per distinct text of the three sets, the training texts first, by number of windows."""
    train_count = collections.Counter(sets["train"]["text"])
    labels = collections.defaultdict(collections.Counter)
    for text, y in zip(sets["train"]["text"], sets["train"]["y"]):
        labels[text][y] += 1
    others = {s: collections.Counter(sets[s]["text"]) for s in ("validation", "test")}
    texts = [t for t, _ in train_count.most_common()]
    texts += sorted({t for s in others for t in others[s]} - set(train_count))
    rows = []
    for number, text in enumerate(texts, 1):
        p = probability_of[text]
        rows.append({"rep_id": f"R{number:02d}", "in_train": int(text in train_count), "train_windows": train_count[text], "train_normal": labels[text][0],
                     "train_anomalous": labels[text][1], "validation_windows": others["validation"][text], "test_windows": others["test"][text],
                     "p_anomaly_mean": statistics.fmean(p), "p_anomaly_std": statistics.stdev(p) if len(p) > 1 else 0.0, "text": text})
    return rows


def measure_inference(model, tokenizer, texts, device):
    """Batch 1, tokenization and forward pass of each window, on a model already in memory."""
    model.eval()
    if device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    process.cpu_percent(None)
    psutil.cpu_percent(None)
    cpu_before = process.cpu_times()
    started = time.perf_counter()
    latencies = []
    with torch.inference_mode():
        for pass_number in range(LATENCY_PASSES):
            for position, text in enumerate(texts):
                begin = time.perf_counter()
                encoded = tokenizer(text, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
                ids, mask = collate([encoded], tokenizer.pad_token_id, device)
                model(ids, mask)
                if device != "cpu":
                    torch.cuda.synchronize()
                elapsed = (time.perf_counter() - begin) * 1000
                if pass_number or position >= LATENCY_WARMUP:
                    latencies.append(elapsed)
    wall = time.perf_counter() - started
    cpu_after = process.cpu_times()
    return {"windows_measured": len(latencies), "passes": LATENCY_PASSES, "discarded_first": LATENCY_WARMUP,
            "latency_ms_mean": statistics.fmean(latencies), "latency_ms_median": statistics.median(latencies),
            "latency_ms_p95": float(np.percentile(latencies, 95)), "latency_ms_max": max(latencies),
            "peak_vram_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if device != "cpu" else None,
            "peak_vram_reserved_mb": torch.cuda.max_memory_reserved() / 2**20 if device != "cpu" else None,
            "process_cpu_core_percent": 100 * ((cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)) / wall,
            "system_cpu_percent": psutil.cpu_percent(None), "logical_cores": psutil.cpu_count()}


def shuffled_control(train, validation, learning_rate, epochs, build, pad_id, device):
    y = np.random.default_rng(c1.SHUFFLE_SEED).permutation(train["y"]).tolist()
    history, best, _, _ = fit({**train, "y": y}, validation, learning_rate, c1.SHUFFLE_SEED, epochs, build, pad_id, device)
    return {"seed": c1.SHUFFLE_SEED, "train_labels_changed": sum(a != b for a, b in zip(y, train["y"])), "best_epoch": best,
            "last_epoch": history[-1], "note": "the best epoch is picked on validation, so its F1 is an optimistic reading of the control"}


def repetition(train, validation, chosen_rate, seed, epochs, build, pad_id, device, checkpoints):
    """The same run again, compared with the saved one on validation only."""
    history, best, state, seconds = fit(train, validation, chosen_rate, seed, epochs, build, pad_id, device)
    folder = checkpoints / run_name(chosen_rate, seed)
    saved = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    first = load_file(folder / "model.safetensors")
    largest = max(float((first[k].float() - state[k].float()).abs().max()) for k in first)
    return {"learning_rate": chosen_rate, "seed": seed, "best_epoch": [saved["best"]["epoch"], best["epoch"]],
            "history_identical": saved["history"] == history, "weights_identical": all(torch.equal(first[k], state[k]) for k in first),
            "largest_weight_difference": largest,
            "largest_validation_loss_difference": max(abs(a["validation_loss"] - b["validation_loss"]) for a, b in zip(saved["history"], history)),
            "largest_train_loss_difference": max(abs(a["train_loss"] - b["train_loss"]) for a, b in zip(saved["history"], history)),
            "validation_f1_per_epoch_identical": [a["validation_f1"] for a in saved["history"]] == [b["validation_f1"] for b in history],
            "train_seconds": [saved["seconds"]["train"], seconds["train"]]}


def stage_select(args, build=pretrained_model):
    base, device = args.output, args.device
    if not (base / PRE_REGISTER).exists():
        raise data.ModelError("the pre-registration must exist before any training")
    tokenizer = load_tokenizer()
    runs = run_selection(args.derived, args.checkpoints, LEARNING_RATES, SEEDS, EPOCHS, build, tokenizer, device)
    chosen, table = choose_rate(runs)
    data.write_new(base / "selection.json", data.json_bytes({
        "chosen": chosen, "table": table, "runs": runs, "rule": "highest mean validation F1 of the best epochs; ties -> lowest mean validation loss, then order of the rates",
        "best_epoch": "highest validation F1, then lowest validation loss, then earliest"}))
    print(f"chosen rate {chosen['learning_rate']} F1 {chosen['f1']['mean']:.4f} loss {chosen['loss']['mean']:.5f} epochs {chosen['best_epochs']}")


def stage_control(args, build=pretrained_model):
    base, device = args.output, args.device
    chosen = read_selection(base)["chosen"]
    tokenizer = load_tokenizer()
    templates = data_v2.read_templates(args.derived, VIEW)
    train = load_split(args.derived, "train", templates, tokenizer)
    validation = load_split(args.derived, "validation", templates, tokenizer)
    rate = chosen["learning_rate"]
    repeated = repetition(train, validation, rate, SEEDS[0], EPOCHS, build, tokenizer.pad_token_id, device, args.checkpoints)
    shuffled = shuffled_control(train, validation, rate, EPOCHS, build, tokenizer.pad_token_id, device)
    data.write_new(base / "controls.json", data.json_bytes({"reproducibility": repeated, "shuffled_labels_validation": shuffled}))
    print(json.dumps({k: v for k, v in repeated.items() if k != "train_seconds"}))
    print(json.dumps(shuffled["best_epoch"]), json.dumps(shuffled["last_epoch"]))


def stage_evaluate(args, build=pretrained_model):
    base, device = args.output, args.device
    selection = read_selection(base)
    controls = json.loads((base / "controls.json").read_text(encoding="utf-8"))
    rate = selection["chosen"]["learning_rate"]
    with open(base / TEST_MARKER, "x", encoding="utf-8") as f:  # a second evaluation stops here
        json.dump({"opened_local": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "learning_rate": rate}, f)
    tokenizer = load_tokenizer()
    templates = data_v2.read_templates(args.derived, VIEW)
    details = data_v2.label_rows(args.derived)
    sets = {s: load_split(args.derived, s, templates, tokenizer) for s in ("train", "validation")}
    gate = c2.TestGate()
    test = gate.read(args.derived, VIEW, templates, rate)
    test["text"] = [wt.window_text(w["templates"], templates) for w in test["windows"]]
    test["input_ids"] = tokenizer(test["text"], add_special_tokens=True, truncation=False, padding=False)["input_ids"]
    wt.check_limit([len(ids) for ids in test["input_ids"]], [w["window_id"] for w in test["windows"]])
    sets["test"] = test
    runs_total = data_v2.check_disjoint({s: sets[s]["rows"] for s in sets})

    texts = sorted({t for s in sets for t in sets[s]["text"]})
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
    probability_of = {t: [] for t in texts}
    probabilities, preds, cuts, predictions, by_condition = {"validation": {}, "test": {}}, {"validation": {}, "test": {}}, {}, {}, {}
    conditions = [details[("test", int(r["line"]))]["condition"] for r in test["rows"]]
    model = build().to(device)
    pad = tokenizer.pad_token_id
    for seed in SEEDS:
        folder = args.checkpoints / run_name(rate, seed)
        model.load_state_dict(load_file(folder / "model.safetensors"))
        for split in ("validation", "test"):
            p, _ = predict(model, sets[split]["input_ids"], pad, device)
            probabilities[split][seed] = p
            preds[split][seed] = [int(x >= THRESHOLD) for x in p]
            predictions[f"predictions/bert_{seed}_{split}.csv"] = data.csv_bytes(c1.PREDICTION_COLUMNS, c1.prediction_rows(sets[split]["windows"], p, preds[split][seed]))
        cuts[str(seed)] = c2.cut_metrics(test["y"], probabilities["test"][seed], preds["test"][seed], test["rows"])
        by_condition[str(seed)] = metrics.by_condition(test["y"], preds["test"][seed], conditions)
        for text, p in zip(texts, predict(model, encoded, pad, device)[0]):
            probability_of[text].append(p)
    efficiency = measure_inference(model, tokenizer, test["text"], device)

    rule = {s: read_predictions(CENARIO1, f"{c1.RULE}_none_{s}", sets[s]["windows"]) for s in ("validation", "test")}
    forest = {s: {seed: read_predictions(CENARIO1, f"{c1.RF}_{seed}_{s}", sets[s]["windows"]) for seed in SEEDS} for s in ("validation", "test")}
    errors = [row for s in ("validation", "test") for row in error_rows(s, sets[s], details, preds[s], probabilities[s], rule[s], forest[s])]
    representations = representation_rows(sets, probability_of)
    summary = c2.summarize_cuts(list(cuts.values()))
    slow = summary["so_lentidao"]
    results = {
        "learning_rate": rate, "best_epochs": selection["chosen"]["best_epochs"],
        "validation": {str(seed): metrics.binary_metrics(sets["validation"]["y"], probabilities["validation"][seed], preds["validation"][seed]) for seed in SEEDS},
        "test_cuts": {"summary": summary, "per_seed": cuts},
        "test_by_condition": metrics.summarize_conditions(list(by_condition.values())),
        "agreement_test": agreement_tables(preds["test"], rule["test"], forest["test"]),
        "agreement_validation": agreement_tables(preds["validation"], rule["validation"], forest["validation"]),
        "expectation_check": {"slow_only_recall": slow["recall"], "slow_only_false_positives_sum": slow["confusion_matrix_sum"]["fp"],
                              "f1_all": summary["todas"]["f1"], "registered_f1": 0.931},
        "efficiency": efficiency,
        "training_seconds": {"all_runs": sum(r["seconds"]["train"] for r in selection["runs"]),
                             "all_runs_with_validation": sum(r["seconds"]["train"] + r["seconds"]["validation"] for r in selection["runs"]),
                             "per_run": {f"{r['learning_rate']:.0e}/{r['seed']}": r["seconds"]["train"] for r in selection["runs"]},
                             "chosen_rate_runs": selection["chosen"]["train_seconds"]},
        "execucoes_sem_sobreposicao": runs_total,
    }
    files = dict(predictions)
    files["metrics.json"] = data.json_bytes(results)
    files["errors.csv"] = data.csv_bytes(ERROR_COLUMNS, errors)
    files["representations.csv"] = data.csv_bytes(REPRESENTATION_COLUMNS, representations)
    files["config.json"] = data.json_bytes({
        "environment": environment(), "inputs": data_v2.verify_inputs(args.derived), "view": VIEW,
        "model": {"encoder": MODEL, "revision": REVISION, "pooling": "mean of the tokens under the attention mask, [CLS] and [SEP] included",
                  "head": "one linear layer with 2 outputs, no dropout of its own", "loss": "cross entropy without class weights",
                  "encoder_dropout": "bert-base-uncased default (0.1) in training", "pooler_layer": "not created"},
        "training": {"optimizer": "torch.optim.AdamW, default betas and eps", "weight_decay": WEIGHT_DECAY, "no_decay": list(NO_DECAY),
                     "warmup_fraction": WARMUP_FRACTION, "schedule": "linear warmup, then linear decay, step by step", "max_grad_norm": MAX_GRAD_NORM,
                     "batch_size": BATCH_SIZE, "max_tokens": wt.MAX_TOKENS, "truncation": False, "epochs": EPOCHS, "precision": "fp32, no mixed precision",
                     "learning_rates": LEARNING_RATES, "seeds": list(SEEDS), "threshold": THRESHOLD},
        "test_reads": dict(gate.reads), "checkpoints": str(args.checkpoints),
        "sets": "labels/sets of the C3: no warm-up or excluded windows; slow-only windows only in test",
    })
    for name, content in files.items():
        data.write_new(base / name, content)
    every = {name: (base / name).read_bytes() for name in (PRE_REGISTER, TEST_MARKER, "selection.json", "controls.json")}
    data.write_new(base / "manifest.sha256", data.manifest_bytes({**files, **every}))
    print(f"written: {base}; test reads {dict(gate.reads)}; reproducibility weights identical {controls['reproducibility']['weights_identical']}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preregister", "select", "control", "evaluate"))
    parser.add_argument("--derived", type=Path, default=data_v2.DERIVED_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        if args.stage == "preregister":
            print(f"pre-registered at {preregister(args.output).isoformat(timespec='seconds')}")
        elif args.stage == "select":
            stage_select(args)
        elif args.stage == "control":
            stage_control(args)
        else:
            stage_evaluate(args)
    except (data.ModelError, wt.WindowTooLong, OSError, KeyError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

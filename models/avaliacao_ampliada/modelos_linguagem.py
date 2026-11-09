"""E2 (and D5): fine-tuning of a language-model encoder in the extended evaluation of the E0 pre-registration: diagnosis (4 classes)
with full data, detection with full data and the 15 low-resource draws of both tasks. The encoder is a parameter: bert-base-uncased in
the E2, the checkpoints of the 2b and of the LUK in the D5 (--model bert_mlm_2b|bert_luk --encoder <path> --output ... --checkpoints ...).

Stages, each a separate process, run from the repository root with C:\\tcc-venv\\bert:
  python -m models.avaliacao_ampliada.modelos_linguagem select    GPU free, hashes; full-data runs (seeds 42-46, best epoch on validation, checkpoints written) and the
                                   shuffled-labels control; validation results and analysis; writes selection.json. The test is not opened.
  python -m models.avaliacao_ampliada.modelos_linguagem evaluate  validation of the saved checkpoints recomputed and compared with select (before the test); writes
                                   test_access.json and refuses a second evaluation; reads the test once; full-data runs on the two tests;
                                   latency; full-data detection copied from the D1 (BERT-base); low-resource runs, each trained with its
                                   best point chosen on validation and only then evaluated on the tests (their weights are never written).
No statistical test is run here (D6). Outputs are never overwritten."""
import argparse
import datetime as dt
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from models.cenario2 import cenario2 as c2  # sets HF_HUB_OFFLINE, HF_HOME and CUBLAS_WORKSPACE_CONFIG before torch is used

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoModel, get_linear_schedule_with_warmup  # noqa: E402

from models.cenario1 import cenario1 as c1  # noqa: E402
from models.cenario1 import cenario1_v2 as c1v2  # noqa: E402
from models.common import data, data_v2, metrics, window_text as wt  # noqa: E402
from models.avaliacao_ampliada import classicos  # noqa: E402

VIEW = classicos.VIEW
CLASSES = classicos.CLASSES
DETECTION, DIAGNOSIS, FULL = classicos.DETECTION, classicos.DIAGNOSIS, classicos.FULL
TASKS = (DETECTION, DIAGNOSIS)
N_LABELS = {DETECTION: 2, DIAGNOSIS: len(CLASSES)}
LEARNING_RATE = 5e-5
STEPS, EVAL_EVERY, WARMUP_STEPS = 780, 39, 78
BERT = "bert_base"
MODELS = ("bert_base", "bert_mlm_2b", "bert_luk")
DEFAULT_PATHS = {BERT: (data_v2.RESULTS_DIR / "e2", Path(r"C:\tcc-data\checkpoints\e2-bert-diag"))}
D1_DIR = data_v2.RESULTS_DIR / "cenario2"
E1_DIR = data_v2.RESULTS_DIR / "e1"
SELECTION_FILE, TEST_MARKER = "selection.json", "test_access.json"
LOW_DIR = "baixos_recursos"
PROBABILITY_TOLERANCE = 1e-6


# ---------------------------------------------------------------- GPU and pre-registration
def gpu_processes():
    """(pid, type, name) of every process on the GPU, from nvidia-smi -q -d PIDS. Under WDDM the desktop applications appear
    as C+G (graphics); a training or inference process appears as C (compute only)."""
    out = subprocess.run(["nvidia-smi", "-q", "-d", "PIDS"], capture_output=True, text=True, check=True).stdout
    processes, current = [], {}
    for line in out.splitlines():
        key, _, value = (part.strip() for part in line.partition(":"))
        if key == "Process ID":
            current = {"pid": int(value)}
            processes.append(current)
        elif key in ("Type", "Name") and current:
            current[key.lower()] = value
    return processes


def busy_processes(processes):
    return [p for p in processes if p.get("type") == "C" or "python" in p.get("name", "").lower()]


def check_gpu_free():
    processes = gpu_processes()
    busy = busy_processes(processes)
    if busy:
        raise data.ModelError(f"another process is using the GPU: {busy}")
    used = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return {"compute_processes": [], "graphics_processes": len(processes), "gpu": used,
            "checked_local": dt.datetime.now().astimezone().isoformat(timespec="seconds")}


def check_checkpoint_manifest(path, manifest):
    """Every file listed in the model.sha256 of a pre-training checkpoint (lines '<sha>  *model/<file>')."""
    base = Path(manifest).parent
    checked = 0
    for line in Path(manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        if classicos.sha(base / name.lstrip("*")) != digest:
            raise data.ModelError(f"{name}: differs from {manifest}")
        checked += 1
    return checked


def check_fine_tuning(pre, model_name, encoder):
    """What this stage runs must be the fine-tuning of the pre-registration, which is the D1 configuration."""
    ft = pre["hiperparametros"]["fine_tuning_bert_2b_luk"]
    low = pre["orcamento_baixos_recursos"]["bert_2b_luk"]
    d1 = json.loads((D1_DIR / "config.json").read_text(encoding="utf-8"))["training"]
    d1_chosen = json.loads((D1_DIR / "selection.json").read_text(encoding="utf-8"))["chosen"]["learning_rate"]
    d1_marker = json.loads((D1_DIR / "test_access.json").read_text(encoding="utf-8"))["learning_rate"]
    steps_per_epoch = math.ceil(ft["full_data_budget"]["train_windows"] / c2.BATCH_SIZE)
    checks = {
        "learning rate 5e-5 = D1 choice": ft["learning_rate"] == LEARNING_RATE == d1_chosen == d1_marker,
        "epochs": ft["epochs"] == c2.EPOCHS == d1["epochs"],
        "batch size": ft["batch_size"] == c2.BATCH_SIZE == d1["batch_size"],
        "weight decay": ft["weight_decay"] == c2.WEIGHT_DECAY == d1["weight_decay"],
        "no decay": ft["no_decay"] == list(c2.NO_DECAY) == d1["no_decay"],
        "warm-up fraction": ft["warmup_fraction"] == c2.WARMUP_FRACTION == d1["warmup_fraction"],
        "schedule": ft["schedule"] == d1["schedule"],
        "gradient clip": ft["max_grad_norm"] == c2.MAX_GRAD_NORM == d1["max_grad_norm"],
        "precision fp32": ft["precision"] == d1["precision"] == "fp32, no mixed precision",
        "512 tokens, no truncation": ft["max_tokens"] == wt.MAX_TOKENS == d1["max_tokens"] and ft["truncation"] is False is d1["truncation"],
        "optimizer": ft["optimizer"] == d1["optimizer"],
        "pooling": "mean of the tokens under the attention mask" in ft["pooling"],
        "loss": ft["loss"] == "cross entropy without class weights",
        "detection threshold": "0.5" in ft["decision"]["deteccao"] and c2.THRESHOLD == c1.RF_THRESHOLD == 0.5,
        "diagnosis argmax": "argmax" in ft["decision"]["diagnostico"],
        "epoch selection": "F1 of the anomalous class" in ft["epoch_selection"]["deteccao"]
                           and "Macro-F1" in ft["epoch_selection"]["diagnostico"]
                           and "lowest validation loss" in ft["epoch_selection"]["diagnostico"] and "earliest" in ft["epoch_selection"]["diagnostico"],
        "full-data budget": ft["full_data_budget"]["steps_per_epoch"] == steps_per_epoch == EVAL_EVERY
                            and ft["full_data_budget"]["steps"] == c2.EPOCHS * steps_per_epoch == STEPS
                            and ft["full_data_budget"]["warmup_steps"] == int(c2.WARMUP_FRACTION * STEPS) == WARMUP_STEPS
                            and ft["full_data_budget"]["evaluation_points"] == STEPS // EVAL_EVERY,
        "low-resource budget": (low["steps"], low["evaluation_every_steps"], low["warmup_steps"], low["evaluation_points"])
                               == (STEPS, EVAL_EVERY, WARMUP_STEPS, STEPS // EVAL_EVERY),
        "seeds": pre["sementes"]["dados_completos"] == list(c2.SEEDS) == d1["seeds"],
        "tokenizer": c2.REVISION in ft["tokenizer"] and c2.MODEL in ft["tokenizer"],
        "BERT encoder": c2.REVISION in ft["encoders"]["bert"] and "add_pooling_layer=False" in ft["encoders"]["bert"],
        "classes": pre["tarefas"]["diagnostico"]["classes"] == list(CLASSES),
    }
    if model_name == BERT:
        checks["encoder argument"] = encoder == c2.MODEL
    else:
        checks["encoder argument"] = Path(encoder).resolve() == Path(pre["modelos"][model_name]["checkpoint"]["path"]).resolve()
    problems = [name for name, ok in checks.items() if not ok]
    if problems:
        raise data.ModelError(f"E2 differs from the pre-registration: {problems}")
    return sorted(checks)


def verify(model_name, encoder):
    pre, verification = classicos.verify_preregistration()
    verification["files"]["results/cenario2 (manifest)"] = data.verify_manifest(D1_DIR)
    verification["files"]["results/e1 (manifest)"] = data.verify_manifest(E1_DIR)
    if model_name != BERT:
        ck = pre["modelos"][model_name]["checkpoint"]
        if check_checkpoint_manifest(ck["path"], ck["manifest"]) != ck["files_checked"]:
            raise data.ModelError(f"{model_name}: number of checkpoint files differs from the pre-registration")
    verification["fine_tuning_agreement"] = check_fine_tuning(pre, model_name, encoder)
    return pre, verification


# ---------------------------------------------------------------- data
def tokenize(loaded, templates, tokenizer):
    loaded["text"] = [wt.window_text(w["templates"], templates) for w in loaded["windows"]]
    loaded["input_ids"] = tokenizer(loaded["text"], add_special_tokens=True, truncation=False, padding=False)["input_ids"]
    wt.check_limit([len(ids) for ids in loaded["input_ids"]], [w["window_id"] for w in loaded["windows"]])
    return loaded


def load_train_validation(tokenizer):
    causes = classicos.cause_rows()
    templates, sets = classicos.load_train_validation(causes)
    for s in sets.values():
        tokenize(s, templates, tokenizer)
    return causes, templates, sets


def evaluated_sets(sets, include_test):
    out = classicos.evaluated_sets(sets, include_test)
    if include_test:
        principal = [i for i, p in enumerate(sets["test"]["principal"]) if p]
        out["test_principal"]["input_ids"] = [sets["test"]["input_ids"][i] for i in principal]
        out["test_principal"]["text"] = [sets["test"]["text"][i] for i in principal]
    return out


def subset(loaded, indexes):
    out = classicos.subset(loaded, indexes)
    out["input_ids"] = [loaded["input_ids"][i] for i in indexes]
    return out


def labels_of(loaded, task):
    return loaded["y"] if task == DETECTION else loaded["cause"]


# ---------------------------------------------------------------- model and training
class Classifier(nn.Module):
    """cenario2.WindowClassifier with the number of outputs of the task: encoder, mean pooling, one linear layer."""

    def __init__(self, encoder, n_labels):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(c2.mean_pool(hidden, attention_mask))


def encoder_loader(encoder):
    """bert-base-uncased at the pinned revision, or a local checkpoint folder (2b, LUK); never with a pooler layer."""
    if encoder == c2.MODEL:
        return lambda: AutoModel.from_pretrained(c2.MODEL, revision=c2.REVISION, local_files_only=True, add_pooling_layer=False)
    return lambda: AutoModel.from_pretrained(str(encoder), local_files_only=True, add_pooling_layer=False)


def batches(size, steps, generator, batch_size=c2.BATCH_SIZE):
    """Indexes of each step: successive passes over the training windows, each a new torch.randperm order, batches of up to
    batch_size that never cross passes. With 307 windows a pass is the epoch of the D1."""
    step = 0
    while True:
        order = torch.randperm(size, generator=generator).tolist()
        for start in range(0, size, batch_size):
            yield order[start:start + batch_size]
            step += 1
            if step == steps:
                return


def predict(model, sequences, pad_id, device, y=None):
    """(softmax probabilities, one row per window; mean cross entropy when y is given). Batches of 8, as the D1."""
    model.eval()
    out, total = [], 0.0
    with torch.inference_mode():
        for start in range(0, len(sequences), c2.BATCH_SIZE):
            ids, mask = c2.collate(sequences[start:start + c2.BATCH_SIZE], pad_id, device)
            logits = model(ids, mask)
            out.append(torch.softmax(logits, dim=1).cpu())
            if y is not None:
                target = torch.tensor(y[start:start + c2.BATCH_SIZE], device=device)
                total += nn.functional.cross_entropy(logits, target, reduction="sum").item()
    return torch.cat(out).double().numpy(), (total / len(sequences) if y is not None else None)


def main_metric(task, y, preds):
    """F1 of the anomalous class (detection) or Macro-F1 over the 4 classes (diagnosis)."""
    if task == DETECTION:
        return float(f1_score(y, preds, zero_division=0))
    return float(f1_score(y, preds, labels=list(range(len(CLASSES))), average="macro", zero_division=0))


def better(candidate, best):
    """Highest validation main metric, then lowest validation loss, then the earliest point."""
    return best is None or (candidate["validation_metric"], -candidate["validation_loss"]) > (best["validation_metric"], -best["validation_loss"])


def train_run(task, train, validation, seed, build, pad_id, device, steps=STEPS, every=EVAL_EVERY, warmup=WARMUP_STEPS,
              learning_rate=LEARNING_RATE):
    """One fine-tuning run. Returns (record, best state on the CPU, validation probabilities of the best point, model on the device)."""
    y_train, y_val = labels_of(train, task), labels_of(validation, task)
    if device != "cpu":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started_run = time.perf_counter()
    c2.set_seed(seed)
    model = Classifier(build(), N_LABELS[task]).to(device)
    if next(model.parameters()).dtype != torch.float32:
        raise data.ModelError("the model must run in fp32")
    optimizer = torch.optim.AdamW(c2.parameter_groups(model), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, steps)
    generator = torch.Generator().manual_seed(seed)
    history, best, best_state, best_probabilities = [], None, None, None
    fit_seconds = eval_seconds = 0.0
    total, seen = 0.0, 0
    model.train()
    started = time.perf_counter()
    for step, chosen in enumerate(batches(len(y_train), steps, generator), 1):
        ids, mask = c2.collate([train["input_ids"][i] for i in chosen], pad_id, device)
        target = torch.tensor([y_train[i] for i in chosen], device=device)
        loss = nn.functional.cross_entropy(model(ids, mask), target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), c2.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total += loss.item() * len(chosen)
        seen += len(chosen)
        if step % every == 0:
            if device != "cpu":
                torch.cuda.synchronize()
            fit_seconds += time.perf_counter() - started
            started = time.perf_counter()
            probabilities, validation_loss = predict(model, validation["input_ids"], pad_id, device, y_val)
            eval_seconds += time.perf_counter() - started
            row = {"point": step // every, "step": step, "train_loss": total / seen, "validation_loss": validation_loss,
                   "validation_metric": main_metric(task, y_val, classicos.decide(probabilities, task))}
            history.append(row)
            if better(row, best):
                best = row
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_probabilities = probabilities
            total, seen = 0.0, 0
            model.train()
            started = time.perf_counter()
    record = {"task": task, "seed": seed, "train_windows": len(y_train), "steps": steps, "evaluation_every": every, "warmup_steps": warmup,
              "learning_rate": learning_rate, "best": best, "history": history,
              "seconds": {"train": fit_seconds, "validation": eval_seconds, "run_total": time.perf_counter() - started_run},
              "peak_vram_mb": ({"allocated": torch.cuda.max_memory_allocated() / 2**20, "reserved": torch.cuda.max_memory_reserved() / 2**20}
                               if device != "cpu" else None)}
    return record, best_state, best_probabilities, model


def evaluate_model(model, task, sets_eval, pad_id, device):
    probabilities = {name: predict(model, s["input_ids"], pad_id, device)[0] for name, s in sets_eval.items()}
    preds = {name: classicos.decide(p, task) for name, p in probabilities.items()}
    result = (classicos.detection_metrics if task == DETECTION else classicos.diagnosis_metrics)(sets_eval, probabilities, preds)
    return result, probabilities, preds


def compare_probabilities(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return {"max_abs_difference": float(np.max(np.abs(a - b))), "same_argmax": bool((a.argmax(1) == b.argmax(1)).all()),
            "same_threshold": bool(((a[:, -1] >= 0.5) == (b[:, -1] >= 0.5)).all()) if a.shape[1] == 2 else None}


# ---------------------------------------------------------------- full-data runs (select)
def full_tasks(model_name, detection_mode):
    return (DIAGNOSIS,) if detection_mode == "copy" else (DETECTION, DIAGNOSIS)


def run_folder(checkpoints, task, seed):
    return checkpoints / f"{task}_seed{seed}"


def full_data_runs(tasks, sets, checkpoints, build, pad_id, device, log):
    """Each run with its files is not trained again (validation only, as the D1 selection)."""
    records = []
    for task in tasks:
        for seed in c2.SEEDS:
            folder = run_folder(checkpoints, task, seed)
            if not (folder / "run.json").exists():
                record, state, best_probabilities, model = train_run(task, sets["train"], sets["validation"], seed, build, pad_id, device)
                del model
                folder.mkdir(parents=True, exist_ok=False)
                save_file(state, folder / "model.safetensors")
                record["checkpoint_sha256"] = classicos.sha(folder / "model.safetensors")
                record["training_validation_probabilities"] = best_probabilities.tolist()
                data.write_new(folder / "run.json", data.json_bytes(record))
                log(f"{task} seed {seed}: point {record['best']['point']} metric {record['best']['validation_metric']:.4f} "
                    f"loss {record['best']['validation_loss']:.5f} train {record['seconds']['train']:.1f}s VRAM {record['peak_vram_mb']}")
            records.append(json.loads((folder / "run.json").read_text(encoding="utf-8")))
    return records


def load_checkpoint(model, folder, expected_sha):
    if classicos.sha(folder / "model.safetensors") != expected_sha:
        raise data.ModelError(f"{folder}: checkpoint differs from its run.json")
    model.load_state_dict(load_file(folder / "model.safetensors"))
    return model


def shuffled_control(sets, build, pad_id, device):
    """Training causes permuted with numpy default_rng(42), as the C4 and E1 controls; trained with seed 42; validation only."""
    y = np.random.default_rng(c1.SHUFFLE_SEED).permutation(sets["train"]["cause"]).tolist()
    train = dict(sets["train"], cause=y)
    record, _, probabilities, model = train_run(DIAGNOSIS, train, sets["validation"], c1.SHUFFLE_SEED, build, pad_id, device)
    del model
    preds = classicos.decide(probabilities, DIAGNOSIS)
    return {"seed": c1.SHUFFLE_SEED, "train_labels_changed": sum(a != b for a, b in zip(y, sets["train"]["cause"])),
            "best_point": record["best"], "last_point": record["history"][-1], "validation_at_best_point": classicos.multiclass_metrics(
                sets["validation"]["cause"], probabilities, preds), "seconds": record["seconds"], "peak_vram_mb": record["peak_vram_mb"],
            "note": "the best point is picked on validation, so its Macro-F1 is an optimistic reading of the control"}


def e1_validation_comparison():
    """The validation analysis of the E1 (TF-IDF + RF presence only receives the same information as the language models)."""
    return json.loads((E1_DIR / SELECTION_FILE).read_text(encoding="utf-8"))["validation_analysis"]


def stage_select(args):
    started = time.perf_counter()
    gpu = check_gpu_free()
    pre, verification = verify(args.model, args.encoder)
    if (args.output / SELECTION_FILE).exists():
        raise data.ModelError("selection.json already exists")
    tokenizer = c2.load_tokenizer()
    _, templates, sets = load_train_validation(tokenizer)
    build = encoder_loader(args.encoder)
    pad = tokenizer.pad_token_id
    tasks = full_tasks(args.model, args.detection_full)
    runs = full_data_runs(tasks, sets, args.checkpoints, build, pad, args.device, log)
    control = shuffled_control(sets, build, pad, args.device)
    log(f"shuffled control: Macro-F1 {control['validation_at_best_point']['macro_f1']:.4f} at point {control['best_point']['point']}")
    results, predictions, consistency = [], [], {}
    model = None
    validation_only = {"validation": sets["validation"]}
    for record in runs:
        task, seed = record["task"], record["seed"]
        if model is None or model.classifier.out_features != N_LABELS[task]:
            model = Classifier(build(), N_LABELS[task]).to(args.device)
        load_checkpoint(model, run_folder(args.checkpoints, task, seed), record["checkpoint_sha256"])
        result, probabilities, preds = evaluate_model(model, task, validation_only, pad, args.device)
        consistency[f"{task}/{seed}"] = compare_probabilities(probabilities["validation"], record["training_validation_probabilities"])
        if main_metric(task, labels_of(sets["validation"], task), preds["validation"]) != record["best"]["validation_metric"]:
            raise data.ModelError(f"{task}/{seed}: the saved checkpoint does not reproduce its best validation point")
        key = {"tarefa": task, "modelo": args.model, "regime": FULL, "sorteio": "", "semente": seed}
        results.append({**key, "params": None, "best": record["best"], "metrics": result})
        predictions += classicos.prediction_rows(task, key, validation_only, probabilities, preds, origin="e2")
    del model
    analysis = classicos.validation_analysis(sets, templates, [r for r in results if r["tarefa"] == DIAGNOSIS],
                                      [p for p in predictions if p["tarefa"] == DIAGNOSIS])
    record = {"created_local": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "model": args.model, "encoder": str(args.encoder),
              "checkpoints": str(args.checkpoints), "gpu_check": gpu, "verification": verification, "environment": c2.environment(),
              "full_data_runs": [{k: v for k, v in r.items() if k != "training_validation_probabilities"} for r in runs],
              "checkpoint_reload_vs_training": consistency, "shuffled_labels_control": control, "validation_results": results,
              "validation_aggregated": classicos.aggregate(results), "validation_predictions": predictions,
              "validation_analysis": analysis, "e1_validation_analysis": e1_validation_comparison(),
              "seconds": time.perf_counter() - started}
    data.write_new(args.output / SELECTION_FILE, data.json_bytes(record))
    log(json.dumps({"seconds": record["seconds"], "best_points": {f"{r['task']}/{r['seed']}": r["best"]["point"] for r in runs}}))


# ---------------------------------------------------------------- detection with full data copied from the D1
def d1_detection(sets_eval, model_name):
    m = json.loads((D1_DIR / "metrics.json").read_text(encoding="utf-8"))
    copied = {"tarefa": DETECTION, "modelo": model_name, "regime": FULL, "runs": len(c2.SEEDS), "origem": str(D1_DIR / "metrics.json"),
              "summary": {"validation": metrics.summarize([m["validation"][str(s)] for s in c2.SEEDS]), "test_cuts": m["test_cuts"]["summary"]},
              "majority_classes": None, "best_epochs_d1": m["best_epochs"], "learning_rate_d1": m["learning_rate"]}
    runs = [{"tarefa": DETECTION, "modelo": model_name, "regime": FULL, "sorteio": "", "semente": s, "origem": str(D1_DIR / "metrics.json"),
             "metrics": {"validation": m["validation"][str(s)], "test_cuts": m["test_cuts"]["per_seed"][str(s)]}} for s in c2.SEEDS]
    rows = []
    for seed in c2.SEEDS:
        for split in ("validation", "test"):
            path = D1_DIR / "predictions" / f"bert_{seed}_{split}.csv"
            by_id = {r["window_id"]: r for r in data_v2.read_csv(path)}
            for name in (["validation"] if split == "validation" else ["test_principal", "test_secundario"]):
                s = sets_eval[name]
                if len(by_id) < len(s["windows"]):
                    raise data.ModelError(f"{path}: fewer windows than the {name} set")
                for i, w in enumerate(s["windows"]):
                    r = by_id[w["window_id"]]
                    score = float(r["score"])
                    rows.append({"tarefa": DETECTION, "modelo": model_name, "regime": FULL, "sorteio": "", "semente": seed, "conjunto": name,
                                 "window_id": w["window_id"], "y_true": s["y"][i], "y_pred": int(r["prediction"]),
                                 "p_normal": repr(1 - score), "p_anomala": r["score"], "score": r["score"], "origem": str(path)})
    return copied, runs, rows


def check_d1_rows(rows, sets_eval):
    """The copied predictions must give the copied metrics."""
    problems = []
    m = json.loads((D1_DIR / "metrics.json").read_text(encoding="utf-8"))
    for seed in c2.SEEDS:
        for name in ("validation", "test_secundario"):
            mine = [r for r in rows if r["semente"] == seed and r["conjunto"] == name]
            y, scores, preds = [r["y_true"] for r in mine], [float(r["score"]) for r in mine], [r["y_pred"] for r in mine]
            if name == "validation":
                ok = metrics.binary_metrics(y, scores, preds) == m["validation"][str(seed)]
            else:
                ok = c1v2.cut_metrics(y, scores, preds, sets_eval[name]["rows"]) == m["test_cuts"]["per_seed"][str(seed)]
            if not ok:
                problems.append(f"{seed}/{name}")
    if problems:
        raise data.ModelError(f"D1 predictions do not reproduce the D1 metrics: {problems}")
    return "the converted D1 predictions reproduce the D1 validation metrics and test cuts of every seed"


# ---------------------------------------------------------------- evaluate
def stage_evaluate(args):
    started = time.perf_counter()
    base = args.output
    selection_path = base / SELECTION_FILE
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    gpu = check_gpu_free()
    pre, verification = verify(args.model, args.encoder)
    if selection["model"] != args.model or selection["encoder"] != str(args.encoder):
        raise data.ModelError("selection.json is of another encoder")
    tokenizer = c2.load_tokenizer()
    causes, templates, sets = load_train_validation(tokenizer)
    build = encoder_loader(args.encoder)
    pad = tokenizer.pad_token_id
    runs = selection["full_data_runs"]

    # 1. the saved checkpoints must give the validation results of select, before the test is opened
    saved = {(p["tarefa"], p["semente"], p["window_id"]): p for p in selection["validation_predictions"]}
    reload_check = {}
    model = None
    for record in runs:
        task, seed = record["task"], record["seed"]
        if model is None or model.classifier.out_features != N_LABELS[task]:
            model = Classifier(build(), N_LABELS[task]).to(args.device)
        load_checkpoint(model, run_folder(args.checkpoints, task, seed), record["checkpoint_sha256"])
        p, _ = predict(model, sets["validation"]["input_ids"], pad, args.device)
        columns = [f"p_{c}" for c in CLASSES] if task == DIAGNOSIS else ["p_normal", "p_anomala"]
        before = [[float(saved[(task, seed, w["window_id"])][c]) for c in columns] for w in sets["validation"]["windows"]]
        reload_check[f"{task}/{seed}"] = compare_probabilities(p, before)
        if reload_check[f"{task}/{seed}"]["max_abs_difference"] > PROBABILITY_TOLERANCE or not reload_check[f"{task}/{seed}"]["same_argmax"]:
            raise data.ModelError(f"{task}/{seed}: validation differs from select ({reload_check[f'{task}/{seed}']}); the test was not opened")

    # 2. marker, then the test, once
    marker = {"created_local": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "model": args.model, "encoder": str(args.encoder),
              "selection_sha256": classicos.sha(selection_path), "reads": 1,
              "note": "written before the test set is opened; a second evaluation is refused"}
    data.write_new(base / TEST_MARKER, data.json_bytes(marker))
    gate = c1v2.TestGate()
    sets["test"] = tokenize(classicos.attach(gate.read(data_v2.DERIVED_DIR, VIEW, templates, marker["selection_sha256"]), causes, "test"),
                            templates, tokenizer)
    runs_total = data_v2.check_disjoint({s: sets[s]["rows"] for s in sets})
    sets_eval = evaluated_sets(sets, include_test=True)

    # 3. full-data runs on validation and the two tests
    results, predictions, latency = [], [], None
    for record in runs:
        task, seed = record["task"], record["seed"]
        if model is None or model.classifier.out_features != N_LABELS[task]:
            model = Classifier(build(), N_LABELS[task]).to(args.device)
        load_checkpoint(model, run_folder(args.checkpoints, task, seed), record["checkpoint_sha256"])
        result, probabilities, preds = evaluate_model(model, task, sets_eval, pad, args.device)
        key = {"tarefa": task, "modelo": args.model, "regime": FULL, "sorteio": "", "semente": seed}
        results.append({**key, "params": None, "best": record["best"], "metrics": result})
        predictions += classicos.prediction_rows(task, key, sets_eval, probabilities, preds, origin="e2")
        if task == DIAGNOSIS and seed == c2.SEEDS[0]:
            latency = c2.measure_inference(model, tokenizer, sets["test"]["text"], args.device)
            latency["model"] = f"{task}, seed {seed}, best epoch {record['best']['point']}; test texts, as the D1"
        log(f"evaluated {task}/{seed}")
    del model

    # 4. detection with full data: copied from the D1 (BERT-base)
    copied, d1_note = [], None
    if args.detection_full == "copy":
        aggregated_d1, d1_runs, d1_rows = d1_detection(sets_eval, args.model)
        d1_note = check_d1_rows(d1_rows, sets_eval)
        copied = [aggregated_d1]
        predictions += d1_rows

    # 5. low-resource runs: train, best point on validation, then validation and tests once; weights are not written
    low_dir = base / LOW_DIR
    low_results, low_records = [], []
    for draw in classicos.read_draws():
        indexes = classicos.draw_indexes(sets["train"], draw)
        train = subset(sets["train"], indexes)
        for task in TASKS:
            name = f"{task}_{draw['fraction']}_{draw['index']}"
            record, state, best_probabilities, model = train_run(task, train, sets["validation"], draw["seed"], build, pad, args.device)
            model.load_state_dict(state)
            result, probabilities, preds = evaluate_model(model, task, sets_eval, pad, args.device)
            del model, state
            record["reload_vs_training"] = compare_probabilities(probabilities["validation"], best_probabilities)
            if main_metric(task, labels_of(sets["validation"], task), preds["validation"]) != record["best"]["validation_metric"]:
                raise data.ModelError(f"{name}: best state does not reproduce its validation point (test already evaluated): stop")
            key = {"tarefa": task, "modelo": args.model, "regime": str(draw["fraction"]), "sorteio": draw["index"], "semente": draw["seed"]}
            rows = classicos.prediction_rows(task, key, sets_eval, probabilities, preds, origin="e2")
            run = {**key, "params": None, "best": record["best"], "metrics": result}
            record.update({"fraction": draw["fraction"], "draw": draw["index"], "by_cause": draw["by_cause"]})
            data.write_new(low_dir / f"{name}.json", data.json_bytes({"run": run, "training": record, "predictions": rows}))
            low_results.append(run)
            low_records.append(record)
            predictions += rows
            log(f"{name}: point {record['best']['point']} metric {record['best']['validation_metric']:.4f} train {record['seconds']['train']:.1f}s")

    # 6. outputs
    results += low_results
    aggregated = classicos.aggregate(results)
    for a in aggregated:
        a["origem"] = "e2"
    aggregated += copied
    seconds = time.perf_counter() - started
    training = {
        "full_data": [{k: v for k, v in r.items()} for r in runs],
        "shuffled_labels_control": selection["shuffled_labels_control"],
        "low_resources": low_records,
        "rule": "best point: highest validation main metric (F1 of the anomalous class / Macro-F1), then lowest validation loss, then the earliest",
    }
    cost = {
        "train_seconds": {"full_data": {f"{r['task']}/{r['seed']}": r["seconds"]["train"] for r in runs},
                          "shuffled_control": selection["shuffled_labels_control"]["seconds"]["train"],
                          "low_resources": {f"{r['task']}/{r['fraction']}/{r['draw']}": r["seconds"]["train"] for r in low_records}},
        "run_seconds_with_validation": {"full_data": {f"{r['task']}/{r['seed']}": r["seconds"]["run_total"] for r in runs},
                                        "low_resources": {f"{r['task']}/{r['fraction']}/{r['draw']}": r["seconds"]["run_total"] for r in low_records}},
        "peak_vram_mb": {"full_data": {f"{r['task']}/{r['seed']}": r["peak_vram_mb"] for r in runs},
                         "low_resources": {f"{r['task']}/{r['fraction']}/{r['draw']}": r["peak_vram_mb"] for r in low_records}},
        "inference": latency, "seconds_select": selection["seconds"], "seconds_evaluate": seconds,
    }
    files = {
        "config.json": data.json_bytes({
            "model": args.model, "encoder": str(args.encoder), "checkpoints": str(args.checkpoints), "environment": c2.environment(),
            "gpu_check": {"select": selection["gpu_check"], "evaluate": gpu}, "verification": verification, "preregistro_sha256": classicos.PREREGISTRO_SHA256,
            "inputs": data_v2.verify_inputs(data_v2.DERIVED_DIR), "view": VIEW, "classes": list(CLASSES),
            "fine_tuning": pre["hiperparametros"]["fine_tuning_bert_2b_luk"], "low_resource_budget": pre["orcamento_baixos_recursos"]["bert_2b_luk"],
            "effective": {"learning_rate": LEARNING_RATE, "steps": STEPS, "evaluation_every": EVAL_EVERY, "warmup_steps": WARMUP_STEPS,
                          "batch_size": c2.BATCH_SIZE, "weight_decay": c2.WEIGHT_DECAY, "no_decay": list(c2.NO_DECAY), "max_grad_norm": c2.MAX_GRAD_NORM,
                          "max_tokens": wt.MAX_TOKENS, "truncation": False, "precision": "fp32", "seeds_full_data": list(c2.SEEDS),
                          "seeds_draws": "seed of the draw", "tokenizer": f"{c2.MODEL} {c2.REVISION}",
                          "head_outputs": {DETECTION: 2, DIAGNOSIS: 4}},
            "detection_full_data": "copied from the D1 (results/cenario2), not trained again" if args.detection_full == "copy" else "trained here",
            "d1_check": d1_note, "checkpoint_reload_check": reload_check, "test_reads": dict(gate.reads),
            "execucoes_sem_sobreposicao": runs_total,
            "sets": {"validation": "normal + visible (102)", "test_principal": "normal + visible (102)", "test_secundario": "every test window (106)",
                     "deteccao_test_cuts": "todas = test_secundario, so_visiveis = test_principal, so_lentidao = normal + slow-only"},
            "selection_json_sha256": marker["selection_sha256"]}),
        "training.json": data.json_bytes(training),
        "metrics.json": data.json_bytes({"runs": results + (d1_runs if args.detection_full == "copy" else []), "aggregated": aggregated,
                                         "cost": cost}),
        "metrics.csv": data.csv_bytes(["tarefa", "modelo", "regime", "runs", "origem", "conjunto", "metrica", "media", "desvio"],
                                      classicos.metric_rows(aggregated)),
        "predictions_diagnostico.csv": data.csv_bytes(classicos.PRED_DIAG, [p for p in predictions if p["tarefa"] == DIAGNOSIS]),
        "predictions_deteccao.csv": data.csv_bytes(classicos.PRED_DET, [p for p in predictions if p["tarefa"] == DETECTION]),
    }
    for name, content in files.items():
        data.write_new(base / name, content)
    every = {name: (base / name).read_bytes() for name in (*files, SELECTION_FILE, TEST_MARKER)}
    every.update({f"{LOW_DIR}/{p.name}": p.read_bytes() for p in sorted(low_dir.iterdir())})
    data.write_new(base / "manifest.sha256", data.manifest_bytes(every))
    log(json.dumps({"runs": len(results), "prediction_rows": len(predictions), "seconds": seconds, "test_reads": dict(gate.reads)}))


def log(message):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("select", "evaluate"))
    parser.add_argument("--model", choices=MODELS, default=BERT, help="name of the model in the pre-registration and in the outputs")
    parser.add_argument("--encoder", default=c2.MODEL, help="bert-base-uncased, or the folder of the 2b / LUK checkpoint")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--detection-full", choices=("copy", "train"), dest="detection_full",
                        help="copy: from the D1 (BERT-base only); train: 5 seeds trained here (2b, LUK)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output is None or args.checkpoints is None:
        if args.model not in DEFAULT_PATHS:
            parser.error(f"--output and --checkpoints are required for {args.model}")
        args.output = args.output or DEFAULT_PATHS[args.model][0]
        args.checkpoints = args.checkpoints or DEFAULT_PATHS[args.model][1]
    if args.detection_full is None:
        args.detection_full = "copy" if args.model == BERT else "train"
    if args.detection_full == "copy" and args.model != BERT:
        parser.error("only BERT-base has full-data detection results to copy (D1)")
    if args.model != BERT:
        args.encoder = str(Path(args.encoder))
    return args


def main(argv=None):
    args = parse(argv)
    {"select": stage_select, "evaluate": stage_evaluate}[args.stage](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

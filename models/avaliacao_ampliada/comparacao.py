"""D6: comparação final da avaliação ampliada (testes estatísticos, intervalos, tabelas e figuras), conforme o pré-registro E0.

Só lê previsões e métricas salvas (results/e1, e2, bert-mlm-2b e bert-luk; results/cenario2 para a perda de validação da detecção do
BERT-base, que veio da D1). Não treina, não avalia modelos e não usa GPU. Os modelos são identificados pela coluna modelo; o campo
origem é ignorado. Executar da raiz do repositório com C:\\tcc-venv\\comparacao (numpy 2.1.3, scipy 1.14.1, scikit-learn 1.5.2, como
nas etapas anteriores, e matplotlib 3.9.2 para as figuras):
  python -m models.avaliacao_ampliada.comparacao
Saídas em results/comparacao-final, pasta criada nesta execução (recusa se já existir); nenhum arquivo é sobrescrito.

Testes (pré-registro, testes_estatisticos):
  estatística   diferença da média, nas 5 unidades, da métrica principal (A - B): Macro-F1 no diagnóstico (teste principal, 102
                janelas), F1 da classe anômala na detecção (todas as janelas do teste, 106).
  unidade       semente (dados completos) ou sorteio (baixos recursos; a semente de treino é a do sorteio, igual nos dois modelos).
  permutação    aleatorização aproximada pareada por janela: em cada permutação, uma moeda por janela decide se as previsões dessa
                janela trocam de modelo; a troca vale para todas as unidades ao mesmo tempo e é sempre entre as rodadas da mesma unidade
                (esclarecimento do orientador de 09/11/2026). 10000 permutações, bilateral sobre a diferença absoluta,
                p = (b + 1) / (N + 1), semente 20261109.
  McNemar       exato (binomial bilateral nas janelas discordantes) sobre o voto majoritário das 5 unidades, certo/errado por janela;
                empate no voto -> maior probabilidade média, depois a ordem das classes.
  IC 95%        bootstrap percentil por execução: as 10 execuções do teste reamostradas com reposição, janelas de cada execução juntas;
                10000 reamostras, semente 20261110.
  Holm          dentro de cada família registrada; a comparação principal (LUK vs 2b, diagnóstico, dados completos) não tem família."""
import csv
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from models.common import data, data_v2, metrics
from models.avaliacao_ampliada import causa, classicos

RESULTS = data_v2.RESULTS_DIR
SOURCES = ("e1", "e2", "bert-mlm-2b", "bert-luk")
C2_DIR = RESULTS / "cenario2"
OUTPUT = RESULTS / "comparacao-final"
CLASSES = tuple(causa.CLASSES)
DET, DIAG, FULL = classicos.DETECTION, classicos.DIAGNOSIS, classicos.FULL
TASKS = (DET, DIAG)
SETS = ("validation", "test_principal", "test_secundario")
FRACTIONS = ("0.01", "0.05", "0.1")
REGIMES = FRACTIONS + (FULL,)
LMS = ("bert_base", "bert_mlm_2b", "bert_luk")
TFIDF = ("tfidf_rf_contagem", "tfidf_rf_presenca")
MAJORITY, RULE = classicos.MAJORITY, classicos.RULE
MODEL_ORDER = (MAJORITY, RULE) + TFIDF + LMS
PERM_SEED, BOOT_SEED = 20261109, 20261110
N_PERM = N_BOOT = 10000
ALPHA = 0.05
TOL = 1e-12
TEST_SET = {DIAG: "test_principal", DET: "test_secundario"}
KIND = {DIAG: "macro", DET: "positiva"}
N_CLASSES = {DIAG: len(CLASSES), DET: 2}
MAIN_METRIC = {DIAG: "macro_f1", DET: "f1"}
NAMES = {MAJORITY: "Classe mais comum", RULE: "Regra simples", "tfidf_rf_contagem": "TF-IDF + RF (contagem)",
         "tfidf_rf_presenca": "TF-IDF + RF (presença)", "bert_base": "BERT-base", "bert_mlm_2b": "BERT + MLM (2b)", "bert_luk": "BERT + LUK"}
SHORT = {"bert_base": "BERT-base", "bert_mlm_2b": "2b", "bert_luk": "LUK", "tfidf_rf_contagem": "TF-IDF contagem",
         "tfidf_rf_presenca": "TF-IDF presença"}
REGIME_NAMES = {"0.01": "1%", "0.05": "5%", "0.1": "10%", FULL: "100%"}
FAMILIES = {"diagnostico_dados_completos": (DIAG, (FULL,)), "diagnostico_baixos_recursos": (DIAG, FRACTIONS),
            "deteccao_dados_completos": (DET, (FULL,)), "deteccao_baixos_recursos": (DET, FRACTIONS)}
FAMILY_NAMES = {None: "Comparação principal (sem família)", "diagnostico_dados_completos": "Diagnóstico, dados completos (Holm, 3)",
                "diagnostico_baixos_recursos": "Diagnóstico, baixos recursos (Holm, 12)",
                "deteccao_dados_completos": "Detecção, dados completos (Holm, 4)", "deteccao_baixos_recursos": "Detecção, baixos recursos (Holm, 12)"}
ADVISOR_NOTE = {
    "data": "2026-11-09",
    "autor": "orientador",
    "texto": ("O teste de permutação pareado é o teste principal, e o emparelhamento deve ser por semente, ou seja, cada rodada do LUK é "
              "comparada com a rodada do 2b (ou do BERT-base) que usou exatamente a mesma semente de treino. Isso é compatível com o "
              "pré-registro: em cada permutação, a troca de previsões de uma janela é feita entre as rodadas de mesma semente dos dois "
              "modelos."),
    "preregistro_alterado": False,
}
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
SERIES_MARKERS = ("o", "s", "^", "D", "v", "P")
SERIES_LINES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1)), (0, (1, 1)))


# ---------------------------------------------------------------- statistics
def confusion(y, preds, k):
    """Confusion counts (U, k, k) of each row of preds (U, W); rows = true class, columns = predicted."""
    preds = np.atleast_2d(np.asarray(preds, dtype=np.int64))
    cells = np.asarray(y, dtype=np.int64)[None, :] * k + preds
    return np.stack([np.bincount(c, minlength=k * k) for c in cells]).reshape(-1, k, k)


def f1_by_class(conf):
    """F1 of each class = 2 TP / (2 TP + FP + FN); 0 when the class has no true and no predicted window (zero_division=0 of sklearn)."""
    conf = np.asarray(conf, dtype=float)
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    den = conf.sum(-1) + conf.sum(-2)
    return np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)


def score(conf, kind):
    """macro: mean F1 of the classes (Macro-F1); positiva: F1 of class 1 (anomalous)."""
    f1 = f1_by_class(conf)
    return f1.mean(-1) if kind == "macro" else f1[..., 1]


def statistic(y, pa, pb, k, kind):
    """Mean over the units of the metric of A minus the same mean of B."""
    return float(score(confusion(y, pa, k), kind).mean() - score(confusion(y, pb, k), kind).mean())


def swap(pa, pb, coins):
    """Predictions after one permutation. Where the coin of a window is 1, the predictions of that window swap model in every unit at
    once, always between the runs of the same unit (row u of A with row u of B); other windows are kept."""
    mask = np.asarray(coins, dtype=bool)[None, :]
    return np.where(mask, pb, pa), np.where(mask, pa, pb)


def permutation_coins(n, windows, seed):
    return np.random.default_rng(seed).integers(0, 2, size=(n, windows), dtype=np.int64)


def permutation_statistics(y, pa, pb, k, kind, coins):
    """Statistic of every permutation, equal to statistic(y, *swap(pa, pb, coins[i]), k, kind), from confusion counts: swapping a
    window moves, in each unit, its cell of B into A and its cell of A into B."""
    pa, pb, y = np.asarray(pa, dtype=np.int64), np.asarray(pb, dtype=np.int64), np.asarray(y, dtype=np.int64)
    units, windows = pa.shape
    eye = np.eye(k * k, dtype=np.int64)
    delta = (eye[y[None, :] * k + pb] - eye[y[None, :] * k + pa]).transpose(1, 0, 2).reshape(windows, units * k * k)
    shift = np.asarray(coins, dtype=np.int64) @ delta
    conf_a = (confusion(y, pa, k).reshape(-1) + shift).reshape(-1, units, k, k)
    conf_b = (confusion(y, pb, k).reshape(-1) - shift).reshape(-1, units, k, k)
    return score(conf_a, kind).mean(1) - score(conf_b, kind).mean(1)


def permutation_test(y, pa, pb, k, kind, n=N_PERM, seed=PERM_SEED):
    observed = statistic(y, pa, pb, k, kind)
    stats = permutation_statistics(y, pa, pb, k, kind, permutation_coins(n, np.shape(pa)[1], seed))
    count = int(np.sum(np.abs(stats) >= abs(observed) - TOL))
    return {"observed": observed, "count_at_least_observed": count, "permutations": n, "seed": seed, "p": (count + 1) / (n + 1)}


def run_indexes(window_ids):
    """Run (experiment_run_id, the part of the window id before '#') of each window, as indexes 0..R-1 in sorted run order."""
    runs = [w.split("#")[0] for w in window_ids]
    names = sorted(set(runs))
    return np.array([names.index(r) for r in runs]), names


def bootstrap_draws(n_runs, n, seed):
    """Each row: n_runs runs drawn with replacement (indexes of whole runs, never of windows)."""
    return np.random.default_rng(seed).integers(0, n_runs, size=(n, n_runs))


def bootstrap_statistics(y, pa, pb, run_index, n_runs, k, kind, draws):
    """Statistic on the windows of the drawn runs, each run with all its windows, as many times as it was drawn."""
    y, pa, pb, run_index = (np.asarray(v) for v in (y, pa, pb, run_index))
    per_run_a = np.stack([confusion(y[run_index == r], pa[:, run_index == r], k) for r in range(n_runs)])
    per_run_b = np.stack([confusion(y[run_index == r], pb[:, run_index == r], k) for r in range(n_runs)])
    weights = np.stack([np.bincount(d, minlength=n_runs) for d in draws])
    conf_a = np.einsum("br,ruij->buij", weights, per_run_a)
    conf_b = np.einsum("br,ruij->buij", weights, per_run_b)
    return score(conf_a, kind).mean(1) - score(conf_b, kind).mean(1)


def bootstrap_ci(y, pa, pb, window_ids, k, kind, n=N_BOOT, seed=BOOT_SEED):
    run_index, names = run_indexes(window_ids)
    draws = bootstrap_draws(len(names), n, seed)
    stats = bootstrap_statistics(y, pa, pb, run_index, len(names), k, kind, draws)
    low, high = np.percentile(stats, [2.5, 97.5])
    return {"lower": float(low), "upper": float(high), "resamples": n, "seed": seed, "runs": len(names), "method": "percentile 2.5 / 97.5 (numpy linear)",
            "bootstrap_mean": float(stats.mean()), "bootstrap_std": float(stats.std(ddof=1))}


def majority_vote(preds, probabilities, k):
    """Most voted class of each window over the units; tie -> highest mean probability among the tied classes, then class order."""
    preds, probabilities = np.asarray(preds), np.asarray(probabilities, dtype=float)
    counts = (preds[..., None] == np.arange(k)).sum(0)
    tied = counts == counts.max(1, keepdims=True)
    return np.argmax(np.where(tied, probabilities.mean(0), -np.inf), axis=1)


def mcnemar(y, vote_a, vote_b):
    ok_a, ok_b = np.asarray(vote_a) == np.asarray(y), np.asarray(vote_b) == np.asarray(y)
    only_a, only_b = int(np.sum(ok_a & ~ok_b)), int(np.sum(~ok_a & ok_b))
    p = 1.0 if only_a + only_b == 0 else float(binomtest(only_a, only_a + only_b, 0.5).pvalue)
    return {"a_certo_b_errado": only_a, "a_errado_b_certo": only_b, "discordantes": only_a + only_b, "p": p,
            "a_acertos": int(ok_a.sum()), "b_acertos": int(ok_b.sum())}


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: (pvalues[i], i))
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


# ---------------------------------------------------------------- saved predictions and metrics
def unit_order(unit):
    draw, seed = unit
    return (int(draw) if draw != "" else -1, int(seed) if seed != "" else -1)


def load_predictions():
    """(tarefa, modelo, regime, sorteio, semente, conjunto) -> run arrays; each key must come from a single source folder."""
    groups = {}
    for task in TASKS:
        for source in SOURCES:
            for row in data_v2.read_csv(RESULTS / source / f"predictions_{task}.csv"):
                if row["tarefa"] != task:
                    raise data.ModelError(f"{source}/predictions_{task}.csv: row of task {row['tarefa']}")
                key = (task, row["modelo"], row["regime"], row["sorteio"], row["semente"], row["conjunto"])
                entry = groups.setdefault(key, {"source": source, "rows": []})
                if entry["source"] != source:
                    raise data.ModelError(f"{key} in {entry['source']} and in {source}")
                entry["rows"].append(row)
    runs = {}
    for key, entry in groups.items():
        rows, task = entry["rows"], key[0]
        ids = [r["window_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise data.ModelError(f"{key}: repeated window")
        if task == DIAG:
            y = [CLASSES.index(r["y_true"]) for r in rows]
            pred = [CLASSES.index(r["y_pred"]) for r in rows]
            prob = [[float(r[f"p_{c}"]) for c in CLASSES] for r in rows]
            scores = None
        else:
            y, pred = [int(r["y_true"]) for r in rows], [int(r["y_pred"]) for r in rows]
            # the simple rule has no probabilities (empty cells), only its score; it is in no test
            prob = [[float(r[c]) if r[c] != "" else np.nan for c in ("p_normal", "p_anomala")] for r in rows]
            scores = np.array([float(r["score"]) for r in rows])
        runs[key] = {"source": entry["source"], "window_ids": ids, "y": np.array(y), "pred": np.array(pred), "prob": np.array(prob),
                     "score": scores}
    return runs


def units_of(runs, task, model, regime):
    return sorted({(k[3], k[4]) for k in runs if k[:3] == (task, model, regime)}, key=unit_order)


def groups_of(runs):
    return sorted({k[:3] for k in runs}, key=lambda g: (g[0], MODEL_ORDER.index(g[1]), REGIMES.index(g[2])))


def detection_cuts(secondary, principal):
    """Indexes of the cuts in the full test (test_secundario): todas; so_visiveis = the windows of test_principal (normal + visible);
    so_lentidao = normal + the windows outside test_principal (slow-only), as cenario1_v2.CUTS."""
    position = {w: i for i, w in enumerate(secondary["window_ids"])}
    inside = set(principal["window_ids"])
    if not inside <= set(position):
        raise data.ModelError("detection: a test_principal window is not in test_secundario")
    for i, w in enumerate(principal["window_ids"]):
        j = position[w]
        if principal["y"][i] != secondary["y"][j] or principal["pred"][i] != secondary["pred"][j]:
            raise data.ModelError(f"detection: window {w} differs between test_principal and test_secundario")
    slow = [i for i, w in enumerate(secondary["window_ids"]) if w not in inside]
    if any(secondary["y"][i] != 1 for i in slow):
        raise data.ModelError("detection: a window outside test_principal is not anomalous")
    every = list(range(len(secondary["window_ids"])))
    return {"todas": every, "so_visiveis": [i for i in every if secondary["window_ids"][i] in inside],
            "so_lentidao": [i for i in every if secondary["window_ids"][i] not in inside or secondary["y"][i] == 0]}


def recompute(runs, task, model, regime, unit):
    """Metrics of one run from its saved predictions, with the functions of the earlier stages (sklearn)."""
    get = {c: runs[(task, model, regime, *unit, c)] for c in SETS}
    if task == DIAG:
        return {c: classicos.multiclass_metrics(get[c]["y"].tolist(), get[c]["prob"], get[c]["pred"].tolist()) for c in SETS}
    val, sec = get["validation"], get["test_secundario"]
    out = {"validation": metrics.binary_metrics(val["y"].tolist(), val["score"].tolist(), val["pred"].tolist()), "test_cuts": {}}
    for cut, idx in detection_cuts(sec, get["test_principal"]).items():
        out["test_cuts"][cut] = metrics.binary_metrics(sec["y"][idx].tolist(), sec["score"][idx].tolist(), sec["pred"][idx].tolist())
    return out


def compare_values(expected, got, path, problems, diffs, subset=False, extra=None):
    """Recursive comparison; with subset, got may hold keys that expected lacks (listed in extra), never the opposite."""
    if isinstance(expected, dict):
        if not isinstance(got, dict) or (not set(expected) <= set(got) if subset else set(expected) != set(got)):
            problems.append(f"{path}: keys differ")
            return
        if extra is not None:
            extra += [f"{path}.{k}" for k in got if k not in expected]
        for key in expected:
            compare_values(expected[key], got[key], f"{path}.{key}", problems, diffs, subset, extra)
    elif isinstance(expected, list):
        if not isinstance(got, list) or len(expected) != len(got):
            problems.append(f"{path}: lengths differ")
            return
        for i, (e, g) in enumerate(zip(expected, got)):
            compare_values(e, g, f"{path}[{i}]", problems, diffs, subset, extra)
    elif isinstance(expected, float) or isinstance(got, float):
        diff = abs(float(expected) - float(got))
        diffs.append(diff)
        if diff > TOL:
            problems.append(f"{path}: {expected} (metrics.json) vs {got} (recomputed)")
    elif expected != got:
        problems.append(f"{path}: {expected!r} (metrics.json) vs {got!r} (recomputed)")


def run_key(r):
    return (r["tarefa"], r["modelo"], r["regime"], str(r["sorteio"]), str(r["semente"]))


def summarize(task, per_run):
    if task == DIAG:
        return {c: classicos.summarize_multiclass([p[c] for p in per_run]) for c in SETS}
    return classicos.summarize_detection(per_run)


def check_metrics(runs):
    """Part 1.2: every run of the predictions recomputed and compared with the metrics.json of its stage (each run and each mean)."""
    recomputed = {g: {u: recompute(runs, *g, u) for u in units_of(runs, *g)} for g in groups_of(runs)}
    report = {"runs_compared": 0, "aggregates_compared": 0, "problems": [], "max_abs_difference": 0.0, "groups": [],
              "aggregate_keys_not_in_metrics_json": []}
    seen_runs, seen_aggregates, diffs = set(), set(), []
    for source in SOURCES:
        saved = json.loads((RESULTS / source / "metrics.json").read_text(encoding="utf-8"))
        for r in saved["runs"]:
            key = run_key(r)
            group, unit = key[:3], key[3:]
            if group not in recomputed or unit not in recomputed[group]:
                report["problems"].append(f"{source}: run {key} of metrics.json has no predictions")
                continue
            compare_values(r["metrics"], recomputed[group][unit], f"{source}:{'/'.join(key)}", report["problems"], diffs)
            seen_runs.add(key)
            report["runs_compared"] += 1
        for a in saved["aggregated"]:
            group = (a["tarefa"], a["modelo"], a["regime"])
            if group not in recomputed:
                report["problems"].append(f"{source}: aggregate {group} has no predictions")
                continue
            per_run = [recomputed[group][u] for u in units_of(runs, *group)]
            if a["runs"] != len(per_run):
                report["problems"].append(f"{source}: aggregate {group}: {a['runs']} runs in metrics.json, {len(per_run)} in the predictions")
            mine = summarize(group[0], per_run)
            compare_values(a["summary"], mine, f"{source}:aggregate:{'/'.join(group)}", report["problems"], diffs, subset=True,
                           extra=report["aggregate_keys_not_in_metrics_json"])
            seen_aggregates.add(group)
            report["aggregates_compared"] += 1
    for group in recomputed:
        without_run = [u for u in recomputed[group] if group + u not in seen_runs]
        entry = {"group": "/".join(group), "units": len(recomputed[group]), "runs_in_metrics_json": len(recomputed[group]) - len(without_run),
                 "aggregate_in_metrics_json": group in seen_aggregates}
        if without_run:
            entry["note"] = "no per-run entry in metrics.json (copied from the C4); checked through the aggregate"
        if group not in seen_aggregates:
            report["problems"].append(f"{group}: no aggregate in any metrics.json")
        report["groups"].append(entry)
    report["max_abs_difference"] = max(diffs) if diffs else 0.0
    report["values_compared"] = len(diffs)
    report["tolerance"] = TOL
    return recomputed, report


def check_pairing(runs, task, regime, a, b):
    """Part 1.3: the same units (seeds, or draws and seeds) and, in each unit and set, the same windows with the same labels."""
    ua, ub = units_of(runs, task, a, regime), units_of(runs, task, b, regime)
    problems = []
    if ua != ub:
        problems.append(f"units: only {a}: {sorted(set(ua) - set(ub))}; only {b}: {sorted(set(ub) - set(ua))}")
    for unit in sorted(set(ua) & set(ub), key=unit_order):
        for c in SETS:
            ra, rb = runs.get((task, a, regime, *unit, c)), runs.get((task, b, regime, *unit, c))
            if ra is None or rb is None:
                problems.append(f"{unit}/{c}: missing for {a if ra is None else b}")
                continue
            if set(ra["window_ids"]) != set(rb["window_ids"]):
                problems.append(f"{unit}/{c}: windows only in {a}: {sorted(set(ra['window_ids']) - set(rb['window_ids']))}; "
                                f"only in {b}: {sorted(set(rb['window_ids']) - set(ra['window_ids']))}")
                continue
            position = {w: i for i, w in enumerate(rb["window_ids"])}
            if not np.array_equal(ra["y"], rb["y"][[position[w] for w in ra["window_ids"]]]):
                problems.append(f"{unit}/{c}: true labels differ")
    return {"units": [list(u) for u in ua], "problems": problems}


# ---------------------------------------------------------------- choice of the best models (validation only)
def validation_losses_d1():
    """Validation loss at the best epoch of the BERT-base detection runs (D1, results/cenario2), at the rate chosen there."""
    selection = json.loads((C2_DIR / "selection.json").read_text(encoding="utf-8"))
    rate = selection["chosen"]["learning_rate"]
    return {str(r["seed"]): r["best"]["validation_loss"] for r in selection["runs"] if r["learning_rate"] == rate}, rate


def validation_loss(saved_runs, d1_losses, task, model, regime, units):
    values = []
    for unit in units:
        r = saved_runs.get((task, model, regime, *unit))
        if r is not None and r.get("best"):
            values.append(r["best"]["validation_loss"])
        elif task == DET and model == "bert_base" and regime == FULL:
            values.append(d1_losses[unit[1]])
        else:
            raise data.ModelError(f"no validation loss for {task}/{model}/{regime}/{unit}")
    return statistics.fmean(values)


def choose_best(recomputed, saved_runs, d1_losses, task, regime):
    """Best language model and best TF-IDF on the full validation, by the mean validation main metric over the units; ties -> lowest
    mean validation loss, then order BERT, 2b, LUK (language models) and contagem before presenca (TF-IDF). Never the test."""
    metric = MAIN_METRIC[task]

    def mean_validation(model):
        return statistics.fmean(m["validation"][metric] for m in recomputed[(task, model, regime)].values())

    lm = {m: mean_validation(m) for m in LMS}
    top = max(lm.values())
    tied = [m for m in LMS if abs(lm[m] - top) <= TOL]
    losses = {}
    if len(tied) > 1:
        units = {m: list(recomputed[(task, m, regime)]) for m in tied}
        losses = {m: validation_loss(saved_runs, d1_losses, task, m, regime, units[m]) for m in tied}
        lowest = min(losses.values())
        best_lm = next(m for m in tied if abs(losses[m] - lowest) <= TOL)
    else:
        best_lm = tied[0]
    tf = {m: mean_validation(m) for m in TFIDF}
    top_tf = max(tf.values())
    tied_tf = [m for m in TFIDF if abs(tf[m] - top_tf) <= TOL]
    return {"tarefa": task, "regime": regime, "metrica_validacao": metric, "lm_validation_mean": lm, "lm_tied": tied,
            "lm_validation_loss_mean": losses, "melhor_lm": best_lm, "tfidf_validation_mean": tf, "tfidf_tied": tied_tf,
            "melhor_tfidf": tied_tf[0]}


# ---------------------------------------------------------------- comparisons
def plan(best):
    comparisons = [{"familia": None, "tarefa": DIAG, "regime": FULL, "a": "bert_luk", "b": "bert_mlm_2b", "principal": True}]
    for family, (task, regimes) in FAMILIES.items():
        for regime in regimes:
            pairs = [("bert_luk", "bert_base"), ("bert_mlm_2b", "bert_base")]
            if family != "diagnostico_dados_completos":
                pairs.insert(0, ("bert_luk", "bert_mlm_2b"))
            choice = best[f"{task}/{regime}"]
            pairs.append((choice["melhor_lm"], choice["melhor_tfidf"]))
            for i, (a, b) in enumerate(pairs):
                comparisons.append({"familia": family, "tarefa": task, "regime": regime, "a": a, "b": b, "principal": False,
                                    "melhor_lm_vs_melhor_tfidf": i == len(pairs) - 1})
    for c in comparisons:
        c["id"] = f"{c['tarefa']}/{c['regime']}/{c['a']}-{c['b']}"
        label = f"{SHORT[c['a']]} vs {SHORT[c['b']]}"
        c["rotulo"] = f"melhor LM ({SHORT[c['a']]}) vs melhor TF-IDF ({SHORT[c['b']]})" if c.get("melhor_lm_vs_melhor_tfidf") else label
    return comparisons


def comparison_arrays(runs, task, regime, a, b):
    units = units_of(runs, task, a, regime)
    conjunto = TEST_SET[task]
    reference = runs[(task, a, regime, *units[0], conjunto)]
    ids, y = reference["window_ids"], reference["y"]

    def stack(model):
        preds, probs = [], []
        for unit in units:
            run = runs[(task, model, regime, *unit, conjunto)]
            position = {w: i for i, w in enumerate(run["window_ids"])}
            index = [position[w] for w in ids]
            if not np.array_equal(run["y"][index], y):
                raise data.ModelError(f"{model}/{regime}/{unit}: true labels differ")
            preds.append(run["pred"][index])
            probs.append(run["prob"][index])
        return np.stack(preds), np.stack(probs)

    pa, proba = stack(a)
    pb, probb = stack(b)
    return units, ids, y, pa, proba, pb, probb


def run_comparison(runs, spec):
    task, regime, k, kind = spec["tarefa"], spec["regime"], N_CLASSES[spec["tarefa"]], KIND[spec["tarefa"]]
    units, ids, y, pa, proba, pb, probb = comparison_arrays(runs, task, regime, spec["a"], spec["b"])
    score_a, score_b = score(confusion(y, pa, k), kind), score(confusion(y, pb, k), kind)
    permutation = permutation_test(y, pa, pb, k, kind)
    per_unit = []
    for i, unit in enumerate(units):
        single = permutation_test(y, pa[i:i + 1], pb[i:i + 1], k, kind)
        per_unit.append({"sorteio": unit[0], "semente": unit[1], "a": float(score_a[i]), "b": float(score_b[i]),
                         "diferenca": float(score_a[i] - score_b[i]), "teste_descritivo_p": single["p"],
                         "teste_descritivo_contagem": single["count_at_least_observed"]})
    vote_a, vote_b = majority_vote(pa, proba, k), majority_vote(pb, probb, k)
    return {**spec, "conjunto": TEST_SET[task], "metrica": "Macro-F1" if task == DIAG else "F1 da classe anômala",
            "unidade": "semente" if regime == FULL else "sorteio (semente do sorteio)", "unidades": len(units), "janelas": len(ids),
            "execucoes": len(run_indexes(ids)[1]), "media_a": float(score_a.mean()), "media_b": float(score_b.mean()),
            "diferenca": permutation["observed"], "permutacao": permutation, "por_unidade": per_unit,
            "ic95": bootstrap_ci(y, pa, pb, ids, k, kind), "mcnemar": mcnemar(y, vote_a, vote_b),
            "janelas_discordantes": int(np.any(pa != pb, axis=0).sum()), "previsoes_identicas": bool(np.array_equal(pa, pb))}


def conclude(results):
    for family in FAMILIES:
        members = [r for r in results if r["familia"] == family]
        for r, adjusted in zip(members, holm([r["permutacao"]["p"] for r in members])):
            r["p_holm"] = adjusted
    for r in results:
        r.setdefault("p_holm", None)
        decisive = r["p_holm"] if r["familia"] else r["permutacao"]["p"]
        r["p_decisao"] = decisive
        r["significativa"] = decisive < ALPHA
        if not r["significativa"]:
            r["conclusao"] = "sem diferença significativa"
        else:
            better = r["a"] if r["diferenca"] > 0 else r["b"]
            r["conclusao"] = f"diferença significativa ({SHORT[better]} maior)"
    return results


# ---------------------------------------------------------------- verification
def sha(path):
    return data.sha256_bytes(Path(path).read_bytes())


def verify_inputs():
    """Part 1.1: pre-registration (and every file it references; moved code through reorganizacao_codigo.json), the run_config of
    the two pre-training checkpoints, and the SHA-256 manifests of the four result folders and of results/cenario2."""
    pre, verification = classicos.verify_preregistration()
    moves = classicos.read_code_moves()
    verification["code_accepted_through_correspondence"] = sorted(n for n in pre["code_sha256"] if n in moves)
    run_configs = {}
    for model in ("bert_mlm_2b", "bert_luk"):
        ck = pre["modelos"][model]["checkpoint"]
        path = Path(ck["path"]).parent / "run_config.json"
        if sha(path) != ck["run_config_sha256"]:
            raise data.ModelError(f"{path} differs from the pre-registration")
        run_configs[str(path)] = ck["run_config_sha256"]
    verification["run_config"] = run_configs
    verification["manifests"] = {s: data.verify_manifest(RESULTS / s) for s in SOURCES + ("cenario2",)}
    return pre, verification


# ---------------------------------------------------------------- descriptive summaries
def mean_std(values):
    return metrics.mean_std(list(values))


def descriptive(runs, recomputed):
    out = {"diagnostico": {}, "deteccao": {}, "deteccao_macro_f1": {}, "diagnostico_confusao_teste_principal": {}}
    for group, per_unit in recomputed.items():
        task, model, regime = group
        name = f"{model}/{regime}"
        values = list(per_unit.values())
        if task == DIAG:
            out["diagnostico"][name] = {"runs": len(values), **{c: classicos.summarize_multiclass([v[c] for v in values]) for c in SETS[1:]}}
            out["diagnostico_confusao_teste_principal"][name] = out["diagnostico"][name]["test_principal"]["confusion_matrix_sum"]
        else:
            out["deteccao"][name] = {"runs": len(values), **classicos.summarize_detection(values)}
            macro = []
            for unit in per_unit:
                sec = runs[(task, model, regime, *unit, "test_secundario")]
                macro.append(float(score(confusion(sec["y"], sec["pred"], 2), "macro")[0]))
            out["deteccao_macro_f1"][name] = {"runs": len(values), **mean_std(macro), "conjunto": "test_secundario (todas)"}
    return out


def cost_summary(pre):
    cost = {}
    for model, source in (("bert_base", "e2"), ("bert_mlm_2b", "bert-mlm-2b"), ("bert_luk", "bert-luk")):
        saved = json.loads((RESULTS / source / "metrics.json").read_text(encoding="utf-8"))["cost"]
        keys = [f"diagnostico/{s}" for s in pre["sementes"]["dados_completos"]]
        train = [saved["train_seconds"]["full_data"][k] for k in keys]
        with_val = [saved["run_seconds_with_validation"]["full_data"][k] for k in keys]
        vram = [saved["peak_vram_mb"]["full_data"][k] for k in keys]
        inference = saved["inference"]
        cost[model] = {
            "fine_tuning": {"runs": "diagnóstico, dados completos, sementes 42-46", "train_seconds": mean_std(train),
                            "seconds_with_validation": mean_std(with_val),
                            "peak_vram_allocated_mib_max": max(v["allocated"] for v in vram),
                            "peak_vram_reserved_mib_max": max(v["reserved"] for v in vram), "source": f"results/{source}/metrics.json cost"},
            "inference": {k: inference[k] for k in ("latency_ms_mean", "latency_ms_median", "latency_ms_p95", "latency_ms_max",
                                                     "peak_vram_allocated_mb", "peak_vram_reserved_mb", "windows_measured", "passes",
                                                     "discarded_first", "model")},
        }
        cost[model]["inference"]["source"] = f"results/{source}/metrics.json cost.inference"
        if model == "bert_base":
            cost[model]["pretraining"] = None
            continue
        folder = Path(pre["modelos"][model]["checkpoint"]["path"]).parent
        config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
        rows = data_v2.read_csv(folder / "train_log.csv")
        logged = [r for r in rows if r["step"] != "finished"]
        finished = [r for r in rows if r["step"] == "finished"]
        steps = sum(int(r["steps_in_row"]) for r in logged)
        seconds_logged = sum(float(r["seconds_per_step"]) * int(r["steps_in_row"]) for r in logged)
        wall = (dt.datetime.fromisoformat(finished[-1]["time_local"]) - dt.datetime.fromisoformat(config["created_local"])).total_seconds()
        cost[model]["pretraining"] = {
            "wall_seconds": wall, "wall_from": config["created_local"], "wall_to": finished[-1]["time_local"],
            "total_steps": config["hyperparameters"]["total_steps"], "steps_logged": steps, "seconds_per_step": seconds_logged / steps,
            "peak_vram_reserved_mib": max(int(r["vram_peak_reserved_mib"]) for r in logged),
            "peak_vram_allocated_mib": max(int(r["vram_allocated_mib"]) for r in logged),
            "vram_cap_fraction": config["vram_cap"]["fraction"],
            "source": f"{folder}\\run_config.json (created_local, hyperparameters) and train_log.csv (time_local, steps_in_row, "
                      "seconds_per_step, vram_allocated_mib, vram_peak_reserved_mib)",
            "train_log_sha256": sha(folder / "train_log.csv")}
    e1 = json.loads((RESULTS / "e1" / "metrics.json").read_text(encoding="utf-8"))
    cost["tfidf_rf"] = {"per_run_train_seconds": None,
                        "note": "the E1 did not record the training time of each run; only the time of each whole stage (CPU, no GPU)",
                        "stage_evaluate_seconds": e1["seconds_evaluate"], "stage_select_seconds": e1["seconds_select"],
                        "stage_evaluate_content": "every E1 run (majority class, TF-IDF + RF contagem and presenca, full data and the 15 "
                                                  "draws of each task), with vectorization, fit and prediction of validation and test",
                        "stage_select_content": "grid of the diagnosis (8 configurations x 5 seeds x 2 versions), shuffled-labels control "
                                                "and every validation result",
                        "source": "results/e1/metrics.json (seconds_evaluate, seconds_select)"}
    return cost


# ---------------------------------------------------------------- formatting
def fmt(x, digits=3):
    if x is None:
        return "--"
    text = f"{x:.{digits}f}"
    if float(text) == 0:
        text = f"{0:.{digits}f}"
    text = text.replace(".", ",")
    return "$-$" + text[1:] if text.startswith("-") else text


def thousands(x):
    """Integer with a thin space between thousands (LaTeX), e.g. 10\\,928."""
    return f"{int(round(x)):,}".replace(",", "\\,")


def pm(summary, runs, digits=3):
    return fmt(summary["mean"], digits) if runs == 1 else f"{fmt(summary['mean'], digits)} $\\pm$ {fmt(summary['std'], digits)}"


def latex(caption, label, header, rows, align, note=None, size="\\small"):
    width = len(header)
    lines = ["% Gerado por models/avaliacao_ampliada/comparacao.py; requer \\usepackage{booktabs}", "\\begin{table}[htbp]",
             "\\centering", size, f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    first = True
    for row in rows:
        if isinstance(row, tuple) and row[0] == "group":
            if not first:
                lines.append("\\midrule")
            lines.append(f"\\multicolumn{{{width}}}{{l}}{{\\textit{{{row[1]}}}}} \\\\")
        else:
            lines.append(" & ".join(row) + " \\\\")
        first = False
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines += ["", "\\smallskip", f"\\parbox{{0.95\\linewidth}}{{\\footnotesize {note}}}"]
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


class Writer:
    """Writes every output once (mode x) and keeps the list for the manifest."""

    def __init__(self, folder):
        self.folder, self.files = folder, []

    def text(self, name, content):
        with open(self.folder / name, "x", encoding="utf-8", newline="\n") as f:
            f.write(content)
        self.files.append(name)

    def csv(self, name, header, rows):
        with open(self.folder / name, "x", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        self.files.append(name)

    def json(self, name, content):
        self.text(name, json.dumps(content, ensure_ascii=False, indent=2, default=to_builtin) + "\n")

    def figure(self, name, fig):
        import matplotlib.pyplot as plt
        fig.savefig(self.folder / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None, "Producer": None}, bbox_inches="tight")
        fig.savefig(self.folder / f"{name}.png", dpi=300, metadata={"Software": None}, bbox_inches="tight")
        plt.close(fig)
        self.files += [f"{name}.pdf", f"{name}.png"]

    def manifest(self):
        lines = [f"{sha(self.folder / n)}  {n}" for n in sorted(self.files)]
        self.text("manifest.sha256", "\n".join(lines) + "\n")


def to_builtin(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


# ---------------------------------------------------------------- tables
DET_SCALARS = (("precision", "Precisão"), ("recall", "Revocação"), ("f1", "F1"), ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"))
CUT_NAMES = {"todas": "Todas as janelas do teste", "so_visiveis": "Só visíveis: normais + visíveis",
             "so_lentidao": "Só lentidão: normais + só lentidão, revocação"}
SET_NAMES = {"test_principal": "Teste principal (normais + visíveis, 102 janelas)",
             "test_secundario": "Teste secundário (todas as janelas, com as 4 só de lentidão, 106 janelas)"}


def table_detection(w, desc, runs):
    sizes = {}
    sec = runs[(DET, "bert_luk", FULL, "", "42", "test_secundario")]
    for cut, idx in detection_cuts(sec, runs[(DET, "bert_luk", FULL, "", "42", "test_principal")]).items():
        sizes[cut] = len(idx)
    csv_rows, tex_rows = [], []
    for cut in ("todas", "so_visiveis", "so_lentidao"):
        tex_rows.append(("group", f"{CUT_NAMES[cut]} ({sizes[cut]} janelas)"))
        for model in MODEL_ORDER:
            entry = desc["deteccao"].get(f"{model}/{FULL}")
            if entry is None:
                continue
            values = entry["test_cuts"][cut]
            for metric, _ in DET_SCALARS:
                csv_rows.append([model, NAMES[model], cut, sizes[cut], metric, values[metric]["mean"], values[metric]["std"], entry["runs"]])
            cells = [pm(values[m], entry["runs"]) if (cut != "so_lentidao" or m == "recall") else "--" for m, _ in DET_SCALARS]
            tex_rows.append([NAMES[model], *cells])
    w.csv("tabela_deteccao.csv", ["modelo", "nome", "recorte", "janelas", "metrica", "media", "desvio", "rodadas"], csv_rows)
    w.text("tabela_deteccao.tex", latex(
        "Detecção com os dados completos: métricas no teste (média $\\pm$ desvio-padrão nas 5 sementes)", "tab:deteccao-completos",
        ["Modelo", *[n for _, n in DET_SCALARS]], tex_rows, "l" + "c" * len(DET_SCALARS),
        "Classe mais comum e regra simples são determinísticas (uma rodada, sem desvio). No recorte só lentidão, o pré-registro usa só a "
        "revocação das janelas só de lentidão; as demais métricas desse recorte estão no CSV."))


def table_diagnosis(w, desc):
    models = (MAJORITY,) + TFIDF + LMS
    csv_rows, tex_rows = [], []
    for conjunto in ("test_principal", "test_secundario"):
        tex_rows.append(("group", SET_NAMES[conjunto]))
        for model in models:
            entry = desc["diagnostico"][f"{model}/{FULL}"]
            s = entry[conjunto]
            for metric in ("accuracy", "weighted_f1", "macro_f1"):
                csv_rows.append([model, NAMES[model], conjunto, metric, s[metric]["mean"], s[metric]["std"], entry["runs"]])
            for c in CLASSES:
                csv_rows.append([model, NAMES[model], conjunto, f"f1_{c}", s["f1_by_class"][c]["mean"], s["f1_by_class"][c]["std"], entry["runs"]])
            tex_rows.append([NAMES[model], pm(s["accuracy"], entry["runs"]), pm(s["weighted_f1"], entry["runs"]), pm(s["macro_f1"], entry["runs"]),
                             *[pm(s["f1_by_class"][c], entry["runs"]) for c in CLASSES]])
    w.csv("tabela_diagnostico.csv", ["modelo", "nome", "conjunto", "metrica", "media", "desvio", "rodadas"], csv_rows)
    w.text("tabela_diagnostico.tex", latex(
        "Diagnóstico com os dados completos: acurácia, F1 ponderado, Macro-F1 e F1 por classe (média $\\pm$ desvio-padrão nas 5 sementes)",
        "tab:diagnostico-completos", ["Modelo", "Acurácia", "F1 ponderado", "Macro-F1", "F1 Normal", "F1 CPU", "F1 Rede", "F1 Combinado"],
        tex_rows, "lccccccc", "Classe mais comum: uma rodada (determinística, sempre Normal). O teste secundário é só descritivo, sem testes "
        "estatísticos.", size="\\scriptsize"))


def table_seeds(w, principal):
    rows = principal["por_unidade"]
    luk = {r["semente"]: r["a"] for r in rows}
    mlm = {r["semente"]: r["b"] for r in rows}
    base = {}
    for r in principal["_bert_base_por_semente"]:
        base[r["semente"]] = r["valor"]
    seeds = [r["semente"] for r in rows]
    csv_rows = [[s, base[s], mlm[s], luk[s], luk[s] - mlm[s]] for s in seeds]
    columns = [list(c) for c in zip(*[[base[s], mlm[s], luk[s], luk[s] - mlm[s]] for s in seeds])]
    csv_rows.append(["media", *[statistics.fmean(c) for c in columns]])
    csv_rows.append(["desvio", *[statistics.stdev(c) for c in columns]])
    tex_rows = [[s, *[fmt(v) for v in row[1:]]] for s, row in zip(seeds, csv_rows)]
    tex_rows.append(["Média", *[fmt(v) for v in csv_rows[-2][1:]]])
    tex_rows.append(["Desvio-padrão", *[fmt(v) for v in csv_rows[-1][1:]]])
    w.csv("tabela_macro_f1_por_semente.csv", ["semente", "bert_base", "bert_mlm_2b", "bert_luk", "luk_menos_2b"], csv_rows)
    w.text("tabela_macro_f1_por_semente.tex", latex(
        "Macro-F1 do diagnóstico por semente no teste principal (102 janelas)", "tab:macro-f1-sementes",
        ["Semente", "BERT-base", "BERT + MLM (2b)", "BERT + LUK", "LUK $-$ 2b"], tex_rows, "lcccc"))


def table_low_resources(w, desc):
    panels = (("diagnostico", "macro_f1", (MAJORITY,) + TFIDF + LMS, "Diagnóstico: Macro-F1 (teste principal)"),
              ("deteccao", "f1", MODEL_ORDER, "Detecção: F1 da classe anômala (todas as janelas do teste)"),
              ("deteccao_macro", "macro_f1", MODEL_ORDER, "Detecção: Macro-F1 (todas as janelas do teste; descritiva)"))
    csv_rows, tex_rows = [], []
    for panel, metric, models, title in panels:
        tex_rows.append(("group", title))
        for model in models:
            cells = []
            for regime in REGIMES:
                source_regime = FULL if model == RULE else regime
                if panel == "diagnostico":
                    entry = desc["diagnostico"].get(f"{model}/{source_regime}")
                    s, n = (entry["test_principal"]["macro_f1"], entry["runs"]) if entry else (None, 0)
                elif panel == "deteccao":
                    entry = desc["deteccao"].get(f"{model}/{source_regime}")
                    s, n = (entry["test_cuts"]["todas"]["f1"], entry["runs"]) if entry else (None, 0)
                else:
                    entry = desc["deteccao_macro_f1"].get(f"{model}/{source_regime}")
                    s, n = (entry, entry["runs"]) if entry else (None, 0)
                if s is None:
                    cells.append("--")
                    continue
                csv_rows.append([panel.split("_")[0], metric, model, NAMES[model], regime, s["mean"], s["std"], n,
                                 "mesmas previsões dos dados completos" if model == RULE else ""])
                cells.append(pm(s, n))
            tex_rows.append([NAMES[model], *cells])
    w.csv("tabela_baixos_recursos.csv", ["tarefa", "metrica", "modelo", "nome", "regime", "media", "desvio", "rodadas", "nota"], csv_rows)
    w.text("tabela_baixos_recursos.tex", latex(
        "Baixos recursos: média $\\pm$ desvio-padrão nos 5 sorteios de cada fração (1\\%, 5\\% e 10\\% do treino) e nas 5 sementes dos "
        "dados completos (100\\%)", "tab:baixos-recursos", ["Modelo", "1\\%", "5\\%", "10\\%", "100\\%"], tex_rows, "lcccc",
        "Validação e teste completos em todas as frações. A regra simples não aprende: as mesmas previsões em todas as frações. A métrica "
        "principal da detecção no pré-registro é o F1 da classe anômala; a Macro-F1 da detecção é só descritiva."))


def table_tests(w, results):
    header = ["id", "familia", "tarefa", "regime", "comparacao", "modelo_a", "modelo_b", "metrica", "conjunto", "unidade", "media_a",
              "media_b", "diferenca", "p_bruto", "p_holm", "ic95_inferior", "ic95_superior", "janelas_discordantes", "janelas",
              "previsoes_identicas", "mcnemar_a_certo_b_errado", "mcnemar_a_errado_b_certo", "mcnemar_p", "conclusao_alfa_0_05"]
    csv_rows, tex_rows, current = [], [], object()
    for r in results:
        csv_rows.append([r["id"], r["familia"] or "principal", r["tarefa"], r["regime"], r["rotulo"], r["a"], r["b"], r["metrica"], r["conjunto"],
                         r["unidade"], r["media_a"], r["media_b"], r["diferenca"], r["permutacao"]["p"], r["p_holm"], r["ic95"]["lower"],
                         r["ic95"]["upper"], r["janelas_discordantes"], r["janelas"], int(r["previsoes_identicas"]),
                         r["mcnemar"]["a_certo_b_errado"], r["mcnemar"]["a_errado_b_certo"], r["mcnemar"]["p"], r["conclusao"]])
        if r["familia"] != current:
            tex_rows.append(("group", FAMILY_NAMES[r["familia"]]))
            current = r["familia"]
        cells = [r["rotulo"], REGIME_NAMES[r["regime"]].replace("%", "\\%"), fmt(r["diferenca"]), fmt(r["permutacao"]["p"], 4),
                 fmt(r["p_holm"], 4) if r["p_holm"] is not None else "--", f"[{fmt(r['ic95']['lower'])}; {fmt(r['ic95']['upper'])}]",
                 f"{r['janelas_discordantes']}/{r['janelas']}", fmt(r["mcnemar"]["p"], 4), "sim" if r["significativa"] else "não"]
        if r["principal"]:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        tex_rows.append(cells)
    w.csv("tabela_testes_estatisticos.csv", header, csv_rows)
    w.text("tabela_testes_estatisticos.tex", latex(
        "Testes estatísticos: diferença da média da métrica principal (A $-$ B), p-valor da aleatorização aproximada pareada (10\\,000 "
        "permutações, bilateral), p-valor ajustado por Holm na família, intervalo de 95\\% por bootstrap das execuções do teste, janelas "
        "com previsões diferentes em pelo menos uma unidade e McNemar exato sobre o voto majoritário",
        "tab:testes", ["Comparação", "Fração", "Diferença", "p", "p Holm", "IC 95\\%", "Discord.", "McNemar p", "Signif."], tex_rows,
        "llccccccc", "Diagnóstico: Macro-F1 no teste principal (102 janelas); detecção: F1 da classe anômala em todas as janelas do teste "
        "(106). Unidade de pareamento: semente (dados completos) ou sorteio (baixos recursos). Significância a $\\alpha = 0{,}05$ pelo p de Holm "
        "nas famílias e pelo p bruto na comparação principal (em negrito). Menor p possível: 1/10\\,001 $\\approx$ 0,0001.",
        size="\\scriptsize"))


def table_cost(w, cost):
    lms = LMS
    hours = lambda s: s / 3600
    rows = [("group", "Pré-treinamento (50 épocas, 189\\,300 passos)")]
    pre_rows = (("Tempo total (h)", lambda p: fmt(hours(p["wall_seconds"]), 2)), ("Segundos por passo", lambda p: fmt(p["seconds_per_step"], 4)),
                ("VRAM pico reservada (MiB)", lambda p: thousands(p["peak_vram_reserved_mib"])),
                ("VRAM alocada máxima (MiB)", lambda p: thousands(p["peak_vram_allocated_mib"])))
    csv_rows = []
    for label, getter in pre_rows:
        rows.append([label, *["não se aplica" if cost[m]["pretraining"] is None else getter(cost[m]["pretraining"]) for m in lms]])
    rows.append(("group", "Fine-tuning por rodada (diagnóstico, dados completos, 780 passos, 5 sementes)"))
    ft = cost
    rows.append(["Tempo de treino (s)", *[pm(ft[m]["fine_tuning"]["train_seconds"], 5, 1) for m in lms]])
    rows.append(["Tempo com validação (s)", *[pm(ft[m]["fine_tuning"]["seconds_with_validation"], 5, 1) for m in lms]])
    rows.append(["VRAM pico alocada (MiB)", *[thousands(ft[m]["fine_tuning"]["peak_vram_allocated_mib_max"]) for m in lms]])
    rows.append(["VRAM pico reservada (MiB)", *[thousands(ft[m]["fine_tuning"]["peak_vram_reserved_mib_max"]) for m in lms]])
    rows.append(("group", "Inferência (GPU, lote 1, por janela do teste)"))
    rows.append(["Latência média (ms)", *[fmt(cost[m]["inference"]["latency_ms_mean"], 2) for m in lms]])
    rows.append(["Latência p95 (ms)", *[fmt(cost[m]["inference"]["latency_ms_p95"], 2) for m in lms]])
    rows.append(["VRAM pico alocada (MiB)", *[fmt(cost[m]["inference"]["peak_vram_allocated_mb"], 0) for m in lms]])
    rows.append(["VRAM pico reservada (MiB)", *[fmt(cost[m]["inference"]["peak_vram_reserved_mb"], 0) for m in lms]])
    tf = cost["tfidf_rf"]
    rows.append(("group", "TF-IDF + RF (CPU, sem GPU)"))
    rows.append(["Tempo de treino por rodada", "\\multicolumn{3}{l}{não registrado pelo E1}"])
    rows.append(["Etapa de avaliação do E1 inteira (s)", f"\\multicolumn{{3}}{{l}}{{{fmt(tf['stage_evaluate_seconds'], 1)} (todas as rodadas do E1)}}"])
    rows.append(["Etapa de seleção do E1 inteira (s)", f"\\multicolumn{{3}}{{l}}{{{fmt(tf['stage_select_seconds'], 1)} (grade, controle e validação)}}"])
    for m in lms:
        p, f, i = cost[m]["pretraining"], cost[m]["fine_tuning"], cost[m]["inference"]
        if p:
            csv_rows += [[m, "pretreino_tempo_total_s", p["wall_seconds"], ""], [m, "pretreino_segundos_por_passo", p["seconds_per_step"], ""],
                         [m, "pretreino_vram_pico_reservada_mib", p["peak_vram_reserved_mib"], ""],
                         [m, "pretreino_vram_alocada_max_mib", p["peak_vram_allocated_mib"], ""]]
        else:
            csv_rows.append([m, "pretreino", "", "não se aplica (checkpoint público bert-base-uncased)"])
        csv_rows += [[m, "finetuning_treino_s", f["train_seconds"]["mean"], f"desvio {f['train_seconds']['std']}"],
                     [m, "finetuning_com_validacao_s", f["seconds_with_validation"]["mean"], f"desvio {f['seconds_with_validation']['std']}"],
                     [m, "finetuning_vram_pico_alocada_mib", f["peak_vram_allocated_mib_max"], ""],
                     [m, "finetuning_vram_pico_reservada_mib", f["peak_vram_reserved_mib_max"], ""],
                     [m, "inferencia_latencia_media_ms", i["latency_ms_mean"], ""], [m, "inferencia_latencia_p95_ms", i["latency_ms_p95"], ""],
                     [m, "inferencia_vram_pico_alocada_mib", i["peak_vram_allocated_mb"], ""],
                     [m, "inferencia_vram_pico_reservada_mib", i["peak_vram_reserved_mb"], ""]]
    csv_rows += [["tfidf_rf", "treino_por_rodada_s", "", "não registrado pelo E1"],
                 ["tfidf_rf", "etapa_avaliacao_e1_s", tf["stage_evaluate_seconds"], tf["stage_evaluate_content"]],
                 ["tfidf_rf", "etapa_selecao_e1_s", tf["stage_select_seconds"], tf["stage_select_content"]]]
    w.csv("tabela_custo.csv", ["modelo", "item", "valor", "nota"], csv_rows)
    w.text("tabela_custo.tex", latex(
        "Custo computacional: pré-treinamento, fine-tuning e inferência dos modelos de linguagem (NVIDIA GeForce RTX 4070, 12\\,282 MiB) e "
        "tempo do TF-IDF + RF", "tab:custo", ["Item", "BERT-base", "BERT + MLM (2b)", "BERT + LUK"], rows, "lccc",
        "Pré-treinamento: do início registrado em run\\_config.json ao fim em train\\_log.csv; segundos por passo = média ponderada dos "
        "registros a cada 500 passos. O LUK processa dois textos por item (perda ChatgptEnhanceLoss). A VRAM reservada reflete o "
        "alocador do PyTorch (limite de 90\\% da GPU no pré-treinamento). Fine-tuning: média $\\pm$ desvio nas 5 sementes do diagnóstico. "
        "O E1 não registrou o tempo de cada rodada do TF-IDF + RF; os tempos mostrados são de etapas inteiras."))


# ---------------------------------------------------------------- figures
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#dddddd",
                         "grid.linewidth": 0.6, "axes.axisbelow": True, "svg.hashsalt": "tcc", "pdf.fonttype": 42})
    return plt


def decimal_comma(axis):
    """Tick labels with a decimal comma (Portuguese)."""
    from matplotlib.ticker import FuncFormatter
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".").replace(".", ",").replace("-", "−")
                                           if v != 0 else "0"))


def figure_low_resources(w, desc, plt):
    models = (MAJORITY,) + TFIDF + LMS
    fractions = [1, 5, 10, 100]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    offsets = np.linspace(-0.05, 0.05, len(models))
    for i, model in enumerate(models):
        means, stds = [], []
        for regime in REGIMES:
            s = desc["diagnostico"][f"{model}/{regime}"]["test_principal"]["macro_f1"]
            means.append(s["mean"])
            stds.append(s["std"])
        x = [f * 10 ** offsets[i] for f in fractions]
        ax.errorbar(x, means, yerr=stds, label=NAMES[model], color=SERIES_COLORS[i], marker=SERIES_MARKERS[i], linestyle=SERIES_LINES[i],
                    linewidth=2, markersize=7, capsize=3, markeredgecolor="#0b0b0b", markeredgewidth=0.6)
    ax.set_xscale("log")
    ax.set_xticks(fractions)
    ax.set_xticklabels(["1%", "5%", "10%", "100%"])
    ax.minorticks_off()
    ax.set_xlabel("Fração do conjunto de treino")
    ax.set_ylabel("Macro-F1 no teste principal")
    ax.set_ylim(0, 1.02)
    decimal_comma(ax.yaxis)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    w.figure("figura_baixos_recursos_diagnostico", fig)


def figure_confusions(w, desc, plt):
    models = TFIDF + LMS
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.0))
    image = None
    for ax, model in zip(axes.flat, models):
        matrix = np.array(desc["diagnostico_confusao_teste_principal"][f"{model}/{FULL}"])
        share = matrix / matrix.sum(1, keepdims=True)
        image = ax.imshow(share, cmap="Greys", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=10, color="white" if share[i, j] > 0.5 else "#0b0b0b")
        ax.set_title(NAMES[model], fontsize=10)
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, fontsize=8)
        ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel("Classe prevista", fontsize=9)
        ax.set_ylabel("Classe verdadeira", fontsize=9)
        ax.grid(False)
    axes.flat[-1].axis("off")
    bar = fig.colorbar(image, ax=axes.flat[-1], fraction=0.5, location="left")
    bar.set_label("Proporção da linha (classe verdadeira)")
    decimal_comma(bar.ax.yaxis)
    fig.tight_layout()
    w.figure("figura_matrizes_confusao_diagnostico", fig)


def figure_difference(w, principal, plt):
    rows = principal["por_unidade"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = list(range(len(rows)))
    ax.axhline(0, color="#52514e", linewidth=1, linestyle="--")
    ax.plot(x, [r["diferenca"] for r in rows], linestyle="none", marker="o", markersize=8, color="#2a78d6", markeredgecolor="#0b0b0b",
            label="Diferença em cada semente")
    mean_x = len(rows) + 0.6
    ci = principal["ic95"]
    ax.errorbar([mean_x], [principal["diferenca"]], yerr=[[principal["diferenca"] - ci["lower"]], [ci["upper"] - principal["diferenca"]]],
                marker="D", markersize=8, color="#0b0b0b", capsize=5, linewidth=2, label="Média das 5 sementes e IC 95% (bootstrap por execução)")
    ax.set_xticks(x + [mean_x])
    ax.set_xticklabels([r["semente"] for r in rows] + ["Média"])
    ax.set_xlabel("Semente de treino")
    ax.set_ylabel("Macro-F1: LUK − 2b")
    decimal_comma(ax.yaxis)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    w.figure("figura_diferenca_luk_2b", fig)


# ---------------------------------------------------------------- main
def main(argv=None):
    start = time.perf_counter()
    started_local = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    if OUTPUT.exists():
        raise data.ModelError(f"{OUTPUT} already exists; nothing is overwritten")
    pre, verification = verify_inputs()
    print("1.1 hashes ok:", len(verification["files"]), "files of the pre-registration;", "manifests", verification["manifests"])
    runs = load_predictions()
    recomputed, metric_check = check_metrics(runs)
    print("1.2 metrics:", metric_check["runs_compared"], "runs,", metric_check["aggregates_compared"], "aggregates, max diff",
          metric_check["max_abs_difference"], "problems", len(metric_check["problems"]))
    if metric_check["problems"]:
        raise data.ModelError("recomputed metrics differ from metrics.json: " + "; ".join(metric_check["problems"][:20]))
    saved_runs = {}
    for source in SOURCES:
        for r in json.loads((RESULTS / source / "metrics.json").read_text(encoding="utf-8"))["runs"]:
            saved_runs[run_key(r)] = r
    d1_losses, d1_rate = validation_losses_d1()
    best = {f"{t}/{g}": choose_best(recomputed, saved_runs, d1_losses, t, g) for t in TASKS for g in REGIMES}
    comparisons = plan(best)
    pairing = {c["id"]: check_pairing(runs, c["tarefa"], c["regime"], c["a"], c["b"]) for c in comparisons}
    for model in MODEL_ORDER:
        for task in TASKS:
            for regime in REGIMES:
                if (task, model, regime) in recomputed and model != "bert_luk" and (task, "bert_luk", regime) in recomputed:
                    if units_of(runs, task, model, regime) == units_of(runs, task, "bert_luk", regime):
                        key = f"{task}/{regime}/bert_luk-{model}"
                        pairing.setdefault(key, check_pairing(runs, task, regime, "bert_luk", model))
    missing = {k: v["problems"] for k, v in pairing.items() if v["problems"]}
    print("1.3 pairing:", len(pairing), "pairs checked, problems:", missing or "none")
    if missing:
        raise data.ModelError(f"pairing problems: {missing}")
    OUTPUT.mkdir(parents=False, exist_ok=False)
    w = Writer(OUTPUT)
    note = {**ADVISOR_NOTE, "registrado_local": started_local, "preregistro_sha256": classicos.PREREGISTRO_SHA256,
            "compatibilidade": "o pré-registro já diz que, em cada permutação, as 5 previsões de sementes de uma janela trocam de modelo juntas; "
                               "o esclarecimento fixa que cada rodada de um modelo é pareada com a rodada do outro de mesma semente",
            "aplicacao": {
                "dados_completos": "unidade = semente (42-46). Para cada janela do teste, uma moeda por permutação: se 1, as previsões da janela "
                                   "trocam entre os dois modelos em todas as sementes ao mesmo tempo, a rodada de semente s de um com a de semente s "
                                   "do outro. Estatística = diferença das médias da métrica nas 5 sementes; 10000 permutações, semente 20261109, "
                                   "bilateral, p = (b + 1) / (N + 1).",
                "baixos_recursos": "unidade = sorteio (0-4); a semente de treino é a do sorteio, a mesma nos dois modelos. A moeda da janela vale "
                                   "para os 5 sorteios ao mesmo tempo, cada troca entre as rodadas do mesmo sorteio (mesma semente).",
                "mcnemar_e_bootstrap": "voto majoritário sobre as mesmas unidades; o bootstrap reamostra as execuções do teste e mantém o "
                                       "pareamento por unidade dentro de cada reamostra",
                "conferencia": "pareamento conferido antes dos testes (mesmas unidades e mesmas janelas em todos os conjuntos): ver "
                               "estatisticas.json, conferencias.pareamento"}}
    w.json("esclarecimento_orientador.json", note)
    results = conclude([run_comparison(runs, c) for c in comparisons])
    principal = results[0]
    base_units = units_of(runs, DIAG, "bert_base", FULL)
    principal["_bert_base_por_semente"] = [
        {"semente": u[1], "valor": float(recomputed[(DIAG, "bert_base", FULL)][u]["test_principal"]["macro_f1"])} for u in base_units]
    print("principal:", principal["diferenca"], principal["permutacao"]["p"], principal["ic95"])
    desc = descriptive(runs, recomputed)
    cost = cost_summary(pre)
    plt = setup_matplotlib()
    table_detection(w, desc, runs)
    table_diagnosis(w, desc)
    table_seeds(w, principal)
    table_low_resources(w, desc)
    table_tests(w, results)
    table_cost(w, cost)
    figure_low_resources(w, desc, plt)
    figure_confusions(w, desc, plt)
    figure_difference(w, principal, plt)
    import matplotlib
    principal.pop("_bert_base_por_semente")
    elapsed = time.perf_counter() - start
    w.json("estatisticas.json", {
        "created_local": started_local, "preregistro_sha256": classicos.PREREGISTRO_SHA256,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": __import__("scipy").__version__,
                        "scikit-learn": __import__("sklearn").__version__, "matplotlib": matplotlib.__version__, "gpu": "not used"},
        "conferencias": {"hashes": verification, "metricas": metric_check, "pareamento": pairing},
        "parametros": {"permutacoes": N_PERM, "semente_permutacao": PERM_SEED, "reamostras_bootstrap": N_BOOT, "semente_bootstrap": BOOT_SEED,
                       "alfa": ALPHA, "tolerancia_contagem": f"|estatística permutada| >= |observada| - {TOL}",
                       "p_valor": "(b + 1) / (N + 1), b = permutações com |estatística| >= |observada|",
                       "conjuntos": {"diagnostico": "test_principal (102)", "deteccao": "test_secundario = todas as janelas (106)"},
                       "f1_classe_ausente": "0 quando a classe não tem janela verdadeira nem prevista (zero_division=0), como nas etapas anteriores",
                       "desempate_mcnemar": "maior probabilidade média nas unidades entre as classes empatadas, depois Normal, CPU, Rede, Combinado",
                       "perda_validacao_bert_base_deteccao_completo": f"results/cenario2/selection.json, taxa {d1_rate}, melhor época de cada semente"},
        "escolha_melhores_modelos": best, "comparacoes": results, "descritivo": desc, "custo": cost,
        "segundos_execucao_ate_estatisticas": elapsed})
    w.manifest()
    print(f"ok: {len(w.files)} files in {OUTPUT}; {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()

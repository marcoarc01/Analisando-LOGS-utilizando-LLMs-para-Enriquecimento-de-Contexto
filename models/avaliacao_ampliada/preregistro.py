"""E0, part 3: pre-registration of the extended evaluation (JSON outside the repository), written before any training of it.
Hyperparameters are read from the code and the result files of Scenarios 1 and 2, not typed in. Checkpoint manifests are verified.
Then the SHA-256 of preregistro.json is written to preregistro.sha256. Run from the repository root with C:\\tcc-venv\\bert after
models.avaliacao_ampliada.causa and models.avaliacao_ampliada.sorteios: python -m models.avaliacao_ampliada.preregistro."""
import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

from models.cenario1 import cenario1 as c1
from models.cenario2 import cenario2 as c2m
from models.common import data, data_v2, window_text as wt
from models.avaliacao_ampliada import causa, sorteios
from models.exploratorio import poucos_dados

PRE_FILE = "preregistro.json"
HASH_FILE = "preregistro.sha256"
REPO = Path(__file__).resolve().parents[2]
CHECKPOINTS = {"2b": Path(r"C:\tcc-data\checkpoints\mlm2b-full"), "luk": Path(r"C:\tcc-data\checkpoints\luk-full")}
CODE = ("models/avaliacao_ampliada/causa.py", "models/avaliacao_ampliada/sorteios.py", "models/avaliacao_ampliada/preregistro.py", "models/common/window_text.py", "models/common/data_v2.py",
        "models/common/metrics.py", "models/cenario1/cenario1.py", "models/cenario1/cenario1_v2.py", "models/cenario2/cenario2.py",
        "models/exploratorio/poucos_dados.py")
INPUTS = ("labels/labels.csv", "labels/sets/train.csv", "labels/sets/validation.csv", "labels/sets/test.csv", "labels/manifest.sha256",
          "control/splits.json")
PERMUTATIONS = 10000
BOOTSTRAP = 10000
ALPHA = 0.05
STAT_SEED = 20261109


def sha(path):
    return data.sha256_bytes(Path(path).read_bytes())


def checkpoint(name, folder):
    """Path, manifest hash and model hash of a pre-trained checkpoint; every file listed in model.sha256 is checked."""
    manifest = folder / "model.sha256"
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        digest, file = line.split(" ", 1)
        file = file.lstrip("*")
        if sha(folder / file) != digest:
            raise data.ModelError(f"{name}: {file} differs from model.sha256")
        listed[file] = digest
    state = json.loads((folder / "state" / "latest.json").read_text(encoding="utf-8"))
    return {"path": str(folder / "model"), "manifest": str(manifest), "manifest_sha256": sha(manifest),
            "model_safetensors_sha256": listed["model/model.safetensors"], "files_checked": len(listed),
            "run_config_sha256": sha(folder / "run_config.json"), "pretraining_global_step": state["global_step"],
            "pretraining_epochs": state["epochs_done"]}


def bert_finetuning():
    """Configuration of the D1 (Scenario 2), from cenario2.py and results/cenario2/{config,selection}.json."""
    config = json.loads((c2m.RESULTS / "config.json").read_text(encoding="utf-8"))
    selection = json.loads((c2m.RESULTS / "selection.json").read_text(encoding="utf-8"))
    train_windows = 307
    steps_per_epoch = math.ceil(train_windows / c2m.BATCH_SIZE)
    steps = c2m.EPOCHS * steps_per_epoch
    return {
        "sources": ["models/cenario2/cenario2.py (constants, WindowClassifier, fit, better, parameter_groups, set_seed)",
                    str(c2m.RESULTS / "config.json"), str(c2m.RESULTS / "selection.json")],
        "learning_rate": selection["chosen"]["learning_rate"],
        "learning_rate_note": f"fixed: the rate chosen in the D1 among {list(c2m.LEARNING_RATES)} (all reached validation F1 1.0; "
                              "tie broken by the validation loss). No new rate search for any model or task.",
        "epochs": c2m.EPOCHS, "batch_size": c2m.BATCH_SIZE, "optimizer": config["training"]["optimizer"],
        "weight_decay": c2m.WEIGHT_DECAY, "no_decay": list(c2m.NO_DECAY), "warmup_fraction": c2m.WARMUP_FRACTION,
        "schedule": config["training"]["schedule"], "max_grad_norm": c2m.MAX_GRAD_NORM, "precision": config["training"]["precision"],
        "max_tokens": wt.MAX_TOKENS, "truncation": False, "pooling": config["model"]["pooling"],
        "head": "one linear layer (hidden 768 -> number of classes), no dropout of its own: 2 outputs in detection, 4 in diagnosis",
        "loss": config["model"]["loss"], "encoder_dropout": config["model"]["encoder_dropout"], "pooler_layer": config["model"]["pooler_layer"],
        "decision": {"deteccao": f"anomalous when the softmax probability of the anomalous class >= {c2m.THRESHOLD} (as the D1)",
                     "diagnostico": "argmax of the softmax over the 4 classes"},
        "determinism": "set_seed(seed): random, numpy, torch, cuda; cudnn.deterministic=True, benchmark=False; batch order torch.randperm "
                       "with a generator seeded by the training seed; CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "tokenizer": f"{c2m.MODEL} revision {c2m.REVISION} for the three models (the saved tokenizers of 2b and LUK give the same ids)",
        "encoders": {"bert": f"{c2m.MODEL} revision {c2m.REVISION}, AutoModel add_pooling_layer=False",
                     "2b": "AutoModel.from_pretrained(<2b checkpoint>, add_pooling_layer=False)",
                     "luk": "AutoModel.from_pretrained(<LUK checkpoint>, add_pooling_layer=False)"},
        "same_for": "BERT, 2b and LUK, detection and diagnosis; only the number of output classes changes",
        "epoch_selection": {"deteccao": "highest validation F1 of the anomalous class, then lowest validation loss, then the earliest "
                                        "(cenario2.better)",
                            "diagnostico": "highest validation Macro-F1, then lowest validation loss, then the earliest"},
        "full_data_budget": {"train_windows": train_windows, "steps_per_epoch": steps_per_epoch, "steps": steps,
                             "warmup_steps": int(c2m.WARMUP_FRACTION * steps), "evaluation_points": c2m.EPOCHS},
    }


def tfidf_rf():
    config = json.loads((c1_results() / "config.json").read_text(encoding="utf-8"))["tfidf_rf"]
    low = json.loads((data_v2.RESULTS_DIR / "viabilidade-poucos-dados" / "config.json").read_text(encoding="utf-8"))
    return {
        "sources": [str(c1_results() / "config.json"), "models/cenario1/cenario1.py", "models/exploratorio/poucos_dados.py",
                    str(data_v2.RESULTS_DIR / "viabilidade-poucos-dados" / "config.json")],
        "document": config["document"],
        "contagem": {"vectorizer": config["tfidf"], "fit": "on the training windows of the run (full set or draw)"},
        "presenca": {"vectorizer": low["versions"]["presenca"] + f" (token_pattern {c1.TOKEN_PATTERN}, lowercase=False)",
                     "fit": "on the training windows of the run (full set or draw)"},
        "random_forest_fixed": config["random_forest_fixed"],
        "grid": config["grid"],
        "chosen_in_c4": config["chosen"]["controlada"],
        "deteccao": {"params": config["chosen"]["controlada"], "applies_to": "both versions, every fraction (as the feasibility study)",
                     "decision": f"anomalous when the forest probability of the anomalous class >= {c1.RF_THRESHOLD}"},
        "diagnostico": {
            "params": "chosen once per version on the full validation, then fixed for every fraction",
            "rule": "highest mean validation Macro-F1 over the seeds 42-46; ties: highest mean validation macro PR-AUC (one-vs-rest "
                    "average precision of the 4 classes), then the grid order",
            "grid": "the 8 configurations of the C4", "decision": "argmax of the forest probabilities"},
    }


def c1_results():
    return data_v2.RESULTS_DIR / "cenario1"


def expectations(counts):
    def majority(normal, windows):
        f1_normal = 2 * normal / (normal + windows)
        return {"accuracy": normal / windows, "macro_f1": f1_normal / 4}
    viab = json.loads((data_v2.RESULTS_DIR / "viabilidade" / "viabilidade.json").read_text(encoding="utf-8"))["ceilings"]["texto_sem_contagem/4_classes"]
    low = json.loads((data_v2.RESULTS_DIR / "viabilidade-poucos-dados" / "poucos_dados.json").read_text(encoding="utf-8"))["summary"]
    presenca_1 = next(r for r in low if r["version"] == "presenca" and r["fraction"] == 0.01)
    main, secondary = counts["test_principal"], counts["test_secundario"]
    return {
        "deteccao": {"teto_f1_teste": 0.931,
                     "motivo": "the 4 slow-only test windows have the same text as 73 normal test windows: no model can tell them apart; "
                               "F1 0.931 is the F1 of Scenario 1 and of the simple rule in the test (results/cenario1/metrics.json, "
                               "results/cenario2/pre_registro.json)"},
        "deteccao_poucos_dados": {"presenca_f1_validacao_1pct": presenca_1["f1_mean"], "draws": presenca_1["draws"],
                                  "note": "feasibility study (results/viabilidade-poucos-dados): little margin between fractions; its draws "
                                          "were by binary class, not these"},
        "diagnostico": {"teto_dentro_da_amostra": {"accuracy": viab["max_accuracy"], "macro_f1": viab["macro_f1"],
                                                    "source": "results/viabilidade/viabilidade.json ceilings texto_sem_contagem/4_classes "
                                                              "(train + validation pooled, majority class of each group of identical "
                                                              "texts: an in-sample upper limit, not an estimate)"},
                        "classe_mais_comum": {"test_principal": majority(main["by_cause"]["Normal"], main["windows"]),
                                              "test_secundario": majority(secondary["by_cause"]["Normal"], secondary["windows"]),
                                              "note": "always Normal, from the counts only; the value 0.234 in the request was not "
                                                      "reproduced and is not used"}},
    }


def statistics_plan():
    return {
        "principal": {"comparison": "LUK vs 2b", "task": "diagnostico", "set": "test_principal", "metric": "Macro-F1",
                      "statistic": "difference of the mean Macro-F1 of the 5 seeds (LUK - 2b)",
                      "test": f"approximate randomization paired by window: in each permutation the 5 seed predictions of a window "
                              f"swap model together (probability 1/2); {PERMUTATIONS} permutations; two-sided on the absolute "
                              f"difference; p = (count >= observed + 1) / ({PERMUTATIONS} + 1); alpha = {ALPHA}; seed {STAT_SEED}",
                      "secondary_test": "exact McNemar (two-sided binomial on the discordant windows) over the majority vote of the 5 "
                                        "seeds, correct/wrong per window; vote tie -> class with the highest mean probability, then "
                                        "the class order Normal, CPU, Rede, Combinado",
                      "confidence_interval": f"95% percentile bootstrap of the difference by run: the 10 test runs resampled with "
                                             f"replacement, their windows kept together; {BOOTSTRAP} resamples; seed {STAT_SEED + 1}"},
        "secundarias": {"comparisons": ["LUK vs BERT", "2b vs BERT", "best language model vs best TF-IDF"],
                        "best_choice": "on the full validation, by the mean validation main metric of the task; ties -> lowest mean "
                                       "validation loss for language models (then order BERT, 2b, LUK) and contagem before presenca "
                                       "for TF-IDF; never on the test",
                        "tests": "the same as the principal (randomization, McNemar, bootstrap CI)",
                        "holm_families": {"diagnostico_dados_completos": "LUK vs BERT, 2b vs BERT, best LM vs best TF-IDF (3)",
                                          "diagnostico_baixos_recursos": "LUK vs 2b, LUK vs BERT, 2b vs BERT, best LM vs best TF-IDF, "
                                                                         "for each of the 3 fractions (12)",
                                          "deteccao_dados_completos": "LUK vs 2b, LUK vs BERT, 2b vs BERT, best LM vs best TF-IDF (4), "
                                                                      "F1 of the anomalous class, test with every window",
                                          "deteccao_baixos_recursos": "the same 4 for each fraction (12)"},
                        "correction": "Holm inside each family"},
        "baixos_recursos": {"pairing": "unit = draw: the statistic is the difference of the mean over the 5 draws; each permutation "
                                       "swaps, inside each draw, the predictions of a window between the two models; McNemar over "
                                       "the majority vote of the 5 draws; bootstrap CI by run of the test, as above"},
        "wilcoxon": "not the main test: with 5 seeds the paired Wilcoxon cannot reach p < 0.05 (smallest two-sided p = 0.0625)",
    }


def document(derived_dir, output, written):
    cause_path, draws_path = output / causa.CAUSE_FILE, output / sorteios.DRAWS_FILE
    cause_rows = data_v2.read_csv(cause_path)
    counts = causa.counts_by_set(cause_rows)
    bert = bert_finetuning()
    budget = bert["full_data_budget"]
    draws = json.loads(draws_path.read_text(encoding="utf-8"))
    return {
        "created_local": written.isoformat(timespec="seconds"), "created_utc": written.astimezone(dt.timezone.utc).isoformat(timespec="seconds"),
        "written_before": "any training or evaluation of the extended evaluation (diagnosis and low resources)",
        "approved": "extension approved by the advisor on 2026-11-07",
        "tarefas": {
            "deteccao": {"role": "baseline", "classes": ["normal (0)", "anomalous (1)"], "label": "binary label of the C3",
                         "sets": {"train": "normal + visible (307)", "validation": "normal + visible (102)",
                                  "test": "every window of the C3 test set (106), with the cuts todas / so_visiveis / so_lentidao"}},
            "diagnostico": {"classes": list(causa.CLASSES), "label": "cause label of cause_labels.csv (C3 label + condition of the run)",
                            "sets": {"train": "normal + visible", "validation": "normal + visible",
                                     "test_principal": "normal + visible", "test_secundario": "every window of the test set, the 4 "
                                     "slow-only windows with the cause of their run"},
                            "counts": counts},
            "baixos_recursos": {"tasks": ["deteccao", "diagnostico"], "fractions": list(sorteios.FRACTIONS), "draws": sorteios.DRAWS,
                                "file": str(draws_path), "validation_and_test": "full in every fraction",
                                "draw_sizes": {str(f): sorteios.draw_size(f, 307) for f in sorteios.FRACTIONS},
                                "draw_class_counts": {str(f): sorteios.allocation(draws["population"]["by_cause"], sorteios.draw_size(f, 307))
                                                      for f in sorteios.FRACTIONS}},
        },
        "condicao_da_execucao": "control metadata: used only to build the answer key (cause label); never a model input",
        "modelos": {
            "classe_mais_comum": {"tasks": ["deteccao", "diagnostico"], "rule": "most frequent class of the training windows of the run "
                                  "(full set or draw); ties -> class order (Normal, CPU, Rede, Combinado; in detection normal before "
                                  "anomalous); constant score for PR-AUC and ROC-AUC"},
            "regra_simples": {"tasks": ["deteccao"], "rule": "anomalous with at least one line of an event of labels/failure_events.json; "
                              "no learning, the same predictions in every fraction", "source": str(c1_results() / "config.json")},
            "tfidf_rf_contagem": {"tasks": ["deteccao", "diagnostico"]},
            "tfidf_rf_presenca": {"tasks": ["deteccao", "diagnostico"]},
            "bert_base": {"tasks": ["deteccao", "diagnostico"], "scenario": "Cenario 2", "encoder": bert["encoders"]["bert"]},
            "bert_mlm_2b": {"tasks": ["deteccao", "diagnostico"], "scenario": "Cenario 2b", "checkpoint": checkpoint("2b", CHECKPOINTS["2b"])},
            "bert_luk": {"tasks": ["deteccao", "diagnostico"], "scenario": "Cenario 3", "checkpoint": checkpoint("luk", CHECKPOINTS["luk"])},
        },
        "entrada": {"view": "controlada", "language_models": "models/common/window_text.py: distinct templates, no counts, fixed order "
                    "by template id, joined by [SEP], no truncation (max 512 tokens)",
                    "tfidf": "one token per template line of the window (counts kept in contagem, 0/1 in presenca)",
                    "operacional": "reference only, not used in the diagnosis",
                    "window_text_py_sha256": sha(REPO / "models/common/window_text.py")},
        "hiperparametros": {"fine_tuning_bert_2b_luk": bert, "tfidf_rf": tfidf_rf()},
        "selecao": {"where": "always on the full validation, never on the test",
                    "metric": {"deteccao": "F1 of the anomalous class", "diagnostico": "Macro-F1"},
                    "test_use": "once per model, seed and draw, after every choice is fixed"},
        "sementes": {"dados_completos": list(c1.SEEDS), "applies_to": "every trainable model (TF-IDF + RF as random_state; BERT, 2b, LUK "
                     "as set_seed and the batch-order generator)",
                     "baixos_recursos": "one run per draw; the training seed is the seed of the draw"},
        "orcamento_baixos_recursos": {
            "rule": "the same number of optimization steps as the full training of the same model, with the validation evaluated at "
                    "the same points (every block of steps equal to one epoch of the full training)",
            "bert_2b_luk": {"steps": budget["steps"], "evaluation_every_steps": budget["steps_per_epoch"],
                            "evaluation_points": budget["evaluation_points"], "warmup_steps": budget["warmup_steps"],
                            "schedule": "linear warm-up and linear decay over these steps",
                            "batches": "successive passes over the draw, each with a new torch.randperm order from the generator of "
                                       "the seed; batches of up to 8 windows that do not cross passes (the last batch of a pass may be "
                                       "smaller, as in the full training); the validation is evaluated after steps 39, 78, ..., 780 and "
                                       "the best point is chosen as the best epoch",
                            "same_for": "the three models and the two tasks"},
            "tfidf_rf": "no optimization steps: fitted once on the draw",
            "classe_mais_comum_e_regra": "no training steps"},
        "metricas": {"deteccao": {"scalars": ["precision", "recall", "f1", "pr_auc", "roc_auc"],
                                  "cuts": {"todas": "normal + every anomalous window", "so_visiveis": "normal + visible",
                                           "so_lentidao": "recall of the slow-only windows"}, "source": "models/common/metrics.py, "
                                  "models/cenario1/cenario1_v2.py (cut_metrics)"},
                     "diagnostico": {"scalars": ["accuracy", "weighted_f1", "macro_f1 (principal)", "f1 by class"],
                                     "confusion_matrix": "rows = true class, columns = predicted, order Normal, CPU, Rede, Combinado",
                                     "sets": ["test_principal", "test_secundario"]},
                     "aggregation": "mean and standard deviation over the 5 seeds (full data) or the 5 draws (low resources)"},
        "testes_estatisticos": statistics_plan(),
        "expectativas": expectations(counts),
        "limitacoes": [
            "in low resources the epoch (evaluation point) is chosen on the full validation, more labelled data than the draw itself",
            "small classes in the validation (CPU 7, Rede 6, Combinado 14 windows): Macro-F1 selection is noisy",
            "the windows with ReadTimeout in the 3 routes form one text group that mixes the three causes",
            "the test has only 10 runs: windows of a run are not independent, hence the bootstrap by run",
            "the 1% draws have 1 window of each class: in detection that is 1 normal and 3 anomalous windows, the inverse of the "
            "proportion of the training set",
            "the 5% and 10% draws have only 1-2 windows of CPU and of Rede",
        ],
        "arquivos": {"cause_labels_csv": {"path": str(cause_path), "sha256": sha(cause_path)},
                     "low_resource_draws_json": {"path": str(draws_path), "sha256": sha(draws_path)}},
        "inputs_sha256": {name: sha(derived_dir / name) for name in INPUTS},
        "code_sha256": {name: sha(REPO / name) for name in CODE},
        "test_reads_in_e0": "ids, C3 status and label, anomaly type and condition of labels.csv / labels/sets/test.csv only; no test "
                            "window text was opened",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived", type=Path, default=data_v2.DERIVED_DIR)
    parser.add_argument("--output", type=Path, default=causa.OUTPUT)
    args = parser.parse_args(argv)
    written = dt.datetime.now().astimezone()
    content = data.json_bytes(document(args.derived, args.output, written))
    data.write_new(args.output / PRE_FILE, content)
    digest = data.sha256_bytes(content)
    data.write_new(args.output / HASH_FILE, f"{digest}  {PRE_FILE}\n".encode("utf-8"))
    print(json.dumps({"preregistro_sha256": digest, "created_local": written.isoformat(timespec="seconds")}))
    return digest


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

"""Build the result table and learning materials from actual saved measurements."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
metrics=json.loads((ROOT/'artifacts/metrics.json').read_text())
meta=json.loads((ROOT/'artifacts/metadata.json').read_text())
def write(name,text):
    (ROOT/name).write_text(text.strip()+'\n',encoding='utf-8')
table='| Model | Test macro-F1 | Positive precision | Positive recall | ROC-AUC | Average precision | Threshold |\n|---|---:|---:|---:|---:|---:|---:|\n'
for name,result in metrics.items():
    m=result['test']; c=m['classification_report']['1']
    table+=f"| {name} | {m['macro_f1']:.3f} | {c['precision']:.3f} | {c['recall']:.3f} | {m['roc_auc']:.3f} | {m['average_precision']:.3f} | {m['threshold']:.2f} |\n"
write('README.md', '''# Explainable liver label classification

An educational, local application that predicts the **ILPD dataset label**, compares three models, and explains the selected pipeline with SHAP and LIME. It does not diagnose liver disease or cirrhosis. Built with AI assistance; the accompanying exercises and interview guide are intended to help the owner understand and maintain it.

![Running application](docs/demo.png)

## Quick start

Python 3.12 was tested on Windows with CPU only. Python 3.11–3.12 is recommended. From this directory:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe download_data.py
.venv/Scripts/python.exe train.py
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
.venv/Scripts/python.exe app.py
```

Open http://127.0.0.1:5051, load the fictional example, and click **Run model and explain**. After installation/training, `./start.ps1` is the Windows startup shortcut. On Linux/macOS use `.venv/bin/python` in the commands above. No database or cloud account is required. Runtime does not contact external inference services. Downloading the data and installing libraries require internet access.

The local workspace already contains trained artifacts. They are excluded from Git so cloning requires training. Only load joblib artifacts you created or trust: pickle-based deserialization can execute code. Full installed versions are recorded in `artifacts/environment-lock.txt`; `requirements.txt` pins direct dependencies.

## Data source and target

[UCI ILPD](https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset), DOI [10.24432/C5D02C](https://doi.org/10.24432/C5D02C), attributed to B. Ramana and N. Venkateswarlu. UCI states **CC BY 4.0**. The download script reads the official archive and saves its SHA-256, CSV hash and original member name in `artifacts/data_provenance.json`; later downloads reject a changed CSV. The full source data is not committed, although the license permits attributed reuse. Aggregate evaluation and source row indices are included.

Selector 1 is encoded as positive (liver patient), Selector 2 as negative. These labels do not establish a particular disease, severity, future outcome, or cirrhosis. Collection is regional, from northeast Andhra Pradesh, India; detailed sampling and diagnostic criteria are not supplied in the repository metadata. The source page contains inconsistent counts and missingness claims: the downloaded file actually has **583 rows, 416 positive labels, 167 negative labels, 13 exact duplicates and four missing A/G ratios**. After feature-profile deduplication there are 570 records. No conflicting labels for identical profiles were found.

| API field | Meaning in UCI |
|---|---|
| age | Age in years; values above 89 top-coded as 90 |
| gender | Female/Male categories supplied by the dataset |
| total_bilirubin / direct_bilirubin | Total/direct bilirubin |
| alkaline_phosphatase | Alkaline phosphatase |
| alt / ast | Alanine/aspartate aminotransferase (source uses SGPT/SGOT) |
| total_protein / albumin | Total proteins/albumin |
| albumin_globulin_ratio | Albumin-to-globulin ratio; null is supported |

**Biochemical units are not specified by UCI.** The software does not infer them. Inputs are dataset-shaped examples, not a way to interpret personal lab results. Three source records have direct bilirubin above total bilirubin; they are flagged in the audit and retained rather than silently corrected. The API rejects that inconsistency for new input. This is a documented training-versus-entry discrepancy and data-quality limitation.

## Architecture and experiment

`official CSV → profile deduplication → stratified train/validation/test → fitted pipeline → saved artifacts → Flask validation → score → pipeline-level SHAP/LIME → browser`

The fixed seed is 42. Splits contain 342/114/114 rows, with 244/81/81 positive labels. Exact feature profiles cannot cross splits; no patient IDs are available to rule out related but nonidentical records. The label is excluded from all input features. Training-only numeric summaries and correlations are saved in `data_audit.json`. Highly related biochemical measurements are retained and make attribution less straightforward.

Each scikit-learn pipeline median-imputes numeric values, standardizes them, and encodes gender as one binary column. All fitted transforms see training data only. The candidates are a prior dummy, balanced logistic regression (C=1), and a balanced 250-tree Random Forest (depth 6, minimum leaf 5). Class weights address imbalance without oversampling or changing test prevalence. Scaling is unnecessary for the forest but keeps a common feature contract. CPU-only, one forest worker.

For learned models, a 0.10–0.90 threshold grid selects validation macro-F1, breaking ties toward 0.5. The dummy stays at 0.5. Model selection includes the dummy and uses validation macro-F1, then average precision. No retraining or threshold adjustment occurs after looking at test scores. Hyperparameters were fixed beforehand; the small validation set can still make selection unstable.

## Actual evaluation

''' + table + f'''
Validation selected **{meta['selected_model']}**, threshold **{meta['threshold']:.2f}**. All scores above use the same 114 held-out records. Precision/recall/F1 for both classes and confusion matrices are in `artifacts/metrics.json`; ROC and PR curves are in `artifacts/test_curves.png`. Average precision is reported separately from trapezoidal PR-AUC. Positive prevalence is 81/114 = 0.711, so a seemingly high AP must be compared with that prior baseline. The selected logistic model's confusion matrix is [[15,18],[21,60]] (rows true 0/1, columns predicted 0/1). Its low negative-class recall is a material limitation, not hidden by accuracy.

The score is **uncalibrated**, especially with class weighting. Brier scores are descriptive, not evidence of clinical calibration. Results at threshold 0.5 are also saved. This single small split has high uncertainty; it is not external validation. Do not choose a new model based on this test table and then claim the same test set remained untouched.

## Explanations

SHAP uses a permutation explainer around the entire fitted pipeline, in the original ten-feature space. Forty seed-selected training rows form its background; 231 evaluations approximate the feature allocations. The code verifies `base + contributions = actual score` within 1e-6. This numerical check does not prove explanation correctness or causality. Correlation and background choice matter.

LIME uses training-derived discretization and 2,000 seeded perturbations. Gender remains categorical, and its prediction callback runs the same pipeline. Its prediction on the original input must equal the actual score. The UI reports local weighted R², surrogate score and absolute score error. Contributions are to the surrogate, not exact SHAP equivalents. Numeric perturbations can create implausible combinations. Missing A/G ratio is filled with the same training median and explicitly flagged before explaining.

## API

- `GET /api/health`: 200 when artifacts load; 503 with recovery instructions otherwise.
- `POST /api/predict`: JSON with exactly ten fields. Returns score, label, threshold, warnings, SHAP and LIME. Unknown keys, invalid category, nonfinite numbers, inconsistent measurements and invalid ages return 400. Wrong content type returns 415, oversized bodies 413, unsupported methods 405. Inference failures return a generic 500 without echoing values.

```powershell
Invoke-RestMethod http://127.0.0.1:5051/api/health
Invoke-RestMethod http://127.0.0.1:5051/api/predict -Method Post -ContentType application/json -Body (Get-Content examples/fictional_input.json -Raw)
```

```bash
curl -H 'Content-Type: application/json' --data-binary @examples/fictional_input.json http://127.0.0.1:5051/api/predict
```

The example is fictional, not a copied patient's record. See `artifacts/example_response.json` for a verified output. No submissions are persisted or logged. Cache-Control is no-store, scripts are local, and the server binds to loopback through Waitress. It has no authentication, clinical safeguards, or public-deployment hardening. The single worker avoids concurrent explainer RNG interactions; this is not a throughput-oriented service.

## Evidence and learning

- [Model card](MODEL_CARD.md)
- [Five-minute demonstration](DEMO.md)
- [Twenty interview questions and exercises](INTERVIEW_GUIDE.md)
- [Resume claims and evidence](EVIDENCE.md)
- [Status and commands run](STATUS.md)

Tests cover validation, pipeline/raw-coordinate equivalence, training-only imputation, duplicate-free splits, explanation reconstruction, HTTP failure paths and independent rescoring. Third-party deprecation warnings are reported in the test log. No production users, clinical impact, deployment, or independently authored achievement is claimed.

Primary implementation references: [scikit-learn evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html), [SHAP API](https://shap.readthedocs.io/en/latest/api.html), [LIME API](https://lime-ml.readthedocs.io/en/latest/lime.html), [Flask](https://flask.palletsprojects.com/en/stable/).
''')
write('MODEL_CARD.md', '''# Model card

Purpose: educational ILPD label classification. Owner: student portfolio project, developed with AI assistance. No clinical approval or independent clinical evaluation.

Inputs and label: ten UCI tabular fields; Selector=1 positive and Selector=2 negative. Age is top-coded; binary gender coverage is a dataset restriction. Biochemical units and detailed diagnostic criteria are not supplied. Never substitute cirrhosis diagnosis for this label.

Data: UCI ILPD, CC BY 4.0, credited in README and provenance. 583 raw records, 570 distinct profiles. Train 342, validation 114, test 114. No user submissions enter training. No protected patient records are collected. Exact duplicates are removed before splitting, but absent identifiers prevent a stronger independence claim.

Model: validation-selected balanced logistic regression with median imputation, scaling and binary gender encoding; threshold 0.39. Dummy and Random Forest comparisons are retained. Model and threshold are fixed before test evaluation.

Evaluation: held-out ROC-AUC 0.722; macro-F1 0.595; positive precision 0.769, recall 0.741, F1 0.755; negative recall 0.455. Confusion matrix [[15,18],[21,60]]. See metrics.json for full precision and all baselines. No external validation, confidence intervals or calibration study was performed.

Risks: false positives/negatives, regional and demographic bias, unknown units, inconsistent records, selection variance, correlated inputs, and potentially misleading explanations. Class weighting changes score behavior; scores are not calibrated disease probabilities. SHAP/LIME explain model behavior, not biological causation. The protected-attribute input is retained for fidelity to the source and should be examined through subgroup evaluation before any broader research use. Subgroup fairness has not been established.

Prohibited uses: diagnosis, triage, treatment, screening, individual risk counseling, or other consequential decisions. Use only fictional dataset-shaped inputs for the demo.

Reproduction: README commands; seed 42; dataset hash and split manifest saved. Only trusted locally trained joblib files should be loaded. Evaluate any future data/model change using a fresh protocol rather than reusing this test set for tuning.
''')
write('DEMO.md', '''# Five-minute demonstration

0:00–0:40 — Open localhost:5051. Explain that the label comes from ILPD, not cirrhosis and not clinical risk. Point out missing source-unit metadata.

0:40–1:30 — Click Load fictional example, then Run model and explain. Show the score, validation-selected threshold, and dataset-label wording. The example was invented for software demonstration, not a fabricated dataset row.

1:30–2:30 — Explain SHAP background plus contributions. Point out the reconstruction error. Compare LIME's local weighted R² and score mismatch; do not equate the two explanations or treat them as causes.

2:30–3:15 — Set direct bilirubin larger than total and submit. Show the clear error. Restore the sample. Blank the A/G ratio and show the imputation warning. Do not enter anyone's actual health information.

3:15–4:15 — Open artifacts/metrics.json and confusion_matrices.png. Show the dummy baseline and 18 false positives / 21 false negatives for selected logistic regression. Explain why negative-class recall matters despite high positive prevalence and AP.

4:15–5:00 — Show test_results.xml, split_manifest.json and the pipeline code. State what you understand, what AI helped implement, and one exercise you completed yourself. End with the unknown units, small regional sample and absence of clinical validation.
''')
questions=[
('What exactly is predicted?','The binary UCI ILPD Selector label: 1 becomes positive and 2 negative. It is not a cirrhosis diagnosis.'),
('Why scikit-learn rather than a neural network?','There are only 570 unique profiles. Simple tabular baselines are fast on CPU, inspectable and appropriate for this limited sample.'),
('How is leakage reduced?','Remove identical feature profiles before splitting, exclude Selector from X, fit transforms only on train, and select on validation. Missing patient IDs leave related-record leakage unresolved.'),
('Why 60/20/20?','It reserves separate validation and test data while keeping a usable training subset. At this size all three subsets are small, so results are uncertain.'),
('Why a dummy model?','A prior classifier exposes the strong positive prevalence and prevents an impressive-looking number from hiding a weak comparison.'),
('Why class weighting?','Balanced weights increase the influence of the minority class in the training objective without changing held-out prevalence. They can distort probability interpretation.'),
('How is missingness handled?','Four source A/G ratios are missing. A training-fitted median imputer handles numeric gaps; the API permits null only for that field.'),
('What is standardized?','Nine numeric features are centered and scaled using training statistics. Gender is a single binary column with a fixed category order.'),
('Which model won?','Logistic regression won validation macro-F1 at threshold 0.39. The test set was used only after freezing this decision.'),
('Why not optimize accuracy?','A majority-positive classifier already gets roughly 71% accuracy. Macro-F1 gives both labels equal importance.'),
('What does ROC-AUC measure?','Ranking positives above negatives across thresholds. It does not describe a single operating threshold or prove calibrated probabilities.'),
('How does AP differ from PR-AUC?','Average precision weights precision by recall increments; the separately saved PR-AUC integrates by trapezoids. They are not interchangeable, and AP must be compared with prevalence.'),
('Are the scores probabilities?','predict_proba supplies a bounded model score, but class weighting and lack of calibration mean it must not be described as a patient disease probability.'),
('How is SHAP mapped to features?','A raw-coordinate callback decodes the categorical value and runs the complete pipeline. This preserves original feature names without mislabeling scaled or one-hot coordinates.'),
('What does the SHAP check prove?','The baseline and contributions reconstruct this prediction within tolerance. It does not prove causality, fairness, or robust attribution under correlated features.'),
('What is LIME doing?','It fits a weighted local linear surrogate to 2000 perturbed examples, with training-derived bins and categorical gender. Its R² and pointwise error reveal imperfect fidelity.'),
('Can explanations mislead?','Yes. Correlated biomarkers and unrealistic perturbations can change attributions; background choice matters, and SHAP/LIME can disagree.'),
('What does the API validate?','Exact keys, finite numeric JSON types, age range/integrality, category, direct≤total bilirubin, albumin≤total protein and body size. These are software checks, not clinical reference ranges.'),
('How is privacy handled?','The local API has no input persistence, no request-body logging and no external inference calls. No-store headers discourage browser caching; this does not make it a production health service.'),
('What would you improve next?','Prespecify repeated validation, assess calibration using training-only procedures, compare subgroup error estimates with uncertainty, and obtain a separate licensed dataset with verified units. Avoid tuning repeatedly on the held-out test.')]
write('INTERVIEW_GUIDE.md', '''# Interview and maintenance guide

Simple explanation: this project takes ten dataset-shaped numbers/categories, estimates a historical binary label, and shows why the fitted model produced that score. Its value is a transparent and tested ML workflow, not medical accuracy.

Data flow: download official CSV → checksum → audit/deduplicate → seeded splits → train-only transforms → dummy/logistic/forest → validation model/threshold choice → one held-out evaluation → local Flask service → SHAP/LIME → browser.

Technology choices: pandas for audit, scikit-learn for pipelines and metrics, joblib for trusted local model persistence, SHAP/LIME for contrasting explanations, Flask for a small API, Waitress for a Windows-compatible local server, pytest for meaningful regression checks. No database is needed because inputs are intentionally not stored.

'''+'\n\n'.join(f'## {i}. {q}\n\n{a}' for i,(q,a) in enumerate(questions,1))+'''

## Five modification exercises

1. Add a test rejecting a boolean for each numeric field; explain why Python bool otherwise behaves like an integer.
2. Add a feature-units column to the interface, explicitly displaying unknown where source metadata is missing.
3. Build training-only five-fold cross-validation in a new experiment folder; compare its variance without selecting on the existing test set.
4. Add calibration curves using training cross-validation and validation selection; keep a genuinely new test protocol for claims.
5. Add a user-controlled explanation visibility toggle without changing inference; test that hidden explanations are not silently computed or logged.

Keep a learning log of changes you personally made and explain their tests. Do not imply independent authorship of AI-assisted code. Failure cases worth rehearsing: missing artifacts (503), wrong JSON (400/415), unseen categories, out-of-range features, model disagreement, poor local LIME fit and minority-class errors.
''')
write('EVIDENCE.md', '''# Defensible resume bullets and evidence

Use these only after reviewing the implementation and completing the demo. Describe the work as an AI-assisted student portfolio project when asked; no employment, users or deployment are implied.

- Built and tested a local Flask application for UCI ILPD dataset-label classification, with strict input validation and SHAP/LIME explanations of the fitted preprocessing-and-model pipeline.
- Compared a dummy baseline, balanced logistic regression and Random Forest on duplicate-separated splits; documented held-out ROC-AUC 0.722 and macro-F1 0.595 for the validation-selected logistic model.
- Added reproducible data checksums, training-only preprocessing, saved evaluation outputs and 17 passing tests covering inference, explanation consistency and API failure cases.

| Claim | Evidence | Reproduce |
|---|---|---|
| Local working Flask flow | app.py, static/, docs/demo.png, artifacts/browser_verification.json | app.py then DEMO.md |
| Validation and errors | liver/schema.py, tests/test_project.py | pytest tests |
| SHAP/LIME match pipeline | liver/service.py, artifacts/example_response.json | explanation test and fictional API call |
| Three actual models | train.py, artifacts/metrics.json, *_test_predictions.csv | train.py |
| Disjoint profiles and training-only transforms | split_manifest.json, data_audit.json, test_training_only_and_disjoint_profiles | pytest tests |
| Reported metrics | metrics.json, test_saved_metrics_match_predictions | rescore saved CSVs |
| Passing tests | artifacts/test_results.xml, artifacts/tests.log | pytest tests |
| Source provenance | download_data.py, data_provenance.json | download_data.py checks unchanged hash |

Not claimed: clinical usefulness, calibrated risk, causality, demographic fairness, production deployment, real users, independent authorship, or improvements over external publications.
''')
print('Documentation written from measured artifacts.')

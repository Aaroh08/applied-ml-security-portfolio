# Defensible resume bullets and evidence

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

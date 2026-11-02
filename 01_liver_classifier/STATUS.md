# Project 1 status

## Completed and tested

- Official licensed data download with checksums and attribution; audit, duplicate removal and fixed train/validation/test split.
- Actual CPU training and evaluation of dummy, logistic regression and Random Forest. Validation chose logistic at threshold 0.39. Test ROC-AUC 0.7224, macro-F1 0.5947; full per-class results saved.
- SHAP and LIME around the real fitted pipeline, including coordinate round-trip and prediction checks. Fictional example score 0.347; SHAP reconstruction error about 1.7e-16. LIME weighted local R² only 0.078 and score error 0.115: the approximation is weak, clearly displayed rather than hidden.
- Flask/Waitress service, loopback binding, health/predict routes, validated JSON, explicit failures, no input persistence and a working browser UI.
- 17 automated tests passed; dependency consistency check passed. Browser prediction and invalid-measurement feedback verified. Actual full-page screenshot in docs/demo.png.
- README, model card, API example, five-minute demo, 20-question interview guide, five modification exercises and evidence-linked resume bullets.

## Commands actually run

```text
python -m venv .venv
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
python download_data.py
python train.py
python -m pytest tests -q -p no:cacheprovider --basetemp=tmp/pytest-verified --junitxml=artifacts/test_results.xml
python write_docs.py
python -m pip freeze
python -m pip check
python app.py
```

Commands used the project virtual environment. Windows sandbox denied initial temporary directories and parallel worker pipes. A project-local TEMP/TMP, one forest worker and a permission-approved test run resolved those environmental failures. Final test evidence is artifacts/tests.log and test_results.xml. Fourteen third-party matplotlib/pyparsing deprecation warnings remained; no failing tests. The browser ran the fictional example and displayed both explanations; then rejected direct bilirubin greater than total.

## Limits and next learning steps

No clinical validation, calibration, external dataset, subgroup fairness study or confidence intervals. Source units are incomplete; three inconsistent bilirubin rows are flagged but retained. This is a local educational application, not a diagnostic service. Explanations are associations in a fitted model; LIME fidelity is weak for the example. Publication does not imply public deployment.

Read the data audit and trace one input through validate → preprocess → predict_proba → explanation. Complete the five exercises, keeping a personal learning log. Only claim understanding you can demonstrate in an interview. Project 2 may proceed after this verified local milestone; repository publication is handled as part of the portfolio delivery.

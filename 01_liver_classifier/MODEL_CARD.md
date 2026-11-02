# Model card

Purpose: educational ILPD label classification. Owner: student portfolio project, developed with AI assistance. No clinical approval or independent clinical evaluation.

Inputs and label: ten UCI tabular fields; Selector=1 positive and Selector=2 negative. Age is top-coded; binary gender coverage is a dataset restriction. Biochemical units and detailed diagnostic criteria are not supplied. Never substitute cirrhosis diagnosis for this label.

Data: UCI ILPD, CC BY 4.0, credited in README and provenance. 583 raw records, 570 distinct profiles. Train 342, validation 114, test 114. No user submissions enter training. No protected patient records are collected. Exact duplicates are removed before splitting, but absent identifiers prevent a stronger independence claim.

Model: validation-selected balanced logistic regression with median imputation, scaling and binary gender encoding; threshold 0.39. Dummy and Random Forest comparisons are retained. Model and threshold are fixed before test evaluation.

Evaluation: held-out ROC-AUC 0.722; macro-F1 0.595; positive precision 0.769, recall 0.741, F1 0.755; negative recall 0.455. Confusion matrix [[15,18],[21,60]]. See metrics.json for full precision and all baselines. No external validation, confidence intervals or calibration study was performed.

Risks: false positives/negatives, regional and demographic bias, unknown units, inconsistent records, selection variance, correlated inputs, and potentially misleading explanations. Class weighting changes score behavior; scores are not calibrated disease probabilities. SHAP/LIME explain model behavior, not biological causation. The protected-attribute input is retained for fidelity to the source and should be examined through subgroup evaluation before any broader research use. Subgroup fairness has not been established.

Prohibited uses: diagnosis, triage, treatment, screening, individual risk counseling, or other consequential decisions. Use only fictional dataset-shaped inputs for the demo.

Reproduction: README commands; seed 42; dataset hash and split manifest saved. Only trusted locally trained joblib files should be loaded. Evaluate any future data/model change using a fresh protocol rather than reusing this test set for tuning.

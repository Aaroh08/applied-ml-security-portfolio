# Interview and maintenance guide

Simple explanation: this project takes ten dataset-shaped numbers/categories, estimates a historical binary label, and shows why the fitted model produced that score. Its value is a transparent and tested ML workflow, not medical accuracy.

Data flow: download official CSV → checksum → audit/deduplicate → seeded splits → train-only transforms → dummy/logistic/forest → validation model/threshold choice → one held-out evaluation → local Flask service → SHAP/LIME → browser.

Technology choices: pandas for audit, scikit-learn for pipelines and metrics, joblib for trusted local model persistence, SHAP/LIME for contrasting explanations, Flask for a small API, Waitress for a Windows-compatible local server, pytest for meaningful regression checks. No database is needed because inputs are intentionally not stored.

## 1. What exactly is predicted?

The binary UCI ILPD Selector label: 1 becomes positive and 2 negative. It is not a cirrhosis diagnosis.

## 2. Why scikit-learn rather than a neural network?

There are only 570 unique profiles. Simple tabular baselines are fast on CPU, inspectable and appropriate for this limited sample.

## 3. How is leakage reduced?

Remove identical feature profiles before splitting, exclude Selector from X, fit transforms only on train, and select on validation. Missing patient IDs leave related-record leakage unresolved.

## 4. Why 60/20/20?

It reserves separate validation and test data while keeping a usable training subset. At this size all three subsets are small, so results are uncertain.

## 5. Why a dummy model?

A prior classifier exposes the strong positive prevalence and prevents an impressive-looking number from hiding a weak comparison.

## 6. Why class weighting?

Balanced weights increase the influence of the minority class in the training objective without changing held-out prevalence. They can distort probability interpretation.

## 7. How is missingness handled?

Four source A/G ratios are missing. A training-fitted median imputer handles numeric gaps; the API permits null only for that field.

## 8. What is standardized?

Nine numeric features are centered and scaled using training statistics. Gender is a single binary column with a fixed category order.

## 9. Which model won?

Logistic regression won validation macro-F1 at threshold 0.39. The test set was used only after freezing this decision.

## 10. Why not optimize accuracy?

A majority-positive classifier already gets roughly 71% accuracy. Macro-F1 gives both labels equal importance.

## 11. What does ROC-AUC measure?

Ranking positives above negatives across thresholds. It does not describe a single operating threshold or prove calibrated probabilities.

## 12. How does AP differ from PR-AUC?

Average precision weights precision by recall increments; the separately saved PR-AUC integrates by trapezoids. They are not interchangeable, and AP must be compared with prevalence.

## 13. Are the scores probabilities?

predict_proba supplies a bounded model score, but class weighting and lack of calibration mean it must not be described as a patient disease probability.

## 14. How is SHAP mapped to features?

A raw-coordinate callback decodes the categorical value and runs the complete pipeline. This preserves original feature names without mislabeling scaled or one-hot coordinates.

## 15. What does the SHAP check prove?

The baseline and contributions reconstruct this prediction within tolerance. It does not prove causality, fairness, or robust attribution under correlated features.

## 16. What is LIME doing?

It fits a weighted local linear surrogate to 2000 perturbed examples, with training-derived bins and categorical gender. Its R² and pointwise error reveal imperfect fidelity.

## 17. Can explanations mislead?

Yes. Correlated biomarkers and unrealistic perturbations can change attributions; background choice matters, and SHAP/LIME can disagree.

## 18. What does the API validate?

Exact keys, finite numeric JSON types, age range/integrality, category, direct≤total bilirubin, albumin≤total protein and body size. These are software checks, not clinical reference ranges.

## 19. How is privacy handled?

The local API has no input persistence, no request-body logging and no external inference calls. No-store headers discourage browser caching; this does not make it a production health service.

## 20. What would you improve next?

Prespecify repeated validation, assess calibration using training-only procedures, compare subgroup error estimates with uncertainty, and obtain a separate licensed dataset with verified units. Avoid tuning repeatedly on the held-out test.

## Five modification exercises

1. Add a test rejecting a boolean for each numeric field; explain why Python bool otherwise behaves like an integer.
2. Add a feature-units column to the interface, explicitly displaying unknown where source metadata is missing.
3. Build training-only five-fold cross-validation in a new experiment folder; compare its variance without selecting on the existing test set.
4. Add calibration curves using training cross-validation and validation selection; keep a genuinely new test protocol for claims.
5. Add a user-controlled explanation visibility toggle without changing inference; test that hidden explanations are not silently computed or logged.

Keep a learning log of changes you personally made and explain their tests. Do not imply independent authorship of AI-assisted code. Failure cases worth rehearsing: missing artifacts (503), wrong JSON (400/415), unseen categories, out-of-range features, model disagreement, poor local LIME fit and minority-class errors.

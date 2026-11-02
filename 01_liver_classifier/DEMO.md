# Five-minute demonstration

0:00–0:40 — Open localhost:5051. Explain that the label comes from ILPD, not cirrhosis and not clinical risk. Point out missing source-unit metadata.

0:40–1:30 — Click Load fictional example, then Run model and explain. Show the score, validation-selected threshold, and dataset-label wording. The example was invented for software demonstration, not a fabricated dataset row.

1:30–2:30 — Explain SHAP background plus contributions. Point out the reconstruction error. Compare LIME's local weighted R² and score mismatch; do not equate the two explanations or treat them as causes.

2:30–3:15 — Set direct bilirubin larger than total and submit. Show the clear error. Restore the sample. Blank the A/G ratio and show the imputation warning. Do not enter anyone's actual health information.

3:15–4:15 — Open artifacts/metrics.json and confusion_matrices.png. Show the dummy baseline and 18 false positives / 21 false negatives for selected logistic regression. Explain why negative-class recall matters despite high positive prevalence and AP.

4:15–5:00 — Show test_results.xml, split_manifest.json and the pipeline code. State what you understand, what AI helped implement, and one exercise you completed yourself. End with the unknown units, small regional sample and absence of clinical validation.

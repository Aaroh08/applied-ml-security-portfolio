# Resume bullets and evidence

- Built a local text-only research demo comparing TF-IDF and frozen DistilBERT embeddings with logistic regression, without storing submitted text or inventing behavioral features.
- Evaluated both representations and a dummy baseline on a shared 600-text holdout from a seeded 3000-text subset; recorded macro-F1 0.900 for the validation-selected model.
- Audited duplicate/label leakage, documented weak-label limitations, and added a keyword-masking diagnostic plus API and embedding-consistency tests.

| Claim | Evidence |
|---|---|
| Actual comparison | train.py, metrics.json, *_test_predictions.csv |
| Real DistilBERT embeddings | signals.py, embedding_metadata.json, provenance.json pinned revision |
| Same splits and no exact overlap | split_manifest.json, data_audit.json, split tests |
| Keyword dependence diagnostic | keyword_mask_diagnostic.json, train.py masking section |
| Working API and privacy | app.py, tests/test_project.py, example_response.json |
| Browser demo | docs/demo.png, browser_verification.json |
| Verified tests | test_results.xml, tests.log |

All artifact paths are under artifacts/. Reproduce with the README commands. No diagnosis, real-user impact, clinical accuracy, multimodal architecture, independent authorship or public deployment is claimed. Understand the code and acknowledge AI assistance before using these bullets.

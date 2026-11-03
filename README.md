# Applied ML and defensive security portfolio

Three local educational projects built with AI assistance, with reproducible experiments and evidence instead of unsupported impact claims.

| Project | Status | Purpose |
|---|---|---|
| [Explainable liver label classification](01_liver_classifier/) | Implemented, evaluated and tested | Training-only tabular pipelines, SHAP/LIME, Flask |
| [Text signal classification](02_text_signal_classifier/) | Implemented, evaluated and tested | Honest text-only TF-IDF versus frozen DistilBERT comparison |
| [Local WAF prototype](03_local_waf/) | Implemented, evaluated and tested | Local defensive request inspection, Random Forest and real MySQL events |

Each project has an isolated directory and runtime. Follow its README to install, train and start locally. No paid infrastructure or public deployment is used. Model weights, raw datasets and credentials are excluded. The repository is private; uploading code does not make the applications publicly accessible.

Do not use the two health-related models for diagnosis or decisions about people. The security project is confined to its bundled localhost target. Resume claims must be backed by the linked experiment/test artifacts, and the student should complete the interview exercises before claiming proficiency.

## Verified delivery · 3 November 2026

| Project | Automated tests | Measured result | Local address |
|---|---:|---|---|
| Liver dataset labels | 17 passed | Test ROC-AUC 0.7224; macro-F1 0.5947 | localhost:5051 |
| Text research tags | 16 passed | TF-IDF macro-F1 0.8996; frozen DistilBERT 0.8322 | localhost:5052 |
| WAF lab | 10 passed | Combined synthetic macro-F1 0.8286; 60 benign false positives, 0 false negatives | localhost:5053 |

These are separate evaluations, not comparable benchmarks. The text scores reflect weak labels and lexical shortcuts; masking explicit terms reduces performance. The WAF figures only describe authored synthetic examples. See each model card and saved artifacts for limitations.

All three folders include source, requirements, a startup script, actual evaluation output, tests, a real application screenshot, a five-minute demo, 20 interview questions with answers, five exercises and three evidence-backed resume statements. Environments and models are prepared locally; a new clone must follow each README's download and training steps. No Hugging Face gated access, GPU or paid service is needed.

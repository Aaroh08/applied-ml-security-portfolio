# Model card

Educational text classification of six creator-supplied, lexicon-derived tags. No human diagnostic inference. AI-assisted student portfolio implementation.

Dataset: Nusrat's Multi-Class Depression Detection Dataset, Zenodo v1.0, CC BY 4.0, DOI 10.5281/zenodo.14233292. Only Tweets/Labels are present. 3,000 deduplicated eligible texts are selected with fixed seed before splitting into 1,800/600/600. User-disjoint evaluation is impossible without IDs; no behavior is inferred.

Models: training-only TF-IDF + balanced logistic regression; frozen pretrained DistilBERT mean embeddings + training-only scaling and balanced logistic regression; most-frequent dummy. C chosen on validation, test evaluated afterward. Selected model: tfidf.

| Model | Validation macro-F1 | Test macro-F1 | Test with label terms masked |
|---|---:|---:|---:|
| tfidf | 0.913 | 0.900 | 0.661 |
| distilbert | 0.830 | 0.832 | 0.612 |
| dummy | 0.067 | 0.067 | N/A |


This table measures agreement with weak dataset labels. It does not establish medical accuracy, fairness, calibration or real-user generalization. Labels can be predicted from collection/annotation words, and the masked diagnostic remains affected by residual clues and domain shift. Class scores are uncalibrated.

Sensitive-data policy: no scraping or private-user collection; source text and submissions excluded from committed outputs. Only fictional demo examples are published. Runtime has no input persistence or request-body logging. No use for diagnosis, monitoring individuals, employment, insurance, eligibility or crisis intervention.

Known failure modes: topic discussion classified as personal condition, negation/quotes/sarcasm, unrelated input forced to one of six tags, out-of-vocabulary words, truncated text, non-English text and user overlap that cannot be audited. Future research needs stronger licensed annotation and verified group IDs, external validation and uncertainty estimates.

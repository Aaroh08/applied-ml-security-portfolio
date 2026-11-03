# Interview guide

Simple explanation: two classifiers look for patterns in invented text and return research dataset tags. The comparison teaches sparse versus pretrained representations and exposes weak-label limitations.

Architecture: creator CSV → normalization/deduplication → fixed splits → training-only TF-IDF or frozen DistilBERT embeddings → logistic regression → validation selection → test artifacts → Flask → browser. pandas audits data; scikit-learn handles classifiers/metrics; Transformers/PyTorch generate embeddings; Flask/Waitress serve locally; pytest verifies behavior. There is no need for a database because inputs must not be saved.

## 1. Why was the project renamed?

The dataset provides text and labels only, so calling it behavioral or multimodal would invent inputs.

## 2. What are the labels?

Six creator-supplied lexicon-derived tags, including no and five condition-named categories. They are not verified diagnoses or a validated clinical taxonomy.

## 3. How is permission documented?

The creator publishes the dataset on Zenodo with CC BY wording. Provenance stores DOI, version, attribution and checksum. We do not scrape accounts or redistribute raw text.

## 4. How many examples were really used?

A fixed stratified subset of 3000 eligible deduplicated texts, split 1800/600/600. The source has 14996 rows; that is not our training count.

## 5. Why cannot you split by user?

There are no user IDs. Text-level deduplication reduces one leakage source but does not establish user independence.

## 6. How are duplicates handled?

Normalize and redact, create punctuation-insensitive keys, remove conflicting-label groups and collapse duplicate keys before sampling/splitting.

## 7. What leakage remains?

Lexicon-defined labels, topic words, semantic duplicates, possible same-author overlap, and pretrained corpus overlap. A cosine audit and masking diagnostic expose some risks without solving all of them.

## 8. What is TF-IDF?

A training-fitted weighting of word/phrase frequency by inverse document frequency. Here unigrams and bigrams are capped at 15000 features with min_df=2.

## 9. Why frozen DistilBERT?

It fits the CPU budget and supplies contextual representations without pretending that a transformer was fine-tuned.

## 10. How are embeddings pooled?

Mean last-layer hidden states using the attention mask, including special tokens and excluding padding. Output dimension is 768.

## 11. What makes inference deterministic?

Evaluation mode disables dropout, gradients are disabled, CPU settings are fixed, and classifier/split seeds are saved. Minor library/platform floating-point differences remain possible.

## 12. How is padding tested?

The same sentence is embedded alone and alongside a longer sentence; results must agree within a numerical tolerance.

## 13. Why scaling embeddings?

StandardScaler is fitted only to training embeddings before logistic regression. It makes the L2 penalty less dependent on coordinate scale.

## 14. How are hyperparameters selected?

C values 0.1, 1 and 10 are compared on validation macro-F1, not test results. The smaller C wins ties.

## 15. Why macro-F1?

It weights each label equally and makes poor minority-class performance harder to hide. Per-class precision/recall and a confusion matrix are also saved.

## 16. What does masking prove?

It measures sensitivity to removing a limited explicit keyword set from fixed test inputs. It creates distribution shift and does not remove every labeling clue.

## 17. Why not call the scores confidence?

They are uncalibrated softmax/logistic class scores over an incomplete tag set, not trustworthy probabilities of conditions or correctness.

## 18. What happens on long text?

The API caps characters at 4000; DistilBERT truncates to 128 tokens and reports a warning. TF-IDF uses the full normalized text, a comparison limitation.

## 19. Does the app save private text?

No. No body logging, persistence or remote inference calls; outputs do not echo the input. The page retains it only until cleared/closed.

## 20. How would you improve the study?

Use better documented labels with user groups, evaluate outside the source domain, prespecify uncertainty/calibration checks and test negation/quoted speech. Never reuse test results for repeated tuning.

## Five modification exercises

1. Add a negation/quotation challenge set of clearly fictional sentences and record model disagreement without pretending the examples are clinical ground truth.
2. Add a token counter before submission and test the exact truncation boundary.
3. Compare max_length 64 versus 128 as a separate validation experiment; do not tune against the current test set.
4. Add a user-controlled toggle to run only one model and measure actual latency with warm-up documented.
5. Add a predeclared low-score abstention option; evaluate coverage and error rates on validation, explaining why it is not a clinical safety guarantee.

Maintain a learning log and be candid about AI assistance. Rehearse the missing-model error, unknown vocabulary, keyword-driven false interpretations and absent user grouping.

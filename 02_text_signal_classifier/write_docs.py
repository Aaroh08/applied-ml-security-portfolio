import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def write(name,text):(ROOT/name).write_text(text.strip()+'\n',encoding='utf-8')
results=json.loads((ROOT/'artifacts/metrics.json').read_text())
meta=json.loads((ROOT/'artifacts/metadata.json').read_text())
audit=json.loads((ROOT/'artifacts/data_audit.json').read_text())
diag=json.loads((ROOT/'artifacts/keyword_mask_diagnostic.json').read_text())
table='| Model | Validation macro-F1 | Test macro-F1 | Test with label terms masked |\n|---|---:|---:|---:|\n'
for name,r in results.items():
    masked=f"{diag[name]['macro_f1']:.3f}" if name in diag else 'N/A'
    table+=f"| {name} | {r['validation']['macro_f1']:.3f} | {r['test']['macro_f1']:.3f} | {masked} |\n"
write('README.md', '''# Text-only depression-related label classification

A local research prototype comparing TF-IDF and frozen DistilBERT representations. It predicts **lexicon-derived dataset tags**, not whether a person has depression. There are no behavioral features and this is not a multimodal system. Built with AI assistance as a learning portfolio project.

![Running application](docs/demo.png)

## Setup and demo

Python 3.12, CPU-only Windows is the tested environment. From this project directory:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe download.py
.venv/Scripts/python.exe train.py
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
.venv/Scripts/python.exe app.py
```

Open http://127.0.0.1:5052. Choose one of the **fictional** examples and compare both models. On Windows `./start.ps1` starts an already-trained installation; on Linux/macOS substitute `.venv/bin/python`. Downloads are free and require internet; inference runs locally with `local_files_only=True`. Model weights, private input, dataset text and trained pickle files are not committed. Train after cloning. Never load untrusted joblib artifacts.

## Source, permission and labels

[Muhammad Osama Nusrat's Zenodo release](https://zenodo.org/records/14233292), version v1.0, DOI 10.5281/zenodo.14233292, explicitly states CC BY 4.0 permission. [The accompanying paper](https://arxiv.org/abs/2404.13104) describes lexicon-based collection and annotation. We download the creator's release, not scrape user accounts. Attribution and SHA-256 are in `artifacts/provenance.json`. The pinned CSV is rejected if its hash changes.

The actual CSV has 14,996 rows and only `Tweets` and `Labels`. Tags and original counts: postpartum 3,747; major depressive 2,517; bipolar 2,443; psychotic 2,312; no 1,985; atypical 1,980. These are inherited dataset names, not a validated diagnostic taxonomy. In particular, topic labels must not be used to infer a clinical condition. “No” is simply one dataset category, not reassurance about health.

There are **no user IDs, timestamps or behavioral columns**. The project therefore cannot perform user-level splits or behavior-only/fusion experiments. It makes no claims about posting frequency, sleep patterns, multimodal signals or user diagnosis. The publisher's usage permission does not establish individual clinical consent or validate labels. Text may remain sensitive even when direct identifiers are absent; raw text is kept out of published outputs and demonstrations.

## Pipeline and leakage controls

`creator CSV → normalization/redaction → conflicting-label and exact-duplicate removal → seeded stratified 3,000-row subset → 60/20/20 splits → representation comparison → validation selection → held-out evaluation → Flask demo`

Normalization applies Unicode NFKC, lowercasing, whitespace collapse and URL/email/handle replacement. It does not invent behavioral measurements. Punctuation-insensitive duplicate keys are removed before splitting; conflicting-label groups are removed entirely. No patient/user-disjoint claim is possible. A training-vocabulary cosine-similarity audit records possible near duplicates rather than pretending exact deduplication removes all related text.

The fixed CPU budget is 3,000 eligible texts, chosen by stratified sampling with seed 42 before modeling. Splits are 1,800 train, 600 validation and 600 test. The full source is not used to claim a larger experiment. All source row indices and counts are saved.

TF-IDF uses training-only unigram/bigram vocabulary, minimum document frequency 2, at most 15,000 features and sublinear term frequency. Its classifier is class-balanced logistic regression. DistilBERT uses the pinned Apache-2.0 `distilbert/distilbert-base-uncased` weights in evaluation mode with gradients disabled. Attention-mask mean pooling (including special tokens) yields 768 features; padding is excluded. Maximum length is 128 tokens, batch size 8, four CPU threads. Training-only StandardScaler and balanced logistic regression are fitted to embeddings. No transformer fine-tuning is claimed.

For each representation, C is selected from 0.1, 1 and 10 by validation macro-F1, preferring the smaller C on ties. A most-frequent dummy is included in model selection. Decisions use argmax, with no clinical threshold. Test labels do not select hyperparameters or the served-model recommendation. Embeddings are cached using a fingerprint of splits, revision and preprocessing settings. Re-running can reuse the cache; delete only the known local `artifacts/embeddings.npz` to force extraction.

## Measured results

''' + table + f'''
Validation selected **{meta['selected_model']}**. All test results use the same 600 records. Per-class precision, recall, F1 and confusion matrices are in `artifacts/metrics.json`; `confusion_matrices.png` visualizes errors. Predictions include source row index, labels and scores, but **no source text**.

Audit found {audit['conflicting_duplicate_groups']} conflicting duplicate groups and removed {audit['same_label_duplicates_removed']} same-label duplicates; {audit['unique_eligible_rows']} unique eligible rows remained before the CPU subset. The near-duplicate audit found {audit['test_rows_with_train_cosine_at_least_0_90']} test rows with training cosine similarity ≥0.90. Exact counts, per-split class balance and explicit-label-term frequency are saved in `data_audit.json`.

The masked diagnostic removes explicit category/depression terms only after model selection. It tests keyword dependence and introduces distribution shift; it does not eliminate the creator's entire labeling lexicon or prove a causal effect. Do not present its result as a new tuned benchmark. Topic leakage and weak labels can make ordinary scores misleadingly strong. Quoted educational content can receive a condition-named tag despite not describing anyone's health.

## API and privacy

- `GET /api/health`: ready (200) or unavailable (503).
- `POST /api/predict`: exactly `{{"text":"fictional English passage"}}`, 10–4,000 characters and at least three alphabetic words. Returns both models' six class scores, dataset tags, disagreement indicator and truncation warning.
- Invalid fields/content return 400; non-JSON 415; oversized bodies 413; missing model files 503. Generic inference errors do not echo text.

```powershell
Invoke-RestMethod http://127.0.0.1:5052/api/predict -Method Post -ContentType application/json -Body (Get-Content examples/fictional_input.json -Raw)
```

Scores are uncalibrated and not probabilities of mental-health conditions. Both models must choose a known tag even for unrelated input; disagreement or a low maximum score is not a validated abstention policy. DistilBERT can truncate; TF-IDF uses all normalized input. The app does not persist, echo, or log submitted text. Clear removes text/output from the page. No external inference API, analytics or input-history feature is used. Loopback-only Waitress, one worker, no-store responses. This is not hardened for public hosting.

## Limitations and learning

English-only source, lexicon/corpus selection bias, class imbalance, missing user independence, potential semantic duplicates, domain shift, sarcasm and negation errors. Source preprocessing may have removed information irreversibly. Pretraining can overlap public internet material. No clinical screening, severity inference, crisis detection or treatment advice is provided.

See [MODEL_CARD.md](MODEL_CARD.md), [DEMO.md](DEMO.md), [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md), [EVIDENCE.md](EVIDENCE.md) and [STATUS.md](STATUS.md). The implementation favors a reproducible CPU study over unsupported claims of diagnostic accuracy.
''')
write('MODEL_CARD.md',f'''# Model card

Educational text classification of six creator-supplied, lexicon-derived tags. No human diagnostic inference. AI-assisted student portfolio implementation.

Dataset: Nusrat's Multi-Class Depression Detection Dataset, Zenodo v1.0, CC BY 4.0, DOI 10.5281/zenodo.14233292. Only Tweets/Labels are present. 3,000 deduplicated eligible texts are selected with fixed seed before splitting into 1,800/600/600. User-disjoint evaluation is impossible without IDs; no behavior is inferred.

Models: training-only TF-IDF + balanced logistic regression; frozen pretrained DistilBERT mean embeddings + training-only scaling and balanced logistic regression; most-frequent dummy. C chosen on validation, test evaluated afterward. Selected model: {meta['selected_model']}.

{table}

This table measures agreement with weak dataset labels. It does not establish medical accuracy, fairness, calibration or real-user generalization. Labels can be predicted from collection/annotation words, and the masked diagnostic remains affected by residual clues and domain shift. Class scores are uncalibrated.

Sensitive-data policy: no scraping or private-user collection; source text and submissions excluded from committed outputs. Only fictional demo examples are published. Runtime has no input persistence or request-body logging. No use for diagnosis, monitoring individuals, employment, insurance, eligibility or crisis intervention.

Known failure modes: topic discussion classified as personal condition, negation/quotes/sarcasm, unrelated input forced to one of six tags, out-of-vocabulary words, truncated text, non-English text and user overlap that cannot be audited. Future research needs stronger licensed annotation and verified group IDs, external validation and uncertainty estimates.
''')
write('DEMO.md','''# Five-minute demo

0:00–0:45 — Explain the revised scope: text-only dataset-tag classification. There are no behavioral fields. The labels were lexicon-derived, not confirmed diagnoses.

0:45–1:45 — Open localhost:5052, load the everyday fictional example and compare both models. Read the dataset-tag wording, all six scores and disagreement. Never describe “no” as proof of wellness.

1:45–2:30 — Load the library-topic example. Explain why a text about a book can trigger a label-named topic. This is a false interpretation risk, even if the predicted dataset tag matches keywords.

2:30–3:15 — Show validation selection and the test confusion matrices. Explain macro-F1, class balance and the dummy baseline. Compare ordinary results with keyword_mask_diagnostic.json without calling the latter a causality test.

3:15–4:15 — Show the embedding code: eval(), requires_grad_(False), attention-mask mean pooling and 128-token cap. Contrast TF-IDF's training vocabulary with a pretrained representation.

4:15–5:00 — Show passing tests, the no-text prediction artifact, and Clear text and output. Explain missing user IDs and the privacy/clinical limits. Discuss one modification exercise you personally completed.
''')
qa=[
('Why was the project renamed?','The dataset provides text and labels only, so calling it behavioral or multimodal would invent inputs.'),
('What are the labels?','Six creator-supplied lexicon-derived tags, including no and five condition-named categories. They are not verified diagnoses or a validated clinical taxonomy.'),
('How is permission documented?','The creator publishes the dataset on Zenodo with CC BY wording. Provenance stores DOI, version, attribution and checksum. We do not scrape accounts or redistribute raw text.'),
('How many examples were really used?','A fixed stratified subset of 3000 eligible deduplicated texts, split 1800/600/600. The source has 14996 rows; that is not our training count.'),
('Why cannot you split by user?','There are no user IDs. Text-level deduplication reduces one leakage source but does not establish user independence.'),
('How are duplicates handled?','Normalize and redact, create punctuation-insensitive keys, remove conflicting-label groups and collapse duplicate keys before sampling/splitting.'),
('What leakage remains?','Lexicon-defined labels, topic words, semantic duplicates, possible same-author overlap, and pretrained corpus overlap. A cosine audit and masking diagnostic expose some risks without solving all of them.'),
('What is TF-IDF?','A training-fitted weighting of word/phrase frequency by inverse document frequency. Here unigrams and bigrams are capped at 15000 features with min_df=2.'),
('Why frozen DistilBERT?','It fits the CPU budget and supplies contextual representations without pretending that a transformer was fine-tuned.'),
('How are embeddings pooled?','Mean last-layer hidden states using the attention mask, including special tokens and excluding padding. Output dimension is 768.'),
('What makes inference deterministic?','Evaluation mode disables dropout, gradients are disabled, CPU settings are fixed, and classifier/split seeds are saved. Minor library/platform floating-point differences remain possible.'),
('How is padding tested?','The same sentence is embedded alone and alongside a longer sentence; results must agree within a numerical tolerance.'),
('Why scaling embeddings?','StandardScaler is fitted only to training embeddings before logistic regression. It makes the L2 penalty less dependent on coordinate scale.'),
('How are hyperparameters selected?','C values 0.1, 1 and 10 are compared on validation macro-F1, not test results. The smaller C wins ties.'),
('Why macro-F1?','It weights each label equally and makes poor minority-class performance harder to hide. Per-class precision/recall and a confusion matrix are also saved.'),
('What does masking prove?','It measures sensitivity to removing a limited explicit keyword set from fixed test inputs. It creates distribution shift and does not remove every labeling clue.'),
('Why not call the scores confidence?','They are uncalibrated softmax/logistic class scores over an incomplete tag set, not trustworthy probabilities of conditions or correctness.'),
('What happens on long text?','The API caps characters at 4000; DistilBERT truncates to 128 tokens and reports a warning. TF-IDF uses the full normalized text, a comparison limitation.'),
('Does the app save private text?','No. No body logging, persistence or remote inference calls; outputs do not echo the input. The page retains it only until cleared/closed.'),
('How would you improve the study?','Use better documented labels with user groups, evaluate outside the source domain, prespecify uncertainty/calibration checks and test negation/quoted speech. Never reuse test results for repeated tuning.')]
write('INTERVIEW_GUIDE.md','''# Interview guide

Simple explanation: two classifiers look for patterns in invented text and return research dataset tags. The comparison teaches sparse versus pretrained representations and exposes weak-label limitations.

Architecture: creator CSV → normalization/deduplication → fixed splits → training-only TF-IDF or frozen DistilBERT embeddings → logistic regression → validation selection → test artifacts → Flask → browser. pandas audits data; scikit-learn handles classifiers/metrics; Transformers/PyTorch generate embeddings; Flask/Waitress serve locally; pytest verifies behavior. There is no need for a database because inputs must not be saved.

'''+'\n\n'.join(f'## {i}. {q}\n\n{a}' for i,(q,a) in enumerate(qa,1))+'''

## Five modification exercises

1. Add a negation/quotation challenge set of clearly fictional sentences and record model disagreement without pretending the examples are clinical ground truth.
2. Add a token counter before submission and test the exact truncation boundary.
3. Compare max_length 64 versus 128 as a separate validation experiment; do not tune against the current test set.
4. Add a user-controlled toggle to run only one model and measure actual latency with warm-up documented.
5. Add a predeclared low-score abstention option; evaluate coverage and error rates on validation, explaining why it is not a clinical safety guarantee.

Maintain a learning log and be candid about AI assistance. Rehearse the missing-model error, unknown vocabulary, keyword-driven false interpretations and absent user grouping.
''')
write('EVIDENCE.md',f'''# Resume bullets and evidence

- Built a local text-only research demo comparing TF-IDF and frozen DistilBERT embeddings with logistic regression, without storing submitted text or inventing behavioral features.
- Evaluated both representations and a dummy baseline on a shared 600-text holdout from a seeded 3000-text subset; recorded macro-F1 {results[meta['selected_model']]['test']['macro_f1']:.3f} for the validation-selected model.
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
''')
print('Wrote measured results and learning documentation.')

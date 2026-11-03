# Five-minute demo

0:00–0:45 — Explain the revised scope: text-only dataset-tag classification. There are no behavioral fields. The labels were lexicon-derived, not confirmed diagnoses.

0:45–1:45 — Open localhost:5052, load the everyday fictional example and compare both models. Read the dataset-tag wording, all six scores and disagreement. Never describe “no” as proof of wellness.

1:45–2:30 — Load the library-topic example. Explain why a text about a book can trigger a label-named topic. This is a false interpretation risk, even if the predicted dataset tag matches keywords.

2:30–3:15 — Show validation selection and the test confusion matrices. Explain macro-F1, class balance and the dummy baseline. Compare ordinary results with keyword_mask_diagnostic.json without calling the latter a causality test.

3:15–4:15 — Show the embedding code: eval(), requires_grad_(False), attention-mask mean pooling and 128-token cap. Contrast TF-IDF's training vocabulary with a pretrained representation.

4:15–5:00 — Show passing tests, the no-text prediction artifact, and Clear text and output. Explain missing user IDs and the privacy/clinical limits. Discuss one modification exercise you personally completed.

# Project 2 status

Completed: licensed creator dataset download, pinned/checksummed DistilBERT files, duplicate and leakage audit, fixed CPU subset/splits, actual TF-IDF/DistilBERT/dummy training and evaluation, keyword-masking diagnostic, Flask API, local interface, model card and learning/evidence guides.

Observed results: validation selected TF-IDF. Held-out macro-F1 was 0.8996 for TF-IDF, 0.8322 for frozen DistilBERT and 0.0667 for the dummy. Masking explicit label terms reduced learned-model macro-F1 to 0.6612 and 0.6124. The ordinary results measure agreement with lexicon-derived dataset labels, not clinical accuracy. Sixteen tests passed, with fourteen third-party plotting-library deprecation warnings and no failures.

Commands actually run using the project virtual environment:

```text
python -m pip install -r requirements.txt
python download.py
python train.py
python -m pytest tests -q -p no:cacheprovider --junitxml=artifacts/test_results.xml
python -m pip check
python -m pip freeze
python write_docs.py
python app.py
```

Initial official weight download stalled; a fresh URL resolved it. A detached demo process did not survive the host command lifecycle, so the verified server was run in a persistent foreground session. `start.ps1` is intended to remain open while the app runs. Actual training output and test evidence are saved under artifacts/.

Limits: no user identifiers or behavioral fields; cannot verify user-level independence. Two test texts have high training-text cosine similarity. Weak lexicon labels, topic bias, semantic duplicates and domain shift remain. Class scores are uncalibrated and every input is assigned to an incomplete fixed tag set. No diagnosis, crisis detection, production deployment or real-user benefit is claimed. Raw source text, submitted text and model weights are excluded from Git.

Learning: trace one fictional input through normalization, tokenization and both representations; reproduce the masking diagnostic; complete the five exercises in INTERVIEW_GUIDE.md. Be candid about AI assistance and use only evidence-backed resume bullets.

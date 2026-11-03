# Completion status · 2026-11-03

Implemented and run: synthetic dataset generation; template-separated training/evaluation; rules, forest and combined comparisons; Flask gateway; protected demo; actual MySQL 8.4.13 database; redacted event dashboard; parameterized SQL; escaping; CSRF; startup readiness; graceful MySQL shutdown; benchmark; browser verification; documentation.

Validation: 10 tests passed in 8.96 seconds; pip check found no broken requirements. The browser dashboard returned HTTP 403 for the SQL pattern and immediately displayed the recorded MySQL event. Screenshot: docs/demo.png.

Commands: install requirements, run train.py, setup_mysql.py, then run_local.py. In a second terminal run pytest and benchmark.py using the project virtual environment. See README for exact commands.

Initial startup was attempted before training completed; the launcher now checks model existence and waits for web health before announcing readiness. A lingering database process was shut down gracefully; the launcher now requests MySQL shutdown during cleanup. The subsequent full integration suite passed.

Known limits: the combined detector has 60/180 benign false positives on synthetic held-out traffic. No production deployment, general attack coverage, authentication, load testing or TLS is claimed. Shared local database credentials are ignored by Git, and notes must contain fictional data only.

Learning focus: explain why the combined detector loses macro-F1, why synthetic scores overestimate realistic effectiveness, why CSRF is distinct, and why application-level defenses remain essential.

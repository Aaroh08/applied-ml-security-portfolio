# Defensible resume statements

Use these only after understanding and demonstrating the implementation. Do not imply production deployment or measured protection of real users.

- Built a localhost Flask security gateway combining transparent SQLi/XSS rules with a 200-tree Random Forest, evaluated on 930 authored examples with separated template families. [Source](train.py), [evaluation](artifacts/metrics.json).
- Implemented real MySQL event storage with payload redaction, a live dashboard, parameterized application queries and independent CSRF validation. [Gateway](app.py), [storage](storage.py), [screenshot](docs/demo.png).
- Verified 10 automated tests, including live database and application defense checks, and measured 50 sequential requests per endpoint with raw latency records. [Tests](tests/test_system.py), [test results](artifacts/test_run.txt), [timing](artifacts/latency.json).

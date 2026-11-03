# Detector model card

Purpose: demonstrate local defensive request classification with explainable rules and a small Random Forest. Labels mean authored benign/attack intent, not verified exploitability.

Training: 330 synthetic examples; 240 validation; 360 held-out examples. Named families are disjoint, but semantic patterns overlap. No real traffic, external scanning, private logs or user data were collected. Source generation and seed are versioned in train.py. No scaling or fitted text vocabulary is used; stateless feature extraction is shared.

The validation-selected RF threshold is 0.5. Test RF macro-F1 is 0.9161; served rules-OR-forest macro-F1 is 0.8286. Combined confusion matrix in benign/attack order is [[120,60],[0,180]]. Rules alone miss 30 obfuscated SQL examples. Both components have false positives on different benign families, so their union increases false positives.

Do not interpret tree vote scores as calibrated real-world attack probabilities. This corpus cannot support a production accuracy claim, claims about zero-day attacks, or a general SQLi/XSS prevention claim. A WAF does not replace parameterized SQL, output encoding or CSRF validation. It does not classify CSRF.

Failure modes include new encodings, uninspected headers, contextual benign code, Unicode/parser differences and concept drift. Fixed synthetic rules and features were authored together, creating evaluation bias. Any future deployment requires independent traffic, permitted collection, realistic prevalence, adversarial testing, latency/load testing and application-specific allowlisting.

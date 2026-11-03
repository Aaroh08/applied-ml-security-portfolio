# Interview practice

1. **What does the gateway protect?** Only the bundled local demo's three allowlisted routes; it is not a general reverse proxy.
2. **Why use a forest?** It models interactions between small numeric request features without expensive inference.
3. **Why keep rules?** They provide explicit pattern IDs, but their union can increase false positives.
4. **Where did data come from?** Authored templates; no real traffic was collected.
5. **How are splits separated?** Named template families and all their encoding variants remain in one split.
6. **Is that enough to remove bias?** No; semantics and feature design still overlap across splits.
7. **How was the threshold selected?** Validation macro-F1 over a fixed grid; test data did not select it.
8. **Why macro-F1?** It weighs benign and attack class F1 equally; deployment costs may require another objective.
9. **What was the surprising result?** Combined macro-F1 fell to 0.8286 because it accumulated both components' false positives.
10. **Does zero FN mean complete protection?** No; it covers only the 180 authored held-out attack examples.
11. **What does normalization do?** NFKC, bounded URL decoding, entity decoding and whitespace/case normalization.
12. **Can normalization fail?** Yes; application parsers and nested encodings may differ.
13. **How is SQLi prevented in the app?** Query parameters are bound through the driver, never concatenated into SQL.
14. **How is output protected?** Jinja autoescaping and a restrictive CSP; no user text is marked safe.
15. **Does the forest stop CSRF?** No; Flask-WTF checks a session token on state-changing requests.
16. **What is logged?** Fixed route, decision, rule IDs, score, status, duration and body size; raw values are omitted.
17. **What happens if MySQL fails?** A failed initial event insert prevents forwarding. A failed completion update is signaled in a header.
18. **Why is the backend accessible directly?** To test layered application defenses in a local lab; production must isolate upstream access.
19. **What does the latency benchmark prove?** Only sequential timing on this machine, including database writes; not scalable throughput.
20. **What would you improve first?** Independent realistic validation and explicit false-positive costs before changing the detector.

## Five exercises

1. Add a new benign template family to a new evaluation set; report FP changes without retuning on it.
2. Compare AND versus OR combination and explain the recall/precision tradeoff.
3. Add a parameterized pagination query and test invalid page bounds.
4. Test a database completion-update failure and confirm the response header and pending event behavior.
5. Add an event retention strategy with a dry-run preview; document how privacy and audit requirements determine retention.

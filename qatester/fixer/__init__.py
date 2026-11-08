"""The Fixer loop: a learned agent repairs the bugs QA reports (docs/FIXER_PLAN.md).

Nothing in this package may read a level's defect list or answer key; only the scoring
that evaluates the loop afterwards does (enforced by tests/test_no_groundtruth_leak.py).
"""

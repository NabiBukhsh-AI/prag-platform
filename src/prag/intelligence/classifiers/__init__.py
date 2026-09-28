"""The classifier cascade: T0 rules, T1 encoder heads, T2 LLM fallback.

T0 ships now. T1 arrives once Phase 2 traffic has produced labels to train it on — building it
before there are labels would mean training on synthetic data and calling the result empirical.
T2 is the low-confidence fallback only, capped at a small share of traffic.
"""

from prag.intelligence.classifiers.rules import RuleVerdict, classify_rules

__all__ = ["RuleVerdict", "classify_rules"]

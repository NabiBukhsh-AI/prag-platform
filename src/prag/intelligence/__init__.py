"""Query understanding and strategy routing — one module, deliberately.

They ship together because routing needs the per-head confidences, not just the argmax, and a
boundary between them would put a network hop between a classifier and its only consumer.

Routing is a small trained classifier, never an LLM call on the default path. A 300 ms model call
cannot sit on the critical path of a 650 ms time-to-first-token target; at that point it is not
routing, it is a second generation.
"""

from prag.intelligence.analysis import CascadeQueryAnalyzer, source_hints_for
from prag.intelligence.classifiers.rules import RuleVerdict, classify_rules
from prag.intelligence.strategy_router import (
    DEFAULT_PROFILES,
    QualityTable,
    StrategyProfile,
    UtilityStrategyRouter,
)
from prag.intelligence.transform import (
    CoreferenceTransformer,
    DecompositionTransformer,
    ExpansionTransformer,
    RewriteTransformer,
    apply_transforms,
)

__all__ = [
    "DEFAULT_PROFILES",
    "CascadeQueryAnalyzer",
    "CoreferenceTransformer",
    "DecompositionTransformer",
    "ExpansionTransformer",
    "QualityTable",
    "RewriteTransformer",
    "RuleVerdict",
    "StrategyProfile",
    "UtilityStrategyRouter",
    "apply_transforms",
    "classify_rules",
    "source_hints_for",
]

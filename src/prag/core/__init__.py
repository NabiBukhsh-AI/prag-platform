"""Protocols, domain models, errors, budget enforcement, and the DI container.

**This package imports nothing internal.** It is the contract every other package is written
against, and the dependency direction is enforced by ``importlinter.ini`` rather than by
convention. If ``core`` ever acquires a dependency on an implementation, the dependency
inversion that makes every other boundary work is gone, and the modular monolith stops being
extractable into services.

The practical consequence for anyone adding a feature: the protocol and its models go here
first, then the implementation goes elsewhere. Writing the protocol afterwards produces a
description of whatever the first implementation happened to do.
"""

from prag.core.budget import BudgetController, DegradationLevel, DegradationPlan
from prag.core.di import Container, Scope
from prag.core.errors import PragError, Severity

__all__ = [
    "BudgetController",
    "Container",
    "DegradationLevel",
    "DegradationPlan",
    "PragError",
    "Scope",
    "Severity",
]

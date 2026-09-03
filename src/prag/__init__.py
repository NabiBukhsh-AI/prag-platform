"""Parametric RAG platform.

A modular monolith. Package boundaries are enforced by ``importlinter.ini`` rather than by
convention: ``core`` imports nothing internal, and every cross-boundary interaction is a
``typing.Protocol`` declared in :mod:`prag.core.protocols`.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

"""The dependency injection container.

The single most important pattern in this codebase, because it is what makes the modular
monolith extractable into services later. Every cross-boundary dependency is a protocol resolved
here, so replacing an in-process implementation with an HTTP client is a registration change.

**This is a container, not a service locator.** The distinction is the whole point and it is
easy to lose. The container is used in exactly two places: the composition root at startup, and
test setup. A module that imports the container and reaches into it during a request has
recreated the global singleton this exists to avoid — its dependencies become invisible at the
call site, and it can no longer be unit tested without building the world.

So: constructors take the protocols they need. The composition root resolves them once and wires
them together. Nothing else touches this file.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar, cast

from prag.core.errors import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = ["Container", "Scope"]

T = TypeVar("T")


class Scope(StrEnum):
    """How long a resolved instance lives."""

    #: One instance for the container's lifetime. Correct for anything holding a connection
    #: pool, a loaded model, or a cache.
    SINGLETON = "singleton"
    #: A fresh instance per resolution. Correct for anything carrying per-request state, which
    #: must not be shared across concurrent requests.
    TRANSIENT = "transient"


@dataclass(slots=True)
class _Registration:
    provider: Callable[[Container], Any]
    scope: Scope
    instance: Any = None
    resolved: bool = False


class Container:
    """Maps protocols to the implementations that satisfy them.

    Registration is by protocol type, and resolution is typed, so ``resolve(LLMProvider)``
    returns something typed as ``LLMProvider`` without a cast at the call site.
    """

    def __init__(self) -> None:
        self._registrations: dict[type, _Registration] = {}
        #: Resolution stack, for cycle detection. A dependency cycle otherwise surfaces as a
        #: recursion error thousands of frames deep, naming nothing useful.
        self._resolving: list[type] = []

    def register(
        self,
        protocol: type[T],
        provider: Callable[[Container], T],
        *,
        scope: Scope = Scope.SINGLETON,
    ) -> Container:
        """Bind a protocol to a factory.

        The factory receives the container so it can resolve its own dependencies, which is what
        lets registrations be declared in any order — nothing is constructed until the first
        resolution.

        Returns ``self`` so a composition root reads as one chained declaration rather than
        fifty separate statements.
        """
        if protocol in self._registrations:
            raise ConfigurationError(
                "duplicate registration",
                protocol=protocol.__name__,
                hint="use override() in tests rather than re-registering",
            )
        self._registrations[protocol] = _Registration(provider=provider, scope=scope)
        return self

    def register_instance(self, protocol: type[T], instance: T) -> Container:
        """Bind a protocol to an already-constructed object.

        For configuration objects and anything built before the container exists.
        """
        return self.register(protocol, lambda _: instance, scope=Scope.SINGLETON)

    def resolve(self, protocol: type[T]) -> T:
        """Get the implementation bound to ``protocol``.

        Raises :class:`ConfigurationError` rather than returning ``None`` when nothing is
        registered. A missing dependency is a wiring mistake that should stop the process at
        startup, not an optional feature that quietly does nothing in production.
        """
        registration = self._registrations.get(protocol)
        if registration is None:
            raise ConfigurationError(
                "no implementation registered",
                protocol=protocol.__name__,
                registered=sorted(p.__name__ for p in self._registrations),
            )

        if registration.scope is Scope.SINGLETON and registration.resolved:
            return cast("T", registration.instance)

        if protocol in self._resolving:
            cycle = [*(p.__name__ for p in self._resolving), protocol.__name__]
            raise ConfigurationError("dependency cycle", cycle=" -> ".join(cycle))

        self._resolving.append(protocol)
        try:
            instance = registration.provider(self)
        finally:
            self._resolving.pop()

        if registration.scope is Scope.SINGLETON:
            registration.instance = instance
            registration.resolved = True

        return cast("T", instance)

    def has(self, protocol: type) -> bool:
        return protocol in self._registrations

    @contextmanager
    def override(self, protocol: type[T], instance: T) -> Iterator[None]:
        """Temporarily replace a binding.

        For tests that need one fake in an otherwise real graph. Restores the previous
        registration on exit, including its cached singleton, so an override cannot leak into
        the next test and produce a failure that only reproduces in suite order.
        """
        previous = self._registrations.get(protocol)
        self._registrations[protocol] = _Registration(
            provider=lambda _: instance,
            scope=Scope.SINGLETON,
            instance=instance,
            resolved=True,
        )
        try:
            yield
        finally:
            if previous is None:
                del self._registrations[protocol]
            else:
                self._registrations[protocol] = previous

    def validate(self) -> None:
        """Eagerly construct every singleton.

        Called at startup so that a broken wiring, a missing credential, or an unreachable
        dependency fails the process immediately rather than on the first request that happens
        to need it. Transient registrations are skipped: they may legitimately require
        per-request arguments that do not exist yet.
        """
        errors: list[str] = []
        for protocol, registration in sorted(
            self._registrations.items(), key=lambda kv: kv[0].__name__
        ):
            if registration.scope is not Scope.SINGLETON:
                continue
            try:
                self.resolve(protocol)
            except Exception as exc:
                errors.append(f"{protocol.__name__}: {exc}")
        if errors:
            raise ConfigurationError("container validation failed", failures=errors)

"""Layered configuration loading and per-tenant resolution.

Four layers, each overriding the last: shipped defaults, an environment file, environment
variables, and per-tenant overrides resolved at request time.

The tenant layer is the one with teeth. Overrides are checked against an explicit allow-list,
and an attempt to override anything else is refused rather than ignored. Silently dropping a
disallowed override would leave a tenant believing they had turned off a guardrail — which is
worse than the override succeeding, because nobody would look again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from prag.config.schema import ALLOWED_OVERRIDE_KEYS, PragSettings
from prag.core.errors import ConfigurationError
from prag.core.models.identity import TenantPolicy, UtilityWeights

__all__ = ["load_settings", "resolve_tenant_policy"]


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge ``overlay`` onto ``base``, recursing into nested mappings.

    Recursive rather than top-level replacement: a tenant overriding one rerank tier must not
    silently delete the others by supplying a partial mapping.
    """
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the install extra
        raise ConfigurationError(
            "PyYAML is required to load configuration files",
            path=str(path),
            hint="install the project with its base dependencies",
        ) from exc

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError("could not read config file", path=str(path)) from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError("config file is not valid YAML", path=str(path)) from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(
            "config file must contain a mapping at the top level",
            path=str(path),
            found=type(loaded).__name__,
        )
    return loaded


def load_settings(
    *,
    defaults_path: Path | None = None,
    env_file_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> PragSettings:
    """Build the settings object, failing at startup on anything invalid.

    ``overrides`` is for tests and for the composition root, not for tenants: it bypasses the
    allow-list, which is exactly why it must never be wired to anything a tenant can reach.
    """
    layered: dict[str, Any] = {}

    if defaults_path is not None and defaults_path.exists():
        layered = _deep_merge(layered, _read_yaml(defaults_path))
    if env_file_path is not None and env_file_path.exists():
        layered = _deep_merge(layered, _read_yaml(env_file_path))
    if overrides:
        layered = _deep_merge(layered, overrides)

    try:
        # Environment variables are applied by BaseSettings itself, on top of what is passed in.
        return PragSettings(**layered)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError("invalid configuration", detail_text=str(exc)) from exc


def _get_path(data: Any, dotted: str) -> Any:
    current = data
    for part in dotted.split("."):
        current = current.get(part) if isinstance(current, dict) else getattr(current, part, None)
        if current is None:
            return None
    return current


def resolve_tenant_policy(
    settings: PragSettings,
    tenant_id: str,
    tenant_overrides: dict[str, Any] | None = None,
) -> TenantPolicy:
    """Resolve the policy in force for one tenant's request.

    Rejects any override outside the allow-list. Refusing loudly beats ignoring quietly: a
    tenant who believes they disabled a guardrail and were silently overruled has a false model
    of the system that nobody will correct, because from the outside it looks like it worked.
    """
    overrides = tenant_overrides or {}
    disallowed = sorted(set(overrides) - ALLOWED_OVERRIDE_KEYS)
    if disallowed:
        raise ConfigurationError(
            "tenant override targets a key that is not tenant-overridable",
            tenant_id=tenant_id,
            disallowed=disallowed,
            allowed=sorted(ALLOWED_OVERRIDE_KEYS),
        )

    sla_tier = str(overrides.get("routing.sla_tier", "standard"))
    weights_override = overrides.get("routing.utility_weights")
    if isinstance(weights_override, dict):
        weights = UtilityWeights(**weights_override)
    else:
        weights = settings.routing.utility_weights.get(
            sla_tier, settings.routing.utility_weights["standard"]
        )

    return TenantPolicy(
        tenant_id=tenant_id,
        config_version=settings.config_version,
        utility_weights=weights,
        strict_mode=bool(overrides.get("guardrails.strict_mode", settings.guardrails.strict_mode)),
        max_evidence_tokens=int(
            overrides.get("context.max_evidence_tokens", settings.context.max_evidence_tokens)
        ),
        abstain_below_knowledge_score=float(
            overrides.get(
                "fusion.abstain_below_knowledge_score",
                settings.fusion.abstain_below_knowledge_score,
            )
        ),
        semantic_cache_similarity_floor=float(
            overrides.get(
                "caching.tiers.semantic.similarity_floor",
                _get_path(settings, "caching.tiers.semantic.similarity_floor") or 0.95,
            )
        ),
        # A tenant may turn the parametric tier off even where the platform has it on, but
        # never on where the platform has it off. Enabling it requires calibrated evaluation,
        # which is a platform-level fact a tenant is in no position to assert.
        parametric_enabled=bool(overrides.get("parametric.enabled", settings.parametric.enabled))
        and settings.parametric.enabled,
    )

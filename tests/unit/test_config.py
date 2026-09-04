"""The configuration contract, its cross-section validation, and tenant resolution."""

from __future__ import annotations

from typing import Any

import pytest

from prag.config import ALLOWED_OVERRIDE_KEYS, PragSettings, load_settings, resolve_tenant_policy
from prag.config.schema import RerankTierConfig, SourceConfig
from prag.core.errors import ConfigurationError


class TestDefaults:
    """The shipped defaults encode the phased roadmap, so they are worth asserting."""

    def test_defaults_load(self) -> None:
        assert PragSettings().config_version

    def test_parametric_ships_off(self) -> None:
        """Phase 1 has no parametric tier.

        You cannot know what is worth parameterizing until a non-parametric baseline has been
        measured, and the system is designed to be fully useful with this false.
        """
        assert PragSettings().parametric.enabled is False

    def test_abstractive_compression_ships_off(self) -> None:
        """A small model summarising evidence can fabricate.

        A fabrication introduced during compression is indistinguishable downstream from one
        the answer model invented.
        """
        compression = PragSettings().context.compression
        assert compression.abstractive_enabled is False
        assert "numeral" in compression.abstractive_forbid_on
        assert "quote" in compression.abstractive_forbid_on

    def test_tool_provenance_gate_ships_on(self) -> None:
        """The defence that holds after the model has already been fooled."""
        assert PragSettings().guardrails.tool_provenance_gate is True

    def test_judge_is_uncalibrated_until_proven_otherwise(self) -> None:
        """No judge score gates anything until it has been calibrated against human labels."""
        assert PragSettings().evaluation.judge_calibrated is False

    def test_adversarial_gate_is_total(self) -> None:
        """A single injection or ACL probe getting through is an incident, not a percentage."""
        assert PragSettings().evaluation.adversarial_pass_rate_min == 1.0

    def test_never_cache_covers_every_warning_class(self) -> None:
        never = PragSettings().caching.never_cache
        assert {"abstained", "coverage_warning", "staleness_warning", "validation_failed"} <= set(
            never
        )

    def test_realtime_content_is_never_cached(self) -> None:
        assert PragSettings().caching.ttl_by_volatility["realtime"] == 0

    def test_long_term_memory_forbids_model_generated(self) -> None:
        memory = PragSettings().memory
        assert "model_generated" in memory.forbid
        assert "model_generated" not in memory.write_sources


class TestCrossSectionValidation:
    """Mistakes a per-field validator cannot see."""

    def test_default_graph_must_be_registered(self) -> None:
        with pytest.raises(ValueError, match="not in the graph registry"):
            PragSettings(orchestration={"default_graph": "nonexistent"})

    def test_rerank_tier_must_be_defined(self) -> None:
        with pytest.raises(ValueError, match="referenced but not defined"):
            PragSettings(reranking={"tier_by_sla": {"standard": "ultra"}})

    def test_fusion_weights_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            PragSettings(fusion={"weights": {"w1": 0.9, "w2": 0.9}})

    def test_rerank_output_cannot_exceed_input(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            RerankTierConfig(model="m", input_k=8, output_k=30, timeout_ms=70)

    def test_at_least_one_source_must_be_required(self) -> None:
        """Otherwise a total retrieval failure cannot be detected.

        Every leg being optional means every leg can drop and the request still looks healthy.
        """
        with pytest.raises(ValueError, match="must be required"):
            PragSettings(
                retrieval={
                    "sources": [SourceConfig(id="vector.primary", impl="pgvector", required=False)]
                }
            )

    def test_a_required_source_satisfies_the_rule(self) -> None:
        settings = PragSettings(
            retrieval={
                "sources": [
                    SourceConfig(id="vector.primary", impl="pgvector", required=True),
                    SourceConfig(id="lexical.primary", impl="opensearch", required=False),
                ]
            }
        )
        assert len(settings.retrieval.sources) == 2

    def test_parametric_requires_calibrated_evaluation(self) -> None:
        """The phase gate, enforced by the config rather than by discipline.

        Enabling the parametric tier without calibrated evaluation means there is no way to
        tell whether an adapter helped, which is the exact situation the roadmap prevents.
        """
        with pytest.raises(ValueError, match="judge_calibrated"):
            PragSettings(parametric={"enabled": True})

    def test_parametric_is_allowed_once_evaluation_is_calibrated(self) -> None:
        settings = PragSettings(parametric={"enabled": True}, evaluation={"judge_calibrated": True})
        assert settings.parametric.enabled

    def test_unknown_top_level_keys_are_refused(self) -> None:
        """A typo in a config file must fail startup, not be silently ignored."""
        with pytest.raises(ValueError, match="Extra inputs"):
            PragSettings(retreival={})  # type: ignore[call-arg]


class TestLoader:
    def test_layers_merge_deeply(self, tmp_path: Any) -> None:
        """A partial override must not delete its siblings."""
        defaults = tmp_path / "defaults.yaml"
        defaults.write_text(
            "context:\n  max_evidence_tokens: 4000\n  memory_tokens: 1000\n", encoding="utf-8"
        )
        env_file = tmp_path / "env.yaml"
        env_file.write_text("context:\n  max_evidence_tokens: 6000\n", encoding="utf-8")

        settings = load_settings(defaults_path=defaults, env_file_path=env_file)
        assert settings.context.max_evidence_tokens == 6000
        assert settings.context.memory_tokens == 1000, "sibling survived the partial override"

    def test_missing_files_are_not_an_error(self, tmp_path: Any) -> None:
        """A deployment without an env file is normal, not broken."""
        assert load_settings(defaults_path=tmp_path / "absent.yaml").config_version

    def test_malformed_yaml_fails_at_startup(self, tmp_path: Any) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("context:\n  - this is a list\n   bad indent:\n", encoding="utf-8")
        with pytest.raises(ConfigurationError):
            load_settings(defaults_path=bad)

    def test_non_mapping_yaml_is_refused(self, tmp_path: Any) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="mapping"):
            load_settings(defaults_path=bad)


class TestTenantResolution:
    def test_allowed_override_applies(self) -> None:
        policy = resolve_tenant_policy(PragSettings(), "tenant-a", {"guardrails.strict_mode": True})
        assert policy.strict_mode
        assert policy.abstains_on_irreconcilable_conflict

    def test_disallowed_override_is_refused_loudly(self) -> None:
        """Refusing beats ignoring.

        A tenant who believes they disabled a guardrail and was silently overruled has a false
        model of the system that nobody will correct, because from outside it looks like it
        worked.
        """
        with pytest.raises(ConfigurationError) as excinfo:
            resolve_tenant_policy(
                PragSettings(), "tenant-a", {"guardrails.tool_provenance_gate": False}
            )
        assert excinfo.value.context["disallowed"] == ["guardrails.tool_provenance_gate"]

    @pytest.mark.parametrize("key", sorted(ALLOWED_OVERRIDE_KEYS))
    def test_every_allow_listed_key_is_accepted(self, key: str) -> None:
        """The allow-list and the resolver must not drift apart."""
        sample: dict[str, Any] = {
            "routing.utility_weights": {"quality": 0.5, "latency": 0.3, "cost": 0.2},
            "guardrails.strict_mode": True,
            "context.max_evidence_tokens": 4000,
            "fusion.abstain_below_knowledge_score": 0.5,
            "caching.tiers.semantic.similarity_floor": 0.97,
            "parametric.enabled": False,
        }
        resolve_tenant_policy(PragSettings(), "tenant-a", {key: sample[key]})

    def test_isolation_and_guardrail_keys_are_not_overridable(self) -> None:
        """The allow-list is a security boundary, so its shape is worth asserting directly."""
        forbidden = {
            "guardrails.tool_provenance_gate",
            "guardrails.input",
            "caching.never_cache",
            "observability.retention_days",
            "evaluation.adversarial_pass_rate_min",
        }
        assert not (forbidden & ALLOWED_OVERRIDE_KEYS)

    def test_tenant_cannot_enable_parametric_the_platform_disabled(self) -> None:
        """A tenant may turn the tier off, never on.

        Enabling it requires calibrated evaluation, which is a platform-level fact a tenant is
        in no position to assert.
        """
        settings = PragSettings()
        assert not settings.parametric.enabled

        policy = resolve_tenant_policy(settings, "tenant-a", {"parametric.enabled": True})
        assert policy.parametric_enabled is False

    def test_tenant_can_disable_parametric_the_platform_enabled(self) -> None:
        settings = PragSettings(parametric={"enabled": True}, evaluation={"judge_calibrated": True})
        policy = resolve_tenant_policy(settings, "tenant-a", {"parametric.enabled": False})
        assert policy.parametric_enabled is False

    def test_config_version_reaches_the_policy(self) -> None:
        """It is part of every cache key, so a config change invalidates caches automatically."""
        settings = PragSettings(config_version="2026.10.01-2")
        assert resolve_tenant_policy(settings, "t").config_version == "2026.10.01-2"

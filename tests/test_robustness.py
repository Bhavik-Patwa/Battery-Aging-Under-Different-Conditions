import copy
import hashlib
import json
import math

from pathlib import Path
from collections import Counter

import pytest

from scripts.assess_robustness import (
    build_model_family_comparisons,
    build_prediction_agreement,
    calculate_file_sha256,
    canonical_json,
    validate_policy,
    weighted_correlation,
    PREDICTION_INTERVAL_DIAGNOSTIC_FIELDS,
    HORIZON_EVALUATION_SENSITIVITY_FIELDS,
    INCIDENT_EVALUATION_SENSITIVITY_FIELDS,
    PREDICTIVE_SENSITIVITY_SUMMARY_FIELDS
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Loading the registered robustness-assessment policy
def load_policy() -> dict:
    return json.loads(
        (
            PROJECT_ROOT
            / "configs"
            / "robustness_assessment_policy.json"
        ).read_text(
            encoding = "utf-8"
        )
    )


# Confirming the registered policy passes validation
def test_registered_policy_passes_validation():
    validate_policy(
        load_policy()
    )


# Rejecting a policy that authorizes causal interpretation
def test_policy_rejects_causal_interpretation():
    policy = copy.deepcopy(
        load_policy()
    )

    policy[
        "scientific_scope"
    ][
        "causal_interpretation_authorized"
    ] = True

    with pytest.raises(
        RuntimeError,
        match = "unsupported interpretation"
    ):
        validate_policy(
            policy
        )


# Rejecting a policy that authorizes a performance-selected winner
def test_policy_rejects_performance_selected_winner():
    policy = copy.deepcopy(
        load_policy()
    )

    policy[
        "scientific_scope"
    ][
        "performance_selected_model_winner_authorized"
    ] = True

    with pytest.raises(
        RuntimeError,
        match = "unsupported interpretation"
    ):
        validate_policy(
            policy
        )


# Confirming the policy binds the current upstream input reports
@pytest.mark.production_artifacts
def test_policy_binds_current_input_reports():
    policy = load_policy()

    paths = {
        "experimental_design": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "experimental_design"
            / "experimental_design_report.json"
        ),
        "causal_identification": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "causal_identification"
            / "causal_identification_report.json"
        ),
        "descriptive_modeling": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "descriptive_models"
            / "descriptive_modeling_report.json"
        ),
        "bayesian_modeling": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "bayesian_models"
            / "bayesian_modeling_report.json"
        ),
        "predictive_modeling": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "predictive_models"
            / "predictive_benchmarking_report.json"
        ),
        "predictive_sensitivity": (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "predictive_sensitivity"
            / "predictive_sensitivity_report.json"
        )
    }

    for input_name, path in paths.items():
        assert calculate_file_sha256(
            path
        ) == policy[
            "inputs"
        ][
            "required_report_sha256"
        ][
            input_name
        ]


# Confirming weighted correlation is one for identical series
def test_weighted_correlation_is_one_for_identical_values():
    correlation = weighted_correlation(
        [
            1.0,
            2.0,
            3.0
        ],
        [
            1.0,
            2.0,
            3.0
        ],
        [
            1.0,
            1.0,
            1.0
        ]
    )

    assert correlation is not None
    assert math.isclose(
        correlation,
        1.0
    )


# Confirming weighted correlation is unavailable without variation
def test_weighted_correlation_is_unavailable_without_variation():
    assert weighted_correlation(
        [
            1.0,
            1.0
        ],
        [
            2.0,
            3.0
        ],
        [
            1.0,
            1.0
        ]
    ) is None


# Confirming family-comparison differences preserve direction
def test_model_family_comparison_preserves_direction():
    predictive_tables = {
        "candidate_registry": [
            {
                "benchmark_identity": "first",
                "candidate_identity": "candidate",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined_within_scope",
                "aging_mode": "cycle_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "model_family": "elastic_net",
                "analytical_status": "available"
            },
            {
                "benchmark_identity": "second",
                "candidate_identity": "candidate",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined_within_scope",
                "aging_mode": "cycle_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "model_family": "random_forest",
                "analytical_status": "available"
            }
        ],
        "validation_metrics": [
            {
                "benchmark_identity": "first",
                "validation_status": "completed",
                "equal_cell_weighted_root_mean_squared_error": "0.02",
                "equal_cell_weighted_mean_absolute_error": "0.01",
                "equal_cell_weighted_predictive_r_squared": "0.8"
            },
            {
                "benchmark_identity": "second",
                "validation_status": "completed",
                "equal_cell_weighted_root_mean_squared_error": "0.01",
                "equal_cell_weighted_mean_absolute_error": "0.005",
                "equal_cell_weighted_predictive_r_squared": "0.9"
            }
        ]
    }

    records = build_model_family_comparisons(
        predictive_tables
    )

    assert len(
        records
    ) == 1

    assert math.isclose(
        records[
            0
        ][
            "root_mean_squared_error_difference"
        ],
        -0.01
    )

    assert records[
        0
    ][
        "comparison_status"
    ] == "completed"


# Confirming prediction agreement is unavailable without shared observations
def test_prediction_agreement_requires_shared_observations():
    bayesian_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "candidate",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined_within_scope",
                "aging_mode": "cycle_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "posterior_analytical_status": "available"
            }
        ],
        "grouped_validation_predictions": []
    }

    predictive_tables = {
        "candidate_registry": [
            {
                "benchmark_identity": "benchmark",
                "candidate_identity": "candidate",
                "model_family": "elastic_net",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined_within_scope",
                "aging_mode": "cycle_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "analytical_status": "available"
            }
        ],
        "validation_predictions": []
    }

    records = build_prediction_agreement(
        bayesian_tables,
        predictive_tables
    )

    assert len(
        records
    ) == 1

    assert records[
        0
    ][
        "assessment_status"
    ] == "unavailable"

    assert records[
        0
    ][
        "assessment_reason_code"
    ] == "NO_SHARED_VALIDATION_PREDICTIONS"


# Confirming external validity does not authorize policy use
def test_external_validity_does_not_authorize_policy_use():
    policy = load_policy()

    statuses = {
        record[
            "domain_identity"
        ]: record[
            "applicability_status"
        ]
        for record in policy[
            "external_validity_domains"
        ]
    }

    assert statuses[
        "operating_policy_recommendations"
    ] == "not_authorized"

    assert statuses[
        "causal_treatment_effects"
    ] == "not_authorized"

    assert statuses[
        "unobserved_operating_conditions"
    ] == "unsupported"


# Confirming prior sensitivity keeps completed-without-classification status
def test_prior_sensitivity_recognizes_completed_without_binary_classification() -> None:
    from scripts.assess_robustness import build_prior_sensitivity

    bayesian_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "candidate",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined_within_scope",
                "aging_mode": "cycle_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days"
            }
        ],
        "prior_sensitivity_summaries": [
            {
                "candidate_identity": "candidate",
                "reference_prior_scale_multiplier": "1.0",
                "sensitivity_prior_scale_multiplier": "0.5",
                "sensitivity_comparison_status": (
                    "completed_without_binary_materiality_classification"
                ),
                "absolute_median_difference": "0.01",
                "relative_median_difference": "0.005"
            },
            {
                "candidate_identity": "candidate",
                "reference_prior_scale_multiplier": "1.0",
                "sensitivity_prior_scale_multiplier": "2.0",
                "sensitivity_comparison_status": (
                    "completed_without_binary_materiality_classification"
                ),
                "absolute_median_difference": "0.02",
                "relative_median_difference": "0.010"
            }
        ]
    }

    records = build_prior_sensitivity(
        bayesian_tables
    )

    assert len(
        records
    ) == 2

    for record in records:
        assert record[
            "assessment_status"
        ] == "completed"

        assert record[
            "median_absolute_median_difference"
        ] is not None

        assert record[
            "median_absolute_relative_difference"
        ] is not None


# Confirming partial prediction overlap is diagnostic only
def test_partial_inventory_overlap_is_diagnostic_only() -> None:
    from scripts.assess_robustness import build_prediction_agreement

    bayesian_tables = {
        "candidate_registry": [{
            "candidate_identity": "candidate",
            "design_scope": "Stage_1",
            "sampling_design_scope": "combined_within_scope",
            "aging_mode": "cycle_aging",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "posterior_analytical_status": "available"
        }],
        "grouped_validation_predictions": [
            {
                "candidate_identity": "candidate",
                "cell_identity": "cell_1",
                "source_identity": "source_1",
                "condition_protocol_identity": "condition_a",
                "horizon": "10.0",
                "observed_outcome": "0.95",
                "posterior_predictive_median": "0.94",
                "prediction_status": "completed"
            },
            {
                "candidate_identity": "candidate",
                "cell_identity": "cell_2",
                "source_identity": "source_2",
                "condition_protocol_identity": "condition_a",
                "horizon": "10.0",
                "observed_outcome": "0.93",
                "posterior_predictive_median": "0.92",
                "prediction_status": "completed"
            }
        ]
    }

    predictive_tables = {
        "candidate_registry": [{
            "benchmark_identity": "benchmark",
            "candidate_identity": "candidate",
            "model_family": "elastic_net",
            "design_scope": "Stage_1",
            "sampling_design_scope": "combined_within_scope",
            "aging_mode": "cycle_aging",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "analytical_status": "available"
        }],
        "validation_predictions": [{
            "benchmark_identity": "benchmark",
            "cell_identity": "cell_1",
            "source_identity": "source_1",
            "condition_protocol_identity": "condition_a",
            "horizon": "10.0",
            "observed_outcome": "0.95",
            "predicted_outcome": "0.945",
            "prediction_status": "completed"
        }]
    }

    records = build_prediction_agreement(
        bayesian_tables,
        predictive_tables
    )

    assert len(records) == 1
    assert records[0]["assessment_status"] == "diagnostic_only"
    assert records[0]["assessment_reason_code"] == "PARTIAL_INVENTORY_OVERLAP"


# Confirming family comparisons preserve diagnostic-only source status
def test_model_family_comparisons_preserve_diagnostic_status() -> None:
    from scripts.assess_robustness import build_model_family_comparisons

    predictive_tables = {
        "candidate_registry": [
            {
                "benchmark_identity": "first",
                "candidate_identity": "candidate",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined_within_scope",
                "aging_mode": "cycle_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "model_family": "elastic_net",
                "analytical_status": "available"
            },
            {
                "benchmark_identity": "second",
                "candidate_identity": "candidate",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined_within_scope",
                "aging_mode": "cycle_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "model_family": "random_forest",
                "analytical_status": "diagnostic_only"
            }
        ],
        "validation_metrics": [
            {
                "benchmark_identity": "first",
                "validation_status": "completed",
                "equal_cell_weighted_root_mean_squared_error": "0.02",
                "equal_cell_weighted_mean_absolute_error": "0.01",
                "equal_cell_weighted_predictive_r_squared": "0.8"
            },
            {
                "benchmark_identity": "second",
                "validation_status": "completed",
                "equal_cell_weighted_root_mean_squared_error": "0.01",
                "equal_cell_weighted_mean_absolute_error": "0.005",
                "equal_cell_weighted_predictive_r_squared": "0.9"
            }
        ]
    }

    records = build_model_family_comparisons(predictive_tables)

    assert len(records) == 1
    assert records[0]["comparison_status"] == "diagnostic_only"
    assert records[0]["comparison_reason_code"] == (
        "SOURCE_BENCHMARK_NOT_ANALYTICALLY_AVAILABLE"
    )


# Confirming structural limitations never reach completed status
def test_structural_limitations_are_never_completed() -> None:
    from scripts.assess_robustness import build_structural_limitations

    design_tables = {
        "analytical_scope": [
            {"stage_batch_relationship_status": "stage_2_used_a_new_battery_cell_batch_relative_to_stage_1"}
        ],
        "stage_overlap": [
            {"exact_overlap_count": "0"}
        ],
        "estimand_support": [
            {"net_calendar_aging_exposure_status": "unavailable"}
        ],
        "design_identifiability": [
            {
                "record_type": "matrix_summary",
                "context_family": "laboratory",
                "full_column_rank": "false"
            },
            {
                "record_type": "term",
                "context_family": "laboratory",
                "structurally_estimable": "false"
            }
        ]
    }

    data_quality_contract = {
        "feature_contract": {
            "dc_internal_resistance": {
                "status": "primary_approved_sensitivity_provisional"
            }
        }
    }

    records = build_structural_limitations(design_tables, data_quality_contract, [])

    for record in records:
        assert record["applicability_status"] in {"unsupported", "unavailable", "not_authorized"}
        assert record["applicability_status"] != "completed"

    by_identity = {r["limitation_identity"]: r for r in records}

    assert by_identity["stage_batch_aliasing"]["evidence_record_count"] == 1
    assert by_identity["no_exact_cross_stage_combinations"]["evidence_record_count"] == 1
    assert by_identity["laboratory_rank_deficient_matrices"]["evidence_record_count"] == 1
    assert by_identity["laboratory_non_estimable_terms"]["evidence_record_count"] == 1
    assert by_identity["net_storage_exposure_unavailable"]["evidence_record_count"] == 1
    assert by_identity["ocv_corrected_resistance_not_approved"]["evidence_record_count"] == 1
    assert by_identity["ocv_corrected_resistance_not_approved"]["evidence_source"] == "data_quality_contract"


# Confirming zero-evidence rows are marked as registered policy restrictions
def test_structural_limitations_zero_evidence_source_is_registered_policy_restriction() -> None:
    from scripts.assess_robustness import build_structural_limitations

    design_tables = {
        "analytical_scope": [],
        "stage_overlap": [],
        "estimand_support": [],
        "design_identifiability": []
    }

    data_quality_contract = {
        "feature_contract": {"dc_internal_resistance": {"status": "approved"}}
    }

    records = build_structural_limitations(design_tables, data_quality_contract, [])
    by_identity = {r["limitation_identity"]: r for r in records}

    for identity in (
        "stage_batch_aliasing",
        "no_exact_cross_stage_combinations",
        "net_storage_exposure_unavailable",
        "ocv_corrected_resistance_not_approved",
        "equipment_allocation_restrictions"
    ):
        assert by_identity[identity]["evidence_record_count"] == 0
        assert by_identity[identity]["evidence_source"] == "registered_policy_restriction"


# Confirming configuration boundary summary publishes every registered configuration with correct fold counts and boundary flags
def test_configuration_boundary_summary_publishes_all_registered_configs() -> None:
    from scripts.assess_robustness import (
        build_configuration_boundary_summary,
        canonical_json
    )

    test_policy = {
        "model_families": {
            "elastic_net": {
                "configurations": [
                    {"alpha": 0.1, "l1_ratio": 0.5},
                    {"alpha": 0.5, "l1_ratio": 0.5}
                ]
            }
        }
    }

    def identity_of(family, config):
        return hashlib.sha256(
            canonical_json({"model_family": family, "configuration": config}).encode("utf-8")
        ).hexdigest()

    cfg_a_id = identity_of("elastic_net", {"alpha": 0.1, "l1_ratio": 0.5})
    cfg_b_id = identity_of("elastic_net", {"alpha": 0.5, "l1_ratio": 0.5})

    predictive_tables = {
        "hyperparameter_selection": [
            {"model_family": "elastic_net", "candidate_identity": "c1",
             "configuration_identity": cfg_a_id,
             "configuration": {"alpha": 0.1, "l1_ratio": 0.5},
             "selected": "true", "outer_fold_index": "0"},
            {"model_family": "elastic_net", "candidate_identity": "c1",
             "configuration_identity": cfg_b_id,
             "configuration": {"alpha": 0.5, "l1_ratio": 0.5},
             "selected": "false", "outer_fold_index": "0"},
            {"model_family": "elastic_net", "candidate_identity": "c1",
             "configuration_identity": cfg_a_id,
             "configuration": {"alpha": 0.1, "l1_ratio": 0.5},
             "selected": "false", "outer_fold_index": "1"},
            {"model_family": "elastic_net", "candidate_identity": "c1",
             "configuration_identity": cfg_b_id,
             "configuration": {"alpha": 0.5, "l1_ratio": 0.5},
             "selected": "true", "outer_fold_index": "1"}
        ]
    }

    records = build_configuration_boundary_summary(predictive_tables, test_policy)
    by_id = {r["configuration_identity"]: r for r in records}

    assert len(records) == 2
    assert by_id[cfg_a_id]["total_outer_folds"] == 2
    assert by_id[cfg_a_id]["selection_count"] == 1
    assert by_id[cfg_a_id]["selected_on_any_outer_fold"] is True
    assert by_id[cfg_a_id]["on_registered_search_boundary"] is None
    assert by_id[cfg_b_id]["selection_count"] == 1


# Confirming zero-selection configurations still appear
def test_configuration_boundary_publishes_zero_selection_configurations() -> None:
    from scripts.assess_robustness import build_configuration_boundary_summary

    test_policy = {
        "model_families": {
            "random_forest": {
                "configurations": [
                    {"max_depth": 4, "min_samples_leaf": 1},
                    {"max_depth": 8, "min_samples_leaf": 1}
                ]
            }
        }
    }

    predictive_tables = {"hyperparameter_selection": []}

    records = build_configuration_boundary_summary(predictive_tables, test_policy)

    assert len(records) == 2
    for record in records:
        assert record["selection_count"] == 0
        assert record["selected_on_any_outer_fold"] is False
        assert record["selection_fraction"] == 0.0


# Rejecting a coverage record with a non-terminal domain status
def test_domain_coverage_requires_registered_terminal_status() -> None:
    from scripts.assess_robustness import (
        build_assessment_coverage,
        domain_evaluation
    )

    policy = load_policy()

    domain_evaluations = {
        domain["assessment_identity"]: domain_evaluation(
            status = "completed",
            total_count = 1,
            completed_count = 1,
            restricted_count = 0,
            unavailable_count = 0,
            reason_code = None
        )
        for domain in policy["assessment_domains"]
    }

    # Inject one non-terminal status
    first_identity = policy["assessment_domains"][0]["assessment_identity"]
    domain_evaluations[first_identity] = domain_evaluation(
        status = "in_progress",
        total_count = 1,
        completed_count = 0,
        restricted_count = 1,
        unavailable_count = 0,
        reason_code = "PARTIAL_EVIDENCE_COVERAGE"
    )

    with pytest.raises(
        RuntimeError,
        match = "non-terminal status"
    ):
        build_assessment_coverage(policy, {}, domain_evaluations)


# Rejecting a coverage record missing a registered domain
def test_domain_coverage_requires_every_registered_domain() -> None:
    from scripts.assess_robustness import (
        build_assessment_coverage,
        domain_evaluation
    )

    policy = load_policy()

    # Missing one domain entirely
    domain_evaluations = {
        domain["assessment_identity"]: domain_evaluation(
            status = "completed",
            total_count = 1,
            completed_count = 1,
            restricted_count = 0,
            unavailable_count = 0,
            reason_code = None
        )
        for domain in policy["assessment_domains"][:-1]
    }

    with pytest.raises(
        RuntimeError,
        match = "was not evaluated"
    ):
        build_assessment_coverage(policy, {}, domain_evaluations)


# Confirming the policy registers every required status and field
def test_policy_registers_all_required_robustness_fields() -> None:
    policy = load_policy()

    terminal = set(policy["assessment_contract"]["terminal_domain_statuses"])
    assert terminal == {
        "completed",
        "completed_with_restrictions",
        "diagnostic_only",
        "unavailable",
        "unsupported",
        "not_authorized"
    }

    matching_fields = set(policy["assessment_contract"]["cohort_sensitivity_matching_fields"])

    for required in (
        "design_scope",
        "sampling_design_scope",
        "aging_mode",
        "outcome",
        "horizon_axis",
        "model_identity",
        "condition_identities",
        "support_interval_minimum",
        "support_interval_maximum"
    ):
        assert required in matching_fields

    assert set(policy["output_contract"]["tables"]) == {
        "validation_summary",
        "model_family_comparisons",
        "prior_sensitivity",
        "influence_summary",
        "prediction_agreement",
        "cohort_sensitivity",
        "descriptive_specification_comparisons",
        "structural_limitations",
        "configuration_boundary_summary",
        "fragility_registry",
        "prediction_interval_diagnostics",
        "horizon_evaluation_sensitivity",
        "incident_evaluation_sensitivity",
        "unsupported_condition_inventory",
        "assessment_coverage",
        "external_validity",
        "predictive_sensitivity_summary"
    }

    for required_reason in (
        "COHORT_COUNTERPART_UNAVAILABLE",
        "COHORT_MATCHING_KEY_AMBIGUOUS",
        "PARTIAL_INVENTORY_OVERLAP",
        "NET_STORAGE_EXPOSURE_UNAVAILABLE",
        "OCV_CORRECTED_RESISTANCE_PROVISIONAL"
    ):
        assert required_reason in policy["assessment_contract"]["registered_reason_codes"]


# Confirming exact primary/sensitivity matching produces a completed comparison
def test_cohort_sensitivity_exact_match_completes() -> None:
    from scripts.assess_robustness import build_cohort_sensitivity

    policy = load_policy()

    design_scope = "Stage_1"
    sampling_scope = "combined_within_scope"
    aging_mode = "calendar_aging"
    outcome = "relative_capacity"
    horizon_axis = "elapsed_calendar_days"
    model_name = "linear_common_trajectory"

    descriptive_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "primary_candidate",
                "design_scope": design_scope,
                "sampling_design_scope": sampling_scope,
                "aging_mode": aging_mode,
                "outcome": outcome,
                "cohort": "primary",
                "horizon_axis": horizon_axis,
                "template_name": model_name,
                "condition_protocol_identities": ["a", "b"],
                "support_interval_minimum": "0.0",
                "support_interval_maximum": "100.0",
                "support_interval_lower_inclusive": "true",
                "support_interval_upper_inclusive": "true",
                "analytical_result_status": "available"
            },
            {
                "candidate_identity": "sensitivity_candidate",
                "design_scope": design_scope,
                "sampling_design_scope": sampling_scope,
                "aging_mode": aging_mode,
                "outcome": outcome,
                "cohort": "sensitivity",
                "horizon_axis": horizon_axis,
                "template_name": model_name,
                "condition_protocol_identities": ["a", "b"],
                "support_interval_minimum": "0.0",
                "support_interval_maximum": "100.0",
                "support_interval_lower_inclusive": "true",
                "support_interval_upper_inclusive": "true",
                "analytical_result_status": "available"
            }
        ],
        "validation_metrics": [
            {
                "candidate_identity": "primary_candidate",
                "validation_status": "completed",
                "root_mean_squared_error": "0.01",
                "mean_absolute_error": "0.005"
            },
            {
                "candidate_identity": "sensitivity_candidate",
                "validation_status": "completed",
                "root_mean_squared_error": "0.012",
                "mean_absolute_error": "0.006"
            }
        ],
        "validation_predictions": [
            {"candidate_identity": "primary_candidate", "cell_identity": "c1",
             "source_identity": "s1", "condition_protocol_identity": "a", "horizon": "10.0"},
            {"candidate_identity": "primary_candidate", "cell_identity": "c2",
             "source_identity": "s2", "condition_protocol_identity": "b", "horizon": "20.0"},
            {"candidate_identity": "sensitivity_candidate", "cell_identity": "c1",
             "source_identity": "s1", "condition_protocol_identity": "a", "horizon": "10.0"},
            {"candidate_identity": "sensitivity_candidate", "cell_identity": "c2",
             "source_identity": "s2", "condition_protocol_identity": "b", "horizon": "20.0"}
        ]
    }

    records = build_cohort_sensitivity(
        descriptive_tables,
        {"candidate_registry": [], "grouped_validation_metrics": []},
        {"candidate_registry": [], "validation_metrics": []},
        policy
    )

    assert len(records) == 1
    assert records[0]["comparison_status"] == "completed"
    assert math.isclose(records[0]["root_mean_squared_error_difference"], 0.002)


# Confirming mismatched support intervals prohibit cohort comparison
def test_cohort_sensitivity_unmatched_counterparts_remain_explicit() -> None:
    from scripts.assess_robustness import build_cohort_sensitivity

    policy = load_policy()

    design_scope = "Stage_1"
    sampling_scope = "combined_within_scope"
    aging_mode = "calendar_aging"
    outcome = "relative_capacity"
    horizon_axis = "elapsed_calendar_days"
    model_name = "linear_common_trajectory"

    descriptive_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "primary_candidate",
                "design_scope": design_scope,
                "sampling_design_scope": sampling_scope,
                "aging_mode": aging_mode,
                "outcome": outcome,
                "cohort": "primary",
                "horizon_axis": horizon_axis,
                "template_name": model_name,
                "estimator": "weighted_least_squares",
                "specification": "main_effects",
                "context_family": "no_additional_context",
                "condition_protocol_identities": ["a", "b"],
                "support_interval_minimum": "0.0",
                "support_interval_maximum": "100.0",
                "support_interval_lower_inclusive": "true",
                "support_interval_upper_inclusive": "true",
                "analytical_result_status": "available"
            },
            {
                "candidate_identity": "sensitivity_candidate",
                "design_scope": design_scope,
                "sampling_design_scope": sampling_scope,
                "aging_mode": aging_mode,
                "outcome": outcome,
                "cohort": "sensitivity",
                "horizon_axis": horizon_axis,
                "template_name": model_name,
                "estimator": "weighted_least_squares",
                "specification": "main_effects",
                "context_family": "no_additional_context",
                "condition_protocol_identities": ["a", "b"],
                "support_interval_minimum": "0.0",
                "support_interval_maximum": "80.0",
                "support_interval_lower_inclusive": "true",
                "support_interval_upper_inclusive": "true",
                "analytical_result_status": "available"
            }
        ],
        "validation_metrics": [
            {
                "candidate_identity": "primary_candidate",
                "validation_status": "completed",
                "root_mean_squared_error": "0.01",
                "mean_absolute_error": "0.005"
            },
            {
                "candidate_identity": "sensitivity_candidate",
                "validation_status": "completed",
                "root_mean_squared_error": "0.012",
                "mean_absolute_error": "0.006"
            }
        ],
        "validation_predictions": []
    }

    records = build_cohort_sensitivity(
        descriptive_tables,
        {"candidate_registry": [], "grouped_validation_metrics": []},
        {"candidate_registry": [], "validation_metrics": []},
        policy
    )

    assert len(records) == 2

    for record in records:
        assert record["comparison_status"] == "unavailable"
        assert record["comparison_reason_code"] == "COHORT_COUNTERPART_UNAVAILABLE"
        assert (
            record["primary_record_count"] + record["sensitivity_record_count"]
        ) == 1


# Confirming ambiguous duplicate cohort matches are unavailable, never first-record selected
def test_cohort_sensitivity_ambiguous_duplicates_are_unavailable() -> None:
    from scripts.assess_robustness import build_cohort_sensitivity

    policy = load_policy()

    design_scope = "Stage_1"
    sampling_scope = "combined_within_scope"
    aging_mode = "calendar_aging"
    outcome = "relative_capacity"
    horizon_axis = "elapsed_calendar_days"
    model_name = "linear_common_trajectory"

    def base_candidate(identity, cohort):
        return {
            "candidate_identity": identity,
            "design_scope": design_scope,
            "sampling_design_scope": sampling_scope,
            "aging_mode": aging_mode,
            "outcome": outcome,
            "cohort": cohort,
            "horizon_axis": horizon_axis,
            "template_name": model_name,
            "estimator": "weighted_least_squares",
            "specification": "main_effects",
            "context_family": "no_additional_context",
            "condition_protocol_identities": ["a", "b"],
            "support_interval_minimum": "0.0",
            "support_interval_maximum": "100.0",
            "support_interval_lower_inclusive": "true",
            "support_interval_upper_inclusive": "true",
            "analytical_result_status": "available"
        }

    descriptive_tables = {
        "candidate_registry": [
            base_candidate("p1", "primary"),
            base_candidate("p2", "primary"),
            base_candidate("s1", "sensitivity")
        ],
        "validation_metrics": [
            {"candidate_identity": "p1", "validation_status": "completed",
             "root_mean_squared_error": "0.01", "mean_absolute_error": "0.005"},
            {"candidate_identity": "p2", "validation_status": "completed",
             "root_mean_squared_error": "0.011", "mean_absolute_error": "0.006"},
            {"candidate_identity": "s1", "validation_status": "completed",
             "root_mean_squared_error": "0.012", "mean_absolute_error": "0.007"}
        ],
        "validation_predictions": []
    }

    records = build_cohort_sensitivity(
        descriptive_tables,
        {"candidate_registry": [], "grouped_validation_metrics": []},
        {"candidate_registry": [], "validation_metrics": []},
        policy
    )

    assert len(records) == 1
    assert records[0]["comparison_status"] == "unavailable"
    assert records[0]["comparison_reason_code"] == "COHORT_MATCHING_KEY_AMBIGUOUS"
    assert records[0]["primary_record_count"] == 2
    assert records[0]["sensitivity_record_count"] == 1


# Confirming prediction agreement publishes inventory diagnostics
def test_prediction_agreement_publishes_inventory_diagnostics() -> None:
    from scripts.assess_robustness import build_prediction_agreement

    bayesian_tables = {
        "candidate_registry": [{
            "candidate_identity": "candidate",
            "design_scope": "Stage_1",
            "sampling_design_scope": "combined_within_scope",
            "aging_mode": "cycle_aging",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "posterior_analytical_status": "available"
        }],
        "grouped_validation_predictions": [
            {
                "candidate_identity": "candidate",
                "cell_identity": "cell_1",
                "source_identity": "source_1",
                "condition_protocol_identity": "condition_a",
                "horizon": "10.0",
                "observed_outcome": "0.95",
                "posterior_predictive_median": "0.94",
                "prediction_status": "completed"
            }
        ]
    }

    predictive_tables = {
        "candidate_registry": [{
            "benchmark_identity": "benchmark",
            "candidate_identity": "candidate",
            "model_family": "elastic_net",
            "design_scope": "Stage_1",
            "sampling_design_scope": "combined_within_scope",
            "aging_mode": "cycle_aging",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "analytical_status": "available"
        }],
        "validation_predictions": [
            {
                "benchmark_identity": "benchmark",
                "cell_identity": "cell_1",
                "source_identity": "source_1",
                "condition_protocol_identity": "condition_a",
                "horizon": "10.0",
                "observed_outcome": "0.95",
                "predicted_outcome": "0.945",
                "prediction_status": "completed"
            },
            {
                "benchmark_identity": "benchmark",
                "cell_identity": "cell_2",
                "source_identity": "source_2",
                "condition_protocol_identity": "condition_a",
                "horizon": "10.0",
                "observed_outcome": "0.93",
                "predicted_outcome": "0.92",
                "prediction_status": "completed"
            }
        ]
    }

    records = build_prediction_agreement(
        bayesian_tables, predictive_tables
    )

    assert len(records) == 1
    record = records[0]

    assert record["bayesian_prediction_count"] == 1
    assert record["predictive_prediction_count"] == 2
    assert record["shared_prediction_count"] == 1
    assert math.isclose(
        record["shared_inventory_fraction_bayesian"], 1.0
    )
    assert math.isclose(
        record["shared_inventory_fraction_predictive"], 0.5
    )
    assert record["exact_inventory_match"] is False
    assert record["assessment_status"] == "diagnostic_only"
    assert record["assessment_reason_code"] == "PARTIAL_INVENTORY_OVERLAP"


# Confirming the post-publication validator rejects a row-count tamper
def test_published_report_tampering_is_rejected(tmp_path: Path) -> None:
    from scripts.assess_robustness import (
        ASSESSMENT_COVERAGE_FIELDS,
        COHORT_SENSITIVITY_FIELDS,
        CONFIGURATION_BOUNDARY_FIELDS,
        DESCRIPTIVE_SPECIFICATION_COMPARISON_FIELDS,
        EXTERNAL_VALIDITY_FIELDS,
        FRAGILITY_REGISTRY_FIELDS,
        INFLUENCE_SUMMARY_FIELDS,
        MODEL_FAMILY_COMPARISON_FIELDS,
        PREDICTION_AGREEMENT_FIELDS,
        PRIOR_SENSITIVITY_FIELDS,
        STRUCTURAL_LIMITATION_FIELDS,
        UNSUPPORTED_CONDITION_FIELDS,
        VALIDATION_SUMMARY_FIELDS,
        validate_published_robustness_directory,
        write_csv_table
    )

    policy = load_policy()

    empty_tables = {
        "validation_summary": [],
        "model_family_comparisons": [],
        "prior_sensitivity": [],
        "influence_summary": [],
        "prediction_agreement": [],
        "cohort_sensitivity": [],
        "descriptive_specification_comparisons": [],
        "structural_limitations": [],
        "configuration_boundary_summary": [],
        "fragility_registry": [],
        "unsupported_condition_inventory": [],
        "prediction_interval_diagnostics": [],
        "horizon_evaluation_sensitivity": [],
        "incident_evaluation_sensitivity": [],
        "assessment_coverage": [
            {
                "assessment_identity": record["assessment_identity"],
                "evidence_source": record["evidence_source"],
                "evidence_unit": record["evidence_unit"],
                "total_evidence_count": 0,
                "completed_evidence_count": 0,
                "restricted_evidence_count": 0,
                "unavailable_evidence_count": 0,
                "assessment_status": "unavailable",
                "assessment_reason_code": "NO_EVIDENCE",
                "interpretation": record["interpretation"],
            }
            for record in policy["assessment_domains"]
        ],
        "external_validity": [
            dict(r) for r in policy["external_validity_domains"]
        ],
        "predictive_sensitivity_summary": []
    }

    field_map = {
        "validation_summary": VALIDATION_SUMMARY_FIELDS,
        "model_family_comparisons": MODEL_FAMILY_COMPARISON_FIELDS,
        "prior_sensitivity": PRIOR_SENSITIVITY_FIELDS,
        "influence_summary": INFLUENCE_SUMMARY_FIELDS,
        "prediction_agreement": PREDICTION_AGREEMENT_FIELDS,
        "cohort_sensitivity": COHORT_SENSITIVITY_FIELDS,
        "descriptive_specification_comparisons": DESCRIPTIVE_SPECIFICATION_COMPARISON_FIELDS,
        "structural_limitations": STRUCTURAL_LIMITATION_FIELDS,
        "configuration_boundary_summary": CONFIGURATION_BOUNDARY_FIELDS,
        "fragility_registry": FRAGILITY_REGISTRY_FIELDS,
        "unsupported_condition_inventory": UNSUPPORTED_CONDITION_FIELDS,
        "assessment_coverage": ASSESSMENT_COVERAGE_FIELDS,
        "external_validity": EXTERNAL_VALIDITY_FIELDS,
        "prediction_interval_diagnostics": PREDICTION_INTERVAL_DIAGNOSTIC_FIELDS,
        "horizon_evaluation_sensitivity": HORIZON_EVALUATION_SENSITIVITY_FIELDS,
        "incident_evaluation_sensitivity": INCIDENT_EVALUATION_SENSITIVITY_FIELDS,
        "predictive_sensitivity_summary": PREDICTIVE_SENSITIVITY_SUMMARY_FIELDS
    }

    table_manifests = {}

    for table_name, records in empty_tables.items():
        table_path = tmp_path / f"{table_name}.csv"
        write_csv_table(table_path, field_map[table_name], records)
        table_manifests[table_name] = {
            "file_name": table_path.name,
            "row_count": len(records),
            "sha256": calculate_file_sha256(table_path),
            "schema": [{"name": field_name} for field_name in field_map[table_name]]
        }

    domain_status_counts = {
        "unavailable": len(policy["assessment_domains"])
    }

    report = {
        "execution_status": "completed",
        "scientific_assessment_status": "restricted_evidence_summary_only",
        "policy_name": policy["policy_name"],
        "policy_version": policy["policy_version"],
        "policy_sha256": calculate_file_sha256(
            PROJECT_ROOT / "configs" / "robustness_assessment_policy.json"
        ),
        "implementation_sha256": calculate_file_sha256(
            Path(__import__("scripts.assess_robustness", fromlist = ["__file__"]).__file__).resolve()
        ),
        "input_hashes": {},
        "causal_interpretation_authorized": False,
        "policy_recommendation_authorized": False,
        "performance_selected_model_winner_declared": False,
        "magnitude_dependent_robustness_classification_used": False,
        "registration_status": policy["registration_status"],
        "amendment": policy["amendment"],
        "domain_status_counts": domain_status_counts,
        "prediction_inventory_reconciliation": {},
        "analytical_handoff": {
            "status": "restricted_evidence_summary_only",
            "fully_assessed_domain_count": 0,
            "restricted_or_unavailable_domain_count": len(policy["assessment_domains"]),
            "causal_interpretation_authorized": False,
            "policy_recommendation_authorized": False,
            "treatment_optimization_authorized": False,
            "cross_stage_transport_authorized": False
        },
        "tables": table_manifests
    }

    (tmp_path / "robustness_assessment_report.json").write_text(
        json.dumps(report, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8"
    )

    validate_published_robustness_directory(
        published_directory = tmp_path,
        expected_policy = policy,
        expected_policy_sha256 = report["policy_sha256"],
        expected_input_hashes = {},
        expected_tables = empty_tables,
        expected_assessment_status = "restricted_evidence_summary_only",
        expected_assessment_status_counts = domain_status_counts,
        expected_prediction_inventory_reconciliation = {}
    )

    tampered_path = tmp_path / "cohort_sensitivity.csv"
    with tampered_path.open("a", encoding = "utf-8", newline = "") as handle:
        handle.write(
            "EXTRA,EXTRA,EXTRA,EXTRA,EXTRA,EXTRA,EXTRA,0,available,available,completed,completed,,,,,,,unavailable,VALIDATION_METRICS_UNAVAILABLE,descriptive_cohort_sensitivity\n"
        )

    with pytest.raises(RuntimeError, match = "does not match"):
        validate_published_robustness_directory(
            published_directory = tmp_path,
            expected_policy = policy,
            expected_policy_sha256 = report["policy_sha256"],
            expected_input_hashes = {},
            expected_tables = empty_tables,
            expected_assessment_status = "restricted_evidence_summary_only",
            expected_assessment_status_counts = domain_status_counts,
            expected_prediction_inventory_reconciliation = {}
        )


# Confirming descriptive specification comparisons preserve upstream unavailable status
def test_descriptive_specification_comparisons_preserve_unavailable_status() -> None:
    from scripts.assess_robustness import (
        build_descriptive_specification_comparisons
    )

    descriptive_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "first",
                "analytical_result_status": "available"
            },
            {
                "candidate_identity": "second",
                "analytical_result_status": "available"
            }
        ],
        "model_comparisons": [
            {
                "first_candidate_identity": "first",
                "second_candidate_identity": "second",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "specification": "main_effects",
                "context_family": "no_additional_context",
                "first_template_name": "linear_common_trajectory",
                "second_template_name": "mixed_linear_common_trajectory",
                "first_root_mean_squared_error": "",
                "second_root_mean_squared_error": "",
                "first_mean_absolute_error": "",
                "second_mean_absolute_error": "",
                "comparison_status": "unavailable",
                "comparison_reason_code": "OUT_OF_FOLD_POPULATIONS_DIFFER"
            }
        ]
    }

    records = build_descriptive_specification_comparisons(
        descriptive_tables
    )

    assert len(records) == 1
    assert records[0]["comparison_status"] == "unavailable"
    assert records[0]["comparison_reason_code"] == "OUT_OF_FOLD_POPULATIONS_DIFFER"


# Confirming descriptive specification comparisons downgrade diagnostic-only sources
def test_descriptive_specification_comparisons_downgrade_diagnostic_source() -> None:
    from scripts.assess_robustness import (
        build_descriptive_specification_comparisons
    )

    descriptive_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "first",
                "analytical_result_status": "diagnostic_only"
            },
            {
                "candidate_identity": "second",
                "analytical_result_status": "available"
            }
        ],
        "model_comparisons": [
            {
                "first_candidate_identity": "first",
                "second_candidate_identity": "second",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "specification": "main_effects",
                "context_family": "no_additional_context",
                "first_template_name": "linear_common_trajectory",
                "second_template_name": "mixed_linear_common_trajectory",
                "first_root_mean_squared_error": "0.02",
                "second_root_mean_squared_error": "0.01",
                "first_mean_absolute_error": "0.01",
                "second_mean_absolute_error": "0.005",
                "comparison_status": "completed",
                "comparison_reason_code": ""
            }
        ]
    }

    records = build_descriptive_specification_comparisons(
        descriptive_tables
    )

    assert len(records) == 1
    assert records[0]["comparison_status"] == "diagnostic_only"
    assert records[0]["comparison_reason_code"] == (
        "SOURCE_MODEL_NOT_ANALYTICALLY_AVAILABLE"
    )
    assert records[0]["first_analytical_status"] == "diagnostic_only"
    assert records[0]["second_analytical_status"] == "available"


# Confirming descriptive specification comparisons complete when both sources available
def test_descriptive_specification_comparisons_complete_when_both_available() -> None:
    from scripts.assess_robustness import (
        build_descriptive_specification_comparisons
    )

    descriptive_tables = {
        "candidate_registry": [
            {"candidate_identity": "first", "analytical_result_status": "available"},
            {"candidate_identity": "second", "analytical_result_status": "available"}
        ],
        "model_comparisons": [
            {
                "first_candidate_identity": "first",
                "second_candidate_identity": "second",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "specification": "main_effects",
                "context_family": "no_additional_context",
                "first_template_name": "linear_common_trajectory",
                "second_template_name": "mixed_linear_common_trajectory",
                "first_root_mean_squared_error": "0.02",
                "second_root_mean_squared_error": "0.01",
                "first_mean_absolute_error": "0.01",
                "second_mean_absolute_error": "0.005",
                "comparison_status": "completed",
                "comparison_reason_code": ""
            }
        ]
    }

    records = build_descriptive_specification_comparisons(
        descriptive_tables
    )

    assert len(records) == 1
    assert records[0]["comparison_status"] == "completed"
    assert records[0]["comparison_reason_code"] is None
    assert math.isclose(records[0]["root_mean_squared_error_difference"], -0.01)


# Confirming fragility registry emits no binary robust label
def test_fragility_registry_emits_no_binary_label() -> None:
    from scripts.assess_robustness import build_fragility_registry

    descriptive_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "desc_candidate",
                "final_disposition": "fitted",
                "analytical_result_status": "diagnostic_only",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days"
            }
        ],
        "validation_metrics": [
            {"candidate_identity": "desc_candidate", "validation_status": "completed"}
        ]
    }

    bayesian_tables = {"candidate_registry": [], "grouped_validation_metrics": []}
    predictive_tables = {"candidate_registry": [], "validation_metrics": []}

    tables = {
        "prior_sensitivity": [],
        "influence_summary": [
            {"candidate_identity": "desc_candidate", "assessment_status": "completed"}
        ],
        "prediction_agreement": []
    }

    records = build_fragility_registry(
        descriptive_tables,
        bayesian_tables,
        predictive_tables,
        [],
        tables
    )

    assert len(records) == 1
    record = records[0]

    assert record["source_analysis"] == "descriptive_modeling"
    assert record["analytical_status"] == "diagnostic_only"
    assert "analytical_diagnostic_only" in record["fragility_indicators"]
    assert record["fragility_indicator_count"] == 1
    assert record["estimate_label"] == "fragility_inspection"

    assert "robust" not in str(record).lower()
    assert "fragile" not in str(record).lower().replace("fragility", "")


# Confirming fragility registry distinguishes not-applicable from false
def test_fragility_registry_distinguishes_null_from_false() -> None:
    from scripts.assess_robustness import build_fragility_registry

    descriptive_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "desc_candidate",
                "final_disposition": "fitted",
                "analytical_result_status": "available",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days"
            }
        ],
        "validation_metrics": [
            {"candidate_identity": "desc_candidate", "validation_status": "completed"}
        ]
    }

    bayesian_tables = {"candidate_registry": [], "grouped_validation_metrics": []}
    predictive_tables = {"candidate_registry": [], "validation_metrics": []}

    tables = {
        "prior_sensitivity": [],
        "influence_summary": [
            {"candidate_identity": "desc_candidate", "assessment_status": "completed"}
        ],
        "prediction_agreement": []
    }

    records = build_fragility_registry(
        descriptive_tables,
        bayesian_tables,
        predictive_tables,
        [],
        tables
    )

    assert len(records) == 1
    record = records[0]

    assert record["validation_available"] is True
    assert record["cell_deletion_refit_complete"] is True
    assert record["prior_sensitivity_complete"] is None
    assert record["condition_calibration_complete"] is None
    assert record["predictive_benchmark_met"] is None
    assert record["cross_framework_inventory_complete"] is None
    assert record["equipment_restriction_present"] is None
    assert "prior_sensitivity_complete" not in record["applicable_indicators"]
    assert "condition_calibration_complete" not in record["applicable_indicators"]


# Confirming fragility registry attaches equipment restriction only through exact estimand decision identity
def test_fragility_registry_attaches_equipment_via_exact_estimand_mapping() -> None:
    from scripts.assess_robustness import build_fragility_registry

    bayesian_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "bayes_candidate",
                "admissibility_status": "admissible",
                "posterior_analytical_status": "available",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "sorted_estimand_decision_identities": '["est_a", "est_b"]'
            }
        ],
        "grouped_validation_metrics": [
            {"candidate_identity": "bayes_candidate", "validation_status": "completed"}
        ]
    }

    estimand_identification_rows = [
        {"estimand_decision_identity": "est_a", "equipment_restriction_present": "false"},
        {"estimand_decision_identity": "est_b", "equipment_restriction_present": "true"}
    ]

    descriptive_tables = {"candidate_registry": [], "validation_metrics": []}
    predictive_tables = {"candidate_registry": [], "validation_metrics": []}

    tables = {
        "prior_sensitivity": [],
        "influence_summary": [],
        "prediction_agreement": []
    }

    records = build_fragility_registry(
        descriptive_tables,
        bayesian_tables,
        predictive_tables,
        estimand_identification_rows,
        tables
    )

    assert len(records) == 1
    record = records[0]

    assert record["source_analysis"] == "bayesian_modeling"
    assert record["equipment_restriction_present"] is True
    assert "equipment_restriction_present" in record["applicable_indicators"]
    assert "equipment_restriction_present" in record["fragility_indicators"]


# Confirming that unknown estimand mappings leave equipment attribution null
def test_fragility_registry_unknown_estimand_returns_null() -> None:
    from scripts.assess_robustness import build_fragility_registry

    bayesian_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "bayes_candidate",
                "admissibility_status": "admissible",
                "posterior_analytical_status": "available",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "sorted_estimand_decision_identities": '["est_unknown"]'
            }
        ],
        "grouped_validation_metrics": [
            {"candidate_identity": "bayes_candidate", "validation_status": "completed"}
        ]
    }

    descriptive_tables = {"candidate_registry": [], "validation_metrics": []}
    predictive_tables = {"candidate_registry": [], "validation_metrics": []}

    tables = {
        "prior_sensitivity": [],
        "influence_summary": [],
        "prediction_agreement": []
    }

    records = build_fragility_registry(
        descriptive_tables,
        bayesian_tables,
        predictive_tables,
        [],
        tables
    )

    assert len(records) == 1
    record = records[0]
    assert record["equipment_restriction_present"] is None
    assert "equipment_restriction_present" not in record["applicable_indicators"]
    assert "equipment_restriction_present" not in record["fragility_indicators"]


# Confirming descriptive candidates never inherit scope-wide equipment
def test_fragility_registry_descriptive_equipment_is_null() -> None:
    from scripts.assess_robustness import build_fragility_registry

    descriptive_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "desc_candidate",
                "final_disposition": "fitted",
                "analytical_result_status": "available",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days"
            }
        ],
        "validation_metrics": [
            {"candidate_identity": "desc_candidate", "validation_status": "completed"}
        ]
    }

    estimand_identification_rows = [
        {"estimand_decision_identity": "est_a", "equipment_restriction_present": "true"}
    ]

    bayesian_tables = {"candidate_registry": [], "grouped_validation_metrics": []}
    predictive_tables = {"candidate_registry": [], "validation_metrics": []}

    tables = {
        "prior_sensitivity": [],
        "influence_summary": [
            {"candidate_identity": "desc_candidate", "assessment_status": "completed"}
        ],
        "prediction_agreement": []
    }

    records = build_fragility_registry(
        descriptive_tables,
        bayesian_tables,
        predictive_tables,
        estimand_identification_rows,
        tables
    )

    assert len(records) == 1
    assert records[0]["equipment_restriction_present"] is None


# Confirming sensitivity-cohort membership is not a fragility indicator
def test_fragility_registry_sensitivity_cohort_is_descriptive_metadata() -> None:
    from scripts.assess_robustness import build_fragility_registry

    descriptive_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "desc_candidate",
                "final_disposition": "fitted",
                "analytical_result_status": "available",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "sensitivity",
                "horizon_axis": "elapsed_calendar_days"
            }
        ],
        "validation_metrics": [
            {"candidate_identity": "desc_candidate", "validation_status": "completed"}
        ]
    }

    bayesian_tables = {"candidate_registry": [], "grouped_validation_metrics": []}
    predictive_tables = {"candidate_registry": [], "validation_metrics": []}

    tables = {
        "prior_sensitivity": [],
        "influence_summary": [
            {"candidate_identity": "desc_candidate", "assessment_status": "completed"}
        ],
        "prediction_agreement": []
    }

    records = build_fragility_registry(
        descriptive_tables,
        bayesian_tables,
        predictive_tables,
        [],
        tables
    )

    assert len(records) == 1
    record = records[0]

    assert record["sensitivity_cohort"] is True
    assert "sensitivity_cohort_only" not in record
    assert "sensitivity_cohort_support_only" not in record["fragility_indicators"]
    assert "sensitivity_cohort" not in record["applicable_indicators"]
    assert record["fragility_indicator_count"] == 0


# Rejecting a policy that declares a purely prospective registration alongside a result-informed amendment
def test_policy_rejects_registration_amendment_contradiction() -> None:
    policy = copy.deepcopy(load_policy())
    policy["registration_status"] = "prospective_for_robustness_classification"

    with pytest.raises(
        RuntimeError,
        match = "Prospective registration_status cannot accompany a result-informed amendment"
    ):
        validate_policy(policy)


# Rejecting a policy that omits the direct-input hash bindings
def test_policy_rejects_missing_direct_input_binding() -> None:
    policy = copy.deepcopy(load_policy())
    del policy["inputs"]["required_file_sha256"]["data_quality_contract"]

    with pytest.raises(
        RuntimeError,
        match = "required_file_sha256"
    ):
        validate_policy(policy)


# Rejecting an unregistered emitted reason code
def test_emitted_reason_codes_are_validated() -> None:
    from scripts.assess_robustness import validate_emitted_codes_and_statuses

    policy = load_policy()

    tables = {
        "validation_summary": [
            {"assessment_status": "completed"}
        ],
        "model_family_comparisons": [
            {"comparison_status": "completed", "comparison_reason_code": "NOT_REGISTERED_CODE"}
        ],
        "prior_sensitivity": [],
        "influence_summary": [],
        "prediction_agreement": [],
        "cohort_sensitivity": [],
        "descriptive_specification_comparisons": [],
        "structural_limitations": []
    }

    with pytest.raises(RuntimeError, match = "unregistered reason code"):
        validate_emitted_codes_and_statuses(tables, policy)


# Rejecting an unregistered emitted status
def test_emitted_status_values_are_validated() -> None:
    from scripts.assess_robustness import validate_emitted_codes_and_statuses

    policy = load_policy()

    tables = {
        "validation_summary": [
            {"assessment_status": "in_progress"}
        ],
        "model_family_comparisons": [],
        "prior_sensitivity": [],
        "influence_summary": [],
        "prediction_agreement": [],
        "cohort_sensitivity": [],
        "descriptive_specification_comparisons": [],
        "structural_limitations": []
    }

    with pytest.raises(RuntimeError, match = "unregistered status"):
        validate_emitted_codes_and_statuses(tables, policy)


# Confirming a summary with zero completed validations is reported as unavailable
def test_partial_validation_summary_is_restricted() -> None:
    from scripts.assess_robustness import build_validation_summary

    base_candidate = {
        "candidate_identity": "c1",
        "template_name": "linear_common_trajectory",
        "design_scope": "Stage_1",
        "sampling_design_scope": "combined_within_scope",
        "aging_mode": "calendar_aging",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": "elapsed_calendar_days",
        "analytical_result_status": "available"
    }

    descriptive_tables = {
        "candidate_registry": [dict(base_candidate)],
        "validation_metrics": [
            {
                "candidate_identity": "c1",
                "validation_status": "unavailable",
                "mean_absolute_error": None,
                "root_mean_squared_error": None,
                "predictive_r_squared": None,
                "prediction_interval_coverage": None
            }
        ]
    }

    summaries = build_validation_summary(
        descriptive_tables,
        {"candidate_registry": [], "grouped_validation_metrics": []},
        {"candidate_registry": [], "validation_metrics": []}
    )

    assert len(summaries) == 1
    assert summaries[0]["assessment_status"] == "unavailable"
    assert summaries[0]["assessment_reason_code"] == "VALIDATION_METRICS_UNAVAILABLE"


# Confirming partial validation coverage is restricted, not completed
def test_partial_validation_coverage_is_completed_with_restrictions() -> None:
    from scripts.assess_robustness import build_validation_summary

    descriptive_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "c1",
                "template_name": "linear_common_trajectory",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined_within_scope",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "analytical_result_status": "available"
            },
            {
                "candidate_identity": "c2",
                "template_name": "linear_common_trajectory",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined_within_scope",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "analytical_result_status": "available"
            }
        ],
        "validation_metrics": [
            {"candidate_identity": "c1", "validation_status": "completed",
             "mean_absolute_error": "0.01", "root_mean_squared_error": "0.02",
             "predictive_r_squared": "0.9", "prediction_interval_coverage": "0.95"},
            {"candidate_identity": "c2", "validation_status": "unavailable",
             "mean_absolute_error": None, "root_mean_squared_error": None,
             "predictive_r_squared": None, "prediction_interval_coverage": None}
        ]
    }

    summaries = build_validation_summary(
        descriptive_tables,
        {"candidate_registry": [], "grouped_validation_metrics": []},
        {"candidate_registry": [], "validation_metrics": []}
    )

    assert len(summaries) == 1
    assert summaries[0]["assessment_status"] == "completed_with_restrictions"
    assert summaries[0]["assessment_reason_code"] == "PARTIAL_VALIDATION_COVERAGE"


# Confirming completed validation with zero available models is diagnostic only
def test_completed_validation_without_available_models_is_diagnostic() -> None:
    from scripts.assess_robustness import build_validation_summary

    descriptive_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "c1",
                "template_name": "linear_common_trajectory",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined_within_scope",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "analytical_result_status": "diagnostic_only"
            }
        ],
        "validation_metrics": [
            {"candidate_identity": "c1", "validation_status": "completed",
             "mean_absolute_error": "0.01", "root_mean_squared_error": "0.02",
             "predictive_r_squared": "0.5", "prediction_interval_coverage": "0.95"}
        ]
    }

    summaries = build_validation_summary(
        descriptive_tables,
        {"candidate_registry": [], "grouped_validation_metrics": []},
        {"candidate_registry": [], "validation_metrics": []}
    )

    assert summaries[0]["assessment_status"] == "diagnostic_only"
    assert summaries[0]["assessment_reason_code"] == "NO_ANALYTICALLY_AVAILABLE_VALIDATION_RECORDS"


# Confirming family comparison requires both validation statuses to be completed
def test_model_family_comparison_requires_completed_validation() -> None:
    from scripts.assess_robustness import build_model_family_comparisons

    predictive_tables = {
        "candidate_registry": [
            {"benchmark_identity": "first", "candidate_identity": "candidate",
             "design_scope": "Stage_1", "sampling_design_scope": "combined",
             "aging_mode": "calendar_aging", "outcome": "relative_capacity",
             "cohort": "primary", "horizon_axis": "elapsed_calendar_days",
             "model_family": "elastic_net", "analytical_status": "available"},
            {"benchmark_identity": "second", "candidate_identity": "candidate",
             "design_scope": "Stage_1", "sampling_design_scope": "combined",
             "aging_mode": "calendar_aging", "outcome": "relative_capacity",
             "cohort": "primary", "horizon_axis": "elapsed_calendar_days",
             "model_family": "random_forest", "analytical_status": "available"}
        ],
        "validation_metrics": [
            {"benchmark_identity": "first", "validation_status": "completed",
             "equal_cell_weighted_root_mean_squared_error": "0.02",
             "equal_cell_weighted_mean_absolute_error": "0.01",
             "equal_cell_weighted_predictive_r_squared": "0.8"},
            {"benchmark_identity": "second", "validation_status": "unavailable",
             "equal_cell_weighted_root_mean_squared_error": "0.01",
             "equal_cell_weighted_mean_absolute_error": "0.005",
             "equal_cell_weighted_predictive_r_squared": "0.9"}
        ]
    }

    records = build_model_family_comparisons(predictive_tables)
    assert len(records) == 1
    assert records[0]["comparison_status"] == "unavailable"
    assert records[0]["comparison_reason_code"] == "VALIDATION_METRICS_UNAVAILABLE"


# Confirming duplicate Bayesian prediction identity is rejected
def test_duplicate_bayesian_prediction_identity_is_rejected() -> None:
    from scripts.assess_robustness import build_prediction_agreement

    duplicate_prediction = {
        "candidate_identity": "candidate",
        "cell_identity": "cell_1",
        "source_identity": "source_1",
        "condition_protocol_identity": "condition_a",
        "horizon": "10.0",
        "observed_outcome": "0.95",
        "posterior_predictive_median": "0.94",
        "prediction_status": "completed"
    }

    bayesian_tables = {
        "candidate_registry": [{
            "candidate_identity": "candidate",
            "design_scope": "Stage_1", "sampling_design_scope": "combined",
            "aging_mode": "calendar_aging", "outcome": "relative_capacity",
            "cohort": "primary", "horizon_axis": "elapsed_calendar_days",
            "posterior_analytical_status": "available"
        }],
        "grouped_validation_predictions": [dict(duplicate_prediction), dict(duplicate_prediction)]
    }

    predictive_tables = {
        "candidate_registry": [],
        "validation_predictions": []
    }

    with pytest.raises(RuntimeError, match = "duplicate identity"):
        build_prediction_agreement(bayesian_tables, predictive_tables)


# Confirming nonfinite paired prediction prevents completed agreement
def test_nonfinite_shared_prediction_blocks_completion() -> None:
    from scripts.assess_robustness import build_prediction_agreement

    bayesian_tables = {
        "candidate_registry": [{
            "candidate_identity": "candidate",
            "design_scope": "Stage_1", "sampling_design_scope": "combined",
            "aging_mode": "calendar_aging", "outcome": "relative_capacity",
            "cohort": "primary", "horizon_axis": "elapsed_calendar_days",
            "posterior_analytical_status": "available"
        }],
        "grouped_validation_predictions": [
            {"candidate_identity": "candidate", "cell_identity": "cell_1",
             "source_identity": "source_1", "condition_protocol_identity": "condition_a",
             "horizon": "10.0", "observed_outcome": "0.95",
             "posterior_predictive_median": "0.94", "prediction_status": "completed"},
            {"candidate_identity": "candidate", "cell_identity": "cell_2",
             "source_identity": "source_2", "condition_protocol_identity": "condition_a",
             "horizon": "10.0", "observed_outcome": "0.93",
             "posterior_predictive_median": "0.92", "prediction_status": "completed"}
        ]
    }

    predictive_tables = {
        "candidate_registry": [{
            "benchmark_identity": "benchmark", "candidate_identity": "candidate",
            "model_family": "elastic_net", "design_scope": "Stage_1",
            "sampling_design_scope": "combined", "aging_mode": "calendar_aging",
            "outcome": "relative_capacity", "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days", "analytical_status": "available"
        }],
        "validation_predictions": [
            {"benchmark_identity": "benchmark", "cell_identity": "cell_1",
             "source_identity": "source_1", "condition_protocol_identity": "condition_a",
             "horizon": "10.0", "observed_outcome": "0.95",
             "predicted_outcome": "0.945", "prediction_status": "completed"},
            {"benchmark_identity": "benchmark", "cell_identity": "cell_2",
             "source_identity": "source_2", "condition_protocol_identity": "condition_a",
             "horizon": "10.0", "observed_outcome": "0.93",
             "predicted_outcome": "", "prediction_status": "completed"}
        ]
    }

    records = build_prediction_agreement(bayesian_tables, predictive_tables)
    assert len(records) == 1
    assert records[0]["assessment_status"] == "unavailable"
    assert records[0]["assessment_reason_code"] == "SHARED_PREDICTION_VALUE_UNAVAILABLE"


# Confirming unregistered predictive configuration identity is rejected
def test_unregistered_predictive_configuration_is_rejected() -> None:
    from scripts.assess_robustness import build_configuration_boundary_summary

    predictive_tables = {
        "hyperparameter_selection": [
            {"model_family": "elastic_net", "candidate_identity": "c1",
             "configuration_identity": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
             "configuration": {"alpha": 0.0001, "l1_ratio": 0.1},
             "selected": "true", "outer_fold_index": "0"}
        ]
    }

    policy = {
        "model_families": {
            "elastic_net": {
                "configurations": [
                    {"alpha": 0.0001, "l1_ratio": 0.1},
                ]
            }
        }
    }

    with pytest.raises(RuntimeError, match = "not registered in the predictive policy"):
        build_configuration_boundary_summary(predictive_tables, policy)


# Confirming configuration identity/value mismatch is rejected
def test_configuration_identity_value_mismatch_is_rejected() -> None:
    from scripts.assess_robustness import (
        build_configuration_boundary_summary,
        canonical_json
    )

    policy = {
        "model_families": {
            "elastic_net": {
                "configurations": [{"alpha": 0.0001, "l1_ratio": 0.1}]
            }
        }
    }

    matching_identity = hashlib.sha256(
        canonical_json({
            "model_family": "elastic_net",
            "configuration": {"alpha": 0.0001, "l1_ratio": 0.1}
        }).encode("utf-8")
    ).hexdigest()

    predictive_tables = {
        "hyperparameter_selection": [
            {"model_family": "elastic_net", "candidate_identity": "c1",
             "configuration_identity": matching_identity,
             "configuration": {"alpha": 0.01, "l1_ratio": 0.1},
             "selected": "true", "outer_fold_index": "0"}
        ]
    }

    with pytest.raises(RuntimeError, match = "does not match its registered identity"):
        build_configuration_boundary_summary(predictive_tables, policy)


# Confirming restricted coverage without a reason is rejected
def test_restricted_coverage_without_reason_is_rejected() -> None:
    from scripts.assess_robustness import (
        build_assessment_coverage,
        domain_evaluation
    )

    policy = load_policy()

    domain_evaluations = {
        domain["assessment_identity"]: domain_evaluation(
            status = "completed",
            total_count = 1,
            completed_count = 1,
            restricted_count = 0,
            unavailable_count = 0,
            reason_code = None
        )
        for domain in policy["assessment_domains"]
    }

    first_identity = policy["assessment_domains"][0]["assessment_identity"]
    domain_evaluations[first_identity] = domain_evaluation(
        status = "completed_with_restrictions",
        total_count = 1,
        completed_count = 0,
        restricted_count = 1,
        unavailable_count = 0,
        reason_code = None
    )

    with pytest.raises(RuntimeError, match = "does not provide a reason code"):
        build_assessment_coverage(policy, {}, domain_evaluations)


# Confirming unregistered coverage reason is rejected
def test_restricted_coverage_with_unregistered_reason_is_rejected() -> None:
    from scripts.assess_robustness import (
        build_assessment_coverage,
        domain_evaluation
    )

    policy = load_policy()

    domain_evaluations = {
        domain["assessment_identity"]: domain_evaluation(
            status = "completed",
            total_count = 1,
            completed_count = 1,
            restricted_count = 0,
            unavailable_count = 0,
            reason_code = None
        )
        for domain in policy["assessment_domains"]
    }

    first_identity = policy["assessment_domains"][0]["assessment_identity"]
    domain_evaluations[first_identity] = domain_evaluation(
        status = "not_authorized",
        total_count = 1,
        completed_count = 0,
        restricted_count = 1,
        unavailable_count = 0,
        reason_code = "UNREGISTERED_REASON"
    )

    with pytest.raises(RuntimeError, match = "unregistered reason code"):
        build_assessment_coverage(policy, {}, domain_evaluations)


# Confirming assessment-coverage codes are validated in emitted-code checks
def test_assessment_coverage_codes_are_validated() -> None:
    from scripts.assess_robustness import validate_emitted_codes_and_statuses

    policy = load_policy()

    tables = {
        "validation_summary": [],
        "model_family_comparisons": [],
        "prior_sensitivity": [],
        "influence_summary": [],
        "prediction_agreement": [],
        "cohort_sensitivity": [],
        "descriptive_specification_comparisons": [],
        "structural_limitations": [],
        "unsupported_condition_inventory": [],
        "prediction_interval_diagnostics": [],
        "horizon_evaluation_sensitivity": [],
        "incident_evaluation_sensitivity": [],
        "assessment_coverage": [
            {
                "assessment_status": "not_authorized",
                "assessment_reason_code": "UNREGISTERED_REASON"
            }
        ]
    }

    with pytest.raises(RuntimeError, match = "unregistered reason code"):
        validate_emitted_codes_and_statuses(tables, policy)


# Confirming published table content mismatch is rejected
def test_published_table_content_mismatch_is_rejected(tmp_path: Path) -> None:
    from scripts.assess_robustness import (
        ASSESSMENT_COVERAGE_FIELDS,
        COHORT_SENSITIVITY_FIELDS,
        CONFIGURATION_BOUNDARY_FIELDS,
        DESCRIPTIVE_SPECIFICATION_COMPARISON_FIELDS,
        EXTERNAL_VALIDITY_FIELDS,
        FRAGILITY_REGISTRY_FIELDS,
        INFLUENCE_SUMMARY_FIELDS,
        MODEL_FAMILY_COMPARISON_FIELDS,
        PREDICTION_AGREEMENT_FIELDS,
        PRIOR_SENSITIVITY_FIELDS,
        STRUCTURAL_LIMITATION_FIELDS,
        UNSUPPORTED_CONDITION_FIELDS,
        VALIDATION_SUMMARY_FIELDS,
        validate_published_robustness_directory,
        write_csv_table
    )

    policy = load_policy()

    field_map = {
        "validation_summary": VALIDATION_SUMMARY_FIELDS,
        "model_family_comparisons": MODEL_FAMILY_COMPARISON_FIELDS,
        "prior_sensitivity": PRIOR_SENSITIVITY_FIELDS,
        "influence_summary": INFLUENCE_SUMMARY_FIELDS,
        "prediction_agreement": PREDICTION_AGREEMENT_FIELDS,
        "cohort_sensitivity": COHORT_SENSITIVITY_FIELDS,
        "descriptive_specification_comparisons": DESCRIPTIVE_SPECIFICATION_COMPARISON_FIELDS,
        "structural_limitations": STRUCTURAL_LIMITATION_FIELDS,
        "configuration_boundary_summary": CONFIGURATION_BOUNDARY_FIELDS,
        "fragility_registry": FRAGILITY_REGISTRY_FIELDS,
        "unsupported_condition_inventory": UNSUPPORTED_CONDITION_FIELDS,
        "assessment_coverage": ASSESSMENT_COVERAGE_FIELDS,
        "external_validity": EXTERNAL_VALIDITY_FIELDS,
        "prediction_interval_diagnostics": PREDICTION_INTERVAL_DIAGNOSTIC_FIELDS,
        "horizon_evaluation_sensitivity": HORIZON_EVALUATION_SENSITIVITY_FIELDS,
        "incident_evaluation_sensitivity": INCIDENT_EVALUATION_SENSITIVITY_FIELDS,
        "predictive_sensitivity_summary": PREDICTIVE_SENSITIVITY_SUMMARY_FIELDS
    }

    empty_tables = {name: [] for name in field_map}

    empty_tables["assessment_coverage"] = [
        {
            "assessment_identity": record["assessment_identity"],
            "evidence_source": record["evidence_source"],
            "evidence_unit": record["evidence_unit"],
            "total_evidence_count": 0,
            "completed_evidence_count": 0,
            "restricted_evidence_count": 0,
            "unavailable_evidence_count": 0,
            "assessment_status": "unavailable",
            "assessment_reason_code": "NO_EVIDENCE",
            "interpretation": record["interpretation"]
        }
        for record in policy["assessment_domains"]
    ]

    empty_tables["external_validity"] = [dict(r) for r in policy["external_validity_domains"]]

    table_manifests = {}

    for table_name, records in empty_tables.items():
        table_path = tmp_path / f"{table_name}.csv"
        write_csv_table(table_path, field_map[table_name], records)
        table_manifests[table_name] = {
            "file_name": table_path.name,
            "row_count": len(records),
            "sha256": calculate_file_sha256(table_path),
            "schema": [{"name": field_name} for field_name in field_map[table_name]]
        }

    policy_sha = calculate_file_sha256(
        PROJECT_ROOT / "configs" / "robustness_assessment_policy.json"
    )
    implementation_sha = calculate_file_sha256(
        Path(__import__("scripts.assess_robustness", fromlist = ["__file__"]).__file__).resolve()
    )

    status_counts = {"unavailable": len(policy["assessment_domains"])}

    report = {
        "execution_status": "completed",
        "scientific_assessment_status": "restricted_evidence_summary_only",
        "policy_name": policy["policy_name"],
        "policy_version": policy["policy_version"],
        "policy_sha256": policy_sha,
        "implementation_sha256": implementation_sha,
        "input_hashes": {},
        "causal_interpretation_authorized": False,
        "policy_recommendation_authorized": False,
        "performance_selected_model_winner_declared": False,
        "magnitude_dependent_robustness_classification_used": False,
        "registration_status": policy["registration_status"],
        "amendment": policy["amendment"],
        "domain_status_counts": status_counts,
        "prediction_inventory_reconciliation": {},
        "analytical_handoff": {
            "status": "restricted_evidence_summary_only",
            "fully_assessed_domain_count": 0,
            "restricted_or_unavailable_domain_count": len(policy["assessment_domains"]),
            "causal_interpretation_authorized": False,
            "policy_recommendation_authorized": False,
            "treatment_optimization_authorized": False,
            "cross_stage_transport_authorized": False
        },
        "tables": table_manifests
    }

    (tmp_path / "robustness_assessment_report.json").write_text(
        json.dumps(report, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8"
    )

    expected_tables_with_different_content = dict(empty_tables)
    expected_tables_with_different_content["assessment_coverage"] = [
        {**record, "total_evidence_count": 999} if record_index == 0 else record
        for record_index, record in enumerate(empty_tables["assessment_coverage"])
    ]

    with pytest.raises(RuntimeError, match = "does not match the computed records"):
        validate_published_robustness_directory(
            published_directory = tmp_path,
            expected_policy = policy,
            expected_policy_sha256 = policy_sha,
            expected_input_hashes = {},
            expected_tables = expected_tables_with_different_content,
            expected_assessment_status = "restricted_evidence_summary_only",
            expected_assessment_status_counts = status_counts,
            expected_prediction_inventory_reconciliation = {}
        )


# Confirming the post-publication validator rejects a mismatched prediction-inventory reconciliation
def test_published_report_rejects_mismatched_prediction_inventory_reconciliation(tmp_path: Path) -> None:
    from scripts.assess_robustness import (
        ASSESSMENT_COVERAGE_FIELDS,
        COHORT_SENSITIVITY_FIELDS,
        CONFIGURATION_BOUNDARY_FIELDS,
        DESCRIPTIVE_SPECIFICATION_COMPARISON_FIELDS,
        EXTERNAL_VALIDITY_FIELDS,
        FRAGILITY_REGISTRY_FIELDS,
        INFLUENCE_SUMMARY_FIELDS,
        MODEL_FAMILY_COMPARISON_FIELDS,
        PREDICTION_AGREEMENT_FIELDS,
        PRIOR_SENSITIVITY_FIELDS,
        STRUCTURAL_LIMITATION_FIELDS,
        UNSUPPORTED_CONDITION_FIELDS,
        VALIDATION_SUMMARY_FIELDS,
        validate_published_robustness_directory,
        write_csv_table
    )

    policy = load_policy()

    field_map = {
        "validation_summary": VALIDATION_SUMMARY_FIELDS,
        "model_family_comparisons": MODEL_FAMILY_COMPARISON_FIELDS,
        "prior_sensitivity": PRIOR_SENSITIVITY_FIELDS,
        "influence_summary": INFLUENCE_SUMMARY_FIELDS,
        "prediction_agreement": PREDICTION_AGREEMENT_FIELDS,
        "cohort_sensitivity": COHORT_SENSITIVITY_FIELDS,
        "descriptive_specification_comparisons": DESCRIPTIVE_SPECIFICATION_COMPARISON_FIELDS,
        "structural_limitations": STRUCTURAL_LIMITATION_FIELDS,
        "configuration_boundary_summary": CONFIGURATION_BOUNDARY_FIELDS,
        "fragility_registry": FRAGILITY_REGISTRY_FIELDS,
        "unsupported_condition_inventory": UNSUPPORTED_CONDITION_FIELDS,
        "assessment_coverage": ASSESSMENT_COVERAGE_FIELDS,
        "external_validity": EXTERNAL_VALIDITY_FIELDS,
        "prediction_interval_diagnostics": PREDICTION_INTERVAL_DIAGNOSTIC_FIELDS,
        "horizon_evaluation_sensitivity": HORIZON_EVALUATION_SENSITIVITY_FIELDS,
        "incident_evaluation_sensitivity": INCIDENT_EVALUATION_SENSITIVITY_FIELDS,
        "predictive_sensitivity_summary": PREDICTIVE_SENSITIVITY_SUMMARY_FIELDS
    }

    empty_tables = {name: [] for name in field_map}

    empty_tables["assessment_coverage"] = [
        {
            "assessment_identity": record["assessment_identity"],
            "evidence_source": record["evidence_source"],
            "evidence_unit": record["evidence_unit"],
            "total_evidence_count": 0,
            "completed_evidence_count": 0,
            "restricted_evidence_count": 0,
            "unavailable_evidence_count": 0,
            "assessment_status": "unavailable",
            "assessment_reason_code": "NO_EVIDENCE",
            "interpretation": record["interpretation"]
        }
        for record in policy["assessment_domains"]
    ]

    empty_tables["external_validity"] = [dict(r) for r in policy["external_validity_domains"]]

    table_manifests = {}

    for table_name, records in empty_tables.items():
        table_path = tmp_path / f"{table_name}.csv"
        write_csv_table(table_path, field_map[table_name], records)
        table_manifests[table_name] = {
            "file_name": table_path.name,
            "row_count": len(records),
            "sha256": calculate_file_sha256(table_path),
            "schema": [{"name": field_name} for field_name in field_map[table_name]]
        }

    policy_sha = calculate_file_sha256(
        PROJECT_ROOT / "configs" / "robustness_assessment_policy.json"
    )
    implementation_sha = calculate_file_sha256(
        Path(__import__("scripts.assess_robustness", fromlist = ["__file__"]).__file__).resolve()
    )

    status_counts = {"unavailable": len(policy["assessment_domains"])}

    report = {
        "execution_status": "completed",
        "scientific_assessment_status": "restricted_evidence_summary_only",
        "policy_name": policy["policy_name"],
        "policy_version": policy["policy_version"],
        "policy_sha256": policy_sha,
        "implementation_sha256": implementation_sha,
        "input_hashes": {},
        "causal_interpretation_authorized": False,
        "policy_recommendation_authorized": False,
        "performance_selected_model_winner_declared": False,
        "magnitude_dependent_robustness_classification_used": False,
        "registration_status": policy["registration_status"],
        "amendment": policy["amendment"],
        "domain_status_counts": status_counts,
        "prediction_inventory_reconciliation": {},
        "analytical_handoff": {
            "status": "restricted_evidence_summary_only",
            "fully_assessed_domain_count": 0,
            "restricted_or_unavailable_domain_count": len(policy["assessment_domains"]),
            "causal_interpretation_authorized": False,
            "policy_recommendation_authorized": False,
            "treatment_optimization_authorized": False,
            "cross_stage_transport_authorized": False
        },
        "tables": table_manifests
    }

    (tmp_path / "robustness_assessment_report.json").write_text(
        json.dumps(report, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8"
    )

    mismatched_expected_reconciliation = {
        "descriptive_modeling": {
            "identity_field": "candidate_identity",
            "excluded_identity_count": 0,
            "excluded_prediction_count": 0,
            "excluded_identities": [],
            "reason_code": None
        }
    }

    with pytest.raises(
        RuntimeError,
        match = "prediction inventory reconciliation is invalid"
    ):
        validate_published_robustness_directory(
            published_directory = tmp_path,
            expected_policy = policy,
            expected_policy_sha256 = policy_sha,
            expected_input_hashes = {},
            expected_tables = empty_tables,
            expected_assessment_status = "restricted_evidence_summary_only",
            expected_assessment_status_counts = status_counts,
            expected_prediction_inventory_reconciliation = (
                mismatched_expected_reconciliation
            )
        )

    report["prediction_inventory_reconciliation"] = []

    (
        tmp_path
        / "robustness_assessment_report.json"
    ).write_text(
        json.dumps(
            report,
            indent = 2,
            sort_keys = True
        )
        + "\n",
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = (
            "prediction inventory reconciliation is not an object"
        )
    ):
        validate_published_robustness_directory(
            published_directory = tmp_path,
            expected_policy = policy,
            expected_policy_sha256 = policy_sha,
            expected_input_hashes = {},
            expected_tables = empty_tables,
            expected_assessment_status = (
                "restricted_evidence_summary_only"
            ),
            expected_assessment_status_counts = (
                status_counts
            ),
            expected_prediction_inventory_reconciliation = []
        )


# Confirming structural evidence counts feed coverage records
def test_structural_evidence_counts_reconcile_into_coverage() -> None:
    from scripts.assess_robustness import (
        build_assessment_coverage,
        domain_evaluation
    )

    policy = load_policy()

    structural_limitations = [
        {"limitation_identity": "laboratory_rank_deficient_matrices", "evidence_record_count": 0},
        {"limitation_identity": "laboratory_non_estimable_terms", "evidence_record_count": 0},
        {"limitation_identity": "equipment_allocation_restrictions", "evidence_record_count": 12828},
        {"limitation_identity": "ocv_corrected_resistance_not_approved", "evidence_record_count": 1},
        {"limitation_identity": "net_storage_exposure_unavailable", "evidence_record_count": 3736}
    ]

    evidence_counts = {
        record["limitation_identity"]: int(record["evidence_record_count"])
        for record in structural_limitations
    }

    domain_evaluations = {
        domain["assessment_identity"]: domain_evaluation(
            status = "completed",
            total_count = 0,
            completed_count = 0,
            restricted_count = 0,
            unavailable_count = 0,
            reason_code = None
        )
        for domain in policy["assessment_domains"]
    }

    laboratory_equipment_category_count = sum(
        1
        for limitation_identity in (
            "laboratory_rank_deficient_matrices",
            "laboratory_non_estimable_terms",
            "equipment_allocation_restrictions"
        )
        if evidence_counts.get(limitation_identity, 0) > 0
    )

    domain_evaluations["laboratory_and_equipment_separability"] = domain_evaluation(
        status = "unsupported",
        total_count = laboratory_equipment_category_count,
        completed_count = 0,
        restricted_count = laboratory_equipment_category_count,
        unavailable_count = 0,
        reason_code = "EQUIPMENT_LAB_ALIASED"
    )

    domain_evaluations["alternative_outcome_definition"] = domain_evaluation(
        status = "not_authorized",
        total_count = evidence_counts.get("ocv_corrected_resistance_not_approved", 0),
        completed_count = 0,
        restricted_count = evidence_counts.get("ocv_corrected_resistance_not_approved", 0),
        unavailable_count = 0,
        reason_code = "OCV_CORRECTED_RESISTANCE_PROVISIONAL"
    )

    domain_evaluations["net_calendar_exposure"] = domain_evaluation(
        status = "unavailable",
        total_count = evidence_counts.get("net_storage_exposure_unavailable", 0),
        completed_count = 0,
        restricted_count = 0,
        unavailable_count = evidence_counts.get("net_storage_exposure_unavailable", 0),
        reason_code = "NET_STORAGE_EXPOSURE_UNAVAILABLE"
    )

    coverage = build_assessment_coverage(policy, {}, domain_evaluations)
    by_identity = {row["assessment_identity"]: row for row in coverage}

    assert (
        by_identity[
            "laboratory_and_equipment_separability"
        ][
            "restricted_evidence_count"
        ]
        == 1
    )

    assert (
        by_identity[
            "alternative_outcome_definition"
        ][
            "restricted_evidence_count"
        ]
        == 1
    )

    assert (
        by_identity[
            "net_calendar_exposure"
        ][
            "unavailable_evidence_count"
        ]
        == 3736
    )


# Confirming exact inventory matches are not relabelled incomplete when the source model is diagnostic-only
def test_exact_inventory_match_is_not_relabelled_incomplete_when_source_is_diagnostic() -> None:
    from scripts.assess_robustness import build_fragility_registry

    bayesian_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "candidate",
                "admissibility_status": "admissible",
                "posterior_analytical_status": "diagnostic_only",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "sorted_estimand_decision_identities": "[]"
            }
        ],
        "grouped_validation_metrics": [
            {"candidate_identity": "candidate", "validation_status": "completed"}
        ]
    }

    predictive_tables = {
        "candidate_registry": [
            {
                "benchmark_identity": "bench",
                "candidate_identity": "candidate",
                "model_family": "elastic_net",
                "analytical_status": "available",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "condition_protocol_identities": []
            }
        ],
        "validation_metrics": [
            {
                "benchmark_identity": "bench",
                "validation_status": "completed",
                "equal_cell_weighted_predictive_r_squared": "0.5"
            }
        ],
        "condition_calibration": []
    }

    tables = {
        "prior_sensitivity": [],
        "influence_summary": [],
        "prediction_agreement": [
            {
                "candidate_identity": "candidate",
                "model_family": "elastic_net",
                "assessment_status": "diagnostic_only",
                "exact_inventory_match": True,
                "bayesian_analytical_status": "diagnostic_only",
                "predictive_analytical_status": "available"
            }
        ]
    }

    records = build_fragility_registry(
        {"candidate_registry": [], "validation_metrics": []},
        bayesian_tables,
        predictive_tables,
        [],
        tables
    )

    predictive_row = next(
        r for r in records if r["source_analysis"] == "predictive_modeling"
    )

    assert predictive_row["cross_framework_inventory_complete"] is True
    assert predictive_row["cross_framework_source_models_available"] is False
    assert "cross_framework_inventory_incomplete" not in predictive_row["fragility_indicators"]
    assert "cross_framework_source_model_unavailable" in predictive_row["fragility_indicators"]


# Confirming cohort sensitivity uses validation-metric cell and observation counts
def test_cohort_sensitivity_uses_validation_metric_counts() -> None:
    from scripts.assess_robustness import build_cohort_sensitivity

    policy = load_policy()

    candidate_common = {
        "design_scope": "Stage_1",
        "sampling_design_scope": "combined",
        "aging_mode": "calendar_aging",
        "outcome": "relative_capacity",
        "horizon_axis": "elapsed_calendar_days",
        "template_name": "linear_common_trajectory",
        "estimator": "weighted_least_squares",
        "specification": "main_effects",
        "context_family": "no_additional_context",
        "condition_protocol_identities": ["a"],
        "support_interval_minimum": "0.0",
        "support_interval_maximum": "100.0",
        "support_interval_lower_inclusive": "true",
        "support_interval_upper_inclusive": "true",
        "analytical_result_status": "available"
    }

    descriptive_tables = {
        "candidate_registry": [
            {**candidate_common, "candidate_identity": "p", "cohort": "primary"},
            {**candidate_common, "candidate_identity": "s", "cohort": "sensitivity"}
        ],
        "validation_metrics": [
            {
                "candidate_identity": "p",
                "validation_status": "completed",
                "root_mean_squared_error": "0.01",
                "mean_absolute_error": "0.005",
                "independent_cell_count": "9",
                "observation_count": "27"
            },
            {
                "candidate_identity": "s",
                "validation_status": "completed",
                "root_mean_squared_error": "0.012",
                "mean_absolute_error": "0.006",
                "independent_cell_count": "9",
                "observation_count": "27"
            }
        ],
        "validation_predictions": [
            {
                "candidate_identity": "p",
                "cell_identity": f"c{i}",
                "source_identity": f"src{i}",
                "condition_protocol_identity": "a",
                "horizon": "10.0"
            }
            for i in range(3)
        ] + [
            {
                "candidate_identity": "s",
                "cell_identity": f"c{i}",
                "source_identity": f"src{i}",
                "condition_protocol_identity": "a",
                "horizon": "10.0"
            }
            for i in range(3)
        ]
    }

    records = build_cohort_sensitivity(
        descriptive_tables,
        {"candidate_registry": [], "grouped_validation_metrics": [],
         "grouped_validation_predictions": []},
        {"candidate_registry": [], "validation_metrics": [],
         "validation_predictions": []},
        policy
    )

    assert len(records) == 1
    row = records[0]
    assert row["primary_validation_cell_count"] == 9
    assert row["primary_validation_observation_count"] == 27
    assert row["sensitivity_validation_cell_count"] == 9
    assert row["sensitivity_validation_observation_count"] == 27
    assert row["validation_population_count_relationship"] == "identical_inventory"


# Confirming cohort sensitivity rejects identical counts with different populations
def test_cohort_sensitivity_rejects_identical_counts_with_different_inventories() -> None:
    from scripts.assess_robustness import build_cohort_sensitivity

    policy = load_policy()

    candidate_common = {
        "design_scope": "Stage_1",
        "sampling_design_scope": "combined",
        "aging_mode": "calendar_aging",
        "outcome": "relative_capacity",
        "horizon_axis": "elapsed_calendar_days",
        "template_name": "linear_common_trajectory",
        "estimator": "weighted_least_squares",
        "specification": "main_effects",
        "context_family": "no_additional_context",
        "condition_protocol_identities": ["a"],
        "support_interval_minimum": "0.0",
        "support_interval_maximum": "100.0",
        "support_interval_lower_inclusive": "true",
        "support_interval_upper_inclusive": "true",
        "analytical_result_status": "available"
    }

    descriptive_tables = {
        "candidate_registry": [
            {**candidate_common, "candidate_identity": "p", "cohort": "primary"},
            {**candidate_common, "candidate_identity": "s", "cohort": "sensitivity"}
        ],
        "validation_metrics": [
            {
                "candidate_identity": "p",
                "validation_status": "completed",
                "root_mean_squared_error": "0.01",
                "mean_absolute_error": "0.005",
                "independent_cell_count": "3",
                "observation_count": "3"
            },
            {
                "candidate_identity": "s",
                "validation_status": "completed",
                "root_mean_squared_error": "0.012",
                "mean_absolute_error": "0.006",
                "independent_cell_count": "3",
                "observation_count": "3"
            }
        ],
        "validation_predictions": [
            {
                "candidate_identity": "p",
                "cell_identity": f"c{i}",
                "source_identity": f"src{i}",
                "condition_protocol_identity": "a",
                "horizon": "10.0"
            }
            for i in range(3)
        ] + [
            {
                "candidate_identity": "s",
                "cell_identity": f"d{i}",
                "source_identity": f"srcd{i}",
                "condition_protocol_identity": "a",
                "horizon": "10.0"
            }
            for i in range(3)
        ]
    }

    records = build_cohort_sensitivity(
        descriptive_tables,
        {"candidate_registry": [], "grouped_validation_metrics": [],
         "grouped_validation_predictions": []},
        {"candidate_registry": [], "validation_metrics": [],
         "validation_predictions": []},
        policy
    )

    assert len(records) == 1
    row = records[0]
    assert row["comparison_status"] == "diagnostic_only"
    assert row["comparison_reason_code"] == "OUT_OF_FOLD_POPULATIONS_DIFFER"
    assert row["validation_population_count_relationship"] == "different_inventory"


# Confirming cohort-inclusion sensitivity is classified as a proxy rather than as isolated incident evidence
def test_incident_sensitivity_is_proxy_only() -> None:
    policy = load_policy()
    domains = {d["assessment_identity"]: d for d in policy["assessment_domains"]}

    assert "cohort_inclusion_sensitivity" in domains
    assert "incident_sensitivity" not in domains
    assert domains["cohort_inclusion_sensitivity"]["evidence_source"] == "cohort_sensitivity"
    assert "INCIDENT_SENSITIVITY_PROXY_ONLY" in policy["assessment_contract"]["registered_reason_codes"]


# Confirming a two-level grid is classified as boundary-uninformative rather than as boundary-reached
def test_two_level_configuration_grid_is_boundary_uninformative() -> None:
    from scripts.assess_robustness import build_configuration_boundary_summary

    test_policy = {
        "model_families": {
            "elastic_net": {
                "configurations": [
                    {"alpha": 0.1},
                    {"alpha": 0.9},
                ]
            }
        }
    }

    selection_records = [
        {
            "model_family": "elastic_net",
            "candidate_identity": "c",
            "configuration_identity": "identity_a",
            "configuration": {"alpha": 0.1},
            "selected": "true",
            "outer_fold_index": "0"
        }
    ]

    expected_identity_a = hashlib.sha256(
        canonical_json({"model_family": "elastic_net", "configuration": {"alpha": 0.1}}).encode("utf-8")
    ).hexdigest()

    selection_records[0]["configuration_identity"] = expected_identity_a

    predictive_tables = {"hyperparameter_selection": selection_records}

    records = build_configuration_boundary_summary(predictive_tables, test_policy)
    row = next(r for r in records if r["configuration_identity"] == expected_identity_a)

    assert row["boundary_assessment_status"] == "uninformative"
    assert row["boundary_classification"] == "boundary_assessment_uninformative"
    assert row["on_registered_search_boundary"] is None
    assert "alpha" in row["uninformative_parameter_names"]
    assert row["informative_parameter_names"] == []


# Confirming validation-summary available-only medians are computed separately from all-completed medians
def test_validation_summary_reports_available_only_medians_separately() -> None:
    from scripts.assess_robustness import build_validation_summary

    descriptive_tables = {
        "candidate_registry": [
            {
                "candidate_identity": "a",
                "template_name": "t",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "analytical_result_status": "available"
            },
            {
                "candidate_identity": "b",
                "template_name": "t",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "analytical_result_status": "diagnostic_only"
            }
        ],
        "validation_metrics": [
            {
                "candidate_identity": "a",
                "validation_status": "completed",
                "mean_absolute_error": "0.01",
                "root_mean_squared_error": "0.02",
                "predictive_r_squared": "0.9",
                "prediction_interval_coverage": "0.95"
            },
            {
                "candidate_identity": "b",
                "validation_status": "completed",
                "mean_absolute_error": "0.99",
                "root_mean_squared_error": "0.99",
                "predictive_r_squared": "0.01",
                "prediction_interval_coverage": "0.5"
            }
        ]
    }

    rows = build_validation_summary(
        descriptive_tables,
        {"candidate_registry": [], "grouped_validation_metrics": []},
        {"candidate_registry": [], "validation_metrics": []}
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["metric_population"] == "all_validation_completed_records"
    assert row["median_mean_absolute_error"] == 0.5
    assert row["available_median_mean_absolute_error"] == 0.01
    assert row["available_validation_completed_count"] == 1


# Confirming the report exposes registration status, amendment and analytical handoff
def test_report_exposes_registration_amendment_and_handoff(tmp_path: Path) -> None:
    from scripts.assess_robustness import (
        ASSESSMENT_COVERAGE_FIELDS,
        COHORT_SENSITIVITY_FIELDS,
        CONFIGURATION_BOUNDARY_FIELDS,
        DESCRIPTIVE_SPECIFICATION_COMPARISON_FIELDS,
        EXTERNAL_VALIDITY_FIELDS,
        FRAGILITY_REGISTRY_FIELDS,
        INFLUENCE_SUMMARY_FIELDS,
        MODEL_FAMILY_COMPARISON_FIELDS,
        PREDICTION_AGREEMENT_FIELDS,
        PRIOR_SENSITIVITY_FIELDS,
        STRUCTURAL_LIMITATION_FIELDS,
        UNSUPPORTED_CONDITION_FIELDS,
        VALIDATION_SUMMARY_FIELDS,
        validate_published_robustness_directory,
        write_csv_table
    )

    policy = load_policy()

    field_map = {
        "validation_summary": VALIDATION_SUMMARY_FIELDS,
        "model_family_comparisons": MODEL_FAMILY_COMPARISON_FIELDS,
        "prior_sensitivity": PRIOR_SENSITIVITY_FIELDS,
        "influence_summary": INFLUENCE_SUMMARY_FIELDS,
        "prediction_agreement": PREDICTION_AGREEMENT_FIELDS,
        "cohort_sensitivity": COHORT_SENSITIVITY_FIELDS,
        "descriptive_specification_comparisons": DESCRIPTIVE_SPECIFICATION_COMPARISON_FIELDS,
        "structural_limitations": STRUCTURAL_LIMITATION_FIELDS,
        "configuration_boundary_summary": CONFIGURATION_BOUNDARY_FIELDS,
        "fragility_registry": FRAGILITY_REGISTRY_FIELDS,
        "unsupported_condition_inventory": UNSUPPORTED_CONDITION_FIELDS,
        "assessment_coverage": ASSESSMENT_COVERAGE_FIELDS,
        "external_validity": EXTERNAL_VALIDITY_FIELDS,
        "prediction_interval_diagnostics": PREDICTION_INTERVAL_DIAGNOSTIC_FIELDS,
        "horizon_evaluation_sensitivity": HORIZON_EVALUATION_SENSITIVITY_FIELDS,
        "incident_evaluation_sensitivity": INCIDENT_EVALUATION_SENSITIVITY_FIELDS,
        "predictive_sensitivity_summary": PREDICTIVE_SENSITIVITY_SUMMARY_FIELDS
    }

    empty_tables = {name: [] for name in field_map}
    empty_tables["assessment_coverage"] = [
        {
            "assessment_identity": record["assessment_identity"],
            "evidence_source": record["evidence_source"],
            "evidence_unit": record["evidence_unit"],
            "total_evidence_count": 0,
            "completed_evidence_count": 0,
            "restricted_evidence_count": 0,
            "unavailable_evidence_count": 0,
            "assessment_status": "unavailable",
            "assessment_reason_code": "NO_EVIDENCE",
            "interpretation": record["interpretation"]
        }
        for record in policy["assessment_domains"]
    ]
    empty_tables["external_validity"] = [dict(r) for r in policy["external_validity_domains"]]

    table_manifests = {}

    for table_name, records in empty_tables.items():
        table_path = tmp_path / f"{table_name}.csv"
        write_csv_table(table_path, field_map[table_name], records)
        table_manifests[table_name] = {
            "file_name": table_path.name,
            "row_count": len(records),
            "sha256": calculate_file_sha256(table_path),
            "schema": [{"name": f} for f in field_map[table_name]]
        }

    policy_sha = calculate_file_sha256(
        PROJECT_ROOT / "configs" / "robustness_assessment_policy.json"
    )
    implementation_sha = calculate_file_sha256(
        Path(__import__("scripts.assess_robustness", fromlist = ["__file__"]).__file__).resolve()
    )

    status_counts = {"unavailable": len(policy["assessment_domains"])}

    report = {
        "execution_status": "completed",
        "scientific_assessment_status": "restricted_evidence_summary_only",
        "policy_name": policy["policy_name"],
        "policy_version": policy["policy_version"],
        "policy_sha256": policy_sha,
        "implementation_sha256": implementation_sha,
        "input_hashes": {},
        "causal_interpretation_authorized": False,
        "policy_recommendation_authorized": False,
        "performance_selected_model_winner_declared": False,
        "magnitude_dependent_robustness_classification_used": False,
        "registration_status": policy["registration_status"],
        "amendment": policy["amendment"],
        "domain_status_counts": status_counts,
        "prediction_inventory_reconciliation": {},
        "analytical_handoff": {
            "status": "restricted_evidence_summary_only",
            "fully_assessed_domain_count": 0,
            "restricted_or_unavailable_domain_count": len(policy["assessment_domains"]),
            "causal_interpretation_authorized": False,
            "policy_recommendation_authorized": False,
            "treatment_optimization_authorized": False,
            "cross_stage_transport_authorized": False
        },
        "tables": table_manifests
    }

    (tmp_path / "robustness_assessment_report.json").write_text(
        json.dumps(report, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8"
    )

    validate_published_robustness_directory(
        published_directory = tmp_path,
        expected_policy = policy,
        expected_policy_sha256 = policy_sha,
        expected_input_hashes = {},
        expected_tables = empty_tables,
        expected_assessment_status = "restricted_evidence_summary_only",
        expected_assessment_status_counts = status_counts,
        expected_prediction_inventory_reconciliation = {}
    )


# Confirming the batch-stability domain reports the observed stage-batch evidence count
def test_batch_aliasing_has_observed_evidence() -> None:
    from scripts.assess_robustness import build_structural_limitations

    design_tables = {
        "analytical_scope": [
            {"stage_batch_relationship_status": "stage_2_used_a_new_battery_cell_batch_relative_to_stage_1"},
            {"stage_batch_relationship_status": "stage_2_used_a_new_battery_cell_batch_relative_to_stage_1"}
        ],
        "stage_overlap": [],
        "estimand_support": [],
        "design_identifiability": []
    }

    records = build_structural_limitations(
        design_tables,
        {"feature_contract": {"dc_internal_resistance": {"status": "approved"}}},
        []
    )

    by_identity = {r["limitation_identity"]: r for r in records}
    assert by_identity["stage_batch_aliasing"]["evidence_record_count"] == 2
    assert by_identity["stage_batch_aliasing"]["evidence_source"] == "analytical_scope"


# Rejecting duplicate JSON fields
def test_duplicate_json_fields_are_rejected(tmp_path: Path) -> None:
    from scripts.assess_robustness import load_json_object

    path = tmp_path / "duplicate.json"
    path.write_text('{"a": 1, "a": 2}', encoding = "utf-8")

    with pytest.raises(RuntimeError, match = "duplicate field"):
        load_json_object(path)


# Rejecting duplicate CSV headers
def test_duplicate_csv_headers_are_rejected(tmp_path: Path) -> None:
    from scripts.assess_robustness import read_csv_table

    path = tmp_path / "duplicate_headers.csv"
    path.write_text("a,a,b\n1,2,3\n", encoding = "utf-8")

    with pytest.raises(RuntimeError, match = "duplicate header"):
        read_csv_table(path)


# Rejecting malformed CSV rows
def test_csv_row_with_extra_fields_is_rejected(tmp_path: Path) -> None:
    from scripts.assess_robustness import read_csv_table

    path = tmp_path / "extra_fields.csv"
    path.write_text("a,b\n1,2,3\n", encoding = "utf-8")

    with pytest.raises(RuntimeError, match = "extra fields"):
        read_csv_table(path)


# Rejecting duplicate prediction identity
def test_duplicate_validation_prediction_identity_is_rejected() -> None:
    from scripts.assess_robustness import build_validation_prediction_inventories

    records = [
        {
            "candidate_identity": "candidate",
            "cell_identity": "cell_1",
            "source_identity": "src_1",
            "condition_protocol_identity": "cond",
            "horizon": "10.0"
        },
        {
            "candidate_identity": "candidate",
            "cell_identity": "cell_1",
            "source_identity": "src_1",
            "condition_protocol_identity": "cond",
            "horizon": "10.0"
        }
    ]

    with pytest.raises(RuntimeError, match = "duplicate record"):
        build_validation_prediction_inventories(records, "candidate_identity")


# Confirming metric counts reconcile with prediction inventory
def test_validation_metric_counts_reconcile_with_prediction_inventory() -> None:
    from scripts.assess_robustness import reconcile_prediction_inventories

    prediction_records = [
        {
            "candidate_identity": "candidate",
            "cell_identity": f"cell_{index}",
            "source_identity": f"src_{index}",
            "condition_protocol_identity": "cond",
            "horizon": "10.0"
        }
        for index in range(3)
    ]

    metric_records_ok = [
        {
            "candidate_identity": "candidate",
            "validation_status": "completed",
            "independent_cell_count": "3",
            "observation_count": "3"
        }
    ]

    reconcile_prediction_inventories(
        prediction_records = prediction_records,
        metric_records = metric_records_ok,
        identity_field = "candidate_identity",
        prediction_status_field = None,
        metric_status_field = "validation_status",
        cell_count_field = "independent_cell_count",
        observation_count_field = "observation_count"
    )

    metric_records_bad = [
        {
            "candidate_identity": "candidate",
            "validation_status": "completed",
            "independent_cell_count": "5",
            "observation_count": "3"
        }
    ]

    with pytest.raises(RuntimeError, match = "cell count does not match"):
        reconcile_prediction_inventories(
            prediction_records = prediction_records,
            metric_records = metric_records_bad,
            identity_field = "candidate_identity",
            prediction_status_field = None,
            metric_status_field = "validation_status",
            cell_count_field = "independent_cell_count",
            observation_count_field = "observation_count"
        )


# Confirming fragility indicators are unique per row
def test_fragility_applicable_indicators_are_unique() -> None:
    from scripts.assess_robustness import build_fragility_registry

    predictive_tables = {
        "candidate_registry": [
            {
                "benchmark_identity": "bench",
                "candidate_identity": "candidate",
                "model_family": "elastic_net",
                "analytical_status": "available",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "condition_protocol_identities": []
            }
        ],
        "validation_metrics": [
            {
                "benchmark_identity": "bench",
                "validation_status": "completed",
                "equal_cell_weighted_predictive_r_squared": "0.5"
            }
        ],
        "condition_calibration": []
    }

    tables = {
        "prior_sensitivity": [],
        "influence_summary": [],
        "prediction_agreement": [
            {
                "candidate_identity": "candidate",
                "model_family": "elastic_net",
                "assessment_status": "completed",
                "exact_inventory_match": True,
                "bayesian_analytical_status": "available",
                "predictive_analytical_status": "available"
            }
        ]
    }

    records = build_fragility_registry(
        {"candidate_registry": [], "validation_metrics": []},
        {"candidate_registry": [], "grouped_validation_metrics": []},
        predictive_tables,
        [],
        tables
    )

    for record in records:
        assert len(record["applicable_indicators"]) == len(set(record["applicable_indicators"]))
        assert len(record["fragility_indicators"]) == len(set(record["fragility_indicators"]))


# Confirming condition calibration is derived directly
def test_calibration_state_comes_from_condition_calibration_rows() -> None:
    from scripts.assess_robustness import build_condition_calibration_states

    predictive_tables = {
        "candidate_registry": [
            {
                "benchmark_identity": "bench",
                "condition_protocol_identities": ["a", "b"]
            }
        ],
        "condition_calibration": [
            {
                "benchmark_identity": "bench",
                "condition_protocol_identity": "a",
                "calibration_status": "completed"
            },
            {
                "benchmark_identity": "bench",
                "condition_protocol_identity": "b",
                "calibration_status": "unavailable"
            }
        ]
    }

    states = build_condition_calibration_states(predictive_tables)

    assert states["bench"] is False

    predictive_tables_ok = {
        "candidate_registry": [
            {"benchmark_identity": "bench2", "condition_protocol_identities": ["a"]}
        ],
        "condition_calibration": [
            {
                "benchmark_identity": "bench2",
                "condition_protocol_identity": "a",
                "calibration_status": "completed"
            }
        ]
    }

    assert build_condition_calibration_states(predictive_tables_ok)["bench2"] is True


# Confirming multiple unmet benchmark criteria remain independent
def test_multiple_unmet_benchmark_criteria_remain_independent() -> None:
    from scripts.assess_robustness import build_condition_calibration_states

    predictive_tables = {
        "candidate_registry": [
            {
                "benchmark_identity": "bench",
                "condition_protocol_identities": ["a"]
            }
        ],
        "condition_calibration": [
            {
                "benchmark_identity": "bench",
                "condition_protocol_identity": "a",
                "calibration_status": "unavailable"
            }
        ]
    }

    # Even if the benchmark reason is PREDICTIVE_BENCHMARK_NOT_MET,
    # direct calibration evidence overrides
    states = build_condition_calibration_states(predictive_tables)

    assert states["bench"] is False


# Requiring explicit accounting for unmatched prediction inventories
def test_policy_requires_unmatched_prediction_inventory_accounting() -> None:
    import copy

    from scripts.assess_robustness import (
        load_json_object,
        validate_policy
    )

    policy = load_json_object(
        PROJECT_ROOT
        / "configs"
        / "robustness_assessment_policy.json"
    )

    invalid_policy = copy.deepcopy(
        policy
    )

    invalid_policy[
        "assessment_contract"
    ][
        "unmatched_prediction_inventory_accounting_required"
    ] = False

    with pytest.raises(
        RuntimeError,
        match = "assessment contract is invalid"
    ):
        validate_policy(
            invalid_policy
        )


# Confirming laboratory-context evidence counts
def test_laboratory_context_evidence_counts() -> None:
    from scripts.assess_robustness import build_structural_limitations

    design_tables = {
        "analytical_scope": [],
        "stage_overlap": [],
        "estimand_support": [],
        "design_identifiability": [
            {"record_type": "matrix_summary", "context_family": "laboratory",
             "full_column_rank": "false"},
            {"record_type": "matrix_summary", "context_family": "laboratory",
             "full_column_rank": "true"},
            {"record_type": "term", "context_family": "laboratory",
             "structurally_estimable": "false"},
            {"record_type": "term", "context_family": "laboratory",
             "structurally_estimable": "true"},
            {"record_type": "term", "context_family": "no_additional_context",
             "structurally_estimable": "false"}
        ]
    }

    data_quality_contract = {
        "feature_contract": {"dc_internal_resistance": {"status": "approved"}}
    }

    records = build_structural_limitations(design_tables, data_quality_contract, [])
    by_identity = {r["limitation_identity"]: r for r in records}

    assert by_identity["laboratory_rank_deficient_matrices"]["evidence_record_count"] == 1
    assert by_identity["laboratory_non_estimable_terms"]["evidence_record_count"] == 1


# Confirming domain evidence counts reconcile
def test_domain_evidence_counts_reconcile() -> None:
    from scripts.assess_robustness import domain_evaluation

    result = domain_evaluation(
        status = "completed_with_restrictions",
        total_count = 10,
        completed_count = 4,
        restricted_count = 3,
        unavailable_count = 3,
        reason_code = "PARTIAL_EVIDENCE_COVERAGE"
    )

    assert result["total_count"] == 10

    with pytest.raises(RuntimeError, match = "do not reconcile"):
        domain_evaluation(
            status = "completed_with_restrictions",
            total_count = 10,
            completed_count = 4,
            restricted_count = 3,
            unavailable_count = 2,
            reason_code = "PARTIAL_EVIDENCE_COVERAGE"
        )


# Confirming execution and scientific status are separate
def test_execution_and_scientific_status_are_separate(tmp_path: Path) -> None:
    # Build a minimal report and check both fields present
    policy = load_policy()

    from scripts.assess_robustness import (
        ASSESSMENT_COVERAGE_FIELDS,
        COHORT_SENSITIVITY_FIELDS,
        CONFIGURATION_BOUNDARY_FIELDS,
        DESCRIPTIVE_SPECIFICATION_COMPARISON_FIELDS,
        EXTERNAL_VALIDITY_FIELDS,
        FRAGILITY_REGISTRY_FIELDS,
        INFLUENCE_SUMMARY_FIELDS,
        MODEL_FAMILY_COMPARISON_FIELDS,
        PREDICTION_AGREEMENT_FIELDS,
        PRIOR_SENSITIVITY_FIELDS,
        STRUCTURAL_LIMITATION_FIELDS,
        UNSUPPORTED_CONDITION_FIELDS,
        VALIDATION_SUMMARY_FIELDS,
        validate_published_robustness_directory,
        write_csv_table
    )

    field_map = {
        "validation_summary": VALIDATION_SUMMARY_FIELDS,
        "model_family_comparisons": MODEL_FAMILY_COMPARISON_FIELDS,
        "prior_sensitivity": PRIOR_SENSITIVITY_FIELDS,
        "influence_summary": INFLUENCE_SUMMARY_FIELDS,
        "prediction_agreement": PREDICTION_AGREEMENT_FIELDS,
        "cohort_sensitivity": COHORT_SENSITIVITY_FIELDS,
        "descriptive_specification_comparisons": DESCRIPTIVE_SPECIFICATION_COMPARISON_FIELDS,
        "structural_limitations": STRUCTURAL_LIMITATION_FIELDS,
        "configuration_boundary_summary": CONFIGURATION_BOUNDARY_FIELDS,
        "fragility_registry": FRAGILITY_REGISTRY_FIELDS,
        "assessment_coverage": ASSESSMENT_COVERAGE_FIELDS,
        "external_validity": EXTERNAL_VALIDITY_FIELDS,
        "unsupported_condition_inventory": UNSUPPORTED_CONDITION_FIELDS,
        "prediction_interval_diagnostics": PREDICTION_INTERVAL_DIAGNOSTIC_FIELDS,
        "horizon_evaluation_sensitivity": HORIZON_EVALUATION_SENSITIVITY_FIELDS,
        "incident_evaluation_sensitivity": INCIDENT_EVALUATION_SENSITIVITY_FIELDS,
        "predictive_sensitivity_summary": PREDICTIVE_SENSITIVITY_SUMMARY_FIELDS
    }

    empty_tables = {name: [] for name in field_map}

    empty_tables["assessment_coverage"] = [
        {
            "assessment_identity": record["assessment_identity"],
            "evidence_source": record["evidence_source"],
            "evidence_unit": record["evidence_unit"],
            "total_evidence_count": 0,
            "completed_evidence_count": 0,
            "restricted_evidence_count": 0,
            "unavailable_evidence_count": 0,
            "assessment_status": "unavailable",
            "assessment_reason_code": "NO_EVIDENCE",
            "interpretation": record["interpretation"]
        }
        for record in policy["assessment_domains"]
    ]

    empty_tables["external_validity"] = [dict(r) for r in policy["external_validity_domains"]]

    table_manifests = {}

    for table_name, records in empty_tables.items():
        table_path = tmp_path / f"{table_name}.csv"
        write_csv_table(table_path, field_map[table_name], records)
        table_manifests[table_name] = {
            "file_name": table_path.name,
            "row_count": len(records),
            "sha256": calculate_file_sha256(table_path),
            "schema": [{"name": f} for f in field_map[table_name]]
        }

    policy_sha = calculate_file_sha256(
        PROJECT_ROOT / "configs" / "robustness_assessment_policy.json"
    )
    implementation_sha = calculate_file_sha256(
        Path(__import__("scripts.assess_robustness", fromlist = ["__file__"]).__file__).resolve()
    )

    report = {
        "execution_status": "completed",
        "scientific_assessment_status": "restricted_evidence_summary_only",
        "policy_name": policy["policy_name"],
        "policy_version": policy["policy_version"],
        "policy_sha256": policy_sha,
        "implementation_sha256": implementation_sha,
        "input_hashes": {},
        "causal_interpretation_authorized": False,
        "policy_recommendation_authorized": False,
        "performance_selected_model_winner_declared": False,
        "magnitude_dependent_robustness_classification_used": False,
        "registration_status": policy["registration_status"],
        "amendment": policy["amendment"],
        "domain_status_counts": {"unavailable": len(policy["assessment_domains"])},
        "prediction_inventory_reconciliation": {},
        "analytical_handoff": {
            "status": "restricted_evidence_summary_only",
            "fully_assessed_domain_count": 0,
            "restricted_or_unavailable_domain_count": len(policy["assessment_domains"]),
            "causal_interpretation_authorized": False,
            "policy_recommendation_authorized": False,
            "treatment_optimization_authorized": False,
            "cross_stage_transport_authorized": False
        },
        "tables": table_manifests
    }

    (tmp_path / "robustness_assessment_report.json").write_text(
        json.dumps(report, indent = 2, sort_keys = True) + "\n", encoding = "utf-8"
    )

    validate_published_robustness_directory(
        published_directory = tmp_path,
        expected_policy = policy,
        expected_policy_sha256 = policy_sha,
        expected_input_hashes = {},
        expected_tables = empty_tables,
        expected_assessment_status = "restricted_evidence_summary_only",
        expected_assessment_status_counts = report["domain_status_counts"],
        expected_prediction_inventory_reconciliation = {}
    )


# Confirming stage performance and cross-stage transportability are separate domains
def test_stage_performance_and_transportability_are_separate() -> None:
    policy = load_policy()
    domains = {d["assessment_identity"]: d for d in policy["assessment_domains"]}

    assert "stage_stratified_performance_comparison" in domains
    assert "cross_stage_transportability" in domains
    assert "dataset_stage_stability" not in domains


# Confirming unsupported_condition_inventory is populated
def test_unsupported_condition_inventory_is_complete() -> None:
    from scripts.assess_robustness import build_unsupported_condition_inventory

    predictive_tables = {
        "candidate_registry": [
            {
                "benchmark_identity": "bench",
                "candidate_identity": "candidate",
                "model_family": "elastic_net",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "condition_protocol_identities": ["a", "b"]
            }
        ],
        "condition_calibration": [
            {
                "benchmark_identity": "bench",
                "condition_protocol_identity": "a",
                "calibration_status": "completed",
                "calibration_reason_code": ""
            },
            {
                "benchmark_identity": "bench",
                "condition_protocol_identity": "b",
                "calibration_status": "unavailable",
                "calibration_reason_code": "EMPTY_CONDITION_PREDICTION_INVENTORY"
            }
        ]
    }

    records = build_unsupported_condition_inventory(predictive_tables)

    assert len(records) == 1
    assert records[0]["condition_protocol_identity"] == "b"
    assert records[0]["calibration_status"] == "unavailable"


# Confirming boundary vocabulary is validated
def test_boundary_status_vocabulary_is_enforced() -> None:
    from scripts.assess_robustness import validate_emitted_codes_and_statuses

    policy = load_policy()

    tables = {
        "validation_summary": [],
        "model_family_comparisons": [],
        "prior_sensitivity": [],
        "influence_summary": [],
        "prediction_agreement": [],
        "cohort_sensitivity": [],
        "descriptive_specification_comparisons": [],
        "structural_limitations": [],
        "configuration_boundary_summary": [
            {
                "boundary_assessment_status": "invented_status",
                "boundary_classification": "selected_interior"
            }
        ]
    }

    with pytest.raises(RuntimeError, match = "unregistered boundary status"):
        validate_emitted_codes_and_statuses(tables, policy)


# Confirming regenerated coverage evidence counts reconcile in every row
def test_coverage_evidence_counts_reconcile_per_row() -> None:
    import csv

    policy_path = (
        PROJECT_ROOT
        / "configs"
        / "robustness_assessment_policy.json"
    )

    output_directory = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "robustness_assessment"
    )

    report_path = (
        output_directory
        / "robustness_assessment_report.json"
    )

    coverage_path = (
        output_directory
        / "assessment_coverage.csv"
    )

    if (
        not report_path.is_file()
        or not coverage_path.is_file()
    ):
        pytest.skip(
            "Robustness artifacts have not been generated."
        )

    report = json.loads(
        report_path.read_text(
            encoding = "utf-8"
        )
    )

    current_policy_sha256 = calculate_file_sha256(
        policy_path
    )

    if (
        report.get(
            "policy_sha256"
        )
        != current_policy_sha256
    ):
        pytest.skip(
            "Robustness artifacts were generated under a different policy."
        )

    with coverage_path.open(
        mode = "r",
        encoding = "utf-8",
        newline = ""
    ) as coverage_stream:
        rows = list(
            csv.DictReader(
                coverage_stream
            )
        )

    assert rows

    for row in rows:
        assert row[
            "evidence_unit"
        ]

        total = int(
            row[
                "total_evidence_count"
            ]
        )

        completed = int(
            row[
                "completed_evidence_count"
            ]
        )

        restricted = int(
            row[
                "restricted_evidence_count"
            ]
        )

        unavailable = int(
            row[
                "unavailable_evidence_count"
            ]
        )

        assert (
            completed
            + restricted
            + unavailable
            == total
        )


# Confirming the domain-status helper distinguishes all four completion levels
def test_derive_domain_status_from_counts():
    from scripts.assess_robustness import derive_domain_status_from_counts

    status, reason = derive_domain_status_from_counts(
        total_count = 0,
        completed_count = 0,
        restricted_count = 0,
        unavailable_count = 0,
        no_evidence_reason_code = "NO_EVIDENCE",
        partial_reason_code = "PARTIAL_VALIDATION_COVERAGE",
        all_diagnostic_reason_code = "ALL_EVIDENCE_DIAGNOSTIC_ONLY",
        all_unavailable_reason_code = "VALIDATION_METRICS_UNAVAILABLE"
    )
    assert status == "unavailable"
    assert reason == "NO_EVIDENCE"

    status, reason = derive_domain_status_from_counts(
        total_count = 5,
        completed_count = 5,
        restricted_count = 0,
        unavailable_count = 0,
        no_evidence_reason_code = "NO_EVIDENCE",
        partial_reason_code = "PARTIAL_VALIDATION_COVERAGE",
        all_diagnostic_reason_code = "ALL_EVIDENCE_DIAGNOSTIC_ONLY",
        all_unavailable_reason_code = "VALIDATION_METRICS_UNAVAILABLE"
    )
    assert status == "completed"
    assert reason is None

    status, reason = derive_domain_status_from_counts(
        total_count = 5,
        completed_count = 2,
        restricted_count = 2,
        unavailable_count = 1,
        no_evidence_reason_code = "NO_EVIDENCE",
        partial_reason_code = "PARTIAL_VALIDATION_COVERAGE",
        all_diagnostic_reason_code = "ALL_EVIDENCE_DIAGNOSTIC_ONLY",
        all_unavailable_reason_code = "VALIDATION_METRICS_UNAVAILABLE"
    )
    assert status == "completed_with_restrictions"
    assert reason == "PARTIAL_VALIDATION_COVERAGE"

    status, reason = derive_domain_status_from_counts(
        total_count = 5,
        completed_count = 0,
        restricted_count = 5,
        unavailable_count = 0,
        no_evidence_reason_code = "NO_EVIDENCE",
        partial_reason_code = "PARTIAL_VALIDATION_COVERAGE",
        all_diagnostic_reason_code = "ALL_EVIDENCE_DIAGNOSTIC_ONLY",
        all_unavailable_reason_code = "VALIDATION_METRICS_UNAVAILABLE"
    )
    assert status == "diagnostic_only"
    assert reason == "ALL_EVIDENCE_DIAGNOSTIC_ONLY"

    status, reason = derive_domain_status_from_counts(
        total_count = 5,
        completed_count = 0,
        restricted_count = 0,
        unavailable_count = 5,
        no_evidence_reason_code = "NO_EVIDENCE",
        partial_reason_code = "PARTIAL_VALIDATION_COVERAGE",
        all_diagnostic_reason_code = "ALL_EVIDENCE_DIAGNOSTIC_ONLY",
        all_unavailable_reason_code = "VALIDATION_METRICS_UNAVAILABLE"
    )
    assert status == "unavailable"
    assert reason == "VALIDATION_METRICS_UNAVAILABLE"


# Confirming horizon sensitivity registers both held-out rescoring and training-window refits
def test_horizon_evaluation_sensitivity_registers_rescoring_and_refitting():
    policy = load_policy()

    domains = {
        domain["assessment_identity"]: domain
        for domain in policy["assessment_domains"]
    }

    domain = domains[
        "horizon_evaluation_sensitivity"
    ]

    assert (
        domain["evaluation_rule"]
        == "held_out_evaluation_windows_and_training_window_refits"
    )

    assert (
        policy[
            "assessment_contract"
        ][
            "held_out_evaluation"
        ][
            "training_data_or_fitted_model_changes_permitted"
        ]
        is False
    )

    sensitivity_contract = policy[
        "assessment_contract"
    ][
        "predictive_sensitivity_evaluation"
    ]

    assert sensitivity_contract[
        "one_factor_at_a_time_required"
    ] is True

    assert "training_window" in sensitivity_contract[
        "required_variant_types"
    ]


# Confirming prior-sensitivity `incomplete` is restricted evidence, not unavailable
def test_prior_sensitivity_incomplete_counts_as_restricted():
    policy = load_policy()
    domains = {d["assessment_identity"]: d for d in policy["assessment_domains"]}

    assert (
        domains["prior_scale_sensitivity_comparison"]["evidence_source"]
        == "prior_sensitivity"
    )


# Confirming cell-influence `incomplete` is restricted evidence, not unavailable
def test_cell_influence_incomplete_counts_as_restricted():
    policy = load_policy()
    domains = {d["assessment_identity"]: d for d in policy["assessment_domains"]}

    assert (
        domains["cell_influence_assessment"]["evidence_source"]
        == "influence_summary"
    )


# Rejecting malformed embedded JSON fields
def test_malformed_embedded_json_is_rejected() -> None:
    from scripts.assess_robustness import parse_csv_value

    with pytest.raises(
        RuntimeError,
        match = "not valid JSON"
    ):
        parse_csv_value(
            '{"field":'
        )


# Rejecting duplicate fields in embedded JSON objects
def test_duplicate_embedded_json_fields_are_rejected() -> None:
    from scripts.assess_robustness import parse_csv_value

    with pytest.raises(
        RuntimeError,
        match = "duplicate field"
    ):
        parse_csv_value(
            '{"field":1,"field":2}'
        )


# Rejecting duplicate required record identities
def test_duplicate_required_record_identities_are_rejected() -> None:
    from scripts.assess_robustness import (
        validate_unique_record_identities
    )

    with pytest.raises(
        RuntimeError,
        match = "duplicate record identity"
    ):
        validate_unique_record_identities(
            records = [
                {
                    "candidate_identity": "candidate"
                },
                {
                    "candidate_identity": "candidate"
                }
            ],
            identity_fields = (
                "candidate_identity",
            ),
            table_name = "candidate_registry"
        )


# Rejecting a completed metric without its prediction inventory
def test_completed_metric_requires_prediction_inventory() -> None:
    from scripts.assess_robustness import (
        reconcile_prediction_inventories
    )

    with pytest.raises(
        RuntimeError,
        match = "no prediction inventory"
    ):
        reconcile_prediction_inventories(
            prediction_records = [],
            metric_records = [
                {
                    "candidate_identity": "candidate",
                    "validation_status": "completed",
                    "independent_cell_count": "2",
                    "observation_count": "4"
                }
            ],
            identity_field = "candidate_identity",
            prediction_status_field = None,
            metric_status_field = "validation_status",
            cell_count_field = "independent_cell_count",
            observation_count_field = "observation_count"
        )


# Rejecting prediction inventories without metric records
def test_prediction_inventory_requires_metric_record() -> None:
    from scripts.assess_robustness import (
        reconcile_prediction_inventories
    )

    with pytest.raises(
        RuntimeError,
        match = "do not map to metric records"
    ):
        reconcile_prediction_inventories(
            prediction_records = [
                {
                    "candidate_identity": "candidate",
                    "cell_identity": "cell",
                    "source_identity": "source",
                    "condition_protocol_identity": "condition",
                    "horizon": "1.0"
                }
            ],
            metric_records = [],
            identity_field = "candidate_identity",
            prediction_status_field = None,
            metric_status_field = "validation_status",
            cell_count_field = "independent_cell_count",
            observation_count_field = "observation_count"
        )


# Rejecting duplicate validation metrics
def test_duplicate_validation_metric_identities_are_rejected() -> None:
    from scripts.assess_robustness import (
        reconcile_prediction_inventories
    )

    metric = {
        "candidate_identity": "candidate",
        "validation_status": "unavailable",
        "independent_cell_count": "",
        "observation_count": ""
    }

    with pytest.raises(
        RuntimeError,
        match = "duplicate identity"
    ):
        reconcile_prediction_inventories(
            prediction_records = [],
            metric_records = [
                dict(
                    metric
                ),
                dict(
                    metric
                )
            ],
            identity_field = "candidate_identity",
            prediction_status_field = None,
            metric_status_field = "validation_status",
            cell_count_field = "independent_cell_count",
            observation_count_field = "observation_count"
        )


# Confirming the handoff and scientific statuses must agree
def test_handoff_status_must_match_scientific_status() -> None:
    report_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "robustness_assessment"
        / "robustness_assessment_report.json"
    )

    if not report_path.is_file():
        pytest.skip(
            "Robustness artifacts have not been generated."
        )

    report = json.loads(
        report_path.read_text(
            encoding = "utf-8"
        )
    )

    if (
        report.get(
            "policy_sha256"
        )
        != calculate_file_sha256(
            PROJECT_ROOT
            / "configs"
            / "robustness_assessment_policy.json"
        )
    ):
        pytest.skip(
            "Robustness artifacts were generated under a different policy."
        )

    assert (
        report[
            "analytical_handoff"
        ][
            "status"
        ]
        == report[
            "scientific_assessment_status"
        ]
    )


# Confirming interval diagnostics use equal-cell weighting across observations
def test_interval_diagnostics_use_equal_cell_weighting():
    from scripts.assess_robustness import (
        build_prediction_interval_diagnostics
    )

    records = [
        {
            "source_analysis": "predictive_modeling",
            "candidate_or_benchmark_identity": "benchmark",
            "model_name": "elastic_net",
            "dataset_stage": "Stage_1",
            "design_scope": "Stage_1",
            "sampling_design_scope": "combined",
            "aging_mode": "calendar_aging",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "analytical_status": "available",
            "fold_identity": "0",
            "cell_identity": "cell_a",
            "source_identity": "source_a_1",
            "condition_protocol_identity": "condition",
            "horizon": 1.0,
            "observed_outcome": 1.0,
            "predicted_outcome": 1.0,
            "interval_status": "completed",
            "interval_lower": 0.9,
            "interval_upper": 1.1,
            "nominal_interval_probability": 0.9
        },
        {
            "source_analysis": "predictive_modeling",
            "candidate_or_benchmark_identity": "benchmark",
            "model_name": "elastic_net",
            "dataset_stage": "Stage_1",
            "design_scope": "Stage_1",
            "sampling_design_scope": "combined",
            "aging_mode": "calendar_aging",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "analytical_status": "available",
            "fold_identity": "0",
            "cell_identity": "cell_a",
            "source_identity": "source_a_2",
            "condition_protocol_identity": "condition",
            "horizon": 2.0,
            "observed_outcome": 2.0,
            "predicted_outcome": 2.0,
            "interval_status": "completed",
            "interval_lower": 0.0,
            "interval_upper": 1.0,
            "nominal_interval_probability": 0.9
        },
        {
            "source_analysis": "predictive_modeling",
            "candidate_or_benchmark_identity": "benchmark",
            "model_name": "elastic_net",
            "dataset_stage": "Stage_1",
            "design_scope": "Stage_1",
            "sampling_design_scope": "combined",
            "aging_mode": "calendar_aging",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "analytical_status": "available",
            "fold_identity": "0",
            "cell_identity": "cell_b",
            "source_identity": "source_b",
            "condition_protocol_identity": "condition",
            "horizon": 1.0,
            "observed_outcome": 1.0,
            "predicted_outcome": 1.0,
            "interval_status": "completed",
            "interval_lower": 0.9,
            "interval_upper": 1.1,
            "nominal_interval_probability": 0.9
        }
    ]

    overall = next(
        record
        for record in build_prediction_interval_diagnostics(
            records
        )
        if record["grouping_dimension"] == "overall"
    )

    assert overall[
        "equal_cell_weighted_interval_coverage"
    ] == 0.75


# Confirming evaluation windows preserve the full inventory and only re-score
def test_horizon_windows_do_not_refit_or_change_full_inventory():
    from scripts.assess_robustness import (
        build_horizon_evaluation_sensitivity
    )

    records = []

    for cell_identity in ("cell_a", "cell_b"):
        for horizon in (1.0, 2.0, 3.0):
            records.append(
                {
                    "source_analysis": "predictive_modeling",
                    "candidate_or_benchmark_identity": "benchmark",
                    "model_name": "elastic_net",
                    "dataset_stage": "Stage_1",
                    "design_scope": "Stage_1",
                    "sampling_design_scope": "combined",
                    "aging_mode": "calendar_aging",
                    "outcome": "relative_capacity",
                    "cohort": "primary",
                    "horizon_axis": "elapsed_calendar_days",
                    "analytical_status": "available",
                    "fold_identity": "0",
                    "cell_identity": cell_identity,
                    "source_identity": f"{cell_identity}_{horizon}",
                    "condition_protocol_identity": "condition",
                    "horizon": horizon,
                    "observed_outcome": 1.0,
                    "predicted_outcome": 1.0,
                    "interval_status": "completed",
                    "interval_lower": 0.9,
                    "interval_upper": 1.1,
                    "nominal_interval_probability": 0.9
                }
            )

    result = build_horizon_evaluation_sensitivity(
        records,
        2
    )

    by_window = {
        record["evaluation_window_identity"]: record
        for record in result
    }

    assert by_window[
        "full_evaluation_inventory"
    ]["prediction_count"] == 6

    assert by_window[
        "first_follow_up_excluded"
    ]["prediction_count"] == 4

    assert by_window[
        "terminal_observation_excluded"
    ]["prediction_count"] == 4


# Confirming incident sensitivity re-scores without refitting
def test_incident_exclusion_is_evaluation_only():
    from scripts.assess_robustness import (
        build_incident_evaluation_sensitivity
    )

    prediction_records = []
    feature_records = []

    for index, cell_identity in enumerate(("cell_a", "cell_b")):
        source_identity = f"source_{index}"

        prediction_records.append(
            {
                "source_analysis": "predictive_modeling",
                "candidate_or_benchmark_identity": "benchmark",
                "model_name": "elastic_net",
                "dataset_stage": "Stage_1",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined",
                "aging_mode": "calendar_aging",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "analytical_status": "available",
                "fold_identity": "0",
                "cell_identity": cell_identity,
                "source_identity": source_identity,
                "condition_protocol_identity": "condition",
                "horizon": 1.0,
                "observed_outcome": 1.0,
                "predicted_outcome": 1.0,
                "interval_status": "completed",
                "interval_lower": 0.9,
                "interval_upper": 1.1,
                "nominal_interval_probability": 0.9
            }
        )

        feature_records.append(
            {
                "source_identity": source_identity,
                "cell_identity": cell_identity,
                "archive_quality_reason_codes": [],
                "quality_reason_codes": (
                    ["DOCUMENTED_INCIDENT"]
                    if index == 0
                    else []
                )
            }
        )

    result = build_incident_evaluation_sensitivity(
        prediction_records,
        feature_records,
        2
    )

    assert len(result) == 1
    assert result[0]["model_refitted"] is False
    assert result[0]["excluded_prediction_count"] == 1
    assert result[0]["assessment_status"] == "unavailable"
    assert result[0]["assessment_reason_code"] == (
        "INCIDENT_EXCLUSION_INSUFFICIENT_CELLS"
    )


# Confirming duplicate feature-provenance identities are rejected
def test_feature_provenance_duplicate_identity_is_rejected():
    from scripts.assess_robustness import (
        build_feature_reason_lookup
    )

    records = [
        {
            "source_identity": "source",
            "cell_identity": "cell",
            "archive_quality_reason_codes": [],
            "quality_reason_codes": []
        },
        {
            "source_identity": "source",
            "cell_identity": "cell",
            "archive_quality_reason_codes": [],
            "quality_reason_codes": []
        }
    ]

    with pytest.raises(
        RuntimeError,
        match = "duplicate identity"
    ):
        build_feature_reason_lookup(records)


# Confirming unmatched prediction inventories are excluded and reported
def test_exclude_unmatched_predictions_removes_orphans(capsys: pytest.CaptureFixture) -> None:
    from scripts.assess_robustness import (
        exclude_unmatched_predictions
    )

    input_tables = {
        "descriptive_modeling": {
            "validation_predictions": [
                {"candidate_identity": "keep_a"},
                {"candidate_identity": "drop_b"},
                {"candidate_identity": "keep_a"}
            ],
            "validation_metrics": [
                {"candidate_identity": "keep_a"}
            ]
        },
        "bayesian_modeling": {
            "grouped_validation_predictions": [
                {"candidate_identity": "keep_c"},
                {"candidate_identity": "drop_d"}
            ],
            "grouped_validation_metrics": [
                {"candidate_identity": "keep_c"}
            ]
        },
        "predictive_modeling": {
            "validation_predictions": [
                {"benchmark_identity": "keep_e"}
            ],
            "validation_metrics": [
                {"benchmark_identity": "keep_e"}
            ]
        }
    }

    reconciliation = exclude_unmatched_predictions(
        input_tables
    )

    assert [
        prediction["candidate_identity"]
        for prediction in input_tables[
            "descriptive_modeling"
        ][
            "validation_predictions"
        ]
    ] == [
        "keep_a",
        "keep_a"
    ]

    assert [
        prediction["candidate_identity"]
        for prediction in input_tables[
            "bayesian_modeling"
        ][
            "grouped_validation_predictions"
        ]
    ] == [
        "keep_c"
    ]

    assert reconciliation[
        "descriptive_modeling"
    ] == {
        "identity_field": "candidate_identity",
        "excluded_identity_count": 1,
        "excluded_prediction_count": 1,
        "excluded_identities": ["drop_b"],
        "reason_code": (
            "PREDICTION_INVENTORY_WITHOUT_METRIC_RECORD"
        )
    }

    assert reconciliation[
        "bayesian_modeling"
    ][
        "excluded_identities"
    ] == [
        "drop_d"
    ]

    assert reconciliation[
        "predictive_modeling"
    ][
        "excluded_identity_count"
    ] == 0

    assert reconciliation[
        "predictive_modeling"
    ][
        "reason_code"
    ] is None

    captured = capsys.readouterr()

    assert (
        "WARNING : 1 descriptive_modeling"
        in captured.out
    )

    assert (
        "WARNING : 1 bayesian_modeling"
        in captured.out
    )


# Confirming an all-unavailable interval population is reported without crashing
def test_interval_diagnostics_handles_empty_available_intervals() -> None:
    from scripts.assess_robustness import (
        build_prediction_interval_diagnostics
    )

    records = [
        {
            "source_analysis": "descriptive_modeling",
            "candidate_or_benchmark_identity": "benchmark",
            "model_name": "linear_common_trajectory",
            "dataset_stage": "Stage_1",
            "design_scope": "Stage_1",
            "sampling_design_scope": "combined",
            "aging_mode": "calendar_aging",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "analytical_status": "available",
            "fold_identity": "0",
            "cell_identity": "cell_a",
            "source_identity": "source_a",
            "condition_protocol_identity": "condition",
            "horizon": 1.0,
            "observed_outcome": 1.0,
            "predicted_outcome": 1.0,
            "interval_status": "unavailable",
            "interval_lower": None,
            "interval_upper": None,
            "nominal_interval_probability": 0.95
        }
    ]

    result = build_prediction_interval_diagnostics(records)

    overall = next(
        record
        for record in result
        if record["grouping_dimension"] == "overall"
    )

    assert overall["interval_available_count"] == 0
    assert overall["interval_covered_count"] == 0
    assert overall["equal_cell_weighted_interval_coverage"] is None
    assert overall["equal_cell_weighted_mean_interval_width"] is None
    assert overall["assessment_status"] == "unavailable"
    assert overall["assessment_reason_code"] == (
        "PREDICTION_INTERVALS_UNAVAILABLE"
    )


# Confirming the predictive-sensitivity summary reconciles every registered variant
@pytest.mark.production_artifacts
def test_predictive_sensitivity_summary_reconciles_current_artifacts() -> None:
    from scripts.assess_robustness import (
        build_predictive_sensitivity_summary,
        verify_report_tables
    )

    policy = load_policy()

    report_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "predictive_sensitivity"
        / "predictive_sensitivity_report.json"
    )

    report, tables = verify_report_tables(
        report_path,
        policy["inputs"]["required_report_sha256"][
            "predictive_sensitivity"
        ],
        policy["inputs"]["required_tables"][
            "predictive_sensitivity"
        ]
    )

    summary = build_predictive_sensitivity_summary(
        tables
    )

    assert len(summary) == report[
        "sensitivity_variant_count"
    ]

    assert {
        record["variant_type"]
        for record in summary
    } == {
        "partition",
        "expanded_config",
        "training_window",
        "incident",
        "ablation"
    }

    observed_status_counts = Counter(
        record["assessment_status"]
        for record in summary
    )

    assert dict(
        sorted(
            observed_status_counts.items()
        )
    ) == report[
        "comparison_status_counts"
    ]


# Rejecting a predictive-sensitivity summary whose variant inventories do not reconcile
def test_predictive_sensitivity_summary_rejects_inventory_mismatch() -> None:
    from scripts.assess_robustness import (
        build_predictive_sensitivity_summary
    )

    tables = {
        "sensitivity_registry": [
            {
                "variant_identity": "variant",
                "variant_type": "partition"
            }
        ],
        "validation_metrics": [],
        "sensitivity_comparisons": [],
        "prediction_interval_diagnostics": []
    }

    with pytest.raises(
        RuntimeError,
        match = "inventories do not reconcile"
    ):
        build_predictive_sensitivity_summary(
            tables
        )


# Confirming a sensitivity-domain evaluation counts diagnostic-only variants as restricted
def test_sensitivity_domain_evaluation_counts_restrictions() -> None:
    from scripts.assess_robustness import (
        sensitivity_domain_evaluation
    )

    records = [
        {
            "variant_type": "partition",
            "assessment_status": "completed"
        },
        {
            "variant_type": "partition",
            "assessment_status": "diagnostic_only"
        },
        {
            "variant_type": "partition",
            "assessment_status": "unavailable"
        },
        {
            "variant_type": "ablation",
            "assessment_status": "completed"
        }
    ]

    result = sensitivity_domain_evaluation(
        records,
        {"partition"}
    )

    assert result == {
        "status": "completed_with_restrictions",
        "total_count": 3,
        "completed_count": 1,
        "restricted_count": 1,
        "unavailable_count": 1,
        "reason_code": "PARTIAL_EVIDENCE_COVERAGE"
    }


# Confirming the empirical-domain evaluation counts diagnostic-only records as restricted
def test_empirical_domain_evaluation_counts_diagnostic_only() -> None:
    from scripts.assess_robustness import (
        empirical_domain_evaluation
    )

    result = empirical_domain_evaluation(
        [
            {
                "assessment_status": "completed"
            },
            {
                "assessment_status": "diagnostic_only"
            }
        ],
        "NO_EVIDENCE"
    )

    assert result["status"] == "completed_with_restrictions"
    assert result["completed_count"] == 1
    assert result["restricted_count"] == 1
    assert result["unavailable_count"] == 0


# Rejecting a result-informed registration status paired with a non-result-informed amendment
def test_result_informed_registration_requires_result_informed_amendment() -> None:
    policy = copy.deepcopy(
        load_policy()
    )

    policy["registration_status"] = (
        "result_informed_after_evidence_review"
    )

    policy["amendment"]["result_informed"] = False
    policy["amendment"]["result_informed_scope"] = ""

    with pytest.raises(
        RuntimeError,
        match = "Result-informed registration requires"
    ):
        validate_policy(
            policy
        )


# Rejecting a sensitivity summary whose ablation or incident domain summary inventory is incomplete
@pytest.mark.production_artifacts
@pytest.mark.parametrize(
    (
        "table_name",
        "expected_message"
    ),
    [
        (
            "predictor_ablation",
            "Predictor-ablation summary inventory"
        ),
        (
            "incident_exclusions",
            "Incident-exclusion summary inventory"
        )
    ]
)
def test_predictive_sensitivity_summary_rejects_missing_domain_summary(table_name: str, expected_message: str) -> None:
    from scripts.assess_robustness import (
        build_predictive_sensitivity_summary,
        verify_report_tables
    )

    policy = load_policy()

    report_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "predictive_sensitivity"
        / "predictive_sensitivity_report.json"
    )

    _, tables = verify_report_tables(
        report_path,
        policy["inputs"]["required_report_sha256"][
            "predictive_sensitivity"
        ],
        policy["inputs"]["required_tables"][
            "predictive_sensitivity"
        ]
    )

    tables = copy.deepcopy(
        tables
    )

    tables[table_name] = tables[table_name][1:]

    with pytest.raises(
        RuntimeError,
        match = expected_message
    ):
        build_predictive_sensitivity_summary(
            tables
        )


# Rejecting an incident exclusion marked ineligible in at least one fold while claiming a completed comparison
@pytest.mark.production_artifacts
def test_incident_ineligibility_cannot_produce_available_comparison() -> None:
    from scripts.assess_robustness import (
        build_predictive_sensitivity_summary,
        verify_report_tables
    )

    policy = load_policy()

    report_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "predictive_sensitivity"
        / "predictive_sensitivity_report.json"
    )

    _, tables = verify_report_tables(
        report_path,
        policy["inputs"]["required_report_sha256"][
            "predictive_sensitivity"
        ],
        policy["inputs"]["required_tables"][
            "predictive_sensitivity"
        ]
    )

    tables = copy.deepcopy(
        tables
    )

    source_record = tables["incident_exclusions"][0]
    source_record["eligible_in_every_fold"] = False
    source_record["comparison_status"] = "completed"

    comparison_record = next(
        record
        for record in tables["sensitivity_comparisons"]
        if record["variant_identity"]
        == source_record["variant_identity"]
    )

    comparison_record["comparison_status"] = "completed"

    metric_record = next(
        record
        for record in tables["validation_metrics"]
        if record["variant_identity"]
        == source_record["variant_identity"]
    )

    source_record["variant_analytical_status"] = (
        metric_record["analytical_status"]
    )

    with pytest.raises(
        RuntimeError,
        match = "Ineligible incident exclusion"
    ):
        build_predictive_sensitivity_summary(
            tables
        )
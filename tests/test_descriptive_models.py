import os
import json
import math
import numpy as np
import pytest

from pathlib import Path
from collections import Counter

import scripts.run_descriptive_models as descriptive_models

from scripts.run_descriptive_models import (
    assemble_candidate_observations,
    build_candidate_registry,
    build_deterministic_identity,
    load_strict_json_object,
    select_model_candidates,
    validate_analytical_scope,
    validate_modeling_policy,
    build_equal_cell_weights,
    build_model_encoding,
    evaluate_candidate_admissibility,
    transform_model_matrix,
    calculate_equal_cell_residual_diagnostics,
    fit_candidate_models,
    fit_linear_mixed_effects,
    fit_weighted_least_squares,
    calculate_uncertainty_and_influence,
    fit_prepared_model,
    build_validation_fold_assignments,
    calculate_grouped_validation_metrics,
    horizon_inside_candidate_support,
    build_condition_calibration,
    build_model_comparisons,
    build_trajectory_summaries,
    build_model_contrasts,
    assign_final_candidate_dispositions,
    build_artifact_table_schema,
    normalize_artifact_records,
    publish_staged_artifacts,
    serialize_artifact_value,
    validate_written_artifact_table,
    write_artifact_table,
    validate_policy_artifact_inventory,
    build_validation_fold_identity,
    validate_selected_matrix_term_accounting,
    build_validation_fixed_effect_covariance,
    build_validation_prediction_intervals,
    build_modeling_execution_plan,
    build_candidate_outcome_summary,
    build_fitted_result_availability,
    resolve_completed_analysis_status,
    build_longitudinal_structure_diagnostics,
    assign_candidate_analytical_result_status,
    build_analytical_handoff_summary,
    validate_descriptive_modeling_report,
    prepare_candidate_model_matrices,
    run_grouped_validation,
    validate_modeling_reconciliation
)


POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "descriptive_modeling_policy.json"
)


# Loading the production modeling policy
def load_test_policy() -> dict:
    return load_strict_json_object(
        POLICY_PATH
    )


# Rejecting duplicate fields in the modeling policy
def test_load_strict_json_object_rejects_duplicate_fields(tmp_path: Path) -> None:
    duplicate_path = (
        tmp_path
        / "duplicate.json"
    )

    duplicate_path.write_text(
        (
            "{\n"
            '  "policy_version": "1.0",\n'
            '  "policy_version": "1.1"\n'
            "}\n"
        ),
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = "duplicate field"
    ):
        load_strict_json_object(
            duplicate_path
        )


# Confirming that deterministic identities ignore dictionary order
def test_build_deterministic_identity_is_order_independent() -> None:
    first_identity = build_deterministic_identity(
        {
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days"
        }
    )

    second_identity = build_deterministic_identity(
        {
            "horizon_axis": "elapsed_calendar_days",
            "cohort": "primary",
            "outcome": "relative_capacity"
        }
    )

    assert first_identity == second_identity


# Confirming the production policy's required invariants
def test_validate_modeling_policy() -> None:
    modeling_policy = load_test_policy()

    validate_modeling_policy(
        modeling_policy
    )

    assert (
        modeling_policy[
            "analysis_scope"
        ][
            "independent_unit"
        ]
        == "cell_identity"
    )

    assert (
        modeling_policy[
            "validation"
        ][
            "cell_leakage_permitted"
        ]
        is False
    )

    assert (
        modeling_policy[
            "missing_data"
        ][
            "imputation_permitted"
        ]
        is False
    )


# Replacing one nested policy value for validation tests
def replace_policy_value(modeling_policy: dict, policy_path: tuple[str, ...],
                         replacement_value: object
) -> None:
    parent_value = modeling_policy

    for field_name in policy_path[
        :-1
    ]:
        parent_value = parent_value[
            field_name
        ]

    parent_value[
        policy_path[-1]
    ] = replacement_value


@pytest.mark.parametrize(
    (
        "policy_path",
        "replacement_value"
    ),
    [
        (
            (
                "admissibility",
                "minimum_independent_cells"
            ),
            5
        ),
        (
            (
                "estimation",
                "weighted_least_squares",
                "observation_weight"
            ),
            "uniform_observation_weight"
        ),
        (
            (
                "restricted_cubic_spline",
                "knot_probabilities"
            ),
            [
                0.0,
                0.25,
                1.0
            ]
        ),
        (
            (
                "validation",
                "maximum_fold_count"
            ),
            5
        ),
        (
            (
                "uncertainty",
                "coefficient_covariance"
            ),
            "model_based_covariance"
        ),
        (
            (
                "artifact_contract",
                "table_schemas_declared_in_implementation"
            ),
            False
        ),
        (
            (
                "candidate_generation",
                "candidate_identity_fields"
            ),
            [
                "support_region_identity",
                "template_name"
            ]
        ),
        (
            (
                "observation_assembly",
                "observations_outside_evaluation_interval"
            ),
            "permitted_for_member_cells_during_model_fitting"
        ),
        (
            (
                "trajectory_summaries",
                "cell_summary_fields"
            ),
            [
                "observation_count"
            ]
        )
    ]
)

# Rejecting policy values not implemented by the analytical workflow
def test_validate_modeling_policy_rejects_unsupported_value(policy_path: tuple[str, ...], replacement_value: object
) -> None:
    modeling_policy = json.loads(
        json.dumps(
            load_test_policy()
        )
    )

    replace_policy_value(
        modeling_policy = modeling_policy,
        policy_path = policy_path,
        replacement_value = replacement_value
    )

    with pytest.raises(
        RuntimeError,
        match = "not supported by the current implementation"
    ):
        validate_modeling_policy(
            modeling_policy
        )


# Rejecting an unimplemented model template
def test_validate_modeling_policy_rejects_unimplemented_template() -> None:
    modeling_policy = json.loads(
        json.dumps(
            load_test_policy()
        )
    )

    modeling_policy[
        "candidate_templates"
    ][0][
        "estimator"
    ] = "unsupported_estimator"

    with pytest.raises(
        RuntimeError,
        match = "candidate templates"
    ):
        validate_modeling_policy(
            modeling_policy
        )


# Rejecting undeclared top-level policy sections
def test_validate_modeling_policy_rejects_unexpected_section() -> None:
    modeling_policy = json.loads(
        json.dumps(
            load_test_policy()
        )
    )

    modeling_policy[
        "undeclared_section"
    ] = {}

    with pytest.raises(
        RuntimeError,
        match = "unexpected"
    ):
        validate_modeling_policy(
            modeling_policy
        )


# Requiring a complete rerun after every policy amendment
def test_validate_modeling_policy_requires_complete_rerun() -> None:
    modeling_policy = json.loads(
        json.dumps(
            load_test_policy()
        )
    )

    modeling_policy[
        "amendment"
    ][
        "complete_rerun_required"
    ] = False

    with pytest.raises(
        RuntimeError,
        match = "complete rerun"
    ):
        validate_modeling_policy(
            modeling_policy
        )


# Accepting the complete declared analytical-artifact inventory
def test_validate_policy_artifact_inventory() -> None:
    validate_policy_artifact_inventory(
        load_test_policy()
    )


# Rejecting an incomplete analytical-artifact inventory
def test_validate_policy_artifact_inventory_rejects_missing_table() -> None:
    modeling_policy = json.loads(
        json.dumps(
            load_test_policy()
        )
    )

    modeling_policy[
        "artifact_contract"
    ][
        "output_tables"
    ].remove(
        "model_contrasts"
    )

    with pytest.raises(
        RuntimeError,
        match = "output-table inventory"
    ):
        validate_policy_artifact_inventory(
            modeling_policy
        )


# Rejecting an unsafe report path
def test_validate_policy_artifact_inventory_rejects_report_path() -> None:
    modeling_policy = json.loads(
        json.dumps(
            load_test_policy()
        )
    )

    modeling_policy[
        "artifact_contract"
    ][
        "report_file_name"
    ] = "../descriptive_modeling_report.json"

    with pytest.raises(
        RuntimeError,
        match = "invalid report file name"
    ):
        validate_policy_artifact_inventory(
            modeling_policy
        )


# Building one valid analytical-scope record
def build_scope_record(term_name: str, term_type: str) -> dict:
    return {
        "support_region_identity": "region-1",
        "support_region_kind": "interval",
        "support_interval_minimum": "10.0",
        "support_interval_maximum": "20.0",
        "support_interval_lower_inclusive": "true",
        "support_interval_upper_inclusive": "false",
        "aging_mode": "calendar_aging",
        "design_scope": "Stage_1",
        "sampling_design_scope": "combined_within_scope",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": "elapsed_calendar_days",
        "analysis_role": "within_cell_bracketing_support",
        "analysis_population": "bracketing_range_cell_design",
        "condition_protocol_count": "3",
        "independent_cell_count": "9",
        "analytical_observation_count": "54",
        "specification": "main_effects",
        "context_family": "no_additional_context",
        "permitted_adjustment_variables": "[]",
        "target_estimand_available": "true",
        "target_estimand_status": "available",
        "term_name": term_name,
        "term_type": term_type,
        "structurally_estimable": "true",
        "exposure_time_interpretation": (
            "whole_assigned_protocol_gross_elapsed_calendar_time"
        ),
        "required_validation_grouping": '["cell_identity"]',
        "stage_pooling_status": "stage_specific",
        "causal_identification_status": "not_established",
        "question_classification": (
            "supported_with_restrictions"
        ),
        "authorization_status": (
            "permitted_for_model_evaluation_with_restrictions"
        )
    }


# Confirming complete candidate-template expansion
def test_build_candidate_registry_expands_every_template() -> None:
    modeling_policy = load_test_policy()

    scope_records = [
        build_scope_record(
            term_name = (
                "ambient_temperature_celsius"
            ),
            term_type = "main_effect"
        ),
        build_scope_record(
            term_name = (
                "maximum_state_of_charge_fraction"
            ),
            term_type = "main_effect"
        )
    ]

    validate_analytical_scope(
        scope_records
    )

    (
        candidate_records,
        candidate_term_records
    ) = build_candidate_registry(
        scope_records = scope_records,
        modeling_policy = modeling_policy
    )

    expected_candidate_count = len(
        modeling_policy[
            "candidate_templates"
        ]
    )

    assert (
        len(
            candidate_records
        )
        == expected_candidate_count
    )

    source_term_count = len(
        scope_records
    )

    expected_term_count = sum(
        (
            source_term_count
            + 1
            + (
                2
                if candidate_template[
                    "horizon_basis"
                ]
                == "restricted_cubic_spline"
                else 1
            )
            + (
                (
                    2
                    if candidate_template[
                        "horizon_basis"
                    ]
                    == "restricted_cubic_spline"
                    else 1
                )
                * source_term_count
            )
        )
        for candidate_template in modeling_policy[
            "candidate_templates"
        ]
    )

    assert (
        len(
            candidate_term_records
        )
        == expected_term_count
    )

    assert len(
        {
            candidate_record[
                "candidate_identity"
            ]
            for candidate_record in candidate_records
        }
    ) == expected_candidate_count


# Confirming complete derived-term accounting for every template
def test_build_candidate_registry_accounts_for_derived_terms() -> None:
    modeling_policy = load_test_policy()

    scope_records = [
        build_scope_record(
            term_name = (
                "ambient_temperature_celsius"
            ),
            term_type = "main_effect"
        )
    ]

    (
        candidate_records,
        candidate_term_records
    ) = build_candidate_registry(
        scope_records = scope_records,
        modeling_policy = modeling_policy
    )

    candidate_by_template = {
        candidate_record[
            "template_name"
        ]: candidate_record
        for candidate_record in candidate_records
    }

    terms_by_candidate = {}

    for candidate_record in candidate_records:
        candidate_identity = candidate_record[
            "candidate_identity"
        ]

        terms_by_candidate[
            candidate_identity
        ] = {
            (
                candidate_term_record[
                    "term_type"
                ],
                candidate_term_record[
                    "term_name"
                ]
            ): candidate_term_record
            for candidate_term_record in (
                candidate_term_records
            )
            if candidate_term_record[
                "candidate_identity"
            ] == candidate_identity
        }

    linear_common_candidate = candidate_by_template[
        "linear_common_trajectory"
    ]

    linear_common_terms = terms_by_candidate[
        linear_common_candidate[
            "candidate_identity"
        ]
    ]

    assert linear_common_terms[
        (
            "intercept",
            "intercept"
        )
    ][
        "term_disposition"
    ] == "selected"

    assert linear_common_terms[
        (
            "horizon_basis",
            "horizon_linear"
        )
    ][
        "term_disposition"
    ] == "selected"

    assert linear_common_terms[
        (
            "horizon_by_treatment_interaction",
            (
                "horizon_linear:"
                "ambient_temperature_celsius"
            )
        )
    ][
        "term_disposition"
    ] == "not_selected"

    linear_differential_candidate = (
        candidate_by_template[
            "linear_differential_trajectory"
        ]
    )

    linear_differential_terms = terms_by_candidate[
        linear_differential_candidate[
            "candidate_identity"
        ]
    ]

    assert linear_differential_terms[
        (
            "horizon_by_treatment_interaction",
            (
                "horizon_linear:"
                "ambient_temperature_celsius"
            )
        )
    ][
        "term_disposition"
    ] == "selected"

    spline_differential_candidate = (
        candidate_by_template[
            (
                "restricted_cubic_spline_"
                "differential_trajectory"
            )
        ]
    )

    spline_differential_terms = terms_by_candidate[
        spline_differential_candidate[
            "candidate_identity"
        ]
    ]

    assert spline_differential_terms[
        (
            "horizon_basis",
            "horizon_restricted_cubic"
        )
    ][
        "term_disposition"
    ] == "selected"

    assert spline_differential_terms[
        (
            "horizon_by_treatment_interaction",
            (
                "horizon_restricted_cubic:"
                "ambient_temperature_celsius"
            )
        )
    ][
        "term_disposition"
    ] == "selected"


# Rejecting conflicting candidate-level fields
def test_build_candidate_registry_rejects_conflicting_fields() -> None:
    modeling_policy = load_test_policy()

    first_record = build_scope_record(
        term_name = (
            "ambient_temperature_celsius"
        ),
        term_type = "main_effect"
    )

    second_record = build_scope_record(
        term_name = (
            "maximum_state_of_charge_fraction"
        ),
        term_type = "main_effect"
    )

    second_record[
        "independent_cell_count"
    ] = "12"

    with pytest.raises(
        RuntimeError,
        match = "conflicting values"
    ):
        build_candidate_registry(
            scope_records = [
                first_record,
                second_record
            ],
            modeling_policy = modeling_policy
        )


# Rejecting non-stage-specific analytical records
def test_validate_analytical_scope_rejects_stage_pooling() -> None:
    scope_record = build_scope_record(
        term_name = (
            "ambient_temperature_celsius"
        ),
        term_type = "main_effect"
    )

    scope_record[
        "stage_pooling_status"
    ] = "pooled_stages"

    with pytest.raises(
        RuntimeError,
        match = "non-stage-specific"
    ):
        validate_analytical_scope(
            [
                scope_record
            ]
        )


# Rejecting unauthorized analytical records
def test_validate_analytical_scope_rejects_unauthorized_record() -> None:
    scope_record = build_scope_record(
        term_name = (
            "ambient_temperature_celsius"
        ),
        term_type = "main_effect"
    )

    scope_record[
        "authorization_status"
    ] = "not_authorized"

    with pytest.raises(
        RuntimeError,
        match = "unauthorized record"
    ):
        validate_analytical_scope(
            [
                scope_record
            ]
        )


# Confirming hierarchy failure produces an inadmissible candidate
def test_build_candidate_registry_marks_missing_hierarchy_component() -> None:
    modeling_policy = load_test_policy()

    interaction_record = build_scope_record(
        term_name = "temperature:SOC",
        term_type = "pairwise_interaction"
    )

    interaction_record["specification"] = (
        "main_effects_and_pairwise_interactions"
    )

    validate_analytical_scope(
        [interaction_record]
    )

    (
        candidate_records,
        candidate_term_records
    ) = build_candidate_registry(
        scope_records = [interaction_record],
        modeling_policy = modeling_policy
    )

    assert all(
        candidate_record[
            "interaction_hierarchy_status"
        ] == "inadmissible"
        for candidate_record in candidate_records
    )

    assert all(
        candidate_record[
            "interaction_hierarchy_reason_code"
        ] == "REQUIRED_LOWER_ORDER_TERM_NOT_AUTHORIZED"
        for candidate_record in candidate_records
    )

    assert all(
        candidate_record[
            "admissibility_status"
        ] == "inadmissible"
        for candidate_record in candidate_records
    )

    required_component_records = [
        candidate_term_record
        for candidate_term_record in (
            candidate_term_records
        )
        if (
            candidate_term_record[
                "term_disposition"
            ]
            == "required_but_not_authorized"
        )
    ]

    assert {
        candidate_term_record[
            "term_name"
        ]
        for candidate_term_record in (
            required_component_records
        )
    } == {
        "temperature",
        "SOC"
    }

    assert all(
        candidate_term_record[
            "term_source"
        ]
        == "interaction_hierarchy"
        for candidate_term_record in (
            required_component_records
        )
    )


# Confirming complete hierarchy produces a pending candidate
def test_build_candidate_registry_marks_complete_hierarchy() -> None:
    modeling_policy = load_test_policy()

    lower_order_records = [
        build_scope_record(
            term_name = "temperature",
            term_type = "main_effect"
        ),
        build_scope_record(
            term_name = "SOC",
            term_type = "main_effect"
        )
    ]

    lower_order_records[0]["specification"] = (
        "main_effects_and_pairwise_interactions"
    )
    lower_order_records[1]["specification"] = (
        "main_effects_and_pairwise_interactions"
    )

    interaction_record = build_scope_record(
        term_name = "temperature:SOC",
        term_type = "pairwise_interaction"
    )

    interaction_record["specification"] = (
        "main_effects_and_pairwise_interactions"
    )

    scope_records = [
        *lower_order_records,
        interaction_record
    ]

    validate_analytical_scope(
        scope_records
    )

    (
        candidate_records,
        candidate_term_records
    ) = build_candidate_registry(
        scope_records = scope_records,
        modeling_policy = modeling_policy
    )

    assert all(
        candidate_record[
            "interaction_hierarchy_status"
        ] == "complete"
        for candidate_record in candidate_records
    )

    assert all(
        candidate_record[
            "admissibility_status"
        ] == "pending"
        for candidate_record in candidate_records
    )

    source_term_records = [
        candidate_term_record
        for candidate_term_record in (
            candidate_term_records
        )
        if candidate_term_record[
            "term_source"
        ] == "analytical_scope.csv"
    ]

    assert source_term_records

    assert all(
        candidate_term_record[
            "term_disposition"
        ]
        == "selected"
        for candidate_term_record in (
            source_term_records
        )
    )


# Building one minimal cell record
def build_cell_record(cell_identity: str, condition_identity: str, cell_number: int) -> dict:
    return {
        "cell_identity": cell_identity,
        "condition_protocol_identity": (
            condition_identity
        ),
        "dataset_stage": "Stage_1",
        "aging_mode": "calendar_aging",
        "ambient_temperature_celsius": "23.0",
        "maximum_state_of_charge_fraction": "0.5",
        "depth_of_discharge_fraction": "0.0",
        "charge_rate_c": "0.0",
        "discharge_rate_c": "0.0",
        "reference_test_schedule": (
            "periodic_reference_tests"
        ),
        "laboratory": "INT",
        "sampling_method": "FF",
        "cell_number": str(
            cell_number
        )
    }


# Building one support-membership record
def build_membership_record(cell_identity: str, condition_identity: str) -> dict:
    return {
        "support_region_identity": "region-1",
        "aging_mode": "calendar_aging",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": "elapsed_calendar_days",
        "region_kind": "interval",
        "lower_horizon": "10.0",
        "upper_horizon": "20.0",
        "lower_inclusive": "true",
        "upper_inclusive": "false",
        "cell_identity": cell_identity,
        "condition_protocol_identity": (
            condition_identity
        ),
        "dataset_stage": "Stage_1",
        "laboratory": "INT",
        "sampling_method": "FF",
        "supporting_cell_eligible_observation_count": "1",
        "cell_horizon_minimum": "10.0",
        "cell_horizon_maximum": "20.0",
        "support_geometry": (
            "within_cell_minimum_maximum_bracketing_range"
        ),
        "continuous_observation_status": (
            "not_established_between_observed_horizons"
        ),
        "interpolation_authorization_status": (
            "requires_model_specific_validation"
        ),
        "combined_region_identity": "combined-1"
    }


# Building one reference-test record
def build_reference_record(cell_identity: str, sequence_number: int,
                           horizon: str, outcome: str,
                           eligible: str = "true", fec_complete: str = ""
) -> dict:
    return {
        "source_identity": (
            f"{cell_identity}/"
            f"measurement_{sequence_number:02d}.csv"
        ),
        "cell_identity": cell_identity,
        "elapsed_calendar_days": horizon,
        "cumulative_full_equivalent_cycles_observed": "",
        "relative_capacity": outcome,
        "relative_dc_internal_resistance": "",
        "relative_capacity_primary_eligible": (
            eligible
        ),
        "relative_capacity_sensitivity_eligible": (
            eligible
        ),
        "relative_dc_internal_resistance_primary_eligible": (
            "false"
        ),
        "relative_dc_internal_resistance_sensitivity_eligible": (
            "false"
        ),
        "cumulative_fec_observation_complete": (
            fec_complete
        )
    }


# Confirming deterministic outcome-blind anchor selection
def test_select_model_candidates_uses_declared_anchor_order() -> None:
    modeling_policy = load_test_policy()

    first_scope_records = [
        build_scope_record(
            term_name = (
                "ambient_temperature_celsius"
            ),
            term_type = "main_effect"
        )
    ]

    second_scope_record = build_scope_record(
        term_name = (
            "ambient_temperature_celsius"
        ),
        term_type = "main_effect"
    )

    second_scope_record[
        "support_region_identity"
    ] = "region-2"

    second_scope_record[
        "support_interval_minimum"
    ] = "10.0"

    second_scope_record[
        "support_interval_maximum"
    ] = "30.0"

    second_scope_record[
        "condition_protocol_count"
    ] = "2"

    second_scope_record[
        "independent_cell_count"
    ] = "6"

    scope_records = [
        *first_scope_records,
        second_scope_record
    ]

    (
        candidate_records,
        candidate_term_records
    ) = build_candidate_registry(
        scope_records = scope_records,
        modeling_policy = modeling_policy
    )

    select_model_candidates(
        candidate_records = candidate_records,
        candidate_term_records = (
            candidate_term_records
        ),
        modeling_policy = modeling_policy
    )

    coverage_candidates = [
        candidate_record
        for candidate_record in candidate_records
        if (
            candidate_record[
                "selection_reason_code"
            ]
            == "COVERAGE_ANCHOR"
        )
    ]

    duration_candidates = [
        candidate_record
        for candidate_record in candidate_records
        if (
            candidate_record[
                "selection_reason_code"
            ]
            == "DURATION_ANCHOR"
        )
    ]

    assert {
        candidate_record[
            "support_region_identity"
        ]
        for candidate_record in coverage_candidates
    } == {
        "region-1"
    }

    assert {
        candidate_record[
            "support_region_identity"
        ]
        for candidate_record in duration_candidates
    } == {
        "region-2"
    }

    assert len(
        coverage_candidates
    ) == 6

    assert len(
        duration_candidates
    ) == 3


# Confirming mutually exclusive observation accounting
def test_assemble_candidate_observations_reconciles_statuses() -> None:
    modeling_policy = load_test_policy()

    cell_records = []

    membership_records = []

    reference_records = []

    condition_identities = [
        "Stage_1/TP_k01/periodic_reference_tests",
        "Stage_1/TP_k02/periodic_reference_tests"
    ]

    for condition_index, condition_identity in enumerate(
        condition_identities,
        start = 1
    ):
        for cell_number in range(
            1,
            4
        ):
            cell_identity = (
                f"Stage_1/TP_k{condition_index:02d}_{cell_number:02d}"
            )

            cell_records.append(
                build_cell_record(
                    cell_identity = cell_identity,
                    condition_identity = (
                        condition_identity
                    ),
                    cell_number = cell_number
                )
            )

            membership_records.append(
                build_membership_record(
                    cell_identity = cell_identity,
                    condition_identity = (
                        condition_identity
                    )
                )
            )

            reference_records.extend(
                [
                    build_reference_record(
                        cell_identity = cell_identity,
                        sequence_number = 1,
                        horizon = "0.0",
                        outcome = "1.0"
                    ),
                    build_reference_record(
                        cell_identity = cell_identity,
                        sequence_number = 2,
                        horizon = "9.0",
                        outcome = "0.99"
                    ),
                    build_reference_record(
                        cell_identity = cell_identity,
                        sequence_number = 3,
                        horizon = "10.0",
                        outcome = "0.985"
                    ),
                    build_reference_record(
                        cell_identity = cell_identity,
                        sequence_number = 4,
                        horizon = "15.0",
                        outcome = "0.98"
                    ),
                    build_reference_record(
                        cell_identity = cell_identity,
                        sequence_number = 5,
                        horizon = "20.0",
                        outcome = "0.97"
                    )
                ]
            )

    scope_record = build_scope_record(
        term_name = (
            "ambient_temperature_celsius"
        ),
        term_type = "main_effect"
    )

    scope_record[
        "condition_protocol_count"
    ] = "2"

    scope_record[
        "independent_cell_count"
    ] = "6"

    scope_record[
        "analytical_observation_count"
    ] = "12"

    (
        candidate_records,
        candidate_term_records
    ) = build_candidate_registry(
        scope_records = [scope_record],
        modeling_policy = modeling_policy
    )

    select_model_candidates(
        candidate_records = candidate_records,
        candidate_term_records = (
            candidate_term_records
        ),
        modeling_policy = modeling_policy
    )

    (
        observation_records,
        accounting_records
    ) = assemble_candidate_observations(
        candidate_records = candidate_records,
        modeling_policy = modeling_policy,
        cell_records = cell_records,
        reference_records = reference_records,
        support_membership_records = (
            membership_records
        )
    )

    assert len(
        accounting_records
    ) == 6

    assert all(
        accounting_record[
            "source_record_count"
        ]
        == 30
        for accounting_record in accounting_records
    )

    assert all(
        accounting_record[
            "included_record_count"
        ]
        == 24
        for accounting_record in accounting_records
    )

    assert all(
        accounting_record[
            "nonpositive_horizon_count"
        ]
        == 6
        for accounting_record in accounting_records
    )

    assert all(
        accounting_record[
            "inside_evaluation_interval_count"
        ]
        == 12
        for accounting_record in accounting_records
    )

    assert all(
        accounting_record[
            "outside_evaluation_interval_count"
        ]
        == 12
        for accounting_record in accounting_records
    )

    assert all(
        accounting_record[
            "inside_evaluation_interval_count"
        ]
        + accounting_record[
            "outside_evaluation_interval_count"
        ]
        == accounting_record[
            "included_record_count"
        ]
        for accounting_record in accounting_records
    )

    assert all(
        "included_count"
        not in accounting_record
        for accounting_record in accounting_records
    )

    assert all(
        "outside_support_interval_count"
        not in accounting_record
        for accounting_record in accounting_records
    )

    assert all(
        accounting_record[
            "excluded_record_count"
        ]
        + accounting_record[
            "included_record_count"
        ]
        == accounting_record[
            "source_record_count"
        ]
        for accounting_record in accounting_records
    )

    assert len(
        observation_records
    ) == 144

    candidates_by_identity = {
        candidate_record[
            "candidate_identity"
        ]: candidate_record
        for candidate_record in candidate_records
    }

    # Every retained observation carries an explicit interval-membership flag
    assert all(
        isinstance(
            observation_record[
                "inside_evaluation_interval"
            ],
            bool
        )
        for observation_record in observation_records
    )

    # Retained observations fall into two disjoint groups : inside the authorized interval, or outside
    # but still eligible for fitting
    for observation_record in observation_records:
        candidate_record = candidates_by_identity[
            observation_record[
                "candidate_identity"
            ]
        ]

        expected_inside = horizon_inside_candidate_support(
            horizon_value = observation_record[
                "horizon"
            ],
            candidate_record = candidate_record
        )

        assert (
            observation_record[
                "inside_evaluation_interval"
            ]
            == expected_inside
        )

    assert len(
        {
            (
                observation_record[
                    "candidate_identity"
                ],
                observation_record[
                    "source_identity"
                ]
            )
            for observation_record in observation_records
        }
    ) == len(
        observation_records
    )


# Building deterministic observations for matrix tests
def build_matrix_observations() -> list[dict]:
    observations = []

    for condition_number, temperature in [
        (
            1,
            "10.0"
        ),
        (
            2,
            "35.0"
        )
    ]:
        condition_identity = (
            f"Stage_1/TP_k{condition_number:02d}/periodic_reference_tests"
        )

        for cell_number in range(
            1,
            5
        ):
            cell_identity = (
                f"Stage_1/TP_k{condition_number:02d}_{cell_number:02d}"
            )

            for observation_number, horizon in enumerate(
                [
                    10.0,
                    20.0,
                    30.0,
                    40.0
                ],
                start = 1
            ):
                observations.append(
                    {
                        "candidate_identity": "candidate-1",
                        "support_region_identity": "region-1",
                        "source_identity": (
                            f"{cell_identity}/measurement_{observation_number:02d}.csv"
                        ),
                        "cell_identity": cell_identity,
                        "condition_protocol_identity": (
                            condition_identity
                        ),
                        "dataset_stage": "Stage_1",
                        "aging_mode": "calendar_aging",
                        "outcome": "relative_capacity",
                        "cohort": "primary",
                        "horizon_axis": (
                            "elapsed_calendar_days"
                        ),
                        "horizon": horizon,
                        "outcome_value": (
                            1.0
                            - 0.0001 * horizon
                            - 0.001 * condition_number
                            - 0.00001 * cell_number
                        ),
                        "ambient_temperature_celsius": (
                            temperature
                        ),
                        "maximum_state_of_charge_fraction": (
                            "0.5"
                        ),
                        "depth_of_discharge_fraction": "0.0",
                        "charge_rate_c": "0.0",
                        "discharge_rate_c": "0.0",
                        "reference_test_schedule": (
                            "periodic_reference_tests"
                        ),
                        "laboratory": "INT",
                        "sampling_method": "FF",
                        "cell_candidate_observation_count": 4
                    }
                )

    return observations


# Building one matrix-test candidate
def build_matrix_candidate(template_name: str, horizon_basis: str, include_interactions: bool) -> dict:
    return {
        "candidate_identity": "candidate-1",
        "support_region_identity": "region-1",
        "template_name": template_name,
        "estimator": "weighted_least_squares",
        "horizon_basis": horizon_basis,
        "include_horizon_by_treatment_interactions": (
            include_interactions
        ),
        "cell_random_intercept": False,
        "context_family": "no_additional_context",
        "permitted_adjustment_variables": "[]",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": "elapsed_calendar_days",
        "selection_status": "selected",
        "interaction_hierarchy_status": "complete",
        "observation_assembly_status": "completed"
    }


# Confirming equal total regression weight per cell
def test_build_equal_cell_weights_equalizes_cell_contribution() -> None:
    observations = build_matrix_observations()

    observations = [
        *observations,
        {
            **observations[0],
            "source_identity": (
                "Stage_1/TP_k01_01/additional.csv"
            ),
            "horizon": 50.0,
            "outcome_value": 0.99
        }
    ]

    weights = build_equal_cell_weights(
        observations
    )

    total_weights_by_cell = {}

    for observation, weight in zip(
        observations,
        weights,
        strict = True
    ):
        total_weights_by_cell.setdefault(
            observation[
                "cell_identity"
            ],
            0.0
        )

        total_weights_by_cell[
            observation[
                "cell_identity"
            ]
        ] += float(
            weight
        )

    assert len(
        {
            round(
                total_weight,
                12
            )
            for total_weight in (
                total_weights_by_cell.values()
            )
        }
    ) == 1

    assert float(
        np.mean(
            weights
        )
    ) == pytest.approx(
        1.0
    )


# Confirming linear matrix construction and treatment interaction
def test_transform_model_matrix_builds_declared_linear_terms() -> None:
    modeling_policy = load_test_policy()

    observations = build_matrix_observations()

    candidate_record = build_matrix_candidate(
        template_name = "linear_differential_trajectory",
        horizon_basis = "linear",
        include_interactions = True
    )

    candidate_terms = [
        {
            "candidate_identity": "candidate-1",
            "term_name": (
                "ambient_temperature_celsius"
            ),
            "term_type": "main_effect",
            "term_disposition": "selected"
        }
    ]

    model_encoding = build_model_encoding(
        candidate_record = candidate_record,
        candidate_term_records = candidate_terms,
        observation_records = observations,
        modeling_policy = modeling_policy
    )

    (
        model_matrix,
        matrix_terms
    ) = transform_model_matrix(
        candidate_record = candidate_record,
        observation_records = observations,
        model_encoding = model_encoding
    )

    assert model_matrix.shape == (
        len(
            observations
        ),
        4
    )

    assert [
        matrix_term[
            "column_name"
        ]
        for matrix_term in matrix_terms
    ] == [
        "intercept",
        "horizon_linear",
        "ambient_temperature_celsius",
        (
            "horizon_linear:"
            "ambient_temperature_celsius"
        )
    ]

    assert np.all(
        np.isfinite(
            model_matrix
        )
    )


# Confirming deterministic restricted-cubic-spline construction
def test_build_model_encoding_constructs_declared_spline() -> None:
    modeling_policy = load_test_policy()

    observations = build_matrix_observations()

    candidate_record = build_matrix_candidate(
        template_name = (
            "restricted_cubic_spline_common_trajectory"
        ),
        horizon_basis = "restricted_cubic_spline",
        include_interactions = False
    )

    candidate_terms = [
        {
            "candidate_identity": "candidate-1",
            "term_name": (
                "ambient_temperature_celsius"
            ),
            "term_type": "main_effect",
            "term_disposition": "selected"
        }
    ]

    model_encoding = build_model_encoding(
        candidate_record = candidate_record,
        candidate_term_records = candidate_terms,
        observation_records = observations,
        modeling_policy = modeling_policy
    )

    assert model_encoding[
        "spline_knots"
    ] == [
        10.0,
        20.0,
        40.0
    ]

    (
        model_matrix,
        matrix_terms
    ) = transform_model_matrix(
        candidate_record = candidate_record,
        observation_records = observations,
        model_encoding = model_encoding
    )

    assert [
        matrix_term[
            "column_name"
        ]
        for matrix_term in matrix_terms
    ] == [
        "intercept",
        "horizon_linear",
        "horizon_restricted_cubic",
        "ambient_temperature_celsius"
    ]

    assert model_matrix.shape == (
        len(
            observations
        ),
        4
    )


# Confirming a valid matrix passes every admissibility check
def test_evaluate_candidate_admissibility_accepts_valid_matrix() -> None:
    modeling_policy = load_test_policy()

    observations = build_matrix_observations()

    candidate_record = build_matrix_candidate(
        template_name = "linear_common_trajectory",
        horizon_basis = "linear",
        include_interactions = False
    )

    candidate_terms = [
        {
            "candidate_identity": "candidate-1",
            "term_name": (
                "ambient_temperature_celsius"
            ),
            "term_type": "main_effect",
            "term_disposition": "selected"
        }
    ]

    model_encoding = build_model_encoding(
        candidate_record = candidate_record,
        candidate_term_records = candidate_terms,
        observation_records = observations,
        modeling_policy = modeling_policy
    )

    (
        model_matrix,
        _
    ) = transform_model_matrix(
        candidate_record = candidate_record,
        observation_records = observations,
        model_encoding = model_encoding
    )

    outcome_values = np.asarray(
        [
            observation[
                "outcome_value"
            ]
            for observation in observations
        ],
        dtype = float
    )

    weights = build_equal_cell_weights(
        observations
    )

    result = evaluate_candidate_admissibility(
        candidate_record = candidate_record,
        observation_records = observations,
        model_matrix = model_matrix,
        outcome_values = outcome_values,
        regression_weights = weights,
        model_encoding = model_encoding,
        modeling_policy = modeling_policy
    )

    assert result[
        "admissibility_status"
    ] == "admissible"

    assert result[
        "admissibility_reason_code"
    ] is None

    assert result[
        "matrix_rank"
    ] == 3

    assert result[
        "residual_independent_cell_degrees_of_freedom"
    ] == 5


# Confirming rank deficiency remains an explicit failure
def test_evaluate_candidate_admissibility_rejects_rank_deficiency() -> None:
    modeling_policy = load_test_policy()

    observations = build_matrix_observations()

    model_matrix = np.column_stack(
        [
            np.ones(
                len(
                    observations
                )
            ),
            np.ones(
                len(
                    observations
                )
            )
        ]
    )

    outcome_values = np.asarray(
        [
            observation[
                "outcome_value"
            ]
            for observation in observations
        ],
        dtype = float
    )

    weights = build_equal_cell_weights(
        observations
    )

    result = evaluate_candidate_admissibility(
        candidate_record = build_matrix_candidate(
            template_name = "linear_common_trajectory",
            horizon_basis = "linear",
            include_interactions = False
        ),
        observation_records = observations,
        model_matrix = model_matrix,
        outcome_values = outcome_values,
        regression_weights = weights,
        model_encoding = {
            "unique_horizon_count": 4,
            "non_boundary_unique_horizon_count": None
        },
        modeling_policy = modeling_policy
    )

    assert result[
        "admissibility_status"
    ] == "inadmissible"

    assert result[
        "admissibility_reason_code"
    ] == "MODEL_MATRIX_RANK_DEFICIENT"


# Building one deterministic prepared regression model
def build_prepared_regression_model(estimator: str) -> dict:
    cell_identities = []

    horizon_values = []

    outcome_values = []

    cell_offsets = [
        -0.18,
        -0.13,
        -0.07,
        -0.02,
        0.02,
        0.07,
        0.13,
        0.18
    ]

    within_cell_errors = [
        -0.06,
        0.06,
        0.06,
        -0.06
    ]

    standardized_horizons = [
        -1.5,
        -0.5,
        0.5,
        1.5
    ]

    for cell_index, cell_offset in enumerate(
        cell_offsets,
        start = 1
    ):
        cell_identity = (
            f"Stage_1/TP_k01_{cell_index:02d}"
        )

        for (
            horizon_value,
            within_cell_error
        ) in zip(
            standardized_horizons,
            within_cell_errors,
            strict = True
        ):
            cell_identities.append(
                cell_identity
            )

            horizon_values.append(
                horizon_value
            )

            outcome_values.append(
                1.0
                + 0.2 * horizon_value
                + cell_offset
                + within_cell_error
            )

    model_matrix = np.column_stack(
        [
            np.ones(
                len(
                    horizon_values
                ),
                dtype = float
            ),
            np.asarray(
                horizon_values,
                dtype = float
            )
        ]
    )

    outcome_array = np.asarray(
        outcome_values,
        dtype = float
    )

    regression_weights = np.ones(
        len(
            outcome_values
        ),
        dtype = float
    )

    candidate_identity = (
        f"candidate-{estimator}"
    )

    return {
        "candidate_identity": candidate_identity,
        "fit_identity": f"fit-{estimator}",
        "candidate_record": {
            "candidate_identity": candidate_identity,
            "support_region_identity": "region-1",
            "template_name": (
                "linear_common_trajectory"
                if estimator
                == "weighted_least_squares"
                else "mixed_linear_common_trajectory"
            ),
            "estimator": estimator,
            "horizon_basis": "linear",
            "include_horizon_by_treatment_interactions": False,
            "fit_status": "not_attempted",
            "uncertainty_status": "not_attempted",
            "validation_status": "not_attempted",
            "influence_status": "not_attempted"
        },
        "observation_records": [
            {
                "source_identity": f"source-{row_index:03d}",
                "cell_identity": cell_identity,
                "condition_protocol_identity": (
                    "Stage_1/TP_k01/periodic_reference_tests"
                ),
                "horizon": horizon_value,
                "outcome_value": outcome_value
            }
            for (
                row_index,
                (
                    cell_identity,
                    horizon_value,
                    outcome_value
                )
            ) in enumerate(
                zip(
                    cell_identities,
                    horizon_values,
                    outcome_values,
                    strict = True
                )
            )
        ],
        "model_encoding": {
            "horizon_basis": "linear",
            "horizon_mean": 0.0,
            "horizon_standard_deviation": 1.0,
            "spline_knots": None,
            "unique_horizon_count": 4,
            "non_boundary_unique_horizon_count": None,
            "ordered_source_terms": [],
            "source_term_encodings": {},
            "context_encodings": []
        },
        "model_matrix": model_matrix,
        "matrix_terms": [
            {
                "column_name": "intercept",
                "term_type": "intercept",
                "source_term_name": None,
                "horizon_basis_name": None
            },
            {
                "column_name": "horizon_linear",
                "term_type": "horizon_basis",
                "source_term_name": None,
                "horizon_basis_name": "horizon_linear"
            }
        ],
        "outcome_values": outcome_array,
        "regression_weights": regression_weights,
        "cell_identities": cell_identities,
        "condition_protocol_identities": [
            "Stage_1/TP_k01/periodic_reference_tests"
            for _ in cell_identities
        ]
    }


# Confirming equal-cell residual diagnostics
def test_calculate_equal_cell_residual_diagnostics() -> None:
    diagnostics = (
        calculate_equal_cell_residual_diagnostics(
            residual_values = np.asarray(
                [
                    1.0,
                    -1.0,
                    2.0,
                    -2.0
                ],
                dtype = float
            ),
            cell_identities = [
                "cell-1",
                "cell-1",
                "cell-2",
                "cell-2"
            ]
        )
    )

    assert diagnostics[
        "residual_mean"
    ] == pytest.approx(
        0.0
    )

    assert diagnostics[
        "residual_standard_deviation"
    ] == pytest.approx(
        math.sqrt(
            2.5
        )
    )

    assert diagnostics[
        "maximum_absolute_standardized_residual"
    ] == pytest.approx(
        2.0
        / math.sqrt(
            2.5
        )
    )


# Confirming the weighted least-squares coefficient calculation
def test_fit_weighted_least_squares() -> None:
    prepared_model = (
        build_prepared_regression_model(
            "weighted_least_squares"
        )
    )

    fitted_result = (
        fit_weighted_least_squares(
            prepared_model = prepared_model,
            modeling_policy = load_test_policy()
        )
    )

    assert fitted_result[
        "coefficients"
    ][0] == pytest.approx(
        1.0
    )

    assert fitted_result[
        "coefficients"
    ][1] == pytest.approx(
        0.2
    )

    assert fitted_result[
        "convergence_status"
    ] == "completed"

    assert fitted_result[
        "random_effect_singularity_status"
    ] == "not_applicable"

    assert (
        fitted_result[
            "model_based_fixed_effect_covariance"
        ].shape
        == (
            2,
            2
        )
    )

    assert np.all(
        np.isfinite(
            fitted_result[
                "model_based_fixed_effect_covariance"
            ]
        )
    )


# Rejecting a weighted model matrix above the condition-number limit
def test_fit_weighted_least_squares_rejects_ill_conditioned_weighted_matrix() -> None:
    modeling_policy = load_test_policy()

    replace_policy_value(
        modeling_policy = modeling_policy,
        policy_path = (
            "admissibility",
            "maximum_effective_condition_number"
        ),
        replacement_value = 1.0
    )

    prepared_model = (
        build_prepared_regression_model(
            "weighted_least_squares"
        )
    )

    with pytest.raises(
        descriptive_models.CandidateFitUnavailableError
    ) as exception:
        fit_weighted_least_squares(
            prepared_model = prepared_model,
            modeling_policy = modeling_policy
        )

    assert exception.value.reason_code == (
        "WEIGHTED_MODEL_MATRIX_ILL_CONDITIONED"
    )


# Confirming random-intercept mixed-effects fitting
def test_fit_linear_mixed_effects() -> None:
    prepared_model = (
        build_prepared_regression_model(
            "linear_mixed_effects"
        )
    )

    fitted_result = (
        fit_linear_mixed_effects(
            prepared_model = prepared_model,
            modeling_policy = load_test_policy()
        )
    )

    assert fitted_result[
        "convergence_status"
    ] == "completed"

    assert fitted_result[
        "random_effect_singularity_status"
    ] == "non_singular"

    assert fitted_result[
        "random_intercept_variance"
    ] > 0

    assert fitted_result[
        "residual_variance"
    ] > 0

    assert len(
        fitted_result[
            "random_effects_by_cell"
        ]
    ) == 8

    assert fitted_result[
        "coefficients"
    ][0] == pytest.approx(
        1.0,
        abs = 0.03
    )

    assert fitted_result[
        "coefficients"
    ][1] == pytest.approx(
        0.2,
        abs = 0.03
    )


# Confirming that one reusable fit serves multiple candidates
def test_fit_candidate_models_reuses_identical_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    first_model = build_prepared_regression_model(
        "weighted_least_squares"
    )

    second_model = build_prepared_regression_model(
        "weighted_least_squares"
    )

    first_model[
        "candidate_identity"
    ] = "candidate-a"

    first_model[
        "candidate_record"
    ][
        "candidate_identity"
    ] = "candidate-a"

    second_model[
        "candidate_identity"
    ] = "candidate-b"

    second_model[
        "candidate_record"
    ][
        "candidate_identity"
    ] = "candidate-b"

    first_model[
        "fit_identity"
    ] = "shared-fit"

    second_model[
        "fit_identity"
    ] = "shared-fit"

    fit_call_count = 0

    original_fit_function = (
        fit_weighted_least_squares
    )

    def count_fit_calls(prepared_model: dict,
                        modeling_policy: dict
    ) -> dict:
        nonlocal fit_call_count

        fit_call_count += 1

        return original_fit_function(
            prepared_model = prepared_model,
            modeling_policy = modeling_policy
        )

    monkeypatch.setattr(
        "scripts.run_descriptive_models."
        "fit_weighted_least_squares",
        count_fit_calls
    )

    (
        fitted_models,
        coefficient_records,
        diagnostic_records
    ) = fit_candidate_models(
        prepared_models = {
            "candidate-a": first_model,
            "candidate-b": second_model
        },
        modeling_policy = load_test_policy()
    )

    assert fit_call_count == 1

    assert set(
        fitted_models
    ) == {
        "candidate-a",
        "candidate-b"
    }

    assert len(
        coefficient_records
    ) == 4

    assert len(
        diagnostic_records
    ) == 2

    assert all(
        diagnostic_record[
            "fit_status"
        ]
        == "completed"
        for diagnostic_record in (
            diagnostic_records
        )
    )


# Confirming complete delete-one-cell uncertainty and influence
def test_calculate_uncertainty_and_influence() -> None:
    modeling_policy = load_test_policy()

    prepared_model = (
        build_prepared_regression_model(
            "weighted_least_squares"
        )
    )

    candidate_identity = prepared_model[
        "candidate_identity"
    ]

    (
        fitted_models,
        coefficient_records,
        diagnostic_records
    ) = fit_candidate_models(
        prepared_models = {
            candidate_identity: prepared_model
        },
        modeling_policy = modeling_policy
    )

    influence_records = (
        calculate_uncertainty_and_influence(
            prepared_models = {
                candidate_identity: prepared_model
            },
            fitted_models_by_candidate = (
                fitted_models
            ),
            coefficient_records = (
                coefficient_records
            ),
            modeling_policy = modeling_policy
        )
    )

    assert diagnostic_records[0][
        "fit_status"
    ] == "completed"

    assert prepared_model[
        "candidate_record"
    ][
        "uncertainty_status"
    ] == "completed"

    assert prepared_model[
        "candidate_record"
    ][
        "influence_status"
    ] == "completed"

    assert all(
        coefficient_record[
            "uncertainty_status"
        ]
        == "completed"
        for coefficient_record in (
            coefficient_records
        )
    )

    assert all(
        coefficient_record[
            "standard_error"
        ]
        is not None
        and coefficient_record[
            "standard_error"
        ]
        >= 0
        for coefficient_record in (
            coefficient_records
        )
    )

    assert all(
        coefficient_record[
            "confidence_interval_lower"
        ]
        <= coefficient_record[
            "estimate"
        ]
        <= coefficient_record[
            "confidence_interval_upper"
        ]
        for coefficient_record in (
            coefficient_records
        )
    )

    independent_cell_count = len(
        set(
            prepared_model[
                "cell_identities"
            ]
        )
    )

    coefficient_count = len(
        prepared_model[
            "matrix_terms"
        ]
    )

    assert len(
        influence_records
    ) == (
        independent_cell_count
        * coefficient_count
    )

    assert {
        influence_record[
            "deleted_cell_identity"
        ]
        for influence_record in influence_records
    } == set(
        prepared_model[
            "cell_identities"
        ]
    )

    assert all(
        influence_record[
            "fit_status"
        ]
        == "completed"
        for influence_record in (
            influence_records
        )
    )

    assert all(
        influence_record[
            "coefficient_change"
        ]
        is not None
        for influence_record in (
            influence_records
        )
    )


# Confirming deletion refits are reused across equivalent candidates
def test_uncertainty_reuses_identical_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    modeling_policy = load_test_policy()

    first_model = build_prepared_regression_model(
        "weighted_least_squares"
    )

    second_model = build_prepared_regression_model(
        "weighted_least_squares"
    )

    first_model[
        "candidate_identity"
    ] = "candidate-a"

    first_model[
        "candidate_record"
    ][
        "candidate_identity"
    ] = "candidate-a"

    second_model[
        "candidate_identity"
    ] = "candidate-b"

    second_model[
        "candidate_record"
    ][
        "candidate_identity"
    ] = "candidate-b"

    first_model[
        "fit_identity"
    ] = "shared-fit"

    second_model[
        "fit_identity"
    ] = "shared-fit"

    (
        fitted_models,
        coefficient_records,
        diagnostic_records
    ) = fit_candidate_models(
        prepared_models = {
            "candidate-a": first_model,
            "candidate-b": second_model
        },
        modeling_policy = modeling_policy
    )

    deletion_fit_count = 0

    original_fit_function = (
        fit_prepared_model
    )

    def count_deleted_fits(prepared_model: dict,
                          modeling_policy: dict
    ) -> dict:
        nonlocal deletion_fit_count

        deletion_fit_count += 1

        return original_fit_function(
            prepared_model = prepared_model,
            modeling_policy = modeling_policy
        )

    monkeypatch.setattr(
        "scripts.run_descriptive_models."
        "fit_prepared_model",
        count_deleted_fits
    )

    influence_records = (
        calculate_uncertainty_and_influence(
            prepared_models = {
                "candidate-a": first_model,
                "candidate-b": second_model
            },
            fitted_models_by_candidate = (
                fitted_models
            ),
            coefficient_records = (
                coefficient_records
            ),
            modeling_policy = modeling_policy
        )
    )

    independent_cell_count = len(
        set(
            first_model[
                "cell_identities"
            ]
        )
    )

    assert deletion_fit_count == (
        independent_cell_count
    )

    assert len(
        influence_records
    ) == (
        2
        * independent_cell_count
        * len(
            first_model[
                "matrix_terms"
            ]
        )
    )

    assert diagnostic_records[0][
        "fit_status"
    ] == "completed"

    assert diagnostic_records[1][
        "fit_status"
    ] == "completed"


# Confirming deterministic condition-stratified fold construction
def test_build_validation_fold_assignments_preserves_conditions() -> None:
    prepared_model = build_prepared_regression_model(
        "weighted_least_squares"
    )

    condition_by_cell = {}

    distinct_cells = sorted(
        set(
            prepared_model[
                "cell_identities"
            ]
        )
    )

    for cell_index, cell_identity in enumerate(
        distinct_cells
    ):
        condition_by_cell[
            cell_identity
        ] = (
            "condition-a"
            if cell_index < 4
            else "condition-b"
        )

    for observation_record in prepared_model[
        "observation_records"
    ]:
        observation_record[
            "condition_protocol_identity"
        ] = condition_by_cell[
            observation_record[
                "cell_identity"
            ]
        ]

    (
        fold_count,
        fold_by_cell
    ) = build_validation_fold_assignments(
        prepared_model = prepared_model,
        modeling_policy = load_test_policy()
    )

    assert fold_count == 3

    assert set(
        fold_by_cell
    ) == set(
        distinct_cells
    )

    for fold_index in range(
        fold_count
    ):
        test_conditions = {
            condition_by_cell[
                cell_identity
            ]
            for (
                cell_identity,
                assigned_fold
            ) in fold_by_cell.items()
            if assigned_fold == fold_index
        }

        training_conditions = {
            condition_by_cell[
                cell_identity
            ]
            for (
                cell_identity,
                assigned_fold
            ) in fold_by_cell.items()
            if assigned_fold != fold_index
        }

        assert test_conditions == {
            "condition-a",
            "condition-b"
        }

        assert training_conditions == {
            "condition-a",
            "condition-b"
        }


# Confirming that equivalent candidate templates share fold identities
def test_build_validation_fold_identity_is_template_independent() -> None:
    modeling_policy = load_test_policy()

    sharing_key_fields = modeling_policy[
        "validation"
    ][
        "fold_sharing_key_fields"
    ]

    first_candidate = {
        field_name: (
            f"value-for-{field_name}"
        )
        for field_name in sharing_key_fields
    }

    first_candidate[
        "candidate_identity"
    ] = "candidate-linear"

    first_candidate[
        "template_name"
    ] = "linear_trajectory"

    second_candidate = {
        **first_candidate,
        "candidate_identity": "candidate-mixed",
        "template_name": (
            "linear_mixed_effects_trajectory"
        )
    }

    test_cell_identities = [
        "Stage_1/TP_k01_01",
        "Stage_1/TP_k01_02"
    ]

    first_fold_identity = (
        build_validation_fold_identity(
            candidate_record = first_candidate,
            fold_index = 0,
            test_cell_identities = (
                test_cell_identities
            ),
            modeling_policy = modeling_policy
        )
    )

    second_fold_identity = (
        build_validation_fold_identity(
            candidate_record = second_candidate,
            fold_index = 0,
            test_cell_identities = list(
                reversed(
                    test_cell_identities
                )
            ),
            modeling_policy = modeling_policy
        )
    )

    assert first_fold_identity == (
        second_fold_identity
    )


# Confirming that fold identities distinguish sharing populations
def test_build_validation_fold_identity_distinguishes_support_regions() -> None:
    modeling_policy = load_test_policy()

    sharing_key_fields = modeling_policy[
        "validation"
    ][
        "fold_sharing_key_fields"
    ]

    first_candidate = {
        field_name: (
            f"value-for-{field_name}"
        )
        for field_name in sharing_key_fields
    }

    second_candidate = {
        **first_candidate,
        "support_region_identity": (
            "different-support-region"
        )
    }

    test_cell_identities = [
        "Stage_1/TP_k01_01",
        "Stage_1/TP_k01_02"
    ]

    first_fold_identity = (
        build_validation_fold_identity(
            candidate_record = first_candidate,
            fold_index = 0,
            test_cell_identities = (
                test_cell_identities
            ),
            modeling_policy = modeling_policy
        )
    )

    second_fold_identity = (
        build_validation_fold_identity(
            candidate_record = second_candidate,
            fold_index = 0,
            test_cell_identities = (
                test_cell_identities
            ),
            modeling_policy = modeling_policy
        )
    )

    assert first_fold_identity != (
        second_fold_identity
    )


# Rejecting validation fold-sharing policy conflicts
def test_build_validation_fold_identity_rejects_excluded_key_fields() -> None:
    modeling_policy = load_test_policy()

    modeling_policy[
        "validation"
    ][
        "fold_sharing_key_fields"
    ].append(
        "template_name"
    )

    candidate_record = {
        field_name: (
            f"value-for-{field_name}"
        )
        for field_name in modeling_policy[
            "validation"
        ][
            "fold_sharing_key_fields"
        ]
    }

    with pytest.raises(
        RuntimeError,
        match = (
            "explicitly excluded fields"
        )
    ):
        build_validation_fold_identity(
            candidate_record = candidate_record,
            fold_index = 0,
            test_cell_identities = [
                "Stage_1/TP_k01_01"
            ],
            modeling_policy = modeling_policy
        )


# Confirming exact support-boundary handling
def test_horizon_inside_candidate_support_respects_boundaries() -> None:
    candidate_record = {
        "support_interval_minimum": "10.0",
        "support_interval_maximum": "20.0",
        "support_interval_lower_inclusive": "true",
        "support_interval_upper_inclusive": "false"
    }

    assert horizon_inside_candidate_support(
        horizon_value = 10.0,
        candidate_record = candidate_record
    )

    assert horizon_inside_candidate_support(
        horizon_value = 15.0,
        candidate_record = candidate_record
    )

    assert not horizon_inside_candidate_support(
        horizon_value = 20.0,
        candidate_record = candidate_record
    )

    assert not horizon_inside_candidate_support(
        horizon_value = 9.999,
        candidate_record = candidate_record
    )


# Confirming equal-cell rather than equal-observation metrics
def test_calculate_grouped_validation_metrics_equalizes_cells() -> None:
    prediction_records = [
        {
            "candidate_identity": "candidate-1",
            "source_identity": "source-1",
            "cell_identity": "cell-1",
            "observed_outcome": 0.0,
            "predicted_outcome": 1.0,
            "prediction_interval_status": "completed",
            "prediction_interval_lower": -1.0,
            "prediction_interval_upper": 2.0
        },
        {
            "candidate_identity": "candidate-1",
            "source_identity": "source-2",
            "cell_identity": "cell-1",
            "observed_outcome": 1.0,
            "predicted_outcome": 2.0,
            "prediction_interval_status": "completed",
            "prediction_interval_lower": 0.0,
            "prediction_interval_upper": 3.0
        },
        {
            "candidate_identity": "candidate-1",
            "source_identity": "source-3",
            "cell_identity": "cell-2",
            "observed_outcome": 0.0,
            "predicted_outcome": 3.0,
            "prediction_interval_status": "completed",
            "prediction_interval_lower": -1.0,
            "prediction_interval_upper": 4.0
        }
    ]

    metrics = calculate_grouped_validation_metrics(
        prediction_records
    )

    assert metrics[
        "independent_cell_count"
    ] == 2

    assert metrics[
        "observation_count"
    ] == 3

    assert metrics[
        "mean_absolute_error"
    ] == pytest.approx(
        2.0
    )

    assert metrics[
        "root_mean_squared_error"
    ] == pytest.approx(
        math.sqrt(
            5.0
        )
    )

    assert metrics[
        "mean_error"
    ] == pytest.approx(
        2.0
    )

    assert metrics[
        "prediction_interval_coverage"
    ] == pytest.approx(
        1.0
    )


# Confirming support-restricted trajectory summaries
def test_build_trajectory_summaries_uses_supported_observations() -> None:
    modeling_policy = load_test_policy()

    condition_identity = (
        "Stage_1/TP_k01/periodic_reference_tests"
    )

    cell_records = [
        build_cell_record(
            cell_identity = "Stage_1/TP_k01_01",
            condition_identity = condition_identity,
            cell_number = 1
        ),
        build_cell_record(
            cell_identity = "Stage_1/TP_k01_02",
            condition_identity = condition_identity,
            cell_number = 2
        )
    ]

    membership_records = [
        build_membership_record(
            cell_identity = cell_record[
                "cell_identity"
            ],
            condition_identity = condition_identity
        )
        for cell_record in cell_records
    ]

    reference_records = []

    for cell_index, cell_record in enumerate(
        cell_records,
        start = 1
    ):
        reference_records.extend(
            [
                build_reference_record(
                    cell_identity = (
                        cell_record[
                            "cell_identity"
                        ]
                    ),
                    sequence_number = 1,
                    horizon = "15.0",
                    outcome = str(
                        1.0
                        - 0.01
                        * cell_index
                    )
                ),
                build_reference_record(
                    cell_identity = (
                        cell_record[
                            "cell_identity"
                        ]
                    ),
                    sequence_number = 2,
                    horizon = "25.0",
                    outcome = "0.90"
                )
            ]
        )

    (
        summary_records,
        accounting_records
    ) = build_trajectory_summaries(
        modeling_policy = modeling_policy,
        cell_records = cell_records,
        reference_records = reference_records,
        support_membership_records = (
            membership_records
        )
    )

    record_type_counts = Counter(
        summary_record[
            "record_type"
        ]
        for summary_record in summary_records
    )

    assert record_type_counts == {
        "cell": 2,
        "condition": 1,
        "condition_horizon": 1
    }

    horizon_record = next(
        summary_record
        for summary_record in summary_records
        if summary_record[
            "record_type"
        ] == "condition_horizon"
    )

    assert horizon_record[
        "independent_cell_count"
    ] == 2

    assert horizon_record[
        "mean_outcome"
    ] == pytest.approx(
        0.985
    )

    assert accounting_records[0][
        "eligible_positive_record_count"
    ] == 4

    assert accounting_records[0][
        "included_record_count"
    ] == 2

    assert accounting_records[0][
        "outside_support_record_count"
    ] == 2


# Confirming equal-cell condition calibration
def test_build_condition_calibration_equalizes_cells() -> None:
    condition_identity = (
        "Stage_1/TP_k01/periodic_reference_tests"
    )

    candidate_record = {
        "candidate_identity": "candidate-1",
        "support_region_identity": "region-1",
        "template_name": "linear_common_trajectory",
        "fit_status": "completed",
        "validation_status": "completed",
        "validation_reason_code": None
    }

    prepared_models = {
        "candidate-1": {
            "candidate_identity": "candidate-1",
            "fit_identity": "fit-1",
            "candidate_record": candidate_record,
            "observation_records": [
                {
                    "cell_identity": "cell-1",
                    "condition_protocol_identity": (
                        condition_identity
                    )
                },
                {
                    "cell_identity": "cell-2",
                    "condition_protocol_identity": (
                        condition_identity
                    )
                }
            ]
        }
    }

    prediction_records = [
        {
            "candidate_identity": "candidate-1",
            "cell_identity": "cell-1",
            "condition_protocol_identity": (
                condition_identity
            ),
            "observed_outcome": 1.0,
            "predicted_outcome": 0.9
        },
        {
            "candidate_identity": "candidate-1",
            "cell_identity": "cell-1",
            "condition_protocol_identity": (
                condition_identity
            ),
            "observed_outcome": 1.0,
            "predicted_outcome": 1.0
        },
        {
            "candidate_identity": "candidate-1",
            "cell_identity": "cell-2",
            "condition_protocol_identity": (
                condition_identity
            ),
            "observed_outcome": 0.8,
            "predicted_outcome": 0.9
        }
    ]

    calibration_records = (
        build_condition_calibration(
            prepared_models = prepared_models,
            prediction_records = (
                prediction_records
            )
        )
    )

    assert len(
        calibration_records
    ) == 1

    calibration_record = (
        calibration_records[0]
    )

    assert calibration_record[
        "calibration_status"
    ] == "completed"

    assert calibration_record[
        "independent_cell_count"
    ] == 2

    assert calibration_record[
        "mean_observed"
    ] == pytest.approx(
        0.9
    )

    assert calibration_record[
        "mean_predicted"
    ] == pytest.approx(
        0.925
    )

    assert calibration_record[
        "mean_absolute_error"
    ] == pytest.approx(
        0.075
    )


# Confirming descriptive comparison without winner selection
def test_build_model_comparisons_reports_metric_differences() -> None:
    modeling_policy = load_test_policy()

    group_fields = modeling_policy[
        "model_comparison"
    ][
        "comparison_group_fields"
    ]

    shared_group_values = {
        "support_region_identity": "region-1",
        "design_scope": "Stage_1",
        "sampling_design_scope": (
            "combined_within_scope"
        ),
        "aging_mode": "calendar_aging",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": "elapsed_calendar_days",
        "specification": "main_effects",
        "context_family": (
            "no_additional_context"
        )
    }

    assert set(
        group_fields
    ) == set(
        shared_group_values
    )

    candidate_records = [
        {
            **shared_group_values,
            "candidate_identity": "candidate-a",
            "template_name": (
                "linear_common_trajectory"
            ),
            "selection_status": "selected"
        },
        {
            **shared_group_values,
            "candidate_identity": "candidate-b",
            "template_name": (
                "mixed_linear_common_trajectory"
            ),
            "selection_status": "selected"
        }
    ]

    prediction_records = [
        {
            "candidate_identity": candidate_identity,
            "source_identity": "source-1",
            "cell_identity": "cell-1"
        }
        for candidate_identity in [
            "candidate-a",
            "candidate-b"
        ]
    ]

    metric_records = [
        {
            "candidate_identity": "candidate-a",
            "validation_status": "completed",
            "mean_absolute_error": 0.02,
            "root_mean_squared_error": 0.03
        },
        {
            "candidate_identity": "candidate-b",
            "validation_status": "completed",
            "mean_absolute_error": 0.015,
            "root_mean_squared_error": 0.025
        }
    ]

    comparison_records = (
        build_model_comparisons(
            candidate_records = candidate_records,
            validation_prediction_records = (
                prediction_records
            ),
            validation_metric_records = (
                metric_records
            ),
            modeling_policy = modeling_policy
        )
    )

    assert len(
        comparison_records
    ) == 1

    comparison_record = (
        comparison_records[0]
    )

    assert comparison_record[
        "comparison_status"
    ] == "completed"

    assert comparison_record[
        "mean_absolute_error_difference"
    ] == pytest.approx(
        -0.005
    )

    assert comparison_record[
        "root_mean_squared_error_difference"
    ] == pytest.approx(
        -0.005
    )

    assert (
        "winner"
        not in comparison_record
    )


# Building one condition used for contrast prediction
def build_contrast_condition(condition_identity: str, reference_test_schedule: str) -> dict:
    return {
        "condition_protocol_identity": (
            condition_identity
        ),
        "dataset_stage": "Stage_1",
        "aging_mode": "calendar_aging",
        "ambient_temperature_celsius": "23.0",
        "maximum_state_of_charge_fraction": "0.5",
        "depth_of_discharge_fraction": "0.0",
        "charge_rate_c": "0.0",
        "discharge_rate_c": "0.0",
        "reference_test_schedule": (
            reference_test_schedule
        ),
        "laboratory": "INT",
        "sampling_method": "FF"
    }


# Confirming authorized condition contrasts and uncertainty
def test_build_model_contrasts() -> None:
    modeling_policy = load_test_policy()

    first_condition_identity = (
        "Stage_1/TP_k01/initial_and_final_only"
    )

    second_condition_identity = (
        "Stage_1/TP_k01/periodic_reference_tests"
    )

    candidate_record = {
        "candidate_identity": "candidate-1",
        "fit_identity": "fit-1",
        "support_region_identity": "region-1",
        "template_name": (
            "linear_common_trajectory"
        ),
        "estimator": "weighted_least_squares",
        "aging_mode": "calendar_aging",
        "design_scope": "Stage_1",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": "elapsed_calendar_days",
        "context_family": (
            "no_additional_context"
        ),
        "horizon_basis": "linear",
        "include_horizon_by_treatment_interactions": (
            False
        ),
        "support_interval_minimum": "10.0",
        "support_interval_maximum": "20.0",
        "support_interval_lower_inclusive": "true",
        "support_interval_upper_inclusive": "true",
        "fit_status": "completed",
        "validation_status": "completed",
        "validation_reason_code": None,
        "uncertainty_status": "completed",
        "uncertainty_reason_code": None,
        "analytical_result_status": "available",
        "analytical_result_reason_code": None
    }

    model_encoding = {
        "horizon_basis": "linear",
        "horizon_mean": 15.0,
        "horizon_standard_deviation": 5.0,
        "spline_knots": None,
        "unique_horizon_count": 2,
        "non_boundary_unique_horizon_count": None,
        "ordered_source_terms": [
            {
                "term_name": (
                    "reference_test_schedule="
                    "periodic_reference_tests"
                ),
                "term_type": "target_treatment"
            }
        ],
        "source_term_encodings": {
            (
                "reference_test_schedule="
                "periodic_reference_tests"
            ): {
                "encoding_type": (
                    "categorical_indicator"
                ),
                "field_name": (
                    "reference_test_schedule"
                ),
                "level_name": (
                    "periodic_reference_tests"
                ),
                "observed_levels": [
                    "initial_and_final_only",
                    "periodic_reference_tests"
                ]
            }
        },
        "context_encodings": []
    }

    matrix_terms = [
        {
            "column_name": "intercept",
            "term_type": "intercept",
            "source_term_name": None,
            "horizon_basis_name": None
        },
        {
            "column_name": "horizon_linear",
            "term_type": "horizon_basis",
            "source_term_name": None,
            "horizon_basis_name": "horizon_linear"
        },
        {
            "column_name": (
                "reference_test_schedule="
                "periodic_reference_tests"
            ),
            "term_type": "target_treatment",
            "source_term_name": (
                "reference_test_schedule=periodic_reference_tests"
            ),
            "horizon_basis_name": None
        }
    ]

    prepared_model = {
        "candidate_identity": "candidate-1",
        "fit_identity": "fit-1",
        "candidate_record": candidate_record,
        "model_encoding": model_encoding,
        "matrix_terms": matrix_terms,
        "observation_records": [
            {
                "source_identity": "source-1",
                "cell_identity": "cell-1",
                "condition_protocol_identity": (
                    first_condition_identity
                ),
                "horizon": 10.0
            },
            {
                "source_identity": "source-2",
                "cell_identity": "cell-2",
                "condition_protocol_identity": (
                    second_condition_identity
                ),
                "horizon": 10.0
            },
            {
                "source_identity": "source-3",
                "cell_identity": "cell-1",
                "condition_protocol_identity": (
                    first_condition_identity
                ),
                "horizon": 20.0
            },
            {
                "source_identity": "source-4",
                "cell_identity": "cell-2",
                "condition_protocol_identity": (
                    second_condition_identity
                ),
                "horizon": 20.0
            }
        ]
    }

    fitted_model = {
        "coefficients": np.asarray(
            [
                1.0,
                -0.1,
                0.05
            ],
            dtype = float
        ),
        "jackknife_covariance": np.diag(
            [
                0.0004,
                0.0004,
                0.0001
            ]
        ),
        "jackknife_degrees_of_freedom": 5
    }

    estimand_record = {
        "aging_mode": "calendar_aging",
        "decision_scope": (
            "condition_protocol_pair"
        ),
        "first_condition_protocol_identity": (
            first_condition_identity
        ),
        "second_condition_protocol_identity": (
            second_condition_identity
        ),
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": "elapsed_calendar_days",
        "target_estimand_available": "true",
        "final_support_classification": (
            "supported_with_restrictions"
        ),
        "permitted_bracketing_ranges": (
            '[{"minimum":10.0,"maximum":20.0}]'
        ),
        "restriction_reasons": (
            '["ASSIGNMENT_MECHANISM_NOT_ESTABLISHED"]'
        )
    }

    contrast_records = build_model_contrasts(
        prepared_models = {
            "candidate-1": prepared_model
        },
        fitted_models_by_candidate = {
            "candidate-1": fitted_model
        },
        estimand_records = [
            estimand_record
        ],
        condition_records = [
            build_contrast_condition(
                condition_identity = (
                    first_condition_identity
                ),
                reference_test_schedule = (
                    "initial_and_final_only"
                )
            ),
            build_contrast_condition(
                condition_identity = (
                    second_condition_identity
                ),
                reference_test_schedule = (
                    "periodic_reference_tests"
                )
            )
        ],
        modeling_policy = modeling_policy
    )

    assert len(
        contrast_records
    ) == 2

    assert {
        contrast_record[
            "horizon"
        ]
        for contrast_record in contrast_records
    } == {
        10.0,
        20.0
    }

    assert all(
        contrast_record[
            "contrast_status"
        ]
        == "completed"
        for contrast_record in contrast_records
    )

    assert all(
        contrast_record[
            "contrast_estimate"
        ]
        == pytest.approx(
            0.05
        )
        for contrast_record in contrast_records
    )

    assert all(
        contrast_record[
            "uncertainty_status"
        ]
        == "completed"
        for contrast_record in contrast_records
    )

    assert all(
        contrast_record[
            "confidence_interval_lower"
        ]
        < contrast_record[
            "contrast_estimate"
        ]
        < contrast_record[
            "confidence_interval_upper"
        ]
        for contrast_record in contrast_records
    )

    assert len(
        {
            contrast_record[
                "estimand_decision_identity"
            ]
            for contrast_record in contrast_records
        }
    ) == 1

    candidate_record[
        "validation_status"
    ] = "unavailable"

    candidate_record[
        "validation_reason_code"
    ] = (
        "GROUPED_VALIDATION_UNAVAILABLE"
    )

    diagnostic_contrasts = (
        build_model_contrasts(
            prepared_models = {
                "candidate-1": prepared_model
            },
            fitted_models_by_candidate = {
                "candidate-1": fitted_model
            },
            estimand_records = [
                estimand_record
            ],
            condition_records = [
                build_contrast_condition(
                    condition_identity = (
                        first_condition_identity
                    ),
                    reference_test_schedule = (
                        "initial_and_final_only"
                    )
                ),
                build_contrast_condition(
                    condition_identity = (
                        second_condition_identity
                    ),
                    reference_test_schedule = (
                        "periodic_reference_tests"
                    )
                )
            ],
            modeling_policy = modeling_policy
        )
    )

    assert all(
        contrast_record[
            "contrast_status"
        ] == "diagnostic_only"
        for contrast_record in (
            diagnostic_contrasts
        )
    )

    assert all(
        contrast_record[
            "contrast_reason_code"
        ] == "GROUPED_VALIDATION_UNAVAILABLE"
        for contrast_record in (
            diagnostic_contrasts
        )
    )

    candidate_record[
        "validation_status"
    ] = "completed"

    candidate_record[
        "validation_reason_code"
    ] = None

    fitted_model[
        "jackknife_covariance"
    ] = None

    uncertainty_limited_contrasts = (
        build_model_contrasts(
            prepared_models = {
                "candidate-1": prepared_model
            },
            fitted_models_by_candidate = {
                "candidate-1": fitted_model
            },
            estimand_records = [
                estimand_record
            ],
            condition_records = [
                build_contrast_condition(
                    condition_identity = (
                        first_condition_identity
                    ),
                    reference_test_schedule = (
                        "initial_and_final_only"
                    )
                ),
                build_contrast_condition(
                    condition_identity = (
                        second_condition_identity
                    ),
                    reference_test_schedule = (
                        "periodic_reference_tests"
                    )
                )
            ],
            modeling_policy = modeling_policy
        )
    )

    assert all(
        contrast_record[
            "contrast_status"
        ] == "diagnostic_only"
        for contrast_record in (
            uncertainty_limited_contrasts
        )
    )

    assert all(
        contrast_record[
            "uncertainty_status"
        ] == "unavailable"
        for contrast_record in (
            uncertainty_limited_contrasts
        )
    )

    fitted_model[
        "jackknife_covariance"
    ] = np.diag(
        [
            0.0004,
            0.0004,
            0.0001
        ]
    )

    fitted_model[
        "coefficients"
    ] = np.asarray(
        [
            -1.0,
            0.0,
            0.0
        ],
        dtype = float
    )

    nonpositive_contrasts = (
        build_model_contrasts(
            prepared_models = {
                "candidate-1": prepared_model
            },
            fitted_models_by_candidate = {
                "candidate-1": fitted_model
            },
            estimand_records = [
                estimand_record
            ],
            condition_records = [
                build_contrast_condition(
                    condition_identity = (
                        first_condition_identity
                    ),
                    reference_test_schedule = (
                        "initial_and_final_only"
                    )
                ),
                build_contrast_condition(
                    condition_identity = (
                        second_condition_identity
                    ),
                    reference_test_schedule = (
                        "periodic_reference_tests"
                    )
                )
            ],
            modeling_policy = modeling_policy
        )
    )

    assert all(
        contrast_record[
            "contrast_status"
        ] == "unavailable"
        for contrast_record in (
            nonpositive_contrasts
        )
    )

    assert all(
        contrast_record[
            "contrast_reason_code"
        ] == "NONPOSITIVE_RATIO_PREDICTION"
        for contrast_record in (
            nonpositive_contrasts
        )
    )


# Confirming that laboratory-adjusted contrasts require shared allocation
def test_build_model_contrasts_rejects_different_laboratories() -> None:
    modeling_policy = load_test_policy()

    first_condition_identity = (
        "Stage_1/TP_k01/periodic_reference_tests"
    )

    second_condition_identity = (
        "Stage_1/TP_k02/periodic_reference_tests"
    )

    candidate_record = {
        "candidate_identity": "candidate-1",
        "fit_identity": "fit-1",
        "support_region_identity": "region-1",
        "template_name": (
            "linear_common_trajectory"
        ),
        "estimator": "weighted_least_squares",
        "aging_mode": "calendar_aging",
        "design_scope": "Stage_1",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": "elapsed_calendar_days",
        "context_family": "laboratory",
        "horizon_basis": "linear",
        "include_horizon_by_treatment_interactions": (
            False
        ),
        "support_interval_minimum": "10.0",
        "support_interval_maximum": "20.0",
        "support_interval_lower_inclusive": "true",
        "support_interval_upper_inclusive": "true",
        "fit_status": "completed"
    }

    prepared_model = {
        "candidate_identity": "candidate-1",
        "fit_identity": "fit-1",
        "candidate_record": candidate_record,
        "model_encoding": {
            "horizon_basis": "linear",
            "horizon_mean": 15.0,
            "horizon_standard_deviation": 5.0,
            "spline_knots": None,
            "ordered_source_terms": [],
            "source_term_encodings": {},
            "context_encodings": []
        },
        "matrix_terms": [],
        "observation_records": [
            {
                "condition_protocol_identity": (
                    first_condition_identity
                ),
                "horizon": 10.0
            },
            {
                "condition_protocol_identity": (
                    second_condition_identity
                ),
                "horizon": 10.0
            }
        ]
    }

    first_condition = build_contrast_condition(
        condition_identity = (
            first_condition_identity
        ),
        reference_test_schedule = (
            "periodic_reference_tests"
        )
    )

    second_condition = build_contrast_condition(
        condition_identity = (
            second_condition_identity
        ),
        reference_test_schedule = (
            "periodic_reference_tests"
        )
    )

    first_condition[
        "laboratory"
    ] = "INT"

    second_condition[
        "laboratory"
    ] = "SIE"

    contrast_records = build_model_contrasts(
        prepared_models = {
            "candidate-1": prepared_model
        },
        fitted_models_by_candidate = {},
        estimand_records = [
            {
                "aging_mode": "calendar_aging",
                "decision_scope": (
                    "condition_protocol_pair"
                ),
                "first_condition_protocol_identity": (
                    first_condition_identity
                ),
                "second_condition_protocol_identity": (
                    second_condition_identity
                ),
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": (
                    "elapsed_calendar_days"
                ),
                "target_estimand_available": "true",
                "final_support_classification": (
                    "supported_with_restrictions"
                ),
                "permitted_bracketing_ranges": (
                    '[{"minimum":10.0,"maximum":20.0}]'
                ),
                "restriction_reasons": "[]"
            }
        ],
        condition_records = [
            first_condition,
            second_condition
        ],
        modeling_policy = modeling_policy
    )

    assert len(
        contrast_records
    ) == 1

    assert contrast_records[0][
        "contrast_status"
    ] == "unavailable"

    assert contrast_records[0][
        "contrast_reason_code"
    ] == "NO_SHARED_OBSERVED_LABORATORY"

    assert contrast_records[0][
        "horizon"
    ] is None


# Confirming mutually exclusive terminal candidate dispositions
def test_assign_final_candidate_dispositions() -> None:
    modeling_policy = load_test_policy()

    candidate_records = [
        {
            "candidate_identity": "not-selected",
            "selection_status": "not_selected",
            "selection_reason_code": (
                "NON_ANCHOR_SUPPORT_REGION"
            ),
            "observation_assembly_status": None,
            "admissibility_status": "not_selected",
            "admissibility_reason_code": None,
            "fit_status": "not_attempted"
        },
        {
            "candidate_identity": "not-assembled",
            "selection_status": "selected",
            "selection_reason_code": "COVERAGE_ANCHOR",
            "observation_assembly_status": "not_assembled",
            "admissibility_status": "not_assembled",
            "admissibility_reason_code": (
                "NO_ELIGIBLE_OBSERVATIONS"
            ),
            "fit_status": "not_attempted"
        },
        {
            "candidate_identity": "inadmissible",
            "selection_status": "selected",
            "selection_reason_code": "COVERAGE_ANCHOR",
            "observation_assembly_status": "completed",
            "admissibility_status": "inadmissible",
            "admissibility_reason_code": (
                "MODEL_MATRIX_RANK_DEFICIENT"
            ),
            "fit_status": "not_attempted"
        },
        {
            "candidate_identity": "fit-failed",
            "selection_status": "selected",
            "selection_reason_code": "COVERAGE_ANCHOR",
            "observation_assembly_status": "completed",
            "admissibility_status": "admissible",
            "admissibility_reason_code": None,
            "fit_status": "failed",
            "fit_reason_code": "MODEL_FIT_FAILED"
        },
        {
            "candidate_identity": "fitted",
            "selection_status": "selected",
            "selection_reason_code": "COVERAGE_ANCHOR",
            "observation_assembly_status": "completed",
            "admissibility_status": "admissible",
            "admissibility_reason_code": None,
            "fit_status": "completed"
        }
    ]

    disposition_counts = (
        assign_final_candidate_dispositions(
            candidate_records = candidate_records,
            modeling_policy = modeling_policy
        )
    )

    assert disposition_counts == {
        "fit_failed": 1,
        "fitted": 1,
        "inadmissible": 1,
        "not_assembled": 1,
        "not_selected": 1
    }

    assert {
        candidate_record[
            "candidate_identity"
        ]: candidate_record[
            "final_disposition"
        ]
        for candidate_record in candidate_records
    } == {
        "not-selected": "not_selected",
        "not-assembled": "not_assembled",
        "inadmissible": "inadmissible",
        "fit-failed": "fit_failed",
        "fitted": "fitted"
    }


# Rejecting an admissible candidate without a terminal fit result
def test_assign_final_candidate_dispositions_rejects_unfinished_fit() -> None:
    modeling_policy = load_test_policy()

    candidate_record = {
        "candidate_identity": "candidate-1",
        "selection_status": "selected",
        "selection_reason_code": "COVERAGE_ANCHOR",
        "observation_assembly_status": "completed",
        "admissibility_status": "admissible",
        "admissibility_reason_code": None,
        "fit_status": "not_attempted"
    }

    with pytest.raises(
        RuntimeError,
        match = "terminal fit status"
    ):
        assign_final_candidate_dispositions(
            candidate_records = [
                candidate_record
            ],
            modeling_policy = modeling_policy
        )


# Confirming deterministic analytical-value serialization
def test_serialize_artifact_value() -> None:
    assert serialize_artifact_value(
        None
    ) == ""

    assert serialize_artifact_value(
        True
    ) == "true"

    assert serialize_artifact_value(
        False
    ) == "false"

    assert serialize_artifact_value(
        1.25
    ) == 1.25

    assert serialize_artifact_value(
        {
            "second": 2,
            "first": 1
        }
    ) == '{"first":1,"second":2}'


# Rejecting non-finite analytical artifact values
@pytest.mark.parametrize(
    "value",
    [
        float(
            "nan"
        ),
        float(
            "inf"
        ),
        float(
            "-inf"
        )
    ]
)
def test_serialize_artifact_value_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(
        RuntimeError,
        match = "non-finite"
    ):
        serialize_artifact_value(
            value
        )


# Sorting records by the declared primary key
def test_normalize_artifact_records_sorts_by_primary_key() -> None:
    normalized_records = normalize_artifact_records(
        table_name = "example_table",
        records = [
            {
                "record_identity": "second",
                "value": 2.0
            },
            {
                "record_identity": "first",
                "value": 1.0
            }
        ],
        field_names = [
            "record_identity",
            "value"
        ],
        primary_key = [
            "record_identity"
        ]
    )

    assert [
        record[
            "record_identity"
        ]
        for record in normalized_records
    ] == [
        "first",
        "second"
    ]


# Rejecting records that differ from the declared table schema
def test_normalize_artifact_records_rejects_schema_difference() -> None:
    with pytest.raises(
        RuntimeError,
        match = "does not match its declared schema"
    ):
        normalize_artifact_records(
            table_name = "example_table",
            records = [
                {
                    "record_identity": "first",
                    "unexpected": 1
                }
            ],
            field_names = [
                "record_identity",
                "value"
            ],
            primary_key = [
                "record_identity"
            ]
        )


# Rejecting duplicate analytical-table primary keys
def test_normalize_artifact_records_rejects_duplicate_primary_key() -> None:
    with pytest.raises(
        RuntimeError,
        match = "duplicate primary key"
    ):
        normalize_artifact_records(
            table_name = "example_table",
            records = [
                {
                    "record_identity": "first",
                    "value": 1.0
                },
                {
                    "record_identity": "first",
                    "value": 2.0
                }
            ],
            field_names = [
                "record_identity",
                "value"
            ],
            primary_key = [
                "record_identity"
            ]
        )


# Writing and validating one deterministic analytical table
def test_write_artifact_table(tmp_path: Path) -> None:
    table_path = (
        tmp_path
        / "example_table.csv"
    )

    table_manifest = write_artifact_table(
        output_path = table_path,
        table_name = "example_table",
        records = [
            {
                "record_identity": "second",
                "value": 2.0,
                "available": False
            },
            {
                "record_identity": "first",
                "value": 1.0,
                "available": True
            }
        ],
        field_names = [
            "record_identity",
            "value",
            "available"
        ],
        primary_key = [
            "record_identity"
        ]
    )

    validate_written_artifact_table(
        table_path = table_path,
        table_manifest = table_manifest
    )

    assert table_manifest[
        "row_count"
    ] == 2

    assert table_manifest[
        "primary_key"
    ] == [
        "record_identity"
    ]

    assert table_path.read_text(
        encoding = "utf-8"
    ) == (
        "record_identity,value,available\n"
        "first,1.0,true\n"
        "second,2.0,false\n"
    )


# Preserving an explicitly declared schema for an empty table
def test_build_artifact_table_schema_for_empty_table() -> None:
    table_schema = build_artifact_table_schema(
        records = [],
        field_names = [
            "candidate_identity",
            "estimate"
        ]
    )

    assert table_schema == [
        {
            "name": "candidate_identity",
            "data_type": "null",
            "nullable": True
        },
        {
            "name": "estimate",
            "data_type": "null",
            "nullable": True
        }
    ]


# Publishing the report after every analytical table
def test_publish_staged_artifacts_replaces_report_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging_directory = (
        tmp_path
        / "staging"
    )

    output_directory = (
        tmp_path
        / "output"
    )

    staging_directory.mkdir()
    output_directory.mkdir()

    (
        staging_directory
        / "table.csv"
    ).write_text(
        "new table\n",
        encoding = "utf-8"
    )

    (
        staging_directory
        / "report.json"
    ).write_text(
        '{"status":"new"}\n',
        encoding = "utf-8"
    )

    published_file_names = []
    original_replace = os.replace

    def record_replace(source: Path, destination: Path) -> None:
        source_path = Path(
            source
        )

        if source_path.parent == staging_directory:
            published_file_names.append(
                source_path.name
            )

        original_replace(
            source,
            destination
        )

    monkeypatch.setattr(
        os,
        "replace",
        record_replace
    )

    publish_staged_artifacts(
        staging_directory = staging_directory,
        output_directory = output_directory,
        expected_file_names = [
            "table.csv",
            "report.json"
        ],
        report_file_name = "report.json"
    )

    assert published_file_names == [
        "table.csv",
        "report.json"
    ]

    assert (
        output_directory
        / "table.csv"
    ).read_text(
        encoding = "utf-8"
    ) == "new table\n"

    assert (
        output_directory
        / "report.json"
    ).read_text(
        encoding = "utf-8"
    ) == '{"status":"new"}\n'


# Restoring the preceding complete artifact set after publication failure
def test_publish_staged_artifacts_rolls_back_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staging_directory = (
        tmp_path
        / "staging"
    )

    output_directory = (
        tmp_path
        / "output"
    )

    staging_directory.mkdir()
    output_directory.mkdir()

    (
        staging_directory
        / "table.csv"
    ).write_text(
        "new table\n",
        encoding = "utf-8"
    )

    (
        staging_directory
        / "report.json"
    ).write_text(
        '{"status":"new"}\n',
        encoding = "utf-8"
    )

    (
        output_directory
        / "table.csv"
    ).write_text(
        "old table\n",
        encoding = "utf-8"
    )

    (
        output_directory
        / "report.json"
    ).write_text(
        '{"status":"old"}\n',
        encoding = "utf-8"
    )

    original_replace = os.replace

    def fail_report_publication(source: Path, destination: Path) -> None:
        source_path = Path(
            source
        )

        if (
            source_path.parent
            == staging_directory
            and source_path.name
            == "report.json"
        ):
            raise OSError(
                "Synthetic report-publication failure."
            )

        original_replace(
            source,
            destination
        )

    monkeypatch.setattr(
        os,
        "replace",
        fail_report_publication
    )

    with pytest.raises(
        OSError,
        match = "Synthetic report-publication failure"
    ):
        publish_staged_artifacts(
            staging_directory = staging_directory,
            output_directory = output_directory,
            expected_file_names = [
                "table.csv",
                "report.json"
            ],
            report_file_name = "report.json"
        )

    assert (
        output_directory
        / "table.csv"
    ).read_text(
        encoding = "utf-8"
    ) == "old table\n"

    assert (
        output_directory
        / "report.json"
    ).read_text(
        encoding = "utf-8"
    ) == '{"status":"old"}\n'


# Restoring the preceding artifact set after final inventory validation fails
def test_publish_staged_artifacts_rolls_back_final_inventory_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_directory = (
        tmp_path
        / "staging"
    )

    output_directory = (
        tmp_path
        / "output"
    )

    staging_directory.mkdir()
    output_directory.mkdir()

    (
        staging_directory
        / "table.csv"
    ).write_text(
        "new table\n",
        encoding = "utf-8"
    )

    (
        staging_directory
        / "report.json"
    ).write_text(
        '{"status":"new"}\n',
        encoding = "utf-8"
    )

    (
        output_directory
        / "table.csv"
    ).write_text(
        "old table\n",
        encoding = "utf-8"
    )

    (
        output_directory
        / "report.json"
    ).write_text(
        '{"status":"old"}\n',
        encoding = "utf-8"
    )

    original_replace = os.replace

    def add_unexpected_file_after_report(source: Path, destination: Path) -> None:
        source_path = Path(
            source
        )

        destination_path = Path(
            destination
        )

        original_replace(
            source,
            destination
        )

        if (
            source_path.parent
            == staging_directory
            and source_path.name
            == "report.json"
        ):
            (
                destination_path.parent
                / "unexpected.txt"
            ).write_text(
                "unexpected\n",
                encoding = "utf-8"
            )

    monkeypatch.setattr(
        os,
        "replace",
        add_unexpected_file_after_report
    )

    with pytest.raises(
        RuntimeError,
        match = "Published artifact inventory"
    ):
        publish_staged_artifacts(
            staging_directory = staging_directory,
            output_directory = output_directory,
            expected_file_names = [
                "table.csv",
                "report.json"
            ],
            report_file_name = "report.json"
        )

    assert (
        output_directory
        / "table.csv"
    ).read_text(
        encoding = "utf-8"
    ) == "old table\n"

    assert (
        output_directory
        / "report.json"
    ).read_text(
        encoding = "utf-8"
    ) == '{"status":"old"}\n'


# Rejecting undeclared files in an analytical-artifact destination
def test_publish_staged_artifacts_rejects_unexpected_existing_file(tmp_path: Path) -> None:
    staging_directory = (
        tmp_path
        / "staging"
    )

    output_directory = (
        tmp_path
        / "output"
    )

    staging_directory.mkdir()
    output_directory.mkdir()

    (
        staging_directory
        / "table.csv"
    ).write_text(
        "new table\n",
        encoding = "utf-8"
    )

    (
        staging_directory
        / "report.json"
    ).write_text(
        '{"status":"new"}\n',
        encoding = "utf-8"
    )

    (
        output_directory
        / "unexpected.txt"
    ).write_text(
        "unexpected\n",
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = "unexpected entries"
    ):
        publish_staged_artifacts(
            staging_directory = staging_directory,
            output_directory = output_directory,
            expected_file_names = [
                "table.csv",
                "report.json"
            ],
            report_file_name = "report.json"
        )


# Running final validation after every declared artifact is published
def test_publish_staged_artifacts_runs_final_validator(tmp_path: Path) -> None:
    staging_directory = (
        tmp_path
        / "staging"
    )

    output_directory = (
        tmp_path
        / "output"
    )

    staging_directory.mkdir()
    output_directory.mkdir()

    (
        staging_directory
        / "table.csv"
    ).write_text(
        "new table\n",
        encoding = "utf-8"
    )

    (
        staging_directory
        / "report.json"
    ).write_text(
        '{"status":"new"}\n',
        encoding = "utf-8"
    )

    validated_file_names = []

    def validate_publication(published_directory: Path) -> None:
        validated_file_names.extend(
            sorted(
                output_path.name
                for output_path in (
                    published_directory.iterdir()
                )
            )
        )

    publish_staged_artifacts(
        staging_directory = staging_directory,
        output_directory = output_directory,
        expected_file_names = [
            "table.csv",
            "report.json"
        ],
        report_file_name = "report.json",
        final_validator = validate_publication
    )

    assert validated_file_names == [
        "report.json",
        "table.csv"
    ]


# Restoring preceding artifacts when final content validation fails
def test_publish_staged_artifacts_rolls_back_final_validation_failure(tmp_path: Path) -> None:
    staging_directory = (
        tmp_path
        / "staging"
    )

    output_directory = (
        tmp_path
        / "output"
    )

    staging_directory.mkdir()
    output_directory.mkdir()

    (
        staging_directory
        / "table.csv"
    ).write_text(
        "new table\n",
        encoding = "utf-8"
    )

    (
        staging_directory
        / "report.json"
    ).write_text(
        '{"status":"new"}\n',
        encoding = "utf-8"
    )

    (
        output_directory
        / "table.csv"
    ).write_text(
        "old table\n",
        encoding = "utf-8"
    )

    (
        output_directory
        / "report.json"
    ).write_text(
        '{"status":"old"}\n',
        encoding = "utf-8"
    )

    def reject_publication(published_directory: Path) -> None:
        raise RuntimeError("Synthetic final validation failure.")

    with pytest.raises(
        RuntimeError,
        match = "Synthetic final validation failure"
    ):
        publish_staged_artifacts(
            staging_directory = staging_directory,
            output_directory = output_directory,
            expected_file_names = [
                "table.csv",
                "report.json"
            ],
            report_file_name = "report.json",
            final_validator = reject_publication
        )

    assert (
        output_directory
        / "table.csv"
    ).read_text(
        encoding = "utf-8"
    ) == "old table\n"

    assert (
        output_directory
        / "report.json"
    ).read_text(
        encoding = "utf-8"
    ) == '{"status":"old"}\n'


# Confirming selected terms reconcile exactly with matrix columns
def test_validate_selected_matrix_term_accounting() -> None:
    candidate_term_records = [
        {
            "term_name": "intercept",
            "term_type": "intercept",
            "term_disposition": "selected"
        },
        {
            "term_name": "horizon_linear",
            "term_type": "horizon_basis",
            "term_disposition": "selected"
        },
        {
            "term_name": (
                "ambient_temperature_celsius"
            ),
            "term_type": "main_effect",
            "term_disposition": "selected"
        },
        {
            "term_name": (
                "horizon_linear:"
                "ambient_temperature_celsius"
            ),
            "term_type": (
                "horizon_by_treatment_interaction"
            ),
            "term_disposition": "not_selected"
        }
    ]

    matrix_terms = [
        {
            "column_name": "intercept",
            "term_type": "intercept"
        },
        {
            "column_name": "horizon_linear",
            "term_type": "horizon_basis"
        },
        {
            "column_name": (
                "ambient_temperature_celsius"
            ),
            "term_type": "main_effect"
        }
    ]

    validate_selected_matrix_term_accounting(
        candidate_identity = "candidate-1",
        candidate_term_records = (
            candidate_term_records
        ),
        matrix_terms = matrix_terms
    )


# Rejecting an unaccounted constructed matrix term
def test_validate_selected_matrix_term_accounting_rejects_unaccounted_term() -> None:
    with pytest.raises(
        RuntimeError,
        match = "do not reconcile"
    ):
        validate_selected_matrix_term_accounting(
            candidate_identity = "candidate-1",
            candidate_term_records = [
                {
                    "term_name": "intercept",
                    "term_type": "intercept",
                    "term_disposition": (
                        "selected"
                    )
                }
            ],
            matrix_terms = [
                {
                    "column_name": "intercept",
                    "term_type": "intercept"
                },
                {
                    "column_name": (
                        "horizon_linear"
                    ),
                    "term_type": (
                        "horizon_basis"
                    )
                }
            ]
        )


# Using cell-jackknife covariance for weighted-regression validation
def test_build_validation_fixed_effect_covariance_uses_training_cell_jackknife() -> None:
    modeling_policy = load_test_policy()

    prepared_model = (
        build_prepared_regression_model(
            "weighted_least_squares"
        )
    )

    fitted_result = fit_prepared_model(
        prepared_model = prepared_model,
        modeling_policy = modeling_policy
    )

    fitted_result[
        "model_based_fixed_effect_covariance"
    ] = np.full(
        (
            prepared_model[
                "model_matrix"
            ].shape[1],
            prepared_model[
                "model_matrix"
            ].shape[1]
        ),
        np.nan,
        dtype = float
    )

    covariance_matrix = (
        build_validation_fixed_effect_covariance(
            training_prepared_model = prepared_model,
            training_fit = fitted_result,
            modeling_policy = modeling_policy
        )
    )

    assert covariance_matrix.shape == (
        prepared_model[
            "model_matrix"
        ].shape[1],
        prepared_model[
            "model_matrix"
        ].shape[1]
    )

    assert np.all(
        np.isfinite(
            covariance_matrix
        )
    )

    assert np.allclose(
        covariance_matrix,
        covariance_matrix.T,
        rtol = 0.0,
        atol = 1e-10
    )


# Retaining point predictions when training-fold covariance is unavailable
def test_build_validation_prediction_intervals_records_failed_training_jackknife(monkeypatch: pytest.MonkeyPatch) -> None:
    modeling_policy = load_test_policy()

    prepared_model = (
        build_prepared_regression_model(
            "weighted_least_squares"
        )
    )

    fitted_result = fit_prepared_model(
        prepared_model = prepared_model,
        modeling_policy = modeling_policy
    )

    def return_failed_deletion(prepared_model: dict, modeling_policy: dict) -> list[dict]:
        return [
            {
                "deleted_cell_identity": "cell-1",
                "fit_status": "failed",
                "fit_reason_code": (
                    "SYNTHETIC_DELETION_FAILURE"
                ),
                "fit_error": (
                    "Synthetic deletion failure."
                ),
                "coefficients": None,
                "optimizer": None,
                "optimizer_attempts": []
            }
        ]

    monkeypatch.setattr(
        (
            "scripts.run_descriptive_models.fit_deleted_cell_coefficients"
        ),
        return_failed_deletion
    )

    prediction_matrix = prepared_model[
        "model_matrix"
    ][
        :2,
        :
    ]

    (
        lower_half_widths,
        upper_half_widths,
        interval_status,
        interval_reason_code
    ) = build_validation_prediction_intervals(
        prediction_matrix = prediction_matrix,
        training_prepared_model = prepared_model,
        training_fit = fitted_result,
        training_cell_count = len(
            set(
                prepared_model[
                    "cell_identities"
                ]
            )
        ),
        modeling_policy = modeling_policy
    )

    assert lower_half_widths == [
        None,
        None
    ]

    assert upper_half_widths == [
        None,
        None
    ]

    assert interval_status == "unavailable"

    assert interval_reason_code == (
        "TRAINING_FOLD_INTERVAL_DELETED_REFIT_FAILED"
    )


# Defaulting command-line execution to validation without model fitting
def test_parse_args_defaults_to_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_descriptive_models"
        ]
    )

    args = descriptive_models.parse_args()

    assert args.operation == "validate"


# Preventing validation-only execution from fitting models
def test_execute_requested_operation_validates_without_fitting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_model_fitting(validated_inputs: dict, output_directory: Path) -> dict:
        raise AssertionError("Validation-only execution attempted to fit models.")

    monkeypatch.setattr(
        descriptive_models,
        "run_descriptive_modeling_analysis",
        reject_model_fitting
    )

    result = (
        descriptive_models.execute_requested_operation(
            validated_inputs = {},
            operation = "validate",
            output_directory = (
                tmp_path
                / "output"
            )
        )
    )

    assert result is None


# Requiring an explicit fitting operation before analytical execution
def test_execute_requested_operation_runs_explicit_fitting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    expected_report = {
        "analysis_status": "completed"
    }

    observed_arguments = {}

    def record_model_fitting(
        validated_inputs: dict,
        output_directory: Path
    ) -> dict:
        observed_arguments[
            "validated_inputs"
        ] = validated_inputs

        observed_arguments[
            "output_directory"
        ] = output_directory

        return expected_report

    monkeypatch.setattr(
        descriptive_models,
        "run_descriptive_modeling_analysis",
        record_model_fitting
    )

    validated_inputs = {
        "validated": True
    }

    output_directory = (
        tmp_path
        / "output"
    )

    report = (
        descriptive_models.execute_requested_operation(
            validated_inputs = validated_inputs,
            operation = "fit",
            output_directory = output_directory
        )
    )

    assert report == expected_report

    assert observed_arguments == {
        "validated_inputs": validated_inputs,
        "output_directory": output_directory
    }


# Rejecting an unsupported analytical operation
def test_execute_requested_operation_rejects_unknown_operation(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match = (
            "Unsupported descriptive-modeling operation"
        )
    ):
        descriptive_models.execute_requested_operation(
            validated_inputs = {},
            operation = "unknown",
            output_directory = (
                tmp_path
                / "output"
            )
        )


# Publishing the complete declared artifact inventory through the public workflow
def test_run_descriptive_modeling_analysis_publishes_complete_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    modeling_policy = load_test_policy()

    output_table_names = modeling_policy[
        "artifact_contract"
    ][
        "output_tables"
    ]

    workflow_results = {
        table_name: []
        for table_name in output_table_names
    }

    workflow_results.update(
        {
            "candidate_disposition_counts": {
                "fitted": 0,
                "fit_failed": 0,
                "inadmissible": 0,
                "not_assembled": 0,
                "not_selected": 0
            },
            "reconciliation": {
                "candidate_count": 0,
                "fitted_candidate_count": 0,
                "failed_fit_candidate_count": 0,
                "inadmissible_candidate_count": 0,
                "not_assembled_candidate_count": 0,
                "not_selected_candidate_count": 0,
                "reconciliation_complete": True
            }
        }
    )

    def return_completed_workflow(validated_inputs: dict) -> dict:
        return workflow_results

    monkeypatch.setattr(
        descriptive_models,
        "execute_descriptive_modeling_workflow",
        return_completed_workflow
    )

    validated_inputs = {
        "modeling_policy": modeling_policy,
        "modeling_policy_sha256": (
            "a"
            * 64
        ),
        "feature_manifest_sha256": (
            "b"
            * 64
        ),
        "design_report_sha256": (
            "c"
            * 64
        ),
        "feature_manifest": {
            "tables": {}
        },
        "design_report": {
            "tables": {}
        }
    }

    output_directory = (
        tmp_path
        / "descriptive_models"
    )

    report = (
        descriptive_models.run_descriptive_modeling_analysis(
            validated_inputs = validated_inputs,
            output_directory = output_directory
        )
    )

    expected_file_names = {
        *(
            f"{table_name}.csv"
            for table_name in output_table_names
        ),
        modeling_policy[
            "artifact_contract"
        ][
            "report_file_name"
        ]
    }

    assert {
        output_path.name
        for output_path in output_directory.iterdir()
    } == expected_file_names

    assert report[
        "analysis_status"
    ] == "completed"

    assert report[
        "modeling_policy"
    ] == modeling_policy

    assert report[
        "modeling_policy_sha256"
    ] == (
        "a"
        * 64
    )

    assert report[
        "reconciliation"
    ][
        "reconciliation_complete"
    ] is True

    for table_manifest in report[
        "tables"
    ].values():
        descriptive_models.validate_written_artifact_table(
            table_path = (
                output_directory
                / table_manifest[
                    "file_name"
                ]
            ),
            table_manifest = table_manifest
        )

    reloaded_report = (
        descriptive_models.load_strict_json_object(
            output_directory
            / modeling_policy[
                "artifact_contract"
            ][
                "report_file_name"
            ]
        )
    )

    assert reloaded_report == report


# Confirming conservative candidate authorization aggregation
def test_build_candidate_registry_uses_most_restrictive_term_authorization() -> None:
    modeling_policy = load_test_policy()

    supported_record = build_scope_record(
        term_name = "charge_rate_c:discharge_rate_c",
        term_type = "pairwise_interaction"
    )

    supported_record[
        "aging_mode"
    ] = "cycle_aging"

    supported_record[
        "horizon_axis"
    ] = (
        "cumulative_full_equivalent_cycles_observed"
    )

    supported_record[
        "cohort"
    ] = "sensitivity"

    supported_record[
        "specification"
    ] = "main_effects_and_pairwise_interactions"

    sensitivity_charge_record = build_scope_record(
        term_name = "charge_rate_c",
        term_type = "main_effect"
    )

    sensitivity_discharge_record = (
        build_scope_record(
            term_name = "discharge_rate_c",
            term_type = "main_effect"
        )
    )

    for sensitivity_record in [
        sensitivity_charge_record,
        sensitivity_discharge_record
    ]:
        sensitivity_record[
            "aging_mode"
        ] = "cycle_aging"

        sensitivity_record[
            "horizon_axis"
        ] = (
            "cumulative_full_equivalent_cycles_observed"
        )

        sensitivity_record[
            "cohort"
        ] = "sensitivity"

        sensitivity_record[
            "specification"
        ] = (
            "main_effects_and_pairwise_interactions"
        )

        sensitivity_record[
            "question_classification"
        ] = "sensitivity_only"

        sensitivity_record[
            "authorization_status"
        ] = (
            "permitted_for_model_evaluation_sensitivity_only"
        )

    scope_records = [
        sensitivity_charge_record,
        sensitivity_discharge_record,
        supported_record
    ]

    (
        candidate_records,
        candidate_term_records
    ) = build_candidate_registry(
        scope_records = scope_records,
        modeling_policy = modeling_policy
    )

    assert all(
        candidate_record[
            "question_classification"
        ]
        == "sensitivity_only"
        for candidate_record in candidate_records
    )

    assert all(
        candidate_record[
            "authorization_status"
        ]
        == (
            "permitted_for_model_evaluation_sensitivity_only"
        )
        for candidate_record in candidate_records
    )

    source_terms = [
        term_record
        for term_record in candidate_term_records
        if term_record[
            "term_source"
        ] == "analytical_scope.csv"
    ]

    assert {
        (
            term_record[
                "term_name"
            ],
            term_record[
                "source_question_classification"
            ]
        )
        for term_record in source_terms
    } == {
        (
            "charge_rate_c",
            "sensitivity_only"
        ),
        (
            "discharge_rate_c",
            "sensitivity_only"
        ),
        (
            "charge_rate_c:discharge_rate_c",
            "supported_with_restrictions"
        )
    }

    derived_terms = [
        term_record
        for term_record in candidate_term_records
        if term_record[
            "term_source"
        ] == "descriptive_modeling_policy"
    ]

    assert all(
        term_record[
            "source_question_classification"
        ]
        is None
        and term_record[
            "source_authorization_status"
        ]
        is None
        for term_record in derived_terms
    )


# Rejecting inconsistent source-term authorization
def test_build_candidate_registry_rejects_mismatched_term_authorization() -> None:
    modeling_policy = load_test_policy()

    scope_record = build_scope_record(
        term_name = (
            "ambient_temperature_celsius"
        ),
        term_type = "main_effect"
    )

    scope_record[
        "question_classification"
    ] = "sensitivity_only"

    with pytest.raises(
        RuntimeError,
        match = (
            "classification conflicts with its authorization status"
        )
    ):
        build_candidate_registry(
            scope_records = [scope_record],
            modeling_policy = modeling_policy
        )


# Confirming complete laboratory-context term accounting
def test_build_candidate_registry_accounts_for_laboratory_context() -> None:
    modeling_policy = load_test_policy()

    scope_record = build_scope_record(
        term_name = (
            "ambient_temperature_celsius"
        ),
        term_type = "main_effect"
    )

    scope_record[
        "context_family"
    ] = "laboratory"

    scope_record[
        "permitted_adjustment_variables"
    ] = '["laboratory"]'

    (
        candidate_records,
        candidate_term_records
    ) = build_candidate_registry(
        scope_records = [
            scope_record
        ],
        modeling_policy = modeling_policy
    )

    for candidate_record in candidate_records:
        context_terms = [
            candidate_term_record
            for candidate_term_record in (
                candidate_term_records
            )
            if (
                candidate_term_record[
                    "candidate_identity"
                ]
                == candidate_record[
                    "candidate_identity"
                ]
                and candidate_term_record[
                    "term_type"
                ]
                == "design_context"
            )
        ]

        assert len(
            context_terms
        ) == 1

        assert context_terms[0][
            "term_name"
        ] == "laboratory"

        assert context_terms[0][
            "term_disposition"
        ] == "selected"

        assert context_terms[0][
            "term_source"
        ] == "analytical_scope.csv"


# Confirming that encoded context columns require an accounted context field
def test_validate_selected_matrix_term_accounting_includes_context() -> None:
    modeling_policy = load_test_policy()

    observations = []

    for observation_record in (
        build_matrix_observations()
    ):
        condition_identity = observation_record[
            "condition_protocol_identity"
        ]

        observations.append(
            {
                **observation_record,
                "laboratory": (
                    "HM"
                    if "TP_k01" in condition_identity
                    else "INT"
                )
            }
        )

    candidate_record = build_matrix_candidate(
        template_name = (
            "linear_common_trajectory"
        ),
        horizon_basis = "linear",
        include_interactions = False
    )

    candidate_record[
        "context_family"
    ] = "laboratory"

    candidate_record[
        "permitted_adjustment_variables"
    ] = '["laboratory"]'

    source_terms = [
        {
            "candidate_identity": "candidate-1",
            "term_name": (
                "ambient_temperature_celsius"
            ),
            "term_type": "main_effect",
            "term_disposition": "selected"
        }
    ]

    model_encoding = build_model_encoding(
        candidate_record = candidate_record,
        candidate_term_records = source_terms,
        observation_records = observations,
        modeling_policy = modeling_policy
    )

    (
        model_matrix,
        matrix_terms
    ) = transform_model_matrix(
        candidate_record = candidate_record,
        observation_records = observations,
        model_encoding = model_encoding
    )

    assert model_matrix.shape[1] == 4

    accounted_terms = [
        *source_terms,
        {
            "candidate_identity": "candidate-1",
            "term_name": "intercept",
            "term_type": "intercept",
            "term_disposition": "selected"
        },
        {
            "candidate_identity": "candidate-1",
            "term_name": "horizon_linear",
            "term_type": "horizon_basis",
            "term_disposition": "selected"
        },
        {
            "candidate_identity": "candidate-1",
            "term_name": "laboratory",
            "term_type": "design_context",
            "term_disposition": "selected"
        }
    ]

    validate_selected_matrix_term_accounting(
        candidate_identity = "candidate-1",
        candidate_term_records = accounted_terms,
        matrix_terms = matrix_terms
    )

    with pytest.raises(
        RuntimeError,
        match = "Selected context terms do not reconcile"
    ):
        validate_selected_matrix_term_accounting(
            candidate_identity = "candidate-1",
            candidate_term_records = [
                candidate_term_record
                for candidate_term_record in (
                    accounted_terms
                )
                if candidate_term_record[
                    "term_type"
                ] != "design_context"
            ],
            matrix_terms = matrix_terms
        )


@pytest.mark.parametrize(
    (
        "estimator",
        "expected_validation_interval_deletions",
        "expected_total_fit_calls"
    ),
    [
        (
            "weighted_least_squares",
            16,
            28
        ),
        (
            "linear_mixed_effects",
            0,
            12
        )
    ]
)

# Confirming deterministic upper bounds for the declared fitting workload
def test_build_modeling_execution_plan(estimator: str, expected_validation_interval_deletions: int,
                                       expected_total_fit_calls: int
) -> None:
    prepared_model = (
        build_prepared_regression_model(
            estimator
        )
    )

    execution_plan = (
        build_modeling_execution_plan(
            prepared_models = {
                prepared_model[
                    "candidate_identity"
                ]: prepared_model
            },
            modeling_policy = load_test_policy()
        )
    )

    assert execution_plan[
        "prepared_candidate_count"
    ] == 1

    assert execution_plan[
        "unique_full_fit_count"
    ] == 1

    assert execution_plan[
        "full_data_deletion_refit_upper_bound"
    ] == 8

    assert execution_plan[
        "validation_fold_fit_upper_bound"
    ] == 3

    assert execution_plan[
        "validation_interval_deletion_refit_upper_bound"
    ] == expected_validation_interval_deletions

    assert execution_plan[
        "validation_assignment_unavailable_count"
    ] == 0

    assert execution_plan[
        "total_fit_call_upper_bound"
    ] == expected_total_fit_calls


# Classifying non-finite validation metrics as unavailable
def test_calculate_grouped_validation_metrics_rejects_nonfinite_result() -> None:
    prediction_records = [
        {
            "candidate_identity": "candidate-1",
            "source_identity": "source-1",
            "cell_identity": "cell-1",
            "observed_outcome": 0.0,
            "predicted_outcome": 1e200,
            "prediction_interval_status": (
                "unavailable"
            ),
            "prediction_interval_lower": None,
            "prediction_interval_upper": None
        },
        {
            "candidate_identity": "candidate-1",
            "source_identity": "source-2",
            "cell_identity": "cell-2",
            "observed_outcome": 0.0,
            "predicted_outcome": 1e200,
            "prediction_interval_status": (
                "unavailable"
            ),
            "prediction_interval_lower": None,
            "prediction_interval_upper": None
        }
    ]

    with np.errstate(
        over = "ignore",
        invalid = "ignore"
    ):
        with pytest.raises(
            descriptive_models.CandidateValidationUnavailableError
        ) as exception:
            calculate_grouped_validation_metrics(
                prediction_records
            )

    assert exception.value.reason_code == (
        "NONFINITE_VALIDATION_METRIC"
    )


# Qualifying completion when an attempted estimator has no accepted fit
def test_resolve_completed_analysis_status_records_estimator_family_failure() -> None:
    estimator_outcomes = {
        "weighted_least_squares": {
            "availability_status": "available"
        },
        "linear_mixed_effects": {
            "availability_status": (
                "unavailable_all_fit_attempts_failed"
            )
        }
    }

    assert (
        resolve_completed_analysis_status(
            estimator_outcomes
        )
        == "completed_with_model_family_failures"
    )


# Preserving ordinary completion when every attempted estimator is available
def test_resolve_completed_analysis_status_accepts_available_estimators() -> None:
    estimator_outcomes = {
        "weighted_least_squares": {
            "availability_status": "available"
        },
        "linear_mixed_effects": {
            "availability_status": (
                "not_evaluated_no_fit_attempts"
            )
        }
    }

    assert (
        resolve_completed_analysis_status(
            estimator_outcomes
        )
        == "completed"
    )


# Summarizing estimator outcomes without changing candidate dispositions
def test_build_candidate_outcome_summary() -> None:
    modeling_policy = load_test_policy()

    candidate_records = [
        {
            "estimator": "weighted_least_squares",
            "final_disposition": "fitted"
        },
        {
            "estimator": "linear_mixed_effects",
            "final_disposition": "fit_failed"
        },
        {
            "estimator": "linear_mixed_effects",
            "final_disposition": "inadmissible"
        }
    ]

    summary = build_candidate_outcome_summary(
        candidate_records = candidate_records,
        grouping_field = "estimator",
        expected_values = [
            "weighted_least_squares",
            "linear_mixed_effects"
        ],
        modeling_policy = modeling_policy
    )

    assert summary[
        "weighted_least_squares"
    ][
        "availability_status"
    ] == "available"

    assert summary[
        "linear_mixed_effects"
    ][
        "fit_attempted_count"
    ] == 1

    assert summary[
        "linear_mixed_effects"
    ][
        "availability_status"
    ] == (
        "unavailable_all_fit_attempts_failed"
    )


# Summarizing result availability only among fitted candidates
def test_build_fitted_result_availability() -> None:
    candidate_records = [
        {
            "final_disposition": "fitted",
            "uncertainty_status": "completed",
            "validation_status": "unavailable",
            "influence_status": "completed"
        },
        {
            "final_disposition": "fitted",
            "uncertainty_status": "unavailable",
            "validation_status": "completed",
            "influence_status": "unavailable"
        },
        {
            "final_disposition": "fit_failed",
            "uncertainty_status": "not_requested",
            "validation_status": "not_requested",
            "influence_status": "not_requested"
        }
    ]

    assert build_fitted_result_availability(
        candidate_records
    ) == {
        "uncertainty": {
            "completed": 1,
            "unavailable": 1
        },
        "validation": {
            "completed": 1,
            "unavailable": 1
        },
        "influence": {
            "completed": 1,
            "unavailable": 1
        }
    }


# Summarizing within-cell and condition-specific horizon replication
def test_build_longitudinal_structure_diagnostics_fully_longitudinal() -> None:
    observation_records = [
        {
            "cell_identity": "cell-1",
            "condition_protocol_identity": "condition-a",
            "horizon": 10.0
        },
        {
            "cell_identity": "cell-1",
            "condition_protocol_identity": "condition-a",
            "horizon": 20.0
        },
        {
            "cell_identity": "cell-2",
            "condition_protocol_identity": "condition-a",
            "horizon": 10.0
        },
        {
            "cell_identity": "cell-2",
            "condition_protocol_identity": "condition-a",
            "horizon": 20.0
        }
    ]

    diagnostics = build_longitudinal_structure_diagnostics(
        observation_records
    )

    assert diagnostics["repeated_cell_count"] == 2
    assert diagnostics["single_observation_cell_count"] == 0
    assert diagnostics["minimum_condition_unique_horizon_count"] == 2
    assert diagnostics["minimum_condition_repeated_cell_count"] == 2
    assert diagnostics["longitudinal_structure_status"] == "fully_longitudinal"


# Reporting a purely cross-sectional dataset
def test_build_longitudinal_structure_diagnostics_cross_sectional() -> None:
    observation_records = [
        {
            "cell_identity": "cell-1",
            "condition_protocol_identity": "condition-a",
            "horizon": 10.0
        },
        {
            "cell_identity": "cell-2",
            "condition_protocol_identity": "condition-a",
            "horizon": 20.0
        }
    ]

    diagnostics = build_longitudinal_structure_diagnostics(
        observation_records
    )

    assert diagnostics["repeated_cell_count"] == 0
    assert diagnostics["single_observation_cell_count"] == 2
    assert diagnostics["longitudinal_structure_status"] == "cross_sectional"


# Cross-sectional data must not be admitted as a mixed-effects trajectory
def test_random_intercept_inadmissible_without_repeated_cells() -> None:
    modeling_policy = load_test_policy()

    observation_records = [
        {
            "candidate_identity": "candidate-1",
            "source_identity": "source-1",
            "cell_identity": "cell-1",
            "condition_protocol_identity": "condition-a",
            "horizon": 10.0,
            "outcome_value": 0.98,
            "inside_evaluation_interval": True
        },
        {
            "candidate_identity": "candidate-1",
            "source_identity": "source-2",
            "cell_identity": "cell-2",
            "condition_protocol_identity": "condition-a",
            "horizon": 20.0,
            "outcome_value": 0.97,
            "inside_evaluation_interval": True
        }
    ]

    candidate_record = {
        "candidate_identity": "candidate-1",
        "estimator": "linear_mixed_effects",
        "include_horizon_by_treatment_interactions": False
    }

    model_matrix = np.asarray(
        [[1.0], [1.0]],
        dtype = float
    )

    outcome_values = np.asarray(
        [0.98, 0.97],
        dtype = float
    )

    weights = np.asarray(
        [1.0, 1.0],
        dtype = float
    )

    result = evaluate_candidate_admissibility(
        candidate_record = candidate_record,
        observation_records = observation_records,
        model_matrix = model_matrix,
        outcome_values = outcome_values,
        regression_weights = weights,
        model_encoding = {
            "unique_horizon_count": 2,
            "non_boundary_unique_horizon_count": None
        },
        modeling_policy = modeling_policy
    )

    assert result["admissibility_status"] == "inadmissible"
    assert result["admissibility_reason_code"] == (
        "RANDOM_INTERCEPT_NOT_STRUCTURALLY_IDENTIFIABLE"
    )


# A common-trajectory model without within-cell replication is inadmissible
def test_common_trajectory_inadmissible_without_repeated_cells() -> None:
    modeling_policy = load_test_policy()

    observation_records = [
        {
            "candidate_identity": "candidate-1",
            "source_identity": "source-1",
            "cell_identity": "cell-1",
            "condition_protocol_identity": "condition-a",
            "horizon": 10.0,
            "outcome_value": 0.98,
            "inside_evaluation_interval": True
        },
        {
            "candidate_identity": "candidate-1",
            "source_identity": "source-2",
            "cell_identity": "cell-2",
            "condition_protocol_identity": "condition-a",
            "horizon": 20.0,
            "outcome_value": 0.97,
            "inside_evaluation_interval": True
        }
    ]

    candidate_record = {
        "candidate_identity": "candidate-1",
        "estimator": "weighted_least_squares",
        "include_horizon_by_treatment_interactions": False
    }

    model_matrix = np.asarray(
        [[1.0, 0.0], [1.0, 1.0]],
        dtype = float
    )

    outcome_values = np.asarray(
        [0.98, 0.97],
        dtype = float
    )

    weights = np.asarray(
        [1.0, 1.0],
        dtype = float
    )

    result = evaluate_candidate_admissibility(
        candidate_record = candidate_record,
        observation_records = observation_records,
        model_matrix = model_matrix,
        outcome_values = outcome_values,
        regression_weights = weights,
        model_encoding = {
            "unique_horizon_count": 2,
            "non_boundary_unique_horizon_count": None
        },
        modeling_policy = modeling_policy
    )

    assert result["admissibility_status"] == "inadmissible"
    assert result["admissibility_reason_code"] == (
        "INSUFFICIENT_REPEATED_CELLS"
    )


# Assignment of downstream analytical availability from fit outcomes
def test_assign_candidate_analytical_result_status() -> None:
    candidate_records = [
        {
            "candidate_identity": "a",
            "fit_status": "completed",
            "validation_status": "completed",
            "uncertainty_status": "completed",
            "influence_status": "completed"
        },
        {
            "candidate_identity": "b",
            "fit_status": "completed",
            "validation_status": "unavailable",
            "validation_reason_code": "NO_SHARED_OBSERVED_LABORATORY",
            "uncertainty_status": "completed",
            "influence_status": "completed"
        },
        {
            "candidate_identity": "c",
            "fit_status": "completed",
            "validation_status": "completed",
            "uncertainty_status": "unavailable",
            "uncertainty_reason_code": "DELETED_REFIT_FAILED",
            "influence_status": "unavailable",
            "influence_reason_code": "DELETED_REFIT_FAILED"
        },
        {
            "candidate_identity": "d",
            "fit_status": "failed",
            "fit_reason_code": "MIXED_EFFECTS_FIT_FAILED"
        }
    ]

    assign_candidate_analytical_result_status(
        candidate_records
    )

    by_identity = {
        candidate_record["candidate_identity"]: candidate_record
        for candidate_record in candidate_records
    }

    assert by_identity["a"]["analytical_result_status"] == "available"
    assert by_identity["a"]["numerical_fit_status"] == "completed"

    assert by_identity["b"]["analytical_result_status"] == "diagnostic_only"
    assert by_identity["b"]["analytical_result_reason_code"] == (
        "NO_SHARED_OBSERVED_LABORATORY"
    )

    assert by_identity["c"]["analytical_result_status"] == "diagnostic_only"
    assert by_identity["c"]["analytical_result_reason_code"] == (
        "DELETED_REFIT_FAILED"
    )

    assert by_identity["d"]["analytical_result_status"] == "not_available"
    assert by_identity["d"]["analytical_result_reason_code"] == (
        "MIXED_EFFECTS_FIT_FAILED"
    )


# Analytical handoff summary reconciles candidate and contrast statuses
def test_build_analytical_handoff_summary() -> None:
    candidate_records = [
        {"analytical_result_status": "available"},
        {"analytical_result_status": "available"},
        {"analytical_result_status": "diagnostic_only"}
    ]

    contrast_records = [
        {"contrast_status": "completed"},
        {"contrast_status": "completed"},
        {"contrast_status": "diagnostic_only"},
        {"contrast_status": "unavailable"}
    ]

    estimator_outcomes = {
        "weighted_least_squares": {
            "availability_status": "available"
        },
        "linear_mixed_effects": {
            "availability_status": (
                "unavailable_all_fit_attempts_failed"
            )
        }
    }

    summary = build_analytical_handoff_summary(
        candidate_records = candidate_records,
        contrast_records = contrast_records,
        estimator_outcomes = estimator_outcomes
    )

    assert summary["candidate_analytical_status_counts"] == {
        "available": 2,
        "diagnostic_only": 1
    }
    assert summary["contrast_status_counts"] == {
        "completed": 2,
        "diagnostic_only": 1,
        "unavailable": 1
    }
    assert summary["unavailable_estimator_families"] == [
        "linear_mixed_effects"
    ]
    assert summary["analytical_handoff_status"] == (
        "ready_with_model_family_and_candidate_restrictions"
    )


# Rejecting a differential trajectory that lacks per-condition horizon variation
def test_differential_trajectory_inadmissible_without_condition_horizon_variation() -> None:
    modeling_policy = load_test_policy()

    observation_records = [
        {
            "cell_identity": "cell-1",
            "condition_protocol_identity": "condition-a",
            "horizon": 10.0,
            "outcome_value": 0.99,
            "inside_evaluation_interval": True
        },
        {
            "cell_identity": "cell-2",
            "condition_protocol_identity": "condition-a",
            "horizon": 10.0,
            "outcome_value": 0.98,
            "inside_evaluation_interval": True
        },
        {
            "cell_identity": "cell-3",
            "condition_protocol_identity": "condition-a",
            "horizon": 10.0,
            "outcome_value": 0.97,
            "inside_evaluation_interval": True
        },
        {
            "cell_identity": "cell-4",
            "condition_protocol_identity": "condition-b",
            "horizon": 10.0,
            "outcome_value": 0.99,
            "inside_evaluation_interval": True
        },
        {
            "cell_identity": "cell-4",
            "condition_protocol_identity": "condition-b",
            "horizon": 20.0,
            "outcome_value": 0.98,
            "inside_evaluation_interval": True
        },
        {
            "cell_identity": "cell-5",
            "condition_protocol_identity": "condition-b",
            "horizon": 10.0,
            "outcome_value": 0.99,
            "inside_evaluation_interval": True
        },
        {
            "cell_identity": "cell-5",
            "condition_protocol_identity": "condition-b",
            "horizon": 20.0,
            "outcome_value": 0.98,
            "inside_evaluation_interval": True
        }
    ]

    candidate_record = {
        "candidate_identity": "candidate-1",
        "estimator": "weighted_least_squares",
        "include_horizon_by_treatment_interactions": True
    }

    result = evaluate_candidate_admissibility(
        candidate_record = candidate_record,
        observation_records = observation_records,
        model_matrix = np.ones(
            (
                7,
                2
            )
        ),
        outcome_values = np.array(
            [
                observation_record[
                    "outcome_value"
                ]
                for observation_record in (
                    observation_records
                )
            ]
        ),
        regression_weights = np.ones(
            7
        ),
        model_encoding = {
            "unique_horizon_count": 2,
            "non_boundary_unique_horizon_count": None
        },
        modeling_policy = modeling_policy
    )

    assert result[
        "admissibility_status"
    ] == "inadmissible"

    assert result[
        "admissibility_reason_code"
    ] == "INSUFFICIENT_CONDITION_HORIZON_VARIATION"


# Rejecting a differential trajectory that lacks per-condition repeated cells
def test_differential_trajectory_inadmissible_without_condition_repeated_cells() -> None:
    modeling_policy = load_test_policy()

    observation_records = [
        {
            "cell_identity": "cell-1",
            "condition_protocol_identity": "condition-a",
            "horizon": 10.0,
            "outcome_value": 0.99,
            "inside_evaluation_interval": True
        },
        {
            "cell_identity": "cell-2",
            "condition_protocol_identity": "condition-a",
            "horizon": 10.0,
            "outcome_value": 0.98,
            "inside_evaluation_interval": True
        },
        {
            "cell_identity": "cell-3",
            "condition_protocol_identity": "condition-a",
            "horizon": 10.0,
            "outcome_value": 0.99,
            "inside_evaluation_interval": True
        },
        {
            "cell_identity": "cell-3",
            "condition_protocol_identity": "condition-a",
            "horizon": 20.0,
            "outcome_value": 0.98,
            "inside_evaluation_interval": True
        },
        {
            "cell_identity": "cell-4",
            "condition_protocol_identity": "condition-b",
            "horizon": 10.0,
            "outcome_value": 0.99,
            "inside_evaluation_interval": True
        },
        {
            "cell_identity": "cell-4",
            "condition_protocol_identity": "condition-b",
            "horizon": 20.0,
            "outcome_value": 0.98,
            "inside_evaluation_interval": True
        },
        {
            "cell_identity": "cell-5",
            "condition_protocol_identity": "condition-b",
            "horizon": 10.0,
            "outcome_value": 0.99,
            "inside_evaluation_interval": True
        },
        {
            "cell_identity": "cell-5",
            "condition_protocol_identity": "condition-b",
            "horizon": 20.0,
            "outcome_value": 0.98,
            "inside_evaluation_interval": True
        }
    ]

    candidate_record = {
        "candidate_identity": "candidate-1",
        "estimator": "weighted_least_squares",
        "include_horizon_by_treatment_interactions": True
    }

    result = evaluate_candidate_admissibility(
        candidate_record = candidate_record,
        observation_records = observation_records,
        model_matrix = np.ones(
            (
                8,
                2
            )
        ),
        outcome_values = np.array(
            [
                observation_record[
                    "outcome_value"
                ]
                for observation_record in (
                    observation_records
                )
            ]
        ),
        regression_weights = np.ones(
            8
        ),
        model_encoding = {
            "unique_horizon_count": 2,
            "non_boundary_unique_horizon_count": None
        },
        modeling_policy = modeling_policy
    )

    assert result[
        "admissibility_status"
    ] == "inadmissible"

    assert result[
        "admissibility_reason_code"
    ] == "INSUFFICIENT_CONDITION_REPEATED_CELLS"


# Rejecting a report whose analysis_registration_status does not match the frozen policy
def test_validate_report_rejects_registration_status_mismatch() -> None:
    modeling_policy = load_test_policy()

    report = {
        "analysis_registration_status": "confirmatory",
        "analysis_registration_rationale": modeling_policy["reporting"]["analysis_registration_rationale"],
        "analysis_status": "completed",
        "modeling_policy": modeling_policy,
        "independent_experimental_unit": "cell",
        "outcome_magnitude_or_direction_used_for_candidate_selection": False,
        "performance_selected_model_declared": False,
        "causal_interpretation_authorized": False,
        "policy_recommendation_authorized": False,
        "estimator_outcomes": {},
        "template_outcomes": {},
        "candidate_disposition_counts": {},
        "reconciliation": {"reconciliation_complete": True},
        "fitted_result_availability": {"uncertainty": {"completed": 0, "unavailable": 0},
                                       "validation":  {"completed": 0, "unavailable": 0},
                                       "influence":   {"completed": 0, "unavailable": 0}},
        "analytical_handoff": {
            "analytical_handoff_status": "not_ready_no_validated_contrasts",
            "candidate_analytical_status_counts": {},
            "contrast_status_counts": {},
            "unavailable_estimator_families": [],
        },
        "tables": {},
    }

    with pytest.raises(RuntimeError, match = "analysis_registration_status"):
        validate_descriptive_modeling_report(report, modeling_policy)


@pytest.mark.parametrize(
    "field_name",
    [
        "calendar_and_cycle_models_separated",
        "primary_and_sensitivity_cohorts_separated"
    ]
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        False,
        None
    ]
)
# Rejecting an invalid required-true separation declaration
def test_validate_report_rejects_invalid_separation_declaration(field_name: str, invalid_value: object) -> None:
    modeling_policy = load_test_policy()

    report = {
        "analysis_registration_status": modeling_policy["reporting"]["analysis_registration_status"],
        "analysis_registration_rationale": modeling_policy["reporting"]["analysis_registration_rationale"],
        "analysis_status": "completed",
        "modeling_policy": modeling_policy,
        "independent_experimental_unit": "cell",
        "calendar_and_cycle_models_separated": True,
        "primary_and_sensitivity_cohorts_separated": True,
        "outcome_magnitude_or_direction_used_for_candidate_selection": False,
        "performance_selected_model_declared": False,
        "causal_interpretation_authorized": False,
        "policy_recommendation_authorized": False,
        "estimator_outcomes": {},
        "template_outcomes": {},
        "candidate_disposition_counts": {},
        "reconciliation": {"reconciliation_complete": True},
        "fitted_result_availability": {
            "uncertainty": {"completed": 0, "unavailable": 0},
            "validation": {"completed": 0, "unavailable": 0},
            "influence": {"completed": 0, "unavailable": 0}
        },
        "analytical_handoff": {
            "analytical_handoff_status": "not_ready_no_validated_contrasts",
            "candidate_analytical_status_counts": {},
            "contrast_status_counts": {},
            "unavailable_estimator_families": []
        },
        "tables": {}
    }

    report[field_name] = invalid_value

    with pytest.raises(
        RuntimeError,
        match = "invalid analytical declarations"
    ):
        validate_descriptive_modeling_report(
            report = report,
            modeling_policy = modeling_policy
        )


# Exercising the complete model-construction and analytical-qualification path
def test_modeling_pipeline_preserves_training_and_evaluation_contract() -> None:
    modeling_policy = load_test_policy()

    first_condition_identity = (
        "Stage_1/TP_k01/"
        "periodic_reference_tests"
    )

    second_condition_identity = (
        "Stage_1/TP_k02/"
        "periodic_reference_tests"
    )

    condition_definitions = [
        {
            "condition_identity": (
                first_condition_identity
            ),
            "temperature": "10.0",
            "horizons": [
                5.0,
                10.0,
                15.0,
                20.0,
                25.0
            ]
        },
        {
            "condition_identity": (
                second_condition_identity
            ),
            "temperature": "23.0",
            "horizons": [
                5.0,
                10.0,
                15.0,
                25.0
            ]
        }
    ]

    cell_records = []
    membership_records = []
    reference_records = []

    for condition_index, condition_definition in enumerate(
        condition_definitions,
        start = 1
    ):
        for cell_number in range(
            1,
            7
        ):
            cell_identity = (
                f"Stage_1/TP_k"
                f"{condition_index:02d}_"
                f"{cell_number:02d}"
            )

            cell_record = build_cell_record(
                cell_identity = cell_identity,
                condition_identity = (
                    condition_definition[
                        "condition_identity"
                    ]
                ),
                cell_number = cell_number
            )

            cell_record[
                "ambient_temperature_celsius"
            ] = condition_definition[
                "temperature"
            ]

            cell_records.append(
                cell_record
            )

            membership_record = (
                build_membership_record(
                    cell_identity = cell_identity,
                    condition_identity = (
                        condition_definition[
                            "condition_identity"
                        ]
                    )
                )
            )

            membership_record[
                "upper_inclusive"
            ] = "true"

            membership_record[
                "supporting_cell_eligible_observation_count"
            ] = (
                "3"
                if condition_index == 1
                else "2"
            )

            membership_record[
                "cell_horizon_maximum"
            ] = (
                "20.0"
                if condition_index == 1
                else "15.0"
            )

            membership_records.append(
                membership_record
            )

            for sequence_number, horizon in enumerate(
                condition_definition[
                    "horizons"
                ],
                start = 1
            ):
                outcome_value = (
                    1.05
                    - 0.001
                    * horizon
                    - 0.00001
                    * horizon ** 2
                    - 0.01
                    * (
                        condition_index
                        - 1
                    )
                    + 0.0002
                    * cell_number
                )

                reference_records.append(
                    build_reference_record(
                        cell_identity = cell_identity,
                        sequence_number = (
                            sequence_number
                        ),
                        horizon = str(
                            horizon
                        ),
                        outcome = str(
                            outcome_value
                        )
                    )
                )

    scope_record = build_scope_record(
        term_name = (
            "ambient_temperature_celsius"
        ),
        term_type = "main_effect"
    )

    scope_record[
        "support_interval_upper_inclusive"
    ] = "true"

    scope_record[
        "condition_protocol_count"
    ] = "2"

    scope_record[
        "independent_cell_count"
    ] = "12"

    scope_record[
        "analytical_observation_count"
    ] = "30"

    (
        candidate_records,
        candidate_term_records
    ) = build_candidate_registry(
        scope_records = [
            scope_record
        ],
        modeling_policy = modeling_policy
    )

    select_model_candidates(
        candidate_records = candidate_records,
        candidate_term_records = (
            candidate_term_records
        ),
        modeling_policy = modeling_policy
    )

    retained_templates = {
        (
            "restricted_cubic_spline_"
            "common_trajectory"
        ),
        (
            "restricted_cubic_spline_"
            "differential_trajectory"
        )
    }

    candidate_records = [
        candidate_record
        for candidate_record in candidate_records
        if candidate_record[
            "template_name"
        ] in retained_templates
    ]

    candidate_identities = {
        candidate_record[
            "candidate_identity"
        ]
        for candidate_record in candidate_records
    }

    candidate_term_records = [
        candidate_term_record
        for candidate_term_record in (
            candidate_term_records
        )
        if candidate_term_record[
            "candidate_identity"
        ] in candidate_identities
    ]

    assert len(
        candidate_records
    ) == 2

    assert all(
        candidate_record[
            "selection_status"
        ] == "selected"
        for candidate_record in candidate_records
    )

    (
        observation_records,
        accounting_records
    ) = assemble_candidate_observations(
        candidate_records = candidate_records,
        modeling_policy = modeling_policy,
        cell_records = cell_records,
        reference_records = reference_records,
        support_membership_records = (
            membership_records
        )
    )

    assert len(
        accounting_records
    ) == 2

    assert all(
        accounting_record[
            "included_record_count"
        ] == 54
        for accounting_record in accounting_records
    )

    assert all(
        accounting_record[
            "inside_evaluation_interval_count"
        ] == 30
        for accounting_record in accounting_records
    )

    assert all(
        accounting_record[
            "outside_evaluation_interval_count"
        ] == 24
        for accounting_record in accounting_records
    )

    assert all(
        accounting_record[
            "inside_evaluation_interval_count"
        ]
        + accounting_record[
            "outside_evaluation_interval_count"
        ]
        == accounting_record[
            "included_record_count"
        ]
        for accounting_record in accounting_records
    )

    assert any(
        not observation_record[
            "inside_evaluation_interval"
        ]
        for observation_record in observation_records
    )

    (
        prepared_models,
        matrix_diagnostic_records
    ) = prepare_candidate_model_matrices(
        candidate_records = candidate_records,
        candidate_term_records = (
            candidate_term_records
        ),
        candidate_observation_records = (
            observation_records
        ),
        modeling_policy = modeling_policy
    )

    assert len(
        prepared_models
    ) == 2

    candidates_by_template = {
        candidate_record[
            "template_name"
        ]: candidate_record
        for candidate_record in candidate_records
    }

    common_candidate = candidates_by_template[
        (
            "restricted_cubic_spline_"
            "common_trajectory"
        )
    ]

    differential_candidate = candidates_by_template[
        (
            "restricted_cubic_spline_"
            "differential_trajectory"
        )
    ]

    for candidate_record in candidate_records:
        assert candidate_record[
            "support_member_cell_count"
        ] == 12

        assert candidate_record[
            "assembled_model_cell_count"
        ] == 12

        assert candidate_record[
            "support_condition_count"
        ] == 2

        assert candidate_record[
            "assembled_model_condition_count"
        ] == 2

        assert candidate_record[
            "assembled_model_observation_count"
        ] == 54

        assert candidate_record[
            "inside_evaluation_interval_observation_count"
        ] == 30

        assert candidate_record[
            "outside_evaluation_interval_observation_count"
        ] == 24

    (
        fitted_models_by_candidate,
        coefficient_records,
        fit_diagnostic_records
    ) = fit_candidate_models(
        prepared_models = prepared_models,
        modeling_policy = modeling_policy
    )

    assert set(
        fitted_models_by_candidate
    ) == candidate_identities

    influence_records = (
        calculate_uncertainty_and_influence(
            prepared_models = prepared_models,
            fitted_models_by_candidate = (
                fitted_models_by_candidate
            ),
            coefficient_records = (
                coefficient_records
            ),
            modeling_policy = modeling_policy
        )
    )

    assert all(
        candidate_record[
            "uncertainty_status"
        ] == "completed"
        for candidate_record in candidate_records
    )

    (
        validation_fold_records,
        validation_prediction_records,
        validation_metric_records
    ) = run_grouped_validation(
        prepared_models = prepared_models,
        fitted_models_by_candidate = (
            fitted_models_by_candidate
        ),
        modeling_policy = modeling_policy
    )

    assert validation_fold_records

    observations_by_key = {
        (
            observation_record[
                "candidate_identity"
            ],
            observation_record[
                "source_identity"
            ]
        ): observation_record
        for observation_record in observation_records
    }

    assert validation_prediction_records

    assert all(
        observations_by_key[
            (
                prediction_record[
                    "candidate_identity"
                ],
                prediction_record[
                    "source_identity"
                ]
            )
        ][
            "inside_evaluation_interval"
        ]
        is True
        for prediction_record in (
            validation_prediction_records
        )
    )

    assert common_candidate[
        "validation_status"
    ] == "completed"

    assert differential_candidate[
        "validation_status"
    ] == "unavailable"

    assert differential_candidate[
        "validation_reason_code"
    ] == (
        "TRAINING_FOLD_"
        "INSUFFICIENT_RESIDUAL_INDEPENDENT_"
        "CELL_DEGREES_OF_FREEDOM"
    )

    assign_candidate_analytical_result_status(
        candidate_records
    )

    assert common_candidate[
        "analytical_result_status"
    ] == "available"

    assert differential_candidate[
        "analytical_result_status"
    ] == "diagnostic_only"

    first_condition_record = (
        build_contrast_condition(
            condition_identity = (
                first_condition_identity
            ),
            reference_test_schedule = (
                "periodic_reference_tests"
            )
        )
    )

    first_condition_record[
        "ambient_temperature_celsius"
    ] = "10.0"

    second_condition_record = (
        build_contrast_condition(
            condition_identity = (
                second_condition_identity
            ),
            reference_test_schedule = (
                "periodic_reference_tests"
            )
        )
    )

    second_condition_record[
        "ambient_temperature_celsius"
    ] = "23.0"

    estimand_record = {
        "aging_mode": "calendar_aging",
        "decision_scope": (
            "condition_protocol_pair"
        ),
        "first_condition_protocol_identity": (
            first_condition_identity
        ),
        "second_condition_protocol_identity": (
            second_condition_identity
        ),
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": (
            "elapsed_calendar_days"
        ),
        "target_estimand_available": "true",
        "final_support_classification": (
            "supported_with_restrictions"
        ),
        "permitted_bracketing_ranges": (
            '[{"minimum":10.0,"maximum":20.0}]'
        ),
        "restriction_reasons": (
            '["ASSIGNMENT_MECHANISM_NOT_ESTABLISHED"]'
        )
    }

    contrast_records = build_model_contrasts(
        prepared_models = prepared_models,
        fitted_models_by_candidate = (
            fitted_models_by_candidate
        ),
        estimand_records = [
            estimand_record
        ],
        condition_records = [
            first_condition_record,
            second_condition_record
        ],
        modeling_policy = modeling_policy
    )

    common_contrasts = [
        contrast_record
        for contrast_record in contrast_records
        if contrast_record[
            "candidate_identity"
        ] == common_candidate[
            "candidate_identity"
        ]
    ]

    differential_contrasts = [
        contrast_record
        for contrast_record in contrast_records
        if contrast_record[
            "candidate_identity"
        ] == differential_candidate[
            "candidate_identity"
        ]
    ]

    assert common_contrasts

    assert all(
        contrast_record[
            "contrast_status"
        ] == "completed"
        for contrast_record in common_contrasts
    )

    assert differential_contrasts

    assert all(
        contrast_record[
            "contrast_status"
        ] == "diagnostic_only"
        for contrast_record in (
            differential_contrasts
        )
    )

    assert any(
        contrast_record[
            "model_based_interpolation"
        ]
        is True
        for contrast_record in (
            differential_contrasts
        )
    )

    negative_coefficients = np.zeros_like(
        fitted_models_by_candidate[
            common_candidate[
                "candidate_identity"
            ]
        ][
            "coefficients"
        ]
    )

    negative_coefficients[0] = -1.0

    nonpositive_fitted_result = {
        **fitted_models_by_candidate[
            common_candidate[
                "candidate_identity"
            ]
        ],
        "coefficients": (
            negative_coefficients
        )
    }

    nonpositive_contrasts = (
        build_model_contrasts(
            prepared_models = {
                common_candidate[
                    "candidate_identity"
                ]: prepared_models[
                    common_candidate[
                        "candidate_identity"
                    ]
                ]
            },
            fitted_models_by_candidate = {
                common_candidate[
                    "candidate_identity"
                ]: nonpositive_fitted_result
            },
            estimand_records = [
                estimand_record
            ],
            condition_records = [
                first_condition_record,
                second_condition_record
            ],
            modeling_policy = modeling_policy
        )
    )

    assert nonpositive_contrasts

    assert all(
        contrast_record[
            "contrast_status"
        ] == "unavailable"
        for contrast_record in (
            nonpositive_contrasts
        )
    )

    assert all(
        contrast_record[
            "contrast_reason_code"
        ] == "NONPOSITIVE_RATIO_PREDICTION"
        for contrast_record in (
            nonpositive_contrasts
        )
    )

    disposition_counts = (
        assign_final_candidate_dispositions(
            candidate_records = candidate_records,
            modeling_policy = modeling_policy
        )
    )

    reconciliation = (
        validate_modeling_reconciliation(
            candidate_records = candidate_records,
            candidate_term_records = (
                candidate_term_records
            ),
            prepared_models = prepared_models,
            fitted_models_by_candidate = (
                fitted_models_by_candidate
            ),
            coefficient_records = (
                coefficient_records
            ),
            matrix_diagnostic_records = (
                matrix_diagnostic_records
            ),
            fit_diagnostic_records = (
                fit_diagnostic_records
            ),
            validation_metric_records = (
                validation_metric_records
            ),
            influence_records = (
                influence_records
            ),
            modeling_policy = modeling_policy
        )
    )

    assert sum(
        disposition_counts.values()
    ) == len(
        candidate_records
    )

    assert reconciliation[
        "candidate_count"
    ] == len(
        candidate_records
    )

    assert reconciliation[
        "fitted_candidate_count"
    ] == 2

    assert reconciliation[
        "reconciliation_complete"
    ] is True

    estimator_outcomes = {
        "weighted_least_squares": {
            "availability_status": (
                "available"
            )
        }
    }

    analytical_handoff = (
        build_analytical_handoff_summary(
            candidate_records = candidate_records,
            contrast_records = (
                contrast_records
            ),
            estimator_outcomes = (
                estimator_outcomes
            )
        )
    )

    assert sum(
        analytical_handoff[
            "candidate_analytical_status_counts"
        ].values()
    ) == len(
        candidate_records
    )

    assert sum(
        analytical_handoff[
            "contrast_status_counts"
        ].values()
    ) == len(
        contrast_records
    )

    assert analytical_handoff[
        "candidate_analytical_status_counts"
    ] == {
        "available": 1,
        "diagnostic_only": 1
    }

    assert analytical_handoff[
        "contrast_status_counts"
    ][
        "completed"
    ] == len(
        common_contrasts
    )

    assert analytical_handoff[
        "contrast_status_counts"
    ][
        "diagnostic_only"
    ] == len(
        differential_contrasts
    )

    assert analytical_handoff[
        "analytical_handoff_status"
    ] == "ready_with_candidate_restrictions"
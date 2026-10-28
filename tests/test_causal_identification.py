import copy
import hashlib
import json

from pathlib import Path

import pytest

from scripts.assess_causal_identification import (
    ASSUMPTION_OUTPUT_COLUMNS,
    ESTIMAND_IDENTITY_FIELDS,
    ESTIMAND_OUTPUT_COLUMNS,
    assess_estimand_record,
    build_assumption_inventory,
    build_estimand_decision_identity,
    build_table_manifest,
    canonical_json,
    load_json_object,
    parse_boolean,
    serialize_csv_value,
    validate_policy,
    validate_upstream_estimand_record,
    write_csv_table
)


# Building the reference test policy
def build_policy() -> dict:
    return {
        "policy_version": "1.5",
        "effective_date": "2026-10-19",
        "previous_policy_sha256": (
            "220b17c7a6ca963971a21765c87c5a544fc9c8ec7704e092cfe8ceda50c7a60f"
        ),
        "registration": {
            "status": "registered_before_bayesian_model_fitting",
            "descriptive_results_available_at_registration": True,
            "causal_model_results_available_at_registration": False,
            "outcome_values_used_for_identification_decisions": False,
            "predictive_performance_used_for_identification_decisions": False,
            "result_informed": False
        },
        "amendment": {
            "reason": "test",
            "result_informed": False,
            "affected_analyses": ["policy_validation"],
            "complete_reassessment_required": True
        },
        "independent_unit": "cell_identity",
        "decision_identity_fields": (
            ESTIMAND_IDENTITY_FIELDS.copy()
        ),
        "upstream_estimand_contract": {
            "aging_modes": [
                "calendar_aging",
                "cycle_aging"
            ],
            "decision_scopes": [
                "condition_protocol_pair"
            ],
            "outcomes": [
                "relative_capacity",
                "relative_dc_internal_resistance"
            ],
            "cohorts": [
                "primary",
                "sensitivity"
            ],
            "horizon_axes_by_aging_mode": {
                "calendar_aging": [
                    "elapsed_calendar_days"
                ],
                "cycle_aging": [
                    "elapsed_calendar_days",
                    "cumulative_full_equivalent_cycles_observed"
                ]
            },
            "support_classifications": [
                "unsupported",
                "descriptive_only",
                "supported_with_restrictions",
                "sensitivity_only"
            ],
            "causal_identification_statuses": [
                "not_established"
            ],
            "cross_stage_transportability_statuses": [
                "not_applicable"
            ],
            "sensitivity_only_permitted_cohorts": [
                "sensitivity"
            ]
        },
        "required_design_evidence": {
            "cell_to_condition_randomization_status": (
                "not_established_in_validated_inputs_or_primary_publication"
            ),
            "causal_interpretation": "not_established",
            "cross_stage_transportability_status": (
                "not_established_requires_model_dependent_joint_support_and_exchangeability_assessment"
            ),
            "calendar_net_storage_exposure_status": (
                "unavailable_not_authorized"
            ),
            "equipment_adjustment_status": (
                "not_authorized_from_aggregated_measurement_history"
            ),
            "baseline_adjustment_status": (
                "descriptive_only_model_adjustment_requires_explicit_model_specific_justification"
            )
        },
        "identification_requirements": {
            "assignment_mechanism_required": True,
            "conditional_exchangeability_required_for_covariate_adjustment": True,
            "conditional_positivity_required": True,
            "consistency_and_well_defined_treatment_versions_required": True,
            "no_interference_required": True,
            "selection_process_identification_required": True,
            "outcome_measurement_assumptions_required": True,
            "cross_stage_transportability_requires_explicit_assumptions": True,
            "absence_of_evidence_never_satisfies_an_assumption": True,
            "bayesian_estimation_alone_establishes_identification": False
        },
        "identification_strategies_assessed": {
            "overall_status": "no_supported_identification_strategy",
            "documented_random_assignment": "not_established",
            "pretreatment_covariate_adjustment": "not_authorized_no_sufficient_adjustment_set",
            "instrumental_variable": "not_established",
            "regression_discontinuity": "not_established",
            "difference_in_differences": "not_established",
            "other_natural_experiment": "not_established"
        },
        "estimand_interpretations": {
            "calendar_elapsed_time": {
                "horizon_axis": "elapsed_calendar_days",
                "authorized_target": "whole_assigned_protocol_outcome_difference_at_common_elapsed_calendar_time",
                "exposure_time_interpretation": "whole_assigned_protocol_gross_elapsed_calendar_time",
                "horizon_conditioning_status": "gross_elapsed_time_not_net_storage_exposure",
                "net_storage_effect_authorized": False,
                "interpretation": "test"
            },
            "cycle_elapsed_time": {
                "horizon_axis": "elapsed_calendar_days",
                "authorized_target": "outcome_difference_at_common_elapsed_calendar_days",
                "exposure_time_interpretation": "assigned_cycling_protocol_at_common_elapsed_calendar_time",
                "horizon_conditioning_status": "elapsed_time_axis",
                "interpretation": "test"
            },
            "cycle_fec": {
                "horizon_axis": "cumulative_full_equivalent_cycles_observed",
                "authorized_target": "outcome_difference_at_common_cumulative_full_equivalent_cycles_observed",
                "exposure_time_interpretation": "assigned_cycling_protocol_at_common_observed_fec",
                "horizon_conditioning_status": "post_assignment_achieved_exposure_axis",
                "interpretation": "test"
            }
        },
        "current_identification_status": {
            "assignment_mechanism": "not_established",
            "conditional_exchangeability": "not_established",
            "sufficient_pretreatment_adjustment_set": "not_authorized",
            "conditional_positivity": "not_established",
            "consistency": "not_established_for_causal_interventions",
            "no_interference": "not_established",
            "selection_process": "not_established",
            "outcome_measurement_error_process": "not_established_for_causal_identification",
            "cross_stage_transportability": "not_established",
            "calendar_net_storage_exposure": "unavailable",
            "causal_point_identification": "not_identified_from_observed_design"
        },
        "assumption_inventory": [
            {
                "assumption_name": "treatment_assignment",
                "assumption_scope": "all_estimands",
                "evidence_status": "not_established",
                "required_for_point_identification": True,
                "evidence_source": "test",
                "observed_evidence": "test",
                "analytical_implication": "test",
                "reason_code": "ASSIGNMENT_MECHANISM_NOT_ESTABLISHED"
            },
            {
                "assumption_name": "conditional_exchangeability",
                "assumption_scope": "all_estimands",
                "evidence_status": "not_established",
                "required_for_point_identification": True,
                "evidence_source": "test",
                "observed_evidence": "test",
                "analytical_implication": "test",
                "reason_code": "SUFFICIENT_PRETREATMENT_ADJUSTMENT_SET_NOT_AUTHORIZED"
            },
            {
                "assumption_name": "conditional_positivity",
                "assumption_scope": "all_estimands",
                "evidence_status": "not_established",
                "required_for_point_identification": True,
                "evidence_source": "test",
                "observed_evidence": "test",
                "analytical_implication": "test",
                "reason_code": "CONDITIONAL_POSITIVITY_NOT_ESTABLISHED"
            },
            {
                "assumption_name": "consistency_and_treatment_versions",
                "assumption_scope": "all_estimands",
                "evidence_status": (
                    "not_established_for_causal_interventions"
                ),
                "required_for_point_identification": True,
                "evidence_source": "test",
                "observed_evidence": "test",
                "analytical_implication": "test",
                "reason_code": "TREATMENT_VERSION_CONSISTENCY_NOT_ESTABLISHED"
            },
            {
                "assumption_name": "no_interference",
                "assumption_scope": "all_estimands",
                "evidence_status": "not_established",
                "required_for_point_identification": True,
                "evidence_source": "test",
                "observed_evidence": "test",
                "analytical_implication": "test",
                "reason_code": "NO_INTERFERENCE_NOT_ESTABLISHED"
            },
            {
                "assumption_name": "selection_and_outcome_availability",
                "assumption_scope": "all_estimands",
                "evidence_status": "not_established",
                "required_for_point_identification": True,
                "evidence_source": "test",
                "observed_evidence": "test",
                "analytical_implication": "test",
                "reason_code": "SELECTION_PROCESS_NOT_IDENTIFIED"
            },
            {
                "assumption_name": "outcome_measurement",
                "assumption_scope": "all_estimands",
                "evidence_status": (
                    "validated_feature_definitions_without_"
                    "causal_measurement_error_identification"
                ),
                "required_for_point_identification": True,
                "evidence_source": "test",
                "observed_evidence": "test",
                "analytical_implication": "test",
                "reason_code": "OUTCOME_MEASUREMENT_ERROR_PROCESS_NOT_ESTABLISHED"
            },
            {
                "assumption_name": "cross_stage_transportability",
                "assumption_scope": "cross_stage_estimands",
                "evidence_status": "not_established",
                "required_for_point_identification": True,
                "evidence_source": "test",
                "observed_evidence": "test",
                "analytical_implication": "test",
                "reason_code": "CROSS_STAGE_TRANSPORTABILITY_NOT_ESTABLISHED"
            },
            {
                "assumption_name": "achieved_fec_horizon",
                "assumption_scope": "cycle_fec_estimands",
                "evidence_status": "requires_explicit_causal_definition",
                "required_for_point_identification": True,
                "evidence_source": "test",
                "observed_evidence": "test",
                "analytical_implication": "test",
                "reason_code": "POST_ASSIGNMENT_FEC_HORIZON_NOT_IDENTIFIED"
            },
            {
                "assumption_name": "calendar_exposure_definition",
                "assumption_scope": "calendar_estimands",
                "evidence_status": "net_storage_exposure_unavailable",
                "required_for_point_identification": False,
                "evidence_source": "test",
                "observed_evidence": "test",
                "analytical_implication": "test",
                "reason_code": "NET_CALENDAR_AGING_EXPOSURE_UNAVAILABLE"
            }
        ],
        "model_authorization": {
            "associational_hierarchical_modeling": {
                "authorization_status": "partially_authorized",
                "authorized_estimand_status": "authorized_with_recorded_restrictions",
                "unauthorized_estimand_status": "not_authorized",
                "permitted_support_classifications": [
                    "supported_with_restrictions",
                    "sensitivity_only"
                ],
                "causal_language_permitted": False,
                "policy_language_permitted": False,
                "unobserved_condition_combinations_permitted": False,
                "cross_stage_pooling_permitted": False,
                "primary_and_sensitivity_cohorts_pooled": False
            },
            "causal_modeling": {
                "authorization_status": "not_authorized",
                "reason": (
                    "required_identification_assumptions_are_not_established"
                ),
                "bayesian_estimation_alone_establishes_identification": False
            },
            "causal_sensitivity_analysis": {
                "authorization_status": "not_authorized",
                "estimand_status": "not_authorized_without_registered_assumption_set",
                "activation_requirement": "test",
                "result_label": "test"
            }
        },
        "reason_codes": {
            "CAUSAL_IDENTIFICATION_NOT_ESTABLISHED_UPSTREAM": "test",
            "ASSIGNMENT_MECHANISM_NOT_ESTABLISHED": "test",
            "SUFFICIENT_PRETREATMENT_ADJUSTMENT_SET_NOT_AUTHORIZED": "test",
            "ALTERNATIVE_IDENTIFICATION_STRATEGY_NOT_ESTABLISHED": "test",
            "CONDITIONAL_POSITIVITY_NOT_ESTABLISHED": "test",
            "TREATMENT_VERSION_CONSISTENCY_NOT_ESTABLISHED": "test",
            "NO_INTERFERENCE_NOT_ESTABLISHED": "test",
            "SELECTION_PROCESS_NOT_IDENTIFIED": "test",
            "OUTCOME_MEASUREMENT_ERROR_PROCESS_NOT_ESTABLISHED": "test",
            "POST_ASSIGNMENT_FEC_HORIZON_NOT_IDENTIFIED": "test",
            "NET_CALENDAR_AGING_EXPOSURE_UNAVAILABLE": "test",
            "CROSS_STAGE_TRANSPORTABILITY_NOT_ESTABLISHED": "test",
            "RECORDED_EQUIPMENT_ALLOCATION_RESTRICTION": "test",
            "DESIGN_SUPPORT_NOT_AUTHORIZED": "test",
            "EXPLICIT_CAUSAL_SENSITIVITY_ASSUMPTIONS_NOT_REGISTERED": "test"
        },
        "reason_code_rules": {
            "universal_point_identification": [
                "CAUSAL_IDENTIFICATION_NOT_ESTABLISHED_UPSTREAM",
                "ASSIGNMENT_MECHANISM_NOT_ESTABLISHED",
                "SUFFICIENT_PRETREATMENT_ADJUSTMENT_SET_NOT_AUTHORIZED",
                "ALTERNATIVE_IDENTIFICATION_STRATEGY_NOT_ESTABLISHED",
                "CONDITIONAL_POSITIVITY_NOT_ESTABLISHED",
                "TREATMENT_VERSION_CONSISTENCY_NOT_ESTABLISHED",
                "NO_INTERFERENCE_NOT_ESTABLISHED",
                "SELECTION_PROCESS_NOT_IDENTIFIED",
                "OUTCOME_MEASUREMENT_ERROR_PROCESS_NOT_ESTABLISHED"
            ],
            "calendar_elapsed_time": {
                "causal_reason_codes": [],
                "interpretation_restriction_codes": [
                    "NET_CALENDAR_AGING_EXPOSURE_UNAVAILABLE"
                ]
            },
            "cycle_elapsed_time": {
                "causal_reason_codes": [],
                "interpretation_restriction_codes": []
            },
            "cycle_fec": {
                "causal_reason_codes": [
                    "POST_ASSIGNMENT_FEC_HORIZON_NOT_IDENTIFIED"
                ],
                "interpretation_restriction_codes": []
            },
            "cross_stage_comparison": {
                "causal_reason_codes": [
                    "CROSS_STAGE_TRANSPORTABILITY_NOT_ESTABLISHED"
                ],
                "interpretation_restriction_codes": []
            },
            "equipment_restriction": {
                "causal_reason_codes": [],
                "interpretation_restriction_codes": [
                    "RECORDED_EQUIPMENT_ALLOCATION_RESTRICTION"
                ]
            },
            "associational_model_not_authorized": {
                "causal_reason_codes": [],
                "interpretation_restriction_codes": [
                    "DESIGN_SUPPORT_NOT_AUTHORIZED"
                ]
            }
        },
        "output_contract": {
            "report_file": "causal_identification_report.json",
            "estimand_table": "estimand_identification.csv",
            "assumption_table": "assumption_inventory.csv",
            "deterministic_ordering_required": True,
            "input_hashes_required": True,
            "table_hashes_required": True,
            "exact_table_schemas_required": True,
            "embedded_policy_required": True,
            "post_publication_validation_required": True,
            "design_evidence_only_for_identification_decisions": True
        }
    }


# Building one representative estimand record
def build_estimand_record() -> dict:
    return {
        "aging_mode": "cycle_aging",
        "decision_scope": "condition_protocol_pair",
        "first_condition_protocol_identity": (
            "Stage_1/TP_z01/periodic_reference_tests"
        ),
        "second_condition_protocol_identity": (
            "Stage_1/TP_z02/periodic_reference_tests"
        ),
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": (
            "cumulative_full_equivalent_cycles_observed"
        ),
        "final_support_classification": (
            "supported_with_restrictions"
        ),
        "target_estimand_available": "true",
        "causal_identification_status": "not_established",
        "equipment_disjoint_dimension_count": "0",
        "equipment_identity_conflicted_dimension_count": "0",
        "calendar_claim_boundary": "",
        "target_estimand_identity": (
            "outcome_difference_at_common_cumulative_full_equivalent_cycles_observed"
        ),
        "condition_comparison_interpretation": (
            "joint_treatment_combination_contrast"
        ),
        "cross_stage_transportability_status": "not_applicable"
    }


# Confirming canonical JSON is order independent
def test_canonical_json_is_deterministic() -> None:
    first_value = {
        "b": 2,
        "a": 1
    }

    second_value = {
        "a": 1,
        "b": 2
    }

    assert (
        canonical_json(
            first_value
        )
        == canonical_json(
            second_value
        )
    )


# Confirming accepted Boolean representations
def test_parse_boolean_accepts_declared_representations() -> None:
    assert parse_boolean(
        True,
        "value"
    ) is True

    assert parse_boolean(
        "true",
        "value"
    ) is True

    assert parse_boolean(
        "false",
        "value"
    ) is False


# Confirming estimand identity is deterministic
def test_estimand_identity_is_deterministic() -> None:
    estimand_record = build_estimand_record()

    first_identity = build_estimand_decision_identity(
        estimand_record
    )

    reordered_record = {
        field_name: field_value
        for field_name, field_value in reversed(
            list(
                estimand_record.items()
            )
        )
    }

    second_identity = build_estimand_decision_identity(
        reordered_record
    )

    assert first_identity == second_identity
    assert len(
        first_identity
    ) == 64


# Confirming supported estimands remain associational
def test_supported_estimand_is_associational_only() -> None:
    assessment = assess_estimand_record(
        estimand_record = build_estimand_record(),
        policy = build_policy()
    )

    assert (
        assessment[
            "point_identification_status"
        ]
        == "not_identified_from_observed_design"
    )

    assert (
        assessment[
            "associational_model_status"
        ]
        == "authorized_with_recorded_restrictions"
    )

    assert (
        assessment[
            "causal_sensitivity_status"
        ]
        == "not_authorized_without_registered_assumption_set"
    )


# Confirming unsupported estimands are not authorized
def test_unsupported_estimand_is_not_authorized() -> None:
    estimand_record = build_estimand_record()

    estimand_record[
        "final_support_classification"
    ] = "unsupported"

    estimand_record[
        "target_estimand_available"
    ] = "false"

    assessment = assess_estimand_record(
        estimand_record = estimand_record,
        policy = build_policy()
    )

    assert (
        assessment[
            "associational_model_status"
        ]
        == "not_authorized"
    )

    assert (
        "DESIGN_SUPPORT_NOT_AUTHORIZED"
        in assessment[
            "interpretation_restriction_codes"
        ]
    )


# Confirming calendar exposure limits persist
def test_calendar_estimand_retains_exposure_limitation() -> None:
    estimand_record = build_estimand_record()

    estimand_record[
        "aging_mode"
    ] = "calendar_aging"

    estimand_record[
        "horizon_axis"
    ] = "elapsed_calendar_days"

    estimand_record[
        "target_estimand_identity"
    ] = (
        "whole_assigned_protocol_outcome_difference_at_common_elapsed_calendar_time"
    )

    assessment = assess_estimand_record(
        estimand_record = estimand_record,
        policy = build_policy()
    )

    assert (
        "NET_CALENDAR_AGING_EXPOSURE_UNAVAILABLE"
        in assessment[
            "interpretation_restriction_codes"
        ]
    )

    assert (
        assessment[
            "exposure_time_interpretation"
        ]
        == "whole_assigned_protocol_gross_elapsed_calendar_time"
    )


# Confirming cross-stage comparisons are not transportable
def test_cross_stage_comparison_is_not_transportable() -> None:
    estimand_record = build_estimand_record()

    estimand_record[
        "second_condition_protocol_identity"
    ] = (
        "Stage_2/TP_z02/periodic_reference_tests"
    )

    assessment = assess_estimand_record(
        estimand_record = estimand_record,
        policy = build_policy()
    )

    assert assessment[
        "cross_stage_comparison"
    ] is True

    assert (
        "CROSS_STAGE_TRANSPORTABILITY_NOT_ESTABLISHED"
        in assessment[
            "causal_identification_reason_codes"
        ]
    )

    assert (
        assessment[
            "associational_model_status"
        ]
        == "not_authorized"
    )

    assert (
        "DESIGN_SUPPORT_NOT_AUTHORIZED"
        in assessment[
            "interpretation_restriction_codes"
        ]
    )


# Confirming recorded equipment restrictions remain visible
def test_equipment_restriction_remains_explicit() -> None:
    estimand_record = build_estimand_record()

    estimand_record[
        "equipment_disjoint_dimension_count"
    ] = "2"

    assessment = assess_estimand_record(
        estimand_record = estimand_record,
        policy = build_policy()
    )

    assert assessment[
        "equipment_restriction_present"
    ] is True

    assert (
        "RECORDED_EQUIPMENT_ALLOCATION_RESTRICTION"
        in assessment[
            "interpretation_restriction_codes"
        ]
    )


# Confirming unsupported causal authorization is rejected
def test_policy_rejects_causal_authorization() -> None:
    policy = build_policy()

    validate_policy(
        policy
    )

    invalid_policy = copy.deepcopy(
        policy
    )

    invalid_policy[
        "model_authorization"
    ][
        "causal_modeling"
    ][
        "authorization_status"
    ] = "authorized"

    with pytest.raises(
        RuntimeError
    ):
        validate_policy(
            invalid_policy
        )


# Confirming the assumption inventory is complete
def test_assumption_inventory_records_identification_limits() -> None:
    policy = build_policy()

    assumption_rows = build_assumption_inventory(
        policy
    )

    assumption_names = {
        assumption_row[
            "assumption_name"
        ]
        for assumption_row in assumption_rows
    }

    assert assumption_names == {
        "treatment_assignment",
        "conditional_exchangeability",
        "conditional_positivity",
        "consistency_and_treatment_versions",
        "no_interference",
        "selection_and_outcome_availability",
        "outcome_measurement",
        "achieved_fec_horizon",
        "calendar_exposure_definition",
        "cross_stage_transportability"
    }

    assert all(
        assumption_row[
            "analytical_implication"
        ]
        for assumption_row in assumption_rows
    )


# Confirming assessment output is byte stable on repeat
def test_estimand_assessment_is_repeatable() -> None:
    policy = build_policy()

    first_assessment = assess_estimand_record(
        estimand_record = build_estimand_record(),
        policy = policy
    )

    second_assessment = assess_estimand_record(
        estimand_record = build_estimand_record(),
        policy = policy
    )

    assert (
        canonical_json(
            first_assessment
        )
        == canonical_json(
            second_assessment
        )
    )


# Confirming the CSV writer produces byte identical output on repeat
def test_csv_writer_is_byte_stable(tmp_path: Path) -> None:
    first_path = tmp_path / "first.csv"

    second_path = tmp_path / "second.csv"

    assumption_rows = build_assumption_inventory(
        build_policy()
    )

    write_csv_table(
        path = first_path,
        columns = ASSUMPTION_OUTPUT_COLUMNS,
        rows = assumption_rows
    )

    write_csv_table(
        path = second_path,
        columns = ASSUMPTION_OUTPUT_COLUMNS,
        rows = assumption_rows
    )

    first_sha256 = hashlib.sha256(
        first_path.read_bytes()
    ).hexdigest()

    second_sha256 = hashlib.sha256(
        second_path.read_bytes()
    ).hexdigest()

    assert first_sha256 == second_sha256

    first_manifest = build_table_manifest(
        path = first_path,
        columns = ASSUMPTION_OUTPUT_COLUMNS,
        row_count = len(
            assumption_rows
        )
    )

    assert first_manifest[
        "sha256"
    ] == first_sha256


# Confirming CSV value serialization is deterministic
def test_csv_value_serialization_is_deterministic() -> None:
    assert serialize_csv_value(
        None
    ) == ""

    assert serialize_csv_value(
        True
    ) == "true"

    assert serialize_csv_value(
        False
    ) == "false"

    assert serialize_csv_value(
        [
            "b",
            "a"
        ]
    ) == '["b","a"]'

    assert serialize_csv_value(
        {
            "b": 2,
            "a": 1
        }
    ) == '{"a":1,"b":2}'


# Confirming calendar net-exposure limitations restrict interpretation without redefining the estimand
def test_calendar_net_exposure_is_an_interpretation_restriction() -> None:
    estimand_record = build_estimand_record()

    estimand_record[
        "aging_mode"
    ] = "calendar_aging"

    estimand_record[
        "horizon_axis"
    ] = "elapsed_calendar_days"

    estimand_record[
        "target_estimand_identity"
    ] = (
        "whole_assigned_protocol_outcome_difference_at_common_elapsed_calendar_time"
    )

    estimand_record[
        "condition_comparison_interpretation"
    ] = "one_factor_structural_contrast"

    assessment = assess_estimand_record(
        estimand_record = estimand_record,
        policy = build_policy()
    )

    assert (
        "NET_CALENDAR_AGING_EXPOSURE_UNAVAILABLE"
        not in assessment[
            "causal_identification_reason_codes"
        ]
    )

    assert (
        "NET_CALENDAR_AGING_EXPOSURE_UNAVAILABLE"
        in assessment[
            "interpretation_restriction_codes"
        ]
    )


# Confirming achieved FEC is not assigned an automatic causal interpretation
def test_achieved_fec_requires_explicit_causal_interpretation() -> None:
    estimand_record = build_estimand_record()

    estimand_record[
        "target_estimand_identity"
    ] = (
        "outcome_difference_at_common_cumulative_full_equivalent_cycles_observed"
    )

    estimand_record[
        "condition_comparison_interpretation"
    ] = "joint_treatment_combination_contrast"

    assessment = assess_estimand_record(
        estimand_record = estimand_record,
        policy = build_policy()
    )

    assert assessment[
        "horizon_conditioning_status"
    ] == "post_assignment_achieved_exposure_axis"

    assert (
        "POST_ASSIGNMENT_FEC_HORIZON_NOT_IDENTIFIED"
        in assessment[
            "causal_identification_reason_codes"
        ]
    )


# Confirming every emitted reason code is registered
def test_estimand_assessment_uses_registered_reason_codes() -> None:
    policy = build_policy()

    assessment = assess_estimand_record(
        estimand_record = build_estimand_record(),
        policy = policy
    )

    emitted_reason_codes = {
        *assessment[
            "causal_identification_reason_codes"
        ],
        *assessment[
            "interpretation_restriction_codes"
        ]
    }

    assert emitted_reason_codes.issubset(
        policy[
            "reason_codes"
        ]
    )


# Confirming descriptive outcomes cannot alter identification
def test_policy_rejects_outcome_informed_identification() -> None:
    policy = build_policy()

    policy[
        "registration"
    ][
        "outcome_values_used_for_identification_decisions"
    ] = True

    with pytest.raises(
        RuntimeError
    ):
        validate_policy(
            policy
        )


def test_policy_rejects_result_informed_identification() -> None:
    policy = build_policy()

    policy["registration"]["result_informed"] = True

    with pytest.raises(RuntimeError):
        validate_policy(policy)


def test_policy_rejects_duplicate_assumptions() -> None:
    policy = build_policy()

    policy["assumption_inventory"].append(
        copy.deepcopy(
            policy["assumption_inventory"][0]
        )
    )

    with pytest.raises(RuntimeError):
        validate_policy(policy)


def test_policy_rejects_unregistered_rule_reason() -> None:
    policy = build_policy()

    policy["reason_code_rules"][
        "universal_point_identification"
    ].append(
        "UNKNOWN_REASON"
    )

    with pytest.raises(RuntimeError):
        validate_policy(policy)


def test_reason_assignment_is_policy_driven() -> None:
    policy = build_policy()

    policy["reason_code_rules"][
        "universal_point_identification"
    ] = [
        "ASSIGNMENT_MECHANISM_NOT_ESTABLISHED"
    ]

    assessment = assess_estimand_record(
        estimand_record = build_estimand_record(),
        policy = policy
    )

    assert assessment[
        "causal_identification_reason_codes"
    ] == [
        "ASSIGNMENT_MECHANISM_NOT_ESTABLISHED",
        "POST_ASSIGNMENT_FEC_HORIZON_NOT_IDENTIFIED"
    ]


# Rejecting a non-Boolean identification requirement
def test_policy_rejects_invalid_identification_requirement() -> None:
    policy = build_policy()

    policy["identification_requirements"][
        "assignment_mechanism_required"
    ] = "yes"

    with pytest.raises(RuntimeError):
        validate_policy(policy)


# Rejecting an identification strategy claiming a supported overall status
def test_policy_rejects_invalid_strategy_status() -> None:
    policy = build_policy()

    policy["identification_strategies_assessed"][
        "overall_status"
    ] = "supported_identification_strategy"

    with pytest.raises(RuntimeError):
        validate_policy(policy)


# Rejecting a cycle elapsed-time target that differs from the authoritative identity
def test_policy_rejects_mismatched_target_identity() -> None:
    policy = build_policy()

    policy["estimand_interpretations"][
        "cycle_elapsed_time"
    ][
        "authorized_target"
    ] = "outcome_difference_at_common_elapsed_calendar_time"

    with pytest.raises(RuntimeError):
        validate_policy(policy)


# Rejecting any causal sensitivity analysis authorization
def test_policy_rejects_unauthorized_causal_sensitivity() -> None:
    policy = build_policy()

    policy["model_authorization"][
        "causal_sensitivity_analysis"
    ][
        "authorization_status"
    ] = "authorized"

    with pytest.raises(RuntimeError):
        validate_policy(policy)


# Rejecting any associational authorization that enables causal language
def test_policy_rejects_associational_authorization_flag() -> None:
    policy = build_policy()

    policy["model_authorization"][
        "associational_hierarchical_modeling"
    ][
        "causal_language_permitted"
    ] = True

    with pytest.raises(RuntimeError):
        validate_policy(policy)


# Confirming the authorized estimand status is read from the policy
def test_assessment_reads_associational_status_from_policy() -> None:
    policy = build_policy()

    policy["model_authorization"][
        "associational_hierarchical_modeling"
    ][
        "authorized_estimand_status"
    ] = "custom_authorized_status"

    assessment = assess_estimand_record(
        estimand_record = build_estimand_record(),
        policy = policy
    )

    assert assessment[
        "associational_model_status"
    ] == "custom_authorized_status"


# Rejecting an estimand whose target identity differs from the registered interpretation
def test_assessment_rejects_mismatched_target_identity() -> None:
    policy = build_policy()

    estimand_record = build_estimand_record()

    estimand_record[
        "target_estimand_identity"
    ] = "unexpected_target"

    with pytest.raises(RuntimeError):
        assess_estimand_record(
            estimand_record = estimand_record,
            policy = policy
        )


# Rejecting a negative equipment-dimension count
def test_assessment_rejects_negative_equipment_count() -> None:
    policy = build_policy()

    estimand_record = build_estimand_record()

    estimand_record[
        "equipment_disjoint_dimension_count"
    ] = "-1"

    with pytest.raises(RuntimeError):
        assess_estimand_record(
            estimand_record = estimand_record,
            policy = policy
        )


# Rejecting a table manifest that omits its exact schema
def test_verify_report_table_rejects_missing_schema(tmp_path: Path) -> None:
    table_path = tmp_path / "table.csv"

    table_path.write_text(
        "a,b\n1,2\n",
        encoding = "utf-8"
    )

    from scripts.assess_causal_identification import (
        verify_report_table
    )

    report = {
        "tables": {
            "demo": {
                "file_name": "table.csv",
                "sha256": hashlib.sha256(
                    table_path.read_bytes()
                ).hexdigest(),
                "row_count": 1
            }
        }
    }

    with pytest.raises(RuntimeError):
        verify_report_table(
            report_path = tmp_path / "report.json",
            report = report,
            table_name = "demo"
        )


# Rejecting a requirement that conflicts with its assumption record
def test_policy_rejects_requirement_assumption_conflict() -> None:
    policy = build_policy()

    for assumption_record in policy[
        "assumption_inventory"
    ]:
        if (
            assumption_record[
                "assumption_name"
            ]
            == "cross_stage_transportability"
        ):
            assumption_record[
                "required_for_point_identification"
            ] = False

            break

    with pytest.raises(
        RuntimeError
    ):
        validate_policy(
            policy
        )


# Rejecting unexpected policy fields
def test_policy_rejects_unexpected_top_level_field() -> None:
    policy = build_policy()

    policy[
        "unexpected_field"
    ] = True

    with pytest.raises(
        RuntimeError
    ):
        validate_policy(
            policy
        )


# Rejecting invalid version provenance
def test_policy_rejects_invalid_previous_digest() -> None:
    policy = build_policy()

    policy[
        "previous_policy_sha256"
    ] = "invalid"

    with pytest.raises(
        RuntimeError
    ):
        validate_policy(
            policy
        )


# Building one valid report directory for mutation tests
def build_report_fixture(tmp_path: Path) -> tuple[Path, dict, dict, list[dict], list[dict]]:
    from scripts.assess_causal_identification import (
        build_table_manifest,
        write_csv_table
    )

    policy = build_policy()

    estimand_rows = [
        assess_estimand_record(
            estimand_record = build_estimand_record(),
            policy = policy
        )
    ]

    assumption_rows = build_assumption_inventory(
        policy
    )

    estimand_path = tmp_path / "estimand_identification.csv"
    assumption_path = tmp_path / "assumption_inventory.csv"

    write_csv_table(
        path = estimand_path,
        columns = ESTIMAND_OUTPUT_COLUMNS,
        rows = estimand_rows
    )

    write_csv_table(
        path = assumption_path,
        columns = ASSUMPTION_OUTPUT_COLUMNS,
        rows = assumption_rows
    )

    report = {
        "assessment_status": "completed",
        "policy_version": policy["policy_version"],
        "policy_effective_date": policy["effective_date"],
        "policy_sha256": "a" * 64,
        "policy": policy,
        "design_report_sha256": "b" * 64,
        "independent_unit": "cell_identity",
        "identification_decision_basis": (
            "registered_policy_and_validated_design_evidence_only"
        ),
        "outcome_values_used_for_identification_decisions": False,
        "predictive_performance_used_for_identification_decisions": False,
        "causal_modeling_status": "not_authorized",
        "associational_hierarchical_modeling_status": (
            "partially_authorized"
        ),
        "causal_sensitivity_analysis_status": "not_authorized",
        "causal_sensitivity_activation_requirement": (
            policy["model_authorization"]["causal_sensitivity_analysis"]["activation_requirement"]
        ),
        "estimand_count": len(estimand_rows),
        "assumption_count": len(assumption_rows),
        "point_identification_status_counts": {
            "not_identified_from_observed_design": 1
        },
        "associational_model_status_counts": {
            "authorized_with_recorded_restrictions": 1
        },
        "tables": {
            "assumption_inventory": build_table_manifest(
                path = assumption_path,
                columns = ASSUMPTION_OUTPUT_COLUMNS,
                row_count = len(assumption_rows)
            ),
            "estimand_identification": build_table_manifest(
                path = estimand_path,
                columns = ESTIMAND_OUTPUT_COLUMNS,
                row_count = len(estimand_rows)
            )
        }
    }

    report_path = tmp_path / "causal_identification_report.json"

    report_path.write_text(
        json.dumps(report, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8"
    )

    return (report_path, report, policy, estimand_rows, assumption_rows)


# Rejecting a report with an unexpected field
def test_validate_report_rejects_unexpected_field(tmp_path: Path) -> None:
    from scripts.assess_causal_identification import (
        validate_identification_report
    )

    (
        report_path,
        report,
        policy,
        estimand_rows,
        assumption_rows
    ) = build_report_fixture(tmp_path)

    report["unexpected_field"] = True

    report_path.write_text(
        json.dumps(report, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8"
    )

    with pytest.raises(RuntimeError):
        validate_identification_report(
            report_path = report_path,
            expected_policy = policy,
            expected_policy_sha256 = "a" * 64,
            expected_design_report_sha256 = "b" * 64,
            expected_estimand_rows = estimand_rows,
            expected_assumption_rows = assumption_rows
        )


# Rejecting a report with a mutated independent unit
def test_validate_report_rejects_mutated_independent_unit(tmp_path: Path) -> None:
    from scripts.assess_causal_identification import (
        validate_identification_report
    )

    (
        report_path,
        report,
        policy,
        estimand_rows,
        assumption_rows
    ) = build_report_fixture(tmp_path)

    report["independent_unit"] = "observation"

    report_path.write_text(
        json.dumps(report, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8"
    )

    with pytest.raises(RuntimeError):
        validate_identification_report(
            report_path = report_path,
            expected_policy = policy,
            expected_policy_sha256 = "a" * 64,
            expected_design_report_sha256 = "b" * 64,
            expected_estimand_rows = estimand_rows,
            expected_assumption_rows = assumption_rows
        )


# Rejecting a report that declares outcome-informed decisions
def test_validate_report_rejects_outcome_informed_declaration(tmp_path: Path) -> None:
    from scripts.assess_causal_identification import (
        validate_identification_report
    )

    (
        report_path,
        report,
        policy,
        estimand_rows,
        assumption_rows
    ) = build_report_fixture(tmp_path)

    report["outcome_values_used_for_identification_decisions"] = True

    report_path.write_text(
        json.dumps(report, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8"
    )

    with pytest.raises(RuntimeError):
        validate_identification_report(
            report_path = report_path,
            expected_policy = policy,
            expected_policy_sha256 = "a" * 64,
            expected_design_report_sha256 = "b" * 64,
            expected_estimand_rows = estimand_rows,
            expected_assumption_rows = assumption_rows
        )


# Confirming that achieved FEC remains a causal-identification requirement
def test_policy_requires_causal_definition_for_achieved_fec() -> None:
    policy = build_policy()

    achieved_fec_assumption = next(
        assumption_record
        for assumption_record in policy[
            "assumption_inventory"
        ]
        if assumption_record[
            "assumption_name"
        ] == "achieved_fec_horizon"
    )

    achieved_fec_assumption[
        "required_for_point_identification"
    ] = False

    with pytest.raises(
        RuntimeError,
        match = "Scope-specific point-identification requirement"
    ):
        validate_policy(
            policy
        )


# Confirming that net storage exposure is an interpretation restriction
def test_policy_does_not_require_net_storage_exposure_for_whole_protocol_estimand() -> None:
    policy = build_policy()

    calendar_assumption = next(
        assumption_record
        for assumption_record in policy[
            "assumption_inventory"
        ]
        if assumption_record[
            "assumption_name"
        ] == "calendar_exposure_definition"
    )

    calendar_assumption[
        "required_for_point_identification"
    ] = True

    with pytest.raises(
        RuntimeError,
        match = "Scope-specific point-identification requirement"
    ):
        validate_policy(
            policy
        )


# Confirming that identification amendments remain outcome independent
def test_policy_rejects_result_informed_amendment() -> None:
    policy = build_policy()

    policy[
        "amendment"
    ][
        "result_informed"
    ] = True

    with pytest.raises(
        RuntimeError,
        match = "Identification-policy amendment is invalid"
    ):
        validate_policy(
            policy
        )


# Rejecting incomplete design-evidence declarations
def test_policy_rejects_incomplete_design_evidence() -> None:
    policy = build_policy()

    del policy[
        "required_design_evidence"
    ][
        "equipment_adjustment_status"
    ]

    with pytest.raises(
        RuntimeError,
        match = "Required design evidence"
    ):
        validate_policy(
            policy
        )


# Rejecting contradictory current-identification declarations
def test_policy_rejects_established_assignment_mechanism() -> None:
    policy = build_policy()

    policy[
        "current_identification_status"
    ][
        "assignment_mechanism"
    ] = "established"

    with pytest.raises(
        RuntimeError,
        match = "internally inconsistent"
    ):
        validate_policy(
            policy
        )


# Rejecting contradictory identification strategies
def test_policy_rejects_established_instrument() -> None:
    policy = build_policy()

    policy[
        "identification_strategies_assessed"
    ][
        "instrumental_variable"
    ] = "established"

    with pytest.raises(
        RuntimeError,
        match = "Identification-strategy"
    ):
        validate_policy(
            policy
        )


# Rejecting missing universal identification reasons
def test_policy_rejects_incomplete_universal_reasons() -> None:
    policy = build_policy()

    policy[
        "reason_code_rules"
    ][
        "universal_point_identification"
    ] = []

    with pytest.raises(
        RuntimeError,
        match = "Universal identification reasons"
    ):
        validate_policy(
            policy
        )


# Rejecting omission of the achieved-FEC limitation
def test_policy_rejects_missing_fec_identification_reason() -> None:
    policy = build_policy()

    policy[
        "reason_code_rules"
    ][
        "cycle_fec"
    ][
        "causal_reason_codes"
    ] = []

    with pytest.raises(
        RuntimeError,
        match = "cycle_fec"
    ):
        validate_policy(
            policy
        )


# Rejecting unexpected model-authorization content
def test_policy_rejects_unexpected_model_authorization() -> None:
    policy = build_policy()

    policy[
        "model_authorization"
    ][
        "unexpected"
    ] = {}

    with pytest.raises(
        RuntimeError,
        match = "Model authorization"
    ):
        validate_policy(
            policy
        )


# Rejecting an unexplained causal non-authorization
def test_policy_rejects_empty_causal_authorization_reason() -> None:
    policy = build_policy()

    policy[
        "model_authorization"
    ][
        "causal_modeling"
    ][
        "reason"
    ] = ""

    with pytest.raises(
        RuntimeError,
        match = "Causal-model authorization"
    ):
        validate_policy(
            policy
        )


# Rejecting duplicate fields in a policy document
def test_strict_json_loader_rejects_duplicate_policy_field(tmp_path: Path) -> None:
    policy_path = (
        tmp_path
        / "policy.json"
    )

    policy_path.write_text(
        (
            "{\n"
            '  "policy_version": "1.5",\n'
            '  "policy_version": "1.6"\n'
            "}\n"
        ),
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = "duplicate field"
    ):
        load_json_object(
            policy_path
        )


# Rejecting duplicate fields in a completed report
def test_strict_json_loader_rejects_duplicate_report_field(tmp_path: Path) -> None:
    report_path = (
        tmp_path
        / "report.json"
    )

    report_path.write_text(
        (
            "{\n"
            '  "assessment_status": "completed",\n'
            '  "assessment_status": "failed"\n'
            "}\n"
        ),
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = "duplicate field"
    ):
        load_json_object(
            report_path
        )


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value"
    ),
    [
        (
            "aging_mode",
            "unknown_aging_mode"
        ),
        (
            "decision_scope",
            "unknown_scope"
        ),
        (
            "outcome",
            "unknown_outcome"
        ),
        (
            "cohort",
            "unknown_cohort"
        ),
        (
            "final_support_classification",
            "unknown_support"
        ),
        (
            "causal_identification_status",
            "unknown_identification"
        ),
        (
            "cross_stage_transportability_status",
            "unknown_transportability"
        )
    ]
)
def test_upstream_estimand_contract_rejects_unknown_values(field_name: str, invalid_value: str) -> None:
    estimand_record = build_estimand_record()

    estimand_record[
        field_name
    ] = invalid_value

    with pytest.raises(
        RuntimeError,
        match = "unsupported categorical value"
    ):
        validate_upstream_estimand_record(
            estimand_record = estimand_record,
            policy = build_policy()
        )


# Rejecting an invalid aging-mode and horizon-axis combination
def test_upstream_estimand_contract_rejects_calendar_fec_axis() -> None:
    estimand_record = build_estimand_record()

    estimand_record[
        "aging_mode"
    ] = "calendar_aging"

    estimand_record[
        "horizon_axis"
    ] = (
        "cumulative_full_equivalent_cycles_observed"
    )

    with pytest.raises(
        RuntimeError,
        match = "aging-mode and horizon-axis"
    ):
        validate_upstream_estimand_record(
            estimand_record = estimand_record,
            policy = build_policy()
        )


# Rejecting sensitivity-only support in the primary cohort
def test_upstream_estimand_contract_rejects_primary_sensitivity_only() -> None:
    estimand_record = build_estimand_record()

    estimand_record[
        "final_support_classification"
    ] = "sensitivity_only"

    estimand_record[
        "cohort"
    ] = "primary"

    with pytest.raises(
        RuntimeError,
        match = "unsupported cohort"
    ):
        validate_upstream_estimand_record(
            estimand_record = estimand_record,
            policy = build_policy()
        )
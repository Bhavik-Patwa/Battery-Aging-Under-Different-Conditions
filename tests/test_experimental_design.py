import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_experimental_design import (
    summarize_common_horizon_support,
    classify_estimand_support,
    measure_categorical_dependence,
    parse_nullable_boolean,
    build_treatment_combination_identity,
    positive_feature_available,
    record_has_positive_horizon,
    build_test_point_design,
    build_design_matrix,
    build_stage_overlap,
    build_equipment_inventory,
    build_analytical_test_point_design,
    build_analytical_cell_design,
    build_context_adjusted_design_matrix,
    build_population_identifiability,
    build_estimand_support,
    build_design_findings,
    OUTCOME_DEFINITIONS,
    CONTEXT_FIELDS,
    DESIGN_PROVENANCE_FIELDS,
    build_context_families,
    CONTEXT_FAMILY_NONE,
    CONTEXT_FAMILY_LABORATORY,
    build_continuous_support_regions,
    apply_final_estimand_classifications,
    build_analytical_scope_table,
    jointly_compact_support_regions,
    validate_cohort_aligned_support_regions
)


# Building one valid analysis-policy payload for deterministic tests
def build_valid_policy_payload() -> dict:
    return {
        "policy_version": "1.4",
        "support_mode": (
            "within_cell_bracketing_ranges"
        ),
        "coverage_rules": {
            "primary": {
                "minimum_assigned_cell_coverage_fraction": 1.0,
                "rationale": "test"
            },
            "sensitivity": {
                "minimum_assigned_cell_coverage_fraction": 2 / 3,
                "rationale": "test"
            }
        },
        "horizon_definitions": [],
        "calendar_exposure_policy": {
            "primary_time_axis": "elapsed_calendar_days",
            "primary_estimand": (
                "whole_assigned_protocol_outcome_difference_at_common_elapsed_calendar_time"
            ),
            "primary_estimand_availability": "available",
            "primary_model_authorization": (
                "permitted_with_restrictions"
            ),
            "net_storage_exposure_estimand": (
                "outcome_difference_at_common_net_calendar_aging_time"
            ),
            "net_storage_exposure_availability": (
                "unavailable"
            ),
            "net_storage_exposure_model_authorization": (
                "not_authorized"
            ),
            "claim_boundary": (
                "Gross elapsed calendar days support whole-protocol comparisons only. They do not "
                "identify time under storage, isolate calendar-aging exposure, or exclude time spent in reference "
                "tests and documented pauses."
            ),
            "rationale": "test"
        },
        "longitudinal_observation_policy": {
            "support_horizon_requirement": "strictly_positive",
            "baseline_horizon_role": (
                "excluded_from_support_determination_and_primary_longitudinal_response"
            ),
            "model_specific_baseline_use": (
                "requires_explicit_normalization_aware_justification"
            ),
            "source_observation_rule": (
                "all_eligible_positive_horizon_observations_from_support_region_member_cells"
            ),
            "support_interval_role": (
                "prediction_and_estimand_evaluation_domain"
            ),
            "observations_outside_support_interval": (
                "permitted_only_for_support_region_member_cells_to_establish_within_cell_bracketing"
            ),
            "estimand_evaluation_mode": (
                "continuous_trajectory_over_supported_ranges"
            ),
            "discrete_horizon_status": (
                "not_authorized_without_outcome_blind_pre_specification"
            ),
            "point_contrast_requirement": (
                "outcome_blind_scientific_or_protocol_pre_specification_before_comparative_results_are_examined"
            ),
            "rationale": "test"
        },
        "treatment_evaluation_policy": {
            "authorized_domain": (
                "observed_condition_protocol_combinations_only"
            ),
            "unobserved_joint_treatment_combinations": (
                "not_authorized_by_design_audit"
            ),
            "continuous_treatment_interpolation": (
                "requires_separate_model_specific_validation"
            ),
            "one_factor_contrast_structure": (
                "requires_exactly_one_changed_treatment_and_no_disjoint_or_identity_conflicted_equipment_allocation"
            ),
            "one_factor_contrast_identification": (
                "not_established_assignment_mechanism_undocumented"
            ),
            "joint_condition_comparison_support": (
                "permitted_with_restrictions_when_replication_horizon_support_and_context_requirements_are_satisfied"
            ),
            "rationale": "test"
        }
    }


# Confirming nullable eligibility remains three-valued
def test_parse_nullable_boolean() -> None:
    assert parse_nullable_boolean(
        "true"
    ) is True

    assert parse_nullable_boolean(
        "false"
    ) is False

    assert parse_nullable_boolean(
        ""
    ) is None

    with pytest.raises(
        RuntimeError,
        match = "invalid boolean value"
    ):
        parse_nullable_boolean(
            "unknown"
        )


# Summarizing exact and bracketing horizon support
def test_summarize_common_horizon_support() -> None:
    support = summarize_common_horizon_support(
        horizon_values_by_cell = {
            "cell_1": [
                10.0,
                20.0,
                30.0
            ],
            "cell_2": [
                10.0,
                25.0,
                30.0
            ],
            "cell_3": [
                10.0,
                30.0
            ]
        },
        assigned_cell_identities = {
            "cell_1",
            "cell_2",
            "cell_3"
        },
        candidate_horizons = [
            10.0,
            20.0,
            30.0
        ],
        minimum_assigned_cell_coverage_fraction = 0.5
    )

    assert support[
        "assigned_cell_count"
    ] == 3

    assert support[
        "eligible_cell_count"
    ] == 3

    assert support[
        "observation_count"
    ] == 8

    assert support[
        "observed_horizon_minimum"
    ] == pytest.approx(
        10.0
    )

    assert support[
        "observed_horizon_maximum"
    ] == pytest.approx(
        30.0
    )

    assert support[
        "complete_eligible_cell_bracketing_minimum"
    ] == pytest.approx(
        10.0
    )

    assert support[
        "complete_eligible_cell_bracketing_maximum"
    ] == pytest.approx(
        30.0
    )

    assert support[
        "exact_common_horizons"
    ] == [
        10.0,
        30.0
    ]

    assert support[
        "coverage_eligible_observed_horizons"
    ] == [
        10.0,
        20.0,
        30.0
    ]


# Rejecting horizon support when cell ranges do not overlap
def test_summarize_common_horizon_support_records_no_overlap() -> None:
    support = summarize_common_horizon_support(
        horizon_values_by_cell = {
            "cell_1": [
                10.0,
                20.0
            ],
            "cell_2": [
                30.0,
                40.0
            ]
        },
        assigned_cell_identities = {
            "cell_1",
            "cell_2"
        },
        candidate_horizons = [
            10.0,
            20.0,
            30.0,
            40.0
        ],
        minimum_assigned_cell_coverage_fraction = 1.0
    )

    assert support[
        "observed_horizon_minimum"
    ] == pytest.approx(
        10.0
    )

    assert support[
        "observed_horizon_maximum"
    ] == pytest.approx(
        40.0
    )

    assert support[
        "complete_eligible_cell_bracketing_minimum"
    ] is None

    assert support[
        "complete_eligible_cell_bracketing_maximum"
    ] is None

    assert support[
        "exact_common_horizons"
    ] == []

    assert support[
        "coverage_eligible_observed_horizons"
    ] == []


# Detecting perfect structural dependence
def test_measure_categorical_dependence() -> None:
    dependence = measure_categorical_dependence(
        records = [
            {
                "laboratory": "A",
                "stage": "1"
            },
            {
                "laboratory": "A",
                "stage": "1"
            },
            {
                "laboratory": "B",
                "stage": "2"
            }
        ],
        first_field = "laboratory",
        second_field = "stage"
    )

    assert dependence[
        "first_determines_second"
    ] is True

    assert dependence[
        "second_determines_first"
    ] is True

    assert dependence[
        "first_to_second_purity"
    ] == pytest.approx(
        1.0
    )


# Confirming a replicated one-factor analytical contrast
def test_classify_estimand_support_records_restricted_support() -> None:
    assert classify_estimand_support(
        changed_treatment_count = 1,
        context_difference_count = 0,
        equipment_disjoint_dimension_count = 0,
        equipment_identity_conflicted_dimension_count = 0,
        minimum_eligible_cell_count = 3,
        observed_horizon_overlap_available = True,
        common_support_interval_available = True,
        target_estimand_available = True
    ) == (
        "supported_with_restrictions",
        "replicated_one_factor_analytical_contrast_with_restrictions"
    )


# Confirming multiply changed comparisons return a joint-condition contrast
def test_classify_estimand_support_returns_joint_condition_contrast() -> None:
    assert classify_estimand_support(
        changed_treatment_count = 2,
        context_difference_count = 0,
        equipment_disjoint_dimension_count = 0,
        equipment_identity_conflicted_dimension_count = 0,
        minimum_eligible_cell_count = 3,
        observed_horizon_overlap_available = True,
        common_support_interval_available = True,
        target_estimand_available = True
    ) == (
        "supported_with_restrictions",
        "replicated_joint_condition_analytical_contrast_with_restrictions"
    )


# Blocking comparisons without observed horizon overlap
def test_classify_estimand_support_requires_observed_horizon_overlap() -> None:
    assert classify_estimand_support(
        changed_treatment_count = 1,
        context_difference_count = 0,
        equipment_disjoint_dimension_count = 0,
        equipment_identity_conflicted_dimension_count = 0,
        minimum_eligible_cell_count = 3,
        observed_horizon_overlap_available = False,
        common_support_interval_available = False,
        target_estimand_available = True
    ) == (
        "unsupported",
        "no_observed_condition_level_horizon_overlap"
    )


# Blocking comparisons without a threshold-qualifying horizon
def test_classify_estimand_support_requires_common_support_interval() -> None:
    assert classify_estimand_support(
        changed_treatment_count = 1,
        context_difference_count = 0,
        equipment_disjoint_dimension_count = 0,
        equipment_identity_conflicted_dimension_count = 0,
        minimum_eligible_cell_count = 3,
        observed_horizon_overlap_available = True,
        common_support_interval_available = False,
        target_estimand_available = True
    ) == (
        "unsupported",
        "no_common_assigned_cell_support_interval"
    )


# Preventing one-cell groups from being treated as replicated
def test_classify_estimand_support_requires_replication() -> None:
    assert classify_estimand_support(
        changed_treatment_count = 1,
        context_difference_count = 0,
        equipment_disjoint_dimension_count = 0,
        equipment_identity_conflicted_dimension_count = 0,
        minimum_eligible_cell_count = 1,
        observed_horizon_overlap_available = True,
        common_support_interval_available = True,
        target_estimand_available = True
    ) == (
        "descriptive_only",
        "unreplicated_observed_contrast"
    )


# Confirming conflicted equipment metadata restricts but does not veto support
def test_classify_estimand_support_records_equipment_conflict() -> None:
    assert classify_estimand_support(
        changed_treatment_count = 1,
        context_difference_count = 0,
        equipment_disjoint_dimension_count = 0,
        equipment_identity_conflicted_dimension_count = 1,
        minimum_eligible_cell_count = 3,
        observed_horizon_overlap_available = True,
        common_support_interval_available = True,
        target_estimand_available = True
    ) == (
        "supported_with_restrictions",
        "replicated_one_factor_analytical_contrast_with_equipment_allocation_restrictions"
    )


# Confirming disjoint equipment allocation restricts but does not veto support
def test_classify_estimand_support_records_equipment_aliasing() -> None:
    assert classify_estimand_support(
        changed_treatment_count = 1,
        context_difference_count = 0,
        equipment_disjoint_dimension_count = 1,
        equipment_identity_conflicted_dimension_count = 0,
        minimum_eligible_cell_count = 3,
        observed_horizon_overlap_available = True,
        common_support_interval_available = True,
        target_estimand_available = True
    ) == (
        "supported_with_restrictions",
        "replicated_one_factor_analytical_contrast_with_equipment_allocation_restrictions"
    )


# Confirming equipment disjointness restricts but does not veto support
def test_equipment_disjointness_is_a_restriction_not_support_veto() -> None:
    status, detail = classify_estimand_support(
        changed_treatment_count = 2,
        context_difference_count = 0,
        equipment_disjoint_dimension_count = 3,
        equipment_identity_conflicted_dimension_count = 0,
        minimum_eligible_cell_count = 3,
        observed_horizon_overlap_available = True,
        common_support_interval_available = True,
        target_estimand_available = True
    )

    assert status == "supported_with_restrictions"

    assert detail == (
        "replicated_joint_condition_analytical_contrast_with_equipment_allocation_restrictions"
    )


# Confirming a design-context difference stays descriptive-only
def test_design_context_difference_remains_descriptive_only() -> None:
    status, detail = classify_estimand_support(
        changed_treatment_count = 1,
        context_difference_count = 1,
        equipment_disjoint_dimension_count = 0,
        equipment_identity_conflicted_dimension_count = 0,
        minimum_eligible_cell_count = 3,
        observed_horizon_overlap_available = True,
        common_support_interval_available = True,
        target_estimand_available = True
    )

    assert status == "descriptive_only"

    assert detail == (
        "replicated_contrast_with_structural_design_context_difference"
    )


# Distinguishing positive exposure from the zero-FEC baseline
def test_positive_feature_available() -> None:
    assert positive_feature_available(
        "1.0"
    ) is True

    assert positive_feature_available(
        "0.0"
    ) is False

    assert positive_feature_available(
        ""
    ) is False


# Building readable deterministic treatment identities
def test_build_treatment_combination_identity() -> None:
    record = {
        "aging_mode": "calendar_aging",
        "ambient_temperature_celsius": "23.0",
        "maximum_state_of_charge_fraction": "0.8",
        "reference_test_schedule": (
            "periodic_reference_tests"
        )
    }

    assert build_treatment_combination_identity(
        record
    ) == (
        "calendar_aging/"
        "ambient_temperature_celsius=23.0/"
        "maximum_state_of_charge_fraction=0.8/"
        "reference_test_schedule=periodic_reference_tests"
    )


# Excluding normalized baselines from degradation support
def test_record_has_positive_horizon_excludes_baseline() -> None:
    record = {
        "elapsed_calendar_days": "0",
        "elapsed_calendar_time_status": "completed",
        "cumulative_full_equivalent_cycles_observed": "0.0",
        "cumulative_fec_observation_complete": "true"
    }

    assert record_has_positive_horizon(
        record
    ) is False


# Collapsing replicated cells to one design row per test point
def test_build_test_point_design_avoids_replication_weighting() -> None:
    records = []

    for cell_number in range(
        1,
        4
    ):
        records.append(
            {
                "cell_identity": (
                    f"Stage_1/TP_z01_0{cell_number}"
                ),
                "condition_protocol_identity": (
                    "Stage_1/TP_z01/"
                    "periodic_reference_tests"
                ),
                "dataset_stage": "Stage_1",
                "aging_mode": "cycle_aging",
                "aging_type": "z",
                "test_point_number": "1",
                "cell_number": str(
                    cell_number
                ),
                "laboratory": "HM",
                "sampling_method": "cLH",
                "ambient_temperature_celsius": "45.0",
                "maximum_state_of_charge_fraction": "0.8",
                "depth_of_discharge_fraction": "0.6",
                "charge_rate_c": "1.0",
                "discharge_rate_c": "2.0",
                "reference_test_schedule": (
                    "periodic_reference_tests"
                )
            }
        )

    design_records = build_test_point_design(
        records
    )

    assert len(design_records) == 1

    assert design_records[0][
        "independent_cell_count"
    ] == 3


# Detecting structurally aliased numerical treatments
def test_build_design_matrix_detects_rank_failure() -> None:
    records = [
        {
            "dataset_stage": "Stage_1",
            "ambient_temperature_celsius": value,
            "maximum_state_of_charge_fraction": value
        }
        for value in [
            "0.0",
            "1.0",
            "2.0"
        ]
    ]

    design_matrix, _ = build_design_matrix(
        records = records,
        aging_mode = "calendar_aging",
        design_scope = "Stage_1",
        specification = "main_effects"
    )

    assert np.linalg.matrix_rank(
        design_matrix
    ) < design_matrix.shape[1]


# Confirming exact cross-stage treatment matches produce zero distance
def test_build_stage_overlap_detects_exact_match() -> None:
    test_point_records = [
        {
            "test_point_identity": "Stage_1/TP_z01",
            "aging_mode": "cycle_aging",
            "dataset_stage": "Stage_1",
            "ambient_temperature_celsius": "45.0",
            "maximum_state_of_charge_fraction": "0.8",
            "depth_of_discharge_fraction": "0.6",
            "charge_rate_c": "1.0",
            "discharge_rate_c": "2.0"
        },
        {
            "test_point_identity": "Stage_2/TP_z01",
            "aging_mode": "cycle_aging",
            "dataset_stage": "Stage_2",
            "ambient_temperature_celsius": "45.0",
            "maximum_state_of_charge_fraction": "0.8",
            "depth_of_discharge_fraction": "0.6",
            "charge_rate_c": "1.0",
            "discharge_rate_c": "2.0"
        }
    ]

    overlap_records = build_stage_overlap(
        test_point_records
    )

    nearest_records = [
        record
        for record in overlap_records
        if record["record_type"] == "nearest_test_point"
    ]

    assert len(nearest_records) == 2

    assert all(
        record["range_scaled_nearest_distance"]
        == pytest.approx(0.0)
        for record in nearest_records
    )


# Confirming non-overlapping stage ranges produce positive distance
def test_build_stage_overlap_detects_non_overlap() -> None:
    test_point_records = [
        {
            "test_point_identity": "Stage_1/TP_z01",
            "aging_mode": "cycle_aging",
            "dataset_stage": "Stage_1",
            "ambient_temperature_celsius": "10.0",
            "maximum_state_of_charge_fraction": "0.8",
            "depth_of_discharge_fraction": "0.6",
            "charge_rate_c": "1.0",
            "discharge_rate_c": "2.0"
        },
        {
            "test_point_identity": "Stage_2/TP_z01",
            "aging_mode": "cycle_aging",
            "dataset_stage": "Stage_2",
            "ambient_temperature_celsius": "45.0",
            "maximum_state_of_charge_fraction": "0.8",
            "depth_of_discharge_fraction": "0.6",
            "charge_rate_c": "1.0",
            "discharge_rate_c": "2.0"
        }
    ]

    overlap_records = build_stage_overlap(
        test_point_records
    )

    nearest_records = [
        record
        for record in overlap_records
        if record["record_type"] == "nearest_test_point"
    ]

    assert all(
        record["range_scaled_nearest_distance"]
        > 0
        for record in nearest_records
    )


# Confirming cycle exposure equipment enters only its own role
def test_build_equipment_inventory_separates_cycle_exposure() -> None:
    audit_result = {
        "archives": [
            {
                "measurements": [
                    {
                        "source_identity": (
                            "Stage_1/TP_z02_01/"
                            "TP_z02_01_02_ZYK.csv"
                        ),
                        "companion_metadata": {
                            "measurement_device": (
                                "cycling_device"
                            ),
                            "test_channel": "CH_CYCLE",
                            "climate_chamber": (
                                "cycle_chamber"
                            ),
                            "climate_chamber_position": (
                                "cycle_position"
                            )
                        },
                        "quality_reasons": []
                    }
                ]
            }
        ]
    }

    equipment_by_cell = build_equipment_inventory(
        audit_result
    )

    cell_equipment = equipment_by_cell[
        "Stage_1/TP_z02_01"
    ]

    assert cell_equipment[
        "cycle_aging_exposure"
    ][
        "validated"
    ][
        "measurement_devices"
    ] == {"cycling_device"}

    assert cell_equipment[
        "reference_measurement"
    ][
        "validated"
    ][
        "measurement_devices"
    ] == set()


# Confirming reference equipment never enters cycle exposure
def test_build_equipment_inventory_separates_reference_measurement() -> None:
    audit_result = {
        "archives": [
            {
                "measurements": [
                    {
                        "source_identity": (
                            "Stage_1/TP_z02_01/"
                            "TP_z02_01_01_ET_T23.csv"
                        ),
                        "companion_metadata": {
                            "measurement_device": (
                                "reference_device"
                            ),
                            "test_channel": "CH_REF",
                            "climate_chamber": (
                                "reference_chamber"
                            ),
                            "climate_chamber_position": (
                                "reference_position"
                            )
                        },
                        "quality_reasons": []
                    }
                ]
            }
        ]
    }

    equipment_by_cell = build_equipment_inventory(
        audit_result
    )

    cell_equipment = equipment_by_cell[
        "Stage_1/TP_z02_01"
    ]

    assert cell_equipment[
        "reference_measurement"
    ][
        "validated"
    ][
        "measurement_devices"
    ] == {"reference_device"}

    assert cell_equipment[
        "cycle_aging_exposure"
    ][
        "validated"
    ][
        "measurement_devices"
    ] == set()


# Detecting rank loss after analytical eligibility is applied
def test_analytical_test_point_design_detects_realized_rank_loss() -> None:
    test_point_records = [
        {
            "test_point_identity": (
                f"Stage_1/TP_k{test_point_number:02d}"
            ),
            "dataset_stage": "Stage_1",
            "aging_mode": "calendar_aging",
            "ambient_temperature_celsius": temperature,
            "maximum_state_of_charge_fraction": (
                state_of_charge
            ),
            "independent_cell_count": 1
        }
        for (
            test_point_number,
            temperature,
            state_of_charge
        ) in [
            (
                1,
                "10.0",
                "0.2"
            ),
            (
                2,
                "23.0",
                "0.8"
            ),
            (
                3,
                "45.0",
                "0.5"
            )
        ]
    ]

    cell_records = [
        {
            "cell_identity": (
                f"Stage_1/TP_k{test_point_number:02d}_01"
            ),
            "dataset_stage": "Stage_1",
            "aging_type": "k",
            "aging_mode": "calendar_aging",
            "test_point_number": str(
                test_point_number
            )
        }
        for test_point_number in range(
            1,
            4
        )
    ]

    reference_records = [
        {
            "cell_identity": (
                f"Stage_1/TP_k{test_point_number:02d}_01"
            ),
            "relative_capacity": "0.98",
            "relative_capacity_primary_eligible": (
                "true"
                if test_point_number < 3
                else "false"
            ),
            "elapsed_calendar_time_status": (
                "completed"
            ),
            "elapsed_calendar_days": "30"
        }
        for test_point_number in range(
            1,
            4
        )
    ]

    analytical_records = (
        build_analytical_test_point_design(
            test_point_records = test_point_records,
            cell_records = cell_records,
            reference_records = reference_records,
            aging_mode = "calendar_aging",
            outcome_definition = (
                OUTCOME_DEFINITIONS[
                    "relative_capacity"
                ]
            ),
            cohort_name = "primary",
            horizon_axis = "elapsed_calendar_days"
        )
    )

    design_matrix, _ = build_design_matrix(
        records = analytical_records,
        aging_mode = "calendar_aging",
        design_scope = "Stage_1",
        specification = "main_effects"
    )

    assert len(analytical_records) == 2

    assert np.linalg.matrix_rank(
        design_matrix
    ) < design_matrix.shape[1]


# Confirming that sampling methodology remains design provenance
def test_sampling_method_is_not_outcome_generating_context() -> None:
    assert "sampling_method" not in CONTEXT_FIELDS
    assert "sampling_method" in DESIGN_PROVENANCE_FIELDS


# Summarizing actual design findings without outcome-value selection
def test_build_design_findings() -> None:
    findings = build_design_findings(
        estimand_records = [
            {
                "aging_mode": "calendar_aging",
                "final_support_classification": (
                    "supported_with_restrictions"
                ),
                "direct_contrast_support_detail": (
                    "replicated_one_factor_descriptive_contrast"
                )
            }
        ],
        identifiability_records = [
            {
                "record_type": "matrix_summary",
                "analysis_population": (
                    "assigned_test_point_design"
                ),
                "aging_mode": "calendar_aging",
                "design_scope": "Stage_1",
                "outcome": None,
                "cohort": None,
                "horizon_axis": None,
                "specification": "main_effects",
                "context_family": CONTEXT_FAMILY_NONE,
                "test_point_count": 11,
                "column_count": 3,
                "matrix_rank": 3,
                "full_column_rank": True,
                "condition_number": 2.0
            },
            {
                "record_type": "term",
                "analysis_population": (
                    "assigned_test_point_design"
                ),
                "aging_mode": "calendar_aging",
                "design_scope": "Stage_1",
                "outcome": None,
                "cohort": None,
                "horizon_axis": None,
                "specification": "main_effects",
                "matrix_scope": (
                    "standardized_treatments_only"
                ),
                "context_family": CONTEXT_FAMILY_NONE,
                "term_name": (
                    "ambient_temperature_celsius"
                ),
                "term_type": "main_effect",
                "structurally_estimable": True
            },
            {
                "record_type": "term",
                "analysis_population": (
                    "assigned_test_point_design"
                ),
                "aging_mode": "calendar_aging",
                "design_scope": "Stage_1",
                "outcome": None,
                "cohort": None,
                "horizon_axis": None,
                "specification": "main_effects",
                "matrix_scope": (
                    "standardized_treatments_only"
                ),
                "context_family": CONTEXT_FAMILY_NONE,
                "term_name": (
                    "ambient_temperature_celsius"
                ),
                "term_type": "main_effect",
                "structurally_estimable": True
            }
        ],
        correlation_records = [
            {
                "aging_mode": "calendar_aging",
                "design_scope": "Stage_1",
                "first_treatment": (
                    "ambient_temperature_celsius"
                ),
                "second_treatment": (
                    "maximum_state_of_charge_fraction"
                ),
                "pearson_correlation": 0.25,
                "correlation_status": "completed"
            }
        ],
        stage_overlap_records = [
            {
                "record_type": (
                    "exact_combination_summary"
                ),
                "aging_mode": "calendar_aging",
                "overlap_basis": "numerical_treatments",
                "exact_overlap_count": 3
            }
        ]
    )

    assert findings[
        "support_counts_by_mode"
    ][0][
        "decision_count"
    ] == 1

    assert findings[
        "matrix_summaries"
    ][0][
        "full_column_rank"
    ] is True

    assert findings[
        "term_estimability_aggregates"
    ] == [
        {
            "analysis_population": "assigned_test_point_design",
            "aging_mode": "calendar_aging",
            "design_scope": "Stage_1",
            "outcome": "",
            "cohort": "",
            "horizon_axis": "",
            "specification": "main_effects",
            "context_family": CONTEXT_FAMILY_NONE,
            "structurally_estimable": True,
            "term_count": 2
        }
    ]

    assert findings[
        "exact_stage_overlap"
    ][0][
        "exact_overlap_count"
    ] == 3


# Confirming cell-level analytical design contains one row per eligible cell
def test_build_analytical_cell_design_produces_one_row_per_cell() -> None:
    cell_records = [
        {
            "cell_identity": f"Stage_1/TP_z01_0{cell_number}",
            "dataset_stage": "Stage_1",
            "aging_type": "z",
            "aging_mode": "cycle_aging",
            "test_point_number": "1"
        }
        for cell_number in range(1, 4)
    ]

    reference_records = [
        {
            "cell_identity": f"Stage_1/TP_z01_0{cell_number}",
            "relative_capacity": "0.98",
            "relative_capacity_primary_eligible": "true",
            "elapsed_calendar_time_status": "completed",
            "elapsed_calendar_days": "30"
        }
        for cell_number in range(1, 4)
    ]

    analytical_records = build_analytical_cell_design(
        cell_records = cell_records,
        reference_records = reference_records,
        aging_mode = "cycle_aging",
        outcome_definition = OUTCOME_DEFINITIONS["relative_capacity"],
        cohort_name = "primary",
        horizon_axis = "elapsed_calendar_days"
    )

    assert len(analytical_records) == 3

    assert all(
        record["analytical_independent_cell_count"] == 1
        for record in analytical_records
    )


# Confirming context-adjusted calendar matrices contain a schedule term
def test_context_adjusted_calendar_matrix_contains_schedule_term() -> None:
    records = [
        {
            "cell_identity": f"Stage_1/TP_k01_0{cell_number}",
            "dataset_stage": "Stage_1",
            "aging_mode": "calendar_aging",
            "ambient_temperature_celsius": "23.0",
            "maximum_state_of_charge_fraction": "0.5",
            "reference_test_schedule": (
                "periodic_reference_tests"
                if cell_number <= 3
                else "initial_and_final_only"
            ),
            "laboratory": "SIE"
        }
        for cell_number in range(1, 7)
    ]

    _, matrix_terms = build_context_adjusted_design_matrix(
        records = records,
        aging_mode = "calendar_aging",
        design_scope = "Stage_1",
        specification = "main_effects",
        equipment_by_cell = {},
        context_family = CONTEXT_FAMILY_NONE
    )

    assert any(
        term["term_name"].startswith(
            "reference_test_schedule="
        )
        for term in matrix_terms
    )


# Confirming two cells from one test point produce two context matrix rows
def test_context_adjusted_matrix_uses_one_row_per_cell() -> None:
    records = [
        {
            "cell_identity": f"Stage_1/TP_k01_0{cell_number}",
            "dataset_stage": "Stage_1",
            "aging_mode": "calendar_aging",
            "ambient_temperature_celsius": "23.0",
            "maximum_state_of_charge_fraction": "0.5",
            "reference_test_schedule": "periodic_reference_tests",
            "laboratory": "SIE"
        }
        for cell_number in range(1, 4)
    ]

    design_matrix, _ = build_context_adjusted_design_matrix(
        records = records,
        aging_mode = "calendar_aging",
        design_scope = "Stage_1",
        specification = "main_effects",
        equipment_by_cell = {},
        context_family = CONTEXT_FAMILY_NONE
    )

    assert design_matrix.shape[0] == 3


# Confirming equipment allocation relationships are retained in estimand output
def test_build_estimand_support_retains_equipment_relationships() -> None:
    condition_records = [
        {
            "condition_protocol_identity": f"Stage_1/TP_k01/{schedule}",
            "aging_mode": "calendar_aging",
            "dataset_stage": "Stage_1",
            "laboratory": "SIE",
            "sampling_method": "FF",
            "ambient_temperature_celsius": "23.0",
            "maximum_state_of_charge_fraction": "0.5",
            "reference_test_schedule": schedule
        }
        for schedule in [
            "periodic_reference_tests",
            "initial_and_final_only"
        ]
    ]

    treatment_support_records = [
        {
            "condition_protocol_identity": (
                f"Stage_1/TP_k01/{schedule}"
            ),
            "reference_measurement_measurement_devices": [
                "device_a"
            ],
            "reference_measurement_measurement_devices_cell_counts": {
                "device_a": 3
            },
            "reference_measurement_identity_conflicted_measurement_devices": [],
            "reference_measurement_identity_conflicted_measurement_devices_cell_counts": {},
            "reference_measurement_test_channels": ["ch_a"],
            "reference_measurement_test_channels_cell_counts": {"ch_a": 3},
            "reference_measurement_identity_conflicted_test_channels": [],
            "reference_measurement_identity_conflicted_test_channels_cell_counts": {},
            "reference_measurement_climate_chambers": ["chamber_a"],
            "reference_measurement_climate_chambers_cell_counts": {"chamber_a": 3},
            "reference_measurement_identity_conflicted_climate_chambers": [],
            "reference_measurement_identity_conflicted_climate_chambers_cell_counts": {},
            "reference_measurement_climate_chamber_positions": ["pos_a"],
            "reference_measurement_climate_chamber_positions_cell_counts": {"pos_a": 3},
            "reference_measurement_identity_conflicted_climate_chamber_positions": [],
            "reference_measurement_identity_conflicted_climate_chamber_positions_cell_counts": {}
        }
        for schedule in [
            "periodic_reference_tests",
            "initial_and_final_only"
        ]
    ]

    horizon_support_records = [
        {
            "condition_protocol_identity": (
                f"Stage_1/TP_k01/{schedule}"
            ),
            "aging_mode": "calendar_aging",
            "outcome": outcome_name,
            "cohort": cohort_name,
            "horizon_axis": "elapsed_calendar_days",
            "assigned_cell_count": 3,
            "eligible_cell_count": 3,
            "eligible_assigned_cell_fraction": 1.0,
            "observed_horizon_minimum": 10.0,
            "observed_horizon_maximum": 300.0,
            "complete_eligible_cell_bracketing_minimum": 10.0,
            "complete_eligible_cell_bracketing_maximum": 300.0,
            "exact_common_horizons": [10.0],
            "coverage_eligible_observed_horizons": [10.0],
            "required_support_cell_count": 2,
            "supported_bracketing_ranges": [
                {
                    "minimum": 10.0,
                    "maximum": 300.0,
                    "required_support_cell_count": 2,
                    "assigned_cell_count": 3,
                    "minimum_assigned_cell_coverage_fraction": 0.5,
                    "support_interpretation": (
                        "within_observed_cell_ranges_without_extrapolation"
                    )
                }
            ]
        }
        for schedule in [
            "periodic_reference_tests",
            "initial_and_final_only"
        ]
        for outcome_name in OUTCOME_DEFINITIONS
        for cohort_name in ["primary", "sensitivity"]
    ]

    estimand_records = build_estimand_support(
        condition_records = condition_records,
        treatment_support_records = treatment_support_records,
        horizon_support_records = horizon_support_records,
        coverage_rules = {
            "primary": {
                "minimum_assigned_cell_coverage_fraction": 0.5,
                "rationale": "test"
            },
            "sensitivity": {
                "minimum_assigned_cell_coverage_fraction": 0.5,
                "rationale": "test"
            }
        },
        analysis_policy = {
            "calendar_exposure_policy": {
                "primary_estimand_availability": "available",
                "primary_estimand": (
                    "whole_assigned_protocol_outcome_difference_at_common_elapsed_calendar_time"
                ),
                "net_storage_exposure_availability": (
                    "unavailable"
                ),
                "net_storage_exposure_model_authorization": (
                    "not_authorized"
                ),
                "claim_boundary": (
                    "Gross elapsed calendar days support whole-protocol comparisons only. They do "
                    "not identify time under storage, isolate calendar-aging exposure, or exclude time "
                    "spent in reference tests and documented pauses."
                )
            }
        }
    )

    assert estimand_records

    assert all(
        "equipment_allocation_relationships" in record
        for record in estimand_records
    )


# Confirming conflicted equipment does not enter validated equipment signatures
def test_build_equipment_signature_rejects_conflicted_equipment() -> None:
    from scripts.audit_experimental_design import build_equipment_signature

    equipment_by_cell = {
        "Stage_1/TP_z01_01": {
            "reference_measurement": {
                "validated": {
                    "measurement_devices": set()
                },
                "identity_conflicted": {
                    "measurement_devices": {"device_x"}
                }
            }
        }
    }

    signature = build_equipment_signature(
        cell_identities = ["Stage_1/TP_z01_01"],
        equipment_by_cell = equipment_by_cell,
        equipment_role = "reference_measurement",
        output_field = "measurement_devices"
    )

    assert signature == "__identity_conflicted__"


# Confirming that calendar context families exclude cycle-exposure equipment
def test_build_context_families_excludes_cycle_equipment_for_calendar() -> None:
    context_families = build_context_families(
        "calendar_aging"
    )

    assert CONTEXT_FAMILY_NONE in context_families
    assert CONTEXT_FAMILY_LABORATORY in context_families

    assert not any(
        context_family.startswith(
            "cycle_aging_exposure_"
        )
        for context_family in context_families
    )


# Confirming that cycle context families include cycle-exposure equipment
def test_build_context_families_includes_cycle_equipment_for_cycle() -> None:
    context_families = build_context_families(
        "cycle_aging"
    )

    cycle_exposure_families = [
        context_family
        for context_family in context_families
        if context_family.startswith(
            "cycle_aging_exposure_"
        )
    ]

    assert len(cycle_exposure_families) == 4

    assert set(
        cycle_exposure_families
    ) == {
        "cycle_aging_exposure_measurement_devices",
        "cycle_aging_exposure_test_channels",
        "cycle_aging_exposure_climate_chambers",
        "cycle_aging_exposure_climate_chamber_positions"
    }


# Confirming that no-additional-context retains schedule without equipment
def test_context_family_none_contains_schedule_only() -> None:
    records = [
        {
            "cell_identity": f"Stage_1/TP_k01_0{cell_number}",
            "dataset_stage": "Stage_1",
            "aging_mode": "calendar_aging",
            "ambient_temperature_celsius": "23.0",
            "maximum_state_of_charge_fraction": "0.5",
            "reference_test_schedule": (
                "periodic_reference_tests"
                if cell_number <= 3
                else "initial_and_final_only"
            ),
            "laboratory": "SIE"
        }
        for cell_number in range(1, 7)
    ]

    _, matrix_terms = build_context_adjusted_design_matrix(
        records = records,
        aging_mode = "calendar_aging",
        design_scope = "Stage_1",
        specification = "main_effects",
        equipment_by_cell = {},
        context_family = CONTEXT_FAMILY_NONE
    )

    term_names = [
        term["term_name"]
        for term in matrix_terms
    ]

    assert any(
        term_name.startswith(
            "reference_test_schedule="
        )
        for term_name in term_names
    )

    assert not any(
        term_name.startswith("laboratory=")
        for term_name in term_names
    )

    assert not any(
        term_name.startswith(
            "reference_measurement_"
        )
        for term_name in term_names
    )


# Confirming that the laboratory family adds only laboratory indicators
def test_context_family_laboratory_adds_only_laboratory() -> None:
    records = [
        {
            "cell_identity": f"Stage_1/TP_k01_0{cell_number}",
            "dataset_stage": "Stage_1",
            "aging_mode": "calendar_aging",
            "ambient_temperature_celsius": "23.0",
            "maximum_state_of_charge_fraction": "0.5",
            "reference_test_schedule": "periodic_reference_tests",
            "laboratory": (
                "SIE"
                if cell_number <= 3
                else "HM"
            )
        }
        for cell_number in range(1, 7)
    ]

    _, matrix_terms = build_context_adjusted_design_matrix(
        records = records,
        aging_mode = "calendar_aging",
        design_scope = "Stage_1",
        specification = "main_effects",
        equipment_by_cell = {},
        context_family = CONTEXT_FAMILY_LABORATORY
    )

    term_names = [
        term["term_name"]
        for term in matrix_terms
    ]

    assert any(
        term_name.startswith("laboratory=")
        for term_name in term_names
    )

    assert not any(
        term_name.startswith(
            "reference_measurement_"
        )
        for term_name in term_names
    )


# Confirming that a single equipment family adds only its own indicators
def test_context_family_adds_one_equipment_family_only() -> None:
    records = [
        {
            "cell_identity": f"Stage_1/TP_z01_0{cell_number}",
            "dataset_stage": "Stage_1",
            "aging_mode": "cycle_aging",
            "ambient_temperature_celsius": "23.0",
            "maximum_state_of_charge_fraction": "0.5",
            "depth_of_discharge_fraction": "0.5",
            "charge_rate_c": "1.0",
            "discharge_rate_c": "1.0",
            "reference_test_schedule": "periodic_reference_tests",
            "laboratory": "SIE"
        }
        for cell_number in range(1, 7)
    ]

    equipment_by_cell = {
        f"Stage_1/TP_z01_0{cell_number}": {
            "reference_measurement": {
                "validated": {
                    "measurement_devices": {
                        f"device_{cell_number % 2}"
                    },
                    "test_channels": {"ch_a"},
                    "climate_chambers": {"chamber_a"},
                    "climate_chamber_positions": {"pos_a"}
                },
                "identity_conflicted": {
                    "measurement_devices": set(),
                    "test_channels": set(),
                    "climate_chambers": set(),
                    "climate_chamber_positions": set()
                }
            },
            "cycle_aging_exposure": {
                "validated": {
                    "measurement_devices": set(),
                    "test_channels": set(),
                    "climate_chambers": set(),
                    "climate_chamber_positions": set()
                },
                "identity_conflicted": {
                    "measurement_devices": set(),
                    "test_channels": set(),
                    "climate_chambers": set(),
                    "climate_chamber_positions": set()
                }
            }
        }
        for cell_number in range(1, 7)
    }

    _, matrix_terms = build_context_adjusted_design_matrix(
        records = records,
        aging_mode = "cycle_aging",
        design_scope = "Stage_1",
        specification = "main_effects",
        equipment_by_cell = equipment_by_cell,
        context_family = (
            "reference_measurement_measurement_devices"
        )
    )

    term_names = [
        term["term_name"]
        for term in matrix_terms
    ]

    assert any(
        term_name.startswith(
            "reference_measurement_measurement_devices="
        )
        for term_name in term_names
    )

    assert not any(
        term_name.startswith(
            "reference_measurement_test_channels="
        )
        for term_name in term_names
    )

    assert not any(
        term_name.startswith(
            "reference_measurement_climate_chambers="
        )
        for term_name in term_names
    )

    assert not any(
        term_name.startswith(
            "cycle_aging_exposure_"
        )
        for term_name in term_names
    )


# Confirming that target term estimability is not governed by full-matrix rank alone
def test_target_term_estimability_is_evaluated_independently() -> None:
    records = [
        {
            "cell_identity": f"Stage_1/TP_k01_0{cell_number}",
            "dataset_stage": "Stage_1",
            "aging_mode": "calendar_aging",
            "ambient_temperature_celsius": temperature,
            "maximum_state_of_charge_fraction": "0.5",
            "reference_test_schedule": schedule,
            "laboratory": "SIE"
        }
        for (cell_number, temperature, schedule) in [
            (1, "10.0", "periodic_reference_tests"),
            (2, "23.0", "periodic_reference_tests"),
            (3, "45.0", "periodic_reference_tests"),
            (4, "10.0", "initial_and_final_only"),
            (5, "23.0", "initial_and_final_only"),
            (6, "45.0", "initial_and_final_only")
        ]
    ]

    design_matrix, matrix_terms = (
        build_context_adjusted_design_matrix(
            records = records,
            aging_mode = "calendar_aging",
            design_scope = "Stage_1",
            specification = "main_effects",
            equipment_by_cell = {},
            context_family = CONTEXT_FAMILY_NONE
        )
    )

    matrix_rank = int(np.linalg.matrix_rank(design_matrix))
    assert matrix_rank < design_matrix.shape[1]

    schedule_term_index = next(
        term_index
        for term_index, term in enumerate(matrix_terms)
        if term["term_name"].startswith("reference_test_schedule=")
    )

    reduced_matrix = np.delete(design_matrix, schedule_term_index, axis=1)
    reduced_rank = int(np.linalg.matrix_rank(reduced_matrix))
    assert matrix_rank > reduced_rank


# Confirming that nuisance-context terms are excluded from the report
def test_nonestimable_terms_exclude_nuisance_context() -> None:
    findings = build_design_findings(
        estimand_records = [],
        identifiability_records = [
            {
                "record_type": "term",
                "analysis_population": (
                    "eligible_positive_horizon_cell_design"
                ),
                "aging_mode": "calendar_aging",
                "design_scope": "Stage_1",
                "outcome": None,
                "cohort": None,
                "horizon_axis": None,
                "specification": "main_effects",
                "matrix_scope": (
                    "context_family_cell_allocation"
                ),
                "context_family": (
                    "reference_measurement_measurement_devices"
                ),
                "term_name": (
                    "reference_measurement_measurement_devices=device_a"
                ),
                "term_type": "design_context",
                "structurally_estimable": False
            },
            {
                "record_type": "term",
                "analysis_population": (
                    "eligible_positive_horizon_cell_design"
                ),
                "aging_mode": "calendar_aging",
                "design_scope": "Stage_1",
                "outcome": None,
                "cohort": None,
                "horizon_axis": None,
                "specification": "main_effects",
                "matrix_scope": (
                    "context_family_cell_allocation"
                ),
                "context_family": CONTEXT_FAMILY_NONE,
                "term_name": (
                    "ambient_temperature_celsius"
                ),
                "term_type": "main_effect",
                "structurally_estimable": False
            }
        ],
        correlation_records = [],
        stage_overlap_records = []
    )

    assert len(
        findings["term_estimability_aggregates"]
    ) == 1

    assert findings[
        "term_estimability_aggregates"
    ][0][
        "structurally_estimable"
    ] is False

    assert findings[
        "term_estimability_aggregates"
    ][0][
        "term_count"
    ] == 1


# Confirming that assigned cell diagnostics include the schedule treatment term
def test_assigned_cell_design_contains_schedule_term() -> None:
    cell_records = [
        {
            "cell_identity": (
                f"Stage_{stage_number}/TP_k01_0{cell_number}"
            ),
            "dataset_stage": f"Stage_{stage_number}",
            "aging_type": "k",
            "aging_mode": "calendar_aging",
            "test_point_number": "1",
            "laboratory": "SIE",
            "ambient_temperature_celsius": "23.0",
            "maximum_state_of_charge_fraction": "0.5",
            "reference_test_schedule": (
                "periodic_reference_tests"
                if cell_number <= 2
                else "initial_and_final_only"
            ),
            "sampling_method": "FF"
        }
        for stage_number in (1, 2)
        for cell_number in range(1, 5)
    ]

    design_records = build_population_identifiability(
        records = cell_records,
        aging_mode = "calendar_aging",
        analysis_population = "assigned_cell_design",
        outcome = None,
        cohort = None,
        horizon_axis = None,
        equipment_by_cell = {},
        context_adjusted = True,
        context_family = CONTEXT_FAMILY_NONE
    )

    schedule_terms = [
        record
        for record in design_records
        if (
            record["record_type"] == "term"
            and record["term_name"].startswith(
                "reference_test_schedule="
            )
        )
    ]

    assert schedule_terms


# Confirming that calendar interaction matrices include schedule interactions
def test_calendar_interaction_matrix_contains_schedule_interactions() -> None:
    records = [
        {
            "cell_identity": f"Stage_1/TP_k01_0{cell_number}",
            "dataset_stage": "Stage_1",
            "aging_mode": "calendar_aging",
            "ambient_temperature_celsius": temperature,
            "maximum_state_of_charge_fraction": "0.5",
            "reference_test_schedule": schedule,
            "laboratory": "SIE"
        }
        for (cell_number, temperature, schedule) in [
            (1, "10.0", "periodic_reference_tests"),
            (2, "23.0", "periodic_reference_tests"),
            (3, "45.0", "periodic_reference_tests"),
            (4, "10.0", "initial_and_final_only"),
            (5, "23.0", "initial_and_final_only"),
            (6, "45.0", "initial_and_final_only")
        ]
    ]

    _, matrix_terms = build_context_adjusted_design_matrix(
        records = records,
        aging_mode = "calendar_aging",
        design_scope = "Stage_1",
        specification = "main_effects_and_pairwise_interactions",
        equipment_by_cell = {},
        context_family = CONTEXT_FAMILY_NONE
    )

    term_names = [
        term["term_name"]
        for term in matrix_terms
    ]

    assert any(
        term_name.startswith(
            "ambient_temperature_celsius:"
            "reference_test_schedule="
        )
        for term_name in term_names
    )

    assert any(
        term_name.startswith(
            "maximum_state_of_charge_fraction:"
            "reference_test_schedule="
        )
        for term_name in term_names
    )


# Confirming that numerical and protocol stage overlap are reported separately
def test_stage_overlap_reports_numerical_and_protocol_bases() -> None:
    test_point_records = [
        {
            "test_point_identity": "Stage_1/TP_k01",
            "aging_mode": "calendar_aging",
            "dataset_stage": "Stage_1",
            "ambient_temperature_celsius": "23.0",
            "maximum_state_of_charge_fraction": "0.5",
            "reference_test_schedules": [
                "periodic_reference_tests"
            ]
        },
        {
            "test_point_identity": "Stage_2/TP_k01",
            "aging_mode": "calendar_aging",
            "dataset_stage": "Stage_2",
            "ambient_temperature_celsius": "23.0",
            "maximum_state_of_charge_fraction": "0.5",
            "reference_test_schedules": [
                "initial_and_final_only"
            ]
        }
    ]

    overlap_records = build_stage_overlap(
        test_point_records
    )

    numerical_summary = next(
        record
        for record in overlap_records
        if record["record_type"] == "exact_combination_summary"
    )

    protocol_summary = next(
        record
        for record in overlap_records
        if (
            record["record_type"]
            == "exact_condition_protocol_summary"
        )
    )

    assert numerical_summary["exact_overlap_count"] == 1

    assert protocol_summary["exact_overlap_count"] == 0


# Confirming that matching schedules produce protocol overlap of one
def test_stage_overlap_protocol_basis_counts_matching_schedules() -> None:
    test_point_records = [
        {
            "test_point_identity": "Stage_1/TP_k01",
            "aging_mode": "calendar_aging",
            "dataset_stage": "Stage_1",
            "ambient_temperature_celsius": "23.0",
            "maximum_state_of_charge_fraction": "0.5",
            "reference_test_schedules": [
                "periodic_reference_tests"
            ]
        },
        {
            "test_point_identity": "Stage_2/TP_k01",
            "aging_mode": "calendar_aging",
            "dataset_stage": "Stage_2",
            "ambient_temperature_celsius": "23.0",
            "maximum_state_of_charge_fraction": "0.5",
            "reference_test_schedules": [
                "periodic_reference_tests"
            ]
        }
    ]

    overlap_records = build_stage_overlap(
        test_point_records
    )

    protocol_summary = next(
        record
        for record in overlap_records
        if (
            record["record_type"]
            == "exact_condition_protocol_summary"
        )
    )

    assert protocol_summary["exact_overlap_count"] == 1


# Confirming that assigned cell diagnostics tolerate missing observation counts
def test_assigned_cell_diagnostics_accept_missing_observation_count() -> None:
    cell_records = [
        {
            "cell_identity": (
                f"Stage_{stage_number}/TP_z01_0{cell_number}"
            ),
            "dataset_stage": f"Stage_{stage_number}",
            "aging_type": "z",
            "aging_mode": "cycle_aging",
            "test_point_number": "1",
            "laboratory": "SIE",
            "ambient_temperature_celsius": "23.0",
            "maximum_state_of_charge_fraction": "0.5",
            "depth_of_discharge_fraction": "0.5",
            "charge_rate_c": "1.0",
            "discharge_rate_c": "1.0",
            "reference_test_schedule": (
                "periodic_reference_tests"
            ),
            "sampling_method": "cLH"
        }
        for stage_number in (1, 2)
        for cell_number in range(1, 4)
    ]

    design_records = build_population_identifiability(
        records = cell_records,
        aging_mode = "cycle_aging",
        analysis_population = "assigned_cell_design",
        outcome = None,
        cohort = None,
        horizon_axis = None,
        equipment_by_cell = {},
        context_adjusted = True,
        context_family = CONTEXT_FAMILY_NONE
    )

    assert design_records

    summary = next(
        record
        for record in design_records
        if record["record_type"] == "matrix_summary"
    )

    assert summary["analytical_observation_count"] is None


# Confirming that every stage-overlap record carries its overlap basis
def test_stage_overlap_records_overlap_basis() -> None:
    test_point_records = [
        {
            "test_point_identity": "Stage_1/TP_z01",
            "aging_mode": "cycle_aging",
            "dataset_stage": "Stage_1",
            "ambient_temperature_celsius": "45.0",
            "maximum_state_of_charge_fraction": "0.8",
            "depth_of_discharge_fraction": "0.6",
            "charge_rate_c": "1.0",
            "discharge_rate_c": "2.0"
        },
        {
            "test_point_identity": "Stage_2/TP_z01",
            "aging_mode": "cycle_aging",
            "dataset_stage": "Stage_2",
            "ambient_temperature_celsius": "45.0",
            "maximum_state_of_charge_fraction": "0.8",
            "depth_of_discharge_fraction": "0.6",
            "charge_rate_c": "1.0",
            "discharge_rate_c": "2.0"
        }
    ]

    overlap_records = build_stage_overlap(
        test_point_records
    )

    assert overlap_records

    assert all(
        "overlap_basis" in record
        for record in overlap_records
    )


# Rejecting discrete-horizon support mode with no approved horizons
def test_load_analysis_policy_rejects_empty_horizon_definitions(tmp_path: Path) -> None:
    from scripts.audit_experimental_design import (
        load_analysis_policy
    )

    policy = build_valid_policy_payload()

    policy["support_mode"] = (
        "approved_discrete_horizons"
    )

    policy["horizon_definitions"] = []

    policy_path = (
        tmp_path
        / "empty_horizons_policy.json"
    )

    policy_path.write_text(
        json.dumps(policy),
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = "at least one approved comparison horizon"
    ):
        load_analysis_policy(
            policy_path = policy_path
        )


# Confirming exact-observation support is available when a horizon matches
def test_evaluate_cell_horizon_support_exact_observation() -> None:
    from scripts.audit_experimental_design import (
        evaluate_cell_horizon_support
    )

    support = evaluate_cell_horizon_support(
        horizon_values = [10.0, 30.0],
        horizon = 10.0,
        interpolation_rule = "exact_observation_only",
        maximum_bracketing_span = None
    )

    assert support["supported"] is True

    assert support["support_type"] == "exact_observation"


# Confirming within-cell bracketing is used when the horizon is interior
def test_evaluate_cell_horizon_support_within_cell_bracketing() -> None:
    from scripts.audit_experimental_design import (
        evaluate_cell_horizon_support
    )

    support = evaluate_cell_horizon_support(
        horizon_values = [10.0, 30.0],
        horizon = 20.0,
        interpolation_rule = "within_cell_bracketing",
        maximum_bracketing_span = 25.0
    )

    assert support["supported"] is True

    assert support["support_type"] == "within_cell_bracketing"

    assert support["bracketing_span"] == pytest.approx(20.0)


# Rejecting bracketing spans that exceed the policy
def test_evaluate_cell_horizon_support_rejects_wide_bracketing() -> None:
    from scripts.audit_experimental_design import (
        evaluate_cell_horizon_support
    )

    support = evaluate_cell_horizon_support(
        horizon_values = [10.0, 100.0],
        horizon = 55.0,
        interpolation_rule = "within_cell_bracketing",
        maximum_bracketing_span = 30.0
    )

    assert support["supported"] is False

    assert support["support_type"] == (
        "bracketing_span_exceeds_policy"
    )


# Confirming an undefined pooled variance returns None
def test_calculate_standardized_mean_difference_undefined_variance() -> None:
    from scripts.audit_experimental_design import (
        calculate_standardized_mean_difference
    )

    result = calculate_standardized_mean_difference(
        first_values = [5.0, 5.0],
        second_values = [5.0, 5.0]
    )

    assert result is None


# Rejecting coverage fractions outside the open-closed unit interval
@pytest.mark.parametrize(
    "invalid_coverage_fraction",
    [0.0, -0.5, 1.5, 2.0]
)
def test_load_analysis_policy_rejects_invalid_coverage_fraction(tmp_path: Path, invalid_coverage_fraction: float) -> None:
    from scripts.audit_experimental_design import (
        load_analysis_policy
    )

    policy = build_valid_policy_payload()

    policy["coverage_rules"]["primary"][
        "minimum_assigned_cell_coverage_fraction"
    ] = invalid_coverage_fraction

    policy_path = (
        tmp_path
        / "invalid_coverage_policy.json"
    )

    policy_path.write_text(
        json.dumps(policy),
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = "Coverage fraction must be greater than zero"
    ):
        load_analysis_policy(
            policy_path = policy_path
        )


# Rejecting policies that omit required top-level fields
@pytest.mark.parametrize(
    "missing_field",
    [
        "policy_version",
        "support_mode",
        "coverage_rules",
        "horizon_definitions",
        "calendar_exposure_policy",
        "longitudinal_observation_policy",
        "treatment_evaluation_policy"
    ]
)
def test_load_analysis_policy_rejects_missing_required_fields(tmp_path: Path, missing_field: str) -> None:
    from scripts.audit_experimental_design import (
        load_analysis_policy
    )

    policy = build_valid_policy_payload()

    del policy[missing_field]

    policy_path = (
        tmp_path
        / "missing_field_policy.json"
    )

    policy_path.write_text(
        json.dumps(policy),
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = "missing or unexpected fields"
    ):
        load_analysis_policy(
            policy_path = policy_path
        )


# Rejecting unsupported analysis roles and interpolation rules
@pytest.mark.parametrize(
    ("invalid_field", "invalid_value", "expected_error"),
    [
        (
            "analysis_role",
            "speculative",
            "Unknown horizon analysis role"
        ),
        (
            "interpolation_rule",
            "extrapolate_freely",
            "Unknown interpolation rule"
        )
    ]
)
def test_load_analysis_policy_rejects_unknown_enumerations(tmp_path: Path, invalid_field: str,
                                                           invalid_value: str, expected_error: str
) -> None:
    from scripts.audit_experimental_design import (
        load_analysis_policy
    )

    policy = build_valid_policy_payload()

    policy["support_mode"] = (
        "approved_discrete_horizons"
    )

    horizon_definition = {
        "aging_mode": "calendar_aging",
        "horizon_axis": "elapsed_calendar_days",
        "horizon": 30.0,
        "analysis_role": "primary",
        "interpolation_rule": "exact_observation_only",
        "maximum_bracketing_span": None,
        "rationale": "test",
        "source": "test"
    }

    horizon_definition[invalid_field] = invalid_value

    policy["horizon_definitions"] = [
        horizon_definition
    ]

    policy_path = (
        tmp_path
        / "invalid_enumeration_policy.json"
    )

    policy_path.write_text(
        json.dumps(policy),
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = expected_error
    ):
        load_analysis_policy(
            policy_path = policy_path
        )


# Confirming primary all-three and sensitivity two-of-three differ
def test_coverage_rules_distinguish_primary_and_sensitivity() -> None:
    horizon_values_by_cell = {
        "cell_1": [10.0, 30.0],
        "cell_2": [10.0, 30.0],
        "cell_3": [10.0]
    }

    assigned_cell_identities = {
        "cell_1",
        "cell_2",
        "cell_3"
    }

    primary_summary = summarize_common_horizon_support(
        horizon_values_by_cell = horizon_values_by_cell,
        assigned_cell_identities = assigned_cell_identities,
        candidate_horizons = [10.0, 30.0],
        minimum_assigned_cell_coverage_fraction = 1.0
    )

    sensitivity_summary = summarize_common_horizon_support(
        horizon_values_by_cell = horizon_values_by_cell,
        assigned_cell_identities = assigned_cell_identities,
        candidate_horizons = [10.0, 30.0],
        minimum_assigned_cell_coverage_fraction = 2 / 3
    )

    assert primary_summary[
        "coverage_eligible_observed_horizons"
    ] == [10.0]

    assert sensitivity_summary[
        "coverage_eligible_observed_horizons"
    ] == [10.0, 30.0]


# Confirming that removing one supporting cell changes rank
def test_horizon_specific_rank_changes_when_supporting_cell_removed() -> None:
    from scripts.audit_experimental_design import (
        build_horizon_specific_cell_design
    )

    cell_records = [
        {
            "cell_identity": f"Stage_1/TP_k01_0{cell_number}",
            "condition_protocol_identity": (
                "Stage_1/TP_k01/periodic_reference_tests"
            ),
            "dataset_stage": "Stage_1",
            "aging_mode": "calendar_aging",
            "aging_type": "k",
            "test_point_number": "1",
            "ambient_temperature_celsius": temperature,
            "maximum_state_of_charge_fraction": "0.5",
            "reference_test_schedule": "periodic_reference_tests",
            "laboratory": "SIE",
            "sampling_method": "FF"
        }
        for (cell_number, temperature) in [
            (1, "10.0"),
            (2, "23.0"),
            (3, "45.0")
        ]
    ]

    reference_records = [
        {
            "cell_identity": f"Stage_1/TP_k01_0{cell_number}",
            "relative_capacity": "0.98",
            "relative_capacity_primary_eligible": "true",
            "elapsed_calendar_time_status": "completed",
            "elapsed_calendar_days": "30"
        }
        for cell_number in range(1, 4)
    ]

    horizon_definition = {
        "aging_mode": "calendar_aging",
        "horizon_axis": "elapsed_calendar_days",
        "horizon": 30.0,
        "analysis_role": "primary",
        "interpolation_rule": "exact_observation_only",
        "maximum_bracketing_span": None,
        "rationale": "test",
        "source": "test"
    }

    (
        supported_records,
        support_audit_records
    ) = build_horizon_specific_cell_design(
        cell_records = cell_records,
        reference_records = reference_records,
        aging_mode = "calendar_aging",
        outcome_definition = OUTCOME_DEFINITIONS["relative_capacity"],
        cohort_name = "primary",
        horizon_axis = "elapsed_calendar_days",
        horizon_definition = horizon_definition
    )

    assert len(supported_records) == 3
    assert all(
        record["horizon_support_type"] == "exact_observation"
        for record in supported_records
    )
    assert len(support_audit_records) == 3

    # Dropping the third cell removes one design row
    reduced_records = [
        record
        for record in supported_records
        if record["cell_identity"] != "Stage_1/TP_k01_03"
    ]

    reduced_matrix, _ = build_design_matrix(
        records = reduced_records,
        aging_mode = "calendar_aging",
        design_scope = "Stage_1",
        specification = "main_effects"
    )

    assert reduced_matrix.shape[0] == 2


# Confirming direct multi-treatment contrasts return joint-condition support
def test_multivariable_support_is_decoupled_from_direct_contrasts() -> None:
    direct_status, direct_detail = classify_estimand_support(
        changed_treatment_count = 3,
        context_difference_count = 0,
        equipment_disjoint_dimension_count = 0,
        equipment_identity_conflicted_dimension_count = 0,
        minimum_eligible_cell_count = 3,
        observed_horizon_overlap_available = True,
        common_support_interval_available = True,
        target_estimand_available = True
    )

    assert direct_status == "supported_with_restrictions"
    assert direct_detail == (
        "replicated_joint_condition_analytical_contrast_with_restrictions"
    )

    from scripts.audit_experimental_design import (
        build_analytical_scope_table
    )

    scope_records = build_analytical_scope_table(
        identifiability_records = [
            {
                "record_type": "term",
                "analysis_population": "bracketing_range_cell_design",
                "aging_mode": "cycle_aging",
                "design_scope": "Stage_1",
                "sampling_design_scope": (
                    "Stage_1/sampling_method=cLH"
                ),
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": (
                    "cumulative_full_equivalent_cycles_observed"
                ),
                "analysis_role": (
                    "within_cell_bracketing_support"
                ),
                "specification": "main_effects",
                "context_family": CONTEXT_FAMILY_NONE,
                "context_term_count": 0,
                "term_name": "ambient_temperature_celsius",
                "term_type": "main_effect",
                "structurally_estimable": True,
                "support_region_identity": "region_1",
                "support_region_kind": "interval",
                "support_interval_minimum": 100.0,
                "support_interval_maximum": 250.0,
                "support_interval_lower_inclusive": False,
                "support_interval_upper_inclusive": False,
                "condition_protocol_count": 2,
                "independent_cell_count": 6,
                "analytical_observation_count": 12,
                "residual_independent_cell_degrees_of_freedom": 1,
                "smallest_rank_singular_value": 0.5,
                "effective_condition_number": 4.0
            }
        ]
    )

    assert len(scope_records) == 1
    assert scope_records[0]["authorization_status"] == (
        "permitted_for_model_evaluation_with_restrictions"
    )
    assert scope_records[0]["term_name"] == (
        "ambient_temperature_celsius"
    )


# Confirming every observed sampling regime enters design scopes
def test_build_design_scopes_covers_all_sampling_regimes() -> None:
    from scripts.audit_experimental_design import (
        build_design_scopes
    )

    records = [
        {
            "dataset_stage": "Stage_1",
            "sampling_method": "FF"
        },
        {
            "dataset_stage": "Stage_1",
            "sampling_method": "cLH"
        },
        {
            "dataset_stage": "Stage_2",
            "sampling_method": "pi"
        },
        {
            "dataset_stage": "Stage_2",
            "sampling_method": "cLH"
        }
    ]

    scopes = build_design_scopes(records)

    assert "pooled_stages" in scopes
    assert "Stage_1" in scopes
    assert "Stage_2" in scopes
    assert "Stage_1/sampling_method=FF" in scopes
    assert "Stage_1/sampling_method=cLH" in scopes
    assert "Stage_2/sampling_method=pi" in scopes
    assert "Stage_2/sampling_method=cLH" in scopes


# Confirming the baseline inventory spans all 279 cells
def test_build_baseline_balance_diagnostics_covers_all_cells() -> None:
    from scripts.audit_experimental_design import (
        build_baseline_balance_diagnostics
    )

    cell_records = [
        {
            "cell_identity": f"Stage_1/TP_k01_{cell_number:02d}",
            "condition_protocol_identity": (
                "Stage_1/TP_k01/periodic_reference_tests"
            ),
            "dataset_stage": "Stage_1",
            "aging_type": "k",
            "aging_mode": "calendar_aging",
            "test_point_number": "1",
            "laboratory": "SIE",
            "sampling_method": "FF"
        }
        for cell_number in range(1, 280)
    ]

    reference_records = [
        {
            "cell_identity": f"Stage_1/TP_k01_{cell_number:02d}",
            "sequence_number": "1",
            "mean_capacity_ah": "4.9",
            "uncorrected_dc_internal_resistance_milliohms": "22.5",
            "baseline_voltage_volts": "3.6"
        }
        for cell_number in range(1, 280)
    ]

    diagnostic_records = build_baseline_balance_diagnostics(
        cell_records = cell_records,
        reference_records = reference_records
    )

    stage_summaries = [
        record
        for record in diagnostic_records
        if (
            record["grouping_name"] == "stage"
            and record["baseline_field"]
            == "mean_capacity_ah"
            and record["analysis_population"]
            == "assigned"
        )
    ]

    assert len(stage_summaries) == 1
    assert stage_summaries[0]["available_cell_count"] == 279
    assert stage_summaries[0]["assigned_cell_count"] == 279


# Confirming the whole-protocol calendar estimand is available without net exposure
def test_calendar_estimands_use_whole_protocol_elapsed_time() -> None:
    condition_records = [
        {
            "condition_protocol_identity": (
                f"Stage_1/TP_k01/{schedule}"
            ),
            "aging_mode": "calendar_aging",
            "dataset_stage": "Stage_1",
            "laboratory": "SIE",
            "sampling_method": "FF",
            "ambient_temperature_celsius": "23.0",
            "maximum_state_of_charge_fraction": "0.5",
            "reference_test_schedule": schedule
        }
        for schedule in [
            "periodic_reference_tests",
            "initial_and_final_only"
        ]
    ]

    treatment_support_records = [
        {
            "condition_protocol_identity": (
                f"Stage_1/TP_k01/{schedule}"
            ),
            "reference_measurement_measurement_devices": [],
            "reference_measurement_test_channels": [],
            "reference_measurement_climate_chambers": [],
            "reference_measurement_climate_chamber_positions": [],
            "reference_measurement_identity_conflicted_measurement_devices": [],
            "reference_measurement_identity_conflicted_test_channels": [],
            "reference_measurement_identity_conflicted_climate_chambers": [],
            "reference_measurement_identity_conflicted_climate_chamber_positions": []
        }
        for schedule in [
            "periodic_reference_tests",
            "initial_and_final_only"
        ]
    ]

    horizon_support_records = [
        {
            "condition_protocol_identity": (
                f"Stage_1/TP_k01/{schedule}"
            ),
            "aging_mode": "calendar_aging",
            "outcome": outcome_name,
            "cohort": cohort_name,
            "horizon_axis": "elapsed_calendar_days",
            "assigned_cell_count": 3,
            "eligible_cell_count": 3,
            "eligible_assigned_cell_fraction": 1.0,
            "observed_horizon_minimum": 10.0,
            "observed_horizon_maximum": 300.0,
            "complete_eligible_cell_bracketing_minimum": 10.0,
            "complete_eligible_cell_bracketing_maximum": 300.0,
            "exact_common_horizons": [10.0],
            "coverage_eligible_observed_horizons": [10.0],
            "required_support_cell_count": 2,
            "supported_bracketing_ranges": [
                {
                    "minimum": 10.0,
                    "maximum": 300.0,
                    "required_support_cell_count": 2,
                    "assigned_cell_count": 3,
                    "minimum_assigned_cell_coverage_fraction": 0.5,
                    "support_interpretation": (
                        "within_observed_cell_ranges_without_extrapolation"
                    )
                }
            ]
        }
        for schedule in [
            "periodic_reference_tests",
            "initial_and_final_only"
        ]
        for outcome_name in OUTCOME_DEFINITIONS
        for cohort_name in ["primary", "sensitivity"]
    ]

    analysis_policy = {
        "calendar_exposure_policy": {
            "primary_estimand_availability": "available",
            "primary_estimand": (
                "whole_assigned_protocol_outcome_difference_at_common_elapsed_calendar_time"
            ),
            "net_storage_exposure_availability": (
                "unavailable"
            ),
            "net_storage_exposure_model_authorization": (
                "not_authorized"
            ),
            "claim_boundary": (
                "Gross elapsed calendar days support whole-protocol comparisons only. They do not identify time"
                " under storage, isolate calendar-aging exposure, or exclude time spent in reference tests and documented "
                "pauses."
            )
        }
    }

    estimand_records = build_estimand_support(
        condition_records = condition_records,
        treatment_support_records = treatment_support_records,
        horizon_support_records = horizon_support_records,
        coverage_rules = {
            "primary": {
                "minimum_assigned_cell_coverage_fraction": 0.5,
                "rationale": "test"
            },
            "sensitivity": {
                "minimum_assigned_cell_coverage_fraction": 0.5,
                "rationale": "test"
            }
        },
        analysis_policy = analysis_policy
    )

    assert estimand_records

    calendar_records = [
        record
        for record in estimand_records
        if record["aging_mode"] == "calendar_aging"
        and record["horizon_axis"] == "elapsed_calendar_days"
    ]

    assert calendar_records

    assert all(
        record["target_estimand_available"] is True
        for record in calendar_records
    )

    assert all(
        record["target_estimand_identity"]
        == (
            "whole_assigned_protocol_outcome_difference_at_common_elapsed_calendar_time"
        )
        for record in calendar_records
    )

    assert all(
        record["net_calendar_aging_exposure_status"]
        == "unavailable"
        for record in calendar_records
    )

    assert all(
        record["net_calendar_aging_model_authorization"]
        == "not_authorized"
        for record in calendar_records
    )

    cycle_records = [
        record
        for record in estimand_records
        if record["aging_mode"] != "calendar_aging"
        or record["horizon_axis"]
        != "elapsed_calendar_days"
    ]

    assert all(
        record["net_calendar_aging_exposure_status"] is None
        for record in cycle_records
    )


# Confirming continuous-support mode permits an empty horizon list
def test_continuous_support_policy_allows_empty_discrete_horizons(tmp_path: Path) -> None:
    from scripts.audit_experimental_design import (
        load_analysis_policy
    )

    policy = build_valid_policy_payload()

    policy_path = (
        tmp_path
        / "continuous_policy.json"
    )

    policy_path.write_text(
        json.dumps(policy),
        encoding = "utf-8"
    )

    loaded = load_analysis_policy(
        policy_path = policy_path
    )

    assert loaded["horizon_definitions"] == []
    assert loaded["support_mode"] == (
        "within_cell_bracketing_ranges"
    )


# Confirming continuous-interval coverage uses the assigned-cell denominator
def test_calculate_supported_intervals_uses_assigned_cell_denominator() -> None:
    from scripts.audit_experimental_design import (
        calculate_supported_intervals
    )

    intervals = calculate_supported_intervals(
        horizon_values_by_cell = {
            "cell_1": [10.0, 100.0],
            "cell_2": [20.0, 90.0],
            "cell_3": [30.0, 80.0]
        },
        assigned_cell_count = 3,
        minimum_assigned_cell_coverage_fraction = 1.0
    )

    assert intervals == [
        {
            "minimum": 30.0,
            "maximum": 80.0,
            "required_support_cell_count": 3,
            "assigned_cell_count": 3,
            "minimum_assigned_cell_coverage_fraction": 1.0,
            "support_interpretation": (
                "within_observed_cell_ranges_without_extrapolation"
            )
        }
    ]


# Confirming the sensitivity rule preserves two-cell replication
def test_sensitivity_interval_retains_two_cell_replication() -> None:
    from scripts.audit_experimental_design import (
        calculate_supported_intervals
    )

    intervals = calculate_supported_intervals(
        horizon_values_by_cell = {
            "cell_1": [10.0, 100.0],
            "cell_2": [20.0, 90.0],
            "cell_3": [30.0, 40.0]
        },
        assigned_cell_count = 3,
        minimum_assigned_cell_coverage_fraction = 2 / 3
    )

    assert intervals == [
        {
            "minimum": 20.0,
            "maximum": 90.0,
            "required_support_cell_count": 2,
            "assigned_cell_count": 3,
            "minimum_assigned_cell_coverage_fraction": (
                2 / 3
            ),
            "support_interpretation": (
                "within_observed_cell_ranges_without_extrapolation"
            )
        }
    ]


# Confirming intersected support never extrapolates either input interval
def test_supported_interval_intersection_never_extrapolates() -> None:
    from scripts.audit_experimental_design import (
        intersect_supported_intervals
    )

    intersections = intersect_supported_intervals(
        first_intervals = [
            {
                "minimum": 20.0,
                "maximum": 90.0,
                "required_support_cell_count": 3
            }
        ],
        second_intervals = [
            {
                "minimum": 40.0,
                "maximum": 120.0,
                "required_support_cell_count": 3
            }
        ]
    )

    assert intersections[0]["minimum"] == pytest.approx(
        40.0
    )

    assert intersections[0]["maximum"] == pytest.approx(
        90.0
    )


# Confirming pooled-stage scope cannot receive analytical authorization
def test_pooled_stage_scope_is_not_authorized() -> None:
    from scripts.audit_experimental_design import (
        build_analytical_scope_table
    )

    records = build_analytical_scope_table(
        identifiability_records = [
            {
                "record_type": "term",
                "analysis_population": (
                    "bracketing_range_cell_design"
                ),
                "aging_mode": "cycle_aging",
                "design_scope": "pooled_stages",
                "sampling_design_scope": "combined_within_scope",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": (
                    "cumulative_full_equivalent_cycles_observed"
                ),
                "analysis_role": (
                    "within_cell_bracketing_support"
                ),
                "specification": "main_effects",
                "context_family": CONTEXT_FAMILY_NONE,
                "context_term_count": 0,
                "term_name": "ambient_temperature_celsius",
                "term_type": "main_effect",
                "structurally_estimable": True,
                "support_region_identity": "region_1",
                "support_region_kind": "open_interval",
                "support_interval_minimum": 100.0,
                "support_interval_maximum": 250.0,
                "support_interval_lower_inclusive": False,
                "support_interval_upper_inclusive": False,
                "condition_protocol_count": 4,
                "independent_cell_count": 12,
                "analytical_observation_count": 24,
                "residual_independent_cell_degrees_of_freedom": 6,
                "smallest_rank_singular_value": 0.5,
                "effective_condition_number": 4.0
            }
        ]
    )

    assert records[0]["authorization_status"] == (
        "unsupported_cross_stage_pooling_not_established"
    )

    assert records[0]["question_classification"] == (
        "unsupported"
    )


# Confirming primary regions require all assigned cells to span the interval
def test_primary_continuous_region_requires_all_assigned_cells() -> None:
    cell_records = [
        {
            "cell_identity": f"cell_{number}",
            "condition_protocol_identity": "condition_1"
        }
        for number in range(
            1,
            4
        )
    ]

    range_records = [
        {
            "cell_identity": "cell_1",
            "condition_protocol_identity": "condition_1",
            "aging_mode": "calendar_aging",
            "dataset_stage": "Stage_1",
            "laboratory": "A",
            "sampling_method": "design",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "supporting_cell_eligible_observation_count": 2,
            "eligible_horizon_minimum": 10.0,
            "eligible_horizon_maximum": 100.0,
            "eligible_horizons": [10.0, 100.0]
        },
        {
            "cell_identity": "cell_2",
            "condition_protocol_identity": "condition_1",
            "aging_mode": "calendar_aging",
            "dataset_stage": "Stage_1",
            "laboratory": "A",
            "sampling_method": "design",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "supporting_cell_eligible_observation_count": 2,
            "eligible_horizon_minimum": 20.0,
            "eligible_horizon_maximum": 90.0,
            "eligible_horizons": [20.0, 90.0]
        },
        {
            "cell_identity": "cell_3",
            "condition_protocol_identity": "condition_1",
            "aging_mode": "calendar_aging",
            "dataset_stage": "Stage_1",
            "laboratory": "A",
            "sampling_method": "design",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "supporting_cell_eligible_observation_count": 2,
            "eligible_horizon_minimum": 30.0,
            "eligible_horizon_maximum": 40.0,
            "eligible_horizons": [30.0, 40.0]
        }
    ]

    regions, memberships = build_continuous_support_regions(
        cell_records = cell_records,
        cell_horizon_range_records = range_records,
        coverage_rules = {
            "primary": {
                "minimum_assigned_cell_coverage_fraction": 1.0
            },
            "sensitivity": {
                "minimum_assigned_cell_coverage_fraction": 2 / 3
            }
        }
    )

    assert regions

    assert all(
        region["lower_horizon"] >= 30.0
        and region["upper_horizon"] <= 40.0
        for region in regions
    )

    assert {
        membership["cell_identity"]
        for membership in memberships
    } == {
        "cell_1",
        "cell_2",
        "cell_3"
    }


# Confirming sensitivity availability cannot elevate a descriptive contrast
def test_sensitivity_classification_retains_descriptive_limit() -> None:
    common_record = {
        "aging_mode": "calendar_aging",
        "first_condition_protocol_identity": (
            "condition_1"
        ),
        "second_condition_protocol_identity": (
            "condition_2"
        ),
        "outcome": "relative_capacity",
        "horizon_axis": "elapsed_calendar_days"
    }

    records = apply_final_estimand_classifications(
        [
            {
                **common_record,
                "cohort": "primary",
                "direct_contrast_support_status": (
                    "unsupported"
                )
            },
            {
                **common_record,
                "cohort": "sensitivity",
                "direct_contrast_support_status": (
                    "descriptive_only"
                )
            }
        ]
    )

    sensitivity_record = next(
        record
        for record in records
        if record["cohort"] == "sensitivity"
    )

    assert sensitivity_record[
        "final_support_classification"
    ] == "descriptive_only"

    assert sensitivity_record[
        "cohort_comparison_status"
    ] == "available_only_in_sensitivity_cohort"

    assert sensitivity_record[
        "final_support_classification_basis"
    ] == (
        "descriptive_limit_retained_"
        "regardless_of_cohort_availability"
    )


# Confirming only the bracketing-range population can authorize analytical scope
def test_only_bracketing_range_population_can_authorize_scope() -> None:
    records = build_analytical_scope_table(
        identifiability_records = [
            {
                "record_type": "term",
                "analysis_population": (
                    "eligible_positive_horizon_cell_design"
                ),
                "term_type": "main_effect"
            }
        ]
    )

    assert records == []


# Preventing cache reuse across populations with different observation counts
def test_interval_identifiability_cache_preserves_observation_counts() -> None:
    from scripts.audit_experimental_design import (
        build_continuous_support_identifiability
    )

    cell_records = [
        {
            "cell_identity": f"Stage_1/TP_k0{condition}_0{cell}",
            "condition_protocol_identity": (
                f"Stage_1/TP_k0{condition}/"
                "periodic_reference_tests"
            ),
            "dataset_stage": "Stage_1",
            "aging_type": "k",
            "aging_mode": "calendar_aging",
            "test_point_number": str(
                condition
            ),
            "cell_number": str(
                cell
            ),
            "laboratory": "SIE",
            "sampling_method": "FF",
            "ambient_temperature_celsius": (
                "23.0"
                if condition == 1
                else "35.0"
            ),
            "maximum_state_of_charge_fraction": "0.5",
            "depth_of_discharge_fraction": "",
            "charge_rate_c": "",
            "discharge_rate_c": "",
            "reference_test_schedule": (
                "periodic_reference_tests"
            )
        }
        for condition in [
            1,
            2
        ]
        for cell in [
            1,
            2
        ]
    ]

    support_regions = [
        {
            "support_region_identity": "capacity_region",
            "aging_mode": "calendar_aging",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "region_kind": "open_interval",
            "lower_horizon": 10.0,
            "upper_horizon": 20.0,
            "lower_inclusive": False,
            "upper_inclusive": False
        },
        {
            "support_region_identity": "resistance_region",
            "aging_mode": "calendar_aging",
            "outcome": (
                "relative_dc_internal_resistance"
            ),
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "region_kind": "open_interval",
            "lower_horizon": 10.0,
            "upper_horizon": 20.0,
            "lower_inclusive": False,
            "upper_inclusive": False
        }
    ]

    memberships = []

    for cell_record in cell_records:
        for (
            region_identity,
            outcome,
            observation_count
        ) in [
            (
                "capacity_region",
                "relative_capacity",
                2
            ),
            (
                "resistance_region",
                "relative_dc_internal_resistance",
                3
            )
        ]:
            memberships.append(
                {
                    "support_region_identity": (
                        region_identity
                    ),
                    "cell_identity": cell_record[
                        "cell_identity"
                    ],
                    "condition_protocol_identity": (
                        cell_record[
                            "condition_protocol_identity"
                        ]
                    ),
                    "dataset_stage": "Stage_1",
                    "laboratory": "SIE",
                    "sampling_method": "FF",
                    "outcome": outcome,
                    "cohort": "primary",
                    "horizon_axis": (
                        "elapsed_calendar_days"
                    ),
                    "supporting_cell_eligible_observation_count": (
                        observation_count
                    )
                }
            )

    (
        records,
        _
    ) = build_continuous_support_identifiability(
        support_region_records = support_regions,
        support_membership_records = memberships,
        cell_records = cell_records,
        equipment_by_cell = {}
    )

    observation_counts_by_region = {
        region_identity: {
            record["analytical_observation_count"]
            for record in records
            if (
                record["support_region_identity"]
                == region_identity
                and record["design_scope"]
                == "Stage_1"
            )
        }
        for region_identity in [
            "capacity_region",
            "resistance_region"
        ]
    }

    assert observation_counts_by_region[
        "capacity_region"
    ] == {
        8
    }

    assert observation_counts_by_region[
        "resistance_region"
    ] == {
        12
    }


# Preventing adjustment authorization when the context has no matrix column
def test_scope_omits_context_family_without_context_term() -> None:
    records = build_analytical_scope_table(
        identifiability_records = [
            {
                "record_type": "term",
                "analysis_population": (
                    "bracketing_range_cell_design"
                ),
                "aging_mode": "calendar_aging",
                "design_scope": "Stage_1",
                "sampling_design_scope": (
                    "combined_within_scope"
                ),
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": (
                    "elapsed_calendar_days"
                ),
                "analysis_role": (
                    "within_cell_bracketing_support"
                ),
                "specification": "main_effects",
                "context_family": (
                    CONTEXT_FAMILY_LABORATORY
                ),
                "context_term_count": 0,
                "term_name": (
                    "ambient_temperature_celsius"
                ),
                "term_type": "main_effect",
                "structurally_estimable": True,
                "support_region_identity": "region_1",
                "support_region_kind": "open_interval",
                "support_interval_minimum": 10.0,
                "support_interval_maximum": 20.0,
                "support_interval_lower_inclusive": False,
                "support_interval_upper_inclusive": False,
                "condition_protocol_count": 2,
                "independent_cell_count": 6,
                "analytical_observation_count": 12,
                "residual_independent_cell_degrees_of_freedom": 3,
                "smallest_rank_singular_value": 0.5,
                "effective_condition_number": 4.0
            }
        ]
    )

    assert records == []


# Confirming baseline balance keeps assigned and eligible populations separate
def test_baseline_balance_separates_eligibility_populations() -> None:
    from scripts.audit_experimental_design import (
        build_baseline_balance_diagnostics
    )

    cell_records = [
        {
            "cell_identity": f"Stage_1/TP_k01_0{cell_number}",
            "condition_protocol_identity": (
                "Stage_1/TP_k01/"
                "periodic_reference_tests"
            ),
            "dataset_stage": "Stage_1",
            "aging_type": "k",
            "aging_mode": "calendar_aging",
            "test_point_number": "1",
            "laboratory": "SIE",
            "sampling_method": "FF"
        }
        for cell_number in [
            1,
            2,
            3
        ]
    ]

    reference_records = [
        {
            "cell_identity": f"Stage_1/TP_k01_0{cell_number}",
            "sequence_number": "1",
            "mean_capacity_ah": capacity,
            "uncorrected_dc_internal_resistance_milliohms": (
                resistance
            ),
            "baseline_voltage_volts": "3.6",
            "relative_capacity_primary_eligible": (
                primary_eligible
            ),
            "relative_capacity_sensitivity_eligible": (
                sensitivity_eligible
            ),
            "relative_dc_internal_resistance_primary_eligible": (
                primary_eligible
            ),
            "relative_dc_internal_resistance_sensitivity_eligible": (
                sensitivity_eligible
            ),
            "primary_cohort_eligible": (
                primary_eligible
            ),
            "sensitivity_cohort_eligible": (
                sensitivity_eligible
            )
        }
        for (
            cell_number,
            capacity,
            resistance,
            primary_eligible,
            sensitivity_eligible
        ) in [
            (
                1,
                "4.90",
                "22.0",
                "true",
                "true"
            ),
            (
                2,
                "4.95",
                "22.5",
                "false",
                "true"
            ),
            (
                3,
                "5.00",
                "23.0",
                "false",
                "false"
            )
        ]
    ]

    diagnostic_records = (
        build_baseline_balance_diagnostics(
            cell_records = cell_records,
            reference_records = reference_records
        )
    )

    summaries = {
        record["analysis_population"]: record
        for record in diagnostic_records
        if (
            record["record_type"]
            == "baseline_group_summary"
            and record["grouping_name"]
            == "stage"
            and record["baseline_field"]
            == "mean_capacity_ah"
        )
    }

    assert summaries[
        "assigned"
    ][
        "available_cell_count"
    ] == 3

    assert summaries[
        "primary"
    ][
        "available_cell_count"
    ] == 1

    assert summaries[
        "sensitivity"
    ][
        "available_cell_count"
    ] == 2

    assert all(
        summary[
            "assigned_cell_count"
        ] == 3
        for summary in summaries.values()
    )


# Confirming analytical support available only in sensitivity remains sensitivity-only
def test_supported_sensitivity_contrast_is_sensitivity_only() -> None:
    common_record = {
        "aging_mode": "calendar_aging",
        "first_condition_protocol_identity": (
            "condition_1"
        ),
        "second_condition_protocol_identity": (
            "condition_2"
        ),
        "outcome": "relative_capacity",
        "horizon_axis": "elapsed_calendar_days"
    }

    records = apply_final_estimand_classifications(
        [
            {
                **common_record,
                "cohort": "primary",
                "direct_contrast_support_status": (
                    "unsupported"
                )
            },
            {
                **common_record,
                "cohort": "sensitivity",
                "direct_contrast_support_status": (
                    "supported_with_restrictions"
                )
            }
        ]
    )

    sensitivity_record = next(
        record
        for record in records
        if record["cohort"] == "sensitivity"
    )

    assert sensitivity_record[
        "final_support_classification"
    ] == "sensitivity_only"

    assert sensitivity_record[
        "underlying_direct_contrast_support_status"
    ] == "supported_with_restrictions"

    assert sensitivity_record[
        "cohort_comparison_status"
    ] == "available_only_in_sensitivity_cohort"


# Confirming cross-scope baseline pairs are retained but prohibited
def test_baseline_pairs_crossing_scope_are_marked_descriptive_only() -> None:
    from scripts.audit_experimental_design import (
        build_baseline_balance_diagnostics
    )

    cell_records = [
        {
            "cell_identity": "Stage_1/TP_k01_01",
            "condition_protocol_identity": (
                "Stage_1/TP_k01/periodic_reference_tests"
            ),
            "dataset_stage": "Stage_1",
            "aging_mode": "calendar_aging",
            "aging_type": "k",
            "test_point_number": "1",
            "laboratory": "SIE",
            "sampling_method": "FF"
        },
        {
            "cell_identity": "Stage_1/TP_z01_01",
            "condition_protocol_identity": (
                "Stage_1/TP_z01/periodic_reference_tests"
            ),
            "dataset_stage": "Stage_1",
            "aging_mode": "cycle_aging",
            "aging_type": "z",
            "test_point_number": "1",
            "laboratory": "SIE",
            "sampling_method": "cLH"
        },
        {
            "cell_identity": "Stage_2/TP_z01_01",
            "condition_protocol_identity": (
                "Stage_2/TP_z01/periodic_reference_tests"
            ),
            "dataset_stage": "Stage_2",
            "aging_mode": "cycle_aging",
            "aging_type": "z",
            "test_point_number": "1",
            "laboratory": "SIE",
            "sampling_method": "cLH"
        }
    ]

    reference_records = [
        {
            "cell_identity": record["cell_identity"],
            "sequence_number": "1",
            "mean_capacity_ah": "4.9",
            "uncorrected_dc_internal_resistance_milliohms": "22.5",
            "baseline_voltage_volts": "3.6"
        }
        for record in cell_records
    ]

    diagnostic_records = build_baseline_balance_diagnostics(
        cell_records = cell_records,
        reference_records = reference_records
    )

    pair_records = [
        record
        for record in diagnostic_records
        if record["record_type"] == "condition_pair_difference"
    ]

    cross_aging_mode_records = [
        record
        for record in pair_records
        if record["comparison_scope_status"] == "cross_aging_mode"
    ]
    cross_stage_records = [
        record
        for record in pair_records
        if record["comparison_scope_status"] == "cross_stage"
    ]

    assert cross_aging_mode_records
    assert cross_stage_records

    assert all(
        record["comparison_scope_authorization"]
        == "descriptive_only_prohibited_for_treatment_support"
        for record in cross_aging_mode_records
        + cross_stage_records
    )


# Confirming bracketing regions do not claim continuous observation
def test_bracketing_regions_declare_no_continuous_observation() -> None:
    from scripts.audit_experimental_design import (
        build_continuous_support_regions
    )

    cell_records = [
        {
            "cell_identity": "cell_1",
            "condition_protocol_identity": "condition_1"
        }
    ]

    range_records = [
        {
            "cell_identity": "cell_1",
            "condition_protocol_identity": "condition_1",
            "aging_mode": "calendar_aging",
            "dataset_stage": "Stage_1",
            "laboratory": "A",
            "sampling_method": "design",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "supporting_cell_eligible_observation_count": 2,
            "eligible_horizon_minimum": 10.0,
            "eligible_horizon_maximum": 100.0,
            "eligible_horizons": [10.0, 100.0]
        }
    ]

    regions, memberships = build_continuous_support_regions(
        cell_records = cell_records,
        cell_horizon_range_records = range_records,
        coverage_rules = {
            "primary": {
                "minimum_assigned_cell_coverage_fraction": 1.0
            },
            "sensitivity": {
                "minimum_assigned_cell_coverage_fraction": 2 / 3
            }
        }
    )

    assert regions
    assert memberships

    assert all(
        region["support_geometry"]
        == "within_cell_minimum_maximum_bracketing_range"
        and region["continuous_observation_status"]
        == "not_established_between_observed_horizons"
        and region["interpolation_authorization_status"]
        == "requires_model_specific_validation"
        for region in regions
    )

    assert all(
        membership["continuous_observation_status"]
        == "not_established_between_observed_horizons"
        for membership in memberships
    )


# Confirming equipment context never receives a permitted authorization
def test_equipment_context_is_not_authorized_for_adjustment() -> None:
    from scripts.audit_experimental_design import (
        build_analytical_scope_table
    )

    common = {
        "record_type": "term",
        "analysis_population": "bracketing_range_cell_design",
        "aging_mode": "cycle_aging",
        "design_scope": "Stage_1",
        "sampling_design_scope": "combined_within_scope",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": (
            "cumulative_full_equivalent_cycles_observed"
        ),
        "analysis_role": "within_cell_bracketing_support",
        "specification": "main_effects",
        "context_term_count": 1,
        "term_name": "ambient_temperature_celsius",
        "term_type": "main_effect",
        "structurally_estimable": True,
        "support_region_identity": "region_1",
        "support_region_kind": "open_interval",
        "support_interval_minimum": 10.0,
        "support_interval_maximum": 20.0,
        "support_interval_lower_inclusive": False,
        "support_interval_upper_inclusive": False,
        "condition_protocol_count": 2,
        "independent_cell_count": 6,
        "analytical_observation_count": 12,
        "residual_independent_cell_degrees_of_freedom": 1,
        "smallest_rank_singular_value": 0.5,
        "effective_condition_number": 4.0
    }

    records = build_analytical_scope_table(
        identifiability_records = [
            {
                **common,
                "context_family": (
                    "reference_measurement_measurement_devices"
                )
            }
        ]
    )

    assert records[0]["authorization_status"] == (
        "unsupported_equipment_adjustment_not_"
        "outcome_horizon_specific"
    )
    assert records[0]["question_classification"] == "unsupported"
    assert records[0]["permitted_adjustment_variables"] == []
    assert records[0]["context_evidence_role"] == (
        "descriptive_allocation_diagnostic_only"
    )


# Confirming diagnostics are carried and inferential claim is not authorized
def test_structural_estimability_is_diagnostic_not_inferential() -> None:
    from scripts.audit_experimental_design import (
        build_analytical_scope_table
    )

    record = {
        "record_type": "term",
        "analysis_population": "bracketing_range_cell_design",
        "aging_mode": "cycle_aging",
        "design_scope": "Stage_1",
        "sampling_design_scope": "combined_within_scope",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": (
            "cumulative_full_equivalent_cycles_observed"
        ),
        "analysis_role": "within_cell_bracketing_support",
        "specification": "main_effects",
        "context_family": CONTEXT_FAMILY_NONE,
        "context_term_count": 0,
        "term_name": "ambient_temperature_celsius",
        "term_type": "main_effect",
        "structurally_estimable": True,
        "support_region_identity": "region_1",
        "support_region_kind": "open_interval",
        "support_interval_minimum": 10.0,
        "support_interval_maximum": 20.0,
        "support_interval_lower_inclusive": False,
        "support_interval_upper_inclusive": False,
        "condition_protocol_count": 2,
        "independent_cell_count": 6,
        "analytical_observation_count": 12,
        "residual_independent_cell_degrees_of_freedom": 1,
        "smallest_rank_singular_value": 0.5,
        "effective_condition_number": 4.0
    }

    scope_records = build_analytical_scope_table(
        identifiability_records = [record]
    )
    assert len(scope_records) == 1
    output = scope_records[0]

    assert output["authorization_status"] == (
        "permitted_for_model_evaluation_with_restrictions"
    )
    assert output["inferential_adequacy_status"] == (
        "requires_model_specific_validation"
    )
    assert output["inferential_claim_authorization"] == (
        "not_authorized_by_design_audit"
    )
    assert output["residual_independent_cell_degrees_of_freedom"] == 1
    assert output["smallest_rank_singular_value"] == pytest.approx(0.5)
    assert output["effective_condition_number"] == pytest.approx(4.0)


# Confirming the production analysis policy loads with the required sections
def test_production_analysis_policy_loads() -> None:
    from scripts.audit_experimental_design import (
        load_analysis_policy
    )

    policy_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experimental_design_policy.json"
    )

    policy = load_analysis_policy(
        policy_path = policy_path
    )

    assert "longitudinal_observation_policy" in policy
    assert "treatment_evaluation_policy" in policy

    assert policy["support_mode"] == (
        "within_cell_bracketing_ranges"
    )

    calendar_policy = policy["calendar_exposure_policy"]

    assert (
        calendar_policy["primary_estimand_availability"]
        == "available"
    )
    assert (
        calendar_policy["net_storage_exposure_availability"]
        == "unavailable"
    )
    assert (
        calendar_policy[
            "net_storage_exposure_model_authorization"
        ]
        == "not_authorized"
    )


# Rejecting a policy that omits the longitudinal observation policy
def test_load_analysis_policy_rejects_missing_longitudinal_policy(tmp_path: Path) -> None:
    from scripts.audit_experimental_design import (
        load_analysis_policy
    )

    policy = build_valid_policy_payload()

    del policy["longitudinal_observation_policy"]

    policy_path = (
        tmp_path
        / "missing_longitudinal.json"
    )

    policy_path.write_text(
        json.dumps(policy),
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = "missing or unexpected fields"
    ):
        load_analysis_policy(
            policy_path = policy_path
        )


# Rejecting an unsupported longitudinal-policy value
def test_load_analysis_policy_rejects_invalid_longitudinal_value(tmp_path: Path) -> None:
    from scripts.audit_experimental_design import (
        load_analysis_policy
    )

    policy = build_valid_policy_payload()

    policy["longitudinal_observation_policy"][
        "support_horizon_requirement"
    ] = "any_sign"

    policy_path = (
        tmp_path
        / "invalid_longitudinal.json"
    )

    policy_path.write_text(
        json.dumps(policy),
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = "Unsupported longitudinal-observation policy value"
    ):
        load_analysis_policy(
            policy_path = policy_path
        )


# Rejecting an unsupported calendar-policy value
def test_load_analysis_policy_rejects_invalid_calendar_value(tmp_path: Path) -> None:
    from scripts.audit_experimental_design import (
        load_analysis_policy
    )

    policy = build_valid_policy_payload()

    policy["calendar_exposure_policy"][
        "primary_time_axis"
    ] = "gross_elapsed_calendar_time"

    policy_path = (
        tmp_path
        / "invalid_calendar.json"
    )

    policy_path.write_text(
        json.dumps(policy),
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = "Unsupported calendar-exposure policy value"
    ):
        load_analysis_policy(
            policy_path = policy_path
        )


# Rejecting empty or non-string rationales
@pytest.mark.parametrize(
    ("section_name", "nested_key"),
    [
        ("coverage_rules", "primary"),
        ("calendar_exposure_policy", None),
        ("longitudinal_observation_policy", None),
        ("treatment_evaluation_policy", None)
    ]
)
@pytest.mark.parametrize(
    "bad_rationale",
    ["", "   ", 42, ["not a string"]]
)
def test_load_analysis_policy_rejects_invalid_rationale(tmp_path: Path, section_name: str,
                                                        nested_key: str | None, bad_rationale: object
) -> None:
    from scripts.audit_experimental_design import (
        load_analysis_policy
    )

    policy = build_valid_policy_payload()

    if nested_key is None:
        policy[section_name]["rationale"] = bad_rationale
    else:
        policy[section_name][nested_key][
            "rationale"
        ] = bad_rationale

    policy_path = (
        tmp_path
        / "bad_rationale.json"
    )

    policy_path.write_text(
        json.dumps(policy),
        encoding = "utf-8"
    )

    with pytest.raises(RuntimeError):
        load_analysis_policy(
            policy_path = policy_path
        )


# Rejecting sensitivity coverage stricter than primary coverage
def test_load_analysis_policy_rejects_sensitivity_stricter_than_primary(tmp_path: Path) -> None:
    from scripts.audit_experimental_design import (
        load_analysis_policy
    )

    policy = build_valid_policy_payload()

    policy["coverage_rules"]["primary"][
        "minimum_assigned_cell_coverage_fraction"
    ] = 0.5

    policy["coverage_rules"]["sensitivity"][
        "minimum_assigned_cell_coverage_fraction"
    ] = 1.0

    policy_path = (
        tmp_path
        / "stricter_sensitivity.json"
    )

    policy_path.write_text(
        json.dumps(policy),
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = "Sensitivity coverage cannot be stricter"
    ):
        load_analysis_policy(
            policy_path = policy_path
        )


# Confirming overlapping but non-aligned cohort support regions are rejected
def test_overlapping_misaligned_primary_and_sensitivity_geometry_raises() -> None:
    from scripts.audit_experimental_design import (
        validate_cohort_aligned_support_regions
    )

    primary_region = {
        "region_kind": "interval",
        "lower_horizon": 10.0,
        "upper_horizon": 20.0,
        "lower_inclusive": False,
        "upper_inclusive": False,
        "aging_mode": "calendar_aging",
        "outcome": "relative_capacity",
        "horizon_axis": "elapsed_calendar_days",
        "cohort": "primary"
    }

    sensitivity_region = {
        **primary_region,
        "upper_horizon": 25.0,
        "cohort": "sensitivity"
    }

    with pytest.raises(
        RuntimeError,
        match = "without joint geometry alignment"
    ):
        validate_cohort_aligned_support_regions(
            [primary_region, sensitivity_region]
        )


# Confirming support membership retains the eligible observation count
def test_support_membership_retains_eligible_observation_count() -> None:
    cell_records = [
        {
            "cell_identity": "cell_1",
            "condition_protocol_identity": "condition_1"
        }
    ]

    range_records = [
        {
            "cell_identity": "cell_1",
            "condition_protocol_identity": "condition_1",
            "aging_mode": "calendar_aging",
            "dataset_stage": "Stage_1",
            "laboratory": "A",
            "sampling_method": "design",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "supporting_cell_eligible_observation_count": 2,
            "eligible_horizon_minimum": 10.0,
            "eligible_horizon_maximum": 30.0,
            "eligible_horizons": [10.0, 30.0]
        }
    ]

    regions, memberships = build_continuous_support_regions(
        cell_records = cell_records,
        cell_horizon_range_records = range_records,
        coverage_rules = {
            "primary": {
                "minimum_assigned_cell_coverage_fraction": 1.0
            },
            "sensitivity": {
                "minimum_assigned_cell_coverage_fraction": 2 / 3
            }
        }
    )

    assert memberships
    assert all(
        "supporting_cell_eligible_observation_count" in membership
        for membership in memberships
    )


# Confirming the observation-count scope is present on analytical scope
def test_analytical_scope_declares_observation_count_scope() -> None:
    records = build_analytical_scope_table(
        identifiability_records = [
            {
                "record_type": "term",
                "analysis_population": "bracketing_range_cell_design",
                "aging_mode": "cycle_aging",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined_within_scope",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": (
                    "cumulative_full_equivalent_cycles_observed"
                ),
                "analysis_role": "within_cell_bracketing_support",
                "specification": "main_effects",
                "context_family": CONTEXT_FAMILY_NONE,
                "context_term_count": 0,
                "term_name": "ambient_temperature_celsius",
                "term_type": "main_effect",
                "structurally_estimable": True,
                "support_region_identity": "region_1",
                "support_region_kind": "open_interval",
                "support_interval_minimum": 10.0,
                "support_interval_maximum": 20.0,
                "support_interval_lower_inclusive": False,
                "support_interval_upper_inclusive": False,
                "condition_protocol_count": 2,
                "independent_cell_count": 6,
                "analytical_observation_count": 12,
                "residual_independent_cell_degrees_of_freedom": 1,
                "smallest_rank_singular_value": 0.5,
                "effective_condition_number": 4.0
            }
        ]
    )

    assert records
    assert all(
        record["analytical_observation_count_scope"]
        == (
            "all_eligible_positive_horizon_observations_"
            "from_support_region_member_cells"
        )
        for record in records
    )


# Confirming cycle elapsed-time scope is not labeled as FEC
def test_cycle_elapsed_time_scope_is_not_labeled_as_fec() -> None:
    records = build_analytical_scope_table(
        identifiability_records = [
            {
                "record_type": "term",
                "analysis_population": "bracketing_range_cell_design",
                "aging_mode": "cycle_aging",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined_within_scope",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "analysis_role": "within_cell_bracketing_support",
                "specification": "main_effects",
                "context_family": CONTEXT_FAMILY_NONE,
                "context_term_count": 0,
                "term_name": "ambient_temperature_celsius",
                "term_type": "main_effect",
                "structurally_estimable": True,
                "support_region_identity": "region_1",
                "support_region_kind": "interval",
                "support_interval_minimum": 10.0,
                "support_interval_maximum": 20.0,
                "support_interval_lower_inclusive": False,
                "support_interval_upper_inclusive": False,
                "condition_protocol_count": 2,
                "independent_cell_count": 6,
                "analytical_observation_count": 12,
                "residual_independent_cell_degrees_of_freedom": 1,
                "smallest_rank_singular_value": 0.5,
                "effective_condition_number": 4.0
            },
            {
                "record_type": "term",
                "analysis_population": "bracketing_range_cell_design",
                "aging_mode": "cycle_aging",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined_within_scope",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": (
                    "cumulative_full_equivalent_cycles_observed"
                ),
                "analysis_role": "within_cell_bracketing_support",
                "specification": "main_effects",
                "context_family": CONTEXT_FAMILY_NONE,
                "context_term_count": 0,
                "term_name": "ambient_temperature_celsius",
                "term_type": "main_effect",
                "structurally_estimable": True,
                "support_region_identity": "region_2",
                "support_region_kind": "interval",
                "support_interval_minimum": 100.0,
                "support_interval_maximum": 250.0,
                "support_interval_lower_inclusive": False,
                "support_interval_upper_inclusive": False,
                "condition_protocol_count": 2,
                "independent_cell_count": 6,
                "analytical_observation_count": 12,
                "residual_independent_cell_degrees_of_freedom": 1,
                "smallest_rank_singular_value": 0.5,
                "effective_condition_number": 4.0
            },
            {
                "record_type": "term",
                "analysis_population": "bracketing_range_cell_design",
                "aging_mode": "calendar_aging",
                "design_scope": "Stage_1",
                "sampling_design_scope": "combined_within_scope",
                "outcome": "relative_capacity",
                "cohort": "primary",
                "horizon_axis": "elapsed_calendar_days",
                "analysis_role": "within_cell_bracketing_support",
                "specification": "main_effects",
                "context_family": CONTEXT_FAMILY_NONE,
                "context_term_count": 0,
                "term_name": "ambient_temperature_celsius",
                "term_type": "main_effect",
                "structurally_estimable": True,
                "support_region_identity": "region_3",
                "support_region_kind": "interval",
                "support_interval_minimum": 10.0,
                "support_interval_maximum": 200.0,
                "support_interval_lower_inclusive": False,
                "support_interval_upper_inclusive": False,
                "condition_protocol_count": 2,
                "independent_cell_count": 6,
                "analytical_observation_count": 12,
                "residual_independent_cell_degrees_of_freedom": 1,
                "smallest_rank_singular_value": 0.5,
                "effective_condition_number": 4.0
            }
        ]
    )

    interpretations = {
        (
            record["aging_mode"],
            record["horizon_axis"]
        ): record["exposure_time_interpretation"]
        for record in records
    }

    assert interpretations[
        (
            "calendar_aging",
            "elapsed_calendar_days"
        )
    ] == (
        "whole_assigned_protocol_gross_elapsed_"
        "calendar_time_not_net_storage_exposure"
    )

    assert interpretations[
        (
            "cycle_aging",
            "elapsed_calendar_days"
        )
    ] == (
        "cycle_aging_elapsed_calendar_days_"
        "not_cumulative_fec"
    )

    assert interpretations[
        (
            "cycle_aging",
            "cumulative_full_equivalent_cycles_observed"
        )
    ] == "complete_cumulative_fec_only"


# Confirming the staging directory helper returns unique paths
def test_create_staging_output_directory_is_unique(tmp_path: Path) -> None:
    from scripts.audit_experimental_design import (
        create_staging_output_directory
    )

    output_directory = tmp_path / "design_output"

    first = create_staging_output_directory(output_directory)
    second = create_staging_output_directory(output_directory)

    try:
        assert first != second
        assert first.exists()
        assert second.exists()
    finally:
        import shutil as _shutil
        if first.exists():
            _shutil.rmtree(first)
        if second.exists():
            _shutil.rmtree(second)


# Confirming the returned report path is the final published path
def test_write_and_publish_returns_final_report_path(tmp_path: Path) -> None:
    from scripts.audit_experimental_design import (
        write_and_publish_design_artifacts
    )

    output_directory = tmp_path / "design_output"

    output_tables = {
        "demo_table": [
            {"key": "value_1", "value": 1},
            {"key": "value_2", "value": 2}
        ]
    }

    report_without_tables = {
        "audit_status": "completed",
        "dataset": {"figshare_article_id": 25975315}
    }

    published_report_path = (
        write_and_publish_design_artifacts(
            output_directory = output_directory,
            output_tables = output_tables,
            report_without_tables = report_without_tables
        )
    )

    assert published_report_path == (
        output_directory
        / "experimental_design_report.json"
    )
    assert published_report_path.exists()
    assert not any(
        entry.name.startswith(
            output_directory.name + ".staging."
        )
        for entry in output_directory.parent.iterdir()
    )
    assert not (
        output_directory.with_name(
            output_directory.name + ".backup"
        )
    ).exists()


# Confirming a mid-write failure leaves the previous output untouched
def test_write_failure_preserves_previous_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.audit_experimental_design import (
        write_and_publish_design_artifacts
    )
    import scripts.audit_experimental_design as module

    output_directory = tmp_path / "design_output"
    output_directory.mkdir()
    sentinel_path = output_directory / "sentinel.txt"
    sentinel_path.write_text("previous")

    def fail_write(output_path, records):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(
        module,
        "write_report_table",
        fail_write
    )

    with pytest.raises(RuntimeError, match = "simulated write failure"):
        write_and_publish_design_artifacts(
            output_directory = output_directory,
            output_tables = {
                "demo_table": [{"key": "value_1"}]
            },
            report_without_tables = {
                "audit_status": "completed"
            }
        )

    assert sentinel_path.exists()
    assert sentinel_path.read_text() == "previous"
    assert not any(
        entry.name.startswith(
            output_directory.name + ".staging."
        )
        for entry in output_directory.parent.iterdir()
    )


# Confirming joint compaction merges only when both cohort signatures are unchanged
def test_joint_compaction_merges_only_on_shared_signature() -> None:
    cell_records = [
        {
            "cell_identity": "cell_1",
            "condition_protocol_identity": "condition_1"
        }
    ]

    range_records = [
        {
            "cell_identity": "cell_1",
            "condition_protocol_identity": "condition_1",
            "aging_mode": "calendar_aging",
            "dataset_stage": "Stage_1",
            "laboratory": "A",
            "sampling_method": "design",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "supporting_cell_eligible_observation_count": 2,
            "eligible_horizon_minimum": 10.0,
            "eligible_horizon_maximum": 50.0,
            "eligible_horizons": [10.0, 50.0]
        }
    ]

    atomic_regions, atomic_memberships = build_continuous_support_regions(
        cell_records = cell_records,
        cell_horizon_range_records = range_records,
        coverage_rules = {
            "primary": {
                "minimum_assigned_cell_coverage_fraction": 1.0
            },
            "sensitivity": {
                "minimum_assigned_cell_coverage_fraction": 2 / 3
            }
        }
    )

    compact_regions, compact_memberships = jointly_compact_support_regions(
        cell_horizon_range_records = range_records,
        atomic_region_records = atomic_regions,
        atomic_membership_records = atomic_memberships
    )

    assert compact_regions
    assert compact_memberships

    assert all(
        region["region_kind"] in {"exact_point", "interval"}
        for region in compact_regions
    )
    assert all(
        "combined_region_identity" in region
        for region in compact_regions
    )
    assert all(
        "combined_region_identity" in membership
        for membership in compact_memberships
    )


# Confirming joint geometry alignment validator accepts jointly compacted regions
def test_validate_cohort_aligned_support_regions_accepts_joint_compaction() -> None:
    cell_records = [
        {
            "cell_identity": "cell_1",
            "condition_protocol_identity": "condition_1"
        }
    ]

    range_records = [
        {
            "cell_identity": "cell_1",
            "condition_protocol_identity": "condition_1",
            "aging_mode": "calendar_aging",
            "dataset_stage": "Stage_1",
            "laboratory": "A",
            "sampling_method": "design",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days",
            "supporting_cell_eligible_observation_count": 2,
            "eligible_horizon_minimum": 10.0,
            "eligible_horizon_maximum": 50.0,
            "eligible_horizons": [10.0, 50.0]
        }
    ]

    atomic_regions, atomic_memberships = build_continuous_support_regions(
        cell_records = cell_records,
        cell_horizon_range_records = range_records,
        coverage_rules = {
            "primary": {
                "minimum_assigned_cell_coverage_fraction": 1.0
            },
            "sensitivity": {
                "minimum_assigned_cell_coverage_fraction": 2 / 3
            }
        }
    )

    compact_regions, _ = jointly_compact_support_regions(
        cell_horizon_range_records = range_records,
        atomic_region_records = atomic_regions,
        atomic_membership_records = atomic_memberships
    )

    validate_cohort_aligned_support_regions(
        compact_regions
    )


# Confirming the whole-protocol calendar estimand is available with restrictions
def test_calendar_estimand_is_available_as_whole_protocol_target() -> None:
    from scripts.audit_experimental_design import (
        load_analysis_policy
    )

    policy_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experimental_design_policy.json"
    )

    policy = load_analysis_policy(
        policy_path = policy_path
    )

    calendar_policy = policy["calendar_exposure_policy"]

    assert (
        calendar_policy["primary_estimand_availability"]
        == "available"
    )
    assert (
        calendar_policy["primary_model_authorization"]
        == "permitted_with_restrictions"
    )
    assert (
        calendar_policy["net_storage_exposure_availability"]
        == "unavailable"
    )
    assert (
        calendar_policy[
            "net_storage_exposure_model_authorization"
        ]
        == "not_authorized"
    )


# Confirming joint-condition contrasts do not authorize one-factor contrast structures
def test_joint_condition_contrast_does_not_authorize_one_factor_effect() -> None:
    from scripts.audit_experimental_design import (
        build_question_disposition
    )

    disposition = build_question_disposition(
        {
            "permitted_bracketing_ranges": [
                {
                    "minimum": 10.0,
                    "maximum": 100.0
                }
            ],
            "first_eligible_cell_count": 3,
            "second_eligible_cell_count": 3,
            "direct_contrast_support_status": (
                "supported_with_restrictions"
            ),
            "target_estimand_available": True,
            "changed_treatments": [
                "ambient_temperature_celsius",
                "maximum_state_of_charge_fraction"
            ]
        }
    )

    assert (
        disposition[
            "condition_comparison_interpretation"
        ]
        == "joint_treatment_combination_contrast"
    )

    assert (
        disposition[
            "joint_condition_comparison_status"
        ]
        == "permitted_with_restrictions"
    )

    assert (
        disposition[
            "one_factor_contrast_structure_status"
        ]
        == "not_supported"
    )


# Confirming one-factor contrast structure requires no equipment restriction
def test_one_factor_contrast_structure_requires_no_equipment_restriction() -> None:
    from scripts.audit_experimental_design import (
        build_question_disposition
    )

    disposition_with_alias = build_question_disposition(
        {
            "permitted_bracketing_ranges": [
                {"minimum": 10.0, "maximum": 100.0}
            ],
            "first_eligible_cell_count": 3,
            "second_eligible_cell_count": 3,
            "direct_contrast_support_status": (
                "supported_with_restrictions"
            ),
            "target_estimand_available": True,
            "changed_treatments": [
                "ambient_temperature_celsius"
            ],
            "equipment_disjoint_dimension_count": 1,
            "equipment_identity_conflicted_dimension_count": 0
        }
    )

    assert (
        disposition_with_alias[
            "joint_condition_comparison_status"
        ] == "permitted_with_restrictions"
    )

    assert (
        disposition_with_alias[
            "one_factor_contrast_structure_status"
        ] == "not_supported"
    )

    disposition_with_conflict = build_question_disposition(
        {
            "permitted_bracketing_ranges": [
                {"minimum": 10.0, "maximum": 100.0}
            ],
            "first_eligible_cell_count": 3,
            "second_eligible_cell_count": 3,
            "direct_contrast_support_status": (
                "supported_with_restrictions"
            ),
            "target_estimand_available": True,
            "changed_treatments": [
                "ambient_temperature_celsius"
            ],
            "equipment_disjoint_dimension_count": 0,
            "equipment_identity_conflicted_dimension_count": 1
        }
    )

    assert (
        disposition_with_conflict[
            "joint_condition_comparison_status"
        ] == "permitted_with_restrictions"
    )

    assert (
        disposition_with_conflict[
            "one_factor_contrast_structure_status"
        ] == "not_supported"
    )

    disposition_without_equipment_restriction = (
        build_question_disposition(
            {
                "permitted_bracketing_ranges": [
                    {"minimum": 10.0, "maximum": 100.0}
                ],
                "first_eligible_cell_count": 3,
                "second_eligible_cell_count": 3,
                "direct_contrast_support_status": (
                    "supported_with_restrictions"
                ),
                "target_estimand_available": True,
                "changed_treatments": [
                    "ambient_temperature_celsius"
                ],
                "equipment_disjoint_dimension_count": 0,
                "equipment_identity_conflicted_dimension_count": 0
            }
        )
    )

    assert (
        disposition_without_equipment_restriction[
            "joint_condition_comparison_status"
        ] == "permitted_with_restrictions"
    )

    assert (
        disposition_without_equipment_restriction[
            "one_factor_contrast_structure_status"
        ] == "permitted_with_restrictions"
    )


# Confirming condition counts accept test-point protocol lists
def test_condition_count_accepts_test_point_protocol_lists() -> None:
    from scripts.audit_experimental_design import (
        collect_condition_protocol_identities
    )

    identities = collect_condition_protocol_identities(
        [
            {
                "condition_protocol_identities": [
                    "condition_1",
                    "condition_2"
                ]
            },
            {
                "condition_protocol_identities": [
                    "condition_2",
                    "condition_3"
                ]
            }
        ]
    )

    assert identities == {
        "condition_1",
        "condition_2",
        "condition_3"
    }


# Rejecting invalid condition-protocol plural fields
def test_condition_count_rejects_invalid_plural_field() -> None:
    from scripts.audit_experimental_design import (
        collect_condition_protocol_identities
    )

    with pytest.raises(
        RuntimeError,
        match = "invalid condition_protocol_identities"
    ):
        collect_condition_protocol_identities(
            [
                {
                    "condition_protocol_identities": [
                        ""
                    ]
                }
            ]
        )


# Confirming direct supported-record selection and question counting
def test_direct_supported_record_selection_and_question_count() -> None:
    from scripts.audit_experimental_design import (
        direct_supported_records,
        direct_supported_question_count
    )

    records = [
        {
            "aging_mode": "calendar_aging",
            "final_support_classification": (
                "supported_with_restrictions"
            ),
            "first_condition_protocol_identity": "condition_a",
            "second_condition_protocol_identity": "condition_b",
            "outcome": "relative_capacity",
            "horizon_axis": "elapsed_calendar_days"
        },
        {
            "aging_mode": "calendar_aging",
            "final_support_classification": "sensitivity_only",
            "first_condition_protocol_identity": "condition_a",
            "second_condition_protocol_identity": "condition_b",
            "outcome": "relative_capacity",
            "horizon_axis": "elapsed_calendar_days"
        },
        {
            "aging_mode": "calendar_aging",
            "final_support_classification": "descriptive_only",
            "first_condition_protocol_identity": "condition_c",
            "second_condition_protocol_identity": "condition_d",
            "outcome": "relative_capacity",
            "horizon_axis": "elapsed_calendar_days"
        },
        {
            "aging_mode": "cycle_aging",
            "final_support_classification": (
                "supported_with_restrictions"
            ),
            "first_condition_protocol_identity": "condition_e",
            "second_condition_protocol_identity": "condition_f",
            "outcome": "relative_dc_internal_resistance",
            "horizon_axis": (
                "cumulative_full_equivalent_cycles_observed"
            )
        }
    ]

    calendar_supported = direct_supported_records(
        records = records,
        aging_mode = "calendar_aging"
    )

    cycle_supported = direct_supported_records(
        records = records,
        aging_mode = "cycle_aging"
    )

    assert len(calendar_supported) == 2
    assert len(cycle_supported) == 1

    assert direct_supported_question_count(
        calendar_supported
    ) == 1

    assert direct_supported_question_count(
        cycle_supported
    ) == 1


# Confirming that delivery-state characteristics enter pairwise baseline diagnostics
def test_baseline_balance_includes_pre_treatment_cell_characteristics() -> None:
    from scripts.audit_experimental_design import (
        CELL_LEVEL_BASELINE_FIELDS,
        build_baseline_balance_diagnostics
    )

    assert "initial_cell_mass_grams" in CELL_LEVEL_BASELINE_FIELDS
    assert (
        "initial_1khz_resistance_source_value"
        in CELL_LEVEL_BASELINE_FIELDS
    )
    assert "initial_voltage_volts" in CELL_LEVEL_BASELINE_FIELDS

    cell_records = [
        {
            "cell_identity": "Stage_1/TP_k01_01",
            "condition_protocol_identity": (
                "Stage_1/TP_k01/periodic_reference_tests"
            ),
            "dataset_stage": "Stage_1",
            "aging_type": "k",
            "aging_mode": "calendar_aging",
            "test_point_number": "1",
            "laboratory": "SIE",
            "sampling_method": "FF",
            "initial_cell_mass_grams": "45.1",
            "initial_1khz_resistance_source_value": "13.44",
            "initial_voltage_volts": "3.61"
        },
        {
            "cell_identity": "Stage_1/TP_k02_01",
            "condition_protocol_identity": (
                "Stage_1/TP_k02/periodic_reference_tests"
            ),
            "dataset_stage": "Stage_1",
            "aging_type": "k",
            "aging_mode": "calendar_aging",
            "test_point_number": "2",
            "laboratory": "SIE",
            "sampling_method": "FF",
            "initial_cell_mass_grams": "45.2",
            "initial_1khz_resistance_source_value": "13.55",
            "initial_voltage_volts": "3.62"
        }
    ]

    reference_records = [
        {
            "cell_identity": "Stage_1/TP_k01_01",
            "sequence_number": "1",
            "mean_capacity_ah": "4.9",
            "uncorrected_dc_internal_resistance_milliohms": "22.5",
            "baseline_voltage_volts": "3.6"
        },
        {
            "cell_identity": "Stage_1/TP_k02_01",
            "sequence_number": "1",
            "mean_capacity_ah": "4.95",
            "uncorrected_dc_internal_resistance_milliohms": "22.6",
            "baseline_voltage_volts": "3.6"
        }
    ]

    diagnostic_records = build_baseline_balance_diagnostics(
        cell_records = cell_records,
        reference_records = reference_records
    )

    observed_baseline_fields = {
        record["baseline_field"]
        for record in diagnostic_records
    }

    assert "initial_cell_mass_grams" in observed_baseline_fields
    assert (
        "initial_1khz_resistance_source_value"
        in observed_baseline_fields
    )
    assert "initial_voltage_volts" in observed_baseline_fields

    pairwise_baseline_fields = {
        record["baseline_field"]
        for record in diagnostic_records
        if record["record_type"] == "condition_pair_difference"
    }

    assert {
        "initial_cell_mass_grams",
        "initial_1khz_resistance_source_value",
        "initial_voltage_volts"
    }.issubset(
        pairwise_baseline_fields
    )


# Confirming the policy declares one-factor contrast structure and required semantics
def test_analysis_policy_declares_one_factor_contrast_semantics() -> None:
    from scripts.audit_experimental_design import (
        load_analysis_policy
    )

    policy_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "experimental_design_policy.json"
    )

    policy = load_analysis_policy(
        policy_path = policy_path
    )

    treatment_policy = policy[
        "treatment_evaluation_policy"
    ]

    assert (
        "one_factor_contrast_structure"
        in treatment_policy
    )

    assert (
        "one_factor_contrast_identification"
        in treatment_policy
    )

    assert (
        treatment_policy[
            "one_factor_contrast_identification"
        ]
        == "not_established_assignment_mechanism_undocumented"
    )

    longitudinal_policy = policy[
        "longitudinal_observation_policy"
    ]

    assert (
        longitudinal_policy["discrete_horizon_status"]
        == "not_authorized_without_outcome_blind_pre_specification"
    )


# Confirming that disjoint delivery-state equipment is reported as such
def test_baseline_balance_reports_disjoint_measurement_equipment() -> None:
    from scripts.audit_experimental_design import (
        build_baseline_balance_diagnostics
    )

    cell_records = [
        {
            "cell_identity": "Stage_1/TP_k01_01",
            "condition_protocol_identity": (
                "Stage_1/TP_k01/periodic_reference_tests"
            ),
            "dataset_stage": "Stage_1",
            "aging_type": "k",
            "aging_mode": "calendar_aging",
            "test_point_number": "1",
            "laboratory": "SIE",
            "sampling_method": "FF",
            "initial_cell_mass_grams": "45.1",
            "initial_cell_mass_scale": "scale_a",
            "initial_1khz_resistance_source_value": "13.44",
            "initial_1khz_resistance_meter": "meter_a",
            "initial_voltage_volts": "3.61",
            "initial_voltage_meter": "volt_meter_a"
        },
        {
            "cell_identity": "Stage_1/TP_k02_01",
            "condition_protocol_identity": (
                "Stage_1/TP_k02/periodic_reference_tests"
            ),
            "dataset_stage": "Stage_1",
            "aging_type": "k",
            "aging_mode": "calendar_aging",
            "test_point_number": "2",
            "laboratory": "SIE",
            "sampling_method": "FF",
            "initial_cell_mass_grams": "45.2",
            "initial_cell_mass_scale": "scale_a",
            "initial_1khz_resistance_source_value": "13.55",
            "initial_1khz_resistance_meter": "meter_b",
            "initial_voltage_volts": "3.62",
            "initial_voltage_meter": "volt_meter_b"
        }
    ]

    reference_records = [
        {
            "cell_identity": "Stage_1/TP_k01_01",
            "sequence_number": "1",
            "mean_capacity_ah": "4.90",
            "uncorrected_dc_internal_resistance_milliohms": "22.5",
            "baseline_voltage_volts": "3.6"
        },
        {
            "cell_identity": "Stage_1/TP_k02_01",
            "sequence_number": "1",
            "mean_capacity_ah": "4.95",
            "uncorrected_dc_internal_resistance_milliohms": "22.6",
            "baseline_voltage_volts": "3.6"
        }
    ]

    diagnostic_records = build_baseline_balance_diagnostics(
        cell_records = cell_records,
        reference_records = reference_records
    )

    resistance_pair = next(
        record
        for record in diagnostic_records
        if (
            record["record_type"] == "condition_pair_difference"
            and record["baseline_field"]
            == "initial_1khz_resistance_source_value"
            and record["analysis_population"] == "assigned"
        )
    )

    assert resistance_pair[
        "measurement_equipment_field"
    ] == "initial_1khz_resistance_meter"

    assert resistance_pair[
        "measurement_equipment_relationship"
    ] == "disjoint"

    assert resistance_pair[
        "first_measurement_equipment_values"
    ] == ["meter_a"]

    assert resistance_pair[
        "second_measurement_equipment_values"
    ] == ["meter_b"]
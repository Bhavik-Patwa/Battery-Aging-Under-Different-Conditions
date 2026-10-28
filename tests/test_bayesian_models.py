import json
import shutil
import hashlib
import pytest
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import inspect

from pathlib import Path
from unittest.mock import patch as mock_patch

import scripts.run_bayesian_models as bayesian_models

from scripts.run_bayesian_models import (
    build_candidate_structure_diagnostics,
    build_execution_plan,
    build_fixed_location_matrix,
    build_fold_assignments,
    calculate_matrix_rank,
    canonical_json,
    deterministic_identity,
    load_json_object,
    reconstruct_source_candidate,
    select_source_candidates,
    serialize_csv_value,
    write_csv_table
)

PROJECT_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "bayesian_modeling_policy.json"
)


@pytest.fixture(
    scope = "module"
)
def published_validation_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    project_root = Path(
        __file__
    ).resolve().parents[1]

    validated_inputs = (
        bayesian_models.load_validated_inputs(
            feature_manifest_path = (
                project_root
                / "data"
                / "processed"
                / "feature_table_manifest.json"
            ),
            design_report_path = (
                project_root
                / "data"
                / "processed"
                / "experimental_design"
                / "experimental_design_report.json"
            ),
            identification_report_path = (
                project_root
                / "data"
                / "processed"
                / "causal_identification"
                / "causal_identification_report.json"
            ),
            descriptive_report_path = (
                project_root
                / "data"
                / "processed"
                / "descriptive_models"
                / "descriptive_modeling_report.json"
            ),
            modeling_policy_path = (
                PROJECT_POLICY_PATH
            )
        )
    )

    workflow = (
        bayesian_models.build_validation_workflow(
            validated_inputs
        )
    )

    output_directory = (
        tmp_path_factory.mktemp(
            "bayesian_validation"
        )
        / "artifacts"
    )

    bayesian_models.publish_validation_artifacts(
        workflow = workflow,
        output_directory = output_directory
    )

    return output_directory


@pytest.fixture(
    scope = "module"
)
def published_prior_predictive_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    project_root = Path(
        __file__
    ).resolve().parents[1]

    validated_inputs = (
        bayesian_models.load_validated_inputs(
            feature_manifest_path = (
                project_root
                / "data"
                / "processed"
                / "feature_table_manifest.json"
            ),
            design_report_path = (
                project_root
                / "data"
                / "processed"
                / "experimental_design"
                / "experimental_design_report.json"
            ),
            identification_report_path = (
                project_root
                / "data"
                / "processed"
                / "causal_identification"
                / "causal_identification_report.json"
            ),
            descriptive_report_path = (
                project_root
                / "data"
                / "processed"
                / "descriptive_models"
                / "descriptive_modeling_report.json"
            ),
            modeling_policy_path = (
                PROJECT_POLICY_PATH
            )
        )
    )

    workflow = (
        bayesian_models.build_validation_workflow(
            validated_inputs
        )
    )

    policy = workflow[
        "validated_inputs"
    ][
        "policy"
    ]

    admissible_candidates = [
        candidate_record
        for candidate_record in (
            workflow[
                "candidate_records"
            ]
        )
        if candidate_record[
            "admissibility_status"
        ]
        == "admissible"
    ]

    if not admissible_candidates:
        pytest.skip(
            "No admissible candidate exists in the current validation workflow."
        )

    first_candidate = admissible_candidates[
        0
    ]

    first_multiplier = (
        policy[
            "priors"
        ][
            "prior_sensitivity_scale_multipliers"
        ][
            0
        ]
    )

    synthetic_diagnostics = [
        {
            "candidate_identity": (
                first_candidate[
                    "candidate_identity"
                ]
            ),
            "outcome": first_candidate[
                "outcome"
            ],
            "prior_scale_multiplier": (
                first_multiplier
            ),
            "diagnostic_population": (
                "assembled_observations"
            ),
            "condition_protocol_identity": None,
            "evaluation_point_count": 1,
            "predictive_draw_count": 1,
            "predictive_value_count": 1,
            "execution_status": "completed",
            "execution_reason_code": (
                "PRIOR_PREDICTIVE_COMPLETED"
            ),
            "execution_error": None,
            "log_scale_median": 0.0,
            "log_scale_median_absolute_deviation": (
                0.0
            ),
            "log_scale_interquartile_range": (
                0.0
            ),
            "log_scale_minimum": 0.0,
            "log_scale_maximum": 0.0,
            "log_scale_quantiles": {
                "0.01": 0.0,
                "0.05": 0.0,
                "0.25": 0.0,
                "0.50": 0.0,
                "0.75": 0.0,
                "0.95": 0.0,
                "0.99": 0.0
            },
            "original_scale_finite_fraction": (
                1.0
            ),
            "original_scale_nonfinite_count": (
                0
            ),
            "original_scale_strictly_positive_finite_fraction": (
                1.0
            ),
            "original_scale_finite_quantiles": {
                "0.01": 1.0,
                "0.05": 1.0,
                "0.25": 1.0,
                "0.50": 1.0,
                "0.75": 1.0,
                "0.95": 1.0,
                "0.99": 1.0
            }
        }
    ]

    synthetic_accounting = {
        "requested_call_count": 1,
        "completed_call_count": 1,
        "failed_call_count": 0,
        "reused_cache_count": 0,
        "new_cache_count": 1
    }

    output_directory = (
        tmp_path_factory.mktemp(
            "bayesian_prior_predictive"
        )
        / "artifacts"
    )

    cache_directory = (
        tmp_path_factory.mktemp(
            "bayesian_prior_predictive_cache"
        )
    )

    with mock_patch.object(
        bayesian_models,
        "execute_prior_predictive_sampling",
        return_value = (
            synthetic_diagnostics,
            synthetic_accounting
        )
    ):
        bayesian_models.publish_prior_predictive_artifacts(
            workflow = workflow,
            output_directory = (
                output_directory
            ),
            cache_directory = cache_directory,
            software_environment_record = {
                "fake_environment": True
            },
            software_environment_sha256 = (
                "0"
                * 64
            )
        )

    return output_directory


# Building a small policy shim for tests that do not require full policy validation
def build_policy_fixture() -> dict:
    return {
        "policy_version": "1.9",
        "candidate_inventory": {
            "required_selection_status": "selected",
            "representative_template": "linear_common_trajectory",
            "required_context_family": "no_additional_context",
            "representative_estimator": "weighted_least_squares",
            "deduplication_fields": [
                "support_region_identity", "design_scope", "sampling_design_scope",
                "aging_mode", "outcome", "cohort", "horizon_axis"
            ],
            "duplicate_source_required_identical_fields": [
                "support_region_identity", "design_scope", "sampling_design_scope",
                "aging_mode", "outcome", "cohort", "horizon_axis",
                "support_interval_minimum", "support_interval_maximum",
                "support_interval_lower_inclusive", "support_interval_upper_inclusive",
                "analysis_population"
            ],
            "authorization_restriction_order": [
                "permitted_for_model_evaluation_with_restrictions",
                "permitted_for_model_evaluation_sensitivity_only"
            ]
        },
        "priors": {
            "prior_sensitivity_scale_multipliers": [0.5, 1.0, 2.0]
        },
        "sampling": {
            "full_data_posterior_scale_multipliers": [0.5, 1.0, 2.0],
            "grouped_validation_scale_multipliers": [1.0]
        }
    }


# Building a candidate from condition-to-(cell, horizon) tuples
def build_candidate_with_conditions(conditions: dict) -> dict:
    included_observations = []
    for condition_identity, cell_horizon_pairs in conditions.items():
        for cell_identity, horizon in cell_horizon_pairs:
            included_observations.append({
                "cell_identity": cell_identity,
                "condition_protocol_identity": condition_identity,
                "horizon": horizon,
                "outcome_value": 1.0,
                "inside_evaluation_interval": True
            })
    return {"included_observations": included_observations}


# Building one minimal source candidate
def build_source_candidate() -> dict:
    return {
        "candidate_identity": "source-candidate-1",
        "support_region_identity": "region-1",
        "design_scope": "Stage_1",
        "sampling_design_scope": "combined_within_scope",
        "aging_mode": "calendar_aging",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": "elapsed_calendar_days",
        "support_interval_minimum": "10.0",
        "support_interval_maximum": "20.0",
        "support_interval_lower_inclusive": "true",
        "support_interval_upper_inclusive": "true"
    }


# Building validated inputs whose only observation has a non-finite outcome
def build_validated_inputs_with_nonfinite_outcome() -> dict:
    return {
        "policy": {
            "outcomes": {
                "relative_capacity": {
                    "source_field": "relative_capacity",
                    "primary_eligibility_field": "relative_capacity_primary_eligible",
                    "sensitivity_eligibility_field": "relative_capacity_sensitivity_eligible"
                }
            }
        },
        "feature_tables": {
            "reference_test_features": [
                {
                    "source_identity": "src-1",
                    "cell_identity": "cell-1",
                    "elapsed_calendar_days": "10.0",
                    "relative_capacity": "nan",
                    "relative_capacity_primary_eligible": "true",
                    "cumulative_fec_observation_complete": ""
                }
            ],
            "cell_features": [
                {
                    "cell_identity": "cell-1",
                    "condition_protocol_identity": "condition-a",
                    "dataset_stage": "Stage_1",
                    "sampling_method": "FF"
                }
            ]
        },
        "design_tables": {
            "bracketing_support_membership": [
                {
                    "support_region_identity": "region-1",
                    "cell_identity": "cell-1",
                    "aging_mode": "calendar_aging",
                    "outcome": "relative_capacity",
                    "cohort": "primary",
                    "horizon_axis": "elapsed_calendar_days",
                    "lower_horizon": "10.0",
                    "upper_horizon": "20.0",
                    "lower_inclusive": "true",
                    "upper_inclusive": "true",
                    "condition_protocol_identity": "condition-a",
                    "dataset_stage": "Stage_1",
                    "sampling_method": "FF"
                }
            ]
        }
    }


# Rejecting duplicate JSON fields
def test_load_json_object_rejects_duplicate_field(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text('{"policy_version":"1.8","policy_version":"1.9"}', encoding = "utf-8")

    with pytest.raises(RuntimeError, match = "duplicate field"):
        load_json_object(path)


# Confirming canonical JSON is order-independent
def test_canonical_json_is_order_independent() -> None:
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


# Confirming deterministic identity is stable
def test_deterministic_identity_is_stable() -> None:
    value = {"a": 1, "b": 2}
    assert deterministic_identity(value) == deterministic_identity(value)
    assert len(deterministic_identity(value)) == 64


# Confirming selection matches the registered filter
def test_select_source_candidates_matches_policy_filter() -> None:
    policy = build_policy_fixture()
    records = [
        {
            "candidate_identity": "match",
            "selection_status": "selected",
            "template_name": "linear_common_trajectory",
            "context_family": "no_additional_context",
            "estimator": "weighted_least_squares"
        },
        {
            "candidate_identity": "wrong-template",
            "selection_status": "selected",
            "template_name": "linear_differential_trajectory",
            "context_family": "no_additional_context",
            "estimator": "weighted_least_squares"
        },
        {
            "candidate_identity": "wrong-context",
            "selection_status": "selected",
            "template_name": "linear_common_trajectory",
            "context_family": "laboratory",
            "estimator": "weighted_least_squares"
        },
        {
            "candidate_identity": "not-selected",
            "selection_status": "not_selected",
            "template_name": "linear_common_trajectory",
            "context_family": "no_additional_context",
            "estimator": "weighted_least_squares"
        }
    ]

    selected = select_source_candidates(records, policy)

    assert [record["candidate_identity"] for record in selected] == ["match"]


# Confirming selection never consults outcome direction, admissibility or performance
def test_selection_ignores_outcome_and_performance() -> None:
    policy = build_policy_fixture()
    records = [
        {
            "candidate_identity": "selected-inadmissible",
            "selection_status": "selected",
            "template_name": "linear_common_trajectory",
            "context_family": "no_additional_context",
            "estimator": "weighted_least_squares",
            "admissibility_status": "inadmissible",
            "fit_status": "fit_failed",
            "validation_status": "unavailable",
            "model_adequacy_status": "diagnostic_only",
            "analytical_result_status": "not_available",
            "predictive_benchmark_status": "not_met"
        }
    ]

    selected = select_source_candidates(records, policy)

    assert len(selected) == 1
    assert selected[0]["candidate_identity"] == "selected-inadmissible"


# Confirming fold assignment groups cells across conditions without leakage
def test_fold_assignment_groups_cells_by_condition() -> None:
    assignments = build_fold_assignments(
        candidate_identity = "candidate",
        cells_by_condition = {
            "condition_a": {"cell_1", "cell_2", "cell_3", "cell_4"},
            "condition_b": {"cell_5", "cell_6", "cell_7", "cell_8"}
        },
        effective_fold_count = 3,
        seed = 17501
    )

    assert set(assignments) == {
        "cell_1", "cell_2", "cell_3", "cell_4",
        "cell_5", "cell_6", "cell_7", "cell_8"
    }

    assert all(fold in {0, 1, 2} for fold in assignments.values())


# Confirming the fold assignment is deterministic
def test_fold_assignment_is_deterministic() -> None:
    first = build_fold_assignments(
        "candidate",
        {"condition_a": {"cell_1", "cell_2", "cell_3"}, "condition_b": {"cell_4", "cell_5", "cell_6"}},
        3,
        17501
    )
    second = build_fold_assignments(
        "candidate",
        {"condition_a": {"cell_3", "cell_1", "cell_2"}, "condition_b": {"cell_6", "cell_4", "cell_5"}},
        3,
        17501
    )

    assert first == second


# Rejecting a cell that appears in two conditions
def test_fold_assignment_rejects_multi_condition_cell() -> None:
    with pytest.raises(RuntimeError, match = "appears in multiple modeled conditions"):
        build_fold_assignments(
            "candidate",
            {"condition_a": {"cell_1"}, "condition_b": {"cell_1"}},
            2,
            17501
        )


# Confirming the fixed-location matrix has the expected shape and full rank
def test_fixed_location_matrix_requires_full_rank() -> None:
    observations = [
        {"condition_protocol_identity": "a", "horizon": 10.0},
        {"condition_protocol_identity": "a", "horizon": 30.0},
        {"condition_protocol_identity": "b", "horizon": 10.0},
        {"condition_protocol_identity": "b", "horizon": 30.0}
    ]

    matrix = build_fixed_location_matrix(
        observation_records = observations,
        condition_identities = ["a", "b"]
    )

    assert matrix.shape == (4, 4)

    rank, tolerance = calculate_matrix_rank(matrix)

    assert rank == 4
    assert tolerance > 0


# Confirming a rank-deficient matrix is detected
def test_fixed_location_matrix_rank_deficiency() -> None:
    observations = [
        {"condition_protocol_identity": "a", "horizon": 10.0},
        {"condition_protocol_identity": "a", "horizon": 10.0},
        {"condition_protocol_identity": "b", "horizon": 20.0},
        {"condition_protocol_identity": "b", "horizon": 20.0}
    ]

    matrix = build_fixed_location_matrix(
        observation_records = observations,
        condition_identities = ["a", "b"]
    )

    rank, tolerance = calculate_matrix_rank(matrix)

    assert rank < 4


# Confirming workload arithmetic matches the policy formulas
def test_execution_plan_uses_registered_formulas() -> None:
    policy = build_policy_fixture()

    candidate_records = [
        {
            "candidate_identity": "a",
            "admissibility_status": "admissible",
            "grouped_validation_status": "available"
        },
        {
            "candidate_identity": "b",
            "admissibility_status": "admissible",
            "grouped_validation_status": "available"
        }
    ]

    fold_records = [
        {"candidate_identity": "a", "fold_status": "available"}
        for _ in range(5)
    ] + [
        {"candidate_identity": "b", "fold_status": "available"}
        for _ in range(5)
    ]

    plan = build_execution_plan(candidate_records, fold_records, policy)

    assert plan["admissible_candidate_count"] == 2
    assert plan["validation_feasible_candidate_count"] == 2
    assert plan["prior_predictive_call_count"] == 6
    assert plan["full_data_posterior_call_count"] == 6
    assert plan["grouped_validation_posterior_call_count"] == 10
    assert plan["nuts_sampler_call_upper_bound"] == 16
    assert plan["posterior_predictive_forward_call_upper_bound"] == 16
    assert plan["all_stochastic_call_upper_bound"] == 38


# Confirming CSV serialization is deterministic
def test_serialize_csv_value() -> None:
    assert serialize_csv_value(None) == ""
    assert serialize_csv_value(True) == "true"
    assert serialize_csv_value(False) == "false"
    assert serialize_csv_value([1, 2]) == "[1,2]"
    assert serialize_csv_value({"b": 2, "a": 1}) == '{"a":1,"b":2}'


# Rejecting non-finite CSV numbers
def test_serialize_csv_value_rejects_nonfinite() -> None:
    with pytest.raises(RuntimeError, match = "non-finite"):
        serialize_csv_value(float("nan"))


# Confirming CSV writing is byte-stable across runs
def test_csv_writing_is_byte_stable(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"

    rows = [
        {"candidate_identity": "b", "value": 2.0},
        {"candidate_identity": "a", "value": 1.0}
    ]
    columns = ["candidate_identity", "value"]

    write_csv_table(first, columns, rows)
    write_csv_table(second, columns, rows)

    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(second.read_bytes()).hexdigest()


# Rejecting invalid row schema
def test_csv_writer_rejects_schema_mismatch(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match = "schema"):
        write_csv_table(
            tmp_path / "bad.csv",
            ["candidate_identity", "value"],
            [{"candidate_identity": "a", "unexpected": 1}]
        )


# Confirming the policy validator rejects missing sections
def test_validate_policy_rejects_missing_section() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    del policy["priors"]
    with pytest.raises(RuntimeError, match = "missing"):
        bayesian_models.validate_policy(policy)


# Confirming the policy validator rejects enabled causal language
def test_validate_policy_rejects_causal_language() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["authorization"]["causal_language_permitted"] = True
    with pytest.raises(RuntimeError, match = "causal_language_permitted"):
        bayesian_models.validate_policy(policy)


# Confirming the policy validator rejects a missing cell_features requirement
def test_validate_policy_rejects_missing_cell_features() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["inputs"]["required_feature_tables"] = ["reference_test_features"]
    with pytest.raises(RuntimeError, match = "cell_features"):
        bayesian_models.validate_policy(policy)


# Confirming the policy validator requires the admissible status
def test_validate_policy_requires_admissible_status() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["observation_contract"]["candidate_execution_statuses"].remove("admissible")
    with pytest.raises(RuntimeError, match = "admissible"):
        bayesian_models.validate_policy(policy)


# Confirming the repeated-cell diagnostic treats zero-repeated conditions as zero
def test_minimum_condition_repeated_count_includes_zero_conditions() -> None:
    candidate = build_candidate_with_conditions(
        conditions = {
            "a": [("c1", 10.0), ("c1", 20.0), ("c2", 10.0), ("c2", 20.0)],
            "b": [("c3", 10.0), ("c4", 20.0)]
        }
    )
    diagnostics = build_candidate_structure_diagnostics(candidate)
    assert diagnostics["minimum_condition_repeated_cell_count"] == 0


# Confirming a non-finite outcome is rejected during reconstruction
def test_reconstruction_rejects_nonfinite_outcome() -> None:
    with pytest.raises(RuntimeError, match = "non-finite outcome"):
        reconstruct_source_candidate(
            source_candidate = build_source_candidate(),
            validated_inputs = build_validated_inputs_with_nonfinite_outcome()
        )


# Confirming workload excludes folds from inadmissible candidates
def test_workload_excludes_inadmissible_folds() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    candidate_records = [
        {"candidate_identity": "adm", "admissibility_status": "admissible",
         "grouped_validation_status": "available"},
        {"candidate_identity": "inadm", "admissibility_status": "inadmissible",
         "grouped_validation_status": "unavailable"}
    ]
    fold_records = [
        {"candidate_identity": "adm", "fold_status": "available"},
        {"candidate_identity": "inadm", "fold_status": "available"}
    ]
    plan = build_execution_plan(candidate_records, fold_records, policy)
    assert plan["validation_feasible_candidate_count"] == 1
    assert plan["grouped_validation_posterior_call_count"] == 1


# Confirming report validation rejects a tampered embedded policy
def test_validate_bayesian_report_rejects_tampered_policy(tmp_path: Path, published_validation_directory: Path) -> None:
    report_path = (
        published_validation_directory
        / "bayesian_modeling_report.json"
    )

    original_report = bayesian_models.load_json_object(report_path)

    candidate_rows = list(
        bayesian_models.read_csv_rows(
            report_path.parent / "candidate_registry.csv"
        )[1]
    )
    accounting_rows = list(
        bayesian_models.read_csv_rows(
            report_path.parent / "observation_accounting.csv"
        )[1]
    )
    fold_rows = list(
        bayesian_models.read_csv_rows(
            report_path.parent / "grouped_validation_folds.csv"
        )[1]
    )

    policy = original_report["policy"]
    policy_sha256 = original_report["policy_sha256"]
    input_hashes = original_report["input_hashes"]
    execution_plan = original_report["execution_plan"]

    tampered = dict(original_report)
    tampered["policy"] = {
        **policy,
        "policy_version": "999.0",
    }

    tampered_path = tmp_path / "bayesian_modeling_report.json"
    tampered_path.write_text(
        json.dumps(tampered, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8",
    )

    with pytest.raises(RuntimeError, match = "embed the registered policy"):
        bayesian_models.validate_bayesian_report(
            report_path = tampered_path,
            expected_policy = policy,
            expected_policy_sha256 = policy_sha256,
            expected_input_hashes = input_hashes,
            expected_candidate_records = candidate_rows,
            expected_accounting_records = accounting_rows,
            expected_fold_records = fold_rows,
            expected_execution_plan = execution_plan,
            expected_operation = "validate",
        )


# Confirming report validation rejects a value-only table tamper
def test_validate_bayesian_report_rejects_value_tamper(tmp_path: Path, published_validation_directory: Path) -> None:
    source_directory = (
        published_validation_directory
    )

    report_path = source_directory / "bayesian_modeling_report.json"

    report = bayesian_models.load_json_object(report_path)

    candidate_registry_path = tmp_path / "candidate_registry.csv"
    observation_accounting_path = tmp_path / "observation_accounting.csv"
    grouped_validation_folds_path = tmp_path / "grouped_validation_folds.csv"

    for source_name, destination_path in (
        ("candidate_registry.csv", candidate_registry_path),
        ("observation_accounting.csv", observation_accounting_path),
        ("grouped_validation_folds.csv", grouped_validation_folds_path),
    ):
        shutil.copyfile(
            source_directory / source_name,
            destination_path,
        )

    tampered_report = dict(report)
    tampered_report["tables"] = {
        table_name: dict(table_record)
        for table_name, table_record in report["tables"].items()
    }

    tampered_path = tmp_path / "bayesian_modeling_report.json"
    tampered_path.write_text(
        json.dumps(tampered_report, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8",
    )

    candidate_rows = list(
        bayesian_models.read_csv_rows(candidate_registry_path)[1]
    )
    accounting_rows = list(
        bayesian_models.read_csv_rows(observation_accounting_path)[1]
    )
    fold_rows = list(
        bayesian_models.read_csv_rows(grouped_validation_folds_path)[1]
    )

    # Mutating one in-memory expected record must now be caught
    tampered_candidate_rows = [
        dict(row) for row in candidate_rows
    ]

    if not tampered_candidate_rows:
        pytest.skip(
            "Published Bayesian validation report contains no candidates."
        )

    tampered_candidate_rows[0]["admissibility_status"] = (
        "inadmissible"
        if tampered_candidate_rows[0]["admissibility_status"] == "admissible"
        else "admissible"
    )

    with pytest.raises(
        RuntimeError,
        match = "contents differ from the validated workflow",
    ):
        bayesian_models.validate_bayesian_report(
            report_path = tampered_path,
            expected_policy = report["policy"],
            expected_policy_sha256 = report["policy_sha256"],
            expected_input_hashes = report["input_hashes"],
            expected_candidate_records = tampered_candidate_rows,
            expected_accounting_records = accounting_rows,
            expected_fold_records = fold_rows,
            expected_execution_plan = report["execution_plan"],
            expected_operation = "validate",
        )


# Confirming post-publication validation failure restores the previous directory
def test_publish_validation_artifacts_rolls_back_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_directory = tmp_path / "published"
    output_directory.mkdir()
    sentinel_path = output_directory / "sentinel.txt"
    sentinel_path.write_text("previous", encoding = "utf-8")

    minimal_workflow = {
        "validated_inputs": {
            "policy": bayesian_models.load_json_object(
                PROJECT_POLICY_PATH
            ),
            "policy_sha256": "a" * 64,
            "feature_manifest_sha256": "b" * 64,
            "design_report_sha256": "c" * 64,
            "identification_report_sha256": "d" * 64,
            "descriptive_report_sha256": "e" * 64,
        },
        "candidate_records": [],
        "observation_accounting_records": [],
        "fold_records": [],
        "execution_plan": {
            "admissible_candidate_count": 0,
            "validation_feasible_candidate_count": 0,
            "prior_predictive_call_count": 0,
            "full_data_posterior_call_count": 0,
            "grouped_validation_posterior_call_count": 0,
            "nuts_sampler_call_upper_bound": 0,
            "posterior_predictive_forward_call_upper_bound": 0,
            "all_stochastic_call_upper_bound": 0,
        },
    }

    def reject_published_directory(report_path: Path, **unused_validation_arguments,) -> None:
        if report_path.parent == output_directory:
            raise RuntimeError("Synthetic post-publication failure.")

    monkeypatch.setattr(
        bayesian_models,
        "validate_bayesian_report",
        reject_published_directory,
    )

    with pytest.raises(
        RuntimeError,
        match = "Synthetic post-publication failure",
    ):
        bayesian_models.publish_validation_artifacts(
            workflow = minimal_workflow,
            output_directory = output_directory,
        )

    assert sentinel_path.is_file()
    assert sentinel_path.read_text(
        encoding = "utf-8"
    ) == "previous"

    for unexpected_name in (
        "candidate_registry.csv",
        "observation_accounting.csv",
        "grouped_validation_folds.csv",
        "bayesian_modeling_report.json",
    ):
        assert not (
            output_directory / unexpected_name
        ).exists()


# Confirming sampling-method scopes use the design report's exact encoding
def test_cell_matches_scope_accepts_sampling_method_scope() -> None:
    cell = {
        "dataset_stage": "Stage_2",
        "sampling_method": "pi"
    }

    assert bayesian_models.cell_matches_scope(
        cell, "Stage_2/sampling_method=pi"
    ) is True

    assert bayesian_models.cell_matches_scope(
        cell, "Stage_2/sampling_method=cLH"
    ) is False

    assert bayesian_models.cell_matches_scope(
        cell, "Stage_1/sampling_method=pi"
    ) is False

    assert bayesian_models.cell_matches_scope(
        cell, "Stage_2"
    ) is True


# Confirming stage-only scopes ignore sampling method
def test_cell_matches_scope_accepts_stage_only_scope() -> None:
    first_cell = {
        "dataset_stage": "Stage_1",
        "sampling_method": "FF"
    }

    second_cell = {
        "dataset_stage": "Stage_1",
        "sampling_method": "cLH"
    }

    assert bayesian_models.cell_matches_scope(first_cell, "Stage_1") is True
    assert bayesian_models.cell_matches_scope(second_cell, "Stage_1") is True


# Rejecting an unsupported design scope
def test_cell_matches_scope_rejects_unknown_scope() -> None:
    with pytest.raises(RuntimeError, match = "Unsupported design scope"):
        bayesian_models.cell_matches_scope(
            {"dataset_stage": "Stage_1", "sampling_method": "FF"},
            "Stage_1/laboratory = HM"
        )


# Confirming that a policy with a missing reason-code contract fails validation
def test_validate_policy_requires_reason_code_contract() -> None:
    policy = bayesian_models.load_json_object(
        PROJECT_POLICY_PATH
    )
    del policy["reason_code_contract"]

    with pytest.raises(RuntimeError, match = "missing"):
        bayesian_models.validate_policy(policy)


# Confirming that a policy with an unregistered admissibility reason code fails
def test_validate_policy_requires_registered_reason_codes() -> None:
    policy = bayesian_models.load_json_object(
        PROJECT_POLICY_PATH
    )
    policy["reason_code_contract"]["candidate_admissibility"] = []

    with pytest.raises(RuntimeError):
        bayesian_models.validate_policy(policy)


# Confirming that a policy with an unsupported likelihood fails validation
def test_validate_policy_rejects_unsupported_likelihood() -> None:
    policy = bayesian_models.load_json_object(
        PROJECT_POLICY_PATH
    )
    policy["model"]["likelihood"] = "gaussian"

    with pytest.raises(RuntimeError, match = "model.likelihood"):
        bayesian_models.validate_policy(policy)


# Confirming that a negative sampler seed fails validation
def test_validate_policy_rejects_negative_sampler_seed() -> None:
    policy = bayesian_models.load_json_object(
        PROJECT_POLICY_PATH
    )
    policy["sampling"]["sampler_seed"] = -1

    with pytest.raises(RuntimeError, match = "sampler_seed"):
        bayesian_models.validate_policy(policy)


# Confirming that the removed chain-seed list is not part of the registered sampling contract
def test_validate_policy_confirms_single_sampler_seed_contract() -> None:
    policy = bayesian_models.load_json_object(
        PROJECT_POLICY_PATH
    )

    assert "chain_seeds" not in policy["sampling"]
    assert isinstance(policy["sampling"]["sampler_seed"], int)


# Confirming that a policy with non-false automatic retry fails validation
def test_validate_policy_rejects_automatic_retry() -> None:
    policy = bayesian_models.load_json_object(
        PROJECT_POLICY_PATH
    )
    policy["sampling"]["automatic_retry_permitted"] = True

    with pytest.raises(RuntimeError, match = "automatic_retry_permitted"):
        bayesian_models.validate_policy(policy)


# Confirming that a policy with a non-positive fold count fails validation
def test_validate_policy_rejects_invalid_fold_count() -> None:
    policy = bayesian_models.load_json_object(
        PROJECT_POLICY_PATH
    )
    policy["predictive_assessment"]["requested_fold_count"] = 0

    with pytest.raises(RuntimeError, match = "requested_fold_count"):
        bayesian_models.validate_policy(policy)


# Confirming that a policy with byte-identical cross-platform promises fails
def test_validate_policy_rejects_cross_platform_draw_promise() -> None:
    policy = bayesian_models.load_json_object(
        PROJECT_POLICY_PATH
    )
    policy["reproducibility"][
        "byte_identical_posterior_draws_promised_across_platforms"
    ] = True

    with pytest.raises(RuntimeError, match = "byte_identical_posterior_draws"):
        bayesian_models.validate_policy(policy)


# Confirming a mutated construction sequence is rejected
def test_validate_policy_rejects_mutated_construction_sequence() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["candidate_inventory"]["candidate_construction_sequence"] = (
        policy["candidate_inventory"]["candidate_construction_sequence"][:-1]
    )
    with pytest.raises(RuntimeError, match = "candidate_construction_sequence"):
        bayesian_models.validate_policy(policy)


# Confirming a design report bound to a different feature manifest is rejected
def test_lineage_rejects_design_report_feature_manifest_mismatch() -> None:
    feature_manifest = {
        "extraction_status": "completed",
        "summary": {
            "reference_extraction_failure_count": 0,
            "cycling_extraction_failure_count": 0
        }
    }
    design_report = {
        "audit_status": "completed",
        "feature_manifest_sha256": "0" * 64,
        "analytical_scope_reconciliation": {
            "reconciliation_complete": True
        },
        "calendar_and_cycle_designs_separated": True,
        "outcome_magnitude_or_direction_used_for_support_selection": False
    }
    identification_report = {
        "design_report_sha256": "1" * 64,
        "outcome_values_used_for_identification_decisions": False,
        "predictive_performance_used_for_identification_decisions": False
    }
    descriptive_report = {
        "inputs": {
            "feature_manifest": {"sha256": "2" * 64},
            "experimental_design_report": {"sha256": "3" * 64}
        },
        "reconciliation": {"reconciliation_complete": True},
        "calendar_and_cycle_models_separated": True,
        "primary_and_sensitivity_cohorts_separated": True,
        "causal_interpretation_authorized": False,
        "policy_recommendation_authorized": False,
        "outcome_magnitude_or_direction_used_for_candidate_selection": False
    }
    with pytest.raises(RuntimeError, match = "feature manifest"):
        bayesian_models.validate_input_lineage(
            feature_manifest = feature_manifest,
            design_report = design_report,
            identification_report = identification_report,
            descriptive_report = descriptive_report,
            feature_manifest_sha256 = "a" * 64,
            design_report_sha256 = "b" * 64
        )


# Confirming an unsafe table path is rejected
def test_verify_report_table_rejects_path_traversal(tmp_path: Path) -> None:
    report = {
        "tables": {
            "example": {
                "file_name": "../outside.csv",
                "sha256": "0" * 64,
                "row_count": 0,
                "schema": []
            }
        }
    }
    with pytest.raises(RuntimeError, match = "unsafe"):
        bayesian_models.verify_report_table(
            report_path = tmp_path / "report.json",
            report = report,
            table_name = "example"
        )


# Rejecting a mutated prior mean
def test_validate_policy_rejects_mutated_prior_mean() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["priors"]["population_intercept"]["mean"] = 5.0
    with pytest.raises(RuntimeError, match = "population_intercept.mean"):
        bayesian_models.validate_policy(policy)


# Rejecting a zero prior scale
def test_validate_policy_rejects_zero_prior_scale() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["priors"]["residual_standard_deviation"]["scale"] = 0.0
    with pytest.raises(RuntimeError, match = "residual_standard_deviation.scale"):
        bayesian_models.validate_policy(policy)


# Rejecting a mutated LKJ eta
def test_validate_policy_rejects_mutated_lkj_eta() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["priors"]["cell_random_effect_correlation"]["eta"] = 10.0
    with pytest.raises(RuntimeError, match = "correlation"):
        bayesian_models.validate_policy(policy)


# Rejecting a mutated Student-t degrees-of-freedom mean
def test_validate_policy_rejects_mutated_degrees_of_freedom_mean() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["priors"]["student_t_degrees_of_freedom"]["exponential_mean"] = 5.0
    with pytest.raises(RuntimeError, match = "degrees-of-freedom"):
        bayesian_models.validate_policy(policy)


# Rejecting a mutated sensitivity-multiplier list
def test_validate_policy_rejects_mutated_multiplier_list() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["priors"]["prior_sensitivity_scale_multipliers"] = [0.5, 0.75, 2.0]
    with pytest.raises(RuntimeError, match = "prior_sensitivity_scale_multipliers"):
        bayesian_models.validate_policy(policy)


# Rejecting a mutated prior-predictive seed
def test_validate_policy_rejects_mutated_prior_predictive_seed() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["priors"]["prior_predictive_seeds"]["0.5"] = 0
    with pytest.raises(RuntimeError, match = "prior_predictive_seeds"):
        bayesian_models.validate_policy(policy)


# Rejecting a missing prior-predictive seed key
def test_validate_policy_rejects_missing_prior_predictive_seed() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    del policy["priors"]["prior_predictive_seeds"]["0.5"]
    with pytest.raises(RuntimeError, match = "prior-predictive seed keys"):
        bayesian_models.validate_policy(policy)


# Rejecting zero prior-predictive draws
def test_validate_policy_rejects_zero_prior_predictive_draws() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["priors"]["prior_predictive_draws_per_multiplier"] = 0
    with pytest.raises(RuntimeError, match = "prior_predictive_draws_per_multiplier"):
        bayesian_models.validate_policy(policy)


# Rejecting an invalid target_accept
def test_validate_policy_rejects_invalid_target_accept() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["sampling"]["target_accept"] = 1.5
    with pytest.raises(RuntimeError, match = "target_accept"):
        bayesian_models.validate_policy(policy)


# Rejecting a zero maximum tree depth
def test_validate_policy_rejects_zero_maximum_tree_depth() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["sampling"]["maximum_tree_depth"] = 0
    with pytest.raises(RuntimeError, match = "maximum_tree_depth"):
        bayesian_models.validate_policy(policy)


# Rejecting multi-core sampling
def test_validate_policy_rejects_multi_core_sampling() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["sampling"]["cores"] = 4
    with pytest.raises(RuntimeError, match = "single-core"):
        bayesian_models.validate_policy(policy)


# Rejecting a non-positive R-hat threshold
def test_validate_policy_rejects_invalid_rhat_threshold() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["diagnostics"]["maximum_rank_normalized_split_rhat"] = 1.0
    with pytest.raises(RuntimeError, match = "R-hat"):
        bayesian_models.validate_policy(policy)


# Rejecting a zero ESS threshold
def test_validate_policy_rejects_zero_ess_threshold() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["diagnostics"]["minimum_bulk_effective_sample_size"] = 0
    with pytest.raises(RuntimeError, match = "minimum_bulk_effective_sample_size"):
        bayesian_models.validate_policy(policy)


# Rejecting a negative divergence threshold
def test_validate_policy_rejects_negative_divergence_threshold() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["diagnostics"]["maximum_divergence_count"] = -1
    with pytest.raises(RuntimeError, match = "maximum_divergence_count"):
        bayesian_models.validate_policy(policy)


# Rejecting a mutated cache metadata field inventory
def test_validate_policy_rejects_mutated_cache_metadata_fields() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["cache_contract"]["cache_metadata_fields"] = ["cache_identity"]
    with pytest.raises(RuntimeError, match = "cache_metadata_fields"):
        bayesian_models.validate_policy(policy)


# Rejecting a mutated review document field inventory
def test_validate_policy_rejects_mutated_review_document_fields() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["prior_predictive_review"]["review_document_fields"] = [
        "policy_sha256"
    ]
    with pytest.raises(RuntimeError, match = "review_document_fields"):
        bayesian_models.validate_policy(policy)


# Rejecting a mutated algorithm declaration
def test_validate_policy_rejects_mutated_candidate_identity_fields() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["candidate_inventory"]["candidate_identity_fields"] = [
        "support_region_identity",
        "template_name"
    ]
    with pytest.raises(RuntimeError, match = "candidate_identity_fields"):
        bayesian_models.validate_policy(policy)


# Rejecting a mutated horizon formula
def test_validate_policy_rejects_mutated_horizon_encoding() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["horizon_encoding"]["formula"] = "(horizon - 0.0) / 1.0"
    with pytest.raises(RuntimeError, match = "horizon_encoding"):
        bayesian_models.validate_policy(policy)


# Rejecting a mutated fold-assignment method
def test_validate_policy_rejects_mutated_fold_assignment_method() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["predictive_assessment"]["fold_assignment_method"] = "random"
    with pytest.raises(RuntimeError, match = "fold_assignment_method"):
        bayesian_models.validate_policy(policy)


# Confirming environment validation rejects a missing PYTHONHASHSEED
def test_validate_stochastic_process_environment_rejects_missing_hashseed(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)

    for variable_name in policy["stochastic_process_environment"]["required_environment_variables"]:
        monkeypatch.setenv(variable_name, policy["stochastic_process_environment"]["required_environment_variables"][variable_name])

    monkeypatch.delenv("PYTHONHASHSEED", raising = False)

    with pytest.raises(RuntimeError, match = "Stochastic process environment"):
        bayesian_models.validate_stochastic_process_environment(policy)


# Confirming environment validation rejects a mismatched thread limit
def test_validate_stochastic_process_environment_rejects_thread_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)

    for variable_name, required_value in policy["stochastic_process_environment"]["required_environment_variables"].items():
        monkeypatch.setenv(variable_name, required_value)

    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    with pytest.raises(RuntimeError, match = "Stochastic process environment"):
        bayesian_models.validate_stochastic_process_environment(policy)


# Confirming environment validation reports the exact mismatches
def test_validate_stochastic_process_environment_reports_mismatches(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)

    for variable_name in policy["stochastic_process_environment"]["required_environment_variables"]:
        monkeypatch.delenv(variable_name, raising = False)

    with pytest.raises(RuntimeError) as exception_information:
        bayesian_models.validate_stochastic_process_environment(policy)

    error_message = str(exception_information.value)

    assert "PYTHONHASHSEED" in error_message
    assert "OMP_NUM_THREADS" in error_message
    assert "Relaunch the process with" in error_message


# Confirming the environment record captures every registered field
def test_build_software_environment_record_captures_all_fields(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)

    for variable_name, required_value in policy["stochastic_process_environment"]["required_environment_variables"].items():
        monkeypatch.setenv(variable_name, required_value)

    environment_record = bayesian_models.build_software_environment_record(policy)

    expected_identity_fields = set(
        policy["cache_contract"]["software_environment_identity_fields"]
    ) - {"environment_values"}

    observed_fields = set(environment_record) - {"environment_values"}

    assert observed_fields == expected_identity_fields

    assert environment_record["environment_values"] == {
        variable_name: policy["stochastic_process_environment"]["required_environment_variables"][variable_name]
        for variable_name in sorted(policy["stochastic_process_environment"]["required_environment_variables"])
    }

    assert environment_record["numpy_configuration_sha256"] is not None
    assert len(environment_record["numpy_configuration_sha256"]) == 64


# Confirming the policy validator requires the stochastic environment section
def test_validate_policy_requires_stochastic_process_environment() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    del policy["stochastic_process_environment"]

    with pytest.raises(RuntimeError, match = "missing"):
        bayesian_models.validate_policy(policy)


# Confirming the policy validator requires the posterior artifact contract
def test_validate_policy_requires_posterior_artifact_contract() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    del policy["posterior_artifact_contract"]

    with pytest.raises(RuntimeError, match = "missing"):
        bayesian_models.validate_policy(policy)


# Confirming PYTHONHASHSEED must equal "0"
def test_validate_policy_requires_zero_hashseed() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["stochastic_process_environment"]["required_environment_variables"]["PYTHONHASHSEED"] = "1"

    with pytest.raises(RuntimeError, match = "PYTHONHASHSEED"):
        bayesian_models.validate_policy(policy)


# Confirming validation-required-before-loading-pymc must be true
def test_validate_policy_requires_environment_validation_before_pymc() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["stochastic_process_environment"]["validation_required_before_loading_pymc"] = False

    with pytest.raises(RuntimeError, match = "validation_required_before_loading_pymc"):
        bayesian_models.validate_policy(policy)


# Confirming the new reason-code categories are accepted by the validator
def test_validate_policy_accepts_new_reason_code_categories() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)

    required_categories = {
        "prior_predictive_execution",
        "cache_validation",
        "posterior_sampling",
        "sampler_diagnostics",
        "posterior_predictive_execution",
        "grouped_validation_execution",
        "analytical_qualification",
        "trajectory_reporting",
    }

    assert required_categories.issubset(set(policy["reason_code_contract"]))

    bayesian_models.validate_policy(policy)


# Confirming a duplicated free-random-variable name is rejected
def test_validate_policy_rejects_duplicate_free_random_variable() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["posterior_artifact_contract"]["free_random_variable_names"].append(
        policy["posterior_artifact_contract"]["free_random_variable_names"][0]
    )

    with pytest.raises(RuntimeError, match = "free_random_variable_names"):
        bayesian_models.validate_policy(policy)


# Confirming an empty coordinate-name list is rejected
def test_validate_policy_rejects_empty_coordinate_list() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["posterior_artifact_contract"]["coordinate_names"] = []

    with pytest.raises(RuntimeError, match = "coordinate_names"):
        bayesian_models.validate_policy(policy)


# Building a deterministic synthetic admissible candidate for model tests
def build_admissible_component_candidate() -> dict:
    conditions = [
        "Stage_1/TP_k01/periodic_reference_tests",
        "Stage_1/TP_k02/periodic_reference_tests"
    ]

    included_observations = []

    for condition_index, condition_identity in enumerate(conditions, start = 1):
        for cell_number in range(1, 7):
            cell_identity = (
                f"Stage_1/TP_k{condition_index:02d}_{cell_number:02d}"
            )

            for horizon in [10.0, 20.0, 30.0, 40.0]:
                included_observations.append({
                    "source_identity": (
                        f"{cell_identity}/measurement_{int(horizon):02d}.csv"
                    ),
                    "cell_identity": cell_identity,
                    "condition_protocol_identity": condition_identity,
                    "horizon": horizon,
                    "outcome_value": (
                        1.0 - 0.001 * horizon - 0.0001 * cell_number
                    ),
                    "inside_evaluation_interval": True
                })

    return {
        "candidate_identity": "admissible-test-candidate",
        "condition_identities": conditions,
        "included_observations": included_observations,
        "collapsed_candidate": {
            "support_interval_minimum": "0.0",
            "support_interval_maximum": "50.0",
            "support_interval_lower_inclusive": "true",
            "support_interval_upper_inclusive": "true"
        }
    }


# Confirming the model-construction input applies the registered scaling
def test_prepare_pymc_candidate_data_transforms_registered_scale() -> None:
    candidate = build_admissible_component_candidate()

    prepared = bayesian_models.prepare_pymc_candidate_data(candidate)

    expected_scaled_horizons = (
        prepared["horizons"] - prepared["training_horizon_midpoint"]
    ) / prepared["training_horizon_width"]

    assert np.allclose(
        prepared["scaled_horizons"],
        expected_scaled_horizons
    )

    assert np.all(np.isfinite(prepared["log_outcomes"]))
    assert prepared["condition_identities"] == sorted(prepared["condition_identities"])
    assert prepared["cell_identities"] == sorted(prepared["cell_identities"])


# Confirming the model-construction input rejects a non-positive outcome
def test_prepare_pymc_candidate_data_rejects_nonpositive_outcome() -> None:
    candidate = build_admissible_component_candidate()

    candidate["included_observations"][0]["outcome_value"] = 0.0

    with pytest.raises(
        RuntimeError,
        match = "strictly positive"
    ):
        bayesian_models.prepare_pymc_candidate_data(candidate)


# Confirming the model-construction input scales horizons over the assembled training range
def test_candidate_horizon_scaling_uses_training_range() -> None:
    candidate = build_admissible_component_candidate()

    candidate["included_observations"][0]["horizon"] = 0.0

    candidate["included_observations"][-1]["horizon"] = 100.0

    prepared = bayesian_models.prepare_pymc_candidate_data(
        candidate
    )

    assert np.min(
        prepared["scaled_horizons"]
    ) == pytest.approx(
        -0.5,
        abs = 1e-12
    )

    assert np.max(
        prepared["scaled_horizons"]
    ) == pytest.approx(
        0.5,
        abs = 1e-12
    )

    assert prepared[
        "training_horizon_width"
    ] == pytest.approx(
        100.0,
        abs = 1e-12
    )


# Confirming floating-point horizon scaling round-off remains admissible
def test_candidate_horizon_scaling_absorbs_float64_roundoff() -> None:
    candidate = build_admissible_component_candidate()

    for observation_index, horizon in enumerate(
        [0.6, 0.7, 0.8, 0.9]
    ):
        candidate[
            "included_observations"
        ][
            observation_index
        ][
            "horizon"
        ] = horizon

    for observation in (
        candidate[
            "included_observations"
        ][
            4:
        ]
    ):
        observation[
            "horizon"
        ] = 0.75

    prepared = bayesian_models.prepare_pymc_candidate_data(
        candidate
    )

    assert np.min(
        prepared[
            "scaled_horizons"
        ]
    ) >= -0.5 - 1e-9

    assert np.max(
        prepared[
            "scaled_horizons"
        ]
    ) <= 0.5 + 1e-9


# Confirming the reporting-support interval remains distinct from the training-scale frame
def test_reporting_support_remains_separate_from_training_scale() -> None:
    candidate = build_admissible_component_candidate()

    candidate[
        "collapsed_candidate"
    ][
        "support_interval_minimum"
    ] = 100.0

    candidate[
        "collapsed_candidate"
    ][
        "support_interval_maximum"
    ] = 200.0

    prepared = bayesian_models.prepare_pymc_candidate_data(
        candidate
    )

    assert prepared[
        "support_minimum"
    ] == pytest.approx(
        100.0
    )

    assert prepared[
        "support_maximum"
    ] == pytest.approx(
        200.0
    )

    assert (
        prepared[
            "training_horizon_width"
        ]
        != prepared[
            "support_width"
        ]
    )


# Confirming the model registers every declared posterior artifact
def test_build_pymc_model_contains_registered_variables() -> None:
    _ = (pm, pt)

    policy = json.loads(
        PROJECT_POLICY_PATH.read_text(encoding = "utf-8")
    )

    candidate = build_admissible_component_candidate()

    model, prepared = bayesian_models.build_pymc_model(
        candidate = candidate,
        policy = policy,
        prior_scale_multiplier = 1.0
    )

    posterior_contract = policy["posterior_artifact_contract"]

    expected_named_variables = {
        *posterior_contract["free_random_variable_names"],
        *posterior_contract["posterior_deterministic_variable_names"],
        posterior_contract["likelihood_variable_name"]
    }

    assert expected_named_variables.issubset(model.named_vars)

    assert (
        len(prepared["observations"])
        == len(model.coords["observation_index"])
    )

    # The registered PyMC posterior-predictive variable must be the observed likelihood variable, because PyMC
    #  replicates that variable into the posterior-predictive group after sampling.
    assert (
        posterior_contract[
            "pymc_posterior_predictive_variable_names"
        ]
        == [
            posterior_contract[
                "likelihood_variable_name"
            ]
        ]
    )

    predictive_variable_name = posterior_contract[
        "pymc_posterior_predictive_variable_names"
    ][0]

    assert predictive_variable_name in model.named_vars


# Confirming the model rejects an unregistered prior multiplier
def test_build_pymc_model_rejects_unregistered_multiplier() -> None:
    policy = json.loads(
        PROJECT_POLICY_PATH.read_text(encoding = "utf-8")
    )

    candidate = build_admissible_component_candidate()

    with pytest.raises(
        RuntimeError,
        match = "not registered"
    ):
        bayesian_models.build_pymc_model(
            candidate = candidate,
            policy = policy,
            prior_scale_multiplier = 3.0
        )


# Confirming the registered observed likelihood supports prior replication
def test_registered_likelihood_supports_prior_replication() -> None:
    policy = json.loads(
        PROJECT_POLICY_PATH.read_text(
            encoding = "utf-8"
        )
    )

    candidate = build_admissible_component_candidate()

    model, prepared = bayesian_models.build_pymc_model(
        candidate = candidate,
        policy = policy,
        prior_scale_multiplier = 1.0
    )

    likelihood_variable_name = policy[
        "posterior_artifact_contract"
    ][
        "likelihood_variable_name"
    ]

    with model:
        inference_data = pm.sample_prior_predictive(
            draws = 2,
            var_names = [
                likelihood_variable_name
            ],
            random_seed = 17389,
            return_inferencedata = True
        )

    assert (
        likelihood_variable_name
        in inference_data.prior_predictive
    )

    replicated = inference_data.prior_predictive[
        likelihood_variable_name
    ]

    assert replicated.sizes[
        "observation_index"
    ] == len(
        prepared[
            "observations"
        ]
    )


# Confirming the validator rejects an unregistered predictive variable
def test_policy_rejects_unregistered_predictive_variable() -> None:
    policy = json.loads(
        PROJECT_POLICY_PATH.read_text(
            encoding = "utf-8"
        )
    )

    policy[
        "posterior_artifact_contract"
    ][
        "pymc_posterior_predictive_variable_names"
    ] = [
        "undeclared_replication"
    ]

    with pytest.raises(
        RuntimeError,
        match = (
            "posterior-predictive variable must be the observed likelihood variable"
        )
    ):
        bayesian_models.validate_policy(
            policy
        )


# Confirming that the locked PyMC runtime supports every registered sampling argument
def test_locked_pymc_runtime_supports_registered_sampling_api() -> None:
    prior_signature = inspect.signature(
        pm.sample_prior_predictive
    )

    posterior_signature = inspect.signature(
        pm.sample
    )

    predictive_signature = inspect.signature(
        pm.sample_posterior_predictive
    )

    assert "draws" in prior_signature.parameters
    assert "random_seed" in prior_signature.parameters
    assert "var_names" in prior_signature.parameters
    assert "return_inferencedata" in prior_signature.parameters

    for parameter_name in (
        "draws",
        "tune",
        "chains",
        "cores",
        "random_seed",
        "init",
        "nuts_sampler",
        "discard_tuned_samples",
        "compute_convergence_checks",
        "return_inferencedata"
    ):
        assert parameter_name in posterior_signature.parameters

    for parameter_name in (
        "var_names",
        "random_seed",
        "return_inferencedata",
        "extend_inferencedata",
        "progressbar"
    ):
        assert parameter_name in predictive_signature.parameters


# Confirming evaluation points, draws, and values are counted separately
def test_prior_predictive_record_separates_evaluation_points_from_draws() -> None:
    candidate = {
        "candidate_identity": "c",
        "collapsed_candidate": {"outcome": "relative_capacity"},
    }

    log_values = np.zeros((10, 7), dtype = float)

    record = bayesian_models.build_prior_predictive_diagnostic_record(
        candidate = candidate,
        prior_scale_multiplier = 1.0,
        diagnostic_population = "assembled_observations",
        log_values = log_values,
    )

    assert record["evaluation_point_count"] == 7
    assert record["predictive_draw_count"] == 10
    assert record["predictive_value_count"] == 70


# Confirming the completed record uses a registered reason code
def test_prior_predictive_completed_record_uses_registered_reason_code() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)

    candidate = {
        "candidate_identity": "c",
        "collapsed_candidate": {"outcome": "relative_capacity"},
    }

    record = bayesian_models.build_prior_predictive_diagnostic_record(
        candidate = candidate,
        prior_scale_multiplier = 1.0,
        diagnostic_population = "assembled_observations",
        log_values = np.zeros((5, 3), dtype = float),
    )

    assert record["execution_reason_code"] in policy["reason_code_contract"]["prior_predictive_execution"]
    assert record["execution_status"] == "completed"


# Confirming the new-cell grid is deterministic for a fixed seed
def test_prior_predictive_new_cell_grid_is_deterministic_for_seed() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    candidate = build_admissible_component_candidate()

    model, data = bayesian_models.build_pymc_model(
        candidate = candidate,
        policy = policy,
        prior_scale_multiplier = 1.0,
    )

    with model:
        prior = pm.sample_prior_predictive(
            draws = 4,
            var_names = [
                "condition_intercept",
                "condition_horizon_slope",
                "cell_random_effect_cholesky",
                "residual_standard_deviation",
                "student_t_degrees_of_freedom",
            ],
            random_seed = 17389,
            return_inferencedata = True,
        )

    scaled_grid = np.asarray([-0.5, 0.0, 0.5], dtype = float)

    first = bayesian_models.build_prior_new_cell_predictive_grid(
        inference_data = prior,
        condition_index = 0,
        scaled_grid = scaled_grid,
        seed = 17390,
    )
    second = bayesian_models.build_prior_new_cell_predictive_grid(
        inference_data = prior,
        condition_index = 0,
        scaled_grid = scaled_grid,
        seed = 17390,
    )

    assert np.array_equal(first, second)

    third = bayesian_models.build_prior_new_cell_predictive_grid(
        inference_data = prior,
        condition_index = 0,
        scaled_grid = scaled_grid,
        seed = 17391,
    )

    assert not np.array_equal(first, third)


# Confirming review keys include the outcome
def test_review_validation_keys_include_outcome(tmp_path: Path) -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)

    diagnostic_records = [
        {
            "candidate_identity": "c1",
            "outcome": "relative_capacity",
            "prior_scale_multiplier": 1.0,
            "execution_status": "completed",
        }
    ]

    review_document = {
        "policy_sha256": "a" * 64,
        "candidate_inventory_sha256": "b" * 64,
        "prior_predictive_diagnostics_sha256": "c" * 64,
        "review_records": [
            {
                "candidate_identity": "c1",
                "outcome": "relative_dc_internal_resistance",
                "prior_scale_multiplier": 1.0,
                "review_status": "acknowledged",
                "review_rationale": "ok",
            }
        ],
    }

    review_path = tmp_path / "review.json"
    review_path.write_text(
        json.dumps(review_document),
        encoding = "utf-8",
    )

    workflow = {
        "validated_inputs": {
            "policy": policy,
            "policy_sha256": "a" * 64,
        }
    }

    with pytest.raises(RuntimeError, match = "does not cover the complete successful pair population"):
        bayesian_models.validate_prior_predictive_review(
            review_path = review_path,
            workflow = workflow,
            candidate_inventory_sha256 = "b" * 64,
            diagnostics_sha256 = "c" * 64,
            diagnostic_records = diagnostic_records,
        )


# Confirming the review rejects unregistered statuses
def test_review_validation_rejects_unregistered_status(tmp_path: Path) -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)

    diagnostic_records = [
        {
            "candidate_identity": "c1",
            "outcome": "relative_capacity",
            "prior_scale_multiplier": 1.0,
            "execution_status": "completed",
        }
    ]

    review_document = {
        "policy_sha256": "a" * 64,
        "candidate_inventory_sha256": "b" * 64,
        "prior_predictive_diagnostics_sha256": "c" * 64,
        "review_records": [
            {
                "candidate_identity": "c1",
                "outcome": "relative_capacity",
                "prior_scale_multiplier": 1.0,
                "review_status": "not_a_status",
                "review_rationale": "x",
            }
        ],
    }

    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review_document), encoding = "utf-8")

    workflow = {
        "validated_inputs": {
            "policy": policy,
            "policy_sha256": "a" * 64,
        }
    }

    with pytest.raises(RuntimeError, match = "unregistered review status"):
        bayesian_models.validate_prior_predictive_review(
            review_path = review_path,
            workflow = workflow,
            candidate_inventory_sha256 = "b" * 64,
            diagnostics_sha256 = "c" * 64,
            diagnostic_records = diagnostic_records,
        )


# Confirming the compiler environment is required
def test_require_compiler_environment_rejects_missing_fields() -> None:
    with pytest.raises(RuntimeError, match = "PyTensor C compiler"):
        bayesian_models.require_compiler_environment(
            {"pytensor_cxx": None, "cxx_compiler_identity": None}
        )

    with pytest.raises(RuntimeError, match = "resolvable C\\+\\+ compiler identity"):
        bayesian_models.require_compiler_environment(
            {"pytensor_cxx": "g++", "cxx_compiler_identity": None}
        )


# Confirming the environment record carries compiler fields
def test_environment_record_includes_compiler_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)

    for variable_name, required_value in policy["stochastic_process_environment"]["required_environment_variables"].items():
        monkeypatch.setenv(variable_name, required_value)

    environment_record = bayesian_models.build_software_environment_record(policy)

    assert "pytensor_cxx" in environment_record
    assert "cxx_compiler_identity" in environment_record

    expected_fields = set(
        policy["cache_contract"]["software_environment_identity_fields"]
    )

    observed_fields = set(environment_record) - {"environment_values"}

    assert expected_fields.issubset(observed_fields)


# Confirming the registered new-cell seed rule is enforced
def test_validate_policy_requires_new_cell_seed_rule() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["priors"]["prior_predictive_new_cell_seed_rule"] = "different_rule"

    with pytest.raises(RuntimeError, match = "prior_predictive_new_cell_seed_rule"):
        bayesian_models.validate_policy(policy)


# Rejecting a prior-predictive diagnostic table that has been modified after publication
def test_validate_bayesian_report_rejects_prior_predictive_diagnostic_tamper(
    tmp_path: Path, published_prior_predictive_directory: Path
) -> None:
    source_directory = (
        published_prior_predictive_directory
    )

    report = bayesian_models.load_json_object(
        source_directory / "bayesian_modeling_report.json"
    )

    diagnostic_records = list(
        bayesian_models.read_csv_rows(
            source_directory / "prior_predictive_diagnostics.csv"
        )[1]
    )

    candidate_rows = list(
        bayesian_models.read_csv_rows(
            source_directory / "candidate_registry.csv"
        )[1]
    )
    accounting_rows = list(
        bayesian_models.read_csv_rows(
            source_directory / "observation_accounting.csv"
        )[1]
    )
    fold_rows = list(
        bayesian_models.read_csv_rows(
            source_directory / "grouped_validation_folds.csv"
        )[1]
    )

    review_template = bayesian_models.load_json_object(
        source_directory / "bayesian_prior_predictive_review_template.json"
    )

    tampered_directory = tmp_path / "tampered"
    tampered_directory.mkdir()

    for file_name in (
        "candidate_registry.csv",
        "observation_accounting.csv",
        "grouped_validation_folds.csv",
        "bayesian_prior_predictive_review_template.json",
    ):
        shutil.copyfile(
            source_directory / file_name,
            tampered_directory / file_name
        )

    tampered_diagnostic_path = (
        tampered_directory
        / "prior_predictive_diagnostics.csv"
    )

    lines = (
        source_directory / "prior_predictive_diagnostics.csv"
    ).read_text(encoding = "utf-8").splitlines()

    tampered_lines = [lines[0]]
    tampered_lines.append(lines[1].replace(",completed,", ",prior_predictive_failed,"))
    tampered_lines.extend(lines[2:])

    tampered_diagnostic_path.write_text(
        "\n".join(tampered_lines) + "\n",
        encoding = "utf-8"
    )

    tampered_report = dict(report)
    tampered_report["tables"] = dict(report["tables"])
    tampered_report["tables"]["prior_predictive_diagnostics"] = {
        **report["tables"]["prior_predictive_diagnostics"],
        "sha256": bayesian_models.calculate_file_sha256(tampered_diagnostic_path),
    }

    tampered_report_path = tampered_directory / "bayesian_modeling_report.json"
    tampered_report_path.write_text(
        json.dumps(tampered_report, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = "contents differ from the validated workflow"
    ):
        bayesian_models.validate_bayesian_report(
            report_path = tampered_report_path,
            expected_policy = report["policy"],
            expected_policy_sha256 = report["policy_sha256"],
            expected_input_hashes = report["input_hashes"],
            expected_candidate_records = candidate_rows,
            expected_accounting_records = accounting_rows,
            expected_fold_records = fold_rows,
            expected_execution_plan = report["execution_plan"],
            expected_operation = "prior_predictive",
            expected_additional_tables = {
                "prior_predictive_diagnostics": (
                    bayesian_models.PRIOR_PREDICTIVE_DIAGNOSTIC_FIELDS,
                    diagnostic_records
                )
            },
            expected_review_template = review_template,
            expected_execution_accounting = report[
                "prior_predictive_execution_accounting"
            ],
        )


# Rejecting a prior-predictive review template that has been modified after publication
def test_validate_bayesian_report_rejects_review_template_tamper(
    tmp_path: Path, published_prior_predictive_directory: Path
) -> None:
    source_directory = (
        published_prior_predictive_directory
    )

    report = bayesian_models.load_json_object(
        source_directory / "bayesian_modeling_report.json"
    )

    diagnostic_records = list(
        bayesian_models.read_csv_rows(
            source_directory / "prior_predictive_diagnostics.csv"
        )[1]
    )

    candidate_rows = list(
        bayesian_models.read_csv_rows(
            source_directory / "candidate_registry.csv"
        )[1]
    )
    accounting_rows = list(
        bayesian_models.read_csv_rows(
            source_directory / "observation_accounting.csv"
        )[1]
    )
    fold_rows = list(
        bayesian_models.read_csv_rows(
            source_directory / "grouped_validation_folds.csv"
        )[1]
    )

    original_review_template = bayesian_models.load_json_object(
        source_directory / "bayesian_prior_predictive_review_template.json"
    )

    tampered_directory = tmp_path / "tampered"
    tampered_directory.mkdir()

    for file_name in (
        "candidate_registry.csv",
        "observation_accounting.csv",
        "grouped_validation_folds.csv",
        "prior_predictive_diagnostics.csv",
    ):
        shutil.copyfile(
            source_directory / file_name,
            tampered_directory / file_name
        )

    tampered_review = dict(original_review_template)
    tampered_review["review_records"] = [
        {**record, "review_status": "acknowledged", "review_rationale": "tampered"}
        for record in original_review_template["review_records"]
    ]

    tampered_review_path = (
        tampered_directory
        / "bayesian_prior_predictive_review_template.json"
    )
    tampered_review_path.write_text(
        json.dumps(tampered_review, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8"
    )

    tampered_report = dict(report)
    tampered_report["tables"] = dict(report["tables"])
    tampered_report["review_template"] = {
        **report["review_template"],
        "sha256": bayesian_models.calculate_file_sha256(tampered_review_path),
    }

    tampered_report_path = tampered_directory / "bayesian_modeling_report.json"
    tampered_report_path.write_text(
        json.dumps(tampered_report, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = "review-template contents differ"
    ):
        bayesian_models.validate_bayesian_report(
            report_path = tampered_report_path,
            expected_policy = report["policy"],
            expected_policy_sha256 = report["policy_sha256"],
            expected_input_hashes = report["input_hashes"],
            expected_candidate_records = candidate_rows,
            expected_accounting_records = accounting_rows,
            expected_fold_records = fold_rows,
            expected_execution_plan = report["execution_plan"],
            expected_operation = "prior_predictive",
            expected_additional_tables = {
                "prior_predictive_diagnostics": (
                    bayesian_models.PRIOR_PREDICTIVE_DIAGNOSTIC_FIELDS,
                    diagnostic_records
                )
            },
            expected_review_template = original_review_template,
            expected_execution_accounting = report[
                "prior_predictive_execution_accounting"
            ],
        )


# Confirming that a well-formed posterior cache diagnostic record passes validation and returns the original fields
def test_validate_posterior_cache_diagnostics_accepts_well_formed_record() -> None:
    posterior_record = {
        "candidate_identity": "c1",
        "outcome": "relative_capacity",
        "prior_scale_multiplier": 1.0,
        "fit_status": "completed",
    }
    parameter_records = [
        {
            "candidate_identity": "c1",
            "outcome": "relative_capacity",
            "prior_scale_multiplier": 1.0,
            "fit_scope": "full_data",
            "fold_index": None,
            "parameter_identity": "population_intercept",
            "rank_normalized_split_rhat": 1.0,
            "bulk_effective_sample_size": 800.0,
            "tail_effective_sample_size": 700.0,
            "rhat_passed": True,
            "bulk_ess_passed": True,
            "tail_ess_passed": True,
            "divergence_count": 0,
            "minimum_bfmi": 0.8,
            "tree_depth_hit_count": 0,
            "diagnostic_status": "completed_passed",
            "diagnostic_reason_code": (
                "SAMPLER_DIAGNOSTICS_COMPLETED"
            ),
        }
    ]

    packed = bayesian_models.build_posterior_cache_diagnostics(
        posterior_record = posterior_record,
        parameter_diagnostic_records = parameter_records,
    )

    returned_record, returned_parameters = (
        bayesian_models.validate_posterior_cache_diagnostics(
            cached_diagnostics = packed,
            candidate_identity = "c1",
            outcome = "relative_capacity",
            prior_scale_multiplier = 1.0,
        )
    )

    assert returned_record == posterior_record
    assert returned_parameters == parameter_records


# Rejecting a posterior cache diagnostic record without a registered prior multiplier
def test_validate_posterior_cache_diagnostics_rejects_missing_multiplier() -> None:
    packed = {
        "posterior_record": {
            "candidate_identity": "c1",
            "outcome": "relative_capacity",
            "fit_status": "completed",
        },
        "parameter_diagnostic_records": [],
    }

    with pytest.raises(
        RuntimeError,
        match = "invalid prior multiplier",
    ):
        bayesian_models.validate_posterior_cache_diagnostics(
            cached_diagnostics = packed,
            candidate_identity = "c1",
            outcome = "relative_capacity",
            prior_scale_multiplier = 1.0,
        )


# Confirming that stochastic input hashes mirror the workflow's verified input hashes
def test_build_stochastic_input_hashes_matches_workflow() -> None:
    workflow = {
        "validated_inputs": {
            "feature_manifest_sha256": "a" * 64,
            "design_report_sha256": "b" * 64,
            "identification_report_sha256": "c" * 64,
            "descriptive_report_sha256": "d" * 64,
        }
    }

    assert bayesian_models.build_stochastic_input_hashes(
        workflow
    ) == {
        "feature_manifest_sha256": "a" * 64,
        "design_report_sha256": "b" * 64,
        "identification_report_sha256": "c" * 64,
        "descriptive_report_sha256": "d" * 64,
    }


# Confirming deterministic trajectory prediction seeds are stable and context-specific
def test_deterministic_prediction_seed_is_stable_and_context_specific() -> None:
    first = bayesian_models.deterministic_prediction_seed(
        candidate_identity = "candidate-a",
        condition_identity = "condition-a",
        registered_seed = 17603
    )
    repeated = bayesian_models.deterministic_prediction_seed(
        candidate_identity = "candidate-a",
        condition_identity = "condition-a",
        registered_seed = 17603
    )
    changed = bayesian_models.deterministic_prediction_seed(
        candidate_identity = "candidate-a",
        condition_identity = "condition-b",
        registered_seed = 17603
    )

    assert first == repeated
    assert first != changed
    assert 0 <= first <= 2 ** 32 - 1


# Confirming equal-cell reduction prevents cells with more observations from receiving more weight
def test_equal_cell_mean_weights_cells_equally() -> None:
    result = bayesian_models.equal_cell_mean(
        {
            "cell-a": [1.0, 1.0, 1.0],
            "cell-b": [3.0]
        }
    )

    assert result == pytest.approx(2.0)


# Confirming the policy prohibits an unregistered binary sensitivity materiality threshold
def test_policy_rejects_binary_prior_sensitivity_threshold() -> None:
    policy = load_json_object(PROJECT_POLICY_PATH)
    policy["priors"]["binary_prior_sensitivity_materiality_threshold"] = 0.1

    with pytest.raises(
        RuntimeError,
        match = "must not register a binary prior-sensitivity materiality threshold"
    ):
        bayesian_models.validate_policy(policy)


def test_posterior_artifact_rejects_missing_predictive_group() -> None:
    policy = load_json_object(
        PROJECT_POLICY_PATH
    )

    candidate = build_admissible_component_candidate()

    model, _ = bayesian_models.build_pymc_model(
        candidate = candidate,
        policy = policy,
        prior_scale_multiplier = 1.0
    )

    with model:
        inference_data = pm.sample(
            draws = 5,
            tune = 5,
            chains = 2,
            cores = 1,
            random_seed = 17401,
            progressbar = False,
            compute_convergence_checks = False,
            return_inferencedata = True
        )

    with pytest.raises(
        RuntimeError,
        match = "missing required groups"
    ):
        bayesian_models.validate_posterior_inference_data(
            inference_data = inference_data,
            candidate = candidate,
            policy = policy
        )


# Confirming cache identity includes the exact implementation digest
def test_cache_identity_includes_implementation_digest() -> None:
    policy = load_json_object(
        PROJECT_POLICY_PATH
    )

    cache_identity = bayesian_models.build_cache_identity(
        policy = policy,
        policy_sha256 = "a" * 64,
        input_hashes = {
            "feature_manifest_sha256": "b" * 64,
            "design_report_sha256": "c" * 64,
            "identification_report_sha256": "d" * 64,
            "descriptive_report_sha256": "e" * 64
        },
        software_environment_sha256 = "f" * 64,
        candidate_identity = "candidate",
        operation = "fit",
        prior_scale_multiplier = 1.0,
        validation_fold_identity = None
    )

    assert cache_identity[
        "implementation_sha256"
    ] == bayesian_models.calculate_file_sha256(
        Path(
            bayesian_models.__file__
        ).resolve()
    )


# Confirming failed publication restores the previous output
def test_transactional_directory_replacement_restores_previous_output(tmp_path: Path) -> None:
    output_directory = (
        tmp_path
        / "published"
    )

    staging_directory = (
        tmp_path
        / "staging"
    )

    output_directory.mkdir()
    staging_directory.mkdir()

    (
        output_directory
        / "previous.txt"
    ).write_text(
        "previous\n",
        encoding = "utf-8"
    )

    (
        staging_directory
        / "replacement.txt"
    ).write_text(
        "replacement\n",
        encoding = "utf-8"
    )

    def reject_published_output() -> None:
        raise RuntimeError(
            "validation failed"
        )

    with pytest.raises(
        RuntimeError,
        match = "validation failed"
    ):
        bayesian_models.replace_directory_transactionally(
            staging_directory = staging_directory,
            output_directory = output_directory,
            validate_published = reject_published_output
        )

    assert (
        output_directory
        / "previous.txt"
    ).read_text(
        encoding = "utf-8"
    ) == "previous\n"

    assert not (
        output_directory
        / "replacement.txt"
    ).exists()

    assert not (
        tmp_path
        / ".published.backup"
    ).exists()


# Confirming successful publication removes the previous backup
def test_transactional_directory_replacement_publishes_valid_output(tmp_path: Path) -> None:
    output_directory = (
        tmp_path
        / "published"
    )

    staging_directory = (
        tmp_path
        / "staging"
    )

    output_directory.mkdir()
    staging_directory.mkdir()

    (
        output_directory
        / "previous.txt"
    ).write_text(
        "previous\n",
        encoding = "utf-8"
    )

    (
        staging_directory
        / "replacement.txt"
    ).write_text(
        "replacement\n",
        encoding = "utf-8"
    )

    def validate_published_output() -> None:
        assert (
            output_directory
            / "replacement.txt"
        ).is_file()

    bayesian_models.replace_directory_transactionally(
        staging_directory = staging_directory,
        output_directory = output_directory,
        validate_published = validate_published_output
    )

    assert (
        output_directory
        / "replacement.txt"
    ).read_text(
        encoding = "utf-8"
    ) == "replacement\n"

    assert not (
        output_directory
        / "previous.txt"
    ).exists()

    assert not (
        tmp_path
        / ".published.backup"
    ).exists()


# Confirming that full-posterior cache jobs partition deterministically across shards
def test_full_posterior_sharding_partitions_cache_identities(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = load_json_object(
        PROJECT_POLICY_PATH
    )

    candidate_identity = "candidate-shard-test"
    workflow = {
        "validated_inputs": {
            "policy": policy,
            "policy_sha256": "a" * 64,
            "feature_manifest_sha256": "b" * 64,
            "design_report_sha256": "c" * 64,
            "identification_report_sha256": "d" * 64,
            "descriptive_report_sha256": "e" * 64,
        },
        "candidate_records": [
            {
                "candidate_identity": candidate_identity,
                "admissibility_status": "admissible",
            }
        ],
        "component_candidates": [
            {
                "candidate_identity": candidate_identity,
                "collapsed_candidate": {
                    "outcome": "relative_capacity"
                },
                "included_observations": [],
                "condition_identities": [],
            }
        ],
    }

    acknowledged_pairs = {
        (
            candidate_identity,
            "relative_capacity",
            float(multiplier),
        )
        for multiplier in policy["sampling"]["full_data_posterior_scale_multipliers"]
    }

    sampled_identities_by_shard: dict[int, list[str]] = {}

    def fake_read(cache_directory, cache_identity, policy):
        return None, None, "CACHE_METADATA_MISSING"

    def fake_sample(candidate, policy, prior_scale_multiplier):
        return (
            None,
            {
                "candidate_identity": candidate_identity,
                "outcome": "relative_capacity",
                "prior_scale_multiplier": prior_scale_multiplier,
                "fit_status": "fit_failed",
                "fit_error": "synthetic",
                "traceback": "",
                "posterior_draw_count": 0,
                "posterior_predictive_value_count": 0,
                "posterior_predictive_nonfinite_count": None,
                "posterior_predictive_positive_finite_fraction": None,
                "sampler_diagnostics_status": "unavailable",
            },
            []
        )

    def fake_write_failure(cache_directory, cache_identity,
                           policy, software_environment_record,
                           software_environment_sha256, failure_record
    ):
        current_shard = fake_write_failure.current_shard
        sampled_identities_by_shard.setdefault(current_shard, []).append(
            cache_identity["cache_identity_sha256"]
        )

    fake_write_failure.current_shard = 0

    monkeypatch.setattr(
        bayesian_models,
        "read_validated_cache_entry",
        fake_read,
    )
    monkeypatch.setattr(
        bayesian_models,
        "sample_posterior_candidate",
        fake_sample,
    )
    monkeypatch.setattr(
        bayesian_models,
        "write_failure_cache_entry_transactionally",
        fake_write_failure,
    )

    shard_count = 8

    for shard_index in range(shard_count):
        fake_write_failure.current_shard = shard_index
        bayesian_models.execute_full_posterior_sampling(
            workflow = workflow,
            acknowledged_pairs = acknowledged_pairs,
            cache_directory = Path("/tmp/unused"),
            software_environment_record = {},
            software_environment_sha256 = "f" * 64,
            shard_count = shard_count,
            shard_index = shard_index,
            sampling_permitted = True,
        )

    observed_identities_per_shard = {
        shard_index: sampled_identities_by_shard.get(shard_index, [])
        for shard_index in range(shard_count)
    }

    flat_identities = [
        identity
        for identities in observed_identities_per_shard.values()
        for identity in identities
    ]

    assert len(flat_identities) == len(set(flat_identities)), (
        "Full posterior shards sampled the same cache identity more than once."
    )

    expected_identity_count = len(acknowledged_pairs)

    assert len(flat_identities) == expected_identity_count, (
        "Full posterior shards did not sample the complete cache inventory."
    )

    with pytest.raises(ValueError):
        bayesian_models.validate_cache_shard(
            shard_count = 8,
            shard_index = 8,
        )

    with pytest.raises(ValueError):
        bayesian_models.validate_cache_shard(
            shard_count = 8,
            shard_index = -1,
        )
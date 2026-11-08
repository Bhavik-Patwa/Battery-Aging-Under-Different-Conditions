import copy
import argparse
import hashlib
import inspect
import json
import math

from collections import defaultdict
from pathlib import Path

import pytest

import numpy as np

from scripts.run_predictive_sensitivity import (
    apply_incident_exclusion,
    apply_training_window,
    build_partition_assignments,
    build_sensitivity_registry,
    canonical_json,
    feature_group_indices,
    load_json_object,
    remove_feature_group,
    validate_policy
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Loading the registered sensitivity policy for fixture-driven tests
def load_policy() -> dict:
    return json.loads(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_text(encoding = "utf-8")
    )


# Building a minimal observation fixture covering two cells in one condition
def build_sample_observations() -> list[dict]:
    observations = []

    for cell_identity in ("cell_a", "cell_b"):
        for horizon in (1.0, 2.0, 3.0):
            observations.append({
                "cell_identity": cell_identity,
                "source_identity": f"source_{cell_identity}_{horizon}",
                "condition_protocol_identity": "condition_a",
                "horizon": horizon
            })

    return observations


# Building a larger fixture with enough cells per condition for grouped partitions
def build_partition_fixture() -> list[dict]:
    observations = []

    for condition_identity in ("condition_a", "condition_b"):
        for cell_index in range(6):
            cell_identity = f"{condition_identity}_cell_{cell_index}"
            observations.append({
                "cell_identity": cell_identity,
                "source_identity": f"source_{cell_identity}",
                "condition_protocol_identity": condition_identity,
                "horizon": 1.0
            })

    return observations


# Confirming partition assignments are deterministic for a fixed namespace
def test_partition_assignments_are_deterministic():
    observations = build_partition_fixture()

    first_assignments, first_fold_count = build_partition_assignments(
        "candidate", observations, 3, 2, "condition_stratified_partition_a"
    )
    second_assignments, second_fold_count = build_partition_assignments(
        "candidate", observations, 3, 2, "condition_stratified_partition_a"
    )

    assert first_assignments == second_assignments
    assert first_fold_count == second_fold_count


# Confirming partition assignments are grouped and cover every condition
def test_partition_assignments_are_grouped_and_complete():
    observations = build_partition_fixture()

    assignments, fold_count = build_partition_assignments(
        "candidate", observations, 3, 2, "condition_stratified_partition_a"
    )

    assert fold_count == 3
    assert set(assignments) == {observation["cell_identity"] for observation in observations}

    for condition_identity in ("condition_a", "condition_b"):
        condition_cells = {
            observation["cell_identity"]
            for observation in observations
            if observation["condition_protocol_identity"] == condition_identity
        }

        assert {
            assignments[cell_identity] for cell_identity in condition_cells
        } == {0, 1, 2}


# Confirming insufficient support returns None instead of raising
def test_partition_assignments_report_insufficient_support():
    observations = [
        {
            "cell_identity": "cell_a",
            "source_identity": "source_a",
            "condition_protocol_identity": "condition_a",
            "horizon": 1.0
        }
    ]

    assignments, fold_count = build_partition_assignments(
        "candidate", observations, 3, 2, "condition_stratified_partition_a"
    )

    assert assignments is None
    assert fold_count == 0


# Confirming the first-observation exclusion drops the minimum horizon within each cell
def test_training_window_excludes_per_cell_endpoint():
    observations = build_sample_observations()

    result = apply_training_window(observations, "first_observation_excluded")

    by_cell = defaultdict(list)
    for observation in result:
        by_cell[observation["cell_identity"]].append(observation["horizon"])

    assert all(min(horizons) > 1.0 for horizons in by_cell.values())


# Confirming the terminal-observation exclusion drops the maximum horizon within each cell
def test_training_window_excludes_terminal_endpoint():
    observations = build_sample_observations()

    result = apply_training_window(observations, "terminal_observation_excluded")

    by_cell = defaultdict(list)
    for observation in result:
        by_cell[observation["cell_identity"]].append(observation["horizon"])

    assert all(max(horizons) < 3.0 for horizons in by_cell.values())


# Rejecting an unregistered training-window variant
def test_training_window_rejects_unregistered_variant():
    with pytest.raises(RuntimeError, match = "Training-window variant is unregistered"):
        apply_training_window(build_sample_observations(), "unknown_variant")


# Building a minimal feature matrix fixture with the registered groups present
def build_sample_matrices() -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_names = [
        "scaled_horizon",
        "condition[condition_b]",
        "scaled_horizon:condition[condition_b]"
    ]

    training_matrix = np.asarray(
        [[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, 0.0, -0.5]],
        dtype = float
    )
    test_matrix = np.asarray(
        [[0.25, 1.0, 0.25]],
        dtype = float
    )

    return training_matrix, test_matrix, feature_names


# Building a minimal reason lookup where cell_a carries one documented reason code
def build_sample_reason_lookup() -> dict[tuple[str, str], frozenset[str]]:
    lookup = {}

    for observation in build_sample_observations():
        identity = (
            observation["source_identity"],
            observation["cell_identity"]
        )
        lookup[identity] = (
            frozenset({"DOCUMENTED_INCIDENT"}) if observation["cell_identity"] == "cell_a"
            else frozenset()
        )

    return lookup


# Confirming hierarchical feature groups preserve both main and interaction columns
def test_feature_group_indices_preserve_hierarchical_groups():
    feature_names = [
        "scaled_horizon",
        "condition[condition_b]",
        "scaled_horizon:condition[condition_b]"
    ]

    groups = feature_group_indices(feature_names)

    assert groups["horizon_main_and_interactions"] == [0, 2]
    assert groups["condition_main_and_interactions"] == [1, 2]
    assert groups["horizon_condition_interactions"] == [2]


# Rejecting an unregistered feature name during group indexing
def test_feature_group_indices_reject_unregistered_feature():
    with pytest.raises(RuntimeError, match = "unsupported feature"):
        feature_group_indices(["unknown_feature"])


# Confirming horizon ablation removes horizon and interactions together
def test_horizon_ablation_removes_horizon_and_interactions():
    training_matrix, test_matrix, feature_names = build_sample_matrices()

    training, test, names = remove_feature_group(
        training_matrix, test_matrix, feature_names, "horizon_main_and_interactions"
    )

    assert "scaled_horizon" not in names
    assert "scaled_horizon:condition[condition_b]" not in names
    assert names == ["condition[condition_b]"]


# Confirming condition ablation removes condition and interactions together
def test_condition_ablation_removes_condition_and_interactions():
    training_matrix, test_matrix, feature_names = build_sample_matrices()

    training, test, names = remove_feature_group(
        training_matrix, test_matrix, feature_names, "condition_main_and_interactions"
    )

    assert "condition[condition_b]" not in names
    assert "scaled_horizon:condition[condition_b]" not in names
    assert names == ["scaled_horizon"]


# Confirming single-condition ablation reports unavailable rather than raising
def test_single_condition_ablation_is_reported_unavailable():
    from scripts.run_predictive_models import EstimatorUnavailableError as ModelError

    feature_names = ["scaled_horizon"]
    training_matrix = np.asarray([[0.5]], dtype = float)
    test_matrix = np.asarray([[0.25]], dtype = float)

    with pytest.raises(ModelError) as exception_info:
        remove_feature_group(
            training_matrix, test_matrix, feature_names, "condition_main_and_interactions"
        )

    assert exception_info.value.reason_code == "PREDICTOR_GROUP_NOT_APPLICABLE"


# Confirming incident exclusion records eligibility, not silent drop
def test_incident_exclusion_requires_supported_cells():
    retained, accounting = apply_incident_exclusion(
        build_sample_observations(),
        build_sample_reason_lookup(),
        "DOCUMENTED_INCIDENT",
        minimum_excluded_cell_count = 2,
        minimum_retained_cell_count = 6
    )

    assert accounting["eligible"] is False
    assert retained
    assert accounting["excluded_independent_cell_count"] == 1
    assert accounting["retained_independent_cell_count"] == 1


# Rejecting an observation that has no matching feature provenance
def test_incident_exclusion_rejects_unmapped_observations():
    observations = build_sample_observations()
    reason_lookup = {}

    with pytest.raises(RuntimeError, match = "does not map to feature provenance"):
        apply_incident_exclusion(observations, reason_lookup, "ANY", 2, 6)


# Confirming horizon_condition_interactions ablation removes only interaction columns
def test_horizon_condition_interactions_ablation_removes_only_interactions():
    training_matrix, test_matrix, feature_names = build_sample_matrices()

    training, test, names = remove_feature_group(
        training_matrix, test_matrix, feature_names, "horizon_condition_interactions"
    )

    assert "scaled_horizon:condition[condition_b]" not in names
    assert "scaled_horizon" in names
    assert "condition[condition_b]" in names


# Rejecting an unregistered predictor group
def test_feature_ablation_rejects_unregistered_group():
    training_matrix, test_matrix, feature_names = build_sample_matrices()

    with pytest.raises(RuntimeError, match = "unsupported"):
        remove_feature_group(training_matrix, test_matrix, feature_names, "unknown_group")


# Confirming the sensitivity registry never crosses two perturbations
def test_sensitivity_variants_are_not_crossed():
    candidates = [
        {
            "candidate_identity": "candidate_a",
            "condition_identities": ["condition_a"],
            "included_observations": build_sample_observations()
        }
    ]

    policy = load_policy()

    registry = build_sensitivity_registry(
        candidates,
        build_sample_reason_lookup(),
        policy
    )

    assert registry

    for record in registry:
        non_primary_dimensions = sum([
            record["partition_identity"] != "primary",
            record["configuration_inventory"] != "primary",
            record["training_window_identity"] != "full",
            record["incident_reason_code"] is not None,
            record["ablation_group"] is not None
        ])

        assert non_primary_dimensions <= 1


# Confirming the registry is deterministic across repeat construction
def test_sensitivity_registry_is_deterministic():
    candidates = [
        {
            "candidate_identity": "candidate_a",
            "condition_identities": ["condition_a"],
            "included_observations": build_sample_observations()
        }
    ]

    policy = load_policy()

    first = build_sensitivity_registry(candidates, build_sample_reason_lookup(), policy)
    second = build_sensitivity_registry(candidates, build_sample_reason_lookup(), policy)

    assert first == second


# Confirming canonical JSON is order-independent
def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


# Confirming the validate branch returns before fitting is invoked
def test_validate_operation_never_calls_fitting():
    from scripts import run_predictive_sensitivity as module
    import inspect

    source = inspect.getsource(module.main)

    validate_branch_position = source.index('if args.operation == "validate":')
    fitting_call_position = source.index("tables = run_sensitivity_fitting(")

    assert validate_branch_position < fitting_call_position, (
        "Validate branch must return before reaching the fitting call."
    )

    validate_region = source[validate_branch_position:fitting_call_position]

    assert "return" in validate_region, "Validate branch must contain a return statement."

    assert "run_sensitivity_fitting" not in validate_region, (
        "Validate branch must not call run_sensitivity_fitting."
    )

    assert "publish_outputs" not in validate_region, (
        "Validate branch must not call publish_outputs."
    )


# Confirming training-window restriction retains only supported held-out horizons
def test_training_window_retains_only_supported_test_horizons():
    from scripts.run_predictive_sensitivity import restrict_test_to_training_support

    training_observations = [
        {"cell_identity": "cell_a", "condition_protocol_identity": "condition_a", "horizon": 1.0},
        {"cell_identity": "cell_a", "condition_protocol_identity": "condition_a", "horizon": 2.0},
        {"cell_identity": "cell_a", "condition_protocol_identity": "condition_a", "horizon": 3.0}
    ]

    test_observations = [
        {"cell_identity": "cell_b", "condition_protocol_identity": "condition_a", "horizon": 0.5},
        {"cell_identity": "cell_b", "condition_protocol_identity": "condition_a", "horizon": 1.5},
        {"cell_identity": "cell_b", "condition_protocol_identity": "condition_a", "horizon": 3.5}
    ]

    retained, excluded_count = restrict_test_to_training_support(
        training_observations, test_observations
    )

    assert excluded_count == 2
    assert [record["horizon"] for record in retained] == [1.5]


# Confirming the incident eligibility aggregation is present in the fitting function
def test_incident_eligibility_aggregates_every_fold():
    from scripts import run_predictive_sensitivity as module
    import inspect

    source = inspect.getsource(module.fit_sensitivity_variant_for_candidate)

    assert "incident_fold_eligibility" in source, (
        "Fitting function must accumulate per-fold incident eligibility."
    )

    assert "eligible_in_every_fold" in source, (
        "Metric record must expose the every-fold eligibility flag."
    )

    assert 'metric_record["incident_eligibility"]' in source, (
        "Metric record must be populated with the aggregated eligibility."
    )

    assert "all(" in source, (
        "Aggregation must use all() to combine fold-level eligibility."
    )

    assert "min(" in source, (
        "Aggregation must use min() for the minimum fold-level cell counts."
    )

    # Confirm the aggregation combines results from every fold, not just the first
    assert source.index("incident_fold_eligibility.append") < source.index(
        "metric_record[\"incident_eligibility\"]"
    ), (
        "Fold eligibility must be appended before being aggregated on the metric record."
    )


# Confirming the ablation group names are hierarchical
def test_ablation_group_names_are_hierarchical():
    policy = load_policy()

    assert policy["predictor_ablation"]["groups"] == [
        "horizon_main_and_interactions",
        "condition_main_and_interactions",
        "horizon_condition_interactions"
    ]


# Confirming the interval metric fields are registered
def test_interval_metric_fields_are_registered():
    from scripts.run_predictive_sensitivity import (
        SENSITIVITY_VALIDATION_METRIC_FIELDS
    )

    assert "equal_cell_weighted_prediction_interval_coverage" in SENSITIVITY_VALIDATION_METRIC_FIELDS
    assert "equal_cell_weighted_mean_prediction_interval_width" in SENSITIVITY_VALIDATION_METRIC_FIELDS


# Confirming the required output tables include the new schema
def test_output_contract_registers_all_new_tables():
    policy = load_policy()

    expected_tables = {
        "sensitivity_registry",
        "validation_folds",
        "configuration_selection",
        "validation_predictions",
        "validation_metrics",
        "sensitivity_comparisons",
        "condition_calibration",
        "predictor_ablation",
        "incident_exclusions",
        "prediction_interval_diagnostics"
    }

    assert set(policy["output_contract"]["tables"]) == expected_tables


# Confirming the primary analytical restriction propagates into comparisons
def test_primary_analytical_restriction_propagates_into_comparisons():
    from scripts.run_predictive_sensitivity import (
        build_sensitivity_comparisons
    )

    metric = {
        "variant_identity": "v1",
        "variant_type": "partition",
        "candidate_identity": "candidate",
        "model_family": "elastic_net",
        "design_scope": "Stage_1",
        "sampling_design_scope": "combined",
        "aging_mode": "calendar_aging",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": "elapsed_calendar_days",
        "validation_status": "completed"
    }

    sensitivity_predictions = [
        {
            "variant_identity": "v1",
            "cell_identity": "cell_a",
            "source_identity": "src_a",
            "condition_protocol_identity": "condition_a",
            "horizon": 1.0,
            "observed_outcome": 1.0,
            "predicted_outcome": 0.99,
            "benchmark_prediction": 1.0
        }
    ]

    primary_predictions_by_benchmark = {
        ("candidate", "elastic_net"): {
            ("cell_a", "src_a", "condition_a", 1.0): {
                "cell_identity": "cell_a",
                "source_identity": "src_a",
                "condition_protocol_identity": "condition_a",
                "horizon": 1.0,
                "observed_outcome": 1.0,
                "predicted_outcome": 0.985,
                "benchmark_prediction": 1.0
            }
        }
    }

    primary_metric_lookup = {
        ("candidate", "elastic_net"): {
            "analytical_status": "diagnostic_only"
        }
    }

    registry_by_variant = {
        "v1": {
            "variant_type": "partition",
            "ablation_group": None,
            "incident_reason_code": None
        }
    }

    rows = build_sensitivity_comparisons(
        [metric],
        sensitivity_predictions,
        primary_predictions_by_benchmark,
        primary_metric_lookup,
        registry_by_variant,
        {}
    )

    assert len(rows) == 1
    assert rows[0]["comparison_status"] == "unavailable"
    assert rows[0]["comparison_reason_code"] == "PRIMARY_BENCHMARK_NOT_ANALYTICALLY_AVAILABLE"


# Confirming a coincident partition variant is reported as uninformative
def test_partition_assignment_unchanged_is_reported_uninformative():
    from scripts.run_predictive_sensitivity import (
        fit_sensitivity_variant_for_candidate
    )

    candidate = {
        "candidate_identity": "candidate",
        "condition_identities": ["condition_a", "condition_b"],
        "included_observations": [],
        "collapsed_candidate": {
            "design_scope": "Stage_1",
            "sampling_design_scope": "combined",
            "aging_mode": "calendar_aging",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days"
        }
    }

    for condition_identity in ("condition_a", "condition_b"):
        for cell_index in range(6):
            cell_identity = f"{condition_identity}_cell_{cell_index}"
            candidate["included_observations"].append({
                "cell_identity": cell_identity,
                "source_identity": f"source_{cell_identity}",
                "condition_protocol_identity": condition_identity,
                "horizon": 1.0,
                "inside_evaluation_interval": True,
                "outcome_value": 1.0
            })

    primary_partition = {
        "condition_a_cell_0": 0,
        "condition_a_cell_1": 1,
        "condition_a_cell_2": 2,
        "condition_a_cell_3": 0,
        "condition_a_cell_4": 1,
        "condition_a_cell_5": 2,
        "condition_b_cell_0": 0,
        "condition_b_cell_1": 1,
        "condition_b_cell_2": 2,
        "condition_b_cell_3": 0,
        "condition_b_cell_4": 1,
        "condition_b_cell_5": 2
    }

    primary_folds = []
    for fold_index in range(3):
        primary_folds.append({
            "fold_index": fold_index,
            "test_cell_identities": sorted(
                cell for cell, f in primary_partition.items() if f == fold_index
            )
        })

    policy = load_policy()
    policy = {
        **policy,
        "metric_contract": {
            **policy["metric_contract"],
            "interval_probability": 0.9
        }
    }

    variant = {
        "variant_identity": "v1",
        "variant_type": "partition",
        "model_family": "elastic_net",
        "partition_identity": "condition_stratified_partition_a",
        "configuration_inventory": "primary",
        "training_window_identity": "full",
        "incident_reason_code": None,
        "ablation_group": None
    }

    import scripts.run_predictive_sensitivity as module

    original_build = module.build_partition_assignments

    def stub_assignment(*args, **kwargs):
        return primary_partition, 3

    module.build_partition_assignments = stub_assignment

    try:
        (
            fold_records,
            selection_records,
            predictions,
            metric
        ) = fit_sensitivity_variant_for_candidate(
            candidate,
            variant,
            {"candidate": primary_folds},
            {},
            {"primary": {"elastic_net": []}},
            policy,
            {}
        )
    finally:
        module.build_partition_assignments = original_build

    assert metric["validation_status"] == "unavailable"
    assert metric["validation_reason_code"] == "PARTITION_ASSIGNMENT_UNCHANGED"
    assert predictions == []


# Confirming a later fold failure suppresses the earlier fold's predictions
def test_partial_outer_predictions_are_not_published(monkeypatch):
    from scripts import run_predictive_sensitivity as module

    observations = []
    for condition_identity in ("condition_a", "condition_b"):
        for cell_index in range(4):
            cell_identity = f"{condition_identity}_cell_{cell_index}"
            for horizon in (1.0, 2.0, 3.0):
                observations.append({
                    "cell_identity": cell_identity,
                    "source_identity": f"src_{cell_identity}_{horizon}",
                    "condition_protocol_identity": condition_identity,
                    "horizon": horizon,
                    "outcome_value": 1.0,
                    "inside_evaluation_interval": True
                })

    candidate = {
        "candidate_identity": "cand",
        "condition_identities": ["condition_a", "condition_b"],
        "included_observations": observations,
        "collapsed_candidate": {
            "design_scope": "Stage_1",
            "sampling_design_scope": "combined",
            "aging_mode": "calendar_aging",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days"
        }
    }

    primary_folds = []
    for fold_index in range(2):
        test_cells = []
        for condition_identity in ("condition_a", "condition_b"):
            for cell_index in (fold_index, fold_index + 2):
                test_cells.append(f"{condition_identity}_cell_{cell_index}")
        primary_folds.append({
            "fold_index": fold_index,
            "test_cell_identities": sorted(test_cells)
        })

    variant = {
        "variant_identity": "v1",
        "variant_type": "ablation",
        "model_family": "elastic_net",
        "partition_identity": "primary",
        "configuration_inventory": "primary",
        "training_window_identity": "full",
        "incident_reason_code": None,
        "ablation_group": "horizon_condition_interactions"
    }

    primary_predictive_policy = {
        "nested_validation": {
            "inner_requested_fold_count": 2,
            "inner_minimum_fold_count": 2,
            "inner_fold_assignment_seed": 190731
        },
        "model_families": {
            "elastic_net": {
                "estimator": "ElasticNet",
                "maximum_iterations": 20000,
                "tolerance": 1e-8,
                "configurations": [{"alpha": 0.01, "l1_ratio": 0.5}]
            },
            "random_forest": {
                "estimator": "RandomForestRegressor",
                "n_estimators": 300,
                "max_features": 0.5,
                "configurations": [{"max_depth": 6, "min_samples_leaf": 1}]
            },
            "gradient_boosting": {
                "estimator": "GradientBoostingRegressor",
                "loss": "squared_error",
                "configurations": [{"n_estimators": 100, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 2}]
            }
        },
        "execution": {"random_seed": 904271}
    }

    fitting_policy = module.build_sensitivity_fitting_policy(
        load_policy(), primary_predictive_policy
    )
    config_inventories = module.build_config_inventories(
        primary_predictive_policy, fitting_policy
    )
    primary_config_identity = config_inventories["primary"]["elastic_net"][0]["configuration_identity"]

    primary_selected_lookup = {
        ("cand", "elastic_net", 0): primary_config_identity,
        ("cand", "elastic_net", 1): primary_config_identity
    }

    monkeypatch.setattr(
        module, "evaluate_sensitivity_configuration",
        lambda *a, **k: {
            "status": "completed", "reason_code": None, "fold_count": 2,
            "metric": 0.01,
            "residuals": np.array([0.1, 0.2, 0.3]),
            "residual_weights": np.array([1.0, 1.0, 1.0])
        }
    )
    monkeypatch.setattr(module, "fit_estimator", lambda *a, **k: None)

    call_count = [0]

    def failing_predict(estimator, matrix):
        call_count[0] += 1
        if call_count[0] >= 2:
            raise module.EstimatorUnavailableError("ESTIMATOR_PREDICTION_FAILED")
        return np.zeros(matrix.shape[0]), np.ones(matrix.shape[0])

    monkeypatch.setattr(module, "predict_original_scale", failing_predict)

    fold_records, _, predictions, metric = module.fit_sensitivity_variant_for_candidate(
        candidate, variant,
        {"cand": primary_folds},
        primary_selected_lookup,
        config_inventories,
        fitting_policy,
        {}
    )

    assert len(fold_records) == 2
    assert fold_records[0]["fold_status"] == "completed"
    assert fold_records[1]["fold_status"] == "unavailable"
    assert predictions == []
    assert metric["validation_status"] == "unavailable"
    assert metric["validation_reason_code"] == "ESTIMATOR_PREDICTION_FAILED"


# Confirming interval calibration derives from inner grouped residuals, not in-sample residuals
def test_fixed_configuration_intervals_use_inner_residuals():
    from scripts import run_predictive_sensitivity as module

    source = inspect.getsource(module.fit_sensitivity_variant_for_candidate)

    assert 'fixed_configuration_evaluation' in source
    assert 'fixed_configuration_evaluation["residuals"]' in source
    assert 'fixed_configuration_evaluation["residual_weights"]' in source
    assert "weighted_quantile(" in source

    assert "training_predicted = predict_original_scale" not in source
    assert "training_target - np.log(training_predicted)" not in source
    assert "if log_residual_quantile is None:" not in source


# Confirming partial overlap is reported unavailable, not completed
def test_comparison_requires_exact_shared_prediction_inventory():
    from scripts.run_predictive_sensitivity import (
        build_sensitivity_comparisons,
        prediction_identity
    )

    metric = {
        "variant_identity": "v1",
        "variant_type": "partition",
        "candidate_identity": "cand",
        "model_family": "elastic_net",
        "design_scope": "Stage_1",
        "sampling_design_scope": "combined",
        "aging_mode": "calendar_aging",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": "elapsed_calendar_days",
        "validation_status": "completed",
        "analytical_status": "available",
        "analytical_reason_code": None
    }

    primary_record = {
        "cell_identity": "cell_a",
        "source_identity": "src_a",
        "condition_protocol_identity": "c",
        "horizon": 1.0,
        "observed_outcome": 1.0,
        "predicted_outcome": 0.985,
        "benchmark_prediction": 1.0
    }

    sensitivity_predictions = [
        {**primary_record, "variant_identity": "v1", "outer_fold_index": 0,
         "predicted_outcome": 0.99},
        {**primary_record, "variant_identity": "v1", "outer_fold_index": 0,
         "cell_identity": "cell_b", "source_identity": "src_b",
         "predicted_outcome": 0.99}
    ]

    primary_predictions_by_benchmark = {
        ("cand", "elastic_net"): {
            prediction_identity(primary_record): primary_record
        }
    }

    rows = build_sensitivity_comparisons(
        [metric], sensitivity_predictions, primary_predictions_by_benchmark,
        {("cand", "elastic_net"): {"analytical_status": "available"}},
        {"v1": {"variant_type": "partition", "ablation_group": None,
                "incident_reason_code": None}},
        {}
    )

    assert len(rows) == 1
    assert rows[0]["comparison_status"] == "unavailable"
    assert rows[0]["comparison_reason_code"] == "PREDICTION_INVENTORY_MISMATCH"
    assert rows[0]["primary_inventory_count"] == 1
    assert rows[0]["variant_inventory_count"] == 2
    assert rows[0]["exact_inventory_match"] is False


# Confirming every evaluated configuration retains a selection record and exactly one is selected per fold
def test_expanded_configuration_selection_is_preserved(monkeypatch):
    from scripts import run_predictive_sensitivity as module

    observations = []
    for condition_identity in ("condition_a", "condition_b"):
        for cell_index in range(4):
            cell_identity = f"{condition_identity}_cell_{cell_index}"
            for horizon in (1.0, 2.0, 3.0):
                observations.append({
                    "cell_identity": cell_identity,
                    "source_identity": f"src_{cell_identity}_{horizon}",
                    "condition_protocol_identity": condition_identity,
                    "horizon": horizon,
                    "outcome_value": 1.0,
                    "inside_evaluation_interval": True
                })

    candidate = {
        "candidate_identity": "cand",
        "condition_identities": ["condition_a", "condition_b"],
        "included_observations": observations,
        "collapsed_candidate": {
            "design_scope": "Stage_1",
            "sampling_design_scope": "combined",
            "aging_mode": "calendar_aging",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days"
        }
    }

    primary_folds = []
    for fold_index in range(2):
        test_cells = []
        for condition_identity in ("condition_a", "condition_b"):
            for cell_index in (fold_index, fold_index + 2):
                test_cells.append(f"{condition_identity}_cell_{cell_index}")
        primary_folds.append({
            "fold_index": fold_index,
            "test_cell_identities": sorted(test_cells)
        })

    variant = {
        "variant_identity": "v1",
        "variant_type": "expanded_config",
        "model_family": "elastic_net",
        "partition_identity": "primary",
        "configuration_inventory": "expanded",
        "training_window_identity": "full",
        "incident_reason_code": None,
        "ablation_group": None
    }

    primary_predictive_policy = {
        "nested_validation": {
            "inner_requested_fold_count": 2,
            "inner_minimum_fold_count": 2,
            "inner_fold_assignment_seed": 190731
        },
        "model_families": {
            "elastic_net": {
                "estimator": "ElasticNet",
                "maximum_iterations": 20000,
                "tolerance": 1e-8,
                "configurations": [{"alpha": 0.01, "l1_ratio": 0.5}]
            },
            "random_forest": {
                "estimator": "RandomForestRegressor",
                "n_estimators": 300,
                "max_features": 0.5,
                "configurations": [{"max_depth": 6, "min_samples_leaf": 1}]
            },
            "gradient_boosting": {
                "estimator": "GradientBoostingRegressor",
                "loss": "squared_error",
                "configurations": [{"n_estimators": 100, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 2}]
            }
        },
        "execution": {"random_seed": 904271}
    }

    fitting_policy = module.build_sensitivity_fitting_policy(
        load_policy(), primary_predictive_policy
    )
    config_inventories = module.build_config_inventories(
        primary_predictive_policy, fitting_policy
    )

    # Each config must report a distinct inner metric so one is deterministically selected
    counter = [0]
    def stub_inner(*a, **k):
        counter[0] += 1
        return {
            "status": "completed", "reason_code": None, "fold_count": 2,
            "metric": 0.01 + counter[0] * 1e-6,
            "residuals": np.array([0.1, 0.2]),
            "residual_weights": np.array([1.0, 1.0])
        }

    monkeypatch.setattr(module, "evaluate_sensitivity_configuration", stub_inner)
    monkeypatch.setattr(module, "fit_estimator", lambda *a, **k: None)
    monkeypatch.setattr(module, "predict_original_scale",
                        lambda e, m: (np.zeros(m.shape[0]), np.ones(m.shape[0])))

    _, selection_records, _, _ = module.fit_sensitivity_variant_for_candidate(
        candidate, variant,
        {"cand": primary_folds},
        {},
        config_inventories,
        fitting_policy,
        {}
    )

    per_fold = defaultdict(list)
    for record in selection_records:
        per_fold[record["outer_fold_index"]].append(record)

    assert set(per_fold) == {0, 1}
    for fold_records in per_fold.values():
        assert len(fold_records) == 9
        assert sum(1 for r in fold_records if r["selected"] is True) == 1


# Confirming every registered condition receives a calibration record, empty ones marked unavailable
def test_condition_calibration_reports_empty_conditions():
    from scripts.run_predictive_sensitivity import build_condition_calibration

    metric = {
        "variant_identity": "v1",
        "variant_type": "partition",
        "candidate_identity": "cand",
        "model_family": "elastic_net",
        "validation_status": "completed"
    }

    sensitivity_predictions = [
        {"variant_identity": "v1", "cell_identity": "cell_a",
         "condition_protocol_identity": "condition_a",
         "observed_outcome": 1.0, "predicted_outcome": 0.99}
    ]

    rows = build_condition_calibration(
        [metric], sensitivity_predictions,
        {"cand": ["condition_a", "condition_b"]}
    )

    assert len(rows) == 2
    by_condition = {
        row["condition_protocol_identity"]: row for row in rows
        if row["condition_protocol_identity"] is not None
    }
    assert by_condition["condition_a"]["calibration_status"] == "completed"
    assert by_condition["condition_b"]["calibration_status"] == "unavailable"
    assert by_condition["condition_b"]["calibration_reason_code"] == (
        "EMPTY_CONDITION_PREDICTION_INVENTORY"
    )


# Confirming the post-publication validator detects table tampering
def test_published_tables_match_expected_content(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    empty_tables = {name: [] for name in policy["output_contract"]["tables"]}
    args = argparse.Namespace(output_directory = tmp_path / "out")

    module.publish_outputs(args, policy, policy_sha256, {}, empty_tables, {})

    report = json.loads(
        (args.output_directory / "predictive_sensitivity_report.json").read_text()
    )

    # Baseline passes
    module.validate_published_directory(
        args.output_directory, policy, policy_sha256, report, empty_tables
    )

    # Tampering must be detected
    victim_path = args.output_directory / "sensitivity_registry.csv"
    victim_path.write_text(victim_path.read_text() + "TAMPERED_LINE\n")

    with pytest.raises(RuntimeError, match = "hash mismatch"):
        module.validate_published_directory(
            args.output_directory, policy, policy_sha256, report, empty_tables
        )


# Confirming a registry variant without a matching metric fails publication
def test_every_registry_variant_has_exactly_one_metric(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    registry_record = {
        "variant_identity": "v1",
        "candidate_identity": "cand",
        "model_family": "elastic_net",
        "variant_type": "partition",
        "partition_identity": "primary",
        "configuration_inventory": "primary",
        "training_window_identity": "full",
        "incident_reason_code": None,
        "ablation_group": None,
        "registration_order": 0
    }

    tables = {name: [] for name in policy["output_contract"]["tables"]}
    tables["sensitivity_registry"] = [registry_record]

    args = argparse.Namespace(output_directory = tmp_path / "out")

    with pytest.raises(RuntimeError, match = "Every registry variant must have exactly one validation metric"):
        module.publish_outputs(args, policy, policy_sha256, {}, tables, {})


# Confirming an unavailable metric cannot ship predictions
def test_unavailable_metric_has_no_predictions(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    registry_record = {
        "variant_identity": "v1",
        "candidate_identity": "cand",
        "model_family": "elastic_net",
        "variant_type": "partition",
        "partition_identity": "primary",
        "configuration_inventory": "primary",
        "training_window_identity": "full",
        "incident_reason_code": None,
        "ablation_group": None,
        "registration_order": 0
    }

    metric_record = {
        "variant_identity": "v1",
        "variant_type": "partition",
        "candidate_identity": "cand",
        "model_family": "elastic_net",
        "design_scope": "Stage_1",
        "sampling_design_scope": "combined",
        "aging_mode": "calendar_aging",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": "elapsed_calendar_days",
        "outer_fold_count": 0,
        "test_cell_count": 0,
        "test_observation_count": 0,
        "equal_cell_weighted_root_mean_squared_error": None,
        "equal_cell_weighted_mean_absolute_error": None,
        "equal_cell_weighted_predictive_r_squared": None,
        "equal_cell_weighted_prediction_interval_coverage": None,
        "equal_cell_weighted_mean_prediction_interval_width": None,
        "validation_status": "unavailable",
        "validation_reason_code": "PARTITION_ASSIGNMENT_UNCHANGED",
        "analytical_status": "not_available",
        "analytical_reason_code": "PARTITION_ASSIGNMENT_UNCHANGED",
        "estimate_label": "predictive_sensitivity"
    }

    prediction_record = {
        "variant_identity": "v1",
        "variant_type": "partition",
        "candidate_identity": "cand",
        "model_family": "elastic_net",
        "outer_fold_index": 0,
        "cell_identity": "cell_a",
        "source_identity": "src_a",
        "condition_protocol_identity": "condition_a",
        "horizon": 1.0,
        "observed_outcome": 1.0,
        "predicted_outcome": 0.99,
        "prediction_interval_lower": 0.95,
        "prediction_interval_upper": 1.05,
        "benchmark_prediction": 1.0,
        "prediction_status": "completed",
        "prediction_reason_code": None,
        "estimate_label": "predictive_sensitivity"
    }

    tables = {name: [] for name in policy["output_contract"]["tables"]}
    tables["sensitivity_registry"] = [registry_record]
    tables["validation_metrics"] = [metric_record]
    tables["validation_predictions"] = [prediction_record]

    args = argparse.Namespace(output_directory = tmp_path / "out")

    with pytest.raises(RuntimeError, match = "Unavailable metric must not publish predictions"):
        module.publish_outputs(args, policy, policy_sha256, {}, tables, {})


# Confirming training-window subset inventories are accepted as valid comparisons
def test_training_window_subset_inventory_is_valid():
    from scripts.run_predictive_sensitivity import (
        build_sensitivity_comparisons,
        prediction_identity
    )

    metric = {
        "variant_identity": "v1",
        "variant_type": "training_window",
        "candidate_identity": "cand",
        "model_family": "elastic_net",
        "design_scope": "Stage_1",
        "sampling_design_scope": "combined",
        "aging_mode": "calendar_aging",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": "elapsed_calendar_days",
        "validation_status": "completed",
        "analytical_status": "available",
        "analytical_reason_code": None
    }

    primary_record = {
        "cell_identity": "cell_a",
        "source_identity": "src_a",
        "condition_protocol_identity": "c",
        "horizon": 1.0,
        "observed_outcome": 1.0,
        "predicted_outcome": 0.985,
        "benchmark_prediction": 1.0
    }

    subset_predictions = [
        {
            **primary_record, "variant_identity": "v1", "outer_fold_index": 0, "predicted_outcome": 0.99
        }
    ]

    primary_predictions_by_benchmark = {
        ("cand", "elastic_net"): {
            prediction_identity(
                primary_record
            ):
                primary_record,
            prediction_identity(
                {**primary_record, "cell_identity": "cell_b", "source_identity": "src_b"}
            ):
                {**primary_record, "cell_identity": "cell_b", "source_identity": "src_b"}
        }
    }

    rows = build_sensitivity_comparisons(
        [metric], subset_predictions, primary_predictions_by_benchmark,
        {
            ("cand", "elastic_net"): {"analytical_status": "available"}
        },
        {
            "v1": {"variant_type": "training_window", "ablation_group": None,
                "incident_reason_code": None}
        },
        {}
    )

    assert len(rows) == 1
    assert rows[0]["comparison_status"] == "completed"
    assert rows[0]["exact_inventory_match"] is False
    assert rows[0]["shared_observation_count"] == 1


# Confirming negative predictive R-squared downgrades to diagnostic-only
def test_negative_r_squared_downgrades_to_diagnostic_only():
    from scripts.run_predictive_sensitivity import qualify_sensitivity_metrics

    policy = load_policy()

    metric = {
        "variant_identity": "v1",
        "variant_type": "partition",
        "candidate_identity": "cand",
        "model_family": "elastic_net",
        "validation_status": "completed",
        "validation_reason_code": None,
        "equal_cell_weighted_predictive_r_squared": -0.05,
        "analytical_status": "not_available",
        "analytical_reason_code": None
    }

    calibration_rows = [
        {
            "variant_identity": "v1",
            "condition_protocol_identity": "condition_a",
            "calibration_status": "completed"
        }
    ]

    qualify_sensitivity_metrics(
        [metric],
        calibration_rows,
        {
            "cand": ["condition_a"]
        },
        policy
    )

    assert metric["analytical_status"] == "diagnostic_only"
    assert metric["analytical_reason_code"] == "PREDICTIVE_BENCHMARK_NOT_MET"


# Confirming incomplete condition calibration downgrades to diagnostic-only
def test_incomplete_calibration_downgrades_to_diagnostic_only():
    from scripts.run_predictive_sensitivity import qualify_sensitivity_metrics

    policy = load_policy()

    metric = {
        "variant_identity": "v1",
        "variant_type": "partition",
        "candidate_identity": "cand",
        "model_family": "elastic_net",
        "validation_status": "completed",
        "validation_reason_code": None,
        "equal_cell_weighted_predictive_r_squared": 0.8,
        "analytical_status": "not_available",
        "analytical_reason_code": None
    }

    calibration_rows = [
        {
            "variant_identity": "v1",
            "condition_protocol_identity": "condition_a",
            "calibration_status": "completed"
        },
        {
            "variant_identity": "v1",
            "condition_protocol_identity": "condition_b",
            "calibration_status": "unavailable"
        }
    ]

    qualify_sensitivity_metrics(
        [metric],
        calibration_rows,
        {"cand": ["condition_a", "condition_b"]},
        policy
    )

    assert metric["analytical_status"] == "diagnostic_only"
    assert metric["analytical_reason_code"] == "INCOMPLETE_CONDITION_CALIBRATION"


# Confirming an inadequately supported observed reason code stays registered and is reported unavailable
def test_inadequately_supported_incident_remains_registered():
    from scripts.run_predictive_sensitivity import (
        build_sensitivity_registry,
        reason_codes_for_candidate
    )

    candidate = {
        "candidate_identity": "candidate_a",
        "condition_identities": ["condition_a"],
        "included_observations": build_sample_observations()
    }

    reason_lookup = build_sample_reason_lookup()

    observed_codes = reason_codes_for_candidate(candidate, reason_lookup)

    assert "DOCUMENTED_INCIDENT" in observed_codes

    policy = load_policy()

    registry = build_sensitivity_registry([candidate], reason_lookup, policy)

    incident_variants = [
        record for record in registry if record["variant_type"] == "incident"
    ]

    assert any(
        record["incident_reason_code"] == "DOCUMENTED_INCIDENT"
        for record in incident_variants
    )


# Confirming interval diagnostics mark one-cell groups as diagnostic-only
def test_interval_diagnostics_require_two_cells():
    from scripts.run_predictive_sensitivity import build_interval_diagnostics

    policy = load_policy()

    metric = {
        "variant_identity": "v1",
        "variant_type": "partition",
        "candidate_identity": "cand",
        "model_family": "elastic_net",
        "validation_status": "completed",
        "analytical_status": "available",
        "analytical_reason_code": None,
        "outer_fold_count": 1
    }

    prediction = {
        "variant_identity": "v1",
        "outer_fold_index": 0,
        "cell_identity": "cell_a",
        "condition_protocol_identity": "condition_a",
        "observed_outcome": 1.0,
        "prediction_interval_lower": 0.9,
        "prediction_interval_upper": 1.1
    }

    rows = build_interval_diagnostics(
        [metric],
        [prediction],
        policy,
        {"cand": ["condition_a"]}
    )

    overall = next(
        record for record in rows if record["grouping_dimension"] == "overall"
    )

    assert overall["independent_cell_count"] == 1
    assert overall["assessment_status"] == "diagnostic_only"
    assert overall["assessment_reason_code"] == "INSUFFICIENT_INDEPENDENT_CELLS"


# Confirming expanded grids contain every primary configuration
def test_expanded_grid_contains_primary_configurations():
    from scripts.run_predictive_models import load_json_object as _primary_loader

    sensitivity_policy = load_policy()
    primary_policy = _primary_loader(
        PROJECT_ROOT / "configs" / "predictive_modeling_policy.json"
    )

    for family_name, family_policy in primary_policy["model_families"].items():
        expanded_configurations = sensitivity_policy[
            "expanded_configuration_inventory"
        ][family_name]["configurations"]

        for primary_configuration in family_policy["configurations"]:
            assert primary_configuration in expanded_configurations


# Confirming incomplete calibration is diagnostic-only even when both source and variant are available
def test_ablation_and_incident_require_analytical_availability():
    from scripts.run_predictive_sensitivity import (
        build_ablation_rows,
        prediction_identity
    )

    metric = {
        "variant_identity": "v1",
        "variant_type": "ablation",
        "candidate_identity": "cand",
        "model_family": "elastic_net",
        "validation_status": "completed",
        "analytical_status": "diagnostic_only",
        "analytical_reason_code": "PREDICTIVE_BENCHMARK_NOT_MET"
    }

    primary_record = {
        "cell_identity": "cell_a",
        "source_identity": "src_a",
        "condition_protocol_identity": "c",
        "horizon": 1.0,
        "observed_outcome": 1.0,
        "predicted_outcome": 0.985,
        "benchmark_prediction": 1.0
    }

    prediction_records = [
        {
            **primary_record, "variant_identity": "v1", "outer_fold_index": 0
        }
    ]

    rows = build_ablation_rows(
        [metric],
        prediction_records,
        {
            ("cand", "elastic_net"): {prediction_identity(primary_record): primary_record}
        },
        {
            ("cand", "elastic_net"): {"analytical_status": "available"}
        },
        {
            "v1": {"variant_type": "ablation", "ablation_group": "horizon_main_and_interactions",
                "incident_reason_code": None}
        }
    )

    assert len(rows) == 1
    assert rows[0]["comparison_status"] == "diagnostic_only"
    assert rows[0]["variant_analytical_status"] == "diagnostic_only"


# Confirming inner ablation residuals use the ablated predictor matrix
def test_inner_ablation_uses_ablated_matrix(monkeypatch):
    from scripts import run_predictive_sensitivity as module

    ablation_calls = []

    def stub_remove_feature_group(training, test, names, group):
        ablation_calls.append(group)
        return training, test, names

    def stub_build_inner_assignments(candidate_identity, observations, policy):
        return ({"cell_a": 0, "cell_b": 1}, 2)

    def stub_build_feature_matrix(observations, conditions, mid, width):
        names = [
            "scaled_horizon",
            "condition[c_b]",
            "scaled_horizon:condition[c_b]"
        ]
        return np.zeros((len(observations), 3)), names

    class _StubEstimator:
        def fit(self, X, y, sample_weight = None):
            pass

        def predict(self, X):
            return np.zeros(X.shape[0])

    monkeypatch.setattr(module, "remove_feature_group", stub_remove_feature_group)
    monkeypatch.setattr(module, "build_inner_assignments", stub_build_inner_assignments)
    monkeypatch.setattr(module, "fold_has_horizon_coverage", lambda *a, **k: True)
    monkeypatch.setattr(module, "calculate_training_horizon_scaling", lambda o: (1.0, 1.0))
    monkeypatch.setattr(module, "build_feature_matrix", stub_build_feature_matrix)
    monkeypatch.setattr(module, "build_estimator", lambda f, c, i, p: _StubEstimator())
    monkeypatch.setattr(module, "fit_estimator", lambda *a, **k: None)
    monkeypatch.setattr(module, "predict_original_scale", lambda e, m: (np.zeros(m.shape[0]), np.ones(m.shape[0])))
    monkeypatch.setattr(module, "equal_cell_metrics", lambda *a, **k: {"equal_cell_weighted_root_mean_squared_error": 0.1})
    monkeypatch.setattr(module, "build_cell_weights", lambda o: np.ones(len(o)))

    observations = [
        {
            "cell_identity": "cell_a",
            "condition_protocol_identity": "c_a",
            "horizon": 1.0,
            "outcome_value": 1.0,
            "inside_evaluation_interval": True
        },
        {
            "cell_identity": "cell_b",
            "condition_protocol_identity": "c_b",
            "horizon": 2.0,
            "outcome_value": 1.0,
            "inside_evaluation_interval": True
        }
    ]

    module.evaluate_sensitivity_configuration(
        "candidate",
        "elastic_net",
        {"configuration_identity": "cfg", "configuration": {"alpha": 0.01, "l1_ratio": 0.5}},
        observations,
        ["c_a", "c_b"],
        {
            "nested_validation": {
                "inner_requested_fold_count": 2,
                "inner_minimum_fold_count": 2,
                "inner_fold_assignment_seed": 1
            },
            "execution": {"random_seed": 1}
        },
        ablation_group = "horizon_main_and_interactions"
    )

    assert ablation_calls, (
        "remove_feature_group must be called inside evaluate_sensitivity_configuration "
        "when ablation_group is set"
    )
    assert all(
        group == "horizon_main_and_interactions" for group in ablation_calls
    ), "The ablation group must propagate into inner validation"


# Confirming completed metric counts reconcile with published predictions
def test_completed_metric_reconciles_with_predictions():
    from scripts.run_predictive_sensitivity import build_metric_record

    candidate = {
        "candidate_identity": "cand",
        "collapsed_candidate": {
            "design_scope": "Stage_1",
            "sampling_design_scope": "combined",
            "aging_mode": "calendar_aging",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "horizon_axis": "elapsed_calendar_days"
        }
    }

    variant = {
        "variant_identity": "v1",
        "variant_type": "partition",
        "model_family": "elastic_net"
    }

    predictions = [
        {
            "outer_fold_index": 0,
            "cell_identity": "cell_a",
            "source_identity": "s1",
            "observed_outcome": 1.0,
            "predicted_outcome": 0.99,
            "prediction_interval_lower": 0.95,
            "prediction_interval_upper": 1.05,
            "benchmark_prediction": 1.0,
            "condition_protocol_identity": "c"
        },
        {
            "outer_fold_index": 0,
            "cell_identity": "cell_a",
            "source_identity": "s2",
            "observed_outcome": 1.0,
            "predicted_outcome": 0.98,
            "prediction_interval_lower": 0.94,
            "prediction_interval_upper": 1.04,
            "benchmark_prediction": 1.0,
            "condition_protocol_identity": "c"
        },
        {
            "outer_fold_index": 1,
            "cell_identity": "cell_b",
            "source_identity": "s3",
            "observed_outcome": 1.0,
            "predicted_outcome": 0.97,
            "prediction_interval_lower": 0.93,
            "prediction_interval_upper": 1.03,
            "benchmark_prediction": 1.0,
            "condition_protocol_identity": "c"
        }
    ]

    policy = {"metric_contract": {"interval_probability": 0.9}}

    metric = build_metric_record(candidate, variant, policy, predictions, None)

    assert metric["validation_status"] == "completed"
    assert metric["test_cell_count"] == len({p["cell_identity"] for p in predictions})
    assert metric["test_observation_count"] == len(predictions)
    assert metric["outer_fold_count"] == len({p["outer_fold_index"] for p in predictions})


# Confirming partition folds separate training and test cells
def test_partition_folds_have_no_cell_leakage():
    from scripts.run_predictive_sensitivity import build_partition_fold_records

    candidate = {
        "included_observations": [
            {
                "cell_identity": f"c{c}",
                "condition_protocol_identity": "cond",
                "horizon": 1.0,
                "inside_evaluation_interval": True,
                "outcome_value": 1.0
            }
            for c in range(6)
        ]
    }

    assignments = {f"c{c}": c % 3 for c in range(6)}

    fold_records = build_partition_fold_records(candidate, assignments, 3)

    for fold in fold_records:
        training_cells = set(fold["training_cell_identities"])
        test_cells = set(fold["test_cell_identities"])

        assert training_cells.isdisjoint(test_cells), (
            f"Fold {fold['fold_index']} has cell leakage"
        )
        assert training_cells | test_cells == set(assignments), (
            f"Fold {fold['fold_index']} does not cover every cell"
        )


# Confirming the expanded grid contains every primary configuration, has no duplicates,
# and matches the estimator-specific schema
def test_expanded_grid_contains_primary_configurations_without_duplicates():
    from scripts.run_predictive_models import load_json_object as _primary_loader

    sensitivity_policy = load_policy()
    primary_policy = _primary_loader(
        PROJECT_ROOT / "configs" / "predictive_modeling_policy.json"
    )

    for family_name, family_policy in primary_policy["model_families"].items():
        expanded_configurations = sensitivity_policy[
            "expanded_configuration_inventory"
        ][family_name]["configurations"]

        # Containment
        for primary_configuration in family_policy["configurations"]:
            assert primary_configuration in expanded_configurations, (
                f"Expanded grid for {family_name} is missing a primary configuration"
            )

        # No duplicates
        serialized = [
            json.dumps(c, sort_keys = True) for c in expanded_configurations
        ]
        assert len(serialized) == len(set(serialized)), (
            f"Expanded grid for {family_name} contains duplicates"
        )

        # Schema match against estimator-specific keys
        expected_keys = set(family_policy["configurations"][0].keys())
        for config in expanded_configurations:
            assert set(config.keys()) == expected_keys, (
                f"Expanded configuration in {family_name} has a mismatched schema"
            )


# Confirming incident sensitivity also downgrades diagnostic-only variants
def test_incident_requires_analytical_availability():
    from scripts.run_predictive_sensitivity import (
        build_incident_rows,
        prediction_identity
    )

    metric = {
        "variant_identity": "v1",
        "variant_type": "incident",
        "candidate_identity": "cand",
        "model_family": "elastic_net",
        "validation_status": "completed",
        "analytical_status": "diagnostic_only",
        "analytical_reason_code": "PREDICTIVE_BENCHMARK_NOT_MET",
        "incident_eligibility": {
            "eligible_in_every_fold": True,
            "minimum_excluded_independent_cell_count": 2,
            "minimum_retained_independent_cell_count": 6
        }
    }

    primary_record = {
        "cell_identity": "cell_a",
        "source_identity": "src_a",
        "condition_protocol_identity": "c",
        "horizon": 1.0,
        "observed_outcome": 1.0,
        "predicted_outcome": 0.985,
        "benchmark_prediction": 1.0
    }

    prediction_records = [
        {
            **primary_record,
            "variant_identity": "v1",
            "outer_fold_index": 0
        }
    ]

    rows = build_incident_rows(
        [metric],
        prediction_records,
        {
            ("cand", "elastic_net"): {
                prediction_identity(primary_record): primary_record
            }
        },
        {
            ("cand", "elastic_net"): {"analytical_status": "available"}
        },
        {
            "v1": {
                "variant_type": "incident",
                "ablation_group": None,
                "incident_reason_code": "DOCUMENTED_INCIDENT"
            }
        }
    )

    assert len(rows) == 1
    assert rows[0]["comparison_status"] == "diagnostic_only"
    assert rows[0]["variant_analytical_status"] == "diagnostic_only"


# Confirming an inadequately supported incident reports unavailable even when observed
def test_inadequate_incident_reports_unavailable():
    from scripts.run_predictive_sensitivity import (
        build_incident_rows,
        prediction_identity
    )

    metric = {
        "variant_identity": "v1",
        "variant_type": "incident",
        "candidate_identity": "cand",
        "model_family": "elastic_net",
        "validation_status": "completed",
        "analytical_status": "available",
        "analytical_reason_code": None,
        "incident_eligibility": {
            "eligible_in_every_fold": False,
            "minimum_excluded_independent_cell_count": 1,
            "minimum_retained_independent_cell_count": 7
        }
    }

    primary_record = {
        "cell_identity": "cell_a",
        "source_identity": "src_a",
        "condition_protocol_identity": "c",
        "horizon": 1.0,
        "observed_outcome": 1.0,
        "predicted_outcome": 0.985,
        "benchmark_prediction": 1.0
    }

    prediction_records = [
        {
            **primary_record,
            "variant_identity": "v1",
            "outer_fold_index": 0
        }
    ]

    rows = build_incident_rows(
        [metric],
        prediction_records,
        {
            ("cand", "elastic_net"): {
                prediction_identity(primary_record): primary_record
            }
        },
        {
            ("cand", "elastic_net"): {"analytical_status": "available"}
        },
        {
            "v1": {
                "variant_type": "incident",
                "ablation_group": None,
                "incident_reason_code": "DOCUMENTED_INCIDENT"
            }
        }
    )

    assert len(rows) == 1
    assert rows[0]["comparison_status"] == "unavailable"
    assert rows[0]["comparison_reason_code"] == "INCIDENT_EXCLUSION_INSUFFICIENT_CELLS"
    assert rows[0]["eligible_in_every_fold"] is False


# Building a minimal valid publication fixture that satisfies every cross-table invariant for one partition variant
def build_minimal_valid_publication_tables() -> dict:
    variant_identity = "v1"
    candidate_identity = "cand"
    model_family = "elastic_net"

    registry_record = {
        "variant_identity": variant_identity,
        "candidate_identity": candidate_identity,
        "model_family": model_family,
        "variant_type": "partition",
        "partition_identity": "primary",
        "configuration_inventory": "primary",
        "training_window_identity": "full",
        "incident_reason_code": None,
        "ablation_group": None,
        "registration_order": 0
    }

    fold_record = {
        "variant_identity": variant_identity,
        "candidate_identity": candidate_identity,
        "model_family": model_family,
        "fold_index": 0,
        "training_cell_count": 2,
        "test_cell_count": 2,
        "training_cell_identities": ["train_a", "train_b"],
        "test_cell_identities": ["test_a", "test_b"],
        "training_observation_count": 4,
        "test_observation_count": 2,
        "excluded_test_observation_count": 0,
        "training_condition_count": 1,
        "fold_status": "completed",
        "fold_reason_code": None
    }

    selection = {
        "variant_identity": variant_identity,
        "candidate_identity": candidate_identity,
        "model_family": model_family,
        "outer_fold_index": 0,
        "configuration_identity": "cfg1",
        "configuration": {"alpha": 0.01, "l1_ratio": 0.5},
        "inner_fold_count": 2,
        "inner_validation_status": "completed",
        "inner_validation_reason_code": None,
        "equal_cell_weighted_root_mean_squared_error": 0.01,
        "selected": True
    }

    prediction_a = {
        "variant_identity": variant_identity,
        "variant_type": "partition",
        "candidate_identity": candidate_identity,
        "model_family": model_family,
        "outer_fold_index": 0,
        "cell_identity": "test_a",
        "source_identity": "src_test_a",
        "condition_protocol_identity": "cond_a",
        "horizon": 1.0,
        "observed_outcome": 1.0,
        "predicted_outcome": 0.99,
        "prediction_interval_lower": 0.95,
        "prediction_interval_upper": 1.05,
        "benchmark_prediction": 1.0,
        "prediction_status": "completed",
        "prediction_reason_code": None,
        "estimate_label": "predictive_sensitivity"
    }
    prediction_b = {
        **prediction_a,
        "cell_identity": "test_b",
        "source_identity": "src_test_b"
    }

    metric = {
        "variant_identity": variant_identity,
        "variant_type": "partition",
        "candidate_identity": candidate_identity,
        "model_family": model_family,
        "design_scope": "Stage_1",
        "sampling_design_scope": "combined",
        "aging_mode": "calendar_aging",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": "elapsed_calendar_days",
        "outer_fold_count": 1,
        "test_cell_count": 2,
        "test_observation_count": 2,
        "equal_cell_weighted_root_mean_squared_error": 0.01,
        "equal_cell_weighted_mean_absolute_error": 0.005,
        "equal_cell_weighted_predictive_r_squared": 0.9,
        "equal_cell_weighted_prediction_interval_coverage": 0.95,
        "equal_cell_weighted_mean_prediction_interval_width": 0.1,
        "validation_status": "completed",
        "validation_reason_code": None,
        "analytical_status": "available",
        "analytical_reason_code": None,
        "estimate_label": "predictive_sensitivity"
    }

    comparison = {
        "variant_identity": variant_identity,
        "variant_type": "partition",
        "candidate_identity": candidate_identity,
        "model_family": model_family,
        "design_scope": "Stage_1",
        "sampling_design_scope": "combined",
        "aging_mode": "calendar_aging",
        "outcome": "relative_capacity",
        "cohort": "primary",
        "horizon_axis": "elapsed_calendar_days",
        "shared_observation_count": 2,
        "shared_independent_cell_count": 2,
        "excluded_test_observation_count": 0,
        "primary_equal_cell_weighted_root_mean_squared_error": 0.01,
        "variant_equal_cell_weighted_root_mean_squared_error": 0.01,
        "root_mean_squared_error_difference": 0.0,
        "primary_equal_cell_weighted_mean_absolute_error": 0.005,
        "variant_equal_cell_weighted_mean_absolute_error": 0.005,
        "mean_absolute_error_difference": 0.0,
        "primary_equal_cell_weighted_predictive_r_squared": 0.9,
        "variant_equal_cell_weighted_predictive_r_squared": 0.9,
        "predictive_r_squared_difference": 0.0,
        "primary_inventory_count": 2,
        "variant_inventory_count": 2,
        "exact_inventory_match": True,
        "variant_analytical_status": "available",
        "comparison_status": "completed",
        "comparison_reason_code": None,
        "estimate_label": "predictive_sensitivity"
    }

    overall_interval = {
        "variant_identity": variant_identity,
        "variant_type": "partition",
        "candidate_identity": candidate_identity,
        "model_family": model_family,
        "grouping_dimension": "overall",
        "grouping_value": "all",
        "nominal_interval_probability": 0.9,
        "prediction_count": 2,
        "independent_cell_count": 2,
        "covered_prediction_count": 2,
        "equal_cell_weighted_interval_coverage": 1.0,
        "equal_cell_weighted_mean_interval_width": 0.1,
        "assessment_status": "completed",
        "assessment_reason_code": None,
        "estimate_label": "predictive_sensitivity"
    }
    fold_interval = {
        **overall_interval,
        "grouping_dimension": "validation_fold",
        "grouping_value": "0"
    }
    condition_interval = {
        **overall_interval,
        "grouping_dimension": "condition_protocol",
        "grouping_value": "cond_a"
    }

    condition_calibration = {
        "variant_identity": variant_identity,
        "variant_type": "partition",
        "candidate_identity": candidate_identity,
        "model_family": model_family,
        "condition_protocol_identity": "cond_a",
        "independent_cell_count": 2,
        "observation_count": 2,
        "equal_cell_weighted_mean_observed": 1.0,
        "equal_cell_weighted_mean_predicted": 0.99,
        "equal_cell_weighted_mean_error": -0.01,
        "equal_cell_weighted_mean_absolute_error": 0.01,
        "equal_cell_weighted_root_mean_squared_error": 0.01,
        "calibration_status": "completed",
        "calibration_reason_code": None,
        "estimate_label": "predictive_sensitivity"
    }

    return {
        "sensitivity_registry": [registry_record],
        "validation_folds": [fold_record],
        "configuration_selection": [selection],
        "validation_predictions": [prediction_a, prediction_b],
        "validation_metrics": [metric],
        "sensitivity_comparisons": [comparison],
        "condition_calibration": [condition_calibration],
        "predictor_ablation": [],
        "incident_exclusions": [],
        "prediction_interval_diagnostics": [
            overall_interval,
            fold_interval,
            condition_interval
        ]
    }


# Confirming the fixture passes the publication invariant validator unchanged
def test_minimal_valid_publication_passes_invariants(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    tables = build_minimal_valid_publication_tables()
    args = argparse.Namespace(output_directory = tmp_path / "out")

    module.publish_outputs(args, policy, policy_sha256, {}, tables, {})


# Rejecting a publication whose sensitivity registry contains duplicate variants
def test_publish_rejects_duplicate_registry_identity(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    tables = build_minimal_valid_publication_tables()
    tables["sensitivity_registry"].append(dict(tables["sensitivity_registry"][0]))

    args = argparse.Namespace(output_directory = tmp_path / "out")

    with pytest.raises(RuntimeError, match = "duplicate variant identities"):
        module.publish_outputs(args, policy, policy_sha256, {}, tables, {})


# Rejecting a publication whose sensitivity metrics contain duplicate variants
def test_publish_rejects_duplicate_metric_identity(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    tables = build_minimal_valid_publication_tables()
    tables["validation_metrics"].append(dict(tables["validation_metrics"][0]))

    args = argparse.Namespace(output_directory = tmp_path / "out")

    with pytest.raises(RuntimeError, match = "duplicate variant identities"):
        module.publish_outputs(args, policy, policy_sha256, {}, tables, {})


# Rejecting a publication whose completed metric observation count does not match
#  its published predictions
def test_publish_rejects_metric_observation_count_mismatch(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    tables = build_minimal_valid_publication_tables()
    tables["validation_metrics"][0]["test_observation_count"] = 99

    args = argparse.Namespace(output_directory = tmp_path / "out")

    with pytest.raises(RuntimeError, match = "observation count does not match"):
        module.publish_outputs(args, policy, policy_sha256, {}, tables, {})


# Rejecting a publication whose validation fold leaks training cells into test cells
def test_publish_rejects_fold_leakage(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    tables = build_minimal_valid_publication_tables()
    tables["validation_folds"][0]["training_cell_identities"] = ["test_a", "train_b"]

    args = argparse.Namespace(output_directory = tmp_path / "out")

    with pytest.raises(RuntimeError, match = "cell leakage"):
        module.publish_outputs(args, policy, policy_sha256, {}, tables, {})


# Rejecting a publication where a completed fold has no selected configuration
def test_publish_rejects_missing_selected_configuration(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    tables = build_minimal_valid_publication_tables()
    tables["configuration_selection"][0]["selected"] = False

    args = argparse.Namespace(output_directory = tmp_path / "out")

    with pytest.raises(RuntimeError, match = "exactly one selected configuration"):
        module.publish_outputs(args, policy, policy_sha256, {}, tables, {})


# Rejecting a publication where a completed fold has multiple selected configurations
def test_publish_rejects_multiple_selected_configurations(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    tables = build_minimal_valid_publication_tables()
    duplicate_selection = dict(tables["configuration_selection"][0])
    duplicate_selection["configuration_identity"] = "cfg2"
    tables["configuration_selection"].append(duplicate_selection)

    args = argparse.Namespace(output_directory = tmp_path / "out")

    with pytest.raises(RuntimeError, match = "exactly one selected configuration"):
        module.publish_outputs(args, policy, policy_sha256, {}, tables, {})


# Rejecting a publication that carries a prediction for an unregistered fold
def test_publish_rejects_prediction_against_missing_fold(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    tables = build_minimal_valid_publication_tables()
    tables["validation_predictions"][0]["outer_fold_index"] = 99

    args = argparse.Namespace(output_directory = tmp_path / "out")

    with pytest.raises(RuntimeError, match = "missing or unavailable fold"):
        module.publish_outputs(args, policy, policy_sha256, {}, tables, {})


# Rejecting a publication that omits the single required comparison for a variant
def test_publish_rejects_missing_comparison_record(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    tables = build_minimal_valid_publication_tables()
    tables["sensitivity_comparisons"] = []

    args = argparse.Namespace(output_directory = tmp_path / "out")

    with pytest.raises(RuntimeError, match = "exactly one comparison record"):
        module.publish_outputs(args, policy, policy_sha256, {}, tables, {})


# Rejecting a publication that omits the overall interval record for a variant
def test_publish_rejects_missing_overall_interval_record(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    tables = build_minimal_valid_publication_tables()
    tables["prediction_interval_diagnostics"] = [
        record
        for record in tables["prediction_interval_diagnostics"]
        if record["grouping_dimension"] != "overall"
    ]

    args = argparse.Namespace(output_directory = tmp_path / "out")

    with pytest.raises(RuntimeError, match = "exactly one overall interval record"):
        module.publish_outputs(args, policy, policy_sha256, {}, tables, {})


# Rejecting a publication that omits fold-level interval records for a completed variant
def test_publish_rejects_missing_fold_interval_record(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    tables = build_minimal_valid_publication_tables()
    tables["prediction_interval_diagnostics"] = [
        record
        for record in tables["prediction_interval_diagnostics"]
        if record["grouping_dimension"] != "validation_fold"
    ]

    args = argparse.Namespace(output_directory = tmp_path / "out")

    with pytest.raises(RuntimeError, match = "missing fold-level interval records"):
        module.publish_outputs(args, policy, policy_sha256, {}, tables, {})


# Rejecting a publication that declares an ablation variant without its summary row
def test_publish_rejects_missing_ablation_summary(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    tables = build_minimal_valid_publication_tables()
    tables["sensitivity_registry"][0]["variant_type"] = "ablation"
    tables["sensitivity_registry"][0]["ablation_group"] = "horizon_main_and_interactions"
    tables["predictor_ablation"] = []

    args = argparse.Namespace(output_directory = tmp_path / "out")

    with pytest.raises(RuntimeError, match = "Ablation summary coverage"):
        module.publish_outputs(args, policy, policy_sha256, {}, tables, {})


# Rejecting a publication that declares an incident variant without its summary row
def test_publish_rejects_missing_incident_summary(tmp_path):
    from scripts import run_predictive_sensitivity as module

    policy = load_policy()
    policy_sha256 = hashlib.sha256(
        (PROJECT_ROOT / "configs" / "predictive_sensitivity_policy.json").read_bytes()
    ).hexdigest()

    tables = build_minimal_valid_publication_tables()
    tables["sensitivity_registry"][0]["variant_type"] = "incident"
    tables["sensitivity_registry"][0]["incident_reason_code"] = "DOCUMENTED_INCIDENT"
    tables["incident_exclusions"] = []

    args = argparse.Namespace(output_directory = tmp_path / "out")

    with pytest.raises(RuntimeError, match = "Incident summary coverage"):
        module.publish_outputs(args, policy, policy_sha256, {}, tables, {})


# Confirming that the interval-diagnostics table covers unavailable variants too
def test_build_interval_diagnostics_emits_unavailable_record():
    from scripts.run_predictive_sensitivity import build_interval_diagnostics

    policy = load_policy()

    metric = {
        "variant_identity": "v_unavail",
        "variant_type": "partition",
        "candidate_identity": "cand",
        "model_family": "elastic_net",
        "validation_status": "unavailable",
        "validation_reason_code": "INSUFFICIENT_GROUPED_PARTITION_SUPPORT",
        "analytical_status": "not_available",
        "analytical_reason_code": "INSUFFICIENT_GROUPED_PARTITION_SUPPORT"
    }

    rows = build_interval_diagnostics(
        [metric],
        [],
        policy,
        {"cand": ["condition_a"]}
    )

    assert len(rows) == 1
    assert rows[0]["grouping_dimension"] == "overall"
    assert rows[0]["assessment_status"] == "unavailable"
    assert rows[0]["assessment_reason_code"] == "INSUFFICIENT_GROUPED_PARTITION_SUPPORT"
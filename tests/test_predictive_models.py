import math

from pathlib import Path

import numpy as np
import pytest

from scripts.run_predictive_models import (
    EstimatorUnavailableError,
    build_cell_weights,
    build_estimator,
    build_feature_importance,
    build_feature_matrix,
    build_inner_assignments,
    build_unavailable_feature_importance_records,
    calculate_condition_benchmark,
    candidate_matches_selection_policy,
    equal_cell_metrics,
    feature_importance_method,
    fit_estimator,
    fold_has_horizon_coverage,
    load_json_object,
    outer_predictions_reconcile,
    select_final_configuration_identity,
    validate_policy,
    weighted_quantile,
    validate_runtime_policy_bindings,
    validate_fit_table_reconciliation,
    deterministic_identity
)


# Confirming normalized equal-cell observation weights
def test_cell_weights_assign_equal_total_weight_to_each_cell():
    observations = [
        {
            "cell_identity": "cell_1"
        },
        {
            "cell_identity": "cell_1"
        },
        {
            "cell_identity": "cell_2"
        }
    ]

    weights = build_cell_weights(
        observations
    )

    assert math.isclose(
        weights[
            0
        ]
        + weights[
            1
        ],
        weights[
            2
        ]
    )

    assert math.isclose(
        float(
            np.mean(
                weights
            )
        ),
        1.0
    )


# Confirming registered condition and interaction columns
def test_feature_matrix_uses_registered_condition_and_interaction_terms():
    observations = [
        {
            "condition_protocol_identity": "condition_a",
            "horizon": 1.0
        },
        {
            "condition_protocol_identity": "condition_b",
            "horizon": 3.0
        }
    ]

    matrix, feature_names = build_feature_matrix(
        observations = observations,
        condition_identities = [
            "condition_a",
            "condition_b"
        ],
        horizon_midpoint = 2.0,
        horizon_width = 2.0
    )

    assert feature_names == [
        "scaled_horizon",
        "condition[condition_b]",
        "scaled_horizon:condition[condition_b]"
    ]

    np.testing.assert_allclose(
        matrix,
        np.asarray(
            [
                [
                    -0.5,
                    0.0,
                    0.0
                ],
                [
                    0.5,
                    1.0,
                    0.5
                ]
            ]
        )
    )


# Rejecting observations with unregistered conditions
def test_feature_matrix_rejects_unregistered_condition():
    with pytest.raises(
        RuntimeError,
        match = "unregistered condition"
    ):
        build_feature_matrix(
            observations = [
                {
                    "condition_protocol_identity": "condition_c",
                    "horizon": 1.0
                }
            ],
            condition_identities = [
                "condition_a",
                "condition_b"
            ],
            horizon_midpoint = 1.0,
            horizon_width = 1.0
        )


# Confirming cells are never split across inner folds
def test_inner_assignments_never_split_one_cell_across_folds():
    observations = []

    for condition_identity in (
        "condition_a",
        "condition_b"
    ):
        for cell_number in range(
            4
        ):
            for horizon in (
                1.0,
                2.0
            ):
                observations.append(
                    {
                        "condition_protocol_identity": condition_identity,
                        "cell_identity": (
                            f"{condition_identity}_cell_{cell_number}"
                        ),
                        "horizon": horizon
                    }
                )

    policy = {
        "nested_validation": {
            "inner_requested_fold_count": 3,
            "inner_minimum_fold_count": 2,
            "inner_fold_assignment_seed": 19
        }
    }

    assignments, fold_count = build_inner_assignments(
        candidate_identity = "candidate",
        observations = observations,
        policy = policy
    )

    assert fold_count == 3

    for observation in observations:
        assert observation[
            "cell_identity"
        ] in assignments


# Confirming deterministic inner-fold assignment
def test_inner_assignments_are_deterministic():
    observations = [
        {
            "condition_protocol_identity": "condition_a",
            "cell_identity": f"cell_{cell_number}",
            "horizon": 1.0
        }
        for cell_number in range(
            5
        )
    ]

    policy = {
        "nested_validation": {
            "inner_requested_fold_count": 3,
            "inner_minimum_fold_count": 2,
            "inner_fold_assignment_seed": 31
        }
    }

    first, first_count = build_inner_assignments(
        "candidate",
        observations,
        policy
    )

    second, second_count = build_inner_assignments(
        "candidate",
        observations,
        policy
    )

    assert first_count == second_count
    assert first == second


# Confirming weighted quantile respects observation weights
def test_weighted_quantile_respects_observation_weights():
    values = np.asarray(
        [
            1.0,
            2.0,
            10.0
        ]
    )

    weights = np.asarray(
        [
            1.0,
            1.0,
            8.0
        ]
    )

    assert weighted_quantile(
        values,
        weights,
        0.5
    ) == 10.0


# Confirming equal-cell weighting prevents observation over-counting
def test_equal_cell_metrics_do_not_overweight_cells_with_more_rows():
    observations = [
        {
            "cell_identity": "cell_1"
        },
        {
            "cell_identity": "cell_1"
        },
        {
            "cell_identity": "cell_2"
        }
    ]

    observed = np.asarray(
        [
            1.0,
            1.0,
            1.0
        ]
    )

    predicted = np.asarray(
        [
            1.0,
            1.0,
            2.0
        ]
    )

    metrics = equal_cell_metrics(
        observations,
        observed,
        predicted
    )

    assert math.isclose(
        metrics[
            "equal_cell_weighted_mean_absolute_error"
        ],
        0.5
    )


# Confirming predictive R-squared compares against the supplied benchmark
def test_predictive_r_squared_compares_against_supplied_benchmark():
    observations = [
        {
            "cell_identity": "cell_1"
        },
        {
            "cell_identity": "cell_2"
        }
    ]

    observed = np.asarray(
        [
            1.0,
            2.0
        ]
    )

    predicted = np.asarray(
        [
            1.0,
            2.0
        ]
    )

    benchmark = np.asarray(
        [
            1.5,
            1.5
        ]
    )

    metrics = equal_cell_metrics(
        observations,
        observed,
        predicted,
        benchmark = benchmark
    )

    assert metrics[
        "equal_cell_weighted_predictive_r_squared"
    ] == 1.0


# Confirming interval metrics on the original outcome scale
def test_prediction_interval_metrics_are_computed_on_original_scale():
    observations = [
        {
            "cell_identity": "cell_1"
        },
        {
            "cell_identity": "cell_2"
        }
    ]

    observed = np.asarray(
        [
            0.9,
            1.1
        ]
    )

    predicted = np.asarray(
        [
            1.0,
            1.0
        ]
    )

    lower = np.asarray(
        [
            0.8,
            0.8
        ]
    )

    upper = np.asarray(
        [
            1.2,
            1.2
        ]
    )

    metrics = equal_cell_metrics(
        observations,
        observed,
        predicted,
        lower,
        upper
    )

    assert metrics[
        "prediction_interval_coverage"
    ] == 1.0

    assert math.isclose(
        metrics[
            "mean_prediction_interval_width"
        ],
        0.4
    )


# Confirming the registered predictive policy passes validation
def test_registered_policy_passes_validation():
    project_root = Path(
        __file__
    ).resolve().parents[
        1
    ]

    policy = load_json_object(
        project_root
        / "configs"
        / "predictive_modeling_policy.json"
    )

    validate_policy(
        policy
    )


# Confirming candidate selection is driven by registered policy fields
def test_candidate_selection_uses_registered_fields():
    policy = {
        "inputs": {
            "candidate_selection_fields": [
                "admissibility_status",
                "grouped_validation_status"
            ],
            "candidate_selection_values": {
                "admissibility_status": "admissible",
                "grouped_validation_status": "available"
            }
        }
    }

    assert candidate_matches_selection_policy(
        {
            "admissibility_status": "admissible",
            "grouped_validation_status": "available"
        },
        policy
    )

    assert not candidate_matches_selection_policy(
        {
            "admissibility_status": "inadmissible",
            "grouped_validation_status": "available"
        },
        policy
    )


# Confirming the condition benchmark gives each training cell equal weight
def test_condition_benchmark_gives_each_training_cell_equal_weight():
    training = [
        {
            "cell_identity": "cell_1",
            "condition_protocol_identity": "condition",
            "outcome_value": 1.0
        },
        {
            "cell_identity": "cell_1",
            "condition_protocol_identity": "condition",
            "outcome_value": 1.0
        },
        {
            "cell_identity": "cell_2",
            "condition_protocol_identity": "condition",
            "outcome_value": 3.0
        }
    ]

    prediction = calculate_condition_benchmark(
        training,
        [
            {
                "condition_protocol_identity": "condition"
            }
        ]
    )

    assert prediction[
        0
    ] == 2.0


# Confirming horizon coverage is evaluated per condition
def test_horizon_coverage_is_condition_specific():
    training = [
        {
            "condition_protocol_identity": "a",
            "horizon": 1.0
        },
        {
            "condition_protocol_identity": "a",
            "horizon": 3.0
        },
        {
            "condition_protocol_identity": "b",
            "horizon": 10.0
        },
        {
            "condition_protocol_identity": "b",
            "horizon": 20.0
        }
    ]

    assert fold_has_horizon_coverage(
        training,
        [
            {
                "condition_protocol_identity": "a",
                "horizon": 2.0
            },
            {
                "condition_protocol_identity": "b",
                "horizon": 15.0
            }
        ]
    )

    assert not fold_has_horizon_coverage(
        training,
        [
            {
                "condition_protocol_identity": "a",
                "horizon": 15.0
            }
        ]
    )


# Rejecting incomplete outer prediction inventories
def test_outer_prediction_reconciliation_rejects_partial_results():
    observations = [
        {
            "condition_protocol_identity": "condition",
            "cell_identity": "cell_1",
            "horizon": 1.0,
            "source_identity": "source_1",
            "inside_evaluation_interval": True
        },
        {
            "condition_protocol_identity": "condition",
            "cell_identity": "cell_2",
            "horizon": 1.0,
            "source_identity": "source_2",
            "inside_evaluation_interval": True
        }
    ]

    predictions = [
        {
            "condition_protocol_identity": "condition",
            "cell_identity": "cell_1",
            "horizon": 1.0,
            "source_identity": "source_1"
        }
    ]

    assert not outer_predictions_reconcile(
        observations,
        predictions
    )


# Rejecting duplicate outer predictions
def test_outer_prediction_reconciliation_rejects_duplicates():
    observation = {
        "condition_protocol_identity": "condition",
        "cell_identity": "cell_1",
        "horizon": 1.0,
        "source_identity": "source_1",
        "inside_evaluation_interval": True
    }

    prediction = {
        key: observation[
            key
        ]
        for key in (
            "condition_protocol_identity",
            "cell_identity",
            "horizon",
            "source_identity"
        )
    }

    assert not outer_predictions_reconcile(
        [
            observation
        ],
        [
            prediction,
            prediction
        ]
    )


# Confirming the final-inspection configuration rule is deterministic
def test_final_configuration_rule_is_deterministic():
    assert select_final_configuration_identity(
        [
            "b",
            "a",
            "b",
            "a"
        ]
    ) == "a"

    assert select_final_configuration_identity(
        [
            "b",
            "b",
            "a"
        ]
    ) == "b"


# Confirming ElasticNet scaling uses registered equal-cell weights
def test_elastic_net_scaler_uses_registered_weights():
    policy = {
        "execution": {
            "random_seed": 17
        },
        "model_families": {
            "elastic_net": {
                "maximum_iterations": 20000,
                "tolerance": 1e-8
            }
        }
    }

    estimator = build_estimator(
        "elastic_net",
        {
            "alpha": 0.0001,
            "l1_ratio": 0.1
        },
        "test",
        policy
    )

    matrix = np.asarray(
        [
            [
                0.0
            ],
            [
                10.0
            ]
        ]
    )

    fit_estimator(
        estimator,
        "elastic_net",
        matrix,
        np.asarray(
            [
                0.0,
                0.1
            ]
        ),
        np.asarray(
            [
                3.0,
                1.0
            ]
        )
    )

    assert math.isclose(
        estimator.named_steps[
            "scale"
        ].mean_[
            0
        ],
        2.5
    )


# Confirming the candidate registry schema includes every scope field
def test_candidate_registry_schema_includes_scope_fields() -> None:
    from scripts.run_predictive_models import (
        CANDIDATE_REGISTRY_FIELDS
    )

    required_scope_fields = {
        "support_region_identity",
        "design_scope",
        "sampling_design_scope",
        "support_interval_minimum",
        "support_interval_maximum",
        "support_interval_lower_inclusive",
        "support_interval_upper_inclusive",
        "condition_protocol_identities"
    }

    missing_fields = sorted(
        required_scope_fields.difference(
            CANDIDATE_REGISTRY_FIELDS
        )
    )

    if missing_fields:
        raise AssertionError("Candidate registry is missing required scope fields : ", str(missing_fields))


# Confirming unqualified models produce explicit unavailable importance records
def test_unqualified_model_importance_is_recorded_unavailable():
    records = build_unavailable_feature_importance_records(
        benchmark_identity = "benchmark",
        candidate_identity = "candidate",
        model_family = "elastic_net",
        configuration_identity = "configuration",
        feature_names = [
            "scaled_horizon",
            "condition[condition_b]"
        ],
        reason_code = "MODEL_NOT_ANALYTICALLY_QUALIFIED"
    )

    assert len(records) == 2

    assert all(
        record["importance_status"] == "unavailable"
        for record in records
    )

    assert all(
        record["importance_value"] is None
        for record in records
    )

    assert all(
        record["importance_reason_code"]
        == "MODEL_NOT_ANALYTICALLY_QUALIFIED"
        for record in records
    )


# Confirming registered importance methods are family-specific
@pytest.mark.parametrize(
    (
        "model_family",
        "expected_method"
    ),
    [
        (
            "elastic_net",
            "absolute_standardized_coefficient"
        ),
        (
            "random_forest",
            "impurity_based_feature_importance"
        ),
        (
            "gradient_boosting",
            "impurity_based_feature_importance"
        )
    ]
)
def test_feature_importance_method_is_registered(model_family, expected_method):
    assert feature_importance_method(
        model_family
    ) == expected_method


# Rejecting a policy that enables outcome-informed candidate selection
def test_policy_rejects_outcome_informed_candidate_selection():
    project_root = Path(
        __file__
    ).resolve().parents[1]

    policy = load_json_object(
        project_root
        / "configs"
        / "predictive_modeling_policy.json"
    )

    policy[
        "inputs"
    ][
        "outcome_magnitude_direction_or_performance_used_for_candidate_selection"
    ] = True

    with pytest.raises(
        RuntimeError,
        match = "input restriction"
    ):
        validate_policy(policy)


# Rejecting causal interpretation of model inspection
def test_policy_rejects_causal_feature_importance():
    project_root = Path(
        __file__
    ).resolve().parents[1]

    policy = load_json_object(
        project_root
        / "configs"
        / "predictive_modeling_policy.json"
    )

    policy[
        "explainability"
    ][
        "causal_interpretation_authorized"
    ] = True

    with pytest.raises(
        RuntimeError,
        match = "explainability policy"
    ):
        validate_policy(policy)


# Rejecting an invalid prediction-interval probability
def test_policy_rejects_invalid_prediction_interval_probability():
    project_root = Path(
        __file__
    ).resolve().parents[1]

    policy = load_json_object(
        project_root
        / "configs"
        / "predictive_modeling_policy.json"
    )

    policy[
        "prediction_intervals"
    ][
        "coverage_probability"
    ] = 1.0

    with pytest.raises(
        RuntimeError,
        match = "Prediction-interval policy"
    ):
        validate_policy(policy)


# Building a small two-condition fixture for end-to-end nested validation
def _build_small_workflow() -> dict:
    condition_identities = [
        "Stage_1/TP_k01/periodic_reference_tests",
        "Stage_1/TP_k02/periodic_reference_tests",
    ]

    observations = []

    for condition_index, condition_identity in enumerate(
        condition_identities
    ):
        for cell_number in range(
            1,
            5
        ):
            cell_identity = (
                f"Stage_1/TP_k{condition_index + 1:02d}_{cell_number:02d}"
            )

            for horizon in (
                10.0,
                30.0
            ):
                observations.append(
                    {
                        "cell_identity": cell_identity,
                        "condition_protocol_identity": (
                            condition_identity
                        ),
                        "horizon": horizon,
                        "outcome_value": (
                            math.exp(
                                -0.01 * horizon
                                + 0.005 * condition_index
                                + 0.001 * cell_number
                            )
                        ),
                        "source_identity": (
                            f"{cell_identity}/h{int(horizon)}.csv"
                        ),
                        "inside_evaluation_interval": True,
                    }
                )

    candidate = {
        "candidate_identity": "test_candidate",
        "condition_identities": condition_identities,
        "included_observations": observations,
        "collapsed_candidate": {
            "support_region_identity": "region_1",
            "design_scope": "Stage_1",
            "sampling_design_scope": "combined",
            "outcome": "relative_capacity",
            "cohort": "primary",
            "aging_mode": "calendar_aging",
            "horizon_axis": "elapsed_calendar_days",
            "support_interval_minimum": "0.0",
            "support_interval_maximum": "100.0",
            "support_interval_lower_inclusive": "true",
            "support_interval_upper_inclusive": "true",
        },
    }

    fold_records = []

    for fold_index in range(
        2
    ):
        training_cell_identities = sorted(set(
            observation["cell_identity"]
            for observation in observations
            if observation["cell_identity"]
            != f"Stage_1/TP_k01_{fold_index * 2 + 1:02d}"
            and observation["cell_identity"]
            != f"Stage_1/TP_k01_{fold_index * 2 + 2:02d}"
            and observation["cell_identity"]
            != f"Stage_1/TP_k02_{fold_index * 2 + 1:02d}"
            and observation["cell_identity"]
            != f"Stage_1/TP_k02_{fold_index * 2 + 2:02d}"
        ))

        test_cell_identities = sorted([
            f"Stage_1/TP_k01_{fold_index * 2 + 1:02d}",
            f"Stage_1/TP_k01_{fold_index * 2 + 2:02d}",
            f"Stage_1/TP_k02_{fold_index * 2 + 1:02d}",
            f"Stage_1/TP_k02_{fold_index * 2 + 2:02d}",
        ])

        fold_records.append(
            {
                "candidate_identity": "test_candidate",
                "fold_index": fold_index,
                "fold_status": "available",
                "fold_reason_code": None,
                "training_cell_identities": (
                    training_cell_identities
                ),
                "test_cell_identities": (
                    test_cell_identities
                ),
                "training_cell_count": len(
                    training_cell_identities
                ),
                "test_cell_count": len(
                    test_cell_identities
                ),
                "training_observation_count": 8,
                "test_observation_count": 8,
                "training_condition_count": 2,
                "training_minimum_condition_cell_count": 2,
                "training_minimum_condition_repeated_cell_count": 2,
            }
        )

    return {
        "candidate_records": [candidate],
        "component_candidates": [candidate],
        "fold_records": fold_records,
    }


# Building the minimal policy required by fit_benchmarks
def _build_small_policy() -> dict:
    return {
        "execution": {
            "random_seed": 904271,
            "parallel_worker_count": 1,
            "operations": [
                "validate",
                "fit"
            ],
            "software_environment": {
                "python_version": "3.11.16",
                "package_versions": {
                    "joblib": "1.6.0",
                    "numpy": "2.4.6",
                    "scikit-learn": "1.9.1",
                    "scipy": "1.17.1"
                },
                "dependency_lock_path": "uv.lock",
                "required_environment_variables": {
                    "PYTHONHASHSEED": "0",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                    "BLIS_NUM_THREADS": "1"
                }
            }
        },
        "model_families": {
            "elastic_net": {
                "estimator": "ElasticNet",
                "maximum_iterations": 20000,
                "tolerance": 1e-8,
                "configuration_rationale": "test",
                "configurations": [
                    {
                        "alpha": 0.0001,
                        "l1_ratio": 0.1
                    }
                ]
            }
        },
        "nested_validation": {
            "inner_requested_fold_count": 2,
            "inner_minimum_fold_count": 2,
            "inner_fold_assignment_seed": 190731
        },
        "prediction_intervals": {
            "coverage_probability": 0.9
        },
        "metrics": {
            "analytical_qualification": {
                "all_outer_folds_completed_required": True,
                "strictly_positive_finite_predictions_required": True,
                "predictive_r_squared_must_exceed_benchmark": True,
                "qualified_status": "available",
                "failed_status": "diagnostic_only"
            }
        },
        "explainability": {
            "analytical_status_required": "available"
        }
    }


# Confirming fit_benchmarks produces a complete fit for a small fixture
def test_fit_benchmarks_completes_for_small_fixture():
    from scripts.run_predictive_models import (
        fit_benchmarks
    )

    workflow = _build_small_workflow()
    policy = _build_small_policy()

    outputs = fit_benchmarks(
        selected_candidates = workflow["component_candidates"],
        workflow = workflow,
        policy = policy
    )

    assert len(outputs["candidate_registry"]) == 1
    assert (
        outputs["candidate_registry"][0]["execution_status"]
        == "completed"
    )

    registry = outputs["candidate_registry"][0]
    registered_cells = {
        observation["cell_identity"]
        for observation in workflow["component_candidates"][0]["included_observations"]
    }

    # Every fold preserves cell identities
    for fold in workflow["fold_records"]:
        assert set(fold["training_cell_identities"]).isdisjoint(
            set(fold["test_cell_identities"])
        )

    # Feature importance exists for the benchmark
    assert len(outputs["feature_importance"]) == (
        2 * len(workflow["component_candidates"][0]["condition_identities"]) - 1
    )


# Confirming a family that fails inner validation still records unavailable importance
def test_family_failure_records_unavailable_importance(monkeypatch: pytest.MonkeyPatch):
    from scripts import run_predictive_models as predictive_models

    workflow = _build_small_workflow()
    policy = _build_small_policy()

    # Force estimator fitting to fail so family_failure_reason is set
    original_fit = predictive_models.fit_estimator

    def failing_fit(estimator, model_family, matrix, target, weights):
        raise predictive_models.EstimatorUnavailableError(
            "ESTIMATOR_FIT_FAILED"
        )

    monkeypatch.setattr(
        predictive_models,
        "fit_estimator",
        failing_fit
    )

    outputs = predictive_models.fit_benchmarks(
        selected_candidates = workflow["component_candidates"],
        workflow = workflow,
        policy = policy
    )

    assert len(outputs["candidate_registry"]) == 1
    assert (
        outputs["candidate_registry"][0]["execution_status"]
        == "unavailable"
    )

    # Unavailable importance records must still be emitted
    assert len(outputs["feature_importance"]) > 0
    assert all(
        record["importance_status"] == "unavailable"
        for record in outputs["feature_importance"]
    )


# Confirming published folds carry complete cell identities
def test_published_fold_records_retain_cell_identities():
    from scripts.run_predictive_models import (
        build_validation_records
    )

    workflow = _build_small_workflow()

    accounting, folds = build_validation_records(
        selected_candidates = workflow["component_candidates"],
        workflow = workflow
    )

    assert len(folds) == 2

    for fold in folds:
        assert set(fold["training_cell_identities"]).isdisjoint(
            set(fold["test_cell_identities"])
        )
        assert len(fold["training_cell_identities"]) == fold["training_cell_count"]
        assert len(fold["test_cell_identities"]) == fold["test_cell_count"]


# Rejecting a policy with missing_data.imputation_authorized
def test_policy_rejects_missing_data_imputation():
    project_root = Path(__file__).resolve().parents[1]

    policy = load_json_object(
        project_root
        / "configs"
        / "predictive_modeling_policy.json"
    )

    policy["missing_data"]["imputation_authorized"] = True

    with pytest.raises(
        RuntimeError,
        match = "missing-data contract"
    ):
        validate_policy(policy)


# Rejecting a policy with a modified weighting rule
def test_policy_rejects_modified_weighting():
    project_root = Path(__file__).resolve().parents[1]

    policy = load_json_object(
        project_root
        / "configs"
        / "predictive_modeling_policy.json"
    )

    policy["weighting"]["training_weight_rule"] = "uniform_observation_weight"

    with pytest.raises(
        RuntimeError,
        match = "weighting contract"
    ):
        validate_policy(policy)


# Rejecting a policy with a modified analytical-qualification contract
def test_policy_rejects_modified_analytical_qualification():
    project_root = Path(__file__).resolve().parents[1]

    policy = load_json_object(
        project_root
        / "configs"
        / "predictive_modeling_policy.json"
    )

    policy["metrics"]["analytical_qualification"]["qualified_status"] = "accepted"

    with pytest.raises(
        RuntimeError,
        match = "analytical-qualification contract"
    ):
        validate_policy(policy)


# Rejecting a policy with a missing software environment variable
def test_policy_rejects_missing_software_environment_variable():
    project_root = Path(__file__).resolve().parents[1]

    policy = load_json_object(
        project_root
        / "configs"
        / "predictive_modeling_policy.json"
    )

    del policy["execution"]["software_environment"]["required_environment_variables"]["PYTHONHASHSEED"]

    with pytest.raises(
        RuntimeError,
        match = "PYTHONHASHSEED"
    ):
        validate_policy(policy)


# Confirming software-environment validation rejects a mismatched package version
def test_software_environment_validation_rejects_mismatched_version(monkeypatch: pytest.MonkeyPatch):
    from scripts import run_predictive_models as predictive_models

    policy = _build_small_policy()

    def mismatched_version(package_name):
        return "0.0.0" if package_name == "scikit-learn" else {
            "joblib": "1.6.0",
            "numpy": "2.4.6",
            "scipy": "1.17.1"
        }.get(package_name, "0.0.0")

    monkeypatch.setattr(
        predictive_models,
        "distribution_version",
        mismatched_version
    )

    with pytest.raises(
        RuntimeError,
        match = "Package versions do not match"
    ):
        predictive_models.validate_software_environment(
            policy
        )


# Building a valid reconciliation fixture
def _build_valid_reconciliation_tables() -> dict:
    configuration = {
        "alpha": 0.1,
        "l1_ratio": 0.5
    }

    configuration_identity = deterministic_identity(
        {
            "model_family": "elastic_net",
            "configuration": configuration
        }
    )

    return {
        "candidate_registry": [
            {
                "benchmark_identity": "b1",
                "candidate_identity": "c1",
                "model_family": "elastic_net",
                "condition_count": 2,
                "outer_fold_count": 2,
                "condition_protocol_identities": [
                    "a",
                    "b"
                ],
                "analytical_status": "diagnostic_only",
                "analytical_reason_code": "PREDICTIVE_BENCHMARK_NOT_MET",
                "execution_status": "completed",
                "selected_configuration_identities": [
                    configuration_identity,
                    configuration_identity
                ]
            }
        ],
        "validation_metrics": [
            {
                "benchmark_identity": "b1",
                "candidate_identity": "c1",
                "model_family": "elastic_net",
                "analytical_status": "diagnostic_only",
                "analytical_reason_code": "PREDICTIVE_BENCHMARK_NOT_MET",
                "validation_status": "completed",
                "test_observation_count": 2
            }
        ],
        "hyperparameter_selection": [
            {
                "benchmark_identity": "b1",
                "outer_fold_index": 0,
                "configuration_identity": configuration_identity,
                "selected": True
            },
            {
                "benchmark_identity": "b1",
                "outer_fold_index": 1,
                "configuration_identity": configuration_identity,
                "selected": True
            }
        ],
        "validation_predictions": [
            {
                "benchmark_identity": "b1",
                "condition_protocol_identity": "a"
            },
            {
                "benchmark_identity": "b1",
                "condition_protocol_identity": "b"
            }
        ],
        "condition_calibration": [
            {
                "benchmark_identity": "b1",
                "condition_protocol_identity": "a"
            },
            {
                "benchmark_identity": "b1",
                "condition_protocol_identity": "b"
            }
        ],
        "feature_importance": [
            {
                "benchmark_identity": "b1",
                "candidate_identity": "c1",
                "model_family": "elastic_net",
                "encoded_feature": feature_name,
                "importance_status": "unavailable",
                "importance_value": None
            }
            for feature_name in [
                "scaled_horizon",
                "condition[b]",
                "scaled_horizon:condition[b]"
            ]
        ]
    }


# Building the minimal policy required by the reconciliation function
def _build_reconciliation_policy() -> dict:
    return {
        "model_families": {
            "elastic_net": {
                "configurations": [
                    {
                        "alpha": 0.1,
                        "l1_ratio": 0.5
                    }
                ]
            }
        }
    }


# Confirming a fully consistent reconciliation fixture is accepted
def test_validate_fit_table_reconciliation_accepts_consistent_tables():
    validate_fit_table_reconciliation(
        _build_valid_reconciliation_tables(),
        _build_reconciliation_policy()
    )


# Rejecting a registry-metric analytical-status mismatch
def test_validate_fit_table_reconciliation_rejects_metric_status_mismatch():
    tables = _build_valid_reconciliation_tables()

    tables["validation_metrics"][0]["analytical_status"] = "available"
    tables["validation_metrics"][0]["analytical_reason_code"] = None

    with pytest.raises(
        RuntimeError,
        match = "registry and metric statuses disagree"
    ):
        validate_fit_table_reconciliation(
            tables,
            _build_reconciliation_policy()
        )


# Rejecting completed importance records for an unqualified benchmark
def test_validate_fit_table_reconciliation_rejects_unqualified_completed_importance():
    tables = _build_valid_reconciliation_tables()

    for record in tables["feature_importance"]:
        record["importance_status"] = "completed"
        record["importance_value"] = 0.5

    with pytest.raises(
        RuntimeError,
        match = "Unqualified benchmark contains completed feature importance"
    ):
        validate_fit_table_reconciliation(
            tables,
            _build_reconciliation_policy()
        )


# Rejecting an unavailable benchmark carrying predictions
def test_validate_fit_table_reconciliation_rejects_unavailable_with_predictions():
    tables = _build_valid_reconciliation_tables()

    tables["candidate_registry"][0]["analytical_status"] = "not_available"
    tables["candidate_registry"][0]["analytical_reason_code"] = "FIT_FAILED"
    tables["candidate_registry"][0]["execution_status"] = "unavailable"
    tables["validation_metrics"][0]["analytical_status"] = "not_available"
    tables["validation_metrics"][0]["analytical_reason_code"] = "FIT_FAILED"
    tables["validation_metrics"][0]["validation_status"] = "unavailable"

    with pytest.raises(
        RuntimeError,
        match = "Unavailable predictive benchmark contains analytical output"
    ):
        validate_fit_table_reconciliation(
            tables,
            _build_reconciliation_policy()
        )


# Rejecting a fold with zero or multiple selected configurations
def test_validate_fit_table_reconciliation_rejects_duplicate_selection():
    tables = _build_valid_reconciliation_tables()

    tables["hyperparameter_selection"].append(
        {
            "benchmark_identity": "b1",
            "outer_fold_index": 0,
            "configuration_identity": "duplicate",
            "selected": True
        }
    )

    with pytest.raises(
        RuntimeError,
        match = "hyperparameter selection is incomplete"
    ):
        validate_fit_table_reconciliation(
            tables,
            _build_reconciliation_policy()
        )


# Rejecting a policy with an altered estimator name
def test_validate_runtime_policy_bindings_rejects_altered_estimator_name():
    policy = load_json_object(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "predictive_modeling_policy.json"
    )

    policy["model_families"]["elastic_net"]["estimator"] = "Ridge"

    with pytest.raises(
        RuntimeError,
        match = "Elastic-net execution settings are invalid"
    ):
        validate_runtime_policy_bindings(policy)


# Rejecting a policy with an altered scientific-scope value
def test_validate_runtime_policy_bindings_rejects_altered_scope():
    policy = load_json_object(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "predictive_modeling_policy.json"
    )

    policy["scientific_scope"]["causal_interpretation_authorized"] = True

    with pytest.raises(
        RuntimeError,
        match = "scientific-scope value is invalid"
    ):
        validate_runtime_policy_bindings(policy)


# Rejecting a policy with a non-single-threaded execution flag
def test_validate_runtime_policy_bindings_rejects_byte_identity_disabled():
    policy = load_json_object(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "predictive_modeling_policy.json"
    )

    policy["execution"]["byte_identical_artifacts_required"] = False

    with pytest.raises(
        RuntimeError,
        match = "Predictive execution contract is invalid"
    ):
        validate_runtime_policy_bindings(policy)


# Rejecting a policy with a modified hyperparameter domain
def test_validate_runtime_policy_bindings_rejects_invalid_hyperparameter():
    policy = load_json_object(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "predictive_modeling_policy.json"
    )

    policy["model_families"]["random_forest"]["configurations"][0]["min_samples_leaf"] = 0

    with pytest.raises(
        RuntimeError,
        match = "Random-forest configuration is invalid"
    ):
        validate_runtime_policy_bindings(policy)


# Confirming that a condition with no inside-interval observations downgrades the benchmark to diagnostic-only
#  instead of crashing
def test_fit_benchmarks_downgrades_when_condition_has_no_predictions(monkeypatch: pytest.MonkeyPatch):
    from scripts import run_predictive_models as predictive_models

    workflow = _build_small_workflow()

    candidate = workflow["component_candidates"][0]

    second_condition = candidate["condition_identities"][1]

    for observation in candidate["included_observations"]:
        if observation["condition_protocol_identity"] == second_condition:
            observation["inside_evaluation_interval"] = False

    for fold in workflow["fold_records"]:
        fold["test_observation_count"] = sum(
            1
            for observation in candidate["included_observations"]
            if observation["cell_identity"]
            in set(fold["test_cell_identities"])
            and observation["inside_evaluation_interval"]
        )

    policy = _build_small_policy()

    outputs = predictive_models.fit_benchmarks(
        selected_candidates=workflow["component_candidates"],
        workflow=workflow,
        policy=policy
    )

    assert len(outputs["candidate_registry"]) == 1

    registry = outputs["candidate_registry"][0]

    assert registry["execution_status"] == "completed"
    assert registry["analytical_status"] == "diagnostic_only"
    assert (
        registry["analytical_reason_code"]
        == "INCOMPLETE_CONDITION_CALIBRATION"
    )

    unavailable_calibration_records = [
        record
        for record in outputs["condition_calibration"]
        if record["calibration_status"] == "unavailable"
    ]

    assert len(unavailable_calibration_records) == 1
    assert (
        unavailable_calibration_records[0]["calibration_reason_code"]
        == "EMPTY_CONDITION_PREDICTION_INVENTORY"
    )
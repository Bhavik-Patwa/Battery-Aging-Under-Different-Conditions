import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
import warnings

from collections import Counter, defaultdict
from importlib.metadata import version as distribution_version
from pathlib import Path

import numpy as np

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

import scripts.run_bayesian_models as bayesian_modeling

from scripts.run_bayesian_models import (
    build_validation_workflow,
    load_validated_inputs
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_REGISTRY_FIELDS = [
    "benchmark_identity",
    "candidate_identity",
    "support_region_identity",
    "design_scope",
    "sampling_design_scope",
    "model_family",
    "outcome",
    "cohort",
    "aging_mode",
    "horizon_axis",
    "support_interval_minimum",
    "support_interval_maximum",
    "support_interval_lower_inclusive",
    "support_interval_upper_inclusive",
    "condition_protocol_identities",
    "condition_count",
    "independent_cell_count",
    "observation_count",
    "outer_fold_count",
    "execution_status",
    "execution_reason_code",
    "analytical_status",
    "analytical_reason_code",
    "selected_configuration_identities"
]

OBSERVATION_ACCOUNTING_FIELDS = [
    "candidate_identity",
    "source_observation_count",
    "inside_evaluation_interval_count",
    "outside_evaluation_interval_count",
    "independent_cell_count",
    "condition_count"
]

VALIDATION_FOLD_FIELDS = [
    "candidate_identity",
    "fold_index",
    "training_cell_count",
    "test_cell_count",
    "training_cell_identities",
    "test_cell_identities",
    "training_observation_count",
    "test_observation_count",
    "training_condition_count",
    "training_minimum_condition_cell_count",
    "training_minimum_condition_repeated_cell_count",
    "fold_status",
    "fold_reason_code"
]

HYPERPARAMETER_SELECTION_FIELDS = [
    "benchmark_identity",
    "candidate_identity",
    "model_family",
    "outer_fold_index",
    "configuration_identity",
    "configuration",
    "inner_fold_count",
    "inner_validation_status",
    "inner_validation_reason_code",
    "equal_cell_weighted_root_mean_squared_error",
    "selected"
]

VALIDATION_PREDICTION_FIELDS = [
    "benchmark_identity",
    "candidate_identity",
    "model_family",
    "outer_fold_index",
    "cell_identity",
    "source_identity",
    "condition_protocol_identity",
    "horizon",
    "observed_outcome",
    "predicted_outcome",
    "prediction_interval_lower",
    "prediction_interval_upper",
    "benchmark_prediction",
    "prediction_status",
    "prediction_reason_code",
    "estimate_label"
]

VALIDATION_METRIC_FIELDS = [
    "benchmark_identity",
    "candidate_identity",
    "model_family",
    "outer_fold_count",
    "test_cell_count",
    "test_observation_count",
    "equal_cell_weighted_mean_absolute_error",
    "equal_cell_weighted_root_mean_squared_error",
    "equal_cell_weighted_median_error",
    "equal_cell_weighted_predictive_r_squared",
    "prediction_interval_coverage",
    "mean_prediction_interval_width",
    "validation_status",
    "validation_reason_code",
    "analytical_status",
    "analytical_reason_code",
    "estimate_label"
]

CONDITION_CALIBRATION_FIELDS = [
    "benchmark_identity",
    "candidate_identity",
    "model_family",
    "condition_protocol_identity",
    "independent_cell_count",
    "observation_count",
    "equal_cell_weighted_mean_observed",
    "equal_cell_weighted_mean_predicted",
    "equal_cell_weighted_mean_error",
    "equal_cell_weighted_mean_absolute_error",
    "equal_cell_weighted_root_mean_squared_error",
    "calibration_status",
    "calibration_reason_code",
    "estimate_label"
]

FEATURE_IMPORTANCE_FIELDS = [
    "benchmark_identity",
    "candidate_identity",
    "model_family",
    "configuration_identity",
    "encoded_feature",
    "feature_group",
    "importance_value",
    "importance_method",
    "importance_status",
    "importance_reason_code",
    "estimate_label"
]


# Representing an estimator failure that requires explicit unavailable recording
class EstimatorUnavailableError(RuntimeError):
    def __init__(self, reason_code: str):
        super().__init__(
            reason_code
        )

        self.reason_code = reason_code


# Parsing validated input paths, policy locations and requested operation
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = (
            "Validating and evaluating registered grouped-cell predictive benchmarks for supported longitudinal outcomes."
        )
    )

    parser.add_argument(
        "--feature-manifest-path",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--design-report-path",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--identification-report-path",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--descriptive-report-path",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--bayesian-policy-path",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--bayesian-report-path",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--predictive-policy-path",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--output-directory",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--operation",
        choices = (
            "validate",
            "fit"
        ),
        required = True
    )

    return parser.parse_args()


# Producing deterministic canonical JSON
def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii = False,
        separators = (
            ",",
            ":"
        ),
        sort_keys = True
    )


# Building one deterministic identity
def deterministic_identity(value: object) -> str:
    return hashlib.sha256(
        canonical_json(
            value
        ).encode(
            "utf-8"
        )
    ).hexdigest()


# Calculating one file SHA-256 digest
def calculate_file_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


# Loading one strict JSON object
def load_json_object(path: Path) -> dict:
    with path.open(
        mode = "r",
        encoding = "utf-8"
    ) as input_stream:
        value = json.load(
            input_stream
        )

    if not isinstance(
        value,
        dict
    ):
        raise RuntimeError(f"JSON document is not an object : {path}")

    return value


# Parsing one CSV value into its typed representation
def parse_csv_value(value: str):
    if value == "":
        return None

    if value == "true":
        return True

    if value == "false":
        return False

    if (
        value.startswith(
            "["
        )
        or value.startswith(
            "{"
        )
    ):
        return json.loads(
            value
        )

    return value


# Reading one CSV table with typed values
def read_csv_table(path: Path) -> list[dict]:
    with path.open(
        mode = "r",
        encoding = "utf-8",
        newline = ""
    ) as input_stream:
        return [
            {
                field_name: parse_csv_value(
                    field_value
                )
                for field_name, field_value
                in row.items()
            }
            for row in csv.DictReader(
                input_stream
            )
        ]


# Serializing one CSV value deterministically
def serialize_csv_value(value: object):
    if value is None:
        return ""

    if isinstance(
        value,
        bool
    ):
        return (
            "true"
            if value
            else "false"
        )

    if isinstance(
        value,
        (
            list,
            dict
        )
    ):
        return canonical_json(
            value
        )

    if isinstance(
        value,
        float
    ):
        if not math.isfinite(
            value
        ):
            raise RuntimeError("Cannot serialize a non-finite numerical value.")

        return repr(
            value
        )

    return value


# Writing one deterministic CSV table
def write_csv_table(path: Path, fields: list[str], records: list[dict]) -> None:
    with path.open(
        mode = "w",
        encoding = "utf-8",
        newline = ""
    ) as output_stream:
        writer = csv.DictWriter(
            output_stream,
            fieldnames = fields,
            lineterminator = "\n"
        )

        writer.writeheader()

        for record in records:
            if set(
                record
            ) != set(
                fields
            ):
                raise RuntimeError(f"Output record does not match its schema : {path.name}")

            writer.writerow(
                {
                    field_name: serialize_csv_value(
                        record[
                            field_name
                        ]
                    )
                    for field_name in fields
                }
            )


# Validating every policy value that controls estimator execution
def validate_runtime_policy_bindings(policy: dict) -> None:
    scientific_scope = policy[
        "scientific_scope"
    ]

    required_scope_values = {
        "estimate_label": "predictive",
        "causal_interpretation_authorized": False,
        "policy_recommendation_authorized": False,
        "treatment_optimization_authorized": False,
        "novel_condition_prediction_authorized": False,
        "calendar_and_cycle_aging_separated": True,
        "elapsed_time_and_cumulative_fec_separated": True,
        "primary_and_sensitivity_cohorts_separated": True,
        "dataset_stages_separated": True
    }

    for field_name, expected_value in (
        required_scope_values.items()
    ):
        if scientific_scope.get(
            field_name
        ) != expected_value:
            raise RuntimeError(f"Predictive scientific-scope value is invalid : {field_name}")

    inputs = policy[
        "inputs"
    ]

    if (
        inputs.get(
            "bayesian_implementation_match_required"
        )
        is not True
        or inputs.get(
            "required_bayesian_report_operation"
        )
        != "fit"
        or inputs.get(
            "required_bayesian_tables"
        )
        != [
            "candidate_registry",
            "grouped_validation_folds"
        ]
        or inputs.get(
            "dependency_lock_hash_required"
        )
        is not True
    ):
        raise RuntimeError("Predictive input-binding contract is invalid.")

    observation_contract = policy[
        "observation_contract"
    ]

    if (
        observation_contract.get(
            "training_population"
        )
        != (
            "all eligible positive-horizon observations from support-region member cells"
        )
        or observation_contract.get(
            "evaluation_population"
        )
        != (
            "eligible observations inside the authorized evaluation interval"
        )
    ):
        raise RuntimeError("Predictive observation-population contract is invalid.")

    predictor_contract = policy[
        "predictor_contract"
    ]

    if (
        predictor_contract.get(
            "horizon_scaling_source"
        )
        != "training_observations_only"
        or predictor_contract.get(
            "condition_encoding"
        )
        != "reference_indicator"
        or predictor_contract.get(
            "reference_condition_rule"
        )
        != (
            "lexicographically_first_observed_training_condition"
        )
    ):
        raise RuntimeError("Predictive feature-construction contract is invalid.")

    interval_policy = policy[
        "prediction_intervals"
    ]

    if (
        interval_policy.get(
            "weighting"
        )
        != "equal_total_weight_per_cell"
        or interval_policy.get(
            "interpretation"
        )
        != (
            "empirical_selection_conditioned_predictive_interval"
        )
    ):
        raise RuntimeError("Predictive interval-construction contract is invalid.")

    explainability = policy[
        "explainability"
    ]

    if (
        explainability.get(
            "method"
        )
        != "native_encoded_feature_importance"
        or explainability.get(
            "elastic_net_importance"
        )
        != "absolute_standardized_coefficient"
        or explainability.get(
            "tree_importance"
        )
        != "impurity_based_feature_importance"
        or explainability.get(
            "held_out_performance_contribution_claim_authorized"
        )
        is not False
        or explainability.get(
            "estimate_label"
        )
        != "predictive_model_inspection"
    ):
        raise RuntimeError("Predictive inspection contract is invalid.")

    final_inspection = policy[
        "final_model_inspection"
    ]

    if (
        final_inspection.get(
            "configuration_rule"
        )
        != (
            "most_frequently_selected_across_outer_folds_then_configuration_identity"
        )
        or final_inspection.get(
            "training_population"
        )
        != (
            "all eligible positive-horizon observations from support-region member cells"
        )
    ):
        raise RuntimeError("Final-model inspection contract is invalid.")

    execution = policy[
        "execution"
    ]

    if (
        not isinstance(
            execution.get(
                "random_seed"
            ),
            int
        )
        or execution[
            "random_seed"
        ] < 0
        or execution.get(
            "byte_identical_artifacts_required"
        )
        is not True
        or execution.get(
            "transactional_publication_required"
        )
        is not True
    ):
        raise RuntimeError("Predictive execution contract is invalid.")

    elastic_net = policy[
        "model_families"
    ][
        "elastic_net"
    ]

    if (
        elastic_net.get(
            "estimator"
        )
        != "ElasticNet"
        or not isinstance(
            elastic_net.get(
                "maximum_iterations"
            ),
            int
        )
        or elastic_net[
            "maximum_iterations"
        ] <= 0
        or not math.isfinite(
            float(
                elastic_net.get(
                    "tolerance"
                )
            )
        )
        or float(
            elastic_net[
                "tolerance"
            ]
        ) <= 0
    ):
        raise RuntimeError("Elastic-net execution settings are invalid.")

    for configuration in elastic_net[
        "configurations"
    ]:
        alpha = float(
            configuration[
                "alpha"
            ]
        )

        l1_ratio = float(
            configuration[
                "l1_ratio"
            ]
        )

        if (
            not math.isfinite(
                alpha
            )
            or alpha <= 0
            or not math.isfinite(
                l1_ratio
            )
            or not (
                0
                <= l1_ratio
                <= 1
            )
        ):
            raise RuntimeError("Elastic-net configuration is invalid.")

    random_forest = policy[
        "model_families"
    ][
        "random_forest"
    ]

    if (
        random_forest.get(
            "estimator"
        )
        != "RandomForestRegressor"
        or not isinstance(
            random_forest.get(
                "n_estimators"
            ),
            int
        )
        or random_forest[
            "n_estimators"
        ] <= 0
        or not (
            0
            < float(
                random_forest.get(
                    "max_features"
                )
            )
            <= 1
        )
    ):
        raise RuntimeError("Random-forest execution settings are invalid.")

    for configuration in random_forest[
        "configurations"
    ]:
        maximum_depth = configuration[
            "max_depth"
        ]

        minimum_leaf_size = configuration[
            "min_samples_leaf"
        ]

        if (
            (
                maximum_depth is not None
                and (
                    not isinstance(
                        maximum_depth,
                        int
                    )
                    or maximum_depth <= 0
                )
            )
            or not isinstance(
                minimum_leaf_size,
                int
            )
            or minimum_leaf_size <= 0
        ):
            raise RuntimeError("Random-forest configuration is invalid.")

    gradient_boosting = policy[
        "model_families"
    ][
        "gradient_boosting"
    ]

    if (
        gradient_boosting.get(
            "estimator"
        )
        != "GradientBoostingRegressor"
        or gradient_boosting.get(
            "loss"
        )
        != "squared_error"
    ):
        raise RuntimeError("Gradient-boosting execution settings are invalid.")

    for configuration in gradient_boosting[
        "configurations"
    ]:
        if (
            not isinstance(
                configuration[
                    "n_estimators"
                ],
                int
            )
            or configuration[
                "n_estimators"
            ] <= 0
            or not math.isfinite(
                float(
                    configuration[
                        "learning_rate"
                    ]
                )
            )
            or float(
                configuration[
                    "learning_rate"
                ]
            ) <= 0
            or not isinstance(
                configuration[
                    "max_depth"
                ],
                int
            )
            or configuration[
                "max_depth"
            ] <= 0
            or not isinstance(
                configuration[
                    "min_samples_leaf"
                ],
                int
            )
            or configuration[
                "min_samples_leaf"
            ] <= 0
        ):
            raise RuntimeError("Gradient-boosting configuration is invalid.")

    output_contract = policy[
        "output_contract"
    ]

    for field_name in (
        "row_counts_required",
        "content_reconciliation_required",
        "fold_cell_identities_required"
    ):
        if output_contract.get(
            field_name
        ) is not True:
            raise RuntimeError(f"Predictive output requirement is invalid : {field_name}")


# Validating the frozen predictive-modeling policy
def validate_policy(policy: dict) -> None:
    required_sections = {
        "policy_name",
        "policy_version",
        "effective_date",
        "previous_policy_sha256",
        "registration_status",
        "scientific_scope",
        "inputs",
        "observation_contract",
        "predictor_contract",
        "missing_data",
        "model_families",
        "weighting",
        "nested_validation",
        "prediction_intervals",
        "metric_definitions",
        "metrics",
        "explainability",
        "final_model_inspection",
        "failure_handling",
        "execution",
        "output_contract",
        "amendment"
    }

    if set(
        policy
    ) != required_sections:
        raise RuntimeError("Predictive policy has an unexpected top-level schema.")

    validate_runtime_policy_bindings(
        policy
    )

    if (
        policy[
            "scientific_scope"
        ][
            "causal_interpretation_authorized"
        ]
        is not False
    ):
        raise RuntimeError("Predictive policy must not authorize causal interpretation.")

    if (
        policy[
            "scientific_scope"
        ][
            "novel_condition_prediction_authorized"
        ]
        is not False
    ):
        raise RuntimeError("Predictive policy must prohibit novel condition prediction.")

    if (
        policy[
            "inputs"
        ][
            "posterior_results_used_for_candidate_selection"
        ]
        is not False
    ):
        raise RuntimeError("Posterior results cannot control predictive candidate selection.")

    if (
        policy[
            "nested_validation"
        ][
            "cell_leakage_prohibited"
        ]
        is not True
    ):
        raise RuntimeError("Grouped validation must prohibit cell leakage.")

    if (
        policy[
            "execution"
        ][
            "parallel_worker_count"
        ]
        != 1
    ):
        raise RuntimeError("Deterministic execution requires one estimator worker.")

    if (
        policy[
            "failure_handling"
        ][
            "partial_outer_predictions_publishable"
        ]
        is not False
    ):
        raise RuntimeError("Partial outer predictions must not be publishable.")

    if (
        policy[
            "final_model_inspection"
        ][
            "performance_estimation_from_final_refit_authorized"
        ]
        is not False
    ):
        raise RuntimeError("Final inspection refit must not authorize performance estimation.")

    if set(
        policy[
            "model_families"
        ]
    ) != {
        "elastic_net",
        "random_forest",
        "gradient_boosting"
    }:
        raise RuntimeError("Predictive model-family inventory is invalid.")

    inputs = policy[
        "inputs"
    ]

    if inputs[
        "candidate_selection_fields"
    ] != [
        "admissibility_status",
        "grouped_validation_status"
    ]:
        raise RuntimeError("Predictive candidate-selection fields are invalid.")

    if inputs[
        "candidate_selection_values"
    ] != {
        "admissibility_status": "admissible",
        "grouped_validation_status": "available"
    }:
        raise RuntimeError("Predictive candidate-selection values are invalid.")

    for field_name in (
        "posterior_results_used_for_candidate_selection",
        "posterior_results_used_as_predictors",
        "outcome_magnitude_direction_or_performance_used_for_candidate_selection"
    ):
        if inputs[
            field_name
        ] is not False:
            raise RuntimeError(f"Predictive input restriction is invalid : {field_name}")

    if inputs[
        "outcome_validity_requirements_inherited_from_upstream_admissibility"
    ] is not True:
        raise RuntimeError("Upstream outcome-validity screening must be declared.")

    observation_contract = policy[
        "observation_contract"
    ]

    if (
        observation_contract[
            "independent_unit"
        ]
        != "cell_identity"
        or observation_contract[
            "outcome_transformation"
        ]
        != "natural_log"
        or observation_contract[
            "reported_prediction_scale"
        ]
        != "original_ratio_scale"
        or observation_contract[
            "strictly_positive_predictions_required"
        ]
        is not True
        or observation_contract[
            "complete_cumulative_fec_required"
        ]
        is not True
        or observation_contract[
            "duplicate_observations_per_candidate_prohibited"
        ]
        is not True
    ):
        raise RuntimeError("Predictive observation contract is invalid.")

    predictor_contract = policy[
        "predictor_contract"
    ]

    if predictor_contract[
        "permitted_predictors"
    ] != [
        "scaled_horizon",
        "condition_protocol_identity",
        "scaled_horizon_by_condition_protocol_identity"
    ]:
        raise RuntimeError("Predictive predictor inventory is invalid.")

    for field_name in (
        "cell_identity_as_predictor",
        "source_identity_as_predictor",
        "outcome_history_as_predictor",
        "laboratory_as_predictor",
        "equipment_as_predictor",
        "incident_reason_as_predictor",
        "stage_as_predictor",
        "unobserved_condition_combinations_permitted"
    ):
        if predictor_contract[
            field_name
        ] is not False:
            raise RuntimeError(f"Prohibited predictor setting is enabled : {field_name}")

    nested_validation = policy[
        "nested_validation"
    ]

    if (
        nested_validation[
            "outer_folds"
        ]
        != "reuse_registered_bayesian_grouped_validation_folds"
        or nested_validation[
            "inner_requested_fold_count"
        ]
        < 2
        or nested_validation[
            "inner_minimum_fold_count"
        ]
        < 2
        or nested_validation[
            "inner_minimum_fold_count"
        ]
        > nested_validation[
            "inner_requested_fold_count"
        ]
        or nested_validation[
            "inner_condition_stratification_required"
        ]
        is not True
        or nested_validation[
            "same_condition_training_required"
        ]
        is not True
        or nested_validation[
            "same_condition_horizon_coverage_required"
        ]
        is not True
        or nested_validation[
            "selection_metric"
        ]
        != "equal_cell_weighted_root_mean_squared_error"
        or nested_validation[
            "selection_metric_scale"
        ]
        != "original_ratio_scale"
        or nested_validation[
            "selection_tie_breaker"
        ]
        != "configuration_identity"
    ):
        raise RuntimeError("Nested-validation policy is invalid.")

    interval_policy = policy[
        "prediction_intervals"
    ]

    if (
        interval_policy[
            "method"
        ]
        != "inner_out_of_fold_absolute_log_residual_quantile"
        or not (
            0
            < float(
                interval_policy[
                    "coverage_probability"
                ]
            )
            < 1
        )
        or interval_policy[
            "formal_distribution_free_coverage_claim_authorized"
        ]
        is not False
    ):
        raise RuntimeError("Prediction-interval policy is invalid.")

    expected_tables = {
        "candidate_registry",
        "observation_accounting",
        "validation_folds",
        "hyperparameter_selection",
        "validation_predictions",
        "validation_metrics",
        "condition_calibration",
        "feature_importance"
    }

    if set(
        policy[
            "output_contract"
        ][
            "tables"
        ]
    ) != expected_tables:
        raise RuntimeError("Predictive output-table inventory is invalid.")

    if policy[
        "execution"
    ][
        "operations"
    ] != [
        "validate",
        "fit"
    ]:
        raise RuntimeError("Predictive operation inventory is invalid.")

    configuration_identities = []

    for family_name, family_policy in policy[
        "model_families"
    ].items():
        configurations = family_policy.get(
            "configurations"
        )

        if not configurations:
            raise RuntimeError(f"Model family has no configurations : {family_name}")

        for configuration in configurations:
            configuration_identity = deterministic_identity(
                {
                    "model_family": family_name,
                    "configuration": configuration
                }
            )

            configuration_identities.append(
                configuration_identity
            )

    if len(
        configuration_identities
    ) != len(
        set(
            configuration_identities
        )
    ):
        raise RuntimeError("Predictive configuration identities are not unique.")

    explainability = policy[
        "explainability"
    ]

    if (
        explainability[
            "analytical_status_required"
        ]
        != "available"
        or explainability[
            "unavailable_records_required"
        ]
        is not True
        or explainability[
            "importance_values_must_be_finite_and_nonnegative"
        ]
        is not True
        or explainability[
            "causal_interpretation_authorized"
        ]
        is not False
        or explainability[
            "counterfactual_interpretation_authorized"
        ]
        is not False
        or explainability[
            "importance_comparability_across_model_families_authorized"
        ]
        is not False
    ):
        raise RuntimeError("Predictive explainability policy is invalid.")

    final_inspection = policy[
        "final_model_inspection"
    ]

    if (
        final_inspection[
            "analytical_status_required"
        ]
        != "available"
        or final_inspection[
            "performance_estimation_from_final_refit_authorized"
        ]
        is not False
        or final_inspection[
            "feature_importance_only"
        ]
        is not True
        or final_inspection[
            "unavailable_record_required"
        ]
        is not True
        or final_inspection[
            "fit_failure_effect"
        ]
        != (
            "feature_importance_unavailable_without_changing_nested_validation_results"
        )
    ):
        raise RuntimeError("Final-inspection policy is invalid.")

    if (
        policy[
            "explainability"
        ][
            "analytical_status_required"
        ]
        != final_inspection[
            "analytical_status_required"
        ]
    ):
        raise RuntimeError("Explainability and final-inspection analytical-status requirements disagree.")

    output_contract = policy[
        "output_contract"
    ]

    if output_contract[
        "candidate_registry_scope_fields"
    ] != [
        "support_region_identity",
        "design_scope",
        "sampling_design_scope",
        "support_interval_minimum",
        "support_interval_maximum",
        "support_interval_lower_inclusive",
        "support_interval_upper_inclusive",
        "condition_protocol_identities"
    ]:
        raise RuntimeError("Predictive candidate-registry scope-field inventory is invalid.")

    if output_contract[
        "analytical_handoff_required"
    ] is not True:
        raise RuntimeError("Predictive output contract must require the analytical handoff.")

    if output_contract[
        "input_hashes_required"
    ] is not True:
        raise RuntimeError("Predictive output contract must require input hashes.")

    if output_contract[
        "implementation_hash_required"
    ] is not True:
        raise RuntimeError("Predictive output contract must require the implementation hash.")

    if output_contract[
        "table_hashes_required"
    ] is not True:
        raise RuntimeError("Predictive output contract must require table hashes.")

    if output_contract[
        "schemas_required"
    ] is not True:
        raise RuntimeError("Predictive output contract must require table schemas.")

    if (
        policy[
            "missing_data"
        ][
            "imputation_authorized"
        ]
        is not False
        or policy[
            "missing_data"
        ][
            "silent_observation_removal_authorized"
        ]
        is not False
        or policy[
            "missing_data"
        ][
            "outcome_missingness_handled_by_upstream_eligibility"
        ]
        is not True
        or policy[
            "missing_data"
        ][
            "nonfinite_predictor_action"
        ]
        != "benchmark_unavailable"
    ):
        raise RuntimeError("Predictive missing-data contract is invalid.")

    if policy[
        "weighting"
    ] != {
        "training_weight_rule": "equal_total_weight_per_cell",
        "validation_metric_weight_rule": "equal_total_weight_per_test_cell",
        "normalization": "mean_observation_weight_equals_one"
    }:
        raise RuntimeError("Predictive weighting contract is invalid.")

    expected_metric_definitions = {
        "equal_cell_weighted_mean_absolute_error": "sum(w_i * abs(y_i - prediction_i)) / sum(w_i)",
        "equal_cell_weighted_root_mean_squared_error": "sqrt(sum(w_i * (y_i - prediction_i)^2) / sum(w_i))",
        "equal_cell_weighted_median_error": "weighted empirical median of prediction_i - y_i",
        "equal_cell_weighted_predictive_r_squared": "1 - weighted_model_squared_error / weighted_same_condition_mean_benchmark_squared_error",
        "prediction_interval_coverage": "sum(w_i * indicator(lower_i <= y_i <= upper_i)) / sum(w_i)",
        "mean_prediction_interval_width": "sum(w_i * (upper_i - lower_i)) / sum(w_i)",
        "weight_definition": "Each cell receives equal total weight; its weight is divided equally among its observations."
    }

    if policy[
        "metric_definitions"
    ] != expected_metric_definitions:
        raise RuntimeError("Predictive metric-definition contract is invalid.")

    if policy[
        "metrics"
    ][
        "primary"
    ] != [
        "equal_cell_weighted_root_mean_squared_error",
        "equal_cell_weighted_mean_absolute_error",
        "equal_cell_weighted_predictive_r_squared"
    ]:
        raise RuntimeError("Predictive primary-metric inventory is invalid.")

    if policy[
        "metrics"
    ][
        "secondary"
    ] != [
        "equal_cell_weighted_median_error",
        "prediction_interval_coverage",
        "mean_prediction_interval_width"
    ]:
        raise RuntimeError("Predictive secondary-metric inventory is invalid.")

    if policy[
        "metrics"
    ][
        "predictive_r_squared_benchmark"
    ] != "same_condition_training_equal_cell_mean":
        raise RuntimeError("Predictive R-squared benchmark is invalid.")

    if policy[
        "metrics"
    ][
        "analytical_qualification"
    ] != {
        "all_outer_folds_completed_required": True,
        "strictly_positive_finite_predictions_required": True,
        "predictive_r_squared_must_exceed_benchmark": True,
        "qualified_status": "available",
        "failed_status": "diagnostic_only"
    }:
        raise RuntimeError("Predictive analytical-qualification contract is invalid.")

    if policy[
        "failure_handling"
    ] != {
        "estimator_convergence_warning": "unavailable",
        "estimator_exception": "unavailable",
        "nonfinite_prediction": "unavailable",
        "nonpositive_prediction": "unavailable",
        "incomplete_outer_prediction_inventory": "unavailable",
        "partial_outer_predictions_publishable": False,
        "simpler_model_fallback_authorized": False,
        "specification_change_after_failure_authorized": False
    }:
        raise RuntimeError("Predictive failure-handling contract is invalid.")

    software_environment = policy[
        "execution"
    ].get(
        "software_environment"
    )

    if not isinstance(
        software_environment,
        dict
    ):
        raise RuntimeError("Predictive software-environment contract is missing.")

    required_software_environment_fields = {
        "python_version",
        "package_versions",
        "dependency_lock_path",
        "required_environment_variables"
    }

    if set(
        software_environment
    ) != required_software_environment_fields:
        raise RuntimeError("Predictive software-environment contract has an invalid schema.")

    if (
        not isinstance(
            software_environment[
                "python_version"
            ],
            str
        )
        or not software_environment[
            "python_version"
        ]
        or not isinstance(
            software_environment[
                "package_versions"
            ],
            dict
        )
        or not software_environment[
            "package_versions"
        ]
        or not isinstance(
            software_environment[
                "dependency_lock_path"
            ],
            str
        )
        or not software_environment[
            "dependency_lock_path"
        ]
        or not isinstance(
            software_environment[
                "required_environment_variables"
            ],
            dict
        )
        or not software_environment[
            "required_environment_variables"
        ]
    ):
        raise RuntimeError("Predictive software-environment contract contains an invalid field.")

    if (
        "PYTHONHASHSEED"
        not in software_environment[
            "required_environment_variables"
        ]
        or software_environment[
            "required_environment_variables"
        ][
            "PYTHONHASHSEED"
        ] != "0"
    ):
        raise RuntimeError("Predictive software-environment contract must require PYTHONHASHSEED=0.")

    amendment = policy[
        "amendment"
    ]

    required_amendment_fields = {
        "reason",
        "result_informed",
        "result_informed_scope",
        "affected_analyses",
        "complete_rerun_required"
    }

    if (
        not isinstance(
            amendment,
            dict
        )
        or set(
            amendment
        ) != required_amendment_fields
    ):
        raise RuntimeError("Predictive amendment record has an invalid schema.")

    if (
        not isinstance(
            amendment[
                "reason"
            ],
            str
        )
        or not amendment[
            "reason"
        ].strip()
        or not isinstance(
            amendment[
                "result_informed"
            ],
            bool
        )
        or not isinstance(
            amendment[
                "result_informed_scope"
            ],
            str
        )
        or (
            amendment[
                "result_informed"
            ] is True
            and not amendment[
                "result_informed_scope"
            ].strip()
        )
        or not isinstance(
            amendment[
                "affected_analyses"
            ],
            list
        )
        or not amendment[
            "affected_analyses"
        ]
        or len(
            amendment[
                "affected_analyses"
            ]
        )
        != len(
            set(
                amendment[
                    "affected_analyses"
                ]
            )
        )
        or amendment[
            "complete_rerun_required"
        ] is not True
    ):
        raise RuntimeError("Predictive amendment record is invalid.")

    family_required_fields = {
        "elastic_net": {
            "estimator",
            "maximum_iterations",
            "tolerance",
            "configuration_rationale",
            "configurations"
        },
        "random_forest": {
            "estimator",
            "n_estimators",
            "max_features",
            "configuration_rationale",
            "configurations"
        },
        "gradient_boosting": {
            "estimator",
            "loss",
            "configuration_rationale",
            "configurations"
        }
    }

    for family_name, required_fields in (
        family_required_fields.items()
    ):
        family_policy = policy[
            "model_families"
        ][
            family_name
        ]

        if set(
            family_policy
        ) != required_fields:
            raise RuntimeError(f"Predictive model-family schema is invalid : {family_name}")

        configuration_required_keys = {
            family_name_key
            for family_name_key in (
                set(
                    family_policy[
                        "configurations"
                    ][
                        0
                    ]
                )
                if family_policy[
                    "configurations"
                ]
                else set()
            )
        }

        if not configuration_required_keys:
            raise RuntimeError(f"Predictive model family has no configuration keys : {family_name}")

        for configuration in family_policy[
            "configurations"
        ]:
            if set(
                configuration
            ) != configuration_required_keys:
                raise RuntimeError(f"Predictive configuration schema is inconsistent : {family_name}")

# Verifying report tables against their published manifests
def verify_report_tables(report_path: Path, report: dict, required_tables: list[str]) -> dict[str, list[dict]]:
    verified_tables = {}

    for table_name in required_tables:
        table_manifest = report.get(
            "tables",
            {}
        ).get(
            table_name
        )

        if table_manifest is None:
            raise RuntimeError(f"Required report table is unavailable : {table_name}")

        table_path = (
            report_path.parent
            / table_manifest[
                "file_name"
            ]
        )

        if (
            calculate_file_sha256(
                table_path
            )
            != table_manifest[
                "sha256"
            ]
        ):
            raise RuntimeError(f"Report table hash does not match : {table_name}")

        verified_tables[
            table_name
        ] = read_csv_table(
            table_path
        )

    return verified_tables


# Building the deterministic configuration inventory from the policy
def build_configuration_inventory(policy: dict) -> dict[str, list[dict]]:
    configurations = {}

    for family_name, family_policy in sorted(
        policy[
            "model_families"
        ].items()
    ):
        family_configurations = []

        for configuration in family_policy[
            "configurations"
        ]:
            configuration_identity = deterministic_identity(
                {
                    "model_family": family_name,
                    "configuration": configuration
                }
            )

            family_configurations.append(
                {
                    "configuration_identity": (
                        configuration_identity
                    ),
                    "configuration": configuration
                }
            )

        configurations[
            family_name
        ] = sorted(
            family_configurations,
            key = lambda record: record[
                "configuration_identity"
            ]
        )

    return configurations


# Building normalized equal-cell observation weights
def build_cell_weights(observations: list[dict]) -> np.ndarray:
    observation_counts = Counter(
        observation[
            "cell_identity"
        ]
        for observation in observations
    )

    weights = np.asarray(
        [
            1.0
            / observation_counts[
                observation[
                    "cell_identity"
                ]
            ]
            for observation in observations
        ],
        dtype = float
    )

    return (
        weights
        / np.mean(
            weights
        )
    )


# Building the registered feature-name list from condition identities
def build_feature_names(condition_identities: list[str]) -> list[str]:
    if (
        not condition_identities
        or condition_identities
        != sorted(
            set(
                condition_identities
            )
        )
    ):
        raise RuntimeError("Condition identities must be nonempty, unique and ordered.")

    encoded_conditions = condition_identities[
        1:
    ]

    return (
        [
            "scaled_horizon"
        ]
        + [
            f"condition[{condition_identity}]"
            for condition_identity in encoded_conditions
        ]
        + [
            f"scaled_horizon:condition[{condition_identity}]"
            for condition_identity in encoded_conditions
        ]
    )


# Building the registered feature matrix from observations
def build_feature_matrix(observations: list[dict], condition_identities: list[str],
                         horizon_midpoint: float, horizon_width: float
) -> tuple[np.ndarray, list[str]]:
    if (
        not math.isfinite(
            horizon_width
        )
        or horizon_width <= 0
    ):
        raise RuntimeError("Feature construction requires a positive horizon width.")

    feature_names = build_feature_names(
        condition_identities
    )

    reference_condition = condition_identities[
        0
    ]

    encoded_conditions = condition_identities[
        1:
    ]

    matrix = np.zeros(
        (
            len(
                observations
            ),
            len(
                feature_names
            )
        ),
        dtype = float
    )

    for row_index, observation in enumerate(
        observations
    ):
        condition_identity = observation[
            "condition_protocol_identity"
        ]

        if condition_identity not in condition_identities:
            raise RuntimeError("Observation contains an unregistered condition.")

        scaled_horizon = (
            float(
                observation[
                    "horizon"
                ]
            )
            - horizon_midpoint
        ) / horizon_width

        matrix[
            row_index,
            0
        ] = scaled_horizon

        if condition_identity != reference_condition:
            condition_index = encoded_conditions.index(
                condition_identity
            )

            indicator_column = (
                1
                + condition_index
            )

            interaction_column = (
                1
                + len(
                    encoded_conditions
                )
                + condition_index
            )

            matrix[
                row_index,
                indicator_column
            ] = 1.0

            matrix[
                row_index,
                interaction_column
            ] = scaled_horizon

    return (
        matrix,
        feature_names
    )


# Calculating training-only horizon midpoint and width
def calculate_training_horizon_scaling(observations: list[dict]) -> tuple[float, float]:
    horizons = np.asarray(
        [
            float(
                observation[
                    "horizon"
                ]
            )
            for observation in observations
        ],
        dtype = float
    )

    horizon_minimum = float(
        np.min(
            horizons
        )
    )

    horizon_maximum = float(
        np.max(
            horizons
        )
    )

    horizon_width = (
        horizon_maximum
        - horizon_minimum
    )

    if horizon_width <= 0:
        raise RuntimeError("Training observations do not span a positive horizon range.")

    return (
        (
            horizon_minimum
            + horizon_maximum
        ) / 2,
        horizon_width
    )


# Deriving a deterministic estimator seed from context and policy
def estimator_seed(identity: object, policy: dict) -> int:
    identity_hash = deterministic_identity(
        {
            "identity": identity,
            "registered_seed": policy[
                "execution"
            ][
                "random_seed"
            ]
        }
    )

    return int(
        identity_hash[
            :8
        ],
        16
    )


# Building one estimator instance for the requested model family
def build_estimator(model_family: str, configuration: dict, identity: object, policy: dict):
    random_seed = estimator_seed(
        identity,
        policy
    )

    family_policy = policy[
        "model_families"
    ][
        model_family
    ]

    if model_family == "elastic_net":
        return Pipeline(
            steps = [
                (
                    "scale",
                    StandardScaler()
                ),
                (
                    "regression",
                    ElasticNet(
                        alpha = float(
                            configuration[
                                "alpha"
                            ]
                        ),
                        l1_ratio = float(
                            configuration[
                                "l1_ratio"
                            ]
                        ),
                        fit_intercept = True,
                        max_iter = int(
                            family_policy[
                                "maximum_iterations"
                            ]
                        ),
                        tol = float(
                            family_policy[
                                "tolerance"
                            ]
                        ),
                        selection = "cyclic",
                        random_state = random_seed
                    )
                )
            ]
        )

    if model_family == "random_forest":
        return RandomForestRegressor(
            n_estimators = int(
                family_policy[
                    "n_estimators"
                ]
            ),
            max_depth = configuration[
                "max_depth"
            ],
            min_samples_leaf = int(
                configuration[
                    "min_samples_leaf"
                ]
            ),
            max_features = float(
                family_policy[
                    "max_features"
                ]
            ),
            random_state = random_seed,
            n_jobs = 1
        )

    if model_family == "gradient_boosting":
        return GradientBoostingRegressor(
            loss = family_policy[
                "loss"
            ],
            n_estimators = int(
                configuration[
                    "n_estimators"
                ]
            ),
            learning_rate = float(
                configuration[
                    "learning_rate"
                ]
            ),
            max_depth = int(
                configuration[
                    "max_depth"
                ]
            ),
            min_samples_leaf = int(
                configuration[
                    "min_samples_leaf"
                ]
            ),
            random_state = random_seed
        )

    raise RuntimeError(f"Unsupported model family : {model_family}")


# Fitting one estimator with explicit failure handling
def fit_estimator(estimator, model_family: str,
                  matrix: np.ndarray, target: np.ndarray,
                  weights: np.ndarray
) -> None:
    if (
        matrix.ndim != 2
        or target.ndim != 1
        or weights.ndim != 1
        or matrix.shape[0] != target.shape[0]
        or matrix.shape[0] != weights.shape[0]
        or matrix.shape[0] == 0
    ):
        raise RuntimeError("Estimator inputs have incompatible dimensions.")

    if not (
        np.all(
            np.isfinite(
                matrix
            )
        )
        and np.all(
            np.isfinite(
                target
            )
        )
        and np.all(
            np.isfinite(
                weights
            )
        )
        and np.all(
            weights > 0
        )
    ):
        raise RuntimeError("Estimator inputs must be finite with positive weights.")

    with warnings.catch_warnings(
        record = True
    ) as captured_warnings:
        warnings.simplefilter(
            "always",
            ConvergenceWarning
        )

        try:
            if model_family == "elastic_net":
                estimator.fit(
                    matrix,
                    target,
                    scale__sample_weight = weights,
                    regression__sample_weight = weights
                )

            else:
                estimator.fit(
                    matrix,
                    target,
                    sample_weight = weights
                )

        except (
            FloatingPointError,
            ValueError,
            np.linalg.LinAlgError
        ) as exc:
            raise EstimatorUnavailableError("ESTIMATOR_FIT_FAILED") from exc

    if any(
        issubclass(
            warning.category,
            ConvergenceWarning
        )
        for warning in captured_warnings
    ):
        raise EstimatorUnavailableError("ESTIMATOR_CONVERGENCE_WARNING")


# Predicting on the original outcome scale with finite-value validation
def predict_original_scale(estimator, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        predicted_log = np.asarray(
            estimator.predict(
                matrix
            ),
            dtype = float
        )

    except (
        FloatingPointError,
        ValueError,
        np.linalg.LinAlgError
    ) as exc:
        raise EstimatorUnavailableError("ESTIMATOR_PREDICTION_FAILED") from exc

    if not np.all(
        np.isfinite(
            predicted_log
        )
    ):
        raise EstimatorUnavailableError("NONFINITE_LOG_SCALE_PREDICTION")

    with np.errstate(
        over = "ignore",
        invalid = "ignore"
    ):
        predicted = np.exp(
            predicted_log
        )

    if not np.all(
        np.isfinite(
            predicted
        )
        & (
            predicted > 0
        )
    ):
        raise EstimatorUnavailableError("NONPOSITIVE_OR_NONFINITE_PREDICTION")

    return (
        predicted_log,
        predicted
    )


# Building deterministic condition-stratified inner-fold assignments
def build_inner_assignments(candidate_identity: str, observations: list[dict],
                            policy: dict
) -> tuple[dict[str, int] | None, int]:
    cells_by_condition = defaultdict(
        set
    )

    for observation in observations:
        cells_by_condition[
            observation[
                "condition_protocol_identity"
            ]
        ].add(
            observation[
                "cell_identity"
            ]
        )

    if not cells_by_condition:
        return (
            None,
            0
        )

    requested_fold_count = int(
        policy[
            "nested_validation"
        ][
            "inner_requested_fold_count"
        ]
    )

    effective_fold_count = min(
        requested_fold_count,
        min(
            len(
                cells
            )
            for cells in cells_by_condition.values()
        )
    )

    if (
        effective_fold_count
        < policy[
            "nested_validation"
        ][
            "inner_minimum_fold_count"
        ]
    ):
        return (
            None,
            0
        )

    assignments = {}

    for condition_identity in sorted(
        cells_by_condition
    ):
        cells = sorted(
            cells_by_condition[
                condition_identity
            ],
            key = lambda cell_identity: (
                deterministic_identity(
                    {
                        "candidate_identity": candidate_identity,
                        "condition_protocol_identity": condition_identity,
                        "cell_identity": cell_identity,
                        "seed": policy[
                            "nested_validation"
                        ][
                            "inner_fold_assignment_seed"
                        ]
                    }
                ),
                cell_identity
            )
        )

        for cell_index, cell_identity in enumerate(
            cells
        ):
            if cell_identity in assignments:
                raise RuntimeError("A cell appears in multiple conditions.")

            assignments[
                cell_identity
            ] = (
                cell_index
                % effective_fold_count
            )

    return (
        assignments,
        effective_fold_count
    )


# Confirming every test observation lies inside same-condition training horizons
def fold_has_horizon_coverage(training_observations: list[dict], test_observations: list[dict]) -> bool:
    training_horizons = defaultdict(
        list
    )

    for observation in training_observations:
        training_horizons[
            observation[
                "condition_protocol_identity"
            ]
        ].append(
            float(
                observation[
                    "horizon"
                ]
            )
        )

    for observation in test_observations:
        condition_identity = observation[
            "condition_protocol_identity"
        ]

        condition_horizons = training_horizons.get(
            condition_identity
        )

        if not condition_horizons:
            return False

        test_horizon = float(
            observation[
                "horizon"
            ]
        )

        if not (
            min(
                condition_horizons
            )
            <= test_horizon
            <= max(
                condition_horizons
            )
        ):
            return False

    return True


# Calculating one deterministic weighted quantile
def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    if not (
        0
        <= probability
        <= 1
    ):
        raise ValueError("Quantile probability must be between zero and one.")

    order = np.argsort(
        values,
        kind = "mergesort"
    )

    ordered_values = values[
        order
    ]

    ordered_weights = weights[
        order
    ]

    cumulative_weights = np.cumsum(
        ordered_weights
    )

    target_weight = (
        probability
        * cumulative_weights[
            -1
        ]
    )

    selected_index = int(
        np.searchsorted(
            cumulative_weights,
            target_weight,
            side = "left"
        )
    )

    return float(
        ordered_values[
            min(
                selected_index,
                len(
                    ordered_values
                )
                - 1
            )
        ]
    )


# Calculating equal-cell weighted prediction metrics
def equal_cell_metrics(observations: list[dict], observed: np.ndarray,
                       predicted: np.ndarray, lower: np.ndarray | None = None,
                       upper: np.ndarray | None = None, benchmark: np.ndarray | None = None
) -> dict:
    weights = build_cell_weights(
        observations
    )

    errors = (
        predicted
        - observed
    )

    mean_absolute_error = float(
        np.average(
            np.abs(
                errors
            ),
            weights = weights
        )
    )

    root_mean_squared_error = float(
        math.sqrt(
            np.average(
                np.square(
                    errors
                ),
                weights = weights
            )
        )
    )

    median_error = weighted_quantile(
        errors,
        weights,
        0.5
    )

    predictive_r_squared = None

    if benchmark is not None:
        residual_sum = float(
            np.sum(
                weights
                * np.square(
                    observed
                    - predicted
                )
            )
        )

        benchmark_sum = float(
            np.sum(
                weights
                * np.square(
                    observed
                    - benchmark
                )
            )
        )

        if benchmark_sum > 0:
            predictive_r_squared = (
                1.0
                - residual_sum
                / benchmark_sum
            )

    interval_coverage = None
    mean_interval_width = None

    if (
        lower is not None
        and upper is not None
    ):
        interval_coverage = float(
            np.average(
                (
                    (
                        observed
                        >= lower
                    )
                    & (
                        observed
                        <= upper
                    )
                ).astype(
                    float
                ),
                weights = weights
            )
        )

        mean_interval_width = float(
            np.average(
                upper
                - lower,
                weights = weights
            )
        )

    return {
        "equal_cell_weighted_mean_absolute_error": (
            mean_absolute_error
        ),
        "equal_cell_weighted_root_mean_squared_error": (
            root_mean_squared_error
        ),
        "equal_cell_weighted_median_error": (
            median_error
        ),
        "equal_cell_weighted_predictive_r_squared": (
            predictive_r_squared
        ),
        "prediction_interval_coverage": (
            interval_coverage
        ),
        "mean_prediction_interval_width": (
            mean_interval_width
        )
    }


# Calculating the same-condition training benchmark for held-out rows
def calculate_condition_benchmark(training_observations: list[dict], test_observations: list[dict]) -> np.ndarray:
    training_by_condition = defaultdict(
        list
    )

    for observation in training_observations:
        training_by_condition[
            observation[
                "condition_protocol_identity"
            ]
        ].append(
            observation
        )

    condition_means = {}

    for condition_identity, observations in training_by_condition.items():
        weights = build_cell_weights(
            observations
        )

        condition_means[
            condition_identity
        ] = float(
            np.average(
                np.asarray(
                    [
                        observation[
                            "outcome_value"
                        ]
                        for observation in observations
                    ],
                    dtype = float
                ),
                weights = weights
            )
        )

    return np.asarray(
        [
            condition_means[
                observation[
                    "condition_protocol_identity"
                ]
            ]
            for observation in test_observations
        ],
        dtype = float
    )


# Evaluating one inner configuration across grouped inner folds
def evaluate_inner_configuration(candidate_identity: str, model_family: str,
                                 configuration_record: dict, training_observations: list[dict],
                                 condition_identities: list[str], policy: dict
) -> dict:
    assignments, fold_count = build_inner_assignments(
        candidate_identity,
        training_observations,
        policy
    )

    if assignments is None:
        return {
            "status": "unavailable",
            "reason_code": "INSUFFICIENT_INNER_FOLDS",
            "fold_count": 0,
            "metric": None,
            "residuals": None,
            "residual_weights": None
        }

    predicted_records = []
    predicted_values = []

    for fold_index in range(
        fold_count
    ):
        inner_training = [
            observation
            for observation in training_observations
            if assignments[
                observation[
                    "cell_identity"
                ]
            ] != fold_index
        ]

        inner_test = [
            observation
            for observation in training_observations
            if (
                assignments[
                    observation[
                        "cell_identity"
                    ]
                ]
                == fold_index
                and observation[
                    "inside_evaluation_interval"
                ]
            )
        ]

        if not inner_test:
            return {
                "status": "unavailable",
                "reason_code": "EMPTY_INNER_TEST_FOLD",
                "fold_count": fold_count,
                "metric": None,
                "residuals": None,
                "residual_weights": None
            }

        if not fold_has_horizon_coverage(
            inner_training,
            inner_test
        ):
            return {
                "status": "unavailable",
                "reason_code": "INNER_TEST_HORIZON_OUTSIDE_TRAINING_RANGE",
                "fold_count": fold_count,
                "metric": None,
                "residuals": None,
                "residual_weights": None
            }

        horizon_midpoint, horizon_width = (
            calculate_training_horizon_scaling(
                inner_training
            )
        )

        training_matrix, _ = build_feature_matrix(
            inner_training,
            condition_identities,
            horizon_midpoint,
            horizon_width
        )

        test_matrix, _ = build_feature_matrix(
            inner_test,
            condition_identities,
            horizon_midpoint,
            horizon_width
        )

        training_target = np.log(
            np.asarray(
                [
                    observation[
                        "outcome_value"
                    ]
                    for observation in inner_training
                ],
                dtype = float
            )
        )

        training_weights = build_cell_weights(
            inner_training
        )

        estimator = build_estimator(
            model_family,
            configuration_record[
                "configuration"
            ],
            {
                "candidate_identity": candidate_identity,
                "model_family": model_family,
                "configuration_identity": configuration_record[
                    "configuration_identity"
                ],
                "inner_fold_index": fold_index
            },
            policy
        )

        try:
            fit_estimator(
                estimator,
                model_family,
                training_matrix,
                training_target,
                training_weights
            )

            (
                predicted_log,
                fold_predictions
            ) = predict_original_scale(
                estimator,
                test_matrix
            )

        except EstimatorUnavailableError as exc:
            return {
                "status": "unavailable",
                "reason_code": exc.reason_code,
                "fold_count": fold_count,
                "metric": None,
                "residuals": None,
                "residual_weights": None
            }

        predicted_records.extend(
            inner_test
        )

        predicted_values.extend(
            fold_predictions.tolist()
        )

    observed_values = np.asarray(
        [
            observation[
                "outcome_value"
            ]
            for observation in predicted_records
        ],
        dtype = float
    )

    predicted_array = np.asarray(
        predicted_values,
        dtype = float
    )

    metrics = equal_cell_metrics(
        predicted_records,
        observed_values,
        predicted_array
    )

    residuals = np.abs(
        np.log(
            observed_values
        )
        - np.log(
            predicted_array
        )
    )

    return {
        "status": "completed",
        "reason_code": None,
        "fold_count": fold_count,
        "metric": metrics[
            "equal_cell_weighted_root_mean_squared_error"
        ],
        "residuals": residuals,
        "residual_weights": build_cell_weights(
            predicted_records
        )
    }


# Building native model-inspection feature importance
def build_feature_importance(estimator, model_family: str, feature_names: list[str]) -> tuple[np.ndarray, str]:
    if model_family == "elastic_net":
        values = np.abs(
            np.asarray(
                estimator.named_steps[
                    "regression"
                ].coef_,
                dtype = float
            )
        )

    else:
        values = np.asarray(
            estimator.feature_importances_,
            dtype = float
        )

    if (
        values.ndim != 1
        or len(
            values
        ) != len(
            feature_names
        )
        or not np.all(
            np.isfinite(
                values
            )
        )
        or np.any(
            values < 0
        )
    ):
        raise EstimatorUnavailableError("INVALID_FEATURE_IMPORTANCE")

    return (
        values,
        feature_importance_method(
            model_family
        )
    )


# Resolving the registered feature-importance method
def feature_importance_method(model_family: str) -> str:
    if model_family == "elastic_net":
        return "absolute_standardized_coefficient"

    if model_family in {
        "random_forest",
        "gradient_boosting"
    }:
        return "impurity_based_feature_importance"

    raise RuntimeError(f"Unsupported feature-importance model family : {model_family}")


# Building explicit unavailable feature-importance records
def build_unavailable_feature_importance_records(benchmark_identity: str, candidate_identity: str,
                                                 model_family: str, configuration_identity: str | None,
                                                 feature_names: list[str], reason_code: str
) -> list[dict]:
    importance_method = feature_importance_method(
        model_family
    )

    return [
        {
            "benchmark_identity": benchmark_identity,
            "candidate_identity": candidate_identity,
            "model_family": model_family,
            "configuration_identity": configuration_identity,
            "encoded_feature": feature_name,
            "feature_group": feature_group(
                feature_name
            ),
            "importance_value": None,
            "importance_method": importance_method,
            "importance_status": "unavailable",
            "importance_reason_code": reason_code,
            "estimate_label": (
                "predictive_model_inspection"
            )
        }
        for feature_name in feature_names
    ]


# Classifying one encoded feature into its semantic group
def feature_group(feature_name: str) -> str:
    if feature_name == "scaled_horizon":
        return "horizon"

    if feature_name.startswith(
        "condition["
    ):
        return "condition"

    return "horizon_by_condition"


# Building one candidate-level benchmark identity
def candidate_benchmark_identity(candidate_identity: str, model_family: str) -> str:
    return deterministic_identity(
        {
            "candidate_identity": candidate_identity,
            "model_family": model_family
        }
    )


# Validating the registered software environment before any fit
def validate_software_environment(policy: dict) -> dict:
    environment_policy = policy[
        "execution"
    ][
        "software_environment"
    ]

    observed_python_version = platform.python_version()

    if (
        observed_python_version
        != environment_policy[
            "python_version"
        ]
    ):
        raise RuntimeError("Python version does not match the predictive policy.")

    observed_package_versions = {
        package_name: distribution_version(
            package_name
        )
        for package_name in environment_policy[
            "package_versions"
        ]
    }

    if (
        observed_package_versions
        != environment_policy[
            "package_versions"
        ]
    ):
        raise RuntimeError("Package versions do not match the predictive policy.")

    observed_environment_variables = {
        variable_name: os.environ.get(
            variable_name
        )
        for variable_name in environment_policy[
            "required_environment_variables"
        ]
    }

    if (
        observed_environment_variables
        != environment_policy[
            "required_environment_variables"
        ]
    ):
        raise RuntimeError("Execution environment does not match the predictive policy.")

    dependency_lock_path = (
        PROJECT_ROOT
        / environment_policy[
            "dependency_lock_path"
        ]
    )

    if not dependency_lock_path.is_file():
        raise RuntimeError("Registered dependency lock is unavailable.")

    return {
        "python_version": observed_python_version,
        "package_versions": observed_package_versions,
        "environment_variables": (
            observed_environment_variables
        ),
        "dependency_lock_sha256": (
            calculate_file_sha256(
                dependency_lock_path
            )
        )
    }


# Validating upstream Bayesian inputs and reconciling candidates
def validate_upstream_inputs(args: argparse.Namespace, predictive_policy: dict) -> tuple[dict, dict, dict]:
    bayesian_policy_sha256 = calculate_file_sha256(
        args.bayesian_policy_path
    )

    if (
        bayesian_policy_sha256
        != predictive_policy[
            "inputs"
        ][
            "required_bayesian_policy_sha256"
        ]
    ):
        raise RuntimeError("Bayesian policy does not match the predictive registration.")

    bayesian_report_sha256 = calculate_file_sha256(
        args.bayesian_report_path
    )

    if (
        bayesian_report_sha256
        != predictive_policy[
            "inputs"
        ][
            "required_bayesian_report_sha256"
        ]
    ):
        raise RuntimeError("Bayesian report does not match the predictive registration.")

    bayesian_report = load_json_object(
        args.bayesian_report_path
    )

    if (
        bayesian_report.get(
            "operation"
        )
        != predictive_policy[
            "inputs"
        ][
            "required_bayesian_report_operation"
        ]
    ):
        raise RuntimeError("Bayesian report operation is not the registered input.")

    bayesian_implementation_sha256 = (
        calculate_file_sha256(
            Path(
                bayesian_modeling.__file__
            ).resolve()
        )
    )

    required_implementation_sha256 = (
        predictive_policy[
            "inputs"
        ][
            "required_bayesian_implementation_sha256"
        ]
    )

    if (
        bayesian_implementation_sha256
        != required_implementation_sha256
        or bayesian_report.get(
            "implementation_sha256"
        )
        != required_implementation_sha256
    ):
        raise RuntimeError("Bayesian implementation does not match the registered input.")

    bayesian_tables = verify_report_tables(
        args.bayesian_report_path,
        bayesian_report,
        predictive_policy[
            "inputs"
        ][
            "required_bayesian_tables"
        ]
    )

    validated_inputs = load_validated_inputs(
        feature_manifest_path = args.feature_manifest_path,
        design_report_path = args.design_report_path,
        identification_report_path = (
            args.identification_report_path
        ),
        descriptive_report_path = (
            args.descriptive_report_path
        ),
        modeling_policy_path = args.bayesian_policy_path
    )

    workflow = build_validation_workflow(
        validated_inputs
    )

    reconstructed_registry = {
        record[
            "candidate_identity"
        ]: record
        for record in workflow[
            "candidate_records"
        ]
    }

    reported_registry = {
        record[
            "candidate_identity"
        ]: record
        for record in bayesian_tables[
            "candidate_registry"
        ]
    }

    if set(
        reconstructed_registry
    ) != set(
        reported_registry
    ):
        raise RuntimeError("Reconstructed and reported Bayesian candidate inventories differ.")

    for candidate_identity, reconstructed in reconstructed_registry.items():
        reported = reported_registry[
            candidate_identity
        ]

        for field_name in (
            "admissibility_status",
            "grouped_validation_status",
            "condition_count",
            "independent_cell_count",
            "observation_count"
        ):
            if str(
                reconstructed[
                    field_name
                ]
            ) != str(
                reported[
                    field_name
                ]
            ):
                raise RuntimeError(
                    f"Bayesian candidate reconstruction does not reconcile : {candidate_identity} : {field_name}"
                )

    return (
        validated_inputs,
        workflow,
        bayesian_report
    )


# Confirming one candidate matches the registered selection policy
def candidate_matches_selection_policy(candidate_record: dict, policy: dict) -> bool:
    selection_fields = policy[
        "inputs"
    ][
        "candidate_selection_fields"
    ]

    selection_values = policy[
        "inputs"
    ][
        "candidate_selection_values"
    ]

    if set(
        selection_fields
    ) != set(
        selection_values
    ):
        raise RuntimeError("Candidate-selection fields and values do not reconcile.")

    return all(
        str(
            candidate_record[
                field_name
            ]
        )
        == str(
            selection_values[
                field_name
            ]
        )
        for field_name in selection_fields
    )


# Building the selection-filtered candidate inventory
def build_selected_candidates(workflow: dict, policy: dict) -> list[dict]:
    candidate_records = {
        record[
            "candidate_identity"
        ]: record
        for record in workflow[
            "candidate_records"
        ]
    }

    component_candidates = {
        candidate[
            "candidate_identity"
        ]: candidate
        for candidate in workflow[
            "component_candidates"
        ]
    }

    if set(
        candidate_records
    ) != set(
        component_candidates
    ):
        raise RuntimeError("Candidate records and assembled candidates do not reconcile.")

    selected = [
        component_candidates[
            candidate_identity
        ]
        for candidate_identity, candidate_record
        in candidate_records.items()
        if candidate_matches_selection_policy(
            candidate_record,
            policy
        )
    ]

    return sorted(
        selected,
        key = lambda candidate: candidate[
            "candidate_identity"
        ]
    )


# Building one observation-to-prediction identity tuple
def observation_prediction_identity(record: dict) -> tuple:
    return (
        record[
            "condition_protocol_identity"
        ],
        record[
            "cell_identity"
        ],
        float(
            record[
                "horizon"
            ]
        ),
        record[
            "source_identity"
        ]
    )


# Confirming outer predictions reconcile exactly to authorized observations
def outer_predictions_reconcile(observations: list[dict], predictions: list[dict]) -> bool:
    expected_identities = [
        observation_prediction_identity(
            observation
        )
        for observation in observations
        if observation[
            "inside_evaluation_interval"
        ]
    ]

    observed_identities = [
        observation_prediction_identity(
            prediction
        )
        for prediction in predictions
    ]

    return (
        len(
            expected_identities
        )
        == len(
            set(
                expected_identities
            )
        )
        == len(
            observed_identities
        )
        == len(
            set(
                observed_identities
            )
        )
        and set(
            expected_identities
        )
        == set(
            observed_identities
        )
    )


# Selecting the final inspection configuration from outer-fold selections
def select_final_configuration_identity(selected_configuration_identities: list[str]) -> str:
    if not selected_configuration_identities:
        raise RuntimeError("Final inspection requires selected outer-fold configurations.")

    identity_counts = Counter(
        selected_configuration_identities
    )

    return min(
        identity_counts,
        key = lambda identity: (
            -identity_counts[
                identity
            ],
            identity
        )
    )


# Building the deterministic workload plan before any fit
def build_execution_plan(selected_candidates: list[dict], workflow: dict, policy: dict) -> dict:
    available_fold_counts = Counter()

    for fold_record in workflow[
        "fold_records"
    ]:
        if fold_record[
            "fold_status"
        ] == "available":
            available_fold_counts[
                fold_record[
                    "candidate_identity"
                ]
            ] += 1

    configuration_count = sum(
        len(
            family_policy[
                "configurations"
            ]
        )
        for family_policy in policy[
            "model_families"
        ].values()
    )

    family_count = len(
        policy[
            "model_families"
        ]
    )

    outer_fold_count = sum(
        available_fold_counts[
            candidate[
                "candidate_identity"
            ]
        ]
        for candidate in selected_candidates
    )

    inner_fold_upper_bound = int(
        policy[
            "nested_validation"
        ][
            "inner_requested_fold_count"
        ]
    )

    return {
        "selected_candidate_count": len(
            selected_candidates
        ),
        "model_family_count": family_count,
        "configuration_count_per_candidate": (
            configuration_count
        ),
        "outer_fold_count": outer_fold_count,
        "inner_estimator_fit_upper_bound": (
            outer_fold_count
            * configuration_count
            * inner_fold_upper_bound
        ),
        "outer_estimator_fit_upper_bound": (
            outer_fold_count
            * family_count
        ),
        "final_inspection_fit_upper_bound": (
            len(
                selected_candidates
            )
            * family_count
        )
    }


# Running the complete predictive benchmark workflow
def fit_benchmarks(selected_candidates: list[dict], workflow: dict, policy: dict) -> dict:
    configuration_inventory = build_configuration_inventory(
        policy
    )

    fold_records_by_candidate = defaultdict(
        list
    )

    for fold_record in workflow[
        "fold_records"
    ]:
        if fold_record[
            "fold_status"
        ] == "available":
            fold_records_by_candidate[
                fold_record[
                    "candidate_identity"
                ]
            ].append(
                fold_record
            )

    candidate_registry = []
    hyperparameter_selection = []
    validation_predictions = []
    validation_metrics = []
    condition_calibration = []
    feature_importance_records = []

    for candidate in selected_candidates:
        candidate_identity = candidate[
            "candidate_identity"
        ]

        condition_identities = sorted(
            candidate[
                "condition_identities"
            ]
        )

        observations = sorted(
            candidate[
                "included_observations"
            ],
            key = lambda observation: (
                observation[
                    "condition_protocol_identity"
                ],
                observation[
                    "cell_identity"
                ],
                float(
                    observation[
                        "horizon"
                    ]
                ),
                observation[
                    "source_identity"
                ]
            )
        )

        outer_folds = sorted(
            fold_records_by_candidate[
                candidate_identity
            ],
            key = lambda record: int(
                record[
                    "fold_index"
                ]
            )
        )

        for model_family in sorted(
            configuration_inventory
        ):
            benchmark_identity = candidate_benchmark_identity(
                candidate_identity,
                model_family
            )

            family_predictions = []
            selected_configuration_identities = []
            family_failure_reason = None

            for outer_fold in outer_folds:
                outer_fold_index = int(
                    outer_fold[
                        "fold_index"
                    ]
                )

                test_cells = set(
                    outer_fold[
                        "test_cell_identities"
                    ]
                )

                outer_training = [
                    observation
                    for observation in observations
                    if observation[
                        "cell_identity"
                    ]
                    not in test_cells
                ]

                outer_test = [
                    observation
                    for observation in observations
                    if (
                        observation[
                            "cell_identity"
                        ]
                        in test_cells
                        and observation[
                            "inside_evaluation_interval"
                        ]
                    )
                ]

                evaluated_configurations = []

                for configuration_record in configuration_inventory[
                    model_family
                ]:
                    evaluation = evaluate_inner_configuration(
                        candidate_identity,
                        model_family,
                        configuration_record,
                        outer_training,
                        condition_identities,
                        policy
                    )

                    hyperparameter_selection.append(
                        {
                            "benchmark_identity": benchmark_identity,
                            "candidate_identity": candidate_identity,
                            "model_family": model_family,
                            "outer_fold_index": outer_fold_index,
                            "configuration_identity": configuration_record[
                                "configuration_identity"
                            ],
                            "configuration": configuration_record[
                                "configuration"
                            ],
                            "inner_fold_count": evaluation[
                                "fold_count"
                            ],
                            "inner_validation_status": evaluation[
                                "status"
                            ],
                            "inner_validation_reason_code": evaluation[
                                "reason_code"
                            ],
                            "equal_cell_weighted_root_mean_squared_error": evaluation[
                                "metric"
                            ],
                            "selected": False
                        }
                    )

                    if evaluation[
                        "status"
                    ] == "completed":
                        evaluated_configurations.append(
                            (
                                evaluation[
                                    "metric"
                                ],
                                configuration_record[
                                    "configuration_identity"
                                ],
                                configuration_record,
                                evaluation
                            )
                        )

                if not evaluated_configurations:
                    family_failure_reason = (
                        "NO_INNER_VALIDATION_CONFIGURATION_AVAILABLE"
                    )

                    break

                (
                    selected_metric,
                    selected_configuration_identity,
                    selected_configuration,
                    selected_evaluation
                ) = min(
                    evaluated_configurations,
                    key = lambda item: (
                        item[
                            0
                        ],
                        item[
                            1
                        ]
                    )
                )

                for selection_record in hyperparameter_selection:
                    if (
                        selection_record[
                            "benchmark_identity"
                        ]
                        == benchmark_identity
                        and selection_record[
                            "outer_fold_index"
                        ]
                        == outer_fold_index
                        and selection_record[
                            "configuration_identity"
                        ]
                        == selected_configuration_identity
                    ):
                        selection_record[
                            "selected"
                        ] = True

                selected_configuration_identities.append(
                    selected_configuration_identity
                )

                interval_probability = float(
                    policy[
                        "prediction_intervals"
                    ][
                        "coverage_probability"
                    ]
                )

                log_residual_quantile = weighted_quantile(
                    selected_evaluation[
                        "residuals"
                    ],
                    selected_evaluation[
                        "residual_weights"
                    ],
                    interval_probability
                )

                if not outer_test:
                    family_failure_reason = (
                        "EMPTY_OUTER_TEST_FOLD"
                    )

                    break

                if not fold_has_horizon_coverage(
                    outer_training,
                    outer_test
                ):
                    family_failure_reason = (
                        "OUTER_TEST_HORIZON_OUTSIDE_TRAINING_RANGE"
                    )

                    break

                horizon_midpoint, horizon_width = (
                    calculate_training_horizon_scaling(
                        outer_training
                    )
                )

                training_matrix, feature_names = build_feature_matrix(
                    outer_training,
                    condition_identities,
                    horizon_midpoint,
                    horizon_width
                )

                test_matrix, _ = build_feature_matrix(
                    outer_test,
                    condition_identities,
                    horizon_midpoint,
                    horizon_width
                )

                training_target = np.log(
                    np.asarray(
                        [
                            observation[
                                "outcome_value"
                            ]
                            for observation in outer_training
                        ],
                        dtype = float
                    )
                )

                estimator = build_estimator(
                    model_family,
                    selected_configuration[
                        "configuration"
                    ],
                    {
                        "benchmark_identity": benchmark_identity,
                        "outer_fold_index": outer_fold_index,
                        "configuration_identity": (
                            selected_configuration_identity
                        )
                    },
                    policy
                )

                try:
                    fit_estimator(
                        estimator,
                        model_family,
                        training_matrix,
                        training_target,
                        build_cell_weights(
                            outer_training
                        )
                    )

                    (
                        predicted_log,
                        predicted
                    ) = predict_original_scale(
                        estimator,
                        test_matrix
                    )

                    with np.errstate(
                        over = "ignore",
                        invalid = "ignore"
                    ):
                        lower = np.exp(
                            predicted_log
                            - log_residual_quantile
                        )

                        upper = np.exp(
                            predicted_log
                            + log_residual_quantile
                        )

                    if not np.all(
                        np.isfinite(
                            lower
                        )
                        & np.isfinite(
                            upper
                        )
                        & (
                            lower > 0
                        )
                        & (
                            upper > 0
                        )
                        & (
                            lower <= predicted
                        )
                        & (
                            predicted <= upper
                        )
                    ):
                        raise EstimatorUnavailableError("INVALID_PREDICTION_INTERVAL")

                except EstimatorUnavailableError as exc:
                    family_failure_reason = (
                        exc.reason_code
                    )

                    break

                benchmark_predictions = calculate_condition_benchmark(
                    outer_training,
                    outer_test
                )

                for (
                    observation,
                    predicted_value,
                    lower_value,
                    upper_value,
                    benchmark_value
                ) in zip(
                    outer_test,
                    predicted,
                    lower,
                    upper,
                    benchmark_predictions,
                    strict = True
                ):
                    prediction_record = {
                        "benchmark_identity": benchmark_identity,
                        "candidate_identity": candidate_identity,
                        "model_family": model_family,
                        "outer_fold_index": outer_fold_index,
                        "cell_identity": observation[
                            "cell_identity"
                        ],
                        "source_identity": observation[
                            "source_identity"
                        ],
                        "condition_protocol_identity": observation[
                            "condition_protocol_identity"
                        ],
                        "horizon": float(
                            observation[
                                "horizon"
                            ]
                        ),
                        "observed_outcome": float(
                            observation[
                                "outcome_value"
                            ]
                        ),
                        "predicted_outcome": float(
                            predicted_value
                        ),
                        "prediction_interval_lower": float(
                            lower_value
                        ),
                        "prediction_interval_upper": float(
                            upper_value
                        ),
                        "benchmark_prediction": float(
                            benchmark_value
                        ),
                        "prediction_status": "completed",
                        "prediction_reason_code": None,
                        "estimate_label": "predictive"
                    }

                    family_predictions.append(
                        prediction_record
                    )

            if (
                family_failure_reason is None
                and not outer_predictions_reconcile(
                    observations,
                    family_predictions
                )
            ):
                family_failure_reason = (
                    "OUTER_PREDICTION_RECONCILIATION_FAILED"
                )

            if family_failure_reason is None:
                validation_predictions.extend(
                    family_predictions
                )

            if family_failure_reason is None:
                prediction_observations = [
                    {
                        "cell_identity": record[
                            "cell_identity"
                        ],
                        "condition_protocol_identity": record[
                            "condition_protocol_identity"
                        ]
                    }
                    for record in family_predictions
                ]

                observed = np.asarray(
                    [
                        record[
                            "observed_outcome"
                        ]
                        for record in family_predictions
                    ],
                    dtype = float
                )

                predicted = np.asarray(
                    [
                        record[
                            "predicted_outcome"
                        ]
                        for record in family_predictions
                    ],
                    dtype = float
                )

                lower = np.asarray(
                    [
                        record[
                            "prediction_interval_lower"
                        ]
                        for record in family_predictions
                    ],
                    dtype = float
                )

                upper = np.asarray(
                    [
                        record[
                            "prediction_interval_upper"
                        ]
                        for record in family_predictions
                    ],
                    dtype = float
                )

                benchmark = np.asarray(
                    [
                        record[
                            "benchmark_prediction"
                        ]
                        for record in family_predictions
                    ],
                    dtype = float
                )

                metrics = equal_cell_metrics(
                    prediction_observations,
                    observed,
                    predicted,
                    lower,
                    upper,
                    benchmark
                )

                predictive_r_squared = metrics[
                    "equal_cell_weighted_predictive_r_squared"
                ]

                qualification_policy = policy[
                    "metrics"
                ][
                    "analytical_qualification"
                ]

                benchmark_requirement_met = (
                    predictive_r_squared
                    is not None
                )

                if qualification_policy[
                    "predictive_r_squared_must_exceed_benchmark"
                ]:
                    benchmark_requirement_met = (
                        benchmark_requirement_met
                        and predictive_r_squared > 0
                    )

                if benchmark_requirement_met:
                    analytical_status = (
                        qualification_policy[
                            "qualified_status"
                        ]
                    )

                    analytical_reason_code = None

                else:
                    analytical_status = (
                        qualification_policy[
                            "failed_status"
                        ]
                    )

                    analytical_reason_code = (
                        "PREDICTIVE_BENCHMARK_NOT_MET"
                    )

                validation_status = "completed"
                validation_reason_code = None

            else:
                metrics = {
                    "equal_cell_weighted_mean_absolute_error": None,
                    "equal_cell_weighted_root_mean_squared_error": None,
                    "equal_cell_weighted_median_error": None,
                    "equal_cell_weighted_predictive_r_squared": None,
                    "prediction_interval_coverage": None,
                    "mean_prediction_interval_width": None
                }

                validation_status = "unavailable"
                validation_reason_code = family_failure_reason
                analytical_status = "not_available"
                analytical_reason_code = family_failure_reason

            validation_metrics.append(
                {
                    "benchmark_identity": benchmark_identity,
                    "candidate_identity": candidate_identity,
                    "model_family": model_family,
                    "outer_fold_count": len(
                        outer_folds
                    ),
                    "test_cell_count": len(
                        {
                            record[
                                "cell_identity"
                            ]
                            for record in family_predictions
                        }
                    ),
                    "test_observation_count": len(
                        family_predictions
                    ),
                    **metrics,
                    "validation_status": validation_status,
                    "validation_reason_code": validation_reason_code,
                    "analytical_status": analytical_status,
                    "analytical_reason_code": analytical_reason_code,
                    "estimate_label": "predictive"
                }
            )

            candidate_registry.append(
                {
                    "benchmark_identity": benchmark_identity,
                    "candidate_identity": candidate_identity,
                    "support_region_identity": candidate[
                        "collapsed_candidate"
                    ][
                        "support_region_identity"
                    ],
                    "design_scope": candidate[
                        "collapsed_candidate"
                    ][
                        "design_scope"
                    ],
                    "sampling_design_scope": candidate[
                        "collapsed_candidate"
                    ][
                        "sampling_design_scope"
                    ],
                    "model_family": model_family,
                    "outcome": candidate[
                        "collapsed_candidate"
                    ][
                        "outcome"
                    ],
                    "cohort": candidate[
                        "collapsed_candidate"
                    ][
                        "cohort"
                    ],
                    "aging_mode": candidate[
                        "collapsed_candidate"
                    ][
                        "aging_mode"
                    ],
                    "horizon_axis": candidate[
                        "collapsed_candidate"
                    ][
                        "horizon_axis"
                    ],
                    "support_interval_minimum": float(
                        candidate[
                            "collapsed_candidate"
                        ][
                            "support_interval_minimum"
                        ]
                    ),
                    "support_interval_maximum": float(
                        candidate[
                            "collapsed_candidate"
                        ][
                            "support_interval_maximum"
                        ]
                    ),
                    "support_interval_lower_inclusive": candidate[
                        "collapsed_candidate"
                    ][
                        "support_interval_lower_inclusive"
                    ],
                    "support_interval_upper_inclusive": candidate[
                        "collapsed_candidate"
                    ][
                        "support_interval_upper_inclusive"
                    ],
                    "condition_protocol_identities": condition_identities,
                    "condition_count": len(
                        condition_identities
                    ),
                    "independent_cell_count": len(
                        {
                            observation[
                                "cell_identity"
                            ]
                            for observation in observations
                        }
                    ),
                    "observation_count": len(
                        observations
                    ),
                    "outer_fold_count": len(
                        outer_folds
                    ),
                    "execution_status": (
                        "completed"
                        if family_failure_reason is None
                        else "unavailable"
                    ),
                    "execution_reason_code": family_failure_reason,
                    "analytical_status": analytical_status,
                    "analytical_reason_code": analytical_reason_code,
                    "selected_configuration_identities": (
                        selected_configuration_identities
                    )
                }
            )

            if family_failure_reason is not None:
                feature_importance_records.extend(
                    build_unavailable_feature_importance_records(
                        benchmark_identity = (
                            benchmark_identity
                        ),
                        candidate_identity = (
                            candidate_identity
                        ),
                        model_family = model_family,
                        configuration_identity = None,
                        feature_names = (
                            build_feature_names(
                                condition_identities
                            )
                        ),
                        reason_code = (
                            family_failure_reason
                        )
                    )
                )

                continue

            has_incomplete_calibration = False

            for condition_identity in condition_identities:
                condition_predictions = [
                    record
                    for record in family_predictions
                    if record[
                        "condition_protocol_identity"
                    ] == condition_identity
                ]

                if not condition_predictions:
                    has_incomplete_calibration = True

                    condition_calibration.append(
                        {
                            "benchmark_identity": benchmark_identity,
                            "candidate_identity": candidate_identity,
                            "model_family": model_family,
                            "condition_protocol_identity": (
                                condition_identity
                            ),
                            "independent_cell_count": 0,
                            "observation_count": 0,
                            "equal_cell_weighted_mean_observed": (
                                None
                            ),
                            "equal_cell_weighted_mean_predicted": (
                                None
                            ),
                            "equal_cell_weighted_mean_error": (
                                None
                            ),
                            "equal_cell_weighted_mean_absolute_error": (
                                None
                            ),
                            "equal_cell_weighted_root_mean_squared_error": (
                                None
                            ),
                            "calibration_status": "unavailable",
                            "calibration_reason_code": (
                                "EMPTY_CONDITION_PREDICTION_INVENTORY"
                            ),
                            "estimate_label": "predictive"
                        }
                    )

                    continue

                calibration_observations = [
                    {
                        "cell_identity": record[
                            "cell_identity"
                        ]
                    }
                    for record in condition_predictions
                ]

                observed_values = np.asarray(
                    [
                        record[
                            "observed_outcome"
                        ]
                        for record in condition_predictions
                    ],
                    dtype = float
                )

                predicted_values = np.asarray(
                    [
                        record[
                            "predicted_outcome"
                        ]
                        for record in condition_predictions
                    ],
                    dtype = float
                )

                condition_weights = build_cell_weights(
                    calibration_observations
                )

                condition_calibration.append(
                    {
                        "benchmark_identity": benchmark_identity,
                        "candidate_identity": candidate_identity,
                        "model_family": model_family,
                        "condition_protocol_identity": condition_identity,
                        "independent_cell_count": len(
                            {
                                record[
                                    "cell_identity"
                                ]
                                for record in condition_predictions
                            }
                        ),
                        "observation_count": len(
                            condition_predictions
                        ),
                        "equal_cell_weighted_mean_observed": float(
                            np.average(
                                observed_values,
                                weights = condition_weights
                            )
                        ),
                        "equal_cell_weighted_mean_predicted": float(
                            np.average(
                                predicted_values,
                                weights = condition_weights
                            )
                        ),
                        "equal_cell_weighted_mean_error": float(
                            np.average(
                                predicted_values
                                - observed_values,
                                weights = condition_weights
                            )
                        ),
                        "equal_cell_weighted_mean_absolute_error": float(
                            np.average(
                                np.abs(
                                    predicted_values
                                    - observed_values
                                ),
                                weights = condition_weights
                            )
                        ),
                        "equal_cell_weighted_root_mean_squared_error": float(
                            math.sqrt(
                                np.average(
                                    np.square(
                                        predicted_values
                                        - observed_values
                                    ),
                                    weights = condition_weights
                                )
                            )
                        ),
                        "calibration_status": "completed",
                        "calibration_reason_code": None,
                        "estimate_label": "predictive"
                    }
                )

            final_configuration_identity = (
                select_final_configuration_identity(
                    selected_configuration_identities
                )
            )

            if (
                has_incomplete_calibration
                and analytical_status
                == policy[
                    "metrics"
                ][
                    "analytical_qualification"
                ][
                    "qualified_status"
                ]
            ):
                analytical_status = policy[
                    "metrics"
                ][
                    "analytical_qualification"
                ][
                    "failed_status"
                ]

                analytical_reason_code = (
                    "INCOMPLETE_CONDITION_CALIBRATION"
                )

                candidate_registry[
                    -1
                ][
                    "analytical_status"
                ] = analytical_status

                candidate_registry[
                    -1
                ][
                    "analytical_reason_code"
                ] = analytical_reason_code

                validation_metrics[
                    -1
                ][
                    "analytical_status"
                ] = analytical_status

                validation_metrics[
                    -1
                ][
                    "analytical_reason_code"
                ] = analytical_reason_code

            final_configuration_identity = (
                select_final_configuration_identity(
                    selected_configuration_identities
                )
            )

            final_configuration = next(
                record
                for record in configuration_inventory[
                    model_family
                ]
                if record[
                    "configuration_identity"
                ] == final_configuration_identity
            )

            horizon_midpoint, horizon_width = (
                calculate_training_horizon_scaling(
                    observations
                )
            )

            full_matrix, feature_names = build_feature_matrix(
                observations,
                condition_identities,
                horizon_midpoint,
                horizon_width
            )

            if analytical_status != policy[
                "explainability"
            ][
                "analytical_status_required"
            ]:
                feature_importance_records.extend(
                    build_unavailable_feature_importance_records(
                        benchmark_identity = benchmark_identity,
                        candidate_identity = candidate_identity,
                        model_family = model_family,
                        configuration_identity = (
                            final_configuration_identity
                        ),
                        feature_names = feature_names,
                        reason_code = (
                            "MODEL_NOT_ANALYTICALLY_QUALIFIED"
                        )
                    )
                )

                continue

            final_estimator = build_estimator(
                model_family,
                final_configuration[
                    "configuration"
                ],
                {
                    "benchmark_identity": benchmark_identity,
                    "fit_scope": "full_data_inspection",
                    "configuration_identity": (
                        final_configuration_identity
                    )
                },
                policy
            )

            try:
                fit_estimator(
                    final_estimator,
                    model_family,
                    full_matrix,
                    np.log(
                        np.asarray(
                            [
                                observation[
                                    "outcome_value"
                                ]
                                for observation in observations
                            ],
                            dtype = float
                        )
                    ),
                    build_cell_weights(
                        observations
                    )
                )

                importances, importance_method = (
                    build_feature_importance(
                        final_estimator,
                        model_family,
                        feature_names
                    )
                )

            except EstimatorUnavailableError as exc:
                feature_importance_records.extend(
                    build_unavailable_feature_importance_records(
                        benchmark_identity = benchmark_identity,
                        candidate_identity = candidate_identity,
                        model_family = model_family,
                        configuration_identity = (
                            final_configuration_identity
                        ),
                        feature_names = feature_names,
                        reason_code = exc.reason_code
                    )
                )

                continue

            for feature_name, importance_value in zip(
                feature_names,
                importances,
                strict = True
            ):
                feature_importance_records.append(
                    {
                        "benchmark_identity": benchmark_identity,
                        "candidate_identity": candidate_identity,
                        "model_family": model_family,
                        "configuration_identity": final_configuration_identity,
                        "encoded_feature": feature_name,
                        "feature_group": feature_group(
                            feature_name
                        ),
                        "importance_value": float(
                            importance_value
                        ),
                        "importance_method": importance_method,
                        "importance_status": "completed",
                        "importance_reason_code": None,
                        "estimate_label": (
                            "predictive_model_inspection"
                        )
                    }
                )

    return {
        "candidate_registry": sorted(
            candidate_registry,
            key = lambda record: record[
                "benchmark_identity"
            ]
        ),
        "hyperparameter_selection": sorted(
            hyperparameter_selection,
            key = lambda record: (
                record[
                    "benchmark_identity"
                ],
                record[
                    "outer_fold_index"
                ],
                record[
                    "configuration_identity"
                ]
            )
        ),
        "validation_predictions": sorted(
            validation_predictions,
            key = lambda record: (
                record[
                    "benchmark_identity"
                ],
                record[
                    "outer_fold_index"
                ],
                record[
                    "condition_protocol_identity"
                ],
                record[
                    "cell_identity"
                ],
                record[
                    "horizon"
                ],
                record[
                    "source_identity"
                ]
            )
        ),
        "validation_metrics": sorted(
            validation_metrics,
            key = lambda record: record[
                "benchmark_identity"
            ]
        ),
        "condition_calibration": sorted(
            condition_calibration,
            key = lambda record: (
                record[
                    "benchmark_identity"
                ],
                record[
                    "condition_protocol_identity"
                ]
            )
        ),
        "feature_importance": sorted(
            feature_importance_records,
            key = lambda record: (
                record[
                    "benchmark_identity"
                ],
                record[
                    "encoded_feature"
                ]
            )
        )
    }


# Building validation accounting and fold records for selected candidates
def build_validation_records(selected_candidates: list[dict], workflow: dict) -> tuple[list[dict], list[dict]]:
    selected_identities = {
        candidate[
            "candidate_identity"
        ]
        for candidate in selected_candidates
    }

    accounting = []

    for candidate in selected_candidates:
        observations = candidate[
            "included_observations"
        ]

        accounting.append(
            {
                "candidate_identity": candidate[
                    "candidate_identity"
                ],
                "source_observation_count": len(
                    observations
                ),
                "inside_evaluation_interval_count": sum(
                    observation[
                        "inside_evaluation_interval"
                    ]
                    for observation in observations
                ),
                "outside_evaluation_interval_count": sum(
                    not observation[
                        "inside_evaluation_interval"
                    ]
                    for observation in observations
                ),
                "independent_cell_count": len(
                    {
                        observation[
                            "cell_identity"
                        ]
                        for observation in observations
                    }
                ),
                "condition_count": len(
                    candidate[
                        "condition_identities"
                    ]
                )
            }
        )

    folds = []

    for fold in workflow[
        "fold_records"
    ]:
        if fold[
            "candidate_identity"
        ] not in selected_identities:
            continue

        training_cells = set(
            fold[
                "training_cell_identities"
            ]
        )

        test_cells = set(
            fold[
                "test_cell_identities"
            ]
        )

        if len(
            training_cells
        ) != int(
            fold[
                "training_cell_count"
            ]
        ):
            raise RuntimeError(
                "Fold training cell count does not match its unique identities : "
                f"{fold['candidate_identity']} : {fold['fold_index']}"
            )

        if len(
            test_cells
        ) != int(
            fold[
                "test_cell_count"
            ]
        ):
            raise RuntimeError(
                "Fold test cell count does not match its unique identities : "
                f"{fold['candidate_identity']} : {fold['fold_index']}"
            )

        if training_cells.intersection(
            test_cells
        ):
            raise RuntimeError(
                "Fold contains cell leakage : "
                f"{fold['candidate_identity']} : {fold['fold_index']}"
            )

        folds.append(
            {
                "candidate_identity": fold[
                    "candidate_identity"
                ],
                "fold_index": int(
                    fold[
                        "fold_index"
                    ]
                ),
                "training_cell_count": int(
                    fold[
                        "training_cell_count"
                    ]
                ),
                "test_cell_count": int(
                    fold[
                        "test_cell_count"
                    ]
                ),
                "training_cell_identities": sorted(
                    training_cells
                ),
                "test_cell_identities": sorted(
                    test_cells
                ),
                "training_observation_count": int(
                    fold[
                        "training_observation_count"
                    ]
                ),
                "test_observation_count": int(
                    fold[
                        "test_observation_count"
                    ]
                ),
                "training_condition_count": int(
                    fold[
                        "training_condition_count"
                    ]
                ),
                "training_minimum_condition_cell_count": int(
                    fold[
                        "training_minimum_condition_cell_count"
                    ]
                ),
                "training_minimum_condition_repeated_cell_count": int(
                    fold[
                        "training_minimum_condition_repeated_cell_count"
                    ]
                ),
                "fold_status": fold[
                    "fold_status"
                ],
                "fold_reason_code": fold[
                    "fold_reason_code"
                ]
            }
        )

    return (
        sorted(
            accounting,
            key = lambda record: record[
                "candidate_identity"
            ]
        ),
        sorted(
            folds,
            key = lambda record: (
                record[
                    "candidate_identity"
                ],
                record[
                    "fold_index"
                ]
            )
        )
    )


# Reconciling fitted analytical tables independently of report summaries
def validate_fit_table_reconciliation(tables: dict, policy: dict) -> None:
    registry_records = tables[
        "candidate_registry"
    ]

    metric_records = tables[
        "validation_metrics"
    ]

    selection_records = tables[
        "hyperparameter_selection"
    ]

    prediction_records = tables[
        "validation_predictions"
    ]

    calibration_records = tables[
        "condition_calibration"
    ]

    importance_records = tables[
        "feature_importance"
    ]

    registry_by_identity = {
        record[
            "benchmark_identity"
        ]: record
        for record in registry_records
    }

    metric_by_identity = {
        record[
            "benchmark_identity"
        ]: record
        for record in metric_records
    }

    if (
        len(
            registry_by_identity
        )
        != len(
            registry_records
        )
        or len(
            metric_by_identity
        )
        != len(
            metric_records
        )
        or set(
            registry_by_identity
        )
        != set(
            metric_by_identity
        )
    ):
        raise RuntimeError("Predictive registry and metric inventories do not reconcile.")

    selections_by_benchmark = defaultdict(
        list
    )

    predictions_by_benchmark = defaultdict(
        list
    )

    calibration_by_benchmark = defaultdict(
        list
    )

    importance_by_benchmark = defaultdict(
        list
    )

    for record in selection_records:
        selections_by_benchmark[
            record[
                "benchmark_identity"
            ]
        ].append(
            record
        )

    for record in prediction_records:
        predictions_by_benchmark[
            record[
                "benchmark_identity"
            ]
        ].append(
            record
        )

    for record in calibration_records:
        calibration_by_benchmark[
            record[
                "benchmark_identity"
            ]
        ].append(
            record
        )

    for record in importance_records:
        importance_by_benchmark[
            record[
                "benchmark_identity"
            ]
        ].append(
            record
        )

    configuration_inventory = (
        build_configuration_inventory(
            policy
        )
    )

    for benchmark_identity, registry_record in (
        registry_by_identity.items()
    ):
        metric_record = metric_by_identity[
            benchmark_identity
        ]

        model_family = registry_record[
            "model_family"
        ]

        candidate_identity = registry_record[
            "candidate_identity"
        ]

        if (
            metric_record[
                "candidate_identity"
            ]
            != candidate_identity
            or metric_record[
                "model_family"
            ]
            != model_family
            or metric_record[
                "analytical_status"
            ]
            != registry_record[
                "analytical_status"
            ]
            or metric_record[
                "analytical_reason_code"
            ]
            != registry_record[
                "analytical_reason_code"
            ]
        ):
            raise RuntimeError("Predictive registry and metric statuses disagree.")

        benchmark_predictions = predictions_by_benchmark[
            benchmark_identity
        ]

        benchmark_calibration = calibration_by_benchmark[
            benchmark_identity
        ]

        benchmark_selections = selections_by_benchmark[
            benchmark_identity
        ]

        benchmark_importance = importance_by_benchmark[
            benchmark_identity
        ]

        expected_feature_count = (
            2
            * int(
                registry_record[
                    "condition_count"
                ]
            )
            - 1
        )

        if len(
            benchmark_importance
        ) != expected_feature_count:
            raise RuntimeError("Predictive feature-importance inventory is incomplete.")

        importance_statuses = {
            record[
                "importance_status"
            ]
            for record in benchmark_importance
        }

        if (
            registry_record[
                "analytical_status"
            ]
            != "available"
            and importance_statuses
            != {
                "unavailable"
            }
        ):
            raise RuntimeError(
                "Unqualified benchmark contains completed feature importance."
            )

        for importance_record in benchmark_importance:
            if (
                importance_record[
                    "candidate_identity"
                ]
                != candidate_identity
                or importance_record[
                    "model_family"
                ]
                != model_family
            ):
                raise RuntimeError("Predictive feature importance has inconsistent provenance.")

            if importance_record[
                "importance_status"
            ] == "completed":
                importance_value = float(
                    importance_record[
                        "importance_value"
                    ]
                )

                if (
                    not math.isfinite(
                        importance_value
                    )
                    or importance_value < 0
                ):
                    raise RuntimeError("Predictive feature importance is invalid.")

        if registry_record[
            "execution_status"
        ] == "completed":
            if (
                metric_record[
                    "validation_status"
                ]
                != "completed"
                or len(
                    benchmark_predictions
                )
                != int(
                    metric_record[
                        "test_observation_count"
                    ]
                )
                or len(
                    benchmark_predictions
                )
                == 0
            ):
                raise RuntimeError("Completed predictive benchmark has incomplete validation.")

            expected_conditions = set(
                registry_record[
                    "condition_protocol_identities"
                ]
            )

            observed_conditions = {
                record[
                    "condition_protocol_identity"
                ]
                for record in benchmark_calibration
            }

            if observed_conditions != expected_conditions:
                raise RuntimeError("Predictive condition calibration is incomplete.")

            outer_fold_count = int(
                registry_record[
                    "outer_fold_count"
                ]
            )

            selected_identities = registry_record[
                "selected_configuration_identities"
            ]

            if len(
                selected_identities
            ) != outer_fold_count:
                raise RuntimeError("Predictive selected-configuration inventory is incomplete.")

            expected_configuration_identities = {
                record[
                    "configuration_identity"
                ]
                for record in configuration_inventory[
                    model_family
                ]
            }

            selections_by_fold = defaultdict(
                list
            )

            for selection_record in benchmark_selections:
                selections_by_fold[
                    int(
                        selection_record[
                            "outer_fold_index"
                        ]
                    )
                ].append(
                    selection_record
                )

            if len(
                selections_by_fold
            ) != outer_fold_count:
                raise RuntimeError("Predictive hyperparameter-selection folds are incomplete.")

            for fold_index, fold_records in sorted(
                selections_by_fold.items()
            ):
                if (
                    {
                        record[
                            "configuration_identity"
                        ]
                        for record in fold_records
                    }
                    != expected_configuration_identities
                    or sum(
                        bool(
                            record[
                                "selected"
                            ]
                        )
                        for record in fold_records
                    )
                    != 1
                ):
                    raise RuntimeError("Predictive hyperparameter selection is incomplete.")

        else:
            if (
                metric_record[
                    "validation_status"
                ]
                != "unavailable"
                or registry_record[
                    "analytical_status"
                ]
                != "not_available"
                or benchmark_predictions
                or benchmark_calibration
            ):
                raise RuntimeError("Unavailable predictive benchmark contains analytical output.")


# Validating one staged or published predictive-benchmarking report
def validate_report(report_path: Path, expected_policy: dict,
                    expected_policy_sha256: str, expected_input_hashes: dict,
                    expected_operation: str, expected_selected_candidate_count: int,
                    expected_upstream_candidate_count: int, expected_execution_plan: dict,
                    expected_benchmark_candidate_count: int, expected_analytical_status_counts: dict,
                    expected_analytical_status_counts_by_model_family: dict, expected_feature_importance_status_counts: dict,
                    expected_analytical_handoff_status: str, expected_scope_fields: list[str]
) -> None:
    report = load_json_object(
        report_path
    )

    if report.get(
        "operation"
    ) != expected_operation:
        raise RuntimeError("Predictive report has an inconsistent operation.")

    if report.get(
        "policy_name"
    ) != expected_policy[
        "policy_name"
    ]:
        raise RuntimeError("Predictive report has an inconsistent policy name.")

    if report.get(
        "policy_version"
    ) != expected_policy[
        "policy_version"
    ]:
        raise RuntimeError("Predictive report has an inconsistent policy version.")

    if report.get(
        "policy_sha256"
    ) != expected_policy_sha256:
        raise RuntimeError("Predictive report has an inconsistent policy digest.")

    if report.get(
        "implementation_sha256"
    ) != calculate_file_sha256(
        Path(
            __file__
        )
    ):
        raise RuntimeError("Predictive report has an inconsistent implementation digest.")

    if report.get(
        "input_hashes"
    ) != expected_input_hashes:
        raise RuntimeError("Predictive report has inconsistent input hashes.")

    if report.get(
        "selected_candidate_count"
    ) != expected_selected_candidate_count:
        raise RuntimeError("Predictive report has an inconsistent selected-candidate count.")

    if report.get(
        "upstream_candidate_count"
    ) != expected_upstream_candidate_count:
        raise RuntimeError(
            "Predictive report has an inconsistent upstream-candidate count."
        )

    if report.get(
        "execution_plan"
    ) != expected_execution_plan:
        raise RuntimeError("Predictive report has an inconsistent execution plan.")

    if report.get(
        "benchmark_candidate_count"
    ) != expected_benchmark_candidate_count:
        raise RuntimeError("Predictive report has an inconsistent benchmark-candidate count.")

    if report.get(
        "analytical_status_counts"
    ) != expected_analytical_status_counts:
        raise RuntimeError("Predictive report has inconsistent analytical-status counts.")

    if report.get(
        "analytical_status_counts_by_model_family"
    ) != expected_analytical_status_counts_by_model_family:
        raise RuntimeError("Predictive report has inconsistent per-family analytical-status counts.")

    if report.get(
        "feature_importance_status_counts"
    ) != expected_feature_importance_status_counts:
        raise RuntimeError("Predictive report has inconsistent feature-importance status counts.")

    if report.get(
        "analytical_handoff",
        {}
    ).get(
        "status"
    ) != expected_analytical_handoff_status:
        raise RuntimeError("Predictive report has an inconsistent analytical-handoff status.")

    if report.get(
        "candidate_selection_uses_outcome_magnitude_direction_or_performance"
    ) is not False:
        raise RuntimeError("Predictive report incorrectly declares outcome-informed candidate selection.")

    if report.get(
        "candidate_selection_inherits_outcome_validity_requirements"
    ) is not True:
        raise RuntimeError("Predictive report does not declare inherited outcome-validity screening.")

    if report.get(
        "causal_interpretation_authorized"
    ) is not False:
        raise RuntimeError("Predictive report incorrectly authorizes causal interpretation.")

    if report.get(
        "policy_recommendation_authorized"
    ) is not False:
        raise RuntimeError("Predictive report incorrectly authorizes policy recommendation.")

    if report.get(
        "novel_condition_prediction_authorized"
    ) is not False:
        raise RuntimeError("Predictive report incorrectly authorizes novel-condition prediction.")

    if report.get(
        "performance_selected_winner_declared"
    ) is not False:
        raise RuntimeError("Predictive report incorrectly declares a performance-selected winner.")

    if report.get(
        "feature_importance_is_causal"
    ) is not False:
        raise RuntimeError("Predictive report incorrectly declares causal feature importance.")

    if report.get(
        "independent_validation_unit"
    ) != "cell_identity":
        raise RuntimeError("Predictive report has an inconsistent independent validation unit.")

    for separation_field in (
        "calendar_and_cycle_models_separated",
        "primary_and_sensitivity_cohorts_separated",
        "elapsed_time_and_cumulative_fec_separated"
    ):
        if report.get(
            separation_field
        ) is not True:
            raise RuntimeError(f"Predictive report has an invalid separation declaration : {separation_field}")

    table_manifest = report.get(
        "tables",
        {}
    )

    if set(
        table_manifest
    ) != set(
        expected_policy[
            "output_contract"
        ][
            "tables"
        ]
    ):
        raise RuntimeError("Predictive report has an unexpected table inventory.")

    verified_tables = {}

    for table_name, table_record in table_manifest.items():
        table_path = (
            report_path.parent
            / table_record[
                "file_name"
            ]
        )

        if not table_path.is_file():
            raise RuntimeError(f"Declared predictive table is missing : {table_name}")

        if (
            calculate_file_sha256(
                table_path
            )
            != table_record[
                "sha256"
            ]
        ):
            raise RuntimeError(f"Predictive table SHA-256 does not match its manifest : {table_name}")

        field_names = list(
            csv.DictReader(
                table_path.open(
                    mode = "r",
                    encoding = "utf-8",
                    newline = ""
                )
            ).fieldnames
            or []
        )

        expected_schema_names = [
            entry[
                "name"
            ]
            for entry in table_record[
                "schema"
            ]
        ]

        if field_names != expected_schema_names:
            raise RuntimeError(f"Predictive table schema does not match its manifest : {table_name}")

        if table_name == "candidate_registry":
            missing_scope_fields = sorted(
                set(
                    expected_scope_fields
                ).difference(
                    field_names
                )
            )

            if missing_scope_fields:
                raise RuntimeError("Predictive candidate registry is missing required scope fields : ", str(missing_scope_fields))

        table_records = read_csv_table(
            table_path
        )

        if (
            len(
                table_records
            )
            != int(
                table_record[
                    "row_count"
                ]
            )
        ):
            raise RuntimeError(f"Predictive table row count does not match its manifest : {table_name}")

        verified_tables[
            table_name
        ] = table_records

    accounting_records = verified_tables[
        "observation_accounting"
    ]

    fold_records = verified_tables[
        "validation_folds"
    ]

    registry_records = verified_tables[
        "candidate_registry"
    ]

    selected_candidate_identities = {
        record[
            "candidate_identity"
        ]
        for record in accounting_records
    }

    if (
        len(
            accounting_records
        )
        != len(
            selected_candidate_identities
        )
        or len(
            accounting_records
        )
        != expected_selected_candidate_count
    ):
        raise RuntimeError("Predictive observation accounting does not reconcile.")

    for fold_record in fold_records:
        training_cells = set(
            fold_record[
                "training_cell_identities"
            ]
        )

        test_cells = set(
            fold_record[
                "test_cell_identities"
            ]
        )

        if training_cells.intersection(
            test_cells
        ):
            raise RuntimeError("Predictive fold contains cell leakage.")

        if (
            len(
                training_cells
            )
            != int(
                fold_record[
                    "training_cell_count"
                ]
            )
            or len(
                test_cells
            )
            != int(
                fold_record[
                    "test_cell_count"
                ]
            )
        ):
            raise RuntimeError("Predictive fold cell counts do not reconcile.")

    if expected_operation == "validate":
        result_tables = (
            "candidate_registry",
            "hyperparameter_selection",
            "validation_predictions",
            "validation_metrics",
            "condition_calibration",
            "feature_importance"
        )

        if any(
            verified_tables[
                table_name
            ]
            for table_name in result_tables
        ):
            raise RuntimeError("Validation-only execution contains fitted results.")

        return

    validate_fit_table_reconciliation(
        verified_tables,
        expected_policy
    )

    benchmark_identities = [
        record[
            "benchmark_identity"
        ]
        for record in registry_records
    ]

    if (
        len(
            benchmark_identities
        )
        != len(
            set(
                benchmark_identities
            )
        )
        or len(
            registry_records
        )
        != expected_benchmark_candidate_count
    ):
        raise RuntimeError("Predictive benchmark registry does not reconcile.")

    expected_benchmark_identities = {
        candidate_benchmark_identity(
            candidate_identity,
            model_family
        )
        for candidate_identity in (
            selected_candidate_identities
        )
        for model_family in expected_policy[
            "model_families"
        ]
    }

    if (
        set(
            benchmark_identities
        )
        != expected_benchmark_identities
    ):
        raise RuntimeError("Predictive benchmark identities are incomplete.")

    metric_records = verified_tables[
        "validation_metrics"
    ]

    if {
        record[
            "benchmark_identity"
        ]
        for record in metric_records
    } != expected_benchmark_identities:
        raise RuntimeError("Predictive validation metrics are incomplete.")

    prediction_records = verified_tables[
        "validation_predictions"
    ]

    prediction_identities = [
        (
            record[
                "benchmark_identity"
            ],
            record[
                "outer_fold_index"
            ],
            record[
                "condition_protocol_identity"
            ],
            record[
                "cell_identity"
            ],
            record[
                "horizon"
            ],
            record[
                "source_identity"
            ]
        )
        for record in prediction_records
    ]

    if len(
        prediction_identities
    ) != len(
        set(
            prediction_identities
        )
    ):
        raise RuntimeError("Predictive output contains duplicate predictions.")

    for prediction_record in prediction_records:
        numerical_values = [
            float(
                prediction_record[
                    field_name
                ]
            )
            for field_name in (
                "observed_outcome",
                "predicted_outcome",
                "prediction_interval_lower",
                "prediction_interval_upper",
                "benchmark_prediction"
            )
        ]

        if (
            not all(
                math.isfinite(
                    value
                )
                for value in numerical_values
            )
            or numerical_values[
                1
            ] <= 0
            or numerical_values[
                2
            ] <= 0
            or numerical_values[
                3
            ] <= 0
            or not (
                numerical_values[
                    2
                ]
                <= numerical_values[
                    1
                ]
                <= numerical_values[
                    3
                ]
            )
        ):
            raise RuntimeError("Predictive output contains an invalid prediction.")

    registry_by_identity = {
        record[
            "benchmark_identity"
        ]: record
        for record in registry_records
    }

    importance_records = verified_tables[
        "feature_importance"
    ]

    importance_by_benchmark = defaultdict(
        list
    )

    for importance_record in importance_records:
        importance_by_benchmark[
            importance_record[
                "benchmark_identity"
            ]
        ].append(
            importance_record
        )

    if set(
        importance_by_benchmark
    ) != expected_benchmark_identities:
        raise RuntimeError("Predictive feature-importance inventory is incomplete.")

    for benchmark_identity, benchmark_record in (
        registry_by_identity.items()
    ):
        expected_feature_count = (
            2
            * int(
                benchmark_record[
                    "condition_count"
                ]
            )
            - 1
        )

        benchmark_importance = (
            importance_by_benchmark[
                benchmark_identity
            ]
        )

        if (
            len(
                benchmark_importance
            )
            != expected_feature_count
            or len(
                {
                    record[
                        "encoded_feature"
                    ]
                    for record in benchmark_importance
                }
            )
            != expected_feature_count
        ):
            raise RuntimeError("Predictive feature-importance rows do not reconcile.")

    observed_status_counts = dict(
        sorted(
            Counter(
                record[
                    "analytical_status"
                ]
                for record in registry_records
            ).items()
        )
    )

    if (
        observed_status_counts
        != report[
            "analytical_status_counts"
        ]
    ):
        raise RuntimeError("Predictive analytical-status counts do not reconcile.")

    observed_status_counts_by_model_family = {}

    for model_family in sorted(
        expected_policy[
            "model_families"
        ]
    ):
        observed_status_counts_by_model_family[
            model_family
        ] = dict(
            sorted(
                Counter(
                    record[
                        "analytical_status"
                    ]
                    for record in registry_records
                    if record[
                        "model_family"
                    ] == model_family
                ).items()
            )
        )

    if (
        observed_status_counts_by_model_family
        != report[
            "analytical_status_counts_by_model_family"
        ]
    ):
        raise RuntimeError("Predictive per-family status counts do not reconcile.")

    observed_importance_status_counts = dict(
        sorted(
            Counter(
                record[
                    "importance_status"
                ]
                for record in importance_records
            ).items()
        )
    )

    if (
        observed_importance_status_counts
        != report[
            "feature_importance_status_counts"
        ]
    ):
        raise RuntimeError("Predictive feature-importance status counts do not reconcile.")

    available_count = observed_status_counts.get(
        "available",
        0
    )

    diagnostic_count = observed_status_counts.get(
        "diagnostic_only",
        0
    )

    unavailable_count = observed_status_counts.get(
        "not_available",
        0
    )

    analytical_handoff = report[
        "analytical_handoff"
    ]

    if (
        analytical_handoff[
            "available_benchmark_count"
        ]
        != available_count
        or analytical_handoff[
            "diagnostic_only_benchmark_count"
        ]
        != diagnostic_count
        or analytical_handoff[
            "unavailable_benchmark_count"
        ]
        != unavailable_count
    ):
        raise RuntimeError("Predictive analytical-handoff counts do not reconcile.")


# Publishing artifacts transactionally through a staged directory
def publish_artifacts(args: argparse.Namespace, policy: dict,
                      workflow: dict, selected_candidates: list[dict],
                      fit_outputs: dict | None, software_environment: dict
) -> None:
    output_parent = args.output_directory.parent

    output_parent.mkdir(
        parents = True,
        exist_ok = True
    )

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix = (
                f".{args.output_directory.name}."
            ),
            dir = output_parent
        )
    )

    try:
        accounting, folds = build_validation_records(
            selected_candidates,
            workflow
        )

        if fit_outputs is None:
            candidate_registry = []
            hyperparameter_selection = []
            validation_predictions = []
            validation_metrics = []
            condition_calibration = []
            feature_importance = []

        else:
            candidate_registry = fit_outputs[
                "candidate_registry"
            ]

            hyperparameter_selection = fit_outputs[
                "hyperparameter_selection"
            ]

            validation_predictions = fit_outputs[
                "validation_predictions"
            ]

            validation_metrics = fit_outputs[
                "validation_metrics"
            ]

            condition_calibration = fit_outputs[
                "condition_calibration"
            ]

            feature_importance = fit_outputs[
                "feature_importance"
            ]

        table_definitions = {
            "candidate_registry": (
                CANDIDATE_REGISTRY_FIELDS,
                candidate_registry
            ),
            "observation_accounting": (
                OBSERVATION_ACCOUNTING_FIELDS,
                accounting
            ),
            "validation_folds": (
                VALIDATION_FOLD_FIELDS,
                folds
            ),
            "hyperparameter_selection": (
                HYPERPARAMETER_SELECTION_FIELDS,
                hyperparameter_selection
            ),
            "validation_predictions": (
                VALIDATION_PREDICTION_FIELDS,
                validation_predictions
            ),
            "validation_metrics": (
                VALIDATION_METRIC_FIELDS,
                validation_metrics
            ),
            "condition_calibration": (
                CONDITION_CALIBRATION_FIELDS,
                condition_calibration
            ),
            "feature_importance": (
                FEATURE_IMPORTANCE_FIELDS,
                feature_importance
            )
        }

        table_manifests = {}

        for table_name, (
            fields,
            records
        ) in table_definitions.items():
            table_path = (
                temporary_directory
                / f"{table_name}.csv"
            )

            write_csv_table(
                table_path,
                fields,
                records
            )

            table_manifests[
                table_name
            ] = {
                "file_name": table_path.name,
                "row_count": len(
                    records
                ),
                "sha256": calculate_file_sha256(
                    table_path
                ),
                "schema": [
                    {
                        "name": field_name
                    }
                    for field_name in fields
                ]
            }

        analytical_status_counts = Counter(
            record[
                "analytical_status"
            ]
            for record in candidate_registry
        )

        feature_importance_status_counts = Counter(
            record[
                "importance_status"
            ]
            for record in feature_importance
        )

        analytical_status_counts_by_model_family = {}

        for model_family in sorted(
            policy[
                "model_families"
            ]
        ):
            analytical_status_counts_by_model_family[
                model_family
            ] = dict(
                sorted(
                    Counter(
                        record[
                            "analytical_status"
                        ]
                        for record in candidate_registry
                        if record[
                            "model_family"
                        ] == model_family
                    ).items()
                )
            )

        if fit_outputs is None:
            analytical_handoff_status = "not_evaluated"

        elif analytical_status_counts.get(
            "available",
            0
        ) == 0:
            analytical_handoff_status = (
                "not_ready_no_qualified_predictive_benchmarks"
            )

        elif analytical_status_counts.get(
            "available",
            0
        ) == len(
            candidate_registry
        ):
            analytical_handoff_status = (
                "ready_for_predictive_reporting"
            )

        else:
            analytical_handoff_status = (
                "ready_for_predictive_reporting_with_candidate_restrictions"
            )

        if fit_outputs is None:
            analysis_status = "validated"

        elif analytical_status_counts.get(
            "available",
            0
        ) == len(
            candidate_registry
        ):
            analysis_status = "completed"

        else:
            analysis_status = (
                "completed_with_candidate_restrictions"
            )

        report = {
            "analysis_status": analysis_status,
            "operation": args.operation,
            "policy_name": policy[
                "policy_name"
            ],
            "policy_version": policy[
                "policy_version"
            ],
            "policy_sha256": calculate_file_sha256(
                args.predictive_policy_path
            ),
            "implementation_sha256": calculate_file_sha256(
                Path(
                    __file__
                )
            ),
            "input_hashes": {
                "feature_manifest": calculate_file_sha256(
                    args.feature_manifest_path
                ),
                "design_report": calculate_file_sha256(
                    args.design_report_path
                ),
                "identification_report": calculate_file_sha256(
                    args.identification_report_path
                ),
                "descriptive_report": calculate_file_sha256(
                    args.descriptive_report_path
                ),
                "bayesian_policy": calculate_file_sha256(
                    args.bayesian_policy_path
                ),
                "bayesian_report": calculate_file_sha256(
                    args.bayesian_report_path
                ),
                "bayesian_implementation": calculate_file_sha256(
                    Path(
                        bayesian_modeling.__file__
                    ).resolve()
                ),
                "dependency_lock": calculate_file_sha256(
                    PROJECT_ROOT
                    / policy[
                        "execution"
                    ][
                        "software_environment"
                    ][
                        "dependency_lock_path"
                    ]
                )
            },
            "software_environment": software_environment,
            "selected_candidate_count": len(
                selected_candidates
            ),
            "benchmark_candidate_count": len(
                candidate_registry
            ),
            "analytical_status_counts": dict(
                sorted(
                    analytical_status_counts.items()
                )
            ),
            "analytical_status_counts_by_model_family": (
                analytical_status_counts_by_model_family
            ),
            "feature_importance_status_counts": dict(
                sorted(
                    feature_importance_status_counts.items()
                )
            ),
            "analytical_handoff": {
                "status": analytical_handoff_status,
                "available_benchmark_count": (
                    analytical_status_counts.get(
                        "available",
                        0
                    )
                ),
                "diagnostic_only_benchmark_count": (
                    analytical_status_counts.get(
                        "diagnostic_only",
                        0
                    )
                ),
                "unavailable_benchmark_count": (
                    analytical_status_counts.get(
                        "not_available",
                        0
                    )
                ),
                "causal_interpretation_authorized": False,
                "policy_use_authorized": False
            },
            "candidate_selection_uses_posterior_results": False,
            "candidate_selection_uses_outcome_magnitude_direction_or_performance": False,
            "candidate_selection_inherits_outcome_validity_requirements": True,
            "upstream_candidate_count": len(
                workflow[
                    "candidate_records"
                ]
            ),
            "planned_benchmark_count": (
                len(
                    selected_candidates
                )
                * len(
                    policy[
                        "model_families"
                    ]
                )
            ),
            "execution_plan": build_execution_plan(
                selected_candidates,
                workflow,
                policy
            ),
            "performance_selected_winner_declared": False,
            "feature_importance_is_causal": False,
            "causal_interpretation_authorized": False,
            "policy_recommendation_authorized": False,
            "novel_condition_prediction_authorized": False,
            "independent_validation_unit": "cell_identity",
            "calendar_and_cycle_models_separated": True,
            "primary_and_sensitivity_cohorts_separated": True,
            "elapsed_time_and_cumulative_fec_separated": True,
            "dataset_stages_separated": True,
            "tables": table_manifests
        }

        report_path = (
            temporary_directory
            / policy[
                "output_contract"
            ][
                "report_file"
            ]
        )

        report_path.write_text(
            json.dumps(
                report,
                ensure_ascii = False,
                indent = 2,
                sort_keys = True
            )
            + "\n",
            encoding = "utf-8"
        )

        backup_directory = None

        if args.output_directory.exists():
            backup_directory = Path(
                tempfile.mkdtemp(
                    prefix = (
                        f".{args.output_directory.name}.backup."
                    ),
                    dir = output_parent
                )
            )

            backup_directory.rmdir()

            os.replace(
                args.output_directory,
                backup_directory
            )

        staged_report_path = (
            temporary_directory
            / policy[
                "output_contract"
            ][
                "report_file"
            ]
        )

        def validate_published_directory(published_directory: Path) -> None:
            validate_report(
                report_path = (
                    published_directory
                    / policy[
                        "output_contract"
                    ][
                        "report_file"
                    ]
                ),
                expected_policy = policy,
                expected_policy_sha256 = calculate_file_sha256(
                    args.predictive_policy_path
                ),
                expected_input_hashes = report[
                    "input_hashes"
                ],
                expected_operation = args.operation,
                expected_selected_candidate_count = (
                    report[
                        "selected_candidate_count"
                    ]
                ),
                expected_upstream_candidate_count = (
                    report[
                        "upstream_candidate_count"
                    ]
                ),
                expected_execution_plan = report[
                    "execution_plan"
                ],
                expected_benchmark_candidate_count = (
                    report[
                        "benchmark_candidate_count"
                    ]
                ),
                expected_analytical_status_counts = (
                    report[
                        "analytical_status_counts"
                    ]
                ),
                expected_analytical_status_counts_by_model_family = (
                    report[
                        "analytical_status_counts_by_model_family"
                    ]
                ),
                expected_feature_importance_status_counts = (
                    report[
                        "feature_importance_status_counts"
                    ]
                ),
                expected_analytical_handoff_status = (
                    report[
                        "analytical_handoff"
                    ][
                        "status"
                    ]
                ),
                expected_scope_fields = policy[
                    "output_contract"
                ][
                    "candidate_registry_scope_fields"
                ]
            )

        validate_report(
            report_path = staged_report_path,
            expected_policy = policy,
            expected_policy_sha256 = calculate_file_sha256(
                args.predictive_policy_path
            ),
            expected_input_hashes = report[
                "input_hashes"
            ],
            expected_operation = args.operation,
            expected_selected_candidate_count = (
                report[
                    "selected_candidate_count"
                ]
            ),
            expected_upstream_candidate_count = (
                report[
                    "upstream_candidate_count"
                ]
            ),
            expected_execution_plan = report[
                "execution_plan"
            ],
            expected_benchmark_candidate_count = (
                report[
                    "benchmark_candidate_count"
                ]
            ),
            expected_analytical_status_counts = (
                report[
                    "analytical_status_counts"
                ]
            ),
            expected_analytical_status_counts_by_model_family = (
                report[
                    "analytical_status_counts_by_model_family"
                ]
            ),
            expected_feature_importance_status_counts = (
                report[
                    "feature_importance_status_counts"
                ]
            ),
            expected_analytical_handoff_status = (
                report[
                    "analytical_handoff"
                ][
                    "status"
                ]
            ),
            expected_scope_fields = policy[
                "output_contract"
            ][
                "candidate_registry_scope_fields"
            ]
        )

        try:
            os.replace(
                temporary_directory,
                args.output_directory
            )

            validate_published_directory(
                args.output_directory
            )

        except Exception:
            if args.output_directory.exists():
                shutil.rmtree(
                    args.output_directory,
                    ignore_errors = True
                )

            if (
                backup_directory is not None
                and backup_directory.exists()
            ):
                os.replace(
                    backup_directory,
                    args.output_directory
                )

            raise

        if (
            backup_directory is not None
            and backup_directory.exists()
        ):
            shutil.rmtree(
                backup_directory
            )

    finally:
        if temporary_directory.exists():
            shutil.rmtree(
                temporary_directory
            )


# Running the requested predictive benchmarking operation
def main() -> None:
    args = parse_args()

    policy = load_json_object(
        args.predictive_policy_path
    )

    validate_policy(
        policy
    )

    software_environment = (
        validate_software_environment(
            policy
        )
    )

    (
        validated_inputs,
        workflow,
        bayesian_report
    ) = validate_upstream_inputs(
        args,
        policy
    )

    selected_candidates = build_selected_candidates(
        workflow,
        policy
    )

    if not selected_candidates:
        raise RuntimeError("No structurally admissible validation-feasible candidates were selected.")

    fit_outputs = None

    if args.operation == "fit":
        fit_outputs = fit_benchmarks(
            selected_candidates,
            workflow,
            policy
        )

        validate_fit_table_reconciliation(
            fit_outputs,
            policy
        )

    publish_artifacts(
        args,
        policy,
        workflow,
        selected_candidates,
        fit_outputs,
        software_environment
    )

    print(
        "Predictive benchmarking finished."
    )

    print(
        "Operation :",
        args.operation
    )

    print(
        "Selected candidate count :",
        len(
            selected_candidates
        )
    )

    if fit_outputs is not None:
        print(
            "Benchmark candidate count :",
            len(
                fit_outputs[
                    "candidate_registry"
                ]
            )
        )

        print(
            "Analytical status counts :",
            dict(
                sorted(
                    Counter(
                        record[
                            "analytical_status"
                        ]
                        for record in fit_outputs[
                            "candidate_registry"
                        ]
                    ).items()
                )
            )
        )

    print(
        "Report :",
        args.output_directory
        / policy[
            "output_contract"
        ][
            "report_file"
        ]
    )


if __name__ == "__main__":
    main()
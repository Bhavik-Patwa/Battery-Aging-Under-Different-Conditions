import subprocess
import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import traceback
import numpy as np

from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from importlib.metadata import version as distribution_version


BAYESIAN_RUNTIME = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ACCOUNTING_STATUSES = [
    "missing_horizon",
    "nonpositive_horizon",
    "missing_outcome",
    "cohort_ineligible",
    "incomplete_cumulative_fec",
    "included"
]

CANDIDATE_REGISTRY_FIELDS = [
    "candidate_identity",
    "support_region_identity",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "support_interval_minimum",
    "support_interval_maximum",
    "support_interval_lower_inclusive",
    "support_interval_upper_inclusive",
    "sorted_condition_protocol_identities",
    "sorted_estimand_decision_identities",
    "source_candidate_identities",
    "source_authorization_status",
    "condition_count",
    "independent_cell_count",
    "observation_count",
    "inside_evaluation_interval_count",
    "outside_evaluation_interval_count",
    "repeated_cell_count",
    "minimum_condition_repeated_cell_count",
    "fixed_location_matrix_rank",
    "fixed_location_matrix_column_count",
    "fixed_location_rank_tolerance",
    "admissibility_status",
    "admissibility_reason_code",
    "candidate_execution_status",
    "posterior_analytical_status",
    "posterior_analytical_reason_code",
    "posterior_qualification_failed_gates",
    "posterior_status_by_prior_multiplier",
    "posterior_reason_codes_by_prior_multiplier",
    "grouped_validation_status",
    "grouped_validation_reason_code",
    "effective_fold_count",
    "available_fold_count",
    "unavailable_fold_count"
]

OBSERVATION_ACCOUNTING_FIELDS = [
    "candidate_identity",
    "source_record_count",
    "included_record_count",
    "excluded_record_count",
    "missing_horizon_count",
    "nonpositive_horizon_count",
    "missing_outcome_count",
    "cohort_ineligible_count",
    "incomplete_cumulative_fec_count",
    "inside_evaluation_interval_count",
    "outside_evaluation_interval_count"
]

GROUPED_VALIDATION_FOLDS_FIELDS = [
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

PRIOR_PREDICTIVE_DIAGNOSTIC_FIELDS = [
    "candidate_identity",
    "outcome",
    "prior_scale_multiplier",
    "diagnostic_population",
    "condition_protocol_identity",
    "evaluation_point_count",
    "predictive_draw_count",
    "predictive_value_count",
    "execution_status",
    "execution_reason_code",
    "execution_error",
    "log_scale_median",
    "log_scale_median_absolute_deviation",
    "log_scale_interquartile_range",
    "log_scale_minimum",
    "log_scale_maximum",
    "log_scale_quantiles",
    "original_scale_finite_fraction",
    "original_scale_nonfinite_count",
    "original_scale_strictly_positive_finite_fraction",
    "original_scale_finite_quantiles"
]

POSTERIOR_SUMMARY_FIELDS = [
    "candidate_identity",
    "outcome",
    "cohort",
    "prior_scale_multiplier",
    "parameter_identity",
    "coordinate_identity",
    "posterior_draw_count",
    "posterior_median",
    "posterior_median_absolute_deviation",
    "posterior_quantiles",
    "summary_status",
    "summary_reason_code"
]

SAMPLER_DIAGNOSTIC_FIELDS = [
    "candidate_identity",
    "outcome",
    "prior_scale_multiplier",
    "fit_scope",
    "fold_index",
    "parameter_identity",
    "rank_normalized_split_rhat",
    "bulk_effective_sample_size",
    "tail_effective_sample_size",
    "rhat_passed",
    "bulk_ess_passed",
    "tail_ess_passed",
    "divergence_count",
    "minimum_bfmi",
    "tree_depth_hit_count",
    "diagnostic_status",
    "diagnostic_reason_code"
]

POSTERIOR_PREDICTIVE_DIAGNOSTIC_FIELDS = [
    "candidate_identity",
    "outcome",
    "cohort",
    "prior_scale_multiplier",
    "diagnostic_population",
    "condition_protocol_identity",
    "horizon_bin_index",
    "observation_count",
    "cell_count",
    "repeated_cell_count",
    "predictive_draw_count",
    "predictive_value_count",
    "predictive_nonfinite_count",
    "predictive_positive_finite_fraction",
    "observed_statistics",
    "posterior_predictive_statistic_medians",
    "posterior_predictive_statistic_quantiles",
    "within_cell_change_status",
    "within_cell_change_reason_code",
    "diagnostic_status",
    "diagnostic_reason_code"
]

GROUPED_VALIDATION_PREDICTION_FIELDS = [
    "candidate_identity",
    "fold_index",
    "cell_identity",
    "source_identity",
    "condition_protocol_identity",
    "horizon",
    "observed_outcome",
    "posterior_predictive_median",
    "posterior_predictive_interval_lower",
    "posterior_predictive_interval_upper",
    "prediction_status",
    "prediction_reason_code",
    "estimate_label"
]

GROUPED_VALIDATION_METRIC_FIELDS = [
    "candidate_identity",
    "outcome",
    "cohort",
    "fold_count",
    "test_cell_count",
    "test_observation_count",
    "equal_cell_weighted_mean_absolute_error",
    "equal_cell_weighted_root_mean_squared_error",
    "equal_cell_weighted_median_error",
    "posterior_predictive_interval_coverage",
    "mean_posterior_predictive_interval_width",
    "equal_cell_weighted_predictive_r_squared",
    "benchmark_variation_status",
    "validation_status",
    "validation_reason_code",
    "estimate_label"
]

GROUPED_VALIDATION_CONDITION_CALIBRATION_FIELDS = [
    "candidate_identity",
    "condition_protocol_identity",
    "independent_cell_count",
    "observation_count",
    "equal_cell_weighted_median_observed",
    "equal_cell_weighted_median_predicted",
    "equal_cell_weighted_median_error",
    "equal_cell_weighted_mean_absolute_error",
    "equal_cell_weighted_root_mean_squared_error",
    "posterior_predictive_interval_coverage",
    "mean_posterior_predictive_interval_width",
    "calibration_status",
    "calibration_reason_code",
    "estimate_label"
]

CONTINUOUS_TRAJECTORY_SUMMARY_FIELDS = [
    "candidate_identity",
    "outcome",
    "cohort",
    "horizon_axis",
    "condition_protocol_identity",
    "horizon",
    "trajectory_target",
    "posterior_mean",
    "posterior_median",
    "credible_interval_lower",
    "credible_interval_upper",
    "predictive_interval_lower",
    "predictive_interval_upper",
    "analytical_status",
    "analytical_reason_code",
    "estimate_label",
    "causal_interpretation"
]

CONTINUOUS_TRAJECTORY_DIFFERENCE_SUMMARY_FIELDS = [
    "candidate_identity",
    "estimand_decision_identity",
    "outcome",
    "cohort",
    "horizon_axis",
    "first_condition_protocol_identity",
    "second_condition_protocol_identity",
    "horizon",
    "posterior_mean_difference",
    "posterior_median_difference",
    "credible_interval_lower",
    "credible_interval_upper",
    "probability_difference_above_zero",
    "posterior_mean_ratio",
    "posterior_median_ratio",
    "ratio_credible_interval_lower",
    "ratio_credible_interval_upper",
    "probability_ratio_above_one",
    "interpretation_restriction_codes",
    "analytical_status",
    "analytical_reason_code",
    "estimate_label",
    "causal_interpretation"
]

PRIOR_SENSITIVITY_SUMMARY_FIELDS = [
    "candidate_identity",
    "outcome",
    "cohort",
    "parameter_or_trajectory_identity",
    "reference_prior_scale_multiplier",
    "sensitivity_prior_scale_multiplier",
    "reference_posterior_median",
    "sensitivity_posterior_median",
    "absolute_median_difference",
    "relative_median_difference",
    "sensitivity_comparison_status",
    "sensitivity_reason_code",
    "estimate_label"
]

OUTPUT_TABLE_FIELDS = {
    "candidate_registry": (
        CANDIDATE_REGISTRY_FIELDS
    ),
    "observation_accounting": (
        OBSERVATION_ACCOUNTING_FIELDS
    ),
    "grouped_validation_folds": (
        GROUPED_VALIDATION_FOLDS_FIELDS
    ),
    "prior_predictive_diagnostics": (
        PRIOR_PREDICTIVE_DIAGNOSTIC_FIELDS
    ),
    "posterior_summaries": (
        POSTERIOR_SUMMARY_FIELDS
    ),
    "sampler_diagnostics": (
        SAMPLER_DIAGNOSTIC_FIELDS
    ),
    "posterior_predictive_diagnostics": (
        POSTERIOR_PREDICTIVE_DIAGNOSTIC_FIELDS
    ),
    "grouped_validation_predictions": (
        GROUPED_VALIDATION_PREDICTION_FIELDS
    ),
    "grouped_validation_metrics": (
        GROUPED_VALIDATION_METRIC_FIELDS
    ),
    "grouped_validation_condition_calibration": (
        GROUPED_VALIDATION_CONDITION_CALIBRATION_FIELDS
    ),
    "continuous_trajectory_summaries": (
        CONTINUOUS_TRAJECTORY_SUMMARY_FIELDS
    ),
    "continuous_trajectory_difference_summaries": (
        CONTINUOUS_TRAJECTORY_DIFFERENCE_SUMMARY_FIELDS
    ),
    "prior_sensitivity_summaries": (
        PRIOR_SENSITIVITY_SUMMARY_FIELDS
    )
}

REQUIRED_POLICY_SECTIONS = {
    "policy_version", "effective_date", "previous_policy_sha256",
    "registration", "authorization", "inputs", "candidate_inventory",
    "observation_contract", "reason_code_contract", "outcomes",
    "horizon_encoding", "model", "priors", "prior_predictive_review",
    "generative_prediction_contract", "admissibility", "sampling",
    "diagnostics", "predictive_assessment", "trajectory_reporting",
    "model_comparison", "reporting", "execution_planning",
    "cache_contract", "cache_import_contract", "reproducibility", "stochastic_process_environment",
    "posterior_artifact_contract", "execution_contract",
    "output_contract", "amendment"
}

REQUIRED_AUTHORIZATION_PROHIBITIONS = {
    "causal_language_permitted",
    "causal_effect_parameters_permitted",
    "policy_recommendations_permitted",
    "causal_sensitivity_analysis_permitted",
    "cross_stage_pooling_permitted",
    "primary_and_sensitivity_cohorts_pooled",
    "elapsed_time_and_fec_axes_pooled",
    "calendar_and_cycle_aging_pooled",
    "unobserved_condition_combinations_permitted"
}

REQUIRED_OPERATIONS = {
    "validate",
    "prior_predictive",
    "populate_fit_cache",
    "fit"
}

POLICY_VERSION_PATTERN = re.compile(r"^[1-9]\d*\.\d+$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

IMPLEMENTED_CANDIDATE_IDENTITY_FIELDS = [
    "support_region_identity",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "sorted_condition_protocol_identities",
    "sorted_estimand_decision_identities"
]

IMPLEMENTED_CANDIDATE_CONSTRUCTION_SEQUENCE = [
    "select_source_candidates_using_registered_outcome_blind_fields",
    "reconstruct_and_reconcile_each_source_candidate_observation_population",
    "group_source_candidates_by_registered_deduplication_fields",
    "validate_duplicate_source_required_fields_and_included_observation_identities",
    "collapse_valid_duplicate_sources_while_retaining_all_source_candidate_identities",
    "resolve_the_most_restrictive_source_authorization",
    "join_authorized_estimand_edges",
    "construct_the_authorized_condition_graph",
    "split_disconnected_graphs_into_deterministically_ordered_connected_components",
    "calculate_candidate_identity_from_registered_identity_fields",
    "assemble_unique_component_observations",
    "apply_registered_admissibility_checks"
]

IMPLEMENTED_HORIZON_ENCODING = {
    "transformation": (
        "assembled_training_range_midpoint_centered_and_width_scaled"
    ),
    "formula": (
        "(horizon - training_horizon_midpoint) "
        "/ training_horizon_width"
    ),
    "training_horizon_minimum": (
        "minimum_horizon_across_assembled_model_observations"
    ),
    "training_horizon_maximum": (
        "maximum_horizon_across_assembled_model_observations"
    ),
    "training_horizon_midpoint": (
        "(training_horizon_minimum "
        "+ training_horizon_maximum) / 2"
    ),
    "training_horizon_width": (
        "training_horizon_maximum "
        "- training_horizon_minimum"
    ),
    "positive_training_horizon_width_required": True,
    "evaluation_support_remains_reporting_domain": True,
    "training_observations_outside_evaluation_interval_retain_transformed_values": True
}

IMPLEMENTED_FOLD_ASSIGNMENT_METHOD = (
    "sort_cells_within_condition_by_sha256_of_canonical_json_candidate_identity_condition_identity_cell_identity_and_seed_"
    "then_assign_round_robin"
)

IMPLEMENTED_PRIOR_SENSITIVITY_SCALE_MULTIPLIERS = [
    0.5,
    1.0,
    2.0
]

IMPLEMENTED_PRIOR_PREDICTIVE_NEW_CELL_SEED_RULE = (
    "registered_prior_predictive_seed_plus_zero_based_condition_index_plus_one"
)

IMPLEMENTED_TRAJECTORY_NEW_CELL_SEED_RULE = (
    "sha256_of_candidate_identity_condition_protocol_identity_and_registered_seed_reduced_to_unsigned_32_bit_integer"
)


# Requiring one object to contain exactly the declared fields
def _require_exact_fields(value, required_fields: set, description: str) -> dict:
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not an object.")

    missing = sorted(set(required_fields).difference(value))
    unexpected = sorted(set(value).difference(required_fields))

    if missing or unexpected:
        raise RuntimeError(f"{description} fields differ from the contract : missing {missing} : unexpected {unexpected}")

    return value


# Confirming a policy value is a strictly positive integer
def _require_positive_integer(value, field_name: str) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise RuntimeError(f"Bayesian policy requires a positive integer for {field_name}.")


# Confirming a policy value is a strictly positive finite number
def _require_positive_number(value, field_name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise RuntimeError(f"Bayesian policy requires a positive finite number for {field_name}.")


# Confirming a policy value is exactly True
def _require_true(value, field_name: str) -> None:
    if value is not True:
        raise RuntimeError(f"Bayesian policy requires {field_name} to be True.")


# Confirming a policy value is exactly False
def _require_false(value, field_name: str) -> None:
    if value is not False:
        raise RuntimeError(f"Bayesian policy requires {field_name} to be False.")


# Confirming a policy value is one of the implementation-supported values
def _require_supported_value(value, supported_values: set, field_name: str) -> None:
    if value not in supported_values:
        raise RuntimeError(f"Bayesian policy requires {field_name} to be one of {sorted(supported_values)}.")


# Confirming a policy section's nested keys exist and no extras are present
def _require_supported_sections(value, supported_section_names: set, field_name: str) -> None:
    if not isinstance(value, dict):
        raise RuntimeError(f"Bayesian policy section is not an object : {field_name}")

    missing = sorted(supported_section_names.difference(value))
    if missing:
        raise RuntimeError(f"Bayesian policy section is missing required entries : {field_name} : {missing}")


# Requiring an implementation-driving declaration to match exactly
def _require_exact_value(observed_value, expected_value, field_name: str) -> None:
    if observed_value != expected_value:
        raise RuntimeError(f"Bayesian policy differs from the implemented contract : {field_name}")


# Confirming a value is a finite number
def _require_finite_number(value, field_name: str) -> None:
    if (
        not isinstance(
            value,
            (int, float)
        )
        or isinstance(
            value,
            bool
        )
        or not math.isfinite(
            float(
                value
            )
        )
    ):
        raise RuntimeError(f"Bayesian policy requires a finite number for {field_name}.")


# Confirming a value is a nonnegative integer
def _require_nonnegative_integer(value, field_name: str) -> None:
    if (
        not isinstance(
            value,
            int
        )
        or isinstance(
            value,
            bool
        )
        or value < 0
    ):
        raise RuntimeError(f"Bayesian policy requires a nonnegative integer for {field_name}.")


# Confirming a value is a probability strictly between zero and one
def _require_open_probability(value, field_name: str) -> None:
    _require_finite_number(
        value,
        field_name
    )

    if not (
        0.0
        < float(
            value
        )
        < 1.0
    ):
        raise RuntimeError(f"Bayesian policy requires {field_name} to be strictly between zero and one.")


# Confirming the process was launched with the registered stochastic environment
def validate_stochastic_process_environment(policy: dict) -> None:
    stochastic_environment = policy["stochastic_process_environment"]
    required_variables = stochastic_environment["required_environment_variables"]

    observed_mismatches = {}

    for variable_name in sorted(required_variables):
        required_value = required_variables[variable_name]
        observed_value = os.environ.get(variable_name)

        if observed_value != required_value:
            observed_mismatches[variable_name] = {
                "required": required_value,
                "observed": observed_value,
            }

    if required_variables.get("PYTHONHASHSEED") == "0":
        if sys.flags.hash_randomization != 0:
            observed_mismatches["sys.flags.hash_randomization"] = {
                "required": 0,
                "observed": sys.flags.hash_randomization,
            }

    if observed_mismatches:
        launch_hint = " ".join(
            f"{variable_name}={variable_value}"
            for variable_name, variable_value in required_variables.items()
        )

        raise RuntimeError(
            "Stochastic process environment does not match the registered policy. "
            f"Mismatches: {observed_mismatches}. "
            f"Relaunch the process with : {launch_hint} python -m scripts.run_bayesian_models ..."
        )


# Building the recorded software-environment identity
def build_software_environment_record(policy: dict) -> dict:
    import pytensor as pytensor_runtime

    required_variables = policy["stochastic_process_environment"]["required_environment_variables"]

    environment_values = {
        variable_name: os.environ.get(variable_name)
        for variable_name in sorted(required_variables)
    }

    config_stream = io.StringIO()

    with contextlib.redirect_stdout(config_stream):
        np.show_config()

    numpy_configuration_sha256 = hashlib.sha256(
        config_stream.getvalue().encode("utf-8")
    ).hexdigest()

    def _safe_version(distribution_name: str) -> str | None:
        try:
            return distribution_version(distribution_name)

        except Exception:
            return None

    pytensor_cxx = None
    cxx_compiler_identity = None

    try:
        pytensor_cxx = (
            pytensor_runtime.config.cxx
            or None
        )

    except Exception:
        pytensor_cxx = None

    if pytensor_cxx:
        try:
            compiler_result = subprocess.run(
                [
                    pytensor_cxx,
                    "--version"
                ],
                check = True,
                capture_output = True,
                text = True
            )

            compiler_text = (
                compiler_result.stdout.strip()
                or compiler_result.stderr.strip()
            )

            cxx_compiler_identity = (
                compiler_text.splitlines()[0]
                if compiler_text
                else None
            )

        except Exception:
            cxx_compiler_identity = None

    return {
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(str(component) for component in sys.version_info[:3]),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "numpy_version": _safe_version("numpy"),
        "scipy_version": _safe_version("scipy"),
        "pymc_version": _safe_version("pymc"),
        "pytensor_version": _safe_version("pytensor"),
        "arviz_version": _safe_version("arviz"),
        "xarray_version": _safe_version("xarray"),
        "numpy_configuration_sha256": numpy_configuration_sha256,
        "pytensor_cxx": pytensor_cxx,
        "cxx_compiler_identity": cxx_compiler_identity,
        "environment_values": environment_values,
    }


# Requiring compiler provenance before any PyMC compile or NUTS run
def require_compiler_environment(environment_record: dict) -> None:
    pytensor_cxx = environment_record.get(
        "pytensor_cxx"
    )

    cxx_compiler_identity = environment_record.get(
        "cxx_compiler_identity"
    )

    if (
        not isinstance(
            pytensor_cxx,
            str
        )
        or not pytensor_cxx
    ):
        raise RuntimeError(
            "Stochastic execution requires a configured PyTensor C compiler. "
            "Install g++ in the execution environment and verify pytensor.config.cxx."
        )

    if (
        not isinstance(
            cxx_compiler_identity,
            str
        )
        or not cxx_compiler_identity
    ):
        raise RuntimeError(
            "Stochastic execution requires a resolvable C++ compiler identity. "
            "Verify that the configured PyTensor compiler responds to --version."
        )


# Loading stochastic dependencies only after environment validation
def load_bayesian_runtime() -> tuple:
    global BAYESIAN_RUNTIME

    if BAYESIAN_RUNTIME is None:
        import arviz
        import pymc
        import pytensor.tensor
        import xarray

        BAYESIAN_RUNTIME = (
            pymc,
            pytensor.tensor,
            arviz,
            xarray
        )

    return BAYESIAN_RUNTIME


# Preparing deterministic candidate arrays for PyMC
def prepare_pymc_candidate_data(candidate: dict, observations: list[dict] | None = None) -> dict:
    if observations is None:
        observations = candidate[
            "included_observations"
        ]

    observations = sorted(
        observations,
        key = lambda record: (
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
    )

    if not observations:
        raise RuntimeError("Cannot construct a Bayesian model without observations.")

    condition_identities = sorted({
        observation[
            "condition_protocol_identity"
        ]
        for observation in observations
    })

    cell_identities = sorted({
        observation[
            "cell_identity"
        ]
        for observation in observations
    })

    condition_index_by_identity = {
        condition_identity: condition_index
        for condition_index, condition_identity
        in enumerate(
            condition_identities
        )
    }

    cell_index_by_identity = {
        cell_identity: cell_index
        for cell_index, cell_identity
        in enumerate(
            cell_identities
        )
    }

    condition_by_cell = {}

    for observation in observations:
        cell_identity = observation[
            "cell_identity"
        ]

        condition_identity = observation[
            "condition_protocol_identity"
        ]

        prior_condition = (
            condition_by_cell.setdefault(
                cell_identity,
                condition_identity
            )
        )

        if prior_condition != condition_identity:
            raise RuntimeError("A physical cell belongs to more than one modeled condition.")

    outcomes = np.asarray(
        [
            float(
                observation[
                    "outcome_value"
                ]
            )
            for observation in observations
        ],
        dtype = float
    )

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

    if (
        not np.all(
            np.isfinite(
                outcomes
            )
        )
        or np.any(
            outcomes <= 0
        )
    ):
        raise RuntimeError("Bayesian outcomes must be finite and strictly positive.")

    if not np.all(
        np.isfinite(
            horizons
        )
    ):
        raise RuntimeError("Bayesian horizons must be finite.")

    training_horizon_minimum = float(
        np.min(
            horizons
        )
    )

    training_horizon_maximum = float(
        np.max(
            horizons
        )
    )

    training_horizon_width = (
        training_horizon_maximum
        - training_horizon_minimum
    )

    if (
        not math.isfinite(
            training_horizon_width
        )
        or training_horizon_width <= 0
    ):
        raise RuntimeError("Bayesian model requires a positive finite training-horizon width.")

    training_horizon_midpoint = (
        training_horizon_minimum
        + training_horizon_maximum
    ) / 2

    scaled_horizons = (
        horizons
        - training_horizon_midpoint
    ) / training_horizon_width

    scaled_horizon_range_tolerance = 1e-6

    if not np.all(
        np.isfinite(
            scaled_horizons
        )
    ):
        raise RuntimeError("Scaled training horizons are non-finite.")

    if (
        np.min(
            scaled_horizons
        )
        < -0.5
        - scaled_horizon_range_tolerance
        or np.max(
            scaled_horizons
        )
        > 0.5
        + scaled_horizon_range_tolerance
    ):
        raise RuntimeError(
            "Scaled training horizons are outside the expected interval : "
            f"minimum {float(np.min(scaled_horizons))} : "
            f"maximum {float(np.max(scaled_horizons))} : "
            f"training_horizon_minimum {training_horizon_minimum} : "
            f"training_horizon_maximum {training_horizon_maximum}"
        )

    support_minimum = float(
        candidate[
            "collapsed_candidate"
        ][
            "support_interval_minimum"
        ]
    )

    support_maximum = float(
        candidate[
            "collapsed_candidate"
        ][
            "support_interval_maximum"
        ]
    )

    support_width = (
        support_maximum
        - support_minimum
    )

    if (
        not math.isfinite(
            support_width
        )
        or support_width <= 0
    ):
        raise RuntimeError("Bayesian reporting requires a positive finite support width.")

    support_midpoint = (
        support_minimum
        + support_maximum
    ) / 2

    return {
        "observations": observations,
        "condition_identities": (
            condition_identities
        ),
        "cell_identities": cell_identities,
        "condition_indices": np.asarray(
            [
                condition_index_by_identity[
                    observation[
                        "condition_protocol_identity"
                    ]
                ]
                for observation in observations
            ],
            dtype = int
        ),
        "cell_indices": np.asarray(
            [
                cell_index_by_identity[
                    observation[
                        "cell_identity"
                    ]
                ]
                for observation in observations
            ],
            dtype = int
        ),
        "outcomes": outcomes,
        "log_outcomes": np.log(
            outcomes
        ),
        "horizons": horizons,
        "scaled_horizons": scaled_horizons,
        "training_horizon_minimum": (
            training_horizon_minimum
        ),
        "training_horizon_maximum": (
            training_horizon_maximum
        ),
        "training_horizon_midpoint": (
            training_horizon_midpoint
        ),
        "training_horizon_width": (
            training_horizon_width
        ),
        "support_minimum": support_minimum,
        "support_maximum": support_maximum,
        "support_midpoint": support_midpoint,
        "support_width": support_width
    }


# Reading one registered prior scale
def scaled_prior_value(prior_definition: dict, prior_scale_multiplier: float) -> float:
    scale_value = (
        prior_definition.get("standard_deviation")
        if "standard_deviation" in prior_definition
        else prior_definition.get("scale")
    )

    if scale_value is None:
        raise RuntimeError("Prior definition does not contain a scale.")

    scaled_value = float(scale_value) * prior_scale_multiplier

    if (
        not math.isfinite(scaled_value)
        or scaled_value <= 0
    ):
        raise RuntimeError("Scaled prior value must be positive and finite.")

    return scaled_value


# Constructing the registered hierarchical trajectory model
def build_pymc_model(candidate: dict, policy: dict,
                     prior_scale_multiplier: float, observations: list[dict] | None = None
) -> tuple[object, dict]:
    if prior_scale_multiplier not in policy["priors"]["posterior_fit_scale_multipliers"]:
        raise RuntimeError("Prior multiplier is not registered.")

    data = prepare_pymc_candidate_data(
        candidate = candidate,
        observations = observations
    )

    pm, pt, _, _ = load_bayesian_runtime()

    priors = policy["priors"]

    coordinates = {
        "condition_protocol_identity": data["condition_identities"],
        "cell_identity": data["cell_identities"],
        "observation_index": list(range(len(data["observations"]))),
        "random_effect_coefficient": ["intercept", "horizon_slope"],
        "random_effect_coefficient_aux": ["intercept", "horizon_slope"]
    }

    with pm.Model(coords = coordinates) as model:
        condition_index = pm.Data(
            "condition_index",
            data["condition_indices"],
            dims = "observation_index"
        )

        cell_index = pm.Data(
            "cell_index",
            data["cell_indices"],
            dims = "observation_index"
        )

        scaled_horizon = pm.Data(
            "scaled_horizon",
            data["scaled_horizons"],
            dims = "observation_index"
        )

        population_intercept = pm.Normal(
            "population_intercept",
            mu = priors["population_intercept"]["mean"],
            sigma = scaled_prior_value(
                priors["population_intercept"],
                prior_scale_multiplier
            )
        )

        population_horizon_slope = pm.Normal(
            "population_horizon_slope",
            mu = priors["population_horizon_slope"]["mean"],
            sigma = scaled_prior_value(
                priors["population_horizon_slope"],
                prior_scale_multiplier
            )
        )

        condition_intercept_standard_deviation = pm.HalfNormal(
            "condition_intercept_standard_deviation",
            sigma = scaled_prior_value(
                priors["condition_intercept_standard_deviation"],
                prior_scale_multiplier
            )
        )

        condition_horizon_slope_standard_deviation = pm.HalfNormal(
            "condition_horizon_slope_standard_deviation",
            sigma = scaled_prior_value(
                priors["condition_horizon_slope_standard_deviation"],
                prior_scale_multiplier
            )
        )

        condition_intercept_noncentered = pm.Normal(
            "condition_intercept_noncentered",
            mu = 0,
            sigma = 1,
            dims = "condition_protocol_identity"
        )

        condition_horizon_slope_noncentered = pm.Normal(
            "condition_horizon_slope_noncentered",
            mu = 0,
            sigma = 1,
            dims = "condition_protocol_identity"
        )

        condition_intercept_deviation = pm.Deterministic(
            "condition_intercept_deviation",
            condition_intercept_noncentered * condition_intercept_standard_deviation,
            dims = "condition_protocol_identity"
        )

        condition_horizon_slope_deviation = pm.Deterministic(
            "condition_horizon_slope_deviation",
            condition_horizon_slope_noncentered * condition_horizon_slope_standard_deviation,
            dims = "condition_protocol_identity"
        )

        condition_intercept = pm.Deterministic(
            "condition_intercept",
            population_intercept + condition_intercept_deviation,
            dims = "condition_protocol_identity"
        )

        condition_horizon_slope = pm.Deterministic(
            "condition_horizon_slope",
            population_horizon_slope + condition_horizon_slope_deviation,
            dims = "condition_protocol_identity"
        )

        cell_scale_distribution = pm.HalfNormal.dist(
            sigma = np.asarray(
                [
                    scaled_prior_value(
                        priors["cell_random_intercept_standard_deviation"],
                        prior_scale_multiplier
                    ),
                    scaled_prior_value(
                        priors["cell_random_horizon_slope_standard_deviation"],
                        prior_scale_multiplier
                    )
                ],
                dtype = float
            ),
            shape = 2
        )

        (
            cell_random_effect_cholesky,
            cell_random_effect_correlation_matrix,
            cell_random_effect_standard_deviations
        ) = pm.LKJCholeskyCov(
            "cell_random_effect_cholesky_packed",
            n = 2,
            eta = priors["cell_random_effect_correlation"]["eta"],
            sd_dist = cell_scale_distribution,
            compute_corr = True,
            store_in_trace = True
        )

        pm.Deterministic(
            "cell_random_effect_cholesky",
            cell_random_effect_cholesky,
            dims = (
                "random_effect_coefficient",
                "random_effect_coefficient_aux"
            )
        )

        pm.Deterministic(
            "cell_random_effect_standard_deviations",
            cell_random_effect_standard_deviations,
            dims = "random_effect_coefficient"
        )

        pm.Deterministic(
            "cell_intercept_slope_correlation",
            cell_random_effect_correlation_matrix[0, 1]
        )

        cell_random_effect_noncentered = pm.Normal(
            "cell_random_effect_noncentered",
            mu = 0,
            sigma = 1,
            dims = (
                "cell_identity",
                "random_effect_coefficient"
            )
        )

        cell_random_effects = pt.dot(
            cell_random_effect_noncentered,
            cell_random_effect_cholesky.T
        )

        cell_intercept = pm.Deterministic(
            "cell_intercept",
            cell_random_effects[:, 0],
            dims = "cell_identity"
        )

        cell_horizon_slope = pm.Deterministic(
            "cell_horizon_slope",
            cell_random_effects[:, 1],
            dims = "cell_identity"
        )

        residual_standard_deviation = pm.HalfNormal(
            "residual_standard_deviation",
            sigma = scaled_prior_value(
                priors["residual_standard_deviation"],
                prior_scale_multiplier
            )
        )

        student_t_degrees_of_freedom_minus_two = pm.Exponential(
            "student_t_degrees_of_freedom_minus_two",
            lam = 1 / priors["student_t_degrees_of_freedom"]["exponential_mean"]
        )

        student_t_degrees_of_freedom = pm.Deterministic(
            "student_t_degrees_of_freedom",
            2 + student_t_degrees_of_freedom_minus_two
        )

        observation_location = pm.Deterministic(
            "observation_location",
            (
                condition_intercept[condition_index]
                + condition_horizon_slope[condition_index] * scaled_horizon
                + cell_intercept[cell_index]
                + cell_horizon_slope[cell_index] * scaled_horizon
            ),
            dims = "observation_index"
        )

        pm.StudentT(
            "log_outcome_observation",
            nu = student_t_degrees_of_freedom,
            mu = observation_location,
            sigma = residual_standard_deviation,
            observed = data["log_outcomes"],
            dims = "observation_index"
        )

    return model, data


# Converting a registered prior multiplier to its stable JSON key
def prior_multiplier_key(prior_scale_multiplier: float) -> str:
    return format(float(prior_scale_multiplier), ".1f")


# Deriving a stable unsigned 32-bit seed from registered prediction context
def deterministic_prediction_seed(candidate_identity: str, condition_identity: str,
                                  registered_seed: int
) -> int:
    seed_payload = {
        "candidate_identity": candidate_identity,
        "condition_protocol_identity": condition_identity,
        "registered_seed": registered_seed
    }

    seed_digest = hashlib.sha256(
        json.dumps(
            seed_payload,
            sort_keys = True,
            separators = (",", ":")
        ).encode(
            "utf-8"
        )
    ).digest()

    return int.from_bytes(
        seed_digest[:4],
        byteorder = "big",
        signed = False
    )


# Deriving original-scale values without silently discarding overflow
def exponentiate_with_nonfinite_accounting(log_values: np.ndarray) -> tuple[np.ndarray, dict]:
    log_values = np.asarray(
        log_values,
        dtype = float
    )

    with np.errstate(
        over = "ignore",
        under = "ignore",
        invalid = "ignore"
    ):
        original_values = np.exp(
            log_values
        )

    finite_mask = np.isfinite(
        original_values
    )

    strictly_positive_finite_mask = (
        finite_mask
        & (original_values > 0)
    )

    return (
        original_values,
        {
            "value_count": int(
                original_values.size
            ),
            "finite_count": int(
                finite_mask.sum()
            ),
            "nonfinite_count": int(
                (~finite_mask).sum()
            ),
            "finite_fraction": (
                float(finite_mask.mean())
                if original_values.size
                else None
            ),
            "strictly_positive_finite_fraction": (
                float(
                    strictly_positive_finite_mask.mean()
                )
                if original_values.size
                else None
            )
        }
    )


# Calculating registered finite quantiles
def finite_quantiles(values: np.ndarray) -> dict:
    finite_values = np.asarray(
        values,
        dtype = float
    )

    finite_values = finite_values[
        np.isfinite(
            finite_values
        )
    ]

    probabilities = [
        0.01,
        0.05,
        0.25,
        0.50,
        0.75,
        0.95,
        0.99
    ]

    if finite_values.size == 0:
        return {
            format(probability, ".2f"): None
            for probability in probabilities
        }

    quantiles = np.quantile(
        finite_values,
        probabilities,
        method = "linear"
    )

    return {
        format(probability, ".2f"): float(
            quantile
        )
        for probability, quantile in zip(
            probabilities,
            quantiles,
            strict = True
        )
    }


# Summarizing one prior-predictive distribution
def summarize_prior_predictive_values(log_values: np.ndarray) -> dict:
    flattened_log_values = np.asarray(
        log_values,
        dtype = float
    ).reshape(-1)

    if (
        flattened_log_values.size == 0
        or not np.all(
            np.isfinite(
                flattened_log_values
            )
        )
    ):
        raise RuntimeError("Prior-predictive log-scale values must be finite and nonempty.")

    log_median = float(
        np.median(
            flattened_log_values
        )
    )

    log_median_absolute_deviation = float(
        np.median(
            np.abs(
                flattened_log_values
                - log_median
            )
        )
    )

    first_quartile, third_quartile = np.quantile(
        flattened_log_values,
        [
            0.25,
            0.75
        ],
        method = "linear"
    )

    (
        original_values,
        original_accounting
    ) = exponentiate_with_nonfinite_accounting(
        flattened_log_values
    )

    return {
        "log_scale_median": log_median,
        "log_scale_median_absolute_deviation": (
            log_median_absolute_deviation
        ),
        "log_scale_interquartile_range": float(
            third_quartile
            - first_quartile
        ),
        "log_scale_minimum": float(
            np.min(
                flattened_log_values
            )
        ),
        "log_scale_maximum": float(
            np.max(
                flattened_log_values
            )
        ),
        "log_scale_quantiles": finite_quantiles(
            flattened_log_values
        ),
        "original_scale_finite_fraction": (
            original_accounting[
                "finite_fraction"
            ]
        ),
        "original_scale_nonfinite_count": (
            original_accounting[
                "nonfinite_count"
            ]
        ),
        "original_scale_strictly_positive_finite_fraction": (
            original_accounting[
                "strictly_positive_finite_fraction"
            ]
        ),
        "original_scale_finite_quantiles": (
            finite_quantiles(
                original_values
            )
        )
    }


# Constructing the registered boundary-aware trajectory grid
def build_trajectory_grid(candidate: dict, policy: dict) -> np.ndarray:
    reporting = policy[
        "trajectory_reporting"
    ]

    grid_size = reporting[
        "trajectory_grid_size"
    ]

    lower = float(
        candidate[
            "collapsed_candidate"
        ][
            "support_interval_minimum"
        ]
    )

    upper = float(
        candidate[
            "collapsed_candidate"
        ][
            "support_interval_maximum"
        ]
    )

    lower_inclusive = candidate[
        "collapsed_candidate"
    ][
        "support_interval_lower_inclusive"
    ]

    upper_inclusive = candidate[
        "collapsed_candidate"
    ][
        "support_interval_upper_inclusive"
    ]

    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or upper <= lower
    ):
        raise RuntimeError("Trajectory grid requires a positive finite interval.")

    lower_offset = (
        0
        if lower_inclusive
        else 1
    )

    upper_offset = (
        0
        if upper_inclusive
        else 1
    )

    denominator = (
        grid_size
        - 1
        + lower_offset
        + upper_offset
    )

    grid = np.asarray(
        [
            lower
            + (
                grid_index
                + lower_offset
            )
            / denominator
            * (
                upper
                - lower
            )
            for grid_index in range(
                grid_size
            )
        ],
        dtype = float
    )

    if (
        not np.all(
            np.isfinite(
                grid
            )
        )
        or np.any(
            np.diff(
                grid
            )
            <= 0
        )
    ):
        raise RuntimeError("Trajectory grid is not finite and strictly increasing.")

    if (
        not lower_inclusive
        and grid[0] <= lower
    ):
        raise RuntimeError("Trajectory grid evaluates an exclusive lower boundary.")

    if (
        not upper_inclusive
        and grid[-1] >= upper
    ):
        raise RuntimeError("Trajectory grid evaluates an exclusive upper boundary.")

    return grid


# Building the new-cell prior-predictive grid for one condition
def build_prior_new_cell_predictive_grid(inference_data, condition_index: int,
                                         scaled_grid: np.ndarray, seed: int
) -> np.ndarray:
    condition_intercept = np.asarray(
        inference_data.prior[
            "condition_intercept"
        ].values[
            ...,
            condition_index
        ],
        dtype = float
    )

    condition_slope = np.asarray(
        inference_data.prior[
            "condition_horizon_slope"
        ].values[
            ...,
            condition_index
        ],
        dtype = float
    )

    cholesky = np.asarray(
        inference_data.prior[
            "cell_random_effect_cholesky"
        ].values,
        dtype = float
    )

    residual_scale = np.asarray(
        inference_data.prior[
            "residual_standard_deviation"
        ].values,
        dtype = float
    )

    degrees_of_freedom = np.asarray(
        inference_data.prior[
            "student_t_degrees_of_freedom"
        ].values,
        dtype = float
    )

    if (
        condition_intercept.shape
        != condition_slope.shape
        or condition_intercept.shape
        != residual_scale.shape
        or condition_intercept.shape
        != degrees_of_freedom.shape
        or cholesky.shape[
            :2
        ] != condition_intercept.shape
        or cholesky.shape[
            -2:
        ] != (
            2,
            2
        )
    ):
        raise RuntimeError("Prior arrays cannot be aligned for new-cell prediction.")

    random_generator = np.random.default_rng(
        seed
    )

    standard_effects = random_generator.normal(
        size = (
            *condition_intercept.shape,
            2
        )
    )

    new_cell_effects = np.einsum(
        "cdj,cdij->cdi",
        standard_effects,
        cholesky
    )

    location = (
        condition_intercept[
            ...,
            None
        ]
        + new_cell_effects[
            ...,
            0,
            None
        ]
        + (
            condition_slope[
                ...,
                None
            ]
            + new_cell_effects[
                ...,
                1,
                None
            ]
        )
        * scaled_grid[
            None,
            None,
            :
        ]
    )

    observation_noise = (
        random_generator.standard_t(
            df = degrees_of_freedom[
                ...,
                None
            ],
            size = location.shape
        )
        * residual_scale[
            ...,
            None
        ]
    )

    result = (
        location
        + observation_noise
    )

    if not np.all(
        np.isfinite(
            result
        )
    ):
        raise RuntimeError("New-cell prior-predictive grid contains non-finite log-scale draws.")

    return result


# Building one prior-predictive diagnostic record
def build_prior_predictive_diagnostic_record(candidate: dict, prior_scale_multiplier: float,
                                             diagnostic_population: str, log_values: np.ndarray,
                                             condition_protocol_identity: str | None = None
) -> dict:
    predictive_values = np.asarray(
        log_values,
        dtype = float
    )

    if predictive_values.ndim < 2:
        raise RuntimeError("Prior-predictive values must contain draw and evaluation-point dimensions.")

    evaluation_point_count = int(
        predictive_values.shape[-1]
    )

    predictive_value_count = int(
        predictive_values.size
    )

    if evaluation_point_count <= 0:
        raise RuntimeError("Prior-predictive diagnostics require at least one evaluation point.")

    predictive_draw_count = (
        predictive_value_count
        // evaluation_point_count
    )

    diagnostics = summarize_prior_predictive_values(
        predictive_values
    )

    reason_code = (
        "PRIOR_PREDICTIVE_NONFINITE_DRAWS"
        if diagnostics[
            "original_scale_nonfinite_count"
        ] > 0
        else "PRIOR_PREDICTIVE_COMPLETED"
    )

    return {
        "candidate_identity": candidate[
            "candidate_identity"
        ],
        "outcome": candidate[
            "collapsed_candidate"
        ][
            "outcome"
        ],
        "prior_scale_multiplier": (
            prior_scale_multiplier
        ),
        "diagnostic_population": (
            diagnostic_population
        ),
        "condition_protocol_identity": (
            condition_protocol_identity
        ),
        "evaluation_point_count": (
            evaluation_point_count
        ),
        "predictive_draw_count": (
            predictive_draw_count
        ),
        "predictive_value_count": (
            predictive_value_count
        ),
        "execution_status": "completed",
        "execution_reason_code": reason_code,
        "execution_error": None,
        **diagnostics
    }


# Sampling one registered prior-predictive candidate
def sample_prior_predictive_candidate(candidate: dict, policy: dict,
                                      prior_scale_multiplier: float
) -> tuple[object | None, list[dict]]:
    multiplier_key = prior_multiplier_key(
        prior_scale_multiplier
    )

    seed = policy[
        "priors"
    ][
        "prior_predictive_seeds"
    ][
        multiplier_key
    ]

    try:
        model, data = build_pymc_model(
            candidate = candidate,
            policy = policy,
            prior_scale_multiplier = (
                prior_scale_multiplier
            )
        )

        pm, _, _, _ = load_bayesian_runtime()

        contract = policy[
            "posterior_artifact_contract"
        ]

        likelihood_variable_name = contract[
            "likelihood_variable_name"
        ]

        required_prior_variables = [
            likelihood_variable_name,
            "condition_intercept",
            "condition_horizon_slope",
            "cell_intercept",
            "cell_horizon_slope",
            "cell_random_effect_cholesky",
            "residual_standard_deviation",
            "student_t_degrees_of_freedom"
        ]

        with model:
            inference_data = pm.sample_prior_predictive(
                draws = policy[
                    "priors"
                ][
                    "prior_predictive_draws_per_multiplier"
                ],
                random_seed = seed,
                var_names = required_prior_variables,
                return_inferencedata = True
            )

        log_values = np.asarray(
            inference_data.prior_predictive[
                likelihood_variable_name
            ].values,
            dtype = float
        )

        if log_values.ndim not in (2, 3):
            raise RuntimeError("Prior-predictive likelihood has unexpected dimensions.")

        observations = data[
            "observations"
        ]

        if log_values.shape[-1] != len(
            observations
        ):
            raise RuntimeError("Prior-predictive observation count does not match model input.")

        inside_mask = np.asarray(
            [
                observation[
                    "inside_evaluation_interval"
                ]
                for observation in observations
            ],
            dtype = bool
        )

        records = [
            build_prior_predictive_diagnostic_record(
                candidate = candidate,
                prior_scale_multiplier = (
                    prior_scale_multiplier
                ),
                diagnostic_population = (
                    "assembled_observations"
                ),
                log_values = log_values
            )
        ]

        if np.any(
            inside_mask
        ):
            records.append(
                build_prior_predictive_diagnostic_record(
                    candidate = candidate,
                    prior_scale_multiplier = (
                        prior_scale_multiplier
                    ),
                    diagnostic_population = (
                        "inside_evaluation_interval_observations"
                    ),
                    log_values = log_values[
                        ...,
                        inside_mask
                    ]
                )
            )

        outside_mask = ~inside_mask

        if np.any(
            outside_mask
        ):
            records.append(
                build_prior_predictive_diagnostic_record(
                    candidate = candidate,
                    prior_scale_multiplier = (
                        prior_scale_multiplier
                    ),
                    diagnostic_population = (
                        "outside_evaluation_interval_observations"
                    ),
                    log_values = log_values[
                        ...,
                        outside_mask
                    ]
                )
            )

        trajectory_grid = build_trajectory_grid(
            candidate = candidate,
            policy = policy
        )

        scaled_grid = (
            trajectory_grid
            - data[
                "training_horizon_midpoint"
            ]
        ) / data[
            "training_horizon_width"
        ]

        condition_intercepts = np.asarray(
            inference_data.prior[
                "condition_intercept"
            ].values,
            dtype = float
        )

        condition_slopes = np.asarray(
            inference_data.prior[
                "condition_horizon_slope"
            ].values,
            dtype = float
        )

        if (
            condition_intercepts.shape
            != condition_slopes.shape
        ):
            raise RuntimeError("Condition prior arrays have inconsistent shapes.")

        for (
            condition_index,
            condition_identity
        ) in enumerate(
            data[
                "condition_identities"
            ]
        ):
            latent_grid = (
                condition_intercepts[
                    ...,
                    condition_index,
                    None
                ]
                + condition_slopes[
                    ...,
                    condition_index,
                    None
                ]
                * scaled_grid[
                    None,
                    None,
                    :
                ]
            )

            records.append(
                build_prior_predictive_diagnostic_record(
                    candidate = candidate,
                    prior_scale_multiplier = (
                        prior_scale_multiplier
                    ),
                    diagnostic_population = (
                        "condition_latent_trajectory_grid"
                    ),
                    condition_protocol_identity = (
                        condition_identity
                    ),
                    log_values = latent_grid
                )
            )

            new_cell_grid = (
                build_prior_new_cell_predictive_grid(
                    inference_data = inference_data,
                    condition_index = condition_index,
                    scaled_grid = scaled_grid,
                    seed = (
                        seed
                        + condition_index
                        + 1
                    )
                )
            )

            records.append(
                build_prior_predictive_diagnostic_record(
                    candidate = candidate,
                    prior_scale_multiplier = (
                        prior_scale_multiplier
                    ),
                    diagnostic_population = (
                        "condition_new_cell_predictive_grid"
                    ),
                    condition_protocol_identity = (
                        condition_identity
                    ),
                    log_values = new_cell_grid
                )
            )

        records.sort(
            key = lambda record: (
                record[
                    "diagnostic_population"
                ],
                record[
                    "condition_protocol_identity"
                ]
                or ""
            )
        )

        return (
            inference_data,
            records
        )

    except Exception as exc:
        return (
            None,
            [
                {
                    "candidate_identity": candidate[
                        "candidate_identity"
                    ],
                    "outcome": candidate[
                        "collapsed_candidate"
                    ][
                        "outcome"
                    ],
                    "prior_scale_multiplier": (
                        prior_scale_multiplier
                    ),
                    "diagnostic_population": "execution",
                    "condition_protocol_identity": None,
                    "evaluation_point_count": 0,
                    "predictive_draw_count": 0,
                    "predictive_value_count": 0,
                    "execution_status": "prior_predictive_failed",
                    "execution_reason_code": (
                        "PRIOR_PREDICTIVE_DRAW_GENERATION_FAILED"
                    ),
                    "execution_error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "log_scale_median": None,
                    "log_scale_median_absolute_deviation": None,
                    "log_scale_interquartile_range": None,
                    "log_scale_minimum": None,
                    "log_scale_maximum": None,
                    "log_scale_quantiles": None,
                    "original_scale_finite_fraction": None,
                    "original_scale_nonfinite_count": None,
                    "original_scale_strictly_positive_finite_fraction": None,
                    "original_scale_finite_quantiles": None
                }
            ]
        )


# Checking all registered free random variables for finite draws
def validate_finite_posterior_draws(inference_data, policy: dict) -> None:
    for variable_name in policy[
        "posterior_artifact_contract"
    ][
        "free_random_variable_names"
    ]:
        if variable_name not in (
            inference_data.posterior
        ):
            raise RuntimeError(f"Posterior is missing a registered free random variable : {variable_name}")

        values = np.asarray(
            inference_data.posterior[
                variable_name
            ].values
        )

        if not np.all(
            np.isfinite(
                values
            )
        ):
            raise RuntimeError(f"Posterior contains a non-finite draw : {variable_name}")


# Validating one posterior artifact against the registered contract
def validate_posterior_inference_data(inference_data, candidate: dict,
                                      policy: dict, observations: list[dict] | None = None,
                                      require_posterior_predictive: bool = True
) -> None:
    required_groups = set(
        policy[
            "output_contract"
        ][
            "posterior_artifact_groups"
        ]
    )

    if not require_posterior_predictive:
        required_groups.discard(
            "posterior_predictive"
        )

    observed_groups = {
        group_name
        for group_name in inference_data.groups()
    }

    missing_groups = (
        required_groups
        - observed_groups
    )

    if missing_groups:
        raise RuntimeError(f"Posterior artifact is missing required groups : {sorted(missing_groups)}")

    artifact_contract = policy[
        "posterior_artifact_contract"
    ]

    required_posterior_variables = (
        set(
            artifact_contract[
                "free_random_variable_names"
            ]
        )
        | set(
            artifact_contract[
                "posterior_deterministic_variable_names"
            ]
        )
    )

    missing_posterior_variables = (
        required_posterior_variables
        - set(
            inference_data.posterior.data_vars
        )
    )

    if missing_posterior_variables:
        raise RuntimeError(f"Posterior artifact is missing registered variables : {sorted(missing_posterior_variables)}")

    if require_posterior_predictive:
        required_predictive_variables = (
            set(
                artifact_contract[
                    "pymc_posterior_predictive_variable_names"
                ]
            )
            | set(
                artifact_contract[
                    "derived_posterior_predictive_variable_names"
                ]
            )
        )

        missing_predictive_variables = (
            required_predictive_variables
            - set(
                inference_data.posterior_predictive.data_vars
            )
        )

        if missing_predictive_variables:
            raise RuntimeError(f"Posterior artifact is missing predictive variables : {sorted(missing_predictive_variables)}")

    expected_data = prepare_pymc_candidate_data(
        candidate = candidate,
        observations = observations
    )

    expected_observation_count = len(
        expected_data[
            "observations"
        ]
    )

    observed_data_count = int(
        inference_data.observed_data.sizes[
            "observation_index"
        ]
    )

    if (
        observed_data_count
        != expected_observation_count
    ):
        raise RuntimeError("Posterior artifact has an inconsistent observation dimension.")

    if require_posterior_predictive:
        predictive_count = int(
            inference_data.posterior_predictive.sizes[
                "observation_index"
            ]
        )

        if (
            predictive_count
            != expected_observation_count
        ):
            raise RuntimeError("Posterior artifact has an inconsistent observation dimension.")

    for coordinate_name, expected_values in {
        "condition_protocol_identity": (
            expected_data[
                "condition_identities"
            ]
        ),
        "cell_identity": (
            expected_data[
                "cell_identities"
            ]
        ),
        "observation_index": list(
            range(
                expected_observation_count
            )
        )
    }.items():
        observed_values = [
            str(
                value
            )
            for value in inference_data.posterior.coords[
                coordinate_name
            ].values.tolist()
        ]

        expected_strings = [
            str(
                value
            )
            for value in expected_values
        ]

        if observed_values != expected_strings:
            raise RuntimeError(f"Posterior artifact coordinate mismatch : {coordinate_name}")

    for variable_name, expected_values in {
        "condition_index": expected_data[
            "condition_indices"
        ],
        "cell_index": expected_data[
            "cell_indices"
        ],
        "scaled_horizon": expected_data[
            "scaled_horizons"
        ]
    }.items():
        if variable_name not in inference_data.constant_data:
            raise RuntimeError(f"Posterior artifact constant data is missing : {variable_name}")

        observed_values = np.asarray(
            inference_data.constant_data[
                variable_name
            ].values
        )

        if not np.array_equal(
            observed_values,
            np.asarray(
                expected_values
            )
        ):
            raise RuntimeError(f"Posterior artifact constant data mismatch : {variable_name}")

    observed_log_outcomes = np.asarray(
        inference_data.observed_data[
            policy[
                "posterior_artifact_contract"
            ][
                "likelihood_variable_name"
            ]
        ].values,
        dtype = float
    )

    if not np.array_equal(
        observed_log_outcomes,
        expected_data[
            "log_outcomes"
        ]
    ):
        raise RuntimeError("Posterior artifact observed outcomes differ from the registered model input.")

    for variable_name in inference_data.posterior.data_vars:
        variable_values = np.asarray(
            inference_data.posterior[
                variable_name
            ].values
        )

        if not np.all(
            np.isfinite(
                variable_values
            )
        ):
            raise RuntimeError(f"Posterior artifact contains non-finite posterior values : {variable_name}")

    if require_posterior_predictive:
        likelihood_variable_name = artifact_contract[
            "likelihood_variable_name"
        ]

        likelihood_predictive_values = np.asarray(
            inference_data.posterior_predictive[
                likelihood_variable_name
            ].values
        )

        if not np.all(
            np.isfinite(
                likelihood_predictive_values
            )
        ):
            raise RuntimeError("Posterior artifact contains non-finite log-scale predictive values.")

        original_scale_variable = (
            artifact_contract[
                "original_scale_predictive_variable_name"
            ]
        )

        original_scale_values = np.asarray(
            inference_data.posterior_predictive[
                original_scale_variable
            ].values
        )

        finite_original_scale_values = original_scale_values[
            np.isfinite(
                original_scale_values
            )
        ]

        if np.any(
            finite_original_scale_values <= 0
        ):
            raise RuntimeError("Posterior artifact contains nonpositive original-scale predictions.")


# Calculating complete registered sampler diagnostics
def calculate_sampler_diagnostics(inference_data, policy: dict) -> tuple[list[dict], dict]:
    _, _, az, _ = load_bayesian_runtime()

    free_random_variables = policy[
        "posterior_artifact_contract"
    ][
        "free_random_variable_names"
    ]

    validate_finite_posterior_draws(
        inference_data,
        policy
    )

    summary = az.summary(
        inference_data,
        var_names = free_random_variables,
        kind = "diagnostics",
        round_to = None
    )

    diagnostic_rows = []

    for parameter_identity, summary_row in (
        summary.iterrows()
    ):
        r_hat = float(
            summary_row["r_hat"]
        )

        effective_sample_size_bulk = float(
            summary_row["ess_bulk"]
        )

        effective_sample_size_tail = float(
            summary_row["ess_tail"]
        )

        diagnostic_rows.append(
            {
                "parameter_identity": str(
                    parameter_identity
                ),
                "rank_normalized_split_rhat": r_hat,
                "bulk_effective_sample_size": (
                    effective_sample_size_bulk
                ),
                "tail_effective_sample_size": (
                    effective_sample_size_tail
                ),
                "rhat_passed": (
                    math.isfinite(r_hat)
                    and r_hat
                    <= policy[
                        "diagnostics"
                    ][
                        "maximum_rank_normalized_split_rhat"
                    ]
                ),
                "bulk_ess_passed": (
                    math.isfinite(
                        effective_sample_size_bulk
                    )
                    and effective_sample_size_bulk
                    >= policy[
                        "diagnostics"
                    ][
                        "minimum_bulk_effective_sample_size"
                    ]
                ),
                "tail_ess_passed": (
                    math.isfinite(
                        effective_sample_size_tail
                    )
                    and effective_sample_size_tail
                    >= policy[
                        "diagnostics"
                    ][
                        "minimum_tail_effective_sample_size"
                    ]
                )
            }
        )

    sample_stats = (
        inference_data.sample_stats
    )

    divergence_count = int(
        np.asarray(
            sample_stats[
                "diverging"
            ].values
        ).sum()
    )

    tree_depth_name = next(
        (
            field_name
            for field_name in (
                "tree_depth",
                "depth"
            )
            if field_name in sample_stats
        ),
        None
    )

    if tree_depth_name is None:
        raise RuntimeError("Sampler output does not contain tree depth.")

    tree_depth_hit_count = int(
        (
            np.asarray(
                sample_stats[
                    tree_depth_name
                ].values
            )
            >= policy[
                "sampling"
            ][
                "maximum_tree_depth"
            ]
        ).sum()
    )

    bfmi_values = np.asarray(
        az.bfmi(
            inference_data
        ),
        dtype = float
    )

    minimum_bfmi = float(
        np.min(
            bfmi_values
        )
    )

    all_parameter_diagnostics_passed = all(
        diagnostic_row["rhat_passed"]
        and diagnostic_row["bulk_ess_passed"]
        and diagnostic_row["tail_ess_passed"]
        for diagnostic_row in diagnostic_rows
    )

    aggregate = {
        "divergence_count": divergence_count,
        "tree_depth_hit_count": (
            tree_depth_hit_count
        ),
        "bfmi_by_chain": [
            float(value)
            for value in bfmi_values
        ],
        "minimum_bfmi": minimum_bfmi,
        "divergence_check_passed": (
            divergence_count
            <= policy[
                "diagnostics"
            ][
                "maximum_divergence_count"
            ]
        ),
        "tree_depth_check_passed": (
            tree_depth_hit_count
            <= policy[
                "diagnostics"
            ][
                "maximum_tree_depth_hit_count"
            ]
        ),
        "bfmi_check_passed": (
            np.all(
                np.isfinite(
                    bfmi_values
                )
            )
            and minimum_bfmi
            >= policy[
                "diagnostics"
            ][
                "minimum_bfmi_per_chain"
            ]
        ),
        "parameter_diagnostics_passed": (
            all_parameter_diagnostics_passed
        )
    }

    aggregate[
        "sampler_diagnostics_status"
    ] = (
        "completed_passed"
        if (
            aggregate[
                "divergence_check_passed"
            ]
            and aggregate[
                "tree_depth_check_passed"
            ]
            and aggregate[
                "bfmi_check_passed"
            ]
            and aggregate[
                "parameter_diagnostics_passed"
            ]
        )
        else "completed_failed"
    )

    return diagnostic_rows, aggregate


# Adding candidate context and aggregate checks to sampler diagnostics
def complete_sampler_diagnostic_records(diagnostic_rows: list[dict], diagnostic_summary: dict,
                                        candidate_identity: str, outcome: str,
                                        prior_scale_multiplier: float,
                                        fit_scope: str = "full_data",
                                        fold_index: int | None = None
) -> list[dict]:
    if (
        diagnostic_summary[
            "sampler_diagnostics_status"
        ]
        == "completed_passed"
    ):
        diagnostic_reason_code = (
            "SAMPLER_DIAGNOSTICS_COMPLETED"
        )

    elif not diagnostic_summary[
        "divergence_check_passed"
    ]:
        diagnostic_reason_code = (
            "SAMPLER_DIVERGENCES_EXCEEDED"
        )

    elif not diagnostic_summary[
        "tree_depth_check_passed"
    ]:
        diagnostic_reason_code = (
            "SAMPLER_TREE_DEPTH_EXCEEDED"
        )

    elif not diagnostic_summary[
        "bfmi_check_passed"
    ]:
        diagnostic_reason_code = (
            "SAMPLER_BFMI_FAILED"
        )

    elif any(
        not diagnostic_row[
            "rhat_passed"
        ]
        for diagnostic_row in diagnostic_rows
    ):
        diagnostic_reason_code = (
            "SAMPLER_RHAT_EXCEEDED"
        )

    else:
        diagnostic_reason_code = (
            "SAMPLER_ESS_FAILED"
        )

    completed_records = []

    for diagnostic_row in diagnostic_rows:
        completed_records.append(
            {
                "candidate_identity": (
                    candidate_identity
                ),
                "outcome": outcome,
                "prior_scale_multiplier": (
                    prior_scale_multiplier
                ),
                "fit_scope": fit_scope,
                "fold_index": fold_index,
                "parameter_identity": (
                    diagnostic_row[
                        "parameter_identity"
                    ]
                ),
                "rank_normalized_split_rhat": (
                    diagnostic_row[
                        "rank_normalized_split_rhat"
                    ]
                ),
                "bulk_effective_sample_size": (
                    diagnostic_row[
                        "bulk_effective_sample_size"
                    ]
                ),
                "tail_effective_sample_size": (
                    diagnostic_row[
                        "tail_effective_sample_size"
                    ]
                ),
                "rhat_passed": (
                    diagnostic_row[
                        "rhat_passed"
                    ]
                ),
                "bulk_ess_passed": (
                    diagnostic_row[
                        "bulk_ess_passed"
                    ]
                ),
                "tail_ess_passed": (
                    diagnostic_row[
                        "tail_ess_passed"
                    ]
                ),
                "divergence_count": (
                    diagnostic_summary[
                        "divergence_count"
                    ]
                ),
                "minimum_bfmi": (
                    diagnostic_summary[
                        "minimum_bfmi"
                    ]
                ),
                "tree_depth_hit_count": (
                    diagnostic_summary[
                        "tree_depth_hit_count"
                    ]
                ),
                "diagnostic_status": (
                    diagnostic_summary[
                        "sampler_diagnostics_status"
                    ]
                ),
                "diagnostic_reason_code": (
                    diagnostic_reason_code
                )
            }
        )

    return completed_records


# Sampling one full-data posterior and its observed-data replication
def sample_posterior_candidate(candidate: dict, policy: dict,
                               prior_scale_multiplier: float
) -> tuple[object | None, dict, list[dict]]:
    try:
        model, data = build_pymc_model(
            candidate = candidate,
            policy = policy,
            prior_scale_multiplier = (
                prior_scale_multiplier
            )
        )

        pm, _, _, _ = load_bayesian_runtime()

        with model:
            inference_data = pm.sample(
                draws = policy[
                    "sampling"
                ][
                    "draws_per_chain"
                ],
                tune = policy[
                    "sampling"
                ][
                    "tuning_draws_per_chain"
                ],
                chains = policy[
                    "sampling"
                ][
                    "chains"
                ],
                cores = policy[
                    "sampling"
                ][
                    "cores"
                ],
                random_seed = policy[
                    "sampling"
                ][
                    "sampler_seed"
                ],
                init = policy[
                    "sampling"
                ][
                    "initialization"
                ].replace(
                    "_plus_",
                    "+"
                ),
                nuts_sampler = "pymc",
                nuts = {
                    "target_accept": policy[
                        "sampling"
                    ][
                        "target_accept"
                    ],
                    "max_treedepth": policy[
                        "sampling"
                    ][
                        "maximum_tree_depth"
                    ]
                },
                discard_tuned_samples = policy[
                    "sampling"
                ][
                    "discard_tuned_samples"
                ],
                compute_convergence_checks = policy[
                    "sampling"
                ][
                    "compute_convergence_checks"
                ],
                return_inferencedata = True,
                progressbar = False
            )

            inference_data = pm.sample_posterior_predictive(
                inference_data,
                var_names = policy[
                    "posterior_artifact_contract"
                ][
                    "pymc_posterior_predictive_variable_names"
                ],
                random_seed = policy[
                    "sampling"
                ][
                    "posterior_predictive_seed"
                ],
                return_inferencedata = True,
                extend_inferencedata = True,
                progressbar = False
            )

        likelihood_variable_name = policy[
            "posterior_artifact_contract"
        ][
            "likelihood_variable_name"
        ]

        log_predictive = (
            inference_data.posterior_predictive[
                likelihood_variable_name
            ]
        )

        with np.errstate(
            over = "ignore",
            under = "ignore",
            invalid = "ignore"
        ):
            predictive_outcome = np.exp(
                log_predictive
            )

        inference_data.posterior_predictive[
            policy[
                "posterior_artifact_contract"
            ][
                "original_scale_predictive_variable_name"
            ]
        ] = predictive_outcome

        validate_posterior_inference_data(
            inference_data = inference_data,
            candidate = candidate,
            policy = policy
        )

        diagnostic_rows, diagnostic_summary = (
            calculate_sampler_diagnostics(
                inference_data = inference_data,
                policy = policy
            )
        )

        outcome = candidate[
            "collapsed_candidate"
        ][
            "outcome"
        ]

        diagnostic_rows = (
            complete_sampler_diagnostic_records(
                diagnostic_rows = diagnostic_rows,
                diagnostic_summary = diagnostic_summary,
                candidate_identity = candidate[
                    "candidate_identity"
                ],
                outcome = outcome,
                prior_scale_multiplier = (
                    prior_scale_multiplier
                )
            )
        )

        predictive_values = np.asarray(
            predictive_outcome.values,
            dtype = float
        )

        predictive_finite = np.isfinite(
            predictive_values
        )

        predictive_positive_finite = (
            predictive_finite
            & (predictive_values > 0)
        )

        posterior_record = {
            "candidate_identity": candidate[
                "candidate_identity"
            ],
            "outcome": outcome,
            "prior_scale_multiplier": (
                prior_scale_multiplier
            ),
            "fit_status": "completed",
            "fit_error": None,
            "posterior_draw_count": int(
                policy["sampling"]["chains"]
                * policy[
                    "sampling"
                ][
                    "draws_per_chain"
                ]
            ),
            "posterior_predictive_value_count": int(
                predictive_values.size
            ),
            "posterior_predictive_nonfinite_count": int(
                (~predictive_finite).sum()
            ),
            "posterior_predictive_positive_finite_fraction": float(
                predictive_positive_finite.mean()
            ),
            **diagnostic_summary
        }

        return (
            inference_data,
            posterior_record,
            diagnostic_rows
        )

    except Exception as exc:
        return (
            None,
            {
                "candidate_identity": candidate[
                    "candidate_identity"
                ],
                "outcome": candidate[
                    "collapsed_candidate"
                ][
                    "outcome"
                ],
                "prior_scale_multiplier": (
                    prior_scale_multiplier
                ),
                "fit_status": "fit_failed",
                "fit_error": (
                    f"{type(exc).__name__}: {exc}"
                ),
                "traceback": traceback.format_exc(),
                "posterior_draw_count": 0,
                "posterior_predictive_value_count": 0,
                "posterior_predictive_nonfinite_count": None,
                "posterior_predictive_positive_finite_fraction": None,
                "sampler_diagnostics_status": (
                    "unavailable"
                )
            },
            []
        )


# Indexing reconstructed candidate objects by deterministic identity
def component_candidates_by_identity(workflow: dict) -> dict[str, dict]:
    indexed_candidates = {}

    for candidate in workflow[
        "component_candidates"
    ]:
        candidate_identity = candidate[
            "candidate_identity"
        ]

        if candidate_identity in indexed_candidates:
            raise RuntimeError(f"Duplicate Bayesian candidate identity : {candidate_identity}")

        indexed_candidates[
            candidate_identity
        ] = candidate

    return indexed_candidates


# Rejecting invalid shard configuration
def validate_cache_shard(shard_count: int, shard_index: int) -> None:
    if shard_count <= 0:
        raise ValueError("Shard count must be positive.")

    if (
        shard_index < 0
        or shard_index >= shard_count
    ):
        raise ValueError("Shard index must be between zero and shard count minus one.")


# Assigning one cache job to a shard by identity hash
def cache_job_assigned_to_shard(cache_identity: dict, shard_count: int, shard_index: int) -> bool:
    cache_identity_integer = int(
        cache_identity[
            "cache_identity_sha256"
        ],
        16
    )

    return (
        cache_identity_integer
        % shard_count
        == shard_index
    )


# Building the deterministic cache identity for one operation
def build_cache_identity(policy: dict, policy_sha256: str,
                         input_hashes: dict, software_environment_sha256: str,
                         candidate_identity: str, operation: str,
                         prior_scale_multiplier: float, validation_fold_identity: str | None
) -> dict:
    identity_fields = policy["execution_planning"]["cache_identity_fields"]

    identity_payload = {
        "policy_sha256": policy_sha256,
        "verified_input_hashes": input_hashes,
        "candidate_identity": candidate_identity,
        "operation": operation,
        "prior_scale_multiplier": float(
            prior_scale_multiplier
        ),
        "validation_fold_identity": (
            validation_fold_identity
        ),
        "software_environment_sha256": (
            software_environment_sha256
        ),
        "implementation_sha256": (
            calculate_file_sha256(
                Path(
                    __file__
                ).resolve()
            )
        )
    }

    if set(identity_payload) != set(identity_fields):
        raise RuntimeError("Cache identity does not match the registered identity fields.")

    identity_payload[
        "cache_identity_sha256"
    ] = deterministic_identity(
        identity_payload
    )

    return identity_payload


# Reconstructing one explicitly authorized source-cache identity
def build_import_source_cache_identity(policy: dict, cache_identity: dict) -> dict | None:
    import_contract = policy["cache_import_contract"]

    if not import_contract["enabled"]:
        return None

    if cache_identity["operation"] not in set(import_contract["permitted_operations"]):
        return None

    if cache_identity["verified_input_hashes"] != import_contract["verified_input_hashes"]:
        raise RuntimeError("Registered cache import input hashes do not match the current analysis inputs.")

    source_identity = {
        "policy_sha256": import_contract["source_policy_sha256"],
        "verified_input_hashes": import_contract["verified_input_hashes"],
        "candidate_identity": cache_identity["candidate_identity"],
        "operation": cache_identity["operation"],
        "prior_scale_multiplier": float(cache_identity["prior_scale_multiplier"]),
        "validation_fold_identity": cache_identity["validation_fold_identity"],
        "software_environment_sha256": import_contract["source_software_environment_sha256"],
        "implementation_sha256": import_contract["source_implementation_sha256"]
    }
    source_identity["cache_identity_sha256"] = deterministic_identity(source_identity)
    return source_identity


# Reconciling the complete registered source-cache inventory before reuse
def validate_registered_source_cache_inventory(cache_directory: Path, policy: dict) -> None:
    import_contract = policy["cache_import_contract"]

    if not import_contract["enabled"]:
        return

    operation_counts = Counter()
    source_entry_count = 0

    if not cache_directory.is_dir():
        raise RuntimeError("Registered source cache directory is unavailable.")

    for entry_directory in sorted(cache_directory.iterdir()):
        metadata_path = entry_directory / policy["cache_contract"]["cache_metadata_file"]

        if not entry_directory.is_dir() or not metadata_path.is_file():
            continue

        try:
            metadata_record = json.loads(metadata_path.read_text(encoding = "utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        source_identity = metadata_record.get("cache_identity", {})

        if source_identity.get("policy_sha256") != import_contract["source_policy_sha256"]:
            continue

        source_entry_count += 1
        operation_counts[source_identity.get("operation")] += 1

        if source_identity.get("implementation_sha256") != import_contract["source_implementation_sha256"]:
            raise RuntimeError("Registered source cache contains an unexpected implementation identity.")

        if source_identity.get("software_environment_sha256") != import_contract["source_software_environment_sha256"]:
            raise RuntimeError("Registered source cache contains an unexpected software environment identity.")

        if source_identity.get("verified_input_hashes") != import_contract["verified_input_hashes"]:
            raise RuntimeError("Registered source cache contains unexpected input hashes.")

        if metadata_record.get("execution_status") != import_contract["required_source_execution_status"]:
            raise RuntimeError("Registered source cache contains an incomplete execution status.")

        if import_contract["inference_data_required"] and not (
            entry_directory / policy["cache_contract"]["cache_inference_data_file"]
        ).is_file():
            raise RuntimeError("Registered source cache is missing inference data.")

    if source_entry_count != import_contract["required_source_entry_count"]:
        raise RuntimeError(
            "Registered source cache entry count differs from the import contract : "
            f"{source_entry_count}"
        )

    if dict(sorted(operation_counts.items())) != dict(
        sorted(import_contract["required_operation_counts"].items())
    ):
        raise RuntimeError("Registered source cache operation counts do not reconcile.")


# Building the deterministic cache entry directory
def build_cache_entry_directory(cache_directory: Path, cache_identity: dict) -> Path:
    return (
        cache_directory
        / cache_identity["cache_identity_sha256"]
    )


# Reading one validated cache entry
def read_validated_cache_entry(cache_directory: Path, cache_identity: dict,
                               policy: dict
) -> tuple[object | None, object | None, str | None]:
    entry_directory = build_cache_entry_directory(
        cache_directory = cache_directory,
        cache_identity = cache_identity
    )

    if not entry_directory.is_dir():
        return None, None, "CACHE_METADATA_MISSING"

    metadata_path = entry_directory / policy["cache_contract"]["cache_metadata_file"]
    inference_data_path = entry_directory / policy["cache_contract"]["cache_inference_data_file"]
    completion_path = entry_directory / policy["cache_contract"]["cache_completion_file"]
    diagnostics_path = entry_directory / "diagnostics.json"
    failure_path = entry_directory / "failure.json"

    if (
        metadata_path.is_file()
        and completion_path.is_file()
        and failure_path.is_file()
        and not inference_data_path.is_file()
    ):
        try:
            metadata_record = json.loads(
                metadata_path.read_text(
                    encoding = "utf-8"
                )
            )

            completion_record = json.loads(
                completion_path.read_text(
                    encoding = "utf-8"
                )
            )

            failure_record = json.loads(
                failure_path.read_text(
                    encoding = "utf-8"
                )
            )

        except (
            OSError,
            json.JSONDecodeError
        ):
            return None, None, "CACHE_METADATA_SCHEMA_INVALID"

        if (
            metadata_record.get(
                "execution_status"
            )
            != "failed"
            or completion_record.get(
                "execution_status"
            )
            != "failed"
        ):
            return None, None, "CACHE_METADATA_SCHEMA_INVALID"

        if (
            metadata_record[
                "cache_identity"
            ]
            != cache_identity
        ):
            return None, None, "CACHE_IDENTITY_MISMATCH"

        stored_identity_payload = {
            key: value for key, value in metadata_record["cache_identity"].items()
            if key != "cache_identity_sha256"
        }

        if deterministic_identity(stored_identity_payload) != metadata_record["cache_identity_sha256"]:
            return None, None, "CACHE_IDENTITY_MISMATCH"

        if (
            entry_directory.name
            != cache_identity[
                "cache_identity_sha256"
            ]
        ):
            return None, None, "CACHE_IDENTITY_MISMATCH"

        if (
            completion_record[
                "cache_metadata_sha256"
            ]
            != calculate_file_sha256(
                metadata_path
            )
        ):
            return None, None, "CACHE_ARTIFACT_HASH_MISMATCH"

        if (
            metadata_record[
                "artifact_hashes"
            ].get(
                "failure_sha256"
            )
            != calculate_file_sha256(
                failure_path
            )
        ):
            return None, None, "CACHE_ARTIFACT_HASH_MISMATCH"

        return (
            None,
            {
                "terminal_failure": (
                    failure_record
                ),
                "cache_metadata": (
                    metadata_record
                )
            },
            "CACHE_TERMINAL_FAILURE"
        )

    if not (
        metadata_path.is_file()
        and inference_data_path.is_file()
        and completion_path.is_file()
        and diagnostics_path.is_file()
    ):
        return None, None, "CACHE_INCOMPLETE_ENTRY"

    try:
        metadata_record = json.loads(
            metadata_path.read_text(
                encoding = "utf-8"
            )
        )

        completion_record = json.loads(
            completion_path.read_text(
                encoding = "utf-8"
            )
        )

        diagnostics = json.loads(
            diagnostics_path.read_text(
                encoding = "utf-8"
            )
        )

    except (OSError, json.JSONDecodeError):
        return None, None, "CACHE_METADATA_SCHEMA_INVALID"

    if set(metadata_record) != set(policy["cache_contract"]["cache_metadata_fields"]):
        return None, None, "CACHE_METADATA_SCHEMA_INVALID"

    if set(completion_record) != set(policy["cache_contract"]["cache_completion_fields"]):
        return None, None, "CACHE_METADATA_SCHEMA_INVALID"

    if metadata_record["cache_identity"] != cache_identity:
        return None, None, "CACHE_IDENTITY_MISMATCH"

    stored_identity_payload = {
        key: value for key, value in metadata_record["cache_identity"].items()
        if key != "cache_identity_sha256"
    }

    if deterministic_identity(stored_identity_payload) != metadata_record["cache_identity_sha256"]:
        return None, None, "CACHE_IDENTITY_MISMATCH"

    if entry_directory.name != cache_identity["cache_identity_sha256"]:
        return None, None, "CACHE_IDENTITY_MISMATCH"

    if completion_record["cache_metadata_sha256"] != calculate_file_sha256(metadata_path):
        return None, None, "CACHE_ARTIFACT_HASH_MISMATCH"

    observed_inference_data_sha256 = calculate_file_sha256(inference_data_path)

    if completion_record["inference_data_sha256"] != observed_inference_data_sha256:
        return None, None, "CACHE_ARTIFACT_HASH_MISMATCH"

    artifact_hashes = metadata_record["artifact_hashes"]

    if artifact_hashes.get("inference_data_sha256") != observed_inference_data_sha256:
        return None, None, "CACHE_ARTIFACT_HASH_MISMATCH"

    if artifact_hashes.get("diagnostics_sha256") != calculate_file_sha256(diagnostics_path):
        return None, None, "CACHE_ARTIFACT_HASH_MISMATCH"

    if completion_record["execution_status"] != "completed":
        return None, None, "CACHE_INCOMPLETE_ENTRY"

    _, _, az, _ = load_bayesian_runtime()

    try:
        inference_data = az.from_netcdf(
            str(
                inference_data_path
            )
        )

    except Exception:
        return None, None, "CACHE_ARTIFACT_HASH_MISMATCH"

    return inference_data, diagnostics, None


# Loading a current cache entry or a registered immutable source entry
def read_cache_entry_with_registered_import(cache_directory: Path, cache_identity: dict,
                                            policy: dict
) -> tuple[
    object | None,
    object | None,
    str | None,
    dict,
    bool
]:
    (
        inference_data,
        diagnostics,
        miss_reason
    ) = read_validated_cache_entry(
        cache_directory,
        cache_identity,
        policy
    )

    if inference_data is not None:
        return (
            inference_data,
            diagnostics,
            None,
            cache_identity,
            False
        )

    source_identity = build_import_source_cache_identity(
        policy,
        cache_identity
    )

    if source_identity is None:
        return (
            inference_data,
            diagnostics,
            miss_reason,
            cache_identity,
            False
        )

    (
        source_inference,
        source_diagnostics,
        source_miss_reason
    ) = read_validated_cache_entry(
        cache_directory,
        source_identity,
        policy
    )

    if source_inference is None:
        raise RuntimeError(
            "Registered immutable source cache entry is unavailable or invalid : "
            f"{source_identity['cache_identity_sha256']} : {source_miss_reason}"
        )

    return (
        source_inference,
        source_diagnostics,
        None,
        source_identity,
        True
    )


# Quarantining a corrupt cache entry
def quarantine_corrupt_cache_entry(cache_directory: Path, cache_identity: dict, reason_code: str) -> None:
    entry_directory = build_cache_entry_directory(
        cache_directory = cache_directory,
        cache_identity = cache_identity
    )

    if not entry_directory.is_dir():
        return

    existing_inventory = [
        (
            entry_path.name,
            calculate_file_sha256(entry_path)
        )
        for entry_path in sorted(entry_directory.iterdir())
        if entry_path.is_file()
    ]

    inventory_sha256 = deterministic_identity(
        {
            "reason_code": reason_code,
            "existing_inventory": existing_inventory
        }
    )

    quarantine_directory = cache_directory / (
        f"{entry_directory.name}.corrupt_{inventory_sha256}"
    )

    if quarantine_directory.exists():
        shutil.rmtree(
            quarantine_directory,
            ignore_errors = True
        )

    entry_directory.rename(quarantine_directory)


# Writing one cache entry transactionally
def write_cache_entry_transactionally(cache_directory: Path, cache_identity: dict,
                                      policy: dict, software_environment_record: dict,
                                      software_environment_sha256: str, inference_data: object,
                                      diagnostics: object
) -> None:
    cache_directory.mkdir(
        parents = True,
        exist_ok = True
    )

    entry_directory = build_cache_entry_directory(
        cache_directory = cache_directory,
        cache_identity = cache_identity
    )

    staging_directory = cache_directory / (
        f".{entry_directory.name}.staging"
    )

    if staging_directory.exists():
        shutil.rmtree(
            staging_directory,
            ignore_errors = True
        )

    staging_directory.mkdir()

    try:
        inference_data_path = staging_directory / policy["cache_contract"]["cache_inference_data_file"]
        diagnostics_path = staging_directory / "diagnostics.json"
        metadata_path = staging_directory / policy["cache_contract"]["cache_metadata_file"]
        completion_path = staging_directory / policy["cache_contract"]["cache_completion_file"]

        inference_data.to_netcdf(str(inference_data_path))

        diagnostics_path.write_text(
            json.dumps(
                diagnostics,
                indent = 2,
                sort_keys = True,
                allow_nan = False
            )
            + "\n",
            encoding = "utf-8"
        )

        inference_data_sha256 = calculate_file_sha256(inference_data_path)
        diagnostics_sha256 = calculate_file_sha256(diagnostics_path)

        metadata_record = {
            "cache_identity": cache_identity,
            "cache_identity_sha256": cache_identity["cache_identity_sha256"],
            "software_environment": software_environment_record,
            "software_environment_sha256": software_environment_sha256,
            "execution_status": "completed",
            "artifact_hashes": {
                "inference_data_sha256": inference_data_sha256,
                "diagnostics_sha256": diagnostics_sha256
            }
        }

        metadata_path.write_text(
            json.dumps(
                metadata_record,
                indent = 2,
                sort_keys = True,
                allow_nan = False
            )
            + "\n",
            encoding = "utf-8"
        )

        completion_record = {
            "cache_identity_sha256": cache_identity["cache_identity_sha256"],
            "cache_metadata_sha256": calculate_file_sha256(metadata_path),
            "inference_data_sha256": inference_data_sha256,
            "execution_status": "completed"
        }

        completion_path.write_text(
            json.dumps(
                completion_record,
                indent = 2,
                sort_keys = True,
                allow_nan = False
            )
            + "\n",
            encoding = "utf-8"
        )

        if entry_directory.exists():
            shutil.rmtree(
                entry_directory,
                ignore_errors = True
            )

        staging_directory.rename(entry_directory)

    finally:
        if staging_directory.exists():
            shutil.rmtree(
                staging_directory,
                ignore_errors = True
            )


# Writing one terminal-failure cache entry transactionally
def write_failure_cache_entry_transactionally(cache_directory: Path, cache_identity: dict,
                                              policy: dict, software_environment_record: dict,
                                              software_environment_sha256: str, failure_record: dict
) -> None:
    cache_directory.mkdir(
        parents = True,
        exist_ok = True
    )

    entry_directory = build_cache_entry_directory(
        cache_directory = cache_directory,
        cache_identity = cache_identity
    )

    staging_directory = cache_directory / (
        f".{entry_directory.name}.failure-staging"
    )

    if staging_directory.exists():
        shutil.rmtree(
            staging_directory,
            ignore_errors = True
        )

    staging_directory.mkdir()

    try:
        failure_path = (
            staging_directory
            / "failure.json"
        )

        failure_path.write_text(
            json.dumps(
                failure_record,
                indent = 2,
                sort_keys = True,
                allow_nan = False
            )
            + "\n",
            encoding = "utf-8"
        )

        failure_sha256 = calculate_file_sha256(
            failure_path
        )

        metadata_record = {
            "cache_identity": cache_identity,
            "cache_identity_sha256": (
                cache_identity[
                    "cache_identity_sha256"
                ]
            ),
            "software_environment": (
                software_environment_record
            ),
            "software_environment_sha256": (
                software_environment_sha256
            ),
            "execution_status": "failed",
            "artifact_hashes": {
                "failure_sha256": (
                    failure_sha256
                )
            }
        }

        metadata_path = (
            staging_directory
            / policy["cache_contract"]["cache_metadata_file"]
        )

        metadata_path.write_text(
            json.dumps(
                metadata_record,
                indent = 2,
                sort_keys = True,
                allow_nan = False
            )
            + "\n",
            encoding = "utf-8"
        )

        completion_record = {
            "cache_identity_sha256": (
                cache_identity[
                    "cache_identity_sha256"
                ]
            ),
            "cache_metadata_sha256": (
                calculate_file_sha256(
                    metadata_path
                )
            ),
            "execution_status": "failed"
        }

        completion_path = (
            staging_directory
            / policy["cache_contract"]["cache_completion_file"]
        )

        completion_path.write_text(
            json.dumps(
                completion_record,
                indent = 2,
                sort_keys = True,
                allow_nan = False
            )
            + "\n",
            encoding = "utf-8"
        )

        if entry_directory.exists():
            shutil.rmtree(
                entry_directory,
                ignore_errors = True
            )

        staging_directory.rename(
            entry_directory
        )

    finally:
        if staging_directory.exists():
            shutil.rmtree(
                staging_directory,
                ignore_errors = True
            )


# Running all registered prior-predictive analyses with cache streaming
def execute_prior_predictive_sampling(workflow: dict, cache_directory: Path,
                                      software_environment_record: dict, software_environment_sha256: str,
                                      shard_count: int = 1, shard_index: int = 0
) -> tuple[list[dict], dict]:
    validate_cache_shard(shard_count, shard_index)

    policy = workflow["validated_inputs"]["policy"]

    input_hashes = {
        "feature_manifest_sha256": workflow["validated_inputs"]["feature_manifest_sha256"],
        "design_report_sha256": workflow["validated_inputs"]["design_report_sha256"],
        "identification_report_sha256": workflow["validated_inputs"]["identification_report_sha256"],
        "descriptive_report_sha256": workflow["validated_inputs"]["descriptive_report_sha256"]
    }

    indexed_candidates = component_candidates_by_identity(
        workflow
    )

    diagnostic_records = []

    execution_accounting = {
        "requested_call_count": 0,
        "completed_call_count": 0,
        "failed_call_count": 0,
        "reused_cache_count": 0,
        "imported_cache_count": 0,
        "new_cache_count": 0
    }

    for candidate_record in workflow["candidate_records"]:
        if candidate_record["admissibility_status"] != "admissible":
            continue

        candidate_identity = candidate_record["candidate_identity"]
        candidate = indexed_candidates[candidate_identity]

        for prior_scale_multiplier in policy["priors"]["prior_sensitivity_scale_multipliers"]:
            cache_identity = build_cache_identity(
                policy = policy,
                policy_sha256 = workflow["validated_inputs"]["policy_sha256"],
                input_hashes = input_hashes,
                software_environment_sha256 = software_environment_sha256,
                candidate_identity = candidate_identity,
                operation = "prior_predictive",
                prior_scale_multiplier = prior_scale_multiplier,
                validation_fold_identity = None
            )

            if not cache_job_assigned_to_shard(
                cache_identity = cache_identity,
                shard_count = shard_count,
                shard_index = shard_index
            ):
                continue

            execution_accounting["requested_call_count"] += 1

            (
                cached_inference_data,
                cached_diagnostics,
                cache_miss_reason,
                loaded_cache_identity,
                imported_cache
            ) = read_cache_entry_with_registered_import(
                cache_directory = cache_directory,
                cache_identity = cache_identity,
                policy = policy
            )

            if (
                cached_inference_data is not None
                and cached_diagnostics is not None
            ):
                diagnostic_records.extend(cached_diagnostics)
                execution_accounting["completed_call_count"] += 1
                execution_accounting[
                    "imported_cache_count" if imported_cache else "reused_cache_count"
                ] += 1
                del cached_inference_data
                continue

            if (
                cache_miss_reason is not None
                and cache_miss_reason != "CACHE_METADATA_MISSING"
            ):
                quarantine_corrupt_cache_entry(
                    cache_directory = cache_directory,
                    cache_identity = loaded_cache_identity,
                    reason_code = cache_miss_reason
                )

            (
                inference_data,
                candidate_diagnostic_records
            ) = sample_prior_predictive_candidate(
                candidate = candidate,
                policy = policy,
                prior_scale_multiplier = prior_scale_multiplier
            )

            if inference_data is not None:
                write_cache_entry_transactionally(
                    cache_directory = cache_directory,
                    cache_identity = cache_identity,
                    policy = policy,
                    software_environment_record = software_environment_record,
                    software_environment_sha256 = software_environment_sha256,
                    inference_data = inference_data,
                    diagnostics = candidate_diagnostic_records
                )
                execution_accounting["completed_call_count"] += 1
                execution_accounting["new_cache_count"] += 1
                del inference_data
            else:
                execution_accounting["failed_call_count"] += 1

            diagnostic_records.extend(candidate_diagnostic_records)

    diagnostic_records.sort(
        key = lambda record: (
            record["candidate_identity"],
            record["outcome"],
            record["prior_scale_multiplier"],
            record["diagnostic_population"],
            record["condition_protocol_identity"] or ""
        )
    )

    return (
        diagnostic_records,
        execution_accounting
    )


# Building the explicit human-review template
def build_prior_predictive_review_template(workflow: dict, diagnostic_records: list[dict],
                                           candidate_inventory_sha256: str, diagnostics_sha256: str
) -> dict:
    validated_inputs = workflow[
        "validated_inputs"
    ]

    successful_pairs = sorted({
        (
            record["candidate_identity"],
            record["outcome"],
            float(
                record["prior_scale_multiplier"]
            )
        )
        for record in diagnostic_records
        if record["execution_status"] == "completed"
    })

    review_records = [
        {
            "candidate_identity": candidate_identity,
            "outcome": outcome,
            "prior_scale_multiplier": prior_scale_multiplier,
            "review_status": "pending",
            "review_rationale": ""
        }
        for (
            candidate_identity,
            outcome,
            prior_scale_multiplier
        ) in successful_pairs
    ]

    return {
        "policy_sha256": validated_inputs[
            "policy_sha256"
        ],
        "candidate_inventory_sha256": (
            candidate_inventory_sha256
        ),
        "prior_predictive_diagnostics_sha256": (
            diagnostics_sha256
        ),
        "review_records": review_records
    }


# Validating prior-predictive review before any posterior call
def validate_prior_predictive_review(review_path: Path, workflow: dict,
                                     candidate_inventory_sha256: str, diagnostics_sha256: str,
                                     diagnostic_records: list[dict]
) -> set[tuple[str, str, float]]:
    review_document = load_json_object(
        review_path
    )

    required_document_fields = set(
        workflow[
            "validated_inputs"
        ][
            "policy"
        ][
            "prior_predictive_review"
        ][
            "review_document_fields"
        ]
    )

    if set(review_document) != required_document_fields:
        raise RuntimeError("Prior-predictive review has an invalid schema.")

    if (
        review_document["policy_sha256"]
        != workflow[
            "validated_inputs"
        ][
            "policy_sha256"
        ]
    ):
        raise RuntimeError("Prior-predictive review policy hash mismatch.")

    if (
        review_document[
            "candidate_inventory_sha256"
        ]
        != candidate_inventory_sha256
    ):
        raise RuntimeError("Prior-predictive review candidate hash mismatch.")

    if (
        review_document[
            "prior_predictive_diagnostics_sha256"
        ]
        != diagnostics_sha256
    ):
        raise RuntimeError("Prior-predictive review diagnostics hash mismatch.")

    successful_pairs = {
        (
            record["candidate_identity"],
            record["outcome"],
            float(
                record["prior_scale_multiplier"]
            )
        )
        for record in diagnostic_records
        if record["execution_status"] == "completed"
    }

    observed_pairs = set()
    acknowledged_pairs = set()

    for record in review_document[
        "review_records"
    ]:
        required_record_fields = set(
            workflow[
                "validated_inputs"
            ][
                "policy"
            ][
                "prior_predictive_review"
            ][
                "review_record_fields"
            ]
        )

        if set(record) != required_record_fields:
            raise RuntimeError("Prior-predictive review record has an invalid schema.")

        pair = (
            record["candidate_identity"],
            record["outcome"],
            float(
                record["prior_scale_multiplier"]
            )
        )

        if pair in observed_pairs:
            raise RuntimeError(f"Prior-predictive review contains a duplicate pair : {pair}")

        observed_pairs.add(pair)

        permitted_statuses = set(
            workflow[
                "validated_inputs"
            ][
                "policy"
            ][
                "prior_predictive_review"
            ][
                "permitted_review_statuses"
            ]
        )

        if record["review_status"] not in permitted_statuses:
            raise RuntimeError("Prior-predictive review contains an unregistered review status.")

        if record["review_status"] == (
            "policy_amendment_required"
        ):
            raise RuntimeError("Posterior sampling is blocked because a policy amendment is required.")

        if record["review_status"] != "acknowledged":
            raise RuntimeError(f"Posterior sampling is blocked by an unacknowledged review record : {pair}")

        if not record[
            "review_rationale"
        ].strip():
            raise RuntimeError(f"Acknowledged prior review requires a nonempty rationale : {pair}")

        acknowledged_pairs.add(pair)

    if observed_pairs != successful_pairs:
        raise RuntimeError("Prior-predictive review does not cover the complete successful pair population.")

    return acknowledged_pairs


# Building the verified input-hash set used by stochastic caches
def build_stochastic_input_hashes(workflow: dict) -> dict:
    validated_inputs = workflow[
        "validated_inputs"
    ]

    return {
        "feature_manifest_sha256": validated_inputs[
            "feature_manifest_sha256"
        ],
        "design_report_sha256": validated_inputs[
            "design_report_sha256"
        ],
        "identification_report_sha256": validated_inputs[
            "identification_report_sha256"
        ],
        "descriptive_report_sha256": validated_inputs[
            "descriptive_report_sha256"
        ]
    }


# Packaging one completed posterior cache record
def build_posterior_cache_diagnostics(posterior_record: dict, parameter_diagnostic_records: list[dict]) -> dict:
    return {
        "posterior_record": posterior_record,
        "parameter_diagnostic_records": (
            parameter_diagnostic_records
        )
    }


# Validating one completed posterior cache record
def validate_posterior_cache_diagnostics(cached_diagnostics: object, candidate_identity: str,
                                         outcome: str, prior_scale_multiplier: float
) -> tuple[dict, list[dict]]:
    if not isinstance(
        cached_diagnostics,
        dict
    ):
        raise RuntimeError("Posterior cache diagnostics are not an object.")

    if set(
        cached_diagnostics
    ) != {
        "posterior_record",
        "parameter_diagnostic_records"
    }:
        raise RuntimeError("Posterior cache diagnostics have an invalid schema.")

    posterior_record = cached_diagnostics[
        "posterior_record"
    ]

    parameter_records = cached_diagnostics[
        "parameter_diagnostic_records"
    ]

    if not isinstance(
        posterior_record,
        dict
    ):
        raise RuntimeError("Posterior cache record is invalid.")

    if not isinstance(
        parameter_records,
        list
    ):
        raise RuntimeError("Posterior parameter diagnostics are invalid.")

    expected_parameter_fields = set(
        SAMPLER_DIAGNOSTIC_FIELDS
    )

    for parameter_record in parameter_records:
        if (
            not isinstance(
                parameter_record,
                dict
            )
            or set(
                parameter_record
            )
            != expected_parameter_fields
        ):
            raise RuntimeError("Posterior parameter diagnostics have an invalid schema.")

        if (
            parameter_record[
                "candidate_identity"
            ]
            != candidate_identity
            or parameter_record[
                "outcome"
            ]
            != outcome
            or float(
                parameter_record[
                    "prior_scale_multiplier"
                ]
            )
            != float(
                prior_scale_multiplier
            )
        ):
            raise RuntimeError("Posterior parameter diagnostics reference the wrong fit.")

    if parameter_records:
        all_parameters_passed = all(
            parameter_record["diagnostic_status"] == "completed_passed"
            for parameter_record in parameter_records
        )

        any_parameter_failed = any(
            parameter_record["diagnostic_status"] == "completed_failed"
            for parameter_record in parameter_records
        )

        aggregate_status = posterior_record.get("sampler_diagnostics_status")

        if aggregate_status == "completed_passed" and not all_parameters_passed:
            raise RuntimeError("Posterior cache aggregate sampler status conflicts with per-parameter diagnostics.")

        if aggregate_status == "completed_failed" and not any_parameter_failed:
            raise RuntimeError("Posterior cache aggregate sampler status conflicts with per-parameter diagnostics.")

    raw_multiplier = posterior_record.get(
        "prior_scale_multiplier"
    )

    if (
        isinstance(
            raw_multiplier,
            bool
        )
        or not isinstance(
            raw_multiplier,
            (int, float)
        )
    ):
        raise RuntimeError("Posterior cache record has an invalid prior multiplier.")

    if (
        posterior_record.get(
            "candidate_identity"
        )
        != candidate_identity
        or posterior_record.get(
            "outcome"
        )
        != outcome
        or float(
            raw_multiplier
        )
        != float(
            prior_scale_multiplier
        )
    ):
        raise RuntimeError("Posterior cache diagnostics reference the wrong candidate or prior multiplier.")

    if posterior_record.get(
        "fit_status"
    ) != "completed":
        raise RuntimeError("Posterior cache does not contain a completed fit.")

    return (
        posterior_record,
        parameter_records
    )


# Fitting all review-authorized full-data posteriors with cache streaming
def execute_full_posterior_sampling(workflow: dict, acknowledged_pairs: set[tuple[str, str, float]],
                                    cache_directory: Path, software_environment_record: dict,
                                    software_environment_sha256: str, shard_count: int = 1,
                                    shard_index: int = 0, sampling_permitted: bool = True
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    dict
]:
    validate_cache_shard(
        shard_count,
        shard_index
    )

    policy = workflow[
        "validated_inputs"
    ][
        "policy"
    ]

    input_hashes = build_stochastic_input_hashes(
        workflow
    )

    indexed_candidates = component_candidates_by_identity(
        workflow
    )

    posterior_records = []
    sampler_diagnostic_records = []
    posterior_cache_records = []

    execution_accounting = {
        "requested_call_count": 0,
        "completed_call_count": 0,
        "failed_call_count": 0,
        "review_blocked_call_count": 0,
        "reused_cache_count": 0,
        "imported_cache_count": 0,
        "new_cache_count": 0
    }

    for candidate_record in workflow[
        "candidate_records"
    ]:
        if (
            candidate_record[
                "admissibility_status"
            ]
            != "admissible"
        ):
            continue

        candidate_identity = candidate_record[
            "candidate_identity"
        ]

        candidate = indexed_candidates[
            candidate_identity
        ]

        outcome = candidate[
            "collapsed_candidate"
        ][
            "outcome"
        ]

        for prior_scale_multiplier in policy[
            "sampling"
        ][
            "full_data_posterior_scale_multipliers"
        ]:
            execution_accounting[
                "requested_call_count"
            ] += 1

            review_pair = (
                candidate_identity,
                outcome,
                float(
                    prior_scale_multiplier
                )
            )

            if review_pair not in acknowledged_pairs:
                execution_accounting[
                    "review_blocked_call_count"
                ] += 1

                posterior_records.append(
                    {
                        "candidate_identity": (
                            candidate_identity
                        ),
                        "outcome": outcome,
                        "prior_scale_multiplier": (
                            prior_scale_multiplier
                        ),
                        "fit_status": (
                            "not_started_review_unavailable"
                        ),
                        "fit_error": (
                            "No acknowledged prior-predictive "
                            "review record."
                        ),
                        "posterior_draw_count": 0,
                        "posterior_predictive_value_count": 0,
                        "posterior_predictive_nonfinite_count": (
                            None
                        ),
                        "posterior_predictive_positive_finite_fraction": (
                            None
                        ),
                        "sampler_diagnostics_status": (
                            "unavailable"
                        )
                    }
                )

                continue

            cache_identity = build_cache_identity(
                policy = policy,
                policy_sha256 = workflow[
                    "validated_inputs"
                ][
                    "policy_sha256"
                ],
                input_hashes = input_hashes,
                software_environment_sha256 = (
                    software_environment_sha256
                ),
                candidate_identity = (
                    candidate_identity
                ),
                operation = "full_data_posterior",
                prior_scale_multiplier = (
                    prior_scale_multiplier
                ),
                validation_fold_identity = None
            )

            if not cache_job_assigned_to_shard(
                cache_identity = cache_identity,
                shard_count = shard_count,
                shard_index = shard_index
            ):
                continue

            (
                cached_inference_data,
                cached_diagnostics,
                cache_miss_reason,
                loaded_cache_identity,
                imported_cache
            ) = read_cache_entry_with_registered_import(
                cache_directory = cache_directory,
                cache_identity = cache_identity,
                policy = policy
            )

            if (
                cache_miss_reason
                == "CACHE_TERMINAL_FAILURE"
            ):
                terminal_failure_record = (
                    cached_diagnostics[
                        "terminal_failure"
                    ][
                        "posterior_record"
                    ]
                )

                posterior_records.append(
                    terminal_failure_record
                )

                execution_accounting[
                    "failed_call_count"
                ] += 1

                continue

            if (
                cached_inference_data is not None
                and cached_diagnostics is not None
            ):
                try:
                    validate_posterior_inference_data(
                        inference_data = cached_inference_data,
                        candidate = candidate,
                        policy = policy
                    )

                    (
                        posterior_record,
                        parameter_records
                    ) = validate_posterior_cache_diagnostics(
                        cached_diagnostics = (
                            cached_diagnostics
                        ),
                        candidate_identity = (
                            candidate_identity
                        ),
                        outcome = outcome,
                        prior_scale_multiplier = (
                            prior_scale_multiplier
                        )
                    )

                except (
                    RuntimeError,
                    TypeError,
                    ValueError
                ) as validation_error:
                    del cached_inference_data

                    if imported_cache:
                        raise RuntimeError(
                            "Registered immutable source cache entry failed compatibility validation : "
                            f"{loaded_cache_identity['cache_identity_sha256']} : {validation_error}"
                        ) from validation_error

                    quarantine_corrupt_cache_entry(
                        cache_directory = (
                            cache_directory
                        ),
                        cache_identity = loaded_cache_identity,
                        reason_code = (
                            "CACHE_METADATA_SCHEMA_INVALID"
                        )
                    )

                    cached_inference_data = None

                else:
                    posterior_records.append(
                        posterior_record
                    )

                    sampler_diagnostic_records.extend(
                        parameter_records
                    )

                    posterior_cache_records.append(
                        {
                            "candidate_identity": (
                                candidate_identity
                            ),
                            "prior_scale_multiplier": (
                                prior_scale_multiplier
                            ),
                            "cache_identity": (
                            loaded_cache_identity
                            )
                        }
                    )

                    execution_accounting[
                        "completed_call_count"
                    ] += 1

                    execution_accounting[
                        "imported_cache_count" if imported_cache else "reused_cache_count"
                    ] += 1

                    del cached_inference_data

                    continue

            if (
                cache_miss_reason is not None
                and cache_miss_reason
                != "CACHE_METADATA_MISSING"
            ):
                quarantine_corrupt_cache_entry(
                    cache_directory = cache_directory,
                    cache_identity = loaded_cache_identity,
                    reason_code = cache_miss_reason
                )

            if not sampling_permitted:
                raise RuntimeError(
                    f"fit operation cannot sample a missing cache entry : {cache_identity['cache_identity_sha256']}"
                )

            (
                inference_data,
                posterior_record,
                parameter_records
            ) = sample_posterior_candidate(
                candidate = candidate,
                policy = policy,
                prior_scale_multiplier = (
                    prior_scale_multiplier
                )
            )

            posterior_records.append(
                posterior_record
            )

            sampler_diagnostic_records.extend(
                parameter_records
            )

            if inference_data is None:
                write_failure_cache_entry_transactionally(
                    cache_directory = cache_directory,
                    cache_identity = cache_identity,
                    policy = policy,
                    software_environment_record = software_environment_record,
                    software_environment_sha256 = software_environment_sha256,
                    failure_record = {
                        "posterior_record": (
                            posterior_record
                        )
                    }
                )

                execution_accounting[
                    "failed_call_count"
                ] += 1

                continue

            cached_diagnostics = (
                build_posterior_cache_diagnostics(
                    posterior_record = (
                        posterior_record
                    ),
                    parameter_diagnostic_records = (
                        parameter_records
                    )
                )
            )

            write_cache_entry_transactionally(
                cache_directory = cache_directory,
                cache_identity = cache_identity,
                policy = policy,
                software_environment_record = (
                    software_environment_record
                ),
                software_environment_sha256 = (
                    software_environment_sha256
                ),
                inference_data = inference_data,
                diagnostics = cached_diagnostics
            )

            posterior_cache_records.append(
                {
                    "candidate_identity": (
                        candidate_identity
                    ),
                    "prior_scale_multiplier": (
                        prior_scale_multiplier
                    ),
                    "cache_identity": cache_identity
                }
            )

            execution_accounting[
                "completed_call_count"
            ] += 1

            execution_accounting[
                "new_cache_count"
            ] += 1

            del inference_data

    posterior_records.sort(
        key = lambda record: (
            record[
                "candidate_identity"
            ],
            record[
                "prior_scale_multiplier"
            ]
        )
    )

    sampler_diagnostic_records.sort(
        key = lambda record: (
            record.get(
                "candidate_identity",
                ""
            ),
            record.get(
                "prior_scale_multiplier",
                0
            ),
            record.get(
                "parameter_identity",
                ""
            )
        )
    )

    posterior_cache_records.sort(
        key = lambda record: (
            record[
                "candidate_identity"
            ],
            record[
                "prior_scale_multiplier"
            ]
        )
    )

    return (
        posterior_records,
        sampler_diagnostic_records,
        posterior_cache_records,
        execution_accounting
    )


# Flattening chain and draw dimensions while preserving parameter coordinates
def stack_posterior_draws(data_array) -> tuple[np.ndarray, list[str], list[object]]:
    parameter_dimensions = [
        dimension_name
        for dimension_name in data_array.dims
        if dimension_name not in {"chain", "draw"}
    ]

    ordered = data_array.transpose(
        "chain",
        "draw",
        *parameter_dimensions
    )

    values = np.asarray(
        ordered.values,
        dtype = float
    )

    flattened = values.reshape(
        values.shape[0] * values.shape[1],
        *values.shape[2:]
    )

    coordinate_records = []

    if not parameter_dimensions:
        coordinate_records.append(
            ((), None)
        )

    else:
        for parameter_index in np.ndindex(
            *flattened.shape[1:]
        ):
            coordinate_identity = {
                dimension_name: str(
                    ordered.coords[
                        dimension_name
                    ].values[
                        dimension_index
                    ]
                )
                for dimension_name, dimension_index in zip(
                    parameter_dimensions,
                    parameter_index,
                    strict = True
                )
            }

            coordinate_records.append(
                (
                    parameter_index,
                    coordinate_identity
                )
            )

    return (
        flattened,
        parameter_dimensions,
        coordinate_records
    )


# Summarizing every registered posterior quantity without arithmetic predictive moments
def summarize_posterior_candidate(inference_data, candidate: dict, policy: dict,
                                  prior_scale_multiplier: float
) -> list[dict]:
    posterior = inference_data.posterior
    outcome = candidate["collapsed_candidate"]["outcome"]
    cohort = candidate["collapsed_candidate"]["cohort"]
    credible_probability = policy["trajectory_reporting"]["credible_interval_probability"]
    lower_probability = (1 - credible_probability) / 2
    upper_probability = 1 - lower_probability
    records = []

    variable_names = (
        policy["posterior_artifact_contract"]["free_random_variable_names"]
        + policy["posterior_artifact_contract"]["posterior_deterministic_variable_names"]
    )

    for variable_name in variable_names:
        if variable_name not in posterior:
            raise RuntimeError(f"Posterior summary variable is missing : {variable_name}")

        flattened, _, coordinate_records = stack_posterior_draws(
            posterior[variable_name]
        )

        for parameter_index, coordinate_identity in coordinate_records:
            values = (
                flattened
                if not parameter_index
                else flattened[(slice(None), *parameter_index)]
            )

            finite_values = values[np.isfinite(values)]

            if finite_values.size != values.size:
                summary_status = "unavailable"
                summary_reason_code = "SAMPLER_NONFINITE_DRAWS"
                posterior_median = None
                posterior_median_absolute_deviation = None
                posterior_quantiles = None

            else:
                posterior_median = float(
                    np.quantile(
                        finite_values,
                        0.5,
                        method = "linear"
                    )
                )

                posterior_median_absolute_deviation = float(
                    np.quantile(
                        np.abs(
                            finite_values
                            - posterior_median
                        ),
                        0.5,
                        method = "linear"
                    )
                )

                posterior_quantiles = {
                    format(probability, ".3f"): float(
                        np.quantile(
                            finite_values,
                            probability,
                            method = "linear"
                        )
                    )
                    for probability in (
                        lower_probability,
                        0.25,
                        0.5,
                        0.75,
                        upper_probability
                    )
                }

                summary_status = "completed"
                summary_reason_code = "POSTERIOR_SAMPLING_COMPLETED"

            records.append(
                {
                    "candidate_identity": candidate["candidate_identity"],
                    "outcome": outcome,
                    "cohort": cohort,
                    "prior_scale_multiplier": prior_scale_multiplier,
                    "parameter_identity": variable_name,
                    "coordinate_identity": coordinate_identity,
                    "posterior_draw_count": int(values.size),
                    "posterior_median": posterior_median,
                    "posterior_median_absolute_deviation": posterior_median_absolute_deviation,
                    "posterior_quantiles": posterior_quantiles,
                    "summary_status": summary_status,
                    "summary_reason_code": summary_reason_code
                }
            )

    return records


# Calculating equal-cell posterior-predictive statistics
def calculate_equal_cell_statistics(cell_values: np.ndarray) -> dict[str, np.ndarray]:
    if (
        cell_values.ndim != 2
        or cell_values.shape[1] == 0
    ):
        return {}

    medians = np.quantile(
        cell_values,
        0.5,
        axis = 1,
        method = "linear"
    )

    return {
        "equal_cell_weighted_median": medians,
        "equal_cell_weighted_median_absolute_deviation": np.quantile(
            np.abs(
                cell_values
                - medians[:, None]
            ),
            0.5,
            axis = 1,
            method = "linear"
        ),
        "equal_cell_weighted_minimum": np.min(
            cell_values,
            axis = 1
        ),
        "equal_cell_weighted_maximum": np.max(
            cell_values,
            axis = 1
        ),
        **{
            f"equal_cell_weighted_quantile_{probability:.2f}": np.quantile(
                cell_values,
                probability,
                axis = 1,
                method = "linear"
            )
            for probability in (
                0.05,
                0.25,
                0.50,
                0.75,
                0.95
            )
        }
    }


# Building one observed-cell posterior-predictive diagnostic row
def build_posterior_predictive_diagnostic(candidate: dict, prior_scale_multiplier: float,
                                          population_name: str, condition_identity: str | None,
                                          observations: list[dict], predictive_values: np.ndarray,
                                          horizon_bin_index: int | None = None
) -> dict:
    outcome = candidate[
        "collapsed_candidate"
    ][
        "outcome"
    ]

    cohort = candidate[
        "collapsed_candidate"
    ][
        "cohort"
    ]

    observations_by_cell = defaultdict(
        list
    )

    for observation_index, observation in enumerate(
        observations
    ):
        observations_by_cell[
            observation["cell_identity"]
        ].append(
            (
                observation_index,
                observation
            )
        )

    ordered_cells = sorted(
        observations_by_cell
    )

    predictive_draw_count = (
        int(
            predictive_values.shape[0]
        )
        if predictive_values.ndim == 2
        else 0
    )

    finite_mask = np.isfinite(
        predictive_values
    )

    positive_finite_mask = (
        finite_mask
        & (
            predictive_values
            > 0
        )
    )

    observed_cell_values = np.asarray(
        [
            [
                float(
                    observation["outcome_value"]
                )
                for _, observation
                in observations_by_cell[
                    cell_identity
                ]
            ]
            for cell_identity in ordered_cells
        ],
        dtype = object
    )

    observed_cell_medians = np.asarray(
        [
            np.quantile(
                np.asarray(
                    values,
                    dtype = float
                ),
                0.5,
                method = "linear"
            )
            for values in observed_cell_values
        ],
        dtype = float
    )

    predictive_cell_medians = (
        np.column_stack(
            [
                np.quantile(
                    predictive_values[
                        :,
                        [
                            observation_index
                            for observation_index, _
                            in observations_by_cell[
                                cell_identity
                            ]
                        ]
                    ],
                    0.5,
                    axis = 1,
                    method = "linear"
                )
                for cell_identity in ordered_cells
            ]
        )
        if ordered_cells
        else np.empty(
            (
                predictive_draw_count,
                0
            ),
            dtype = float
        )
    )

    observed_statistic_arrays = calculate_equal_cell_statistics(
        observed_cell_medians[
            None,
            :
        ]
    )

    predictive_statistic_arrays = calculate_equal_cell_statistics(
        predictive_cell_medians
    )

    repeated_cells = [
        cell_identity
        for cell_identity in ordered_cells
        if len(
            observations_by_cell[
                cell_identity
            ]
        ) >= 2
    ]

    observed_changes = []
    predictive_changes = []

    for cell_identity in repeated_cells:
        ordered_records = sorted(
            observations_by_cell[
                cell_identity
            ],
            key = lambda item: (
                float(
                    item[1]["horizon"]
                ),
                item[1]["source_identity"]
            )
        )

        first_index, first_observation = (
            ordered_records[0]
        )

        last_index, last_observation = (
            ordered_records[-1]
        )

        observed_changes.append(
            float(
                last_observation[
                    "outcome_value"
                ]
            )
            - float(
                first_observation[
                    "outcome_value"
                ]
            )
        )

        predictive_changes.append(
            predictive_values[
                :,
                last_index
            ]
            - predictive_values[
                :,
                first_index
            ]
        )

    if repeated_cells:
        observed_statistic_arrays[
            "equal_cell_weighted_within_cell_first_to_last_change_median"
        ] = np.asarray(
            [
                np.quantile(
                    np.asarray(
                        observed_changes,
                        dtype = float
                    ),
                    0.5,
                    method = "linear"
                )
            ],
            dtype = float
        )

        predictive_statistic_arrays[
            "equal_cell_weighted_within_cell_first_to_last_change_median"
        ] = np.quantile(
            np.column_stack(
                predictive_changes
            ),
            0.5,
            axis = 1,
            method = "linear"
        )

        within_cell_change_status = (
            "completed"
        )

        within_cell_change_reason_code = (
            "POSTERIOR_PREDICTIVE_COMPLETED"
        )

    else:
        within_cell_change_status = (
            "unavailable"
        )

        within_cell_change_reason_code = (
            "POSTERIOR_PREDICTIVE_WITHIN_CELL_CHANGE_UNAVAILABLE"
        )

    observed_statistics = {
        statistic_name: float(
            statistic_values[0]
        )
        for statistic_name, statistic_values
        in observed_statistic_arrays.items()
    }

    posterior_predictive_statistic_medians = {
        statistic_name: float(
            np.quantile(
                statistic_values,
                0.5,
                method = "linear"
            )
        )
        for statistic_name, statistic_values
        in predictive_statistic_arrays.items()
        if (
            statistic_values.size
            and np.all(
                np.isfinite(
                    statistic_values
                )
            )
        )
    }

    posterior_predictive_statistic_quantiles = {
        statistic_name: {
            f"{probability:.2f}": float(
                np.quantile(
                    statistic_values,
                    probability,
                    method = "linear"
                )
            )
            for probability in (
                0.05,
                0.25,
                0.50,
                0.75,
                0.95
            )
        }
        for statistic_name, statistic_values
        in predictive_statistic_arrays.items()
        if (
            statistic_values.size
            and np.all(
                np.isfinite(
                    statistic_values
                )
            )
        )
    }

    if not observations:
        diagnostic_status = (
            "unavailable"
        )

        diagnostic_reason_code = (
            "POSTERIOR_PREDICTIVE_POPULATION_EMPTY"
        )

    elif (
        predictive_values.ndim != 2
        or predictive_values.shape[1]
        != len(
            observations
        )
        or not np.all(
            positive_finite_mask
        )
    ):
        diagnostic_status = (
            "completed_failed"
        )

        diagnostic_reason_code = (
            "POSTERIOR_PREDICTIVE_FAILED"
        )

    else:
        diagnostic_status = (
            "completed"
        )

        diagnostic_reason_code = (
            "POSTERIOR_PREDICTIVE_COMPLETED"
        )

    return {
        "candidate_identity": candidate[
            "candidate_identity"
        ],
        "outcome": outcome,
        "cohort": cohort,
        "prior_scale_multiplier": (
            prior_scale_multiplier
        ),
        "diagnostic_population": (
            population_name
        ),
        "condition_protocol_identity": (
            condition_identity
        ),
        "horizon_bin_index": (
            horizon_bin_index
        ),
        "observation_count": len(
            observations
        ),
        "cell_count": len(
            ordered_cells
        ),
        "repeated_cell_count": len(
            repeated_cells
        ),
        "predictive_draw_count": (
            predictive_draw_count
        ),
        "predictive_value_count": int(
            predictive_values.size
        ),
        "predictive_nonfinite_count": int(
            (
                ~finite_mask
            ).sum()
        ),
        "predictive_positive_finite_fraction": (
            float(
                positive_finite_mask.mean()
            )
            if predictive_values.size
            else None
        ),
        "observed_statistics": (
            observed_statistics
        ),
        "posterior_predictive_statistic_medians": (
            posterior_predictive_statistic_medians
        ),
        "posterior_predictive_statistic_quantiles": (
            posterior_predictive_statistic_quantiles
        ),
        "within_cell_change_status": (
            within_cell_change_status
        ),
        "within_cell_change_reason_code": (
            within_cell_change_reason_code
        ),
        "diagnostic_status": (
            diagnostic_status
        ),
        "diagnostic_reason_code": (
            diagnostic_reason_code
        )
    }


# Summarizing conditional replications of fitted cells
def summarize_observed_cell_posterior_predictive(inference_data, candidate: dict,
                                                 prior_scale_multiplier: float, policy: dict
) -> list[dict]:
    predictive = np.asarray(
        inference_data.posterior_predictive[
            "posterior_predictive_outcome"
        ].values,
        dtype = float
    ).reshape(
        -1,
        len(
            candidate[
                "included_observations"
            ]
        )
    )

    observations = prepare_pymc_candidate_data(
        candidate
    )[
        "observations"
    ]

    all_indices = list(
        range(
            len(
                observations
            )
        )
    )

    def repeated_indices(indices: list[int]) -> list[int]:
        counts = Counter(
            observations[
                index
            ][
                "cell_identity"
            ]
            for index in indices
        )

        return [
            index
            for index in indices
            if counts[
                observations[
                    index
                ][
                    "cell_identity"
                ]
            ] >= 2
        ]

    populations = [
        (
            "all_observations",
            None,
            None,
            all_indices
        ),
        (
            "all_repeated_cell_observations",
            None,
            None,
            repeated_indices(
                all_indices
            )
        )
    ]

    support_minimum = float(
        candidate[
            "collapsed_candidate"
        ][
            "support_interval_minimum"
        ]
    )

    support_maximum = float(
        candidate[
            "collapsed_candidate"
        ][
            "support_interval_maximum"
        ]
    )

    bin_count = int(
        policy[
            "predictive_assessment"
        ][
            "horizon_bin_count"
        ]
    )

    boundaries = np.linspace(
        support_minimum,
        support_maximum,
        bin_count + 1
    )

    for condition_identity in sorted(
        candidate[
            "condition_identities"
        ]
    ):
        condition_indices = [
            index
            for index, observation
            in enumerate(
                observations
            )
            if observation[
                "condition_protocol_identity"
            ] == condition_identity
        ]

        populations.extend(
            [
                (
                    "condition_all_observations",
                    condition_identity,
                    None,
                    condition_indices
                ),
                (
                    "condition_repeated_cell_observations",
                    condition_identity,
                    None,
                    repeated_indices(
                        condition_indices
                    )
                )
            ]
        )

        interval_indices = [
            index
            for index in condition_indices
            if observations[
                index
            ][
                "inside_evaluation_interval"
            ]
        ]

        for bin_index in range(
            bin_count
        ):
            if bin_index == (
                bin_count
                - 1
            ):
                indices = [
                    index
                    for index in interval_indices
                    if (
                        boundaries[
                            bin_index
                        ]
                        <= float(
                            observations[
                                index
                            ][
                                "horizon"
                            ]
                        )
                        <= boundaries[
                            bin_index + 1
                        ]
                    )
                ]

            else:
                indices = [
                    index
                    for index in interval_indices
                    if (
                        boundaries[
                            bin_index
                        ]
                        <= float(
                            observations[
                                index
                            ][
                                "horizon"
                            ]
                        )
                        < boundaries[
                            bin_index + 1
                        ]
                    )
                ]

            populations.append(
                (
                    "condition_horizon_bin",
                    condition_identity,
                    bin_index,
                    indices
                )
            )

    records = []

    for (
        population_name,
        condition_identity,
        horizon_bin_index,
        indices
    ) in populations:
        records.append(
            build_posterior_predictive_diagnostic(
                candidate = candidate,
                prior_scale_multiplier = (
                    prior_scale_multiplier
                ),
                population_name = (
                    population_name
                ),
                condition_identity = (
                    condition_identity
                ),
                observations = [
                    observations[
                        index
                    ]
                    for index in indices
                ],
                predictive_values = (
                    predictive[
                        :,
                        indices
                    ]
                    if indices
                    else np.empty(
                        (
                            predictive.shape[0],
                            0
                        ),
                        dtype = float
                    )
                ),
                horizon_bin_index = (
                    horizon_bin_index
                )
            )
        )

    return records


# Sampling one grouped-validation training posterior without reusing held-out cell effects
def sample_grouped_validation_posterior(candidate: dict, training_observations: list[dict],
                                        policy: dict, prior_scale_multiplier: float,
                                        fold_index: int
) -> tuple[object | None, dict, list[dict]]:
    try:
        model, data = build_pymc_model(
            candidate = candidate,
            policy = policy,
            prior_scale_multiplier = prior_scale_multiplier,
            observations = training_observations
        )
        pm, _, _, _ = load_bayesian_runtime()

        with model:
            inference_data = pm.sample(
                draws = policy["sampling"]["draws_per_chain"],
                tune = policy["sampling"]["tuning_draws_per_chain"],
                chains = policy["sampling"]["chains"],
                cores = policy["sampling"]["cores"],
                random_seed = policy["sampling"]["sampler_seed"],
                init = policy["sampling"]["initialization"].replace("_plus_", "+"),
                nuts_sampler = "pymc",
                nuts = {
                    "target_accept": policy["sampling"]["target_accept"],
                    "max_treedepth": policy["sampling"]["maximum_tree_depth"]
                },
                discard_tuned_samples = policy["sampling"]["discard_tuned_samples"],
                compute_convergence_checks = policy["sampling"]["compute_convergence_checks"],
                return_inferencedata = True,
                progressbar = False
            )

        validate_finite_posterior_draws(
            inference_data,
            policy
        )

        diagnostic_rows, diagnostic_summary = calculate_sampler_diagnostics(
            inference_data = inference_data,
            policy = policy
        )

        outcome = candidate["collapsed_candidate"]["outcome"]
        completed_diagnostics = complete_sampler_diagnostic_records(
            diagnostic_rows = diagnostic_rows,
            diagnostic_summary = diagnostic_summary,
            candidate_identity = candidate["candidate_identity"],
            outcome = outcome,
            prior_scale_multiplier = prior_scale_multiplier,
            fit_scope = "grouped_validation",
            fold_index = fold_index
        )

        return (
            inference_data,
            {
                "candidate_identity": candidate["candidate_identity"],
                "outcome": outcome,
                "prior_scale_multiplier": prior_scale_multiplier,
                "fit_status": "completed",
                "fit_error": None,
                "posterior_draw_count": int(
                    policy["sampling"]["chains"]
                    * policy["sampling"]["draws_per_chain"]
                ),
                "training_observation_count": len(data["observations"]),
                "training_horizon_midpoint": data["training_horizon_midpoint"],
                "training_horizon_width": data["training_horizon_width"],
                **diagnostic_summary
            },
            completed_diagnostics
        )

    except Exception as exc:
        return (
            None,
            {
                "candidate_identity": candidate["candidate_identity"],
                "outcome": candidate["collapsed_candidate"]["outcome"],
                "prior_scale_multiplier": prior_scale_multiplier,
                "fit_status": "fit_failed",
                "fit_error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "posterior_draw_count": 0,
                "training_observation_count": len(training_observations),
                "training_horizon_midpoint": None,
                "training_horizon_width": None,
                "sampler_diagnostics_status": "unavailable"
            },
            []
        )


# Drawing an unseen-cell predictive distribution from one training posterior
def generate_held_out_cell_predictions(inference_data, candidate: dict,
                                       training_observations: list[dict],
                                       test_observations: list[dict], policy: dict,
                                       fold_index: int
) -> tuple[list[dict], dict[str, np.ndarray]]:
    training_data = prepare_pymc_candidate_data(
        candidate,
        observations = training_observations
    )
    posterior = inference_data.posterior

    def flattened(variable_name: str) -> np.ndarray:
        values, _, _ = stack_posterior_draws(
            posterior[variable_name]
        )
        return values

    condition_intercepts = flattened("condition_intercept")
    condition_slopes = flattened("condition_horizon_slope")
    cholesky = flattened("cell_random_effect_cholesky")
    residual_scale = flattened("residual_standard_deviation").reshape(-1)
    degrees_of_freedom = flattened("student_t_degrees_of_freedom").reshape(-1)
    draw_count = condition_intercepts.shape[0]

    if not (
        condition_slopes.shape[0]
        == cholesky.shape[0]
        == residual_scale.shape[0]
        == degrees_of_freedom.shape[0]
        == draw_count
    ):
        raise RuntimeError("Grouped-validation posterior draw dimensions are inconsistent.")

    condition_index = {
        identity: index
        for index, identity in enumerate(training_data["condition_identities"])
    }
    test_observations = sorted(
        test_observations,
        key = lambda record: (
            record["cell_identity"],
            record["condition_protocol_identity"],
            record["horizon"],
            record["source_identity"]
        )
    )
    cells = sorted({record["cell_identity"] for record in test_observations})
    random_generator = np.random.default_rng(
        policy["sampling"]["posterior_predictive_seed"]
        + fold_index
    )
    cell_effects = {}

    for cell_identity in cells:
        standard_effects = random_generator.normal(
            size = (draw_count, 2)
        )
        cell_effects[cell_identity] = np.einsum(
            "dj,dkj->dk",
            standard_effects,
            cholesky
        )

    prediction_records = []
    predictive_by_source = {}
    lower_probability = policy["predictive_assessment"]["predictive_interval_lower_probability"]
    upper_probability = policy["predictive_assessment"]["predictive_interval_upper_probability"]

    for observation in test_observations:
        condition_identity = observation["condition_protocol_identity"]

        if condition_identity not in condition_index:
            raise RuntimeError("Held-out observation condition is absent from its training posterior.")

        scaled_horizon = (
            float(observation["horizon"])
            - training_data["training_horizon_midpoint"]
        ) / training_data["training_horizon_width"]
        index = condition_index[condition_identity]
        cell_random_effect = cell_effects[observation["cell_identity"]]
        location = (
            condition_intercepts[:, index]
            + condition_slopes[:, index] * scaled_horizon
            + cell_random_effect[:, 0]
            + cell_random_effect[:, 1] * scaled_horizon
        )
        noise = random_generator.standard_t(
            degrees_of_freedom
        ) * residual_scale
        log_prediction = location + noise

        with np.errstate(over = "ignore", under = "ignore", invalid = "ignore"):
            predictive_values = np.exp(log_prediction)

        predictive_by_source[observation["source_identity"]] = predictive_values
        finite_positive = np.isfinite(predictive_values) & (predictive_values > 0)

        if not np.all(finite_positive):
            prediction_status = "unavailable"
            prediction_reason_code = "GROUPED_VALIDATION_INTERVAL_UNAVAILABLE"
            median = None
            lower = None
            upper = None

        else:
            prediction_status = "completed"
            prediction_reason_code = "GROUPED_VALIDATION_EXECUTION_COMPLETED"
            median = float(np.quantile(predictive_values, 0.5, method = "linear"))
            lower = float(np.quantile(predictive_values, lower_probability, method = "linear"))
            upper = float(np.quantile(predictive_values, upper_probability, method = "linear"))

        prediction_records.append(
            {
                "candidate_identity": candidate["candidate_identity"],
                "fold_index": fold_index,
                "cell_identity": observation["cell_identity"],
                "source_identity": observation["source_identity"],
                "condition_protocol_identity": condition_identity,
                "horizon": observation["horizon"],
                "observed_outcome": observation["outcome_value"],
                "posterior_predictive_median": median,
                "posterior_predictive_interval_lower": lower,
                "posterior_predictive_interval_upper": upper,
                "prediction_status": prediction_status,
                "prediction_reason_code": prediction_reason_code,
                "estimate_label": "predictive"
            }
        )

    return prediction_records, predictive_by_source


# Reducing observation losses so each held-out cell contributes equal total weight
def equal_cell_mean(values_by_cell: dict[str, list[float]]) -> float:
    if not values_by_cell or any(not values for values in values_by_cell.values()):
        raise RuntimeError("Equal-cell reduction requires at least one value for every cell.")

    return float(
        np.mean(
            [
                np.mean(
                    np.asarray(values, dtype = float)
                )
                for values in values_by_cell.values()
            ]
        )
    )


# Calculating registered grouped unseen-cell validation metrics
def summarize_grouped_validation(candidate: dict, prediction_records: list[dict],
                                 training_observations_by_fold: dict[int, list[dict]]
) -> tuple[dict, list[dict]]:
    completed = [
        record
        for record in prediction_records
        if record["prediction_status"] == "completed"
    ]
    expected = len(prediction_records)
    outcome = candidate["collapsed_candidate"]["outcome"]
    cohort = candidate["collapsed_candidate"]["cohort"]

    if not completed or len(completed) != expected:
        return (
            {
                "candidate_identity": candidate["candidate_identity"],
                "outcome": outcome,
                "cohort": cohort,
                "fold_count": len(training_observations_by_fold),
                "test_cell_count": len({record["cell_identity"] for record in prediction_records}),
                "test_observation_count": expected,
                "equal_cell_weighted_mean_absolute_error": None,
                "equal_cell_weighted_root_mean_squared_error": None,
                "equal_cell_weighted_median_error": None,
                "posterior_predictive_interval_coverage": None,
                "mean_posterior_predictive_interval_width": None,
                "equal_cell_weighted_predictive_r_squared": None,
                "benchmark_variation_status": "unavailable",
                "validation_status": "unavailable",
                "validation_reason_code": "GROUPED_VALIDATION_INTERVAL_UNAVAILABLE",
                "estimate_label": "predictive"
            },
            []
        )

    absolute_errors = defaultdict(list)
    squared_errors = defaultdict(list)
    signed_errors = defaultdict(list)
    coverage_values = defaultdict(list)
    interval_widths = defaultdict(list)
    benchmark_squared_errors = defaultdict(list)

    for record in completed:
        cell_identity = record["cell_identity"]
        observed = float(record["observed_outcome"])
        predicted = float(record["posterior_predictive_median"])
        error = predicted - observed
        absolute_errors[cell_identity].append(abs(error))
        squared_errors[cell_identity].append(error ** 2)
        signed_errors[cell_identity].append(error)
        coverage_values[cell_identity].append(
            float(
                float(record["posterior_predictive_interval_lower"])
                <= observed
                <= float(record["posterior_predictive_interval_upper"])
            )
        )
        interval_widths[cell_identity].append(
            float(record["posterior_predictive_interval_upper"])
            - float(record["posterior_predictive_interval_lower"])
        )

        fold_index = int(record["fold_index"])
        condition_identity = record["condition_protocol_identity"]
        training_records = [
            training_record
            for training_record in training_observations_by_fold[fold_index]
            if training_record["condition_protocol_identity"] == condition_identity
        ]
        training_values_by_cell = defaultdict(list)

        for training_record in training_records:
            training_values_by_cell[training_record["cell_identity"]].append(
                float(training_record["outcome_value"])
            )

        benchmark = equal_cell_mean(training_values_by_cell)
        benchmark_squared_errors[cell_identity].append((benchmark - observed) ** 2)

    model_squared_error = equal_cell_mean(squared_errors)
    benchmark_squared_error = equal_cell_mean(benchmark_squared_errors)

    if benchmark_squared_error <= 0 or not math.isfinite(benchmark_squared_error):
        predictive_r_squared = None
        benchmark_status = "unavailable"
    else:
        predictive_r_squared = 1 - model_squared_error / benchmark_squared_error
        benchmark_status = "completed"

    metric_record = {
        "candidate_identity": candidate["candidate_identity"],
        "outcome": outcome,
        "cohort": cohort,
        "fold_count": len(training_observations_by_fold),
        "test_cell_count": len(absolute_errors),
        "test_observation_count": len(completed),
        "equal_cell_weighted_mean_absolute_error": equal_cell_mean(absolute_errors),
        "equal_cell_weighted_root_mean_squared_error": math.sqrt(model_squared_error),
        "equal_cell_weighted_median_error": float(
            np.quantile(
                np.asarray(
                    [
                        np.quantile(values, 0.5, method = "linear")
                        for values in signed_errors.values()
                    ],
                    dtype = float
                ),
                0.5,
                method = "linear"
            )
        ),
        "posterior_predictive_interval_coverage": equal_cell_mean(coverage_values),
        "mean_posterior_predictive_interval_width": equal_cell_mean(interval_widths),
        "equal_cell_weighted_predictive_r_squared": predictive_r_squared,
        "benchmark_variation_status": benchmark_status,
        "validation_status": (
            "completed"
            if predictive_r_squared is not None
            else "unavailable"
        ),
        "validation_reason_code": (
            "GROUPED_VALIDATION_EXECUTION_COMPLETED"
            if predictive_r_squared is not None
            else "GROUPED_VALIDATION_INTERVAL_UNAVAILABLE"
        ),
        "estimate_label": "predictive"
    }

    calibration_records = []

    for condition_identity in sorted({record["condition_protocol_identity"] for record in completed}):
        condition_records = [
            record
            for record in completed
            if record["condition_protocol_identity"] == condition_identity
        ]
        observed_by_cell = defaultdict(list)
        predicted_by_cell = defaultdict(list)
        error_by_cell = defaultdict(list)
        coverage_by_cell = defaultdict(list)
        absolute_error_by_cell = defaultdict(list)
        squared_error_by_cell = defaultdict(list)
        interval_width_by_cell = defaultdict(list)

        for record in condition_records:
            cell_identity = record["cell_identity"]
            observed = float(record["observed_outcome"])
            predicted = float(record["posterior_predictive_median"])
            observed_by_cell[cell_identity].append(observed)
            predicted_by_cell[cell_identity].append(predicted)
            error_by_cell[cell_identity].append(predicted - observed)
            absolute_error_by_cell[cell_identity].append(abs(predicted - observed))
            squared_error_by_cell[cell_identity].append((predicted - observed) ** 2)
            coverage_by_cell[cell_identity].append(
                float(
                    float(record["posterior_predictive_interval_lower"])
                    <= observed
                    <= float(record["posterior_predictive_interval_upper"])
                )
            )
            interval_width_by_cell[cell_identity].append(
                float(record["posterior_predictive_interval_upper"])
                - float(record["posterior_predictive_interval_lower"])
            )

        def median_across_cell_medians(values_by_cell: dict[str, list[float]]) -> float:
            return float(
                np.quantile(
                    np.asarray(
                        [
                            np.quantile(values, 0.5, method = "linear")
                            for values in values_by_cell.values()
                        ],
                        dtype = float
                    ),
                    0.5,
                    method = "linear"
                )
            )

        calibration_records.append(
            {
                "candidate_identity": candidate["candidate_identity"],
                "condition_protocol_identity": condition_identity,
                "independent_cell_count": len(observed_by_cell),
                "observation_count": len(condition_records),
                "equal_cell_weighted_median_observed": median_across_cell_medians(observed_by_cell),
                "equal_cell_weighted_median_predicted": median_across_cell_medians(predicted_by_cell),
                "equal_cell_weighted_median_error": median_across_cell_medians(error_by_cell),
                "equal_cell_weighted_mean_absolute_error": equal_cell_mean(absolute_error_by_cell),
                "equal_cell_weighted_root_mean_squared_error": math.sqrt(equal_cell_mean(squared_error_by_cell)),
                "posterior_predictive_interval_coverage": equal_cell_mean(coverage_by_cell),
                "mean_posterior_predictive_interval_width": equal_cell_mean(interval_width_by_cell),
                "calibration_status": "completed",
                "calibration_reason_code": "GROUPED_VALIDATION_EXECUTION_COMPLETED",
                "estimate_label": "predictive"
            }
        )

    return metric_record, calibration_records


# Executing every feasible grouped held-out-cell validation fold with cache streaming
def execute_grouped_validation(workflow: dict, cache_directory: Path,
                               software_environment_record: dict, software_environment_sha256: str,
                               shard_count: int = 1, shard_index: int = 0,
                               sampling_permitted: bool = True
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    validate_cache_shard(
        shard_count,
        shard_index
    )

    policy = workflow["validated_inputs"]["policy"]
    input_hashes = build_stochastic_input_hashes(workflow)
    indexed_candidates = component_candidates_by_identity(workflow)

    candidate_records = {
        record["candidate_identity"]: record
        for record in workflow["candidate_records"]
    }

    fold_records_by_candidate = defaultdict(list)

    for fold_record in workflow["fold_records"]:
        fold_records_by_candidate[fold_record["candidate_identity"]].append(fold_record)

    prediction_records = []
    metric_records = []
    calibration_records = []
    sampler_records = []
    accounting = {
        "requested_call_count": 0,
        "completed_call_count": 0,
        "failed_call_count": 0,
        "reused_cache_count": 0,
        "imported_cache_count": 0,
        "new_cache_count": 0
    }

    for candidate_identity in sorted(indexed_candidates):
        registry_record = candidate_records[candidate_identity]

        if registry_record["grouped_validation_status"] != "available":
            continue

        candidate = indexed_candidates[candidate_identity]
        all_candidate_predictions = []
        training_observations_by_fold = {}
        candidate_failed = False

        for fold_record in sorted(
            fold_records_by_candidate[candidate_identity],
            key = lambda record: int(record["fold_index"])
        ):
            if fold_record["fold_status"] != "available":
                candidate_failed = True
                continue

            fold_index = int(fold_record["fold_index"])
            test_cells = set(fold_record["test_cell_identities"])

            training_observations = [
                observation
                for observation in candidate["included_observations"]
                if observation["cell_identity"] not in test_cells
            ]
            test_observations = [
                observation
                for observation in candidate["included_observations"]
                if (
                    observation["cell_identity"] in test_cells
                    and observation["inside_evaluation_interval"]
                )
            ]

            training_observations_by_fold[fold_index] = training_observations

            prior_scale_multiplier = policy["priors"]["grouped_validation_scale_multiplier"]

            accounting["requested_call_count"] += 1

            fold_identity = f"{candidate_identity}:{fold_index}"

            cache_identity = build_cache_identity(
                policy = policy,
                policy_sha256 = workflow["validated_inputs"]["policy_sha256"],
                input_hashes = input_hashes,
                software_environment_sha256 = software_environment_sha256,
                candidate_identity = candidate_identity,
                operation = "grouped_validation_posterior",
                prior_scale_multiplier = prior_scale_multiplier,
                validation_fold_identity = fold_identity
            )

            if not cache_job_assigned_to_shard(
                cache_identity = cache_identity,
                shard_count = shard_count,
                shard_index = shard_index
            ):
                continue

            (
                inference_data,
                cached_diagnostics,
                cache_miss_reason,
                loaded_cache_identity,
                imported_cache
            ) = read_cache_entry_with_registered_import(
                cache_directory = cache_directory,
                cache_identity = cache_identity,
                policy = policy
            )

            if (
                cache_miss_reason
                == "CACHE_TERMINAL_FAILURE"
            ):
                accounting["failed_call_count"] += 1
                candidate_failed = True
                continue

            posterior_record = None
            parameter_records = []

            if inference_data is not None and cached_diagnostics is not None:
                try:
                    posterior_record, parameter_records = validate_posterior_cache_diagnostics(
                        cached_diagnostics = cached_diagnostics,
                        candidate_identity = candidate_identity,
                        outcome = candidate["collapsed_candidate"]["outcome"],
                        prior_scale_multiplier = prior_scale_multiplier
                    )

                    validate_posterior_inference_data(
                        inference_data = inference_data,
                        candidate = candidate,
                        policy = policy,
                        observations = training_observations,
                        require_posterior_predictive = False
                    )
                except (RuntimeError, TypeError, ValueError) as validation_error:
                    del inference_data
                    inference_data = None

                    if imported_cache:
                        raise RuntimeError(
                            "Registered immutable source cache entry failed compatibility validation : "
                            f"{loaded_cache_identity['cache_identity_sha256']} : {validation_error}"
                        ) from validation_error

                    quarantine_corrupt_cache_entry(
                        cache_directory = cache_directory,
                        cache_identity = loaded_cache_identity,
                        reason_code = "CACHE_METADATA_SCHEMA_INVALID"
                    )
                else:
                    accounting[
                        "imported_cache_count" if imported_cache else "reused_cache_count"
                    ] += 1

            if inference_data is None:
                if cache_miss_reason not in {None, "CACHE_METADATA_MISSING"}:
                    quarantine_corrupt_cache_entry(
                        cache_directory = cache_directory,
                        cache_identity = loaded_cache_identity,
                        reason_code = cache_miss_reason
                    )

                if not sampling_permitted:
                    raise RuntimeError(
                        "fit operation cannot sample a missing grouped-validation cache entry : "
                        f"{cache_identity['cache_identity_sha256']}"
                    )

                inference_data, posterior_record, parameter_records = sample_grouped_validation_posterior(
                    candidate = candidate,
                    training_observations = training_observations,
                    policy = policy,
                    prior_scale_multiplier = prior_scale_multiplier,
                    fold_index = fold_index
                )

                if inference_data is not None:
                    write_cache_entry_transactionally(
                        cache_directory = cache_directory,
                        cache_identity = cache_identity,
                        policy = policy,
                        software_environment_record = software_environment_record,
                        software_environment_sha256 = software_environment_sha256,
                        inference_data = inference_data,
                        diagnostics = build_posterior_cache_diagnostics(
                            posterior_record,
                            parameter_records
                        )
                    )
                    accounting["new_cache_count"] += 1

            if inference_data is None:
                write_failure_cache_entry_transactionally(
                    cache_directory = cache_directory,
                    cache_identity = cache_identity,
                    policy = policy,
                    software_environment_record = software_environment_record,
                    software_environment_sha256 = software_environment_sha256,
                    failure_record = {
                        "posterior_record": (
                            posterior_record
                        )
                    }
                )

                accounting["failed_call_count"] += 1
                candidate_failed = True
                continue

            if posterior_record["sampler_diagnostics_status"] != "completed_passed":
                accounting["completed_call_count"] += 1
                candidate_failed = True
                sampler_records.extend(parameter_records)
                del inference_data
                continue

            fold_predictions, _ = generate_held_out_cell_predictions(
                inference_data = inference_data,
                candidate = candidate,
                training_observations = training_observations,
                test_observations = test_observations,
                policy = policy,
                fold_index = fold_index
            )
            all_candidate_predictions.extend(fold_predictions)
            sampler_records.extend(parameter_records)
            accounting["completed_call_count"] += 1
            del inference_data

        prediction_records.extend(all_candidate_predictions)

        if candidate_failed:
            continue

        metric_record, candidate_calibration = summarize_grouped_validation(
            candidate = candidate,
            prediction_records = all_candidate_predictions,
            training_observations_by_fold = training_observations_by_fold
        )
        metric_records.append(metric_record)
        calibration_records.extend(candidate_calibration)

    prediction_records.sort(key = lambda record: (
        record["candidate_identity"],
        record["fold_index"],
        record["cell_identity"],
        record["source_identity"]
    ))
    metric_records.sort(key = lambda record: record["candidate_identity"])
    calibration_records.sort(key = lambda record: (
        record["candidate_identity"],
        record["condition_protocol_identity"]
    ))
    sampler_records.sort(key = lambda record: (
        record["candidate_identity"],
        record["fit_scope"],
        -1 if record["fold_index"] is None else record["fold_index"],
        record["parameter_identity"]
    ))

    return prediction_records, metric_records, calibration_records, sampler_records, accounting


# Loading one completed inference artifact from its exact cache identity
def load_completed_cached_inference(cache_directory: Path, cache_identity: dict,
                                    policy: dict
):
    inference_data, diagnostics, cache_miss_reason = read_validated_cache_entry(
        cache_directory = cache_directory,
        cache_identity = cache_identity,
        policy = policy
    )

    if inference_data is None or diagnostics is None:
        raise RuntimeError(f"A completed posterior cache entry could not be reloaded : {cache_miss_reason}")

    return inference_data, diagnostics


# Producing streamed summaries from all completed full-data posterior caches
def summarize_full_posterior_caches(workflow: dict, cache_directory: Path,
                                    posterior_cache_records: list[dict]
) -> tuple[list[dict], list[dict]]:
    policy = workflow["validated_inputs"]["policy"]
    indexed_candidates = component_candidates_by_identity(workflow)
    posterior_summaries = []
    predictive_diagnostics = []

    for cache_record in posterior_cache_records:
        candidate = indexed_candidates[cache_record["candidate_identity"]]
        inference_data, _ = load_completed_cached_inference(
            cache_directory = cache_directory,
            cache_identity = cache_record["cache_identity"],
            policy = policy
        )
        validate_posterior_inference_data(inference_data, candidate, policy)
        posterior_summaries.extend(
            summarize_posterior_candidate(
                inference_data = inference_data,
                candidate = candidate,
                policy = policy,
                prior_scale_multiplier = cache_record["prior_scale_multiplier"]
            )
        )
        predictive_diagnostics.extend(
            summarize_observed_cell_posterior_predictive(
                inference_data = inference_data,
                candidate = candidate,
                prior_scale_multiplier = cache_record["prior_scale_multiplier"],
                policy = policy
            )
        )
        del inference_data

    posterior_summaries.sort(key = lambda record: (
        record["candidate_identity"],
        record["prior_scale_multiplier"],
        record["parameter_identity"],
        canonical_json(record["coordinate_identity"])
        if record["coordinate_identity"] is not None
        else ""
    ))
    predictive_diagnostics.sort(key = lambda record: (
        record["candidate_identity"],
        record["prior_scale_multiplier"],
        record["diagnostic_population"],
        record["condition_protocol_identity"] or ""
    ))
    return posterior_summaries, predictive_diagnostics


# Qualifying reference posteriors under every registered diagnostic requirement
def qualify_reference_posteriors(workflow: dict, posterior_records: list[dict],
                                 predictive_diagnostics: list[dict], validation_metrics: list[dict],
                                 calibration_records: list[dict]
) -> dict[str, dict]:
    policy = workflow["validated_inputs"]["policy"]
    reference_multiplier = policy["sampling"]["reference_posterior_multiplier"]
    registry = {record["candidate_identity"]: record for record in workflow["candidate_records"]}
    posterior_by_candidate = {
        record["candidate_identity"]: record
        for record in posterior_records
        if float(record["prior_scale_multiplier"]) == float(reference_multiplier)
    }
    predictive_by_candidate = defaultdict(list)
    metric_by_candidate = {record["candidate_identity"]: record for record in validation_metrics}
    calibration_by_candidate = defaultdict(list)

    for record in predictive_diagnostics:
        if float(record["prior_scale_multiplier"]) == float(reference_multiplier):
            predictive_by_candidate[record["candidate_identity"]].append(record)

    for record in calibration_records:
        calibration_by_candidate[record["candidate_identity"]].append(record)

    registered_statistics = set(
        policy["predictive_assessment"]["posterior_predictive_statistics"]
    )

    within_cell_statistic = (
        "equal_cell_weighted_within_cell_first_to_last_change_median"
    )

    base_statistics = (
        registered_statistics
        - {
            within_cell_statistic
        }
    )

    quantile_keys = {
        "0.05",
        "0.25",
        "0.50",
        "0.75",
        "0.95"
    }

    def finite_numeric_mapping(value: dict) -> bool:
        return all(
            isinstance(
                item,
                (
                    int,
                    float
                )
            )
            and not isinstance(
                item,
                bool
            )
            and math.isfinite(
                float(
                    item
                )
            )
            for item in value.values()
        )

    def diagnostic_record_is_acceptable(record: dict) -> bool:
        observation_count = int(
            record[
                "observation_count"
            ]
        )

        if observation_count == 0:
            return (
                record[
                    "diagnostic_status"
                ]
                == "unavailable"
                and record[
                    "diagnostic_reason_code"
                ]
                == "POSTERIOR_PREDICTIVE_POPULATION_EMPTY"
            )

        if (
            record[
                "diagnostic_status"
            ]
            != "completed"
            or int(
                record[
                    "predictive_nonfinite_count"
                ]
            )
            != 0
            or float(
                record[
                    "predictive_positive_finite_fraction"
                ]
            )
            != 1.0
        ):
            return False

        expected_statistics = set(
            base_statistics
        )

        if int(
            record[
                "repeated_cell_count"
            ]
        ) > 0:
            expected_statistics.add(
                within_cell_statistic
            )

            if (
                record[
                    "within_cell_change_status"
                ]
                != "completed"
            ):
                return False

        elif (
            record[
                "within_cell_change_status"
            ]
            != "unavailable"
            or record[
                "within_cell_change_reason_code"
            ]
            != "POSTERIOR_PREDICTIVE_WITHIN_CELL_CHANGE_UNAVAILABLE"
        ):
            return False

        observed = record[
            "observed_statistics"
        ]

        predictive_medians = record[
            "posterior_predictive_statistic_medians"
        ]

        predictive_quantiles = record[
            "posterior_predictive_statistic_quantiles"
        ]

        if not (
            set(
                observed
            )
            == expected_statistics
            == set(
                predictive_medians
            )
            == set(
                predictive_quantiles
            )
        ):
            return False

        if not (
            finite_numeric_mapping(
                observed
            )
            and finite_numeric_mapping(
                predictive_medians
            )
        ):
            return False

        return all(
            set(
                statistic_quantiles
            )
            == quantile_keys
            and finite_numeric_mapping(
                statistic_quantiles
            )
            for statistic_quantiles
            in predictive_quantiles.values()
        )

    qualification = {}

    for candidate_identity, registry_record in registry.items():
        if registry_record["admissibility_status"] != "admissible":
            qualification[candidate_identity] = {
                "status": "not_available",
                "reason_code": "ANALYTICAL_QUALIFICATION_NOT_AVAILABLE"
            }
            continue

        posterior_record = posterior_by_candidate.get(candidate_identity)
        metric_record = metric_by_candidate.get(candidate_identity)
        predictive_records = predictive_by_candidate.get(candidate_identity, [])
        candidate_calibration = calibration_by_candidate.get(candidate_identity, [])
        condition_identities = json.loads(
            registry_record["sorted_condition_protocol_identities"]
        ) if isinstance(
            registry_record["sorted_condition_protocol_identities"],
            str
        ) else registry_record["sorted_condition_protocol_identities"]

        expected_predictive_inventory = {
            (
                "all_observations",
                None,
                None
            ),
            (
                "all_repeated_cell_observations",
                None,
                None
            )
        }

        for condition_identity in condition_identities:
            expected_predictive_inventory.update(
                {
                    (
                        "condition_all_observations",
                        condition_identity,
                        None
                    ),
                    (
                        "condition_repeated_cell_observations",
                        condition_identity,
                        None
                    )
                }
            )

            for horizon_bin_index in range(
                policy[
                    "predictive_assessment"
                ][
                    "horizon_bin_count"
                ]
            ):
                expected_predictive_inventory.add(
                    (
                        "condition_horizon_bin",
                        condition_identity,
                        horizon_bin_index
                    )
                )

        observed_predictive_inventory = {
            (
                record["diagnostic_population"],
                record["condition_protocol_identity"],
                record["horizon_bin_index"]
            )
            for record in predictive_records
        }

        predictive_inventory_complete = (
            observed_predictive_inventory
            == expected_predictive_inventory
            and len(predictive_records) == len(expected_predictive_inventory)
        )

        failed_gates = []

        if posterior_record is None:
            failed_gates.append("reference_posterior_unavailable")
        elif posterior_record["fit_status"] != "completed":
            failed_gates.append("reference_posterior_not_completed")
        elif posterior_record["sampler_diagnostics_status"] != "completed_passed":
            failed_gates.append("reference_sampler_diagnostics_failed")

        if not predictive_inventory_complete:
            failed_gates.append("posterior_predictive_inventory_incomplete")
        elif not all(
            diagnostic_record_is_acceptable(record)
            for record in predictive_records
        ):
            failed_gates.append("posterior_predictive_diagnostics_failed")

        if metric_record is None or metric_record["validation_status"] != "completed":
            failed_gates.append("grouped_validation_not_completed")
        else:
            predictive_r_squared = metric_record["equal_cell_weighted_predictive_r_squared"]

            if predictive_r_squared is None:
                failed_gates.append("predictive_r_squared_unavailable")
            elif float(predictive_r_squared) < policy["predictive_assessment"]["analytical_qualification"]["minimum_predictive_r_squared"]:
                failed_gates.append("predictive_benchmark_not_met")

        if len(candidate_calibration) != int(registry_record["condition_count"]):
            failed_gates.append("condition_calibration_incomplete")
        elif not all(record["calibration_status"] == "completed" for record in candidate_calibration):
            failed_gates.append("condition_calibration_diagnostics_failed")

        available = not failed_gates

        qualification[candidate_identity] = {
            "status": "available" if available else "diagnostic_only",
            "reason_code": (
                "ANALYTICAL_QUALIFICATION_AVAILABLE"
                if available
                else "ANALYTICAL_QUALIFICATION_DIAGNOSTIC_ONLY"
            ),
            "failed_gates": failed_gates
        }

    return qualification


# Drawing latent and new-cell trajectories for analytically qualified reference posteriors
def summarize_candidate_trajectories(inference_data, candidate: dict, policy: dict,
                                     qualification: dict
) -> list[dict]:
    if qualification["status"] != "available":
        return []

    data = prepare_pymc_candidate_data(candidate)
    posterior = inference_data.posterior
    condition_intercepts, _, _ = stack_posterior_draws(posterior["condition_intercept"])
    condition_slopes, _, _ = stack_posterior_draws(posterior["condition_horizon_slope"])
    cholesky, _, _ = stack_posterior_draws(posterior["cell_random_effect_cholesky"])
    residual_scale, _, _ = stack_posterior_draws(posterior["residual_standard_deviation"])
    degrees_of_freedom, _, _ = stack_posterior_draws(posterior["student_t_degrees_of_freedom"])
    residual_scale = residual_scale.reshape(-1)
    degrees_of_freedom = degrees_of_freedom.reshape(-1)
    grid = build_trajectory_grid(candidate, policy)
    probability = policy["trajectory_reporting"]["credible_interval_probability"]
    lower_probability = (1 - probability) / 2
    upper_probability = 1 - lower_probability
    records = []

    for condition_index, condition_identity in enumerate(data["condition_identities"]):
        seed = deterministic_prediction_seed(
            candidate_identity = candidate["candidate_identity"],
            condition_identity = condition_identity,
            registered_seed = policy["trajectory_reporting"]["new_cell_predictive_seed"]
        )
        random_generator = np.random.default_rng(seed)
        standard_effects = random_generator.normal(
            size = (condition_intercepts.shape[0], 2)
        )
        random_effects = np.einsum("dj,dkj->dk", standard_effects, cholesky)

        for horizon in grid:
            scaled_horizon = (
                float(horizon) - data["training_horizon_midpoint"]
            ) / data["training_horizon_width"]
            latent_log = (
                condition_intercepts[:, condition_index]
                + condition_slopes[:, condition_index] * scaled_horizon
            )
            latent = np.exp(latent_log)
            predictive_log = (
                latent_log
                + random_effects[:, 0]
                + random_effects[:, 1] * scaled_horizon
                + random_generator.standard_t(degrees_of_freedom) * residual_scale
            )

            with np.errstate(over = "ignore", under = "ignore", invalid = "ignore"):
                predictive = np.exp(predictive_log)

            if not np.all(np.isfinite(latent)) or not np.all(np.isfinite(predictive)) or np.any(predictive <= 0):
                for trajectory_target in (
                    "latent_reference_cell",
                    "new_cell_predictive"
                ):
                    records.append({
                        "candidate_identity": candidate["candidate_identity"],
                        "outcome": candidate["collapsed_candidate"]["outcome"],
                        "cohort": candidate["collapsed_candidate"]["cohort"],
                        "horizon_axis": candidate["collapsed_candidate"]["horizon_axis"],
                        "condition_protocol_identity": condition_identity,
                        "horizon": float(horizon),
                        "trajectory_target": trajectory_target,
                        "posterior_mean": None,
                        "posterior_median": None,
                        "credible_interval_lower": None,
                        "credible_interval_upper": None,
                        "predictive_interval_lower": None,
                        "predictive_interval_upper": None,
                        "analytical_status": "unavailable",
                        "analytical_reason_code": "NONFINITE_TRAJECTORY_DERIVATION",
                        "estimate_label": policy["trajectory_reporting"]["estimate_label"],
                        "causal_interpretation": policy["trajectory_reporting"]["causal_label"]
                    })
                continue

            common = {
                "candidate_identity": candidate["candidate_identity"],
                "outcome": candidate["collapsed_candidate"]["outcome"],
                "cohort": candidate["collapsed_candidate"]["cohort"],
                "horizon_axis": candidate["collapsed_candidate"]["horizon_axis"],
                "condition_protocol_identity": condition_identity,
                "horizon": float(horizon),
                "analytical_status": "available",
                "analytical_reason_code": "TRAJECTORY_REPORTING_COMPLETED",
                "estimate_label": policy["trajectory_reporting"]["estimate_label"],
                "causal_interpretation": policy["trajectory_reporting"]["causal_label"]
            }
            records.append({
                **common,
                "trajectory_target": "latent_reference_cell",
                "posterior_mean": float(np.mean(latent)),
                "posterior_median": float(np.quantile(latent, 0.5, method = "linear")),
                "credible_interval_lower": float(np.quantile(latent, lower_probability, method = "linear")),
                "credible_interval_upper": float(np.quantile(latent, upper_probability, method = "linear")),
                "predictive_interval_lower": None,
                "predictive_interval_upper": None
            })
            records.append({
                **common,
                "trajectory_target": "new_cell_predictive",
                "posterior_mean": None,
                "posterior_median": float(np.quantile(predictive, 0.5, method = "linear")),
                "credible_interval_lower": None,
                "credible_interval_upper": None,
                "predictive_interval_lower": float(np.quantile(predictive, lower_probability, method = "linear")),
                "predictive_interval_upper": float(np.quantile(predictive, upper_probability, method = "linear"))
            })

    return records


# Reporting authorized continuous latent-trajectory differences without point-estimand claims
def summarize_candidate_differences(inference_data, candidate: dict,
                                    workflow: dict, qualification: dict
) -> list[dict]:
    policy = workflow["validated_inputs"]["policy"]
    identification_records = workflow["validated_inputs"]["identification_tables"]["estimand_identification"]
    support_records = workflow["validated_inputs"]["design_tables"]["estimand_support"]
    data = None
    condition_index = None
    condition_intercepts = None
    condition_slopes = None

    if qualification["status"] == "available":
        data = prepare_pymc_candidate_data(candidate)
        condition_index = {
            identity: index
            for index, identity in enumerate(data["condition_identities"])
        }
        condition_intercepts, _, _ = stack_posterior_draws(
            inference_data.posterior["condition_intercept"]
        )
        condition_slopes, _, _ = stack_posterior_draws(
            inference_data.posterior["condition_horizon_slope"]
        )
    probability = policy["trajectory_reporting"]["continuous_pairwise_difference_reporting"]["credible_interval_probability"]
    lower_probability = (1 - probability) / 2
    upper_probability = 1 - lower_probability
    records = []

    def unavailable_record(estimand_identity: str, identification: dict,
                           reason_code: str) -> dict:
        return {
            "candidate_identity": candidate["candidate_identity"],
            "estimand_decision_identity": estimand_identity,
            "outcome": candidate["collapsed_candidate"]["outcome"],
            "cohort": candidate["collapsed_candidate"]["cohort"],
            "horizon_axis": candidate["collapsed_candidate"]["horizon_axis"],
            "first_condition_protocol_identity": identification["first_condition_protocol_identity"],
            "second_condition_protocol_identity": identification["second_condition_protocol_identity"],
            "horizon": None,
            "posterior_mean_difference": None,
            "posterior_median_difference": None,
            "credible_interval_lower": None,
            "credible_interval_upper": None,
            "probability_difference_above_zero": None,
            "posterior_mean_ratio": None,
            "posterior_median_ratio": None,
            "ratio_credible_interval_lower": None,
            "ratio_credible_interval_upper": None,
            "probability_ratio_above_one": None,
            "interpretation_restriction_codes": json.loads(
                identification["interpretation_restriction_codes"]
            ),
            "analytical_status": "unavailable",
            "analytical_reason_code": reason_code,
            "estimate_label": policy["trajectory_reporting"]["estimate_label"],
            "causal_interpretation": policy["trajectory_reporting"]["continuous_pairwise_difference_reporting"]["causal_interpretation"]
        }

    for estimand_identity in candidate["estimand_decision_identities"]:
        matching_identification = [
            record
            for record in identification_records
            if record["estimand_decision_identity"] == estimand_identity
        ]

        if len(matching_identification) != 1:
            raise RuntimeError(f"Authorized estimand identity is not unique : {estimand_identity}")

        identification = matching_identification[0]
        first_condition = identification["first_condition_protocol_identity"]
        second_condition = identification["second_condition_protocol_identity"]

        if qualification["status"] != "available":
            records.append(
                unavailable_record(
                    estimand_identity,
                    identification,
                    "REFERENCE_POSTERIOR_NOT_ANALYTICALLY_AVAILABLE"
                )
            )
            continue

        if first_condition not in condition_index or second_condition not in condition_index:
            records.append(
                unavailable_record(
                    estimand_identity,
                    identification,
                    "CONDITIONS_NOT_PRESENT_IN_CANDIDATE"
                )
            )
            continue

        matching_support = [
            record
            for record in support_records
            if all(
                record[field_name] == identification[field_name]
                for field_name in (
                    "aging_mode",
                    "decision_scope",
                    "first_condition_protocol_identity",
                    "second_condition_protocol_identity",
                    "outcome",
                    "cohort",
                    "horizon_axis"
                )
            )
        ]

        if len(matching_support) != 1:
            raise RuntimeError(f"Authorized estimand support record is not unique : {estimand_identity}")

        permitted_ranges = json.loads(matching_support[0]["permitted_bracketing_ranges"])
        positive_width_range_found = False

        for permitted_range in permitted_ranges:
            lower = max(
                float(permitted_range["minimum"]),
                float(candidate["collapsed_candidate"]["support_interval_minimum"])
            )
            upper = min(
                float(permitted_range["maximum"]),
                float(candidate["collapsed_candidate"]["support_interval_maximum"])
            )

            if upper <= lower:
                continue

            positive_width_range_found = True

            range_candidate = {
                **candidate,
                "collapsed_candidate": {
                    **candidate["collapsed_candidate"],
                    "support_interval_minimum": lower,
                    "support_interval_maximum": upper,
                    "support_interval_lower_inclusive": True,
                    "support_interval_upper_inclusive": True
                }
            }
            grid = build_trajectory_grid(range_candidate, policy)

            for horizon in grid:
                scaled_horizon = (
                    float(horizon) - data["training_horizon_midpoint"]
                ) / data["training_horizon_width"]
                first = np.exp(
                    condition_intercepts[:, condition_index[first_condition]]
                    + condition_slopes[:, condition_index[first_condition]] * scaled_horizon
                )
                second = np.exp(
                    condition_intercepts[:, condition_index[second_condition]]
                    + condition_slopes[:, condition_index[second_condition]] * scaled_horizon
                )
                difference = second - first
                ratio = second / first

                if (
                    not np.all(np.isfinite(difference))
                    or not np.all(np.isfinite(ratio))
                    or np.any(ratio <= 0)
                ):
                    records.append(
                        {
                            **unavailable_record(
                                estimand_identity,
                                identification,
                                "NONFINITE_TRAJECTORY_DERIVATION"
                            ),
                            "horizon": float(horizon)
                        }
                    )
                    continue

                records.append({
                    "candidate_identity": candidate["candidate_identity"],
                    "estimand_decision_identity": estimand_identity,
                    "outcome": candidate["collapsed_candidate"]["outcome"],
                    "cohort": candidate["collapsed_candidate"]["cohort"],
                    "horizon_axis": candidate["collapsed_candidate"]["horizon_axis"],
                    "first_condition_protocol_identity": first_condition,
                    "second_condition_protocol_identity": second_condition,
                    "horizon": float(horizon),
                    "posterior_mean_difference": float(np.mean(difference)),
                    "posterior_median_difference": float(np.quantile(difference, 0.5, method = "linear")),
                    "credible_interval_lower": float(np.quantile(difference, lower_probability, method = "linear")),
                    "credible_interval_upper": float(np.quantile(difference, upper_probability, method = "linear")),
                    "probability_difference_above_zero": float(np.mean(difference > 0)),
                    "posterior_mean_ratio": float(np.mean(ratio)),
                    "posterior_median_ratio": float(np.quantile(ratio, 0.5, method = "linear")),
                    "ratio_credible_interval_lower": float(np.quantile(ratio, lower_probability, method = "linear")),
                    "ratio_credible_interval_upper": float(np.quantile(ratio, upper_probability, method = "linear")),
                    "probability_ratio_above_one": float(np.mean(ratio > 1)),
                    "interpretation_restriction_codes": json.loads(identification["interpretation_restriction_codes"]),
                    "analytical_status": "available",
                    "analytical_reason_code": "TRAJECTORY_REPORTING_COMPLETED",
                    "estimate_label": policy["trajectory_reporting"]["estimate_label"],
                    "causal_interpretation": policy["trajectory_reporting"]["continuous_pairwise_difference_reporting"]["causal_interpretation"]
                })

        if not positive_width_range_found:
            records.append(
                unavailable_record(
                    estimand_identity,
                    identification,
                    "NO_POSITIVE_WIDTH_CONTINUOUS_EVALUATION_RANGE"
                )
            )

    return records


# Comparing registered prior-scale posteriors without imposing an unregistered materiality threshold
def summarize_prior_sensitivity(workflow: dict, posterior_summaries: list[dict],
                                posterior_records: list[dict], policy: dict) -> list[dict]:
    registry_by_candidate = {
        record["candidate_identity"]: record
        for record in workflow["candidate_records"]
    }

    reference_multiplier = policy["sampling"]["reference_posterior_multiplier"]
    diagnostic_status = {
        (
            record["candidate_identity"],
            float(record["prior_scale_multiplier"])
        ): record.get("sampler_diagnostics_status")
        for record in posterior_records
    }
    indexed = {}

    for record in posterior_summaries:
        if registry_by_candidate[record["candidate_identity"]]["admissibility_status"] != "admissible":
            continue

        key = (
            record["candidate_identity"],
            record["parameter_identity"],
            canonical_json(record["coordinate_identity"])
            if record["coordinate_identity"] is not None
            else ""
        )
        indexed.setdefault(key, {})[float(record["prior_scale_multiplier"])] = record

    records = []

    for key in sorted(indexed):
        multiplier_records = indexed[key]
        any_record = next(iter(multiplier_records.values()))
        reference = multiplier_records.get(float(reference_multiplier))

        for sensitivity_multiplier in policy["priors"]["prior_sensitivity_scale_multipliers"]:
            if float(sensitivity_multiplier) == float(reference_multiplier):
                continue

            sensitivity = multiplier_records.get(float(sensitivity_multiplier))

            reference_passed = (
                reference is not None
                and reference["summary_status"] == "completed"
                and diagnostic_status.get((key[0], float(reference_multiplier))) == "completed_passed"
            )
            sensitivity_passed = (
                sensitivity is not None
                and sensitivity["summary_status"] == "completed"
                and diagnostic_status.get((key[0], float(sensitivity_multiplier))) == "completed_passed"
            )

            if not reference_passed or not sensitivity_passed:
                records.append({
                    "candidate_identity": key[0],
                    "outcome": any_record["outcome"],
                    "cohort": any_record["cohort"],
                    "parameter_or_trajectory_identity": (
                        key[1]
                        + (f":{key[2]}" if key[2] else "")
                    ),
                    "reference_prior_scale_multiplier": reference_multiplier,
                    "sensitivity_prior_scale_multiplier": sensitivity_multiplier,
                    "reference_posterior_median": None,
                    "sensitivity_posterior_median": None,
                    "absolute_median_difference": None,
                    "relative_median_difference": None,
                    "sensitivity_comparison_status": "unavailable",
                    "sensitivity_reason_code": (
                        "REFERENCE_POSTERIOR_DIAGNOSTICS_FAILED"
                        if not reference_passed
                        else "SENSITIVITY_POSTERIOR_DIAGNOSTICS_FAILED"
                    ),
                    "estimate_label": "bayesian_associational_sensitivity"
                })
                continue

            reference_median = float(reference["posterior_median"])

            sensitivity_median = float(sensitivity["posterior_median"])
            absolute_difference = abs(sensitivity_median - reference_median)
            relative_difference = (
                absolute_difference / abs(reference_median)
                if reference_median != 0
                else None
            )
            records.append({
                "candidate_identity": reference["candidate_identity"],
                "outcome": reference["outcome"],
                "cohort": reference["cohort"],
                "parameter_or_trajectory_identity": (
                    reference["parameter_identity"]
                    + (f":{key[2]}" if key[2] else "")
                ),
                "reference_prior_scale_multiplier": reference_multiplier,
                "sensitivity_prior_scale_multiplier": sensitivity_multiplier,
                "reference_posterior_median": reference_median,
                "sensitivity_posterior_median": sensitivity_median,
                "absolute_median_difference": absolute_difference,
                "relative_median_difference": relative_difference,
                "sensitivity_comparison_status": "completed_without_binary_materiality_classification",
                "sensitivity_reason_code": None,
                "estimate_label": "bayesian_associational_sensitivity"
            })

    return records


# Comparing latent trajectory medians across registered prior scales
def summarize_trajectory_prior_sensitivity(workflow: dict, cache_directory: Path,
                                           posterior_cache_records: list[dict],
                                           posterior_records: list[dict]) -> list[dict]:
    policy = workflow["validated_inputs"]["policy"]
    reference_multiplier = float(policy["sampling"]["reference_posterior_multiplier"])
    indexed_candidates = component_candidates_by_identity(workflow)

    registry_by_candidate = {
        record["candidate_identity"]: record
        for record in workflow["candidate_records"]
    }

    cache_by_candidate_multiplier = {
        (
            record["candidate_identity"],
            float(record["prior_scale_multiplier"])
        ): record
        for record in posterior_cache_records
    }
    diagnostics_by_candidate_multiplier = {
        (
            record["candidate_identity"],
            float(record["prior_scale_multiplier"])
        ): record.get("sampler_diagnostics_status")
        for record in posterior_records
    }
    records = []

    for candidate_identity in sorted(indexed_candidates):
        if registry_by_candidate[candidate_identity]["admissibility_status"] != "admissible":
            continue

        candidate = indexed_candidates[candidate_identity]
        data = prepare_pymc_candidate_data(candidate)
        grid = build_trajectory_grid(candidate, policy)
        reference_record = cache_by_candidate_multiplier.get(
            (candidate_identity, reference_multiplier)
        )
        reference_passed = (
            reference_record is not None
            and diagnostics_by_candidate_multiplier.get(
                (candidate_identity, reference_multiplier)
            ) == "completed_passed"
        )
        reference_inference = None
        reference_intercepts = None
        reference_slopes = None

        if reference_passed:
            reference_inference, _ = load_completed_cached_inference(
                cache_directory,
                reference_record["cache_identity"],
                policy
            )
            reference_intercepts, _, _ = stack_posterior_draws(
                reference_inference.posterior["condition_intercept"]
            )
            reference_slopes, _, _ = stack_posterior_draws(
                reference_inference.posterior["condition_horizon_slope"]
            )

        for sensitivity_multiplier in policy["priors"]["prior_sensitivity_scale_multipliers"]:
            sensitivity_multiplier = float(sensitivity_multiplier)

            if sensitivity_multiplier == reference_multiplier:
                continue

            sensitivity_record = cache_by_candidate_multiplier.get(
                (candidate_identity, sensitivity_multiplier)
            )
            sensitivity_passed = (
                sensitivity_record is not None
                and diagnostics_by_candidate_multiplier.get(
                    (candidate_identity, sensitivity_multiplier)
                ) == "completed_passed"
            )
            sensitivity_inference = None
            sensitivity_intercepts = None
            sensitivity_slopes = None

            if sensitivity_passed:
                sensitivity_inference, _ = load_completed_cached_inference(
                    cache_directory,
                    sensitivity_record["cache_identity"],
                    policy
                )
                sensitivity_intercepts, _, _ = stack_posterior_draws(
                    sensitivity_inference.posterior["condition_intercept"]
                )
                sensitivity_slopes, _, _ = stack_posterior_draws(
                    sensitivity_inference.posterior["condition_horizon_slope"]
                )

            for condition_index, condition_identity in enumerate(data["condition_identities"]):
                for horizon in grid:
                    trajectory_identity = (
                        "latent_trajectory:"
                        f"{condition_identity}:"
                        f"{format(float(horizon), '.17g')}"
                    )

                    if not reference_passed or not sensitivity_passed:
                        records.append({
                            "candidate_identity": candidate_identity,
                            "outcome": candidate["collapsed_candidate"]["outcome"],
                            "cohort": candidate["collapsed_candidate"]["cohort"],
                            "parameter_or_trajectory_identity": trajectory_identity,
                            "reference_prior_scale_multiplier": reference_multiplier,
                            "sensitivity_prior_scale_multiplier": sensitivity_multiplier,
                            "reference_posterior_median": None,
                            "sensitivity_posterior_median": None,
                            "absolute_median_difference": None,
                            "relative_median_difference": None,
                            "sensitivity_comparison_status": "unavailable",
                            "sensitivity_reason_code": (
                                "REFERENCE_POSTERIOR_DIAGNOSTICS_FAILED"
                                if not reference_passed
                                else "SENSITIVITY_POSTERIOR_DIAGNOSTICS_FAILED"
                            ),
                            "estimate_label": "bayesian_associational_sensitivity"
                        })
                        continue

                    scaled_horizon = (
                        float(horizon) - data["training_horizon_midpoint"]
                    ) / data["training_horizon_width"]
                    reference_values = np.exp(
                        reference_intercepts[:, condition_index]
                        + reference_slopes[:, condition_index] * scaled_horizon
                    )
                    sensitivity_values = np.exp(
                        sensitivity_intercepts[:, condition_index]
                        + sensitivity_slopes[:, condition_index] * scaled_horizon
                    )

                    if (
                        not np.all(np.isfinite(reference_values))
                        or not np.all(np.isfinite(sensitivity_values))
                    ):
                        records.append({
                            "candidate_identity": candidate_identity,
                            "outcome": candidate["collapsed_candidate"]["outcome"],
                            "cohort": candidate["collapsed_candidate"]["cohort"],
                            "parameter_or_trajectory_identity": trajectory_identity,
                            "reference_prior_scale_multiplier": reference_multiplier,
                            "sensitivity_prior_scale_multiplier": sensitivity_multiplier,
                            "reference_posterior_median": None,
                            "sensitivity_posterior_median": None,
                            "absolute_median_difference": None,
                            "relative_median_difference": None,
                            "sensitivity_comparison_status": "unavailable",
                            "sensitivity_reason_code": "NONFINITE_TRAJECTORY_DERIVATION",
                            "estimate_label": "bayesian_associational_sensitivity"
                        })
                        continue

                    reference_median = float(
                        np.quantile(reference_values, 0.5, method = "linear")
                    )
                    sensitivity_median = float(
                        np.quantile(sensitivity_values, 0.5, method = "linear")
                    )
                    absolute_difference = abs(sensitivity_median - reference_median)
                    records.append({
                        "candidate_identity": candidate_identity,
                        "outcome": candidate["collapsed_candidate"]["outcome"],
                        "cohort": candidate["collapsed_candidate"]["cohort"],
                        "parameter_or_trajectory_identity": trajectory_identity,
                        "reference_prior_scale_multiplier": reference_multiplier,
                        "sensitivity_prior_scale_multiplier": sensitivity_multiplier,
                        "reference_posterior_median": reference_median,
                        "sensitivity_posterior_median": sensitivity_median,
                        "absolute_median_difference": absolute_difference,
                        "relative_median_difference": (
                            absolute_difference / abs(reference_median)
                            if reference_median != 0
                            else None
                        ),
                        "sensitivity_comparison_status": "completed_without_binary_materiality_classification",
                        "sensitivity_reason_code": None,
                        "estimate_label": "bayesian_associational_sensitivity"
                    })

            if sensitivity_inference is not None:
                del sensitivity_inference

        if reference_inference is not None:
            del reference_inference

    return records


# Revalidating the published prior-predictive operation before posterior execution
def load_prior_predictive_handoff(output_directory: Path, workflow: dict) -> tuple[list[dict], dict, set]:
    report_path = output_directory / "bayesian_modeling_report.json"

    if not report_path.is_file():
        raise RuntimeError("Posterior fitting requires a completed prior-predictive report.")

    report = load_json_object(report_path)

    if report.get("operation") != "prior_predictive":
        raise RuntimeError("Posterior fitting requires the prior-predictive operation output.")

    diagnostics_path = output_directory / "prior_predictive_diagnostics.csv"
    template_path = output_directory / workflow["validated_inputs"]["policy"]["output_contract"]["prior_predictive_review_template_file"]
    _, diagnostics = read_csv_rows(diagnostics_path)
    review_template = load_json_object(template_path)
    validate_bayesian_report(
        report_path = report_path,
        expected_policy = workflow["validated_inputs"]["policy"],
        expected_policy_sha256 = workflow["validated_inputs"]["policy_sha256"],
        expected_input_hashes = report["input_hashes"],
        expected_candidate_records = workflow["candidate_records"],
        expected_accounting_records = workflow["observation_accounting_records"],
        expected_fold_records = workflow["fold_records"],
        expected_execution_plan = workflow["execution_plan"],
        expected_operation = "prior_predictive",
        expected_additional_tables = {
            "prior_predictive_diagnostics": (
                PRIOR_PREDICTIVE_DIAGNOSTIC_FIELDS,
                diagnostics
            )
        },
        expected_review_template = review_template,
        expected_execution_accounting = report["prior_predictive_execution_accounting"]
    )
    review_path = PROJECT_ROOT / workflow["validated_inputs"]["policy"]["prior_predictive_review"]["review_artifact_path"]
    acknowledged_pairs = validate_prior_predictive_review(
        review_path = review_path,
        workflow = workflow,
        candidate_inventory_sha256 = calculate_file_sha256(output_directory / "candidate_registry.csv"),
        diagnostics_sha256 = calculate_file_sha256(diagnostics_path),
        diagnostic_records = diagnostics
    )
    return diagnostics, review_template, acknowledged_pairs


# Recording execution and analytical status without conflating the two contracts
def build_final_candidate_records(workflow: dict, qualification: dict[str, dict],
                                  posterior_records: list[dict]) -> list[dict]:
    posterior_by_candidate = defaultdict(list)

    for posterior_record in posterior_records:
        posterior_by_candidate[
            posterior_record["candidate_identity"]
        ].append(
            posterior_record
        )

    policy = workflow[
        "validated_inputs"
    ][
        "policy"
    ]

    reference_multiplier = float(
        policy[
            "sampling"
        ][
            "reference_posterior_multiplier"
        ]
    )

    expected_prior_scale_multipliers = {
        float(
            prior_scale_multiplier
        )
        for prior_scale_multiplier in policy[
            "sampling"
        ][
            "full_data_posterior_scale_multipliers"
        ]
    }

    expected_multiplier_keys = sorted(
        format(
            multiplier,
            ".1f"
        )
        for multiplier in expected_prior_scale_multipliers
    )

    records = []

    for source_record in workflow["candidate_records"]:
        record = dict(source_record)
        candidate_identity = record["candidate_identity"]
        candidate_qualification = qualification.get(candidate_identity)
        candidate_posteriors = posterior_by_candidate.get(candidate_identity, [])

        if candidate_qualification is not None:
            record["posterior_analytical_status"] = candidate_qualification["status"]
            record["posterior_analytical_reason_code"] = candidate_qualification["reason_code"]
            record["posterior_qualification_failed_gates"] = list(
                candidate_qualification.get("failed_gates", [])
            )
        else:
            record["posterior_qualification_failed_gates"] = []

        if record["admissibility_status"] == "admissible":
            observed_prior_scale_multipliers = [
                float(
                    posterior_record[
                        "prior_scale_multiplier"
                    ]
                )
                for posterior_record in candidate_posteriors
            ]

            posterior_multiplier_inventory_complete = (
                len(
                    observed_prior_scale_multipliers
                )
                == len(
                    expected_prior_scale_multipliers
                )
                and set(
                    observed_prior_scale_multipliers
                )
                == expected_prior_scale_multipliers
            )

            posterior_execution_completed = (
                posterior_multiplier_inventory_complete
                and all(
                    posterior_record.get(
                        "fit_status"
                    )
                    == "completed"
                    for posterior_record in candidate_posteriors
                )
            )

            record["candidate_execution_status"] = (
                "fitted"
                if posterior_execution_completed
                else "fit_failed"
            )

        status_by_multiplier = {}
        reason_by_multiplier = {}

        for posterior_record in candidate_posteriors:
            multiplier = float(
                posterior_record[
                    "prior_scale_multiplier"
                ]
            )

            multiplier_key = format(
                multiplier,
                ".1f"
            )

            fit_completed = (
                posterior_record.get(
                    "fit_status"
                )
                == "completed"
            )

            sampler_passed = (
                posterior_record.get(
                    "sampler_diagnostics_status"
                )
                == "completed_passed"
            )

            if multiplier == reference_multiplier:
                status_by_multiplier[
                    multiplier_key
                ] = candidate_qualification[
                    "status"
                ]

                reason_by_multiplier[
                    multiplier_key
                ] = candidate_qualification[
                    "reason_code"
                ]

            elif not fit_completed:
                status_by_multiplier[
                    multiplier_key
                ] = "not_available"

                reason_by_multiplier[
                    multiplier_key
                ] = "POSTERIOR_SAMPLING_FAILED"

            elif sampler_passed:
                status_by_multiplier[
                    multiplier_key
                ] = "available"

                reason_by_multiplier[
                    multiplier_key
                ] = "SAMPLER_DIAGNOSTICS_COMPLETED"

            else:
                status_by_multiplier[
                    multiplier_key
                ] = "diagnostic_only"

                reason_by_multiplier[
                    multiplier_key
                ] = "SENSITIVITY_POSTERIOR_DIAGNOSTICS_FAILED"

        if record["admissibility_status"] == "admissible":
            for multiplier_key in expected_multiplier_keys:
                if multiplier_key not in status_by_multiplier:
                    status_by_multiplier[
                        multiplier_key
                    ] = "not_available"

                    reason_by_multiplier[
                        multiplier_key
                    ] = "POSTERIOR_SAMPLING_FAILED"

        record["posterior_status_by_prior_multiplier"] = status_by_multiplier
        record["posterior_reason_codes_by_prior_multiplier"] = reason_by_multiplier

        records.append(record)

    return records


# Building complete posterior outputs by streaming identity-bound cache entries
def build_fit_outputs(workflow: dict, output_directory: Path, cache_directory: Path,
                      software_environment_record: dict,
                      software_environment_sha256: str
) -> dict:
    prior_diagnostics, review_template, acknowledged_pairs = load_prior_predictive_handoff(
        output_directory,
        workflow
    )

    posterior_records, full_sampler_records, posterior_cache_records, full_accounting = (
        execute_full_posterior_sampling(
            workflow = workflow,
            acknowledged_pairs = acknowledged_pairs,
            cache_directory = cache_directory,
            software_environment_record = software_environment_record,
            software_environment_sha256 = software_environment_sha256,
            sampling_permitted = False
        )
    )
    predictions, validation_metrics, calibration, fold_sampler_records, validation_accounting = (
        execute_grouped_validation(
            workflow = workflow,
            cache_directory = cache_directory,
            software_environment_record = software_environment_record,
            software_environment_sha256 = software_environment_sha256,
            sampling_permitted = False
        )
    )

    posterior_summaries, predictive_diagnostics = summarize_full_posterior_caches(
        workflow = workflow,
        cache_directory = cache_directory,
        posterior_cache_records = posterior_cache_records
    )

    qualification = qualify_reference_posteriors(
        workflow = workflow,
        posterior_records = posterior_records,
        predictive_diagnostics = predictive_diagnostics,
        validation_metrics = validation_metrics,
        calibration_records = calibration
    )

    trajectories = []
    differences = []

    indexed_candidates = component_candidates_by_identity(workflow)

    policy = workflow["validated_inputs"]["policy"]

    reference_multiplier = policy["sampling"]["reference_posterior_multiplier"]

    difference_candidates_processed = set()

    for cache_record in posterior_cache_records:
        if float(cache_record["prior_scale_multiplier"]) != float(reference_multiplier):
            continue

        candidate_identity = cache_record["candidate_identity"]
        inference_data, _ = load_completed_cached_inference(
            cache_directory,
            cache_record["cache_identity"],
            policy
        )
        candidate = indexed_candidates[candidate_identity]
        trajectories.extend(
            summarize_candidate_trajectories(
                inference_data,
                candidate,
                policy,
                qualification[candidate_identity]
            )
        )
        differences.extend(
            summarize_candidate_differences(
                inference_data,
                candidate,
                workflow,
                qualification[candidate_identity]
            )
        )
        difference_candidates_processed.add(candidate_identity)
        del inference_data

    for candidate_identity in sorted(indexed_candidates):
        if candidate_identity in difference_candidates_processed:
            continue

        differences.extend(
            summarize_candidate_differences(
                None,
                indexed_candidates[candidate_identity],
                workflow,
                qualification[candidate_identity]
            )
        )

    trajectories.sort(key = lambda record: (
        record["candidate_identity"],
        record["condition_protocol_identity"],
        record["trajectory_target"],
        record["horizon"]
    ))
    differences.sort(key = lambda record: (
        record["candidate_identity"],
        record["estimand_decision_identity"],
        float("-inf") if record["horizon"] is None else record["horizon"]
    ))
    prior_sensitivity = summarize_prior_sensitivity(
        workflow,
        posterior_summaries,
        posterior_records,
        policy
    )
    prior_sensitivity.extend(
        summarize_trajectory_prior_sensitivity(
            workflow,
            cache_directory,
            posterior_cache_records,
            posterior_records
        )
    )
    prior_sensitivity.sort(key = lambda record: (
        record["candidate_identity"],
        record["parameter_or_trajectory_identity"],
        record["sensitivity_prior_scale_multiplier"]
    ))
    sampler_records = sorted(
        full_sampler_records + fold_sampler_records,
        key = lambda record: (
            record["candidate_identity"],
            record["fit_scope"],
            -1 if record["fold_index"] is None else record["fold_index"],
            record["prior_scale_multiplier"],
            record["parameter_identity"]
        )
    )

    _validate_all_reason_codes(
        fit_outputs = {
            "sampler_diagnostics": sampler_records,
            "posterior_predictive_diagnostics": predictive_diagnostics,
            "grouped_validation_predictions": predictions,
            "grouped_validation_metrics": validation_metrics,
            "grouped_validation_condition_calibration": calibration,
            "continuous_trajectory_summaries": trajectories,
            "continuous_trajectory_difference_summaries": differences,
            "prior_sensitivity_summaries": prior_sensitivity,
            "posterior_summaries": posterior_summaries,
            "posterior_records": posterior_records
        },
        policy = policy
    )

    return {
        "prior_diagnostics": prior_diagnostics,
        "review_template": review_template,
        "posterior_records": posterior_records,
        "posterior_cache_records": posterior_cache_records,
        "posterior_summaries": posterior_summaries,
        "sampler_diagnostics": sampler_records,
        "posterior_predictive_diagnostics": predictive_diagnostics,
        "grouped_validation_predictions": predictions,
        "grouped_validation_metrics": validation_metrics,
        "grouped_validation_condition_calibration": calibration,
        "continuous_trajectory_summaries": trajectories,
        "continuous_trajectory_difference_summaries": differences,
        "prior_sensitivity_summaries": prior_sensitivity,
        "qualification": qualification,
        "candidate_records": build_final_candidate_records(
            workflow,
            qualification,
            posterior_records
        ),
        "full_posterior_execution_accounting": full_accounting,
        "grouped_validation_execution_accounting": validation_accounting
    }


# Validating the registered Bayesian modeling policy
def validate_policy(policy: dict) -> None:
    _require_exact_fields(
        policy, REQUIRED_POLICY_SECTIONS, "Bayesian policy"
    )

    if not isinstance(policy["policy_version"], str) or not POLICY_VERSION_PATTERN.fullmatch(policy["policy_version"]):
        raise RuntimeError("Bayesian policy has an invalid version.")

    try:
        date.fromisoformat(policy["effective_date"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Bayesian policy has an invalid effective date.") from exc

    if not isinstance(policy["previous_policy_sha256"], str) or not SHA256_PATTERN.fullmatch(policy["previous_policy_sha256"]):
        raise RuntimeError("Bayesian policy has an invalid previous-policy digest.")

    registration = _require_exact_fields(
        policy["registration"],
        {
            "status", "descriptive_results_available_at_registration",
            "prior_predictive_results_available_at_registration",
            "posterior_results_available_at_registration",
            "bayesian_results_available_at_registration",
            "causal_identification_status", "result_informed",
            "result_informed_scope", "analysis_classification"
        },
        "Bayesian policy registration"
    )

    if registration["status"] != "amended_after_bayesian_results":
        raise RuntimeError("Bayesian policy does not disclose its amendment timing.")

    if registration["causal_identification_status"] != "not_identified_from_observed_design":
        raise RuntimeError("Bayesian policy is inconsistent with causal non-identification.")

    for availability_field in (
        "prior_predictive_results_available_at_registration",
        "posterior_results_available_at_registration",
        "bayesian_results_available_at_registration"
    ):
        if registration[availability_field] is not True:
            raise RuntimeError(
                f"Bayesian policy does not disclose available results : {availability_field}"
            )

    authorization = _require_exact_fields(
        policy["authorization"],
        {
            "identification_report_status_required",
            "causal_modeling_status_required",
            "associational_modeling_status_required",
            "required_estimand_status",
            "permitted_support_classifications",
            "causal_language_permitted",
            "causal_effect_parameters_permitted",
            "policy_recommendations_permitted",
            "causal_sensitivity_analysis_permitted",
            "cross_stage_pooling_permitted",
            "primary_and_sensitivity_cohorts_pooled",
            "elapsed_time_and_fec_axes_pooled",
            "calendar_and_cycle_aging_pooled",
            "unobserved_condition_combinations_permitted"
        },
        "Bayesian policy authorization"
    )

    for prohibited_field in REQUIRED_AUTHORIZATION_PROHIBITIONS:
        if authorization[prohibited_field] is not False:
            raise RuntimeError(f"Bayesian authorization incorrectly enables : {prohibited_field}")

    if authorization["causal_modeling_status_required"] != "not_authorized":
        raise RuntimeError("Bayesian policy requires causal modeling, which is prohibited.")

    if authorization["required_estimand_status"] != "authorized_with_recorded_restrictions":
        raise RuntimeError("Bayesian policy has an unexpected required estimand status.")

    if authorization["permitted_support_classifications"] != ["supported_with_restrictions", "sensitivity_only"]:
        raise RuntimeError("Bayesian policy has unexpected permitted support classifications.")

    inputs = _require_exact_fields(
        policy["inputs"],
        {
            "feature_manifest", "design_report", "identification_report",
            "descriptive_report", "required_feature_tables",
            "required_design_tables", "required_identification_table",
            "required_descriptive_tables",
            "descriptive_observation_accounting_reconciliation_required",
            "reconstructed_candidate_counts_must_match_descriptive_source_candidate_counts",
            "reconstructed_included_observation_identities_must_be_unique",
            "descriptive_analytical_result_status_used_for_candidate_selection",
            "every_input_hash_verification_required",
            "embedded_policy_object_verification_required"
        },
        "Bayesian policy inputs"
    )

    if "cell_features" not in inputs["required_feature_tables"]:
        raise RuntimeError("Bayesian inputs do not require cell_features.")

    if "reference_test_features" not in inputs["required_feature_tables"]:
        raise RuntimeError("Bayesian inputs do not require reference_test_features.")

    if inputs["descriptive_analytical_result_status_used_for_candidate_selection"] is not False:
        raise RuntimeError("Bayesian inputs permit candidate selection from descriptive performance.")

    candidate_inventory = _require_exact_fields(
        policy["candidate_inventory"],
        {
            "source", "descriptive_result_fields_used_for_candidate_selection",
            "required_selection_status", "representative_template",
            "required_context_family", "deduplication_fields",
            "authorization_join", "authorization_join_key",
            "candidate_selection_uses_outcome_values",
            "candidate_selection_uses_descriptive_fit_status",
            "candidate_selection_uses_validation_performance",
            "candidate_selection_uses_statistical_significance",
            "candidate_dispositions", "representative_estimator",
            "duplicate_source_group_rule",
            "duplicate_source_required_identical_fields",
            "duplicate_authorization_resolution",
            "authorization_restriction_order",
            "source_specifications_retained_for_provenance_only",
            "source_specifications_used_as_bayesian_terms",
            "authorized_condition_graph_rule",
            "disconnected_authorized_condition_graph_action",
            "candidate_identity_fields",
            "source_candidate_provenance_required",
            "candidate_construction_sequence",
            "candidate_identity_calculated_before_admissibility",
            "inadmissible_candidates_retain_candidate_identity"
        },
        "Bayesian policy candidate inventory"
    )

    if candidate_inventory["required_selection_status"] != "selected":
        raise RuntimeError("Bayesian candidate inventory does not select 'selected' status.")

    for prohibited_flag in (
        "candidate_selection_uses_outcome_values",
        "candidate_selection_uses_descriptive_fit_status",
        "candidate_selection_uses_validation_performance",
        "candidate_selection_uses_statistical_significance",
        "descriptive_result_fields_used_for_candidate_selection"
    ):
        if candidate_inventory[prohibited_flag] is not False:
            raise RuntimeError(f"Bayesian candidate selection permits prohibited field : {prohibited_flag}")

    _require_exact_value(
        candidate_inventory["candidate_identity_fields"],
        IMPLEMENTED_CANDIDATE_IDENTITY_FIELDS,
        "candidate_inventory.candidate_identity_fields"
    )

    _require_exact_value(
        candidate_inventory["candidate_construction_sequence"],
        IMPLEMENTED_CANDIDATE_CONSTRUCTION_SEQUENCE,
        "candidate_inventory.candidate_construction_sequence"
    )

    _require_exact_value(
        policy[
            "horizon_encoding"
        ],
        IMPLEMENTED_HORIZON_ENCODING,
        "horizon_encoding"
    )

    _require_exact_value(
        policy[
            "predictive_assessment"
        ][
            "fold_assignment_method"
        ],
        IMPLEMENTED_FOLD_ASSIGNMENT_METHOD,
        "predictive_assessment.fold_assignment_method"
    )

    observation_contract = policy["observation_contract"]
    if observation_contract["candidate_execution_statuses"][:4] != [
        "not_authorized", "not_assembled", "inadmissible", "admissible"
    ]:
        raise RuntimeError(
            "Bayesian candidate execution statuses must begin with 'not_authorized', 'not_assembled',"
            " 'inadmissible', 'admissible'."
        )

    # Validating the registered model section against implementation assumptions
    model_section = policy["model"]

    _require_supported_value(
        model_section.get("likelihood"),
        {"student_t"},
        "model.likelihood"
    )

    _require_supported_value(
        model_section.get("location_scale"),
        {"log_outcome"},
        "model.location_scale"
    )

    for required_true_field in (
        "condition_specific_intercept",
        "condition_specific_horizon_slope",
        "condition_coefficients_partially_pooled",
        "cell_random_intercept",
        "cell_random_horizon_slope",
        "cell_random_effect_correlation",
    ):
        _require_true(
            model_section.get(required_true_field),
            f"model.{required_true_field}"
        )

    for required_false_field in (
        "laboratory_adjustment_included",
        "equipment_adjustment_included",
        "baseline_covariate_adjustment_included",
        "condition_specific_residual_scales_modeled",
        "horizon_dependent_residual_scale_modeled",
        "decomposed_treatment_main_effects_reported",
        "dose_response_parameters_reported",
        "separate_treatment_main_effects_reported",
        "treatment_interaction_parameters_reported",
    ):
        _require_false(
            model_section.get(required_false_field),
            f"model.{required_false_field}"
        )

    _require_supported_value(
        model_section.get("condition_intercept_slope_correlation_mode"),
        {"fixed_zero"},
        "model.condition_intercept_slope_correlation_mode"
    )

    # Validating the registered horizon encoding
    horizon_encoding = policy["horizon_encoding"]

    _require_supported_value(
        horizon_encoding.get("transformation"),
        {"assembled_training_range_midpoint_centered_and_width_scaled"},
        "horizon_encoding.transformation"
    )

    _require_true(
        horizon_encoding.get("positive_training_horizon_width_required"),
        "horizon_encoding.positive_training_horizon_width_required"
    )

    _require_true(
        horizon_encoding.get("evaluation_support_remains_reporting_domain"),
        "horizon_encoding.evaluation_support_remains_reporting_domain"
    )

    _require_exact_value(
        horizon_encoding,
        IMPLEMENTED_HORIZON_ENCODING,
        "horizon_encoding"
    )

    # Validating the registered priors
    priors = policy["priors"]

    normal_prior_names = (
        "population_intercept",
        "population_horizon_slope"
    )

    half_normal_prior_names = (
        "condition_intercept_standard_deviation",
        "condition_horizon_slope_standard_deviation",
        "cell_random_intercept_standard_deviation",
        "cell_random_horizon_slope_standard_deviation",
        "residual_standard_deviation"
    )

    normal_prior_expected_values = {
        "population_intercept": {
            "mean": 0.0,
            "standard_deviation": 0.25
        },
        "population_horizon_slope": {
            "mean": 0.0,
            "standard_deviation": 0.25
        }
    }

    half_normal_prior_expected_scales = {
        "condition_intercept_standard_deviation": 0.15,
        "condition_horizon_slope_standard_deviation": 0.15,
        "cell_random_intercept_standard_deviation": 0.1,
        "cell_random_horizon_slope_standard_deviation": 0.15,
        "residual_standard_deviation": 0.1
    }

    for prior_name in normal_prior_names:
        prior_record = _require_exact_fields(
            priors[prior_name],
            {
                "distribution",
                "mean",
                "standard_deviation"
            },
            f"Bayesian prior {prior_name}"
        )

        _require_exact_value(
            prior_record["distribution"],
            "normal",
            f"priors.{prior_name}.distribution"
        )

        _require_exact_value(
            prior_record["mean"],
            normal_prior_expected_values[prior_name]["mean"],
            f"priors.{prior_name}.mean"
        )

        _require_exact_value(
            prior_record["standard_deviation"],
            normal_prior_expected_values[prior_name]["standard_deviation"],
            f"priors.{prior_name}.standard_deviation"
        )

    for prior_name in half_normal_prior_names:
        prior_record = _require_exact_fields(
            priors[prior_name],
            {
                "distribution",
                "scale"
            },
            f"Bayesian prior {prior_name}"
        )

        _require_exact_value(
            prior_record["distribution"],
            "half_normal",
            f"priors.{prior_name}.distribution"
        )

        _require_exact_value(
            prior_record["scale"],
            half_normal_prior_expected_scales[prior_name],
            f"priors.{prior_name}.scale"
        )

    correlation_prior = _require_exact_fields(
        priors["cell_random_effect_correlation"],
        {
            "distribution",
            "eta"
        },
        "Bayesian cell random-effect correlation prior"
    )

    _require_exact_value(
        correlation_prior["distribution"],
        "lkj",
        "priors.cell_random_effect_correlation.distribution"
    )

    _require_exact_value(
        correlation_prior["eta"],
        2.0,
        "priors.cell_random_effect_correlation.eta"
    )

    degrees_of_freedom_prior = _require_exact_fields(
        priors["student_t_degrees_of_freedom"],
        {
            "distribution",
            "exponential_mean"
        },
        "Bayesian Student-t degrees-of-freedom prior"
    )

    _require_exact_value(
        degrees_of_freedom_prior["distribution"],
        "two_plus_exponential",
        "priors.student_t_degrees_of_freedom.distribution"
    )

    if degrees_of_freedom_prior["exponential_mean"] != 10.0:
        raise RuntimeError("Bayesian policy requires the Student-t degrees-of-freedom exponential_mean to be 10.0.")

    _require_exact_value(
        priors["prior_sensitivity_scale_multipliers"],
        IMPLEMENTED_PRIOR_SENSITIVITY_SCALE_MULTIPLIERS,
        "priors.prior_sensitivity_scale_multipliers"
    )

    if priors.get(
        "binary_prior_sensitivity_materiality_threshold"
    ) is not None:
        raise RuntimeError(
            "Bayesian policy must not register a binary prior-sensitivity materiality threshold."
        )

    _require_false(
        priors.get(
            "binary_prior_sensitivity_materiality_classification_permitted"
        ),
        "priors.binary_prior_sensitivity_materiality_classification_permitted"
    )

    _require_exact_value(
        priors.get(
            "prior_predictive_new_cell_seed_rule"
        ),
        IMPLEMENTED_PRIOR_PREDICTIVE_NEW_CELL_SEED_RULE,
        "priors.prior_predictive_new_cell_seed_rule"
    )

    _require_positive_integer(
        priors["prior_predictive_draws_per_multiplier"],
        "priors.prior_predictive_draws_per_multiplier"
    )

    expected_prior_seed_keys = {
        format(
            float(multiplier),
            ".1f"
        )
        for multiplier in IMPLEMENTED_PRIOR_SENSITIVITY_SCALE_MULTIPLIERS
    }

    if set(
        priors["prior_predictive_seeds"]
    ) != expected_prior_seed_keys:
        raise RuntimeError("Bayesian policy prior-predictive seed keys do not match the registered multipliers.")

    for (
        multiplier_key,
        prior_predictive_seed
    ) in priors["prior_predictive_seeds"].items():
        _require_positive_integer(
            prior_predictive_seed,
            f"priors.prior_predictive_seeds.{multiplier_key}"
        )

    _require_exact_value(
        priors[
            "prior_predictive_new_cell_seed_rule"
        ],
        (
            "registered_prior_predictive_seed_plus_zero_based_condition_index_plus_one"
        ),
        "priors.prior_predictive_new_cell_seed_rule"
    )

    _require_false(
        priors["prior_scales_selected_from_fitted_outcome_direction"],
        "priors.prior_scales_selected_from_fitted_outcome_direction"
    )

    _require_true(
        priors["prior_predictive_sampling_required"],
        "priors.prior_predictive_sampling_required"
    )

    _require_true(
        priors["prior_predictive_review_required_before_posterior_sampling"],
        "priors.prior_predictive_review_required_before_posterior_sampling"
    )

    # Validating the registered admissibility thresholds
    admissibility = policy["admissibility"]

    for required_int_field in (
        "minimum_conditions",
        "minimum_independent_cells",
        "minimum_repeated_cells",
        "minimum_repeated_cells_per_condition",
        "minimum_distinct_positive_horizons",
        "minimum_distinct_positive_horizons_per_condition",
        "minimum_independent_cells_per_condition",
    ):
        _require_positive_integer(
            admissibility.get(required_int_field),
            f"admissibility.{required_int_field}"
        )

    _require_supported_value(
        admissibility.get("rank_method"),
        {"singular_value_decomposition"},
        "admissibility.rank_method"
    )

    _require_true(
        admissibility.get("fixed_location_design_rank_check_required"),
        "admissibility.fixed_location_design_rank_check_required"
    )

    _require_false(
        admissibility.get("automatic_model_simplification_permitted"),
        "admissibility.automatic_model_simplification_permitted"
    )

    _require_false(
        admissibility.get("automatic_prior_change_permitted"),
        "admissibility.automatic_prior_change_permitted"
    )

    # Validating the registered sampling configuration
    sampling = policy["sampling"]

    _require_supported_value(
        sampling.get("sampler"),
        {"native_nuts"},
        "sampling.sampler"
    )

    _require_supported_value(
        sampling.get("engine"),
        {"pymc"},
        "sampling.engine"
    )

    _require_positive_integer(
        sampling.get("chains"),
        "sampling.chains"
    )

    _require_positive_integer(
        sampling.get("draws_per_chain"),
        "sampling.draws_per_chain"
    )

    _require_positive_integer(
        sampling.get("tuning_draws_per_chain"),
        "sampling.tuning_draws_per_chain"
    )

    _require_positive_integer(
        sampling.get("cores"),
        "sampling.cores"
    )

    sampler_seed = sampling.get(
        "sampler_seed"
    )

    if (
        not isinstance(
            sampler_seed,
            int
        )
        or isinstance(
            sampler_seed,
            bool
        )
        or sampler_seed < 0
    ):
        raise RuntimeError("Bayesian policy requires a nonnegative integer sampler_seed.")

    _require_false(
        sampling.get("automatic_retry_permitted"),
        "sampling.automatic_retry_permitted"
    )

    _require_false(
        sampling.get("automatic_sampler_change_permitted"),
        "sampling.automatic_sampler_change_permitted"
    )

    _require_true(
        sampling.get("discard_tuned_samples"),
        "sampling.discard_tuned_samples"
    )

    if sampling.get("full_data_posterior_scale_multipliers") != priors["prior_sensitivity_scale_multipliers"]:
        raise RuntimeError(
            "Bayesian policy requires full_data_posterior_scale_multipliers to equal prior_sensitivity_scale_multipliers."
        )

    _require_false(
        sampling.get("validation_operation_samples_model"),
        "sampling.validation_operation_samples_model"
    )

    _require_false(
        sampling.get("prior_predictive_operation_conditions_on_observed_outcomes"),
        "sampling.prior_predictive_operation_conditions_on_observed_outcomes"
    )

    _require_true(
        sampling.get("analytical_qualification_uses_reference_posterior_only"),
        "sampling.analytical_qualification_uses_reference_posterior_only"
    )

    _require_open_probability(
        sampling["target_accept"],
        "sampling.target_accept"
    )

    _require_positive_integer(
        sampling["maximum_tree_depth"],
        "sampling.maximum_tree_depth"
    )

    _require_exact_value(
        sampling["initialization"],
        "jitter_plus_adapt_diag",
        "sampling.initialization"
    )

    if sampling["cores"] != 1:
        raise RuntimeError("Bayesian policy requires single-core sampling.")

    # Validating the registered diagnostic thresholds
    diagnostics = policy["diagnostics"]

    _require_positive_number(
        diagnostics["maximum_rank_normalized_split_rhat"],
        "diagnostics.maximum_rank_normalized_split_rhat"
    )

    if (
        diagnostics["maximum_rank_normalized_split_rhat"]
        <= 1.0
    ):
        raise RuntimeError("Bayesian policy requires the maximum R-hat threshold to exceed one.")

    _require_exact_value(
        diagnostics["maximum_rank_normalized_split_rhat"],
        1.01,
        "diagnostics.maximum_rank_normalized_split_rhat"
    )

    _require_exact_value(
        diagnostics["minimum_bulk_effective_sample_size"],
        400,
        "diagnostics.minimum_bulk_effective_sample_size"
    )

    _require_exact_value(
        diagnostics["minimum_tail_effective_sample_size"],
        400,
        "diagnostics.minimum_tail_effective_sample_size"
    )

    _require_exact_value(
        diagnostics["maximum_divergence_count"],
        0,
        "diagnostics.maximum_divergence_count"
    )

    _require_exact_value(
        diagnostics["minimum_bfmi_per_chain"],
        0.3,
        "diagnostics.minimum_bfmi_per_chain"
    )

    _require_exact_value(
        diagnostics["maximum_tree_depth_hit_count"],
        0,
        "diagnostics.maximum_tree_depth_hit_count"
    )

    # Validating the registered predictive assessment
    predictive_assessment = policy["predictive_assessment"]

    _require_positive_integer(
        predictive_assessment.get("requested_fold_count"),
        "predictive_assessment.requested_fold_count"
    )

    _require_positive_integer(
        predictive_assessment.get("minimum_effective_fold_count"),
        "predictive_assessment.minimum_effective_fold_count"
    )

    if predictive_assessment["minimum_effective_fold_count"] < 2:
        raise RuntimeError("Bayesian policy requires minimum_effective_fold_count to be at least 2.")

    _require_supported_value(
        predictive_assessment.get("fold_assignment_unit"),
        {"cell_identity"},
        "predictive_assessment.fold_assignment_unit"
    )

    _require_false(
        predictive_assessment.get("cell_leakage_permitted"),
        "predictive_assessment.cell_leakage_permitted"
    )

    _require_false(
        predictive_assessment.get("outcome_dependent_model_selection_permitted"),
        "predictive_assessment.outcome_dependent_model_selection_permitted"
    )

    _require_true(
        predictive_assessment.get("all_effective_folds_required_for_completed_validation"),
        "predictive_assessment.all_effective_folds_required_for_completed_validation"
    )

    _require_exact_value(
        predictive_assessment["fold_assignment_method"],
        IMPLEMENTED_FOLD_ASSIGNMENT_METHOD,
        "predictive_assessment.fold_assignment_method"
    )

    # Validating the registered trajectory reporting
    trajectory_reporting = policy["trajectory_reporting"]

    _require_positive_integer(
        trajectory_reporting.get(
            "new_cell_predictive_seed"
        ),
        "trajectory_reporting.new_cell_predictive_seed"
    )

    _require_exact_value(
        trajectory_reporting.get(
            "new_cell_predictive_seed_rule"
        ),
        IMPLEMENTED_TRAJECTORY_NEW_CELL_SEED_RULE,
        "trajectory_reporting.new_cell_predictive_seed_rule"
    )

    _require_true(
        trajectory_reporting.get("continuous_trajectory_reporting_authorized"),
        "trajectory_reporting.continuous_trajectory_reporting_authorized"
    )

    _require_false(
        trajectory_reporting.get("discrete_point_contrasts_authorized"),
        "trajectory_reporting.discrete_point_contrasts_authorized"
    )

    _require_supported_value(
        trajectory_reporting.get("causal_label"),
        {"not_authorized"},
        "trajectory_reporting.causal_label"
    )

    _require_supported_value(
        trajectory_reporting.get("estimate_label"),
        {"bayesian_associational"},
        "trajectory_reporting.estimate_label"
    )

    # Validating the cache contract
    cache_contract = _require_exact_fields(
        policy["cache_contract"],
        {
            "default_cache_directory",
            "cache_entry_directory_identity",
            "cache_metadata_file",
            "cache_inference_data_file",
            "cache_completion_file",
            "cache_entry_written_transactionally",
            "completion_file_written_last",
            "cache_metadata_schema_required",
            "cache_artifact_hashes_required",
            "incomplete_cache_entry_reuse_permitted",
            "mismatched_cache_entry_reuse_permitted",
            "corrupt_cache_entry_action",
            "corrupt_cache_quarantine_suffix",
            "quarantined_cache_entry_reuse_permitted",
            "quarantined_cache_entry_preserved_for_diagnostics",
            "software_environment_identity_fields",
            "software_environment_identity_method",
            "numpy_configuration_sha256_source",
            "cache_identity_uses_absolute_paths",
            "cache_metadata_fields",
            "cache_completion_fields",
        },
        "Bayesian policy cache contract"
    )

    prior_predictive_review = _require_exact_fields(
        policy["prior_predictive_review"],
        {
            "review_required_before_posterior_sampling",
            "review_artifact_path",
            "review_template_file",
            "review_scope_fields",
            "permitted_review_statuses",
            "posterior_sampling_required_status",
            "observed_outcomes_used_for_review",
            "descriptive_coefficients_used_for_review",
            "comparative_model_performance_used_for_review",
            "review_uses_prior_predictive_diagnostics_only",
            "policy_sha256_binding_required",
            "candidate_inventory_sha256_binding_required",
            "prior_predictive_diagnostics_sha256_binding_required",
            "prior_predictive_execution_status_required_for_every_admissible_candidate_multiplier",
            "review_population",
            "complete_review_population_coverage_required",
            "inadmissible_candidates_require_review_records",
            "failed_prior_predictive_pairs_require_review_records",
            "failed_prior_predictive_pairs_require_acknowledgement",
            "failed_prior_predictive_pair_execution_status",
            "failed_prior_predictive_pair_blocks_only_its_corresponding_posterior",
            "failed_reference_multiplier_blocks_reference_posterior_for_that_candidate",
            "failed_sensitivity_multiplier_blocks_only_that_sensitivity_posterior",
            "successful_pairs_may_proceed_when_another_pair_failed",
            "missing_or_mismatched_review_action",
            "policy_amendment_required_before_changing_priors",
            "silent_review_default_permitted",
            "review_record_fields",
            "nonempty_review_rationale_required",
            "acknowledged_status_interpretation",
            "acknowledgement_is_automatic_prior_approval",
            "policy_amendment_required_status_interpretation",
            "policy_amendment_required_record_blocks_fit_operation",
            "nonfinite_prior_predictive_values_require_policy_amendment_or_explicit_scientific_rationale",
            "review_document_fields",
            "review_records_sorted_by"
        },
        "Bayesian policy prior-predictive review"
    )

    _require_exact_value(
        prior_predictive_review["review_document_fields"],
        [
            "policy_sha256",
            "candidate_inventory_sha256",
            "prior_predictive_diagnostics_sha256",
            "review_records"
        ],
        "prior_predictive_review.review_document_fields"
    )

    _require_exact_value(
        prior_predictive_review["review_records_sorted_by"],
        [
            "candidate_identity",
            "outcome",
            "prior_scale_multiplier"
        ],
        "prior_predictive_review.review_records_sorted_by"
    )

    _require_exact_value(
        cache_contract["cache_metadata_fields"],
        [
            "cache_identity",
            "cache_identity_sha256",
            "software_environment",
            "software_environment_sha256",
            "execution_status",
            "artifact_hashes"
        ],
        "cache_contract.cache_metadata_fields"
    )

    _require_exact_value(
        cache_contract["cache_completion_fields"],
        [
            "cache_identity_sha256",
            "cache_metadata_sha256",
            "inference_data_sha256",
            "execution_status"
        ],
        "cache_contract.cache_completion_fields"
    )

    _require_true(
        cache_contract.get("cache_entry_written_transactionally"),
        "cache_contract.cache_entry_written_transactionally"
    )

    _require_true(
        cache_contract.get("completion_file_written_last"),
        "cache_contract.completion_file_written_last"
    )

    _require_false(
        cache_contract.get("incomplete_cache_entry_reuse_permitted"),
        "cache_contract.incomplete_cache_entry_reuse_permitted"
    )

    _require_false(
        cache_contract.get("cache_identity_uses_absolute_paths"),
        "cache_contract.cache_identity_uses_absolute_paths"
    )

    cache_import_contract = _require_exact_fields(
        policy["cache_import_contract"],
        {
            "enabled",
            "source_policy_sha256",
            "source_implementation_sha256",
            "source_software_environment_sha256",
            "verified_input_hashes",
            "permitted_operations",
            "required_source_entry_count",
            "required_operation_counts",
            "required_source_execution_status",
            "inference_data_required",
            "source_draws_are_read_only",
            "source_model_prior_likelihood_sampling_candidate_seed_and_input_contract_unchanged",
            "source_cache_identity_and_artifact_hashes_revalidated",
            "posterior_derived_statistics_recomputed_under_current_reporting_contract",
            "source_sampler_diagnostics_reused_only_after_schema_and_identity_validation",
            "missing_or_invalid_source_entry_action"
        },
        "Bayesian policy cache import contract"
    )

    _require_true(
        cache_import_contract["enabled"],
        "cache_import_contract.enabled"
    )

    for digest_field in (
        "source_policy_sha256",
        "source_implementation_sha256",
        "source_software_environment_sha256"
    ):
        if not SHA256_PATTERN.fullmatch(cache_import_contract[digest_field]):
            raise RuntimeError(
                f"Bayesian cache import contract has an invalid digest : {digest_field}"
            )

    _require_exact_value(
        cache_import_contract["permitted_operations"],
        [
            "prior_predictive",
            "full_data_posterior",
            "grouped_validation_posterior"
        ],
        "cache_import_contract.permitted_operations"
    )

    _require_exact_value(
        cache_import_contract["required_operation_counts"],
        {
            "prior_predictive": 528,
            "full_data_posterior": 528,
            "grouped_validation_posterior": 393
        },
        "cache_import_contract.required_operation_counts"
    )

    if cache_import_contract["required_source_entry_count"] != sum(
        cache_import_contract["required_operation_counts"].values()
    ):
        raise RuntimeError("Bayesian cache import entry count does not reconcile.")

    for required_true_field in (
        "inference_data_required",
        "source_draws_are_read_only",
        "source_model_prior_likelihood_sampling_candidate_seed_and_input_contract_unchanged",
        "source_cache_identity_and_artifact_hashes_revalidated",
        "posterior_derived_statistics_recomputed_under_current_reporting_contract",
        "source_sampler_diagnostics_reused_only_after_schema_and_identity_validation"
    ):
        _require_true(
            cache_import_contract[required_true_field],
            f"cache_import_contract.{required_true_field}"
        )

    for required_cache_identity_field in (
        "policy_sha256",
        "verified_input_hashes",
        "candidate_identity",
        "operation",
        "prior_scale_multiplier",
        "validation_fold_identity",
        "software_environment_sha256",
        "implementation_sha256",
    ):
        if required_cache_identity_field not in policy["execution_planning"]["cache_identity_fields"]:
            raise RuntimeError(
                f"Bayesian policy cache identity is missing a required field : {required_cache_identity_field}"
            )

    # Validating the reproducibility contract
    reproducibility = policy["reproducibility"]

    _require_true(
        reproducibility.get("single_threaded_numerical_execution"),
        "reproducibility.single_threaded_numerical_execution"
    )

    _require_true(
        reproducibility.get("fixed_chain_seeds"),
        "reproducibility.fixed_chain_seeds"
    )

    _require_true(
        reproducibility.get("exact_input_hashes_recorded"),
        "reproducibility.exact_input_hashes_recorded"
    )

    _require_false(
        reproducibility.get("byte_identical_posterior_draws_promised_across_platforms"),
        "reproducibility.byte_identical_posterior_draws_promised_across_platforms"
    )

    # Validating the newly added reason-code contract
    reason_code_contract = _require_exact_fields(
        policy["reason_code_contract"],
        {
            "candidate_admissibility", "grouped_validation", "validation_abort", "prior_predictive_execution",
            "cache_validation", "posterior_sampling", "sampler_diagnostics", "posterior_predictive_execution",
            "grouped_validation_execution", "analytical_qualification", "trajectory_reporting"
        },
        "Bayesian policy reason code contract"
    )

    for category_name, category_codes in reason_code_contract.items():
        if (
            not isinstance(category_codes, list)
            or not category_codes
            or any(
                not isinstance(code, str) or not code
                for code in category_codes
            )
        ):
            raise RuntimeError(f"Bayesian reason-code contract contains an invalid category : {category_name}")

        if len(category_codes) != len(set(category_codes)):
            raise RuntimeError(f"Bayesian reason-code contract contains duplicate codes : {category_name}")

    stochastic_process_environment = _require_exact_fields(
        policy["stochastic_process_environment"],
        {
            "required_environment_variables",
            "validation_required_before_loading_pymc",
            "environment_values_recorded",
            "environment_values_included_in_software_environment_identity",
            "mismatched_environment_action",
        },
        "Bayesian policy stochastic process environment"
    )

    required_environment_variables = stochastic_process_environment[
        "required_environment_variables"
    ]

    if (
        not isinstance(required_environment_variables, dict)
        or not required_environment_variables
    ):
        raise RuntimeError("Bayesian stochastic process environment requires a non-empty required_environment_variables object.")

    for variable_name, variable_value in required_environment_variables.items():
        if not isinstance(variable_name, str) or not variable_name:
            raise RuntimeError("Bayesian stochastic process environment contains an invalid variable name.")

        if not isinstance(variable_value, str) or not variable_value:
            raise RuntimeError(f"Bayesian stochastic process environment contains an invalid value for {variable_name}.")

    if "PYTHONHASHSEED" not in required_environment_variables:
        raise RuntimeError("Bayesian stochastic process environment must register PYTHONHASHSEED.")

    if required_environment_variables["PYTHONHASHSEED"] != "0":
        raise RuntimeError("Bayesian stochastic process environment must require PYTHONHASHSEED == '0'.")

    _require_true(
        stochastic_process_environment["validation_required_before_loading_pymc"],
        "stochastic_process_environment.validation_required_before_loading_pymc"
    )

    _require_true(
        stochastic_process_environment["environment_values_recorded"],
        "stochastic_process_environment.environment_values_recorded"
    )

    _require_true(
        stochastic_process_environment[
            "environment_values_included_in_software_environment_identity"
        ],
        "stochastic_process_environment.environment_values_included_in_software_environment_identity"
    )

    _require_supported_value(
        stochastic_process_environment["mismatched_environment_action"],
        {"abort_before_stochastic_execution"},
        "stochastic_process_environment.mismatched_environment_action"
    )

    posterior_artifact_contract = _require_exact_fields(
        policy["posterior_artifact_contract"],
        {
            "likelihood_variable_name",
            "likelihood_scale",
            "likelihood_distribution",
            "original_scale_predictive_variable_name",
            "original_scale_predictive_derivation",
            "nonfinite_original_scale_predictions_recorded",
            "nonfinite_original_scale_predictions_silently_removed",
            "free_random_variable_names",
            "posterior_deterministic_variable_names",
            "pymc_posterior_predictive_variable_names",
            "derived_posterior_predictive_variable_names",
            "coordinate_names",
            "variable_dimensions",
            "diagnostics_cover_every_free_random_variable",
            "coordinate_dimension_required",
            "posterior_predictive_requires_coordinate_dimension",
        },
        "Bayesian policy posterior artifact contract"
    )

    for string_field_name in (
        "likelihood_variable_name",
        "likelihood_scale",
        "likelihood_distribution",
        "original_scale_predictive_variable_name",
        "original_scale_predictive_derivation",
    ):
        field_value = posterior_artifact_contract[string_field_name]

        if (
            not isinstance(field_value, str)
            or not field_value
        ):
            raise RuntimeError(f"Bayesian posterior artifact contract requires a nonempty string : {string_field_name}")

    _require_exact_value(
        posterior_artifact_contract["likelihood_scale"],
        "natural_log_outcome",
        "posterior_artifact_contract.likelihood_scale"
    )

    _require_exact_value(
        posterior_artifact_contract["likelihood_distribution"],
        "student_t",
        "posterior_artifact_contract.likelihood_distribution"
    )

    _require_exact_value(
        posterior_artifact_contract["original_scale_predictive_derivation"],
        (
            "exponential_of_posterior_predictive_log_outcome_observation"
        ),
        (
            "posterior_artifact_contract."
            "original_scale_predictive_derivation"
        )
    )

    expected_predictive_variables = [
        posterior_artifact_contract[
            "likelihood_variable_name"
        ]
    ]

    if (
        posterior_artifact_contract[
            "pymc_posterior_predictive_variable_names"
        ]
        != expected_predictive_variables
    ):
        raise RuntimeError("The registered PyMC posterior-predictive variable must be the observed likelihood variable.")

    _require_true(
        posterior_artifact_contract["nonfinite_original_scale_predictions_recorded"],
        "posterior_artifact_contract.nonfinite_original_scale_predictions_recorded"
    )

    _require_false(
        posterior_artifact_contract["nonfinite_original_scale_predictions_silently_removed"],
        "posterior_artifact_contract.nonfinite_original_scale_predictions_silently_removed"
    )

    for variable_list_field in (
        "free_random_variable_names",
        "posterior_deterministic_variable_names",
        "pymc_posterior_predictive_variable_names",
        "derived_posterior_predictive_variable_names",
        "coordinate_names",
    ):
        variable_list = posterior_artifact_contract[variable_list_field]

        if (
            not isinstance(variable_list, list)
            or not variable_list
            or any(
                not isinstance(entry, str) or not entry
                for entry in variable_list
            )
            or len(variable_list) != len(set(variable_list))
        ):
            raise RuntimeError(f"Bayesian posterior artifact contract contains an invalid field : {variable_list_field}")

    registered_coordinate_names = set(
        posterior_artifact_contract["coordinate_names"]
    )

    registered_variable_names = (
        set(posterior_artifact_contract["free_random_variable_names"])
        | set(posterior_artifact_contract["posterior_deterministic_variable_names"])
        | set(posterior_artifact_contract["pymc_posterior_predictive_variable_names"])
        | set(posterior_artifact_contract["derived_posterior_predictive_variable_names"])
        | {posterior_artifact_contract["likelihood_variable_name"]}
    )

    variable_dimensions = posterior_artifact_contract["variable_dimensions"]

    if not isinstance(variable_dimensions, dict):
        raise RuntimeError("Bayesian posterior artifact contract requires an object for variable_dimensions.")

    for variable_name, dimension_names in variable_dimensions.items():
        if (
            not isinstance(variable_name, str)
            or not variable_name
        ):
            raise RuntimeError("Bayesian posterior artifact contract contains an invalid variable_dimensions key.")

        if variable_name not in registered_variable_names:
            raise RuntimeError(
                f"Bayesian posterior artifact contract declares dimensions for an unregistered variable : {variable_name}"
            )

        if (
            not isinstance(dimension_names, list)
            or not dimension_names
            or any(
                not isinstance(dimension_name, str) or not dimension_name
                for dimension_name in dimension_names
            )
        ):
            raise RuntimeError(f"Bayesian posterior artifact contract contains an invalid dimension list : {variable_name}")

        for dimension_name in dimension_names:
            if dimension_name not in registered_coordinate_names:
                raise RuntimeError(
                    "Bayesian posterior artifact contract references an unregistered coordinate : "
                    f"{variable_name} : {dimension_name}"
                )

    _require_true(
        posterior_artifact_contract["diagnostics_cover_every_free_random_variable"],
        "posterior_artifact_contract.diagnostics_cover_every_free_random_variable"
    )

    _require_true(
        posterior_artifact_contract["coordinate_dimension_required"],
        "posterior_artifact_contract.coordinate_dimension_required"
    )

    _require_true(
        posterior_artifact_contract["posterior_predictive_requires_coordinate_dimension"],
        "posterior_artifact_contract.posterior_predictive_requires_coordinate_dimension"
    )

    execution_contract = _require_exact_fields(
        policy["execution_contract"],
        {"operations", "validate", "prior_predictive", "populate_fit_cache", "fit",
         "operation_specific_absent_tables_are_recorded_as_not_produced",
         "partial_or_failed_operation_never_replaces_a_completed_output_directory"},
        "Bayesian policy execution contract"
    )

    populate_fit_cache_contract = _require_exact_fields(
        execution_contract["populate_fit_cache"],
        {"sampling_permitted", "posterior_sampling_permitted",
         "prior_predictive_review_required", "publishes_analytical_artifacts",
         "required_outputs"},
        "Bayesian populate_fit_cache execution contract"
    )

    _require_true(
        populate_fit_cache_contract["sampling_permitted"],
        "execution_contract.populate_fit_cache.sampling_permitted"
    )
    _require_true(
        populate_fit_cache_contract["posterior_sampling_permitted"],
        "execution_contract.populate_fit_cache.posterior_sampling_permitted"
    )
    _require_true(
        populate_fit_cache_contract["prior_predictive_review_required"],
        "execution_contract.populate_fit_cache.prior_predictive_review_required"
    )
    _require_false(
        populate_fit_cache_contract["publishes_analytical_artifacts"],
        "execution_contract.populate_fit_cache.publishes_analytical_artifacts"
    )

    if set(execution_contract["operations"]) != REQUIRED_OPERATIONS:
        raise RuntimeError("Bayesian execution contract has an invalid operations inventory.")

    if execution_contract["validate"]["sampling_permitted"] is not False:
        raise RuntimeError("Bayesian validate operation permits sampling.")

    if execution_contract["prior_predictive"]["posterior_sampling_permitted"] is not False:
        raise RuntimeError("Bayesian prior_predictive operation permits posterior sampling.")

    output_contract = _require_exact_fields(
        policy["output_contract"],
        {
            "report_file", "tables", "posterior_directory",
            "failed_candidates_remain_in_registry", "exact_table_schemas_required",
            "input_hashes_required", "table_hashes_required", "posterior_hashes_required",
            "embedded_policy_required", "implementation_sha256_recorded",
            "transactional_publication_required",
            "post_publication_validation_required", "silent_failure_permitted",
            "silent_candidate_exclusion_permitted", "silent_prior_change_permitted",
            "silent_specification_change_permitted",
            "prior_predictive_review_template_file", "posterior_artifact_format",
            "posterior_artifact_groups", "reference_and_sensitivity_posterior_artifacts_separate",
            "analytical_status_recorded_per_candidate_and_prior_multiplier",
            "continuous_difference_records_require_estimand_decision_identity",
            "continuous_difference_records_require_interpretation_restriction_codes"
        },
        "Bayesian policy output contract"
    )

    if output_contract["embedded_policy_required"] is not True:
        raise RuntimeError("Bayesian output contract does not require the embedded policy.")

    if output_contract["post_publication_validation_required"] is not True:
        raise RuntimeError("Bayesian output contract does not require post-publication validation.")

    if output_contract["transactional_publication_required"] is not True:
        raise RuntimeError("Bayesian output contract does not require transactional publication.")

    for prohibited_flag in (
        "silent_failure_permitted",
        "silent_candidate_exclusion_permitted",
        "silent_prior_change_permitted",
        "silent_specification_change_permitted"
    ):
        if output_contract[prohibited_flag] is not False:
            raise RuntimeError(f"Bayesian output contract incorrectly enables : {prohibited_flag}")

    if "candidate_registry" not in output_contract["tables"]:
        raise RuntimeError("Bayesian output contract omits candidate_registry.")

    if "observation_accounting" not in output_contract["tables"]:
        raise RuntimeError("Bayesian output contract omits observation_accounting.")

    if "grouped_validation_folds" not in output_contract["tables"]:
        raise RuntimeError("Bayesian output contract omits grouped_validation_folds.")

    if set(
        output_contract[
            "tables"
        ]
    ) != set(
        OUTPUT_TABLE_FIELDS
    ):
        raise RuntimeError(
            "Bayesian output-table schemas do not match the registered output inventory : "
            f"missing {sorted(set(OUTPUT_TABLE_FIELDS) - set(output_contract['tables']))} : "
            f"unexpected {sorted(set(output_contract['tables']) - set(OUTPUT_TABLE_FIELDS))}"
        )

    for (
        table_name,
        field_names
    ) in OUTPUT_TABLE_FIELDS.items():
        if (
            not field_names
            or len(
                field_names
            )
            != len(
                set(
                    field_names
                )
            )
        ):
            raise RuntimeError(f"Bayesian output table has an invalid schema : {table_name}")

    for outcome_name, outcome_definition in policy["outcomes"].items():
        for required_outcome_field in (
            "source_field",
            "primary_eligibility_field",
            "sensitivity_eligibility_field",
            "transformation",
            "inverse_transformation",
            "support",
            "interpretation"
        ):
            if required_outcome_field not in outcome_definition:
                raise RuntimeError(
                    f"Bayesian outcome definition is missing a required field : {outcome_name} : {required_outcome_field}"
                )

        if outcome_definition["transformation"] != "natural_logarithm":
            raise RuntimeError(f"Bayesian outcome definition uses an unsupported transformation : {outcome_name}")

        if outcome_definition["support"] != "strictly_positive":
            raise RuntimeError(f"Bayesian outcome definition uses an unsupported support : {outcome_name}")

    # Validating runtime-driving predictive_assessment fields
    predictive_assessment = policy["predictive_assessment"]

    _require_positive_integer(
        predictive_assessment["horizon_bin_count"],
        "predictive_assessment.horizon_bin_count"
    )

    _require_open_probability(
        predictive_assessment["posterior_predictive_interval_probability"],
        "predictive_assessment.posterior_predictive_interval_probability"
    )

    _require_open_probability(
        predictive_assessment["predictive_interval_lower_probability"],
        "predictive_assessment.predictive_interval_lower_probability"
    )

    _require_open_probability(
        predictive_assessment["predictive_interval_upper_probability"],
        "predictive_assessment.predictive_interval_upper_probability"
    )

    if (
        predictive_assessment["predictive_interval_lower_probability"]
        >= predictive_assessment["predictive_interval_upper_probability"]
    ):
        raise RuntimeError(
            "Bayesian policy requires predictive_interval_lower_probability < predictive_interval_upper_probability."
        )

    expected_posterior_predictive_statistics = {
        "equal_cell_weighted_median",
        "equal_cell_weighted_median_absolute_deviation",
        "equal_cell_weighted_minimum",
        "equal_cell_weighted_maximum",
        "equal_cell_weighted_quantile_0.05",
        "equal_cell_weighted_quantile_0.25",
        "equal_cell_weighted_quantile_0.50",
        "equal_cell_weighted_quantile_0.75",
        "equal_cell_weighted_quantile_0.95",
        "equal_cell_weighted_within_cell_first_to_last_change_median"
    }

    if set(predictive_assessment["posterior_predictive_statistics"]) != expected_posterior_predictive_statistics:
        raise RuntimeError(
            "Bayesian policy posterior_predictive_statistics do not match the implemented statistic inventory."
        )

    expected_population_inventory = {
        "all_observations",
        "all_repeated_cell_observations",
        "condition_all_observations",
        "condition_repeated_cell_observations",
        "condition_horizon_bin"
    }

    if set(predictive_assessment["posterior_predictive_population_inventory"]) != expected_population_inventory:
        raise RuntimeError(
            "Bayesian policy posterior_predictive_population_inventory does not match the implemented population inventory."
        )

    _require_true(
        predictive_assessment["posterior_predictive_inventory_must_be_exact"],
        "predictive_assessment.posterior_predictive_inventory_must_be_exact"
    )

    for required_positive_integer_field in (
        "training_fold_minimum_independent_cells_per_condition",
        "training_fold_minimum_repeated_cells_per_condition"
    ):
        _require_positive_integer(
            predictive_assessment[required_positive_integer_field],
            f"predictive_assessment.{required_positive_integer_field}"
        )

    for required_true_field in (
        "training_fold_full_rank_required",
        "training_fold_finite_transformed_values_required",
        "training_fold_same_condition_horizon_coverage_required",
        "strictly_positive_finite_predictions_required",
        "complete_prediction_for_every_valid_test_observation_required",
        "condition_calibration_required",
        "empty_population_disqualifies_candidate_required"
    ):
        if required_true_field not in predictive_assessment:
            continue
        _require_true(
            predictive_assessment[required_true_field],
            f"predictive_assessment.{required_true_field}"
        )

    # Validating runtime-driving trajectory_reporting fields
    trajectory_reporting = policy["trajectory_reporting"]

    _require_positive_integer(
        trajectory_reporting["trajectory_grid_size"],
        "trajectory_reporting.trajectory_grid_size"
    )

    if trajectory_reporting["trajectory_grid_size"] < 2:
        raise RuntimeError("Bayesian policy requires trajectory_grid_size of at least 2.")

    _require_open_probability(
        trajectory_reporting["credible_interval_probability"],
        "trajectory_reporting.credible_interval_probability"
    )

    _require_open_probability(
        trajectory_reporting["continuous_pairwise_difference_reporting"]["credible_interval_probability"],
        "trajectory_reporting.continuous_pairwise_difference_reporting.credible_interval_probability"
    )

    if set(trajectory_reporting["continuous_pairwise_difference_reporting"]["summary_statistics"]) != {
        "posterior_draw_mean",
        "posterior_draw_median",
        "equal_tailed_credible_interval"
    }:
        raise RuntimeError(
            "Bayesian policy continuous_pairwise_difference_reporting summary_statistics do not match the implemented inventory."
        )

    # Validating analytical_qualification runtime-driving fields
    analytical_qualification = policy["predictive_assessment"]["analytical_qualification"]

    for required_qualification_true_field in (
        "prior_predictive_review_acknowledged_required",
        "full_posterior_sampler_diagnostics_completed_required",
        "posterior_predictive_diagnostics_completed_required",
        "grouped_validation_completed_required",
        "predictive_r_squared_completed_required",
        "prediction_intervals_completed_required",
        "condition_calibration_completed_required",
        "strictly_positive_finite_predictions_required"
    ):
        _require_true(
            analytical_qualification[required_qualification_true_field],
            f"predictive_assessment.analytical_qualification.{required_qualification_true_field}"
        )

    _require_exact_value(
        analytical_qualification["failed_qualification_status"],
        "diagnostic_only",
        "predictive_assessment.analytical_qualification.failed_qualification_status"
    )

    _require_exact_value(
        analytical_qualification["reference_prior_multiplier_required"],
        policy["sampling"]["reference_posterior_multiplier"],
        "predictive_assessment.analytical_qualification.reference_prior_multiplier_required"
    )

    # Validating observation-order and accounting field inventories
    _require_exact_value(
        policy["observation_contract"]["candidate_observation_order"],
        [
            "condition_protocol_identity",
            "cell_identity",
            "horizon",
            "source_identity"
        ],
        "observation_contract.candidate_observation_order"
    )

    _require_exact_value(
        policy["observation_contract"]["reconstructed_source_candidate_accounting_fields"],
        [
            "source_record_count",
            "included_record_count",
            "excluded_record_count",
            "missing_horizon_count",
            "nonpositive_horizon_count",
            "missing_outcome_count",
            "cohort_ineligible_count",
            "incomplete_cumulative_fec_count",
            "inside_evaluation_interval_count",
            "outside_evaluation_interval_count"
        ],
        "observation_contract.reconstructed_source_candidate_accounting_fields"
    )

    # Validating output contract implementation binding
    _require_true(
        policy["output_contract"]["implementation_sha256_recorded"],
        "output_contract.implementation_sha256_recorded"
    )

    # Validating execution contract output inventories against output_contract.tables
    all_declared_tables = set(policy["output_contract"]["tables"])

    validate_required_outputs = set(policy["execution_contract"]["validate"]["required_outputs"])
    expected_validate_tables = all_declared_tables - {
        table_name for table_name in all_declared_tables
        if table_name not in {
            "candidate_registry", "observation_accounting",
            "grouped_validation_folds", "bayesian_modeling_report"
        }
    }

    if not expected_validate_tables.issubset(validate_required_outputs):
        raise RuntimeError(
            "Bayesian execution_contract.validate.required_outputs are inconsistent with the declared tables."
        )

    prior_predictive_outputs = set(policy["execution_contract"]["prior_predictive"]["required_outputs"])

    if "prior_predictive_diagnostics" not in prior_predictive_outputs:
        raise RuntimeError(
            "Bayesian execution_contract.prior_predictive.required_outputs must include prior_predictive_diagnostics."
        )

    fit_outputs = set(policy["execution_contract"]["fit"]["required_outputs"])

    if not all_declared_tables.issubset(fit_outputs):
        raise RuntimeError("Bayesian execution_contract.fit.required_outputs must include every declared output_contract table.")

    amendment = _require_exact_fields(
        policy["amendment"],
        {
            "reason",
            "result_informed",
            "result_informed_scope",
            "affected_analyses",
            "complete_rerun_required"
        },
        "Bayesian policy amendment"
    )

    if not isinstance(amendment["reason"], str) or not amendment["reason"].strip():
        raise RuntimeError("Bayesian amendment reason is empty.")

    if not isinstance(amendment["result_informed"], bool):
        raise RuntimeError("Bayesian amendment must contain a boolean result_informed flag.")

    if not isinstance(amendment["result_informed_scope"], str):
        raise RuntimeError("Bayesian amendment must contain a result_informed_scope string.")

    if (
        amendment["result_informed"] is True
        and not amendment["result_informed_scope"].strip()
    ):
        raise RuntimeError("Result-informed Bayesian amendment requires a non-empty result_informed_scope.")

    if (
        not isinstance(amendment["affected_analyses"], list)
        or not amendment["affected_analyses"]
        or len(amendment["affected_analyses"]) != len(set(amendment["affected_analyses"]))
    ):
        raise RuntimeError("Bayesian amendment affected analyses are invalid.")

    if amendment["complete_rerun_required"] is not True:
        raise RuntimeError("Bayesian amendment must require a complete rerun.")


# Validating completion declarations and cross-artifact lineage
def validate_input_lineage(feature_manifest: dict, design_report: dict,
                           identification_report: dict, descriptive_report: dict,
                           feature_manifest_sha256: str, design_report_sha256: str
) -> None:
    if feature_manifest.get(
        "extraction_status"
    ) != "completed":
        raise RuntimeError("Feature extraction is not completed.")

    feature_summary = feature_manifest.get(
        "summary",
        {}
    )

    if (
        feature_summary.get(
            "reference_extraction_failure_count"
        ) != 0
        or feature_summary.get(
            "cycling_extraction_failure_count"
        ) != 0
    ):
        raise RuntimeError("Feature extraction contains execution failures.")

    if design_report.get(
        "audit_status"
    ) != "completed":
        raise RuntimeError("Experimental-design assessment is not completed.")

    if design_report.get(
        "feature_manifest_sha256"
    ) != feature_manifest_sha256:
        raise RuntimeError("Experimental-design input does not match the supplied feature manifest.")

    if design_report.get(
        "analytical_scope_reconciliation",
        {}
    ).get(
        "reconciliation_complete"
    ) is not True:
        raise RuntimeError("Experimental-design analytical scope is not reconciled.")

    if design_report.get(
        "calendar_and_cycle_designs_separated"
    ) is not True:
        raise RuntimeError("Experimental-design aging modes are not separated.")

    if design_report.get(
        "outcome_magnitude_or_direction_used_for_support_selection"
    ) is not False:
        raise RuntimeError("Experimental-design support used outcome magnitude or direction.")

    if identification_report.get(
        "design_report_sha256"
    ) != design_report_sha256:
        raise RuntimeError("Identification input does not match the supplied experimental-design report.")

    if identification_report.get(
        "outcome_values_used_for_identification_decisions"
    ) is not False:
        raise RuntimeError("Identification decisions used outcome values.")

    if identification_report.get(
        "predictive_performance_used_for_identification_decisions"
    ) is not False:
        raise RuntimeError("Identification decisions used predictive performance.")

    descriptive_inputs = descriptive_report.get(
        "inputs",
        {}
    )

    if descriptive_inputs.get(
        "feature_manifest",
        {}
    ).get(
        "sha256"
    ) != feature_manifest_sha256:
        raise RuntimeError("Descriptive-modeling input does not match the supplied feature manifest.")

    if descriptive_inputs.get(
        "experimental_design_report",
        {}
    ).get(
        "sha256"
    ) != design_report_sha256:
        raise RuntimeError("Descriptive-modeling input does not match the supplied experimental-design report.")

    if descriptive_report.get(
        "reconciliation",
        {}
    ).get(
        "reconciliation_complete"
    ) is not True:
        raise RuntimeError("Descriptive-modeling candidates are not reconciled.")

    for separation_field in (
        "calendar_and_cycle_models_separated",
        "primary_and_sensitivity_cohorts_separated"
    ):
        if descriptive_report.get(
            separation_field
        ) is not True:
            raise RuntimeError(f"Descriptive-modeling separation declaration is invalid : {separation_field}")

    if descriptive_report.get(
        "causal_interpretation_authorized"
    ) is not False:
        raise RuntimeError("Descriptive modeling authorizes causal interpretation.")

    if descriptive_report.get(
        "policy_recommendation_authorized"
    ) is not False:
        raise RuntimeError("Descriptive modeling authorizes policy recommendations.")

    if descriptive_report.get(
        "outcome_magnitude_or_direction_used_for_candidate_selection"
    ) is not False:
        raise RuntimeError("Descriptive candidate selection used outcome magnitude or direction.")


# Parsing the analytical input, policy and output locations
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = (
            "Validating and estimating registered Bayesian longitudinal trajectory models."
        )
    )

    parser.add_argument("--feature-manifest-path", type = Path, required = True)
    parser.add_argument("--design-report-path", type = Path, required = True)
    parser.add_argument("--identification-report-path", type = Path, required = True)
    parser.add_argument("--descriptive-report-path", type = Path, required = True)
    parser.add_argument("--modeling-policy-path", type = Path, required = True)
    parser.add_argument("--output-directory", type = Path, required = True)
    parser.add_argument(
        "--cache-directory",
        type = Path,
        default = Path("data/cache/bayesian_models")
    )
    parser.add_argument(
        "--operation",
        choices = (
            "validate",
            "prior_predictive",
            "populate_fit_cache",
            "fit"
        ),
        required = True
    )
    parser.add_argument(
        "--shard-count",
        type = int,
        default = 1
    )
    parser.add_argument(
        "--shard-index",
        type = int,
        default = 0
    )

    return parser.parse_args()


# Rejecting duplicate JSON object fields
def reject_duplicate_json_fields(object_pairs):
    parsed_object = {}

    for field_name, field_value in object_pairs:
        if field_name in parsed_object:
            raise RuntimeError(f"JSON object contains a duplicate field : {field_name}")

        parsed_object[field_name] = field_value

    return parsed_object


# Loading one strict JSON object
def load_json_object(path: Path) -> dict:
    with path.open(mode = "r", encoding = "utf-8") as input_stream:
        parsed_value = json.load(
            input_stream,
            object_pairs_hook = reject_duplicate_json_fields
        )

    if not isinstance(parsed_value, dict):
        raise RuntimeError(f"JSON document is not an object : {path}")

    return parsed_value


# Producing deterministic canonical JSON
def canonical_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii = False,
        allow_nan = False,
        sort_keys = True,
        separators = (",", ":")
    )


# Building one deterministic identity
def deterministic_identity(value) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


# Calculating one file SHA-256 digest
def calculate_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open(mode = "rb") as input_stream:
        while content_chunk := input_stream.read(chunk_size):
            digest.update(content_chunk)

    return digest.hexdigest()


# Reading one CSV table
def read_csv_rows(path: Path):
    with path.open(mode = "r", encoding = "utf-8", newline = "") as input_stream:
        reader = csv.DictReader(input_stream)
        field_names = reader.fieldnames

        if field_names is None:
            raise RuntimeError(f"CSV table does not contain a header : {path}")

        return field_names, list(reader)


# Verifying one declared report table
def verify_report_table(report_path: Path, report: dict, table_name: str):
    table_record = report.get("tables", {}).get(table_name)

    if not isinstance(table_record, dict):
        raise RuntimeError(f"Report does not declare table : {report_path.name} : {table_name}")

    file_name = table_record.get("file_name")
    expected_sha256 = table_record.get("sha256")
    expected_row_count = table_record.get("row_count")

    if not isinstance(file_name, str) or not file_name:
        raise RuntimeError(f"Invalid table file name : {table_name}")

    declared_table_path = Path(
        file_name
    )

    if (
        declared_table_path.is_absolute()
        or declared_table_path.name != file_name
    ):
        raise RuntimeError(f"Report table file name is unsafe : {table_name}")

    table_path = report_path.parent / file_name

    if not table_path.is_file():
        raise RuntimeError(f"Declared table is unavailable : {table_path}")

    if calculate_file_sha256(table_path) != expected_sha256:
        raise RuntimeError(f"Table SHA-256 does not match its report : {table_name}")

    field_names, rows = read_csv_rows(table_path)

    if len(rows) != expected_row_count:
        raise RuntimeError(f"Table row count does not match its report : {table_name}")

    schema_field_names = [entry["name"] for entry in table_record.get("schema", [])]

    if field_names != schema_field_names:
        raise RuntimeError(f"Table schema does not match its report : {table_name}")

    return rows


# Loading every validated input
def load_validated_inputs(feature_manifest_path: Path, design_report_path: Path,
                          identification_report_path: Path, descriptive_report_path: Path,
                          modeling_policy_path: Path
) -> dict:
    policy = load_json_object(modeling_policy_path)

    validate_policy(policy)

    feature_manifest = load_json_object(feature_manifest_path)
    design_report = load_json_object(design_report_path)
    identification_report = load_json_object(identification_report_path)
    descriptive_report = load_json_object(descriptive_report_path)

    observed_feature_manifest_sha256 = (
        calculate_file_sha256(
            feature_manifest_path
        )
    )

    observed_design_report_sha256 = (
        calculate_file_sha256(
            design_report_path
        )
    )

    validate_input_lineage(
        feature_manifest = feature_manifest,
        design_report = design_report,
        identification_report = identification_report,
        descriptive_report = descriptive_report,
        feature_manifest_sha256 = (
            observed_feature_manifest_sha256
        ),
        design_report_sha256 = (
            observed_design_report_sha256
        )
    )

    if identification_report.get("assessment_status") != "completed":
        raise RuntimeError("Identification report is not completed.")

    if identification_report.get("causal_modeling_status") != "not_authorized":
        raise RuntimeError("Identification report authorizes causal modeling.")

    if identification_report.get("associational_hierarchical_modeling_status") not in {
        "partially_authorized",
        "fully_authorized_with_recorded_restrictions"
    }:
        raise RuntimeError("Identification report does not authorize associational modeling.")

    if descriptive_report.get("analysis_status") not in {
        "completed",
        "completed_with_model_family_failures"
    }:
        raise RuntimeError("Descriptive report is not completed.")

    feature_tables = {
        table_name: verify_report_table(
            feature_manifest_path,
            feature_manifest,
            table_name
        )
        for table_name in policy["inputs"]["required_feature_tables"]
    }

    design_tables = {
        table_name: verify_report_table(
            design_report_path,
            design_report,
            table_name
        )
        for table_name in policy["inputs"]["required_design_tables"]
    }

    required_identification_table = policy["inputs"]["required_identification_table"]

    identification_tables = {
        required_identification_table: verify_report_table(
            identification_report_path,
            identification_report,
            required_identification_table
        )
    }

    descriptive_tables = {
        table_name: verify_report_table(
            descriptive_report_path,
            descriptive_report,
            table_name
        )
        for table_name in policy["inputs"]["required_descriptive_tables"]
    }

    return {
        "policy": policy,
        "policy_sha256": calculate_file_sha256(modeling_policy_path),
        "feature_manifest_sha256": calculate_file_sha256(feature_manifest_path),
        "design_report_sha256": calculate_file_sha256(design_report_path),
        "identification_report_sha256": calculate_file_sha256(identification_report_path),
        "descriptive_report_sha256": calculate_file_sha256(descriptive_report_path),
        "feature_tables": feature_tables,
        "design_tables": design_tables,
        "identification_tables": identification_tables,
        "descriptive_tables": descriptive_tables
    }


# Parsing a serialized boolean
def parse_boolean(value, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "true":
        return True
    if value == "false":
        return False
    raise RuntimeError(f"Invalid Boolean value : {field_name} : {value!r}")


# Confirming that a cell matches a declared design scope
def cell_matches_scope(cell: dict, design_scope: str) -> bool:
    if "/" in design_scope:
        dataset_stage, scope_component = design_scope.split("/", 1)
        prefix = "sampling_method="
        if not scope_component.startswith(prefix):
            raise RuntimeError(f"Unsupported design scope : {design_scope}")
        required_sampling = scope_component[len(prefix):]
        return (
            cell["dataset_stage"] == dataset_stage
            and cell["sampling_method"] == required_sampling
        )

    if design_scope not in {"Stage_1", "Stage_2"}:
        raise RuntimeError(f"Unsupported design scope : {design_scope}")

    return cell["dataset_stage"] == design_scope


# Confirming whether one descriptive record meets the registered selection filter
def source_candidate_selected(record: dict, policy: dict) -> bool:
    inventory = policy["candidate_inventory"]

    return (
        record["selection_status"]
        == inventory["required_selection_status"]
        and record["template_name"]
        == inventory["representative_template"]
        and record["context_family"]
        == inventory["required_context_family"]
        and record["estimator"]
        == inventory["representative_estimator"]
    )


# Selecting source candidates from the descriptive registry
def select_source_candidates(descriptive_candidate_records, policy):
    return [
        record
        for record in descriptive_candidate_records
        if source_candidate_selected(record, policy)
    ]


# Confirming whether one horizon lies inside the candidate support
def horizon_inside_support(horizon_value: float, candidate_record: dict) -> bool:
    lower = float(candidate_record["support_interval_minimum"])
    upper = float(candidate_record["support_interval_maximum"])
    lower_inclusive = parse_boolean(
        candidate_record["support_interval_lower_inclusive"], "lower_inclusive"
    )
    upper_inclusive = parse_boolean(
        candidate_record["support_interval_upper_inclusive"], "upper_inclusive"
    )
    lower_ok = horizon_value >= lower if lower_inclusive else horizon_value > lower
    upper_ok = horizon_value <= upper if upper_inclusive else horizon_value < upper
    return lower_ok and upper_ok


# Reconstructing accounting for one source candidate
def reconstruct_source_candidate(source_candidate: dict, validated_inputs: dict) -> dict:
    policy = validated_inputs["policy"]
    feature_tables = validated_inputs["feature_tables"]
    design_tables = validated_inputs["design_tables"]

    region_identity = source_candidate["support_region_identity"]
    design_scope = source_candidate["design_scope"]
    outcome = source_candidate["outcome"]
    cohort = source_candidate["cohort"]
    horizon_axis = source_candidate["horizon_axis"]

    outcome_definition = policy["outcomes"][outcome]
    outcome_field = outcome_definition["source_field"]
    eligibility_field = outcome_definition[
        "primary_eligibility_field"
        if cohort == "primary"
        else "sensitivity_eligibility_field"
    ]

    membership_records = [
        membership
        for membership in design_tables[
            "bracketing_support_membership"
        ]
        if membership[
            "support_region_identity"
        ] == region_identity
    ]

    expected_membership_values = {
        "aging_mode": source_candidate[
            "aging_mode"
        ],
        "outcome": outcome,
        "cohort": cohort,
        "horizon_axis": horizon_axis
    }

    for membership_record in membership_records:
        for field_name, expected_value in (
            expected_membership_values.items()
        ):
            if membership_record[
                field_name
            ] != expected_value:
                raise RuntimeError(f"Support membership disagrees with its candidate : {region_identity} : {field_name}")

        if (
            float(
                membership_record[
                    "lower_horizon"
                ]
            )
            != float(
                source_candidate[
                    "support_interval_minimum"
                ]
            )
            or float(
                membership_record[
                    "upper_horizon"
                ]
            )
            != float(
                source_candidate[
                    "support_interval_maximum"
                ]
            )
            or parse_boolean(
                membership_record[
                    "lower_inclusive"
                ],
                "membership.lower_inclusive"
            )
            != parse_boolean(
                source_candidate[
                    "support_interval_lower_inclusive"
                ],
                "candidate.support_interval_lower_inclusive"
            )
            or parse_boolean(
                membership_record[
                    "upper_inclusive"
                ],
                "membership.upper_inclusive"
            )
            != parse_boolean(
                source_candidate[
                    "support_interval_upper_inclusive"
                ],
                "candidate.support_interval_upper_inclusive"
            )
        ):
            raise RuntimeError(f"Support membership interval disagrees with its candidate : {region_identity}")

    membership_cells = {
        membership_record[
            "cell_identity"
        ]
        for membership_record in membership_records
    }

    cell_records = feature_tables["cell_features"] if "cell_features" in feature_tables else None

    if cell_records is None:
        raise RuntimeError("cell_features table is required but not loaded in feature tables.")

    cell_map = {}

    for cell_record in cell_records:
        cell_identity = cell_record["cell_identity"]

        if cell_identity in cell_map:
            raise RuntimeError(f"Cell feature table contains a duplicate cell identity : {cell_identity}")

        cell_map[cell_identity] = cell_record

    for membership_record in membership_records:
        cell_identity = membership_record[
            "cell_identity"
        ]

        if cell_identity not in cell_map:
            raise RuntimeError(f"Support membership references an unavailable cell : {cell_identity}")

        cell_record = cell_map[
            cell_identity
        ]

        expected_cell_values = {
            "condition_protocol_identity": cell_record[
                "condition_protocol_identity"
            ],
            "dataset_stage": cell_record[
                "dataset_stage"
            ],
            "sampling_method": cell_record[
                "sampling_method"
            ]
        }

        for field_name, expected_value in (
            expected_cell_values.items()
        ):
            if membership_record[
                field_name
            ] != expected_value:
                raise RuntimeError(f"Support membership disagrees with cell metadata : {cell_identity} : {field_name}")

    filtered_cells = {
        cell_identity
        for cell_identity in membership_cells
        if cell_identity in cell_map
        and cell_matches_scope(cell_map[cell_identity], design_scope)
    }

    missing_membership_cells = sorted(
        membership_cells.difference(cell_map)
    )

    if missing_membership_cells:
        raise RuntimeError(
            f"Support-region membership references cells absent from cell_features : {missing_membership_cells}"
        )

    statuses_by_condition = defaultdict(lambda: dict.fromkeys(ACCOUNTING_STATUSES, 0))
    inside_by_condition = defaultdict(int)
    outside_by_condition = defaultdict(int)
    included_by_condition = defaultdict(list)
    observed_source_identities = set()

    for reference_record in feature_tables["reference_test_features"]:
        cell_identity = reference_record["cell_identity"]

        if cell_identity not in filtered_cells:
            continue

        source_identity = reference_record["source_identity"]

        if source_identity in observed_source_identities:
            raise RuntimeError(f"Duplicate source_identity in reconstruction : {source_identity}")

        observed_source_identities.add(source_identity)

        condition_identity = cell_map[cell_identity]["condition_protocol_identity"]
        horizon_text = reference_record.get(horizon_axis, "")
        outcome_text = reference_record.get(outcome_field, "")
        eligibility_text = reference_record.get(eligibility_field, "")

        horizon_value = None
        outcome_value = None

        if horizon_text == "":
            status = "missing_horizon"

        else:
            horizon_value = float(horizon_text)

            if not math.isfinite(horizon_value):
                raise RuntimeError(
                    f"Validated feature input contains a non-finite horizon : {source_identity} : {horizon_axis}"
                )

            if horizon_value <= 0:
                status = "nonpositive_horizon"

            elif outcome_text == "":
                status = "missing_outcome"

            else:
                outcome_value = float(outcome_text)

                if not math.isfinite(outcome_value):
                    raise RuntimeError(
                        f"Validated feature input contains a non-finite outcome : {source_identity} : {outcome_field}"
                    )

                if eligibility_text != "true":
                    status = "cohort_ineligible"

                elif horizon_axis == "cumulative_full_equivalent_cycles_observed":
                    completeness = reference_record.get(
                        "cumulative_fec_observation_complete", ""
                    )
                    status = (
                        "included"
                        if completeness == "true"
                        else "incomplete_cumulative_fec"
                    )

                else:
                    status = "included"

        statuses_by_condition[condition_identity][status] += 1

        if status == "included":
            if horizon_inside_support(horizon_value, source_candidate):
                inside_by_condition[condition_identity] += 1
            else:
                outside_by_condition[condition_identity] += 1

            included_by_condition[condition_identity].append({
                "source_identity": source_identity,
                "cell_identity": cell_identity,
                "condition_protocol_identity": condition_identity,
                "horizon": horizon_value,
                "outcome_value": outcome_value,
                "inside_evaluation_interval": horizon_inside_support(
                    horizon_value, source_candidate
                )
            })

    aggregated_statuses = {status: 0 for status in ACCOUNTING_STATUSES}

    for condition_statuses in statuses_by_condition.values():
        for status in ACCOUNTING_STATUSES:
            aggregated_statuses[status] += condition_statuses[status]

    return {
        "source_candidate_identity": source_candidate["candidate_identity"],
        "source_candidate_record": source_candidate,
        "aggregated_statuses": aggregated_statuses,
        "statuses_by_condition": dict(statuses_by_condition),
        "inside_by_condition": dict(inside_by_condition),
        "outside_by_condition": dict(outside_by_condition),
        "included_by_condition": dict(included_by_condition),
        "filtered_cells": filtered_cells
    }


# Verifying reconstruction against descriptive observation accounting
def verify_reconstruction_against_descriptive(reconstruction: dict, descriptive_accounting_records: list[dict]
) -> tuple[bool, str | None]:
    source_identity = reconstruction["source_candidate_identity"]

    matching_records = [
        record
        for record in descriptive_accounting_records
        if record["candidate_identity"] == source_identity
    ]

    if len(matching_records) != 1:
        return False, "DESCRIPTIVE_ACCOUNTING_MISSING"

    reference = matching_records[0]
    aggregated = reconstruction["aggregated_statuses"]

    expected_values = {
        "source_record_count": sum(aggregated.values()),
        "included_record_count": aggregated["included"],
        "excluded_record_count": sum(
            aggregated[status] for status in ACCOUNTING_STATUSES if status != "included"
        ),
        "missing_horizon_count": aggregated["missing_horizon"],
        "nonpositive_horizon_count": aggregated["nonpositive_horizon"],
        "missing_outcome_count": aggregated["missing_outcome"],
        "cohort_ineligible_count": aggregated["cohort_ineligible"],
        "incomplete_cumulative_fec_count": aggregated["incomplete_cumulative_fec"],
        "inside_evaluation_interval_count": sum(reconstruction["inside_by_condition"].values()),
        "outside_evaluation_interval_count": sum(reconstruction["outside_by_condition"].values())
    }

    for field, expected in expected_values.items():
        if int(reference[field]) != expected:
            return False, f"DESCRIPTIVE_ACCOUNTING_MISMATCH_{field}"

    return True, None


# Collapsing duplicate source candidates per the deduplication contract
def collapse_duplicate_source_candidates(reconstructed_sources: list[dict], policy: dict) -> list[dict]:
    inventory = policy["candidate_inventory"]
    deduplication_fields = inventory["deduplication_fields"]
    required_identical_fields = inventory["duplicate_source_required_identical_fields"]
    authorization_order = inventory["authorization_restriction_order"]

    groups = defaultdict(list)

    for source in reconstructed_sources:
        key = tuple(source["source_candidate_record"][field] for field in deduplication_fields)
        groups[key].append(source)

    collapsed_candidates = []

    for key in sorted(groups):
        sources = groups[key]
        first = sources[0]

        for source in sources[1:]:
            for field in required_identical_fields:
                if source["source_candidate_record"][field] != first["source_candidate_record"][field]:
                    raise RuntimeError(
                        f"Duplicate source candidates disagree on required field "
                        f"{field} : {first['source_candidate_identity']} vs {source['source_candidate_identity']}"
                    )

        first_observation_identities = {
            record["source_identity"]
            for condition_records in first["included_by_condition"].values()
            for record in condition_records
        }

        for source in sources[1:]:
            source_identities = {
                record["source_identity"]
                for condition_records in source["included_by_condition"].values()
                for record in condition_records
            }

            if source_identities != first_observation_identities:
                raise RuntimeError(
                    f"Duplicate source candidates contain different included observations : "
                    f"{first['source_candidate_identity']} vs {source['source_candidate_identity']}"
                )

        authorization_values = [
            source["source_candidate_record"]["authorization_status"]
            for source in sources
        ]

        most_restrictive = max(
            authorization_values,
            key = lambda value: authorization_order.index(value)
        )

        collapsed_candidates.append({
            "support_region_identity": first["source_candidate_record"]["support_region_identity"],
            "design_scope": first["source_candidate_record"]["design_scope"],
            "sampling_design_scope": first["source_candidate_record"]["sampling_design_scope"],
            "aging_mode": first["source_candidate_record"]["aging_mode"],
            "outcome": first["source_candidate_record"]["outcome"],
            "cohort": first["source_candidate_record"]["cohort"],
            "horizon_axis": first["source_candidate_record"]["horizon_axis"],
            "support_interval_minimum": first["source_candidate_record"]["support_interval_minimum"],
            "support_interval_maximum": first["source_candidate_record"]["support_interval_maximum"],
            "support_interval_lower_inclusive": first["source_candidate_record"]["support_interval_lower_inclusive"],
            "support_interval_upper_inclusive": first["source_candidate_record"]["support_interval_upper_inclusive"],
            "most_restrictive_authorization": most_restrictive,
            "source_candidate_identities": sorted(
                source["source_candidate_identity"] for source in sources
            ),
            "reconstruction": first
        })

    return collapsed_candidates


# Building the authorized condition-pair edges for one collapsed candidate
def build_authorized_edges(collapsed_candidate: dict, identification_records: list[dict], policy: dict) -> list[dict]:
    reconstruction = collapsed_candidate["reconstruction"]
    condition_identities = sorted(reconstruction["included_by_condition"].keys())

    aging_mode = collapsed_candidate["aging_mode"]
    outcome = collapsed_candidate["outcome"]
    cohort = collapsed_candidate["cohort"]
    horizon_axis = collapsed_candidate["horizon_axis"]

    allowed_associational_status = policy["authorization"]["required_estimand_status"]
    permitted_support = set(policy["authorization"]["permitted_support_classifications"])

    edges = []

    for first_condition in condition_identities:
        for second_condition in condition_identities:
            if first_condition >= second_condition:
                continue

            matching = [
                record
                for record in identification_records
                if (
                    record["aging_mode"] == aging_mode
                    and record["outcome"] == outcome
                    and record["cohort"] == cohort
                    and record["horizon_axis"] == horizon_axis
                    and record["first_condition_protocol_identity"] == first_condition
                    and record["second_condition_protocol_identity"] == second_condition
                    and record["associational_model_status"] == allowed_associational_status
                    and record["final_support_classification"] in permitted_support
                )
            ]

            if len(matching) == 1:
                edges.append({
                    "first_condition": first_condition,
                    "second_condition": second_condition,
                    "estimand_decision_identity": matching[0]["estimand_decision_identity"]
                })

            elif len(matching) > 1:
                raise RuntimeError(
                    f"Multiple authorized identification records for the same pair : "
                    f"{first_condition} : {second_condition}"
                )

    return edges


# Finding deterministic connected components in the authorized condition graph
def build_connected_components(condition_identities: list[str], edges: list[dict]) -> list[list[str]]:
    adjacency = {condition: set() for condition in condition_identities}

    for edge in edges:
        adjacency[edge["first_condition"]].add(edge["second_condition"])
        adjacency[edge["second_condition"]].add(edge["first_condition"])

    visited = set()
    components = []

    for condition in sorted(condition_identities):
        if condition in visited:
            continue

        component = []
        stack = [condition]

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            component.append(current)

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    stack.append(neighbor)

        components.append(sorted(component))

    return sorted(components)


# Building component-level candidates from one collapsed candidate
def build_component_candidates(collapsed_candidate: dict, identification_records: list[dict]) -> list[dict]:
    reconstruction = collapsed_candidate["reconstruction"]
    condition_identities = sorted(reconstruction["included_by_condition"].keys())

    if not condition_identities:
        return []

    edges = build_authorized_edges(
        collapsed_candidate, identification_records, collapsed_candidate_policy(collapsed_candidate)
    )

    components = build_connected_components(condition_identities, edges)

    candidates = []

    for component in components:
        component_edges = [
            edge
            for edge in edges
            if edge["first_condition"] in component and edge["second_condition"] in component
        ]

        if not component_edges:
            continue

        component_statuses = {status: 0 for status in ACCOUNTING_STATUSES}
        inside_count = 0
        outside_count = 0
        included_observations = []

        for condition_identity in component:
            for status in ACCOUNTING_STATUSES:
                component_statuses[status] += reconstruction["statuses_by_condition"][condition_identity][status]
            inside_count += reconstruction["inside_by_condition"].get(condition_identity, 0)
            outside_count += reconstruction["outside_by_condition"].get(condition_identity, 0)
            included_observations.extend(
                reconstruction["included_by_condition"].get(condition_identity, [])
            )

        candidate_identity_payload = {
            "support_region_identity": collapsed_candidate["support_region_identity"],
            "design_scope": collapsed_candidate["design_scope"],
            "sampling_design_scope": collapsed_candidate["sampling_design_scope"],
            "aging_mode": collapsed_candidate["aging_mode"],
            "outcome": collapsed_candidate["outcome"],
            "cohort": collapsed_candidate["cohort"],
            "horizon_axis": collapsed_candidate["horizon_axis"],
            "sorted_condition_protocol_identities": component,
            "sorted_estimand_decision_identities": sorted(
                edge["estimand_decision_identity"] for edge in component_edges
            )
        }

        candidate_identity = deterministic_identity(candidate_identity_payload)

        candidates.append({
            "candidate_identity": candidate_identity,
            "collapsed_candidate": collapsed_candidate,
            "condition_identities": component,
            "estimand_decision_identities": sorted(
                edge["estimand_decision_identity"] for edge in component_edges
            ),
            "statuses": component_statuses,
            "inside_count": inside_count,
            "outside_count": outside_count,
            "included_observations": included_observations
        })

    return candidates


# Passing policy through the collapsed candidate for edge construction
def collapsed_candidate_policy(collapsed_candidate: dict) -> dict:
    return collapsed_candidate.get("__policy__", {})


# Filtering the reference policy through the edge-authorization context
def attach_policy_to_collapsed_candidates(collapsed_candidates: list[dict], policy: dict) -> None:
    for candidate in collapsed_candidates:
        candidate["__policy__"] = policy


# Building deterministic condition-stratified cell fold assignments
def build_fold_assignments(candidate_identity: str, cells_by_condition: dict[str, set[str]],
                           effective_fold_count: int, seed: int
) -> dict[str, int]:
    assignments = {}

    for condition_identity in sorted(cells_by_condition):
        ranked_cells = sorted(
            cells_by_condition[condition_identity],
            key = lambda cell_identity: (
                deterministic_identity({
                    "candidate_identity": candidate_identity,
                    "condition_protocol_identity": condition_identity,
                    "cell_identity": cell_identity,
                    "seed": seed
                }),
                cell_identity
            )
        )

        for cell_index, cell_identity in enumerate(ranked_cells):
            if cell_identity in assignments:
                raise RuntimeError(
                    "A cell appears in multiple modeled conditions within one candidate : "
                    f"{candidate_identity} : {cell_identity}"
                )

            assignments[cell_identity] = cell_index % effective_fold_count

    return assignments


# Building the fixed-location matrix for the rank check
def build_fixed_location_matrix(observation_records: list[dict], condition_identities: list[str]) -> np.ndarray:
    if not observation_records:
        raise RuntimeError("Fixed-location design requires observations.")

    horizons = np.asarray(
        [
            float(
                observation[
                    "horizon"
                ]
            )
            for observation in observation_records
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

    if (
        not math.isfinite(
            horizon_width
        )
        or horizon_width <= 0
    ):
        raise RuntimeError("Fixed-location design requires a positive horizon width.")

    horizon_midpoint = (
        horizon_minimum
        + horizon_maximum
    ) / 2

    scaled_horizons = (
        horizons
        - horizon_midpoint
    ) / horizon_width

    condition_index = {
        condition_identity: index
        for index, condition_identity
        in enumerate(
            condition_identities
        )
    }

    matrix = np.zeros(
        (
            len(
                observation_records
            ),
            2
            * len(
                condition_identities
            )
        ),
        dtype = float
    )

    for row_index, (
        observation,
        scaled_horizon
    ) in enumerate(
        zip(
            observation_records,
            scaled_horizons,
            strict = True
        )
    ):
        column_index = condition_index[
            observation[
                "condition_protocol_identity"
            ]
        ]

        matrix[
            row_index,
            column_index
        ] = 1.0

        matrix[
            row_index,
            (
                len(
                    condition_identities
                )
                + column_index
            )
        ] = scaled_horizon

    return matrix


# Calculating the SVD-based matrix rank with the registered tolerance
def calculate_matrix_rank(matrix: np.ndarray) -> tuple[int, float]:
    singular_values = np.linalg.svd(matrix, compute_uv = False)

    if singular_values.size == 0:
        return 0, 0.0

    tolerance = (
        max(matrix.shape)
        * singular_values[0]
        * np.finfo(np.float64).eps
    )

    rank = int(np.count_nonzero(singular_values > tolerance))

    return rank, float(tolerance)


# Building per-condition cell, horizon and repetition diagnostics
def build_candidate_structure_diagnostics(candidate: dict) -> dict:
    horizons_by_cell = defaultdict(set)
    cells_by_condition = defaultdict(set)
    repeated_cells_by_condition = defaultdict(set)

    for observation in candidate["included_observations"]:
        cell_identity = observation["cell_identity"]
        condition_identity = observation["condition_protocol_identity"]
        horizon_value = observation["horizon"]

        horizons_by_cell[cell_identity].add(horizon_value)
        cells_by_condition[condition_identity].add(cell_identity)

    for cell_identity, horizons in horizons_by_cell.items():
        if len(horizons) >= 2:
            for condition_identity, condition_cells in cells_by_condition.items():
                if cell_identity in condition_cells:
                    repeated_cells_by_condition[condition_identity].add(cell_identity)

    return {
        "independent_cell_count": len(horizons_by_cell),
        "repeated_cell_count": sum(
            1 for horizons in horizons_by_cell.values() if len(horizons) >= 2
        ),
        "minimum_condition_repeated_cell_count": min(
            (
                len(
                    repeated_cells_by_condition.get(
                        condition_identity,
                        set()
                    )
                )
                for condition_identity in sorted(
                    cells_by_condition
                )
            ),
            default = 0
        ),
        "cells_by_condition": dict(cells_by_condition),
        "horizons_by_cell": dict(horizons_by_cell)
    }


# Evaluating candidate admissibility
def evaluate_candidate_admissibility(candidate: dict, policy: dict) -> dict:
    admissibility = policy["admissibility"]
    diagnostics = build_candidate_structure_diagnostics(candidate)

    condition_count = len(candidate["condition_identities"])
    independent_cell_count = diagnostics["independent_cell_count"]
    repeated_cell_count = diagnostics["repeated_cell_count"]
    minimum_condition_repeated = diagnostics["minimum_condition_repeated_cell_count"]

    min_conditions = admissibility["minimum_conditions"]
    min_cells = admissibility["minimum_independent_cells"]
    min_repeated = admissibility["minimum_repeated_cells"]
    min_repeated_per_condition = admissibility["minimum_repeated_cells_per_condition"]
    min_distinct_horizons = admissibility["minimum_distinct_positive_horizons"]
    min_distinct_horizons_per_condition = admissibility["minimum_distinct_positive_horizons_per_condition"]

    if condition_count < min_conditions:
        return {"admissibility_status": "inadmissible", "admissibility_reason_code": "INSUFFICIENT_CONDITIONS", "diagnostics": diagnostics, "matrix_rank": None, "matrix_column_count": None, "rank_tolerance": None}

    if independent_cell_count < min_cells:
        return {"admissibility_status": "inadmissible", "admissibility_reason_code": "INSUFFICIENT_INDEPENDENT_CELLS", "diagnostics": diagnostics, "matrix_rank": None, "matrix_column_count": None, "rank_tolerance": None}

    if repeated_cell_count < min_repeated:
        return {"admissibility_status": "inadmissible", "admissibility_reason_code": "INSUFFICIENT_REPEATED_CELLS", "diagnostics": diagnostics, "matrix_rank": None, "matrix_column_count": None, "rank_tolerance": None}

    if minimum_condition_repeated < min_repeated_per_condition:
        return {"admissibility_status": "inadmissible", "admissibility_reason_code": "INSUFFICIENT_CONDITION_REPEATED_CELLS", "diagnostics": diagnostics, "matrix_rank": None, "matrix_column_count": None, "rank_tolerance": None}

    distinct_horizons = set(
        observation["horizon"] for observation in candidate["included_observations"]
    )

    if len(distinct_horizons) < min_distinct_horizons:
        return {"admissibility_status": "inadmissible", "admissibility_reason_code": "INSUFFICIENT_DISTINCT_HORIZONS", "diagnostics": diagnostics, "matrix_rank": None, "matrix_column_count": None, "rank_tolerance": None}

    for condition_identity, cells in diagnostics["cells_by_condition"].items():
        if len(cells) < admissibility.get("minimum_independent_cells_per_condition", 2):
            return {"admissibility_status": "inadmissible", "admissibility_reason_code": "INSUFFICIENT_INDEPENDENT_CELLS_PER_CONDITION", "diagnostics": diagnostics, "matrix_rank": None, "matrix_column_count": None, "rank_tolerance": None}

        condition_horizons = set(
            observation["horizon"]
            for observation in candidate["included_observations"]
            if observation["condition_protocol_identity"] == condition_identity
        )

        if len(condition_horizons) < min_distinct_horizons_per_condition:
            return {"admissibility_status": "inadmissible", "admissibility_reason_code": "INSUFFICIENT_DISTINCT_HORIZONS_PER_CONDITION", "diagnostics": diagnostics, "matrix_rank": None, "matrix_column_count": None, "rank_tolerance": None}

    lower = float(candidate["collapsed_candidate"]["support_interval_minimum"])
    upper = float(candidate["collapsed_candidate"]["support_interval_maximum"])
    support_width = upper - lower

    if support_width <= 0:
        return {"admissibility_status": "inadmissible", "admissibility_reason_code": "NONPOSITIVE_SUPPORT_WIDTH", "diagnostics": diagnostics, "matrix_rank": None, "matrix_column_count": None, "rank_tolerance": None}

    matrix = build_fixed_location_matrix(
        observation_records = candidate["included_observations"],
        condition_identities = candidate["condition_identities"]
    )

    observed_rank, rank_tolerance = calculate_matrix_rank(matrix)
    expected_rank = 2 * len(candidate["condition_identities"])

    if admissibility["fixed_location_design_rank_check_required"] and observed_rank != expected_rank:
        return {
            "admissibility_status": "inadmissible",
            "admissibility_reason_code": "MODEL_MATRIX_RANK_DEFICIENT",
            "diagnostics": diagnostics,
            "matrix_rank": observed_rank,
            "matrix_column_count": int(matrix.shape[1]),
            "rank_tolerance": rank_tolerance
        }

    for observation in candidate["included_observations"]:
        if not math.isfinite(observation["outcome_value"]) or observation["outcome_value"] <= 0:
            return {
                "admissibility_status": "inadmissible",
                "admissibility_reason_code": "NONPOSITIVE_OR_NONFINITE_OUTCOME",
                "diagnostics": diagnostics,
                "matrix_rank": observed_rank,
                "matrix_column_count": int(matrix.shape[1]),
                "rank_tolerance": rank_tolerance
            }

    return {
        "admissibility_status": "admissible",
        "admissibility_reason_code": None,
        "diagnostics": diagnostics,
        "matrix_rank": observed_rank,
        "matrix_column_count": int(matrix.shape[1]),
        "rank_tolerance": rank_tolerance
    }


# Evaluating fold admissibility against the registered fold contract
def evaluate_fold_admissibility(candidate: dict, training_observations: list[dict],
                                test_observations: list[dict], policy: dict
) -> tuple[bool, str | None]:
    predictive = policy["predictive_assessment"]

    if not training_observations:
        return False, "EMPTY_TRAINING_OBSERVATIONS"

    if not test_observations:
        return False, "NO_TEST_OBSERVATIONS_IN_EVALUATION_INTERVAL"

    condition_identities = candidate["condition_identities"]

    training_cells_by_condition = defaultdict(set)
    training_horizons_by_condition = defaultdict(set)
    training_horizons_by_cell = defaultdict(set)

    for observation in training_observations:
        horizon = observation["horizon"]
        outcome = observation["outcome_value"]

        if (
            not math.isfinite(horizon)
            or not math.isfinite(outcome)
            or horizon <= 0
            or outcome <= 0
        ):
            return False, "NONFINITE_OR_NONPOSITIVE_TRAINING_VALUE"

        condition_identity = observation["condition_protocol_identity"]
        cell_identity = observation["cell_identity"]

        training_cells_by_condition[condition_identity].add(cell_identity)
        training_horizons_by_condition[condition_identity].add(horizon)
        training_horizons_by_cell[cell_identity].add(horizon)

    if set(training_cells_by_condition) != set(condition_identities):
        return False, "TRAINING_CONDITION_MISSING"

    repeated_cells = {
        cell_identity
        for cell_identity, horizons in training_horizons_by_cell.items()
        if len(horizons) >= 2
    }

    for condition_identity in condition_identities:
        condition_cells = training_cells_by_condition[condition_identity]

        if len(condition_cells) < predictive["training_fold_minimum_independent_cells_per_condition"]:
            return False, "INSUFFICIENT_TRAINING_CELLS_PER_CONDITION"

        if len(condition_cells.intersection(repeated_cells)) < predictive["training_fold_minimum_repeated_cells_per_condition"]:
            return False, "INSUFFICIENT_TRAINING_REPEATED_CELLS_PER_CONDITION"

    lower = float(candidate["collapsed_candidate"]["support_interval_minimum"])
    upper = float(candidate["collapsed_candidate"]["support_interval_maximum"])
    support_width = upper - lower

    if support_width <= 0:
        return False, "NONPOSITIVE_SUPPORT_WIDTH"

    matrix = build_fixed_location_matrix(
        observation_records = training_observations,
        condition_identities = condition_identities
    )

    observed_rank, _ = calculate_matrix_rank(matrix)

    if observed_rank != 2 * len(condition_identities):
        return False, "TRAINING_MODEL_MATRIX_RANK_DEFICIENT"

    if predictive["training_fold_same_condition_horizon_coverage_required"]:
        for test_observation in test_observations:
            condition_identity = test_observation["condition_protocol_identity"]
            test_horizon = test_observation["horizon"]

            condition_horizons = training_horizons_by_condition.get(condition_identity)

            if not condition_horizons:
                return False, "TEST_CONDITION_MISSING_FROM_TRAINING"

            if not (min(condition_horizons) <= test_horizon <= max(condition_horizons)):
                return False, "TEST_HORIZON_OUTSIDE_CONDITION_TRAINING_RANGE"

    return True, None


# Deriving the candidate-level grouped-validation status from its fold summary
def _compute_grouped_validation_status(fold_summary: dict) -> str:
    if fold_summary["effective_fold_count"] == 0:
        return "unavailable"
    if fold_summary["unavailable_fold_count"] != 0:
        return "unavailable"
    return "available"


# Deriving the candidate-level grouped-validation reason code from its fold summary
def _compute_grouped_validation_reason(fold_summary: dict) -> str | None:
    if fold_summary["effective_fold_count"] == 0:
        return "INSUFFICIENT_EFFECTIVE_FOLDS"
    if fold_summary["unavailable_fold_count"] != 0:
        return "ONE_OR_MORE_FOLDS_UNAVAILABLE"
    return None


# Building fold records for one admissible candidate
def build_grouped_validation_folds(candidate: dict, policy: dict) -> tuple[list[dict], dict]:
    predictive = policy["predictive_assessment"]
    seed = predictive["fold_assignment_seed"]
    diagnostics = candidate["admissibility"]["diagnostics"]
    cells_by_condition = diagnostics["cells_by_condition"]

    if not cells_by_condition:
        return (
            [_build_unavailable_fold_record(candidate["candidate_identity"], 0, "NO_MODELED_CONDITIONS")],
            {"effective_fold_count": 0, "available_fold_count": 0, "unavailable_fold_count": 1}
        )

    effective_fold_count = min(
        predictive["requested_fold_count"],
        min(len(cells) for cells in cells_by_condition.values())
    )

    if effective_fold_count < predictive["minimum_effective_fold_count"]:
        return (
            [_build_unavailable_fold_record(
                candidate["candidate_identity"], 0, "INSUFFICIENT_CELLS_FOR_GROUPED_VALIDATION"
            )],
            {"effective_fold_count": 0, "available_fold_count": 0, "unavailable_fold_count": 1}
        )

    assignments = build_fold_assignments(
        candidate["candidate_identity"], cells_by_condition, effective_fold_count, seed
    )

    fold_records = []
    available_fold_count = 0
    unavailable_fold_count = 0

    for fold_index in range(effective_fold_count):
        test_cell_identities = sorted(
            cell_identity
            for cell_identity, assigned_fold in assignments.items()
            if assigned_fold == fold_index
        )

        training_observations = [
            observation
            for observation in candidate["included_observations"]
            if assignments.get(observation["cell_identity"]) != fold_index
        ]

        test_observations = [
            observation
            for observation in candidate["included_observations"]
            if (
                assignments.get(observation["cell_identity"]) == fold_index
                and observation["inside_evaluation_interval"]
            )
        ]

        is_admissible, reason_code = evaluate_fold_admissibility(
            candidate = candidate,
            training_observations = training_observations,
            test_observations = test_observations,
            policy = policy
        )

        training_cells_by_condition = defaultdict(set)
        horizons_by_cell = defaultdict(set)

        for observation in training_observations:
            training_cells_by_condition[observation["condition_protocol_identity"]].add(
                observation["cell_identity"]
            )
            horizons_by_cell[observation["cell_identity"]].add(observation["horizon"])

        repeated_cells = {
            cell_identity
            for cell_identity, horizons in horizons_by_cell.items()
            if len(horizons) >= 2
        }

        training_condition_count = len(training_cells_by_condition)
        training_min_condition_cells = min(
            (len(cells) for cells in training_cells_by_condition.values()), default = 0
        )
        training_min_repeated_cells = min(
            (
                len(cells.intersection(repeated_cells))
                for cells in training_cells_by_condition.values()
            ),
            default = 0
        )

        if is_admissible:
            available_fold_count += 1
            fold_status = "available"
        else:
            unavailable_fold_count += 1
            fold_status = "unavailable"

        fold_records.append({
            "candidate_identity": candidate["candidate_identity"],
            "fold_index": fold_index,
            "training_cell_count": len(set(
                observation["cell_identity"] for observation in training_observations
            )),
            "test_cell_count": len(test_cell_identities),
            "training_cell_identities": sorted(set(
                observation["cell_identity"] for observation in training_observations
            )),
            "test_cell_identities": test_cell_identities,
            "training_observation_count": len(training_observations),
            "test_observation_count": len(test_observations),
            "training_condition_count": training_condition_count,
            "training_minimum_condition_cell_count": training_min_condition_cells,
            "training_minimum_condition_repeated_cell_count": training_min_repeated_cells,
            "fold_status": fold_status,
            "fold_reason_code": reason_code
        })

    return (
        fold_records,
        {
            "effective_fold_count": effective_fold_count,
            "available_fold_count": available_fold_count,
            "unavailable_fold_count": unavailable_fold_count
        }
    )


# Building one placeholder fold record for an unavailable candidate
def _build_unavailable_fold_record(candidate_identity: str, fold_index: int, reason_code: str) -> dict:
    return {
        "candidate_identity": candidate_identity,
        "fold_index": fold_index,
        "training_cell_count": 0,
        "test_cell_count": 0,
        "training_cell_identities": [],
        "test_cell_identities": [],
        "training_observation_count": 0,
        "test_observation_count": 0,
        "training_condition_count": 0,
        "training_minimum_condition_cell_count": 0,
        "training_minimum_condition_repeated_cell_count": 0,
        "fold_status": "unavailable",
        "fold_reason_code": reason_code
    }


# Building the workload plan
def build_execution_plan(candidate_records: list[dict], fold_records: list[dict], policy: dict) -> dict:
    prior_multipliers = len(policy["priors"]["prior_sensitivity_scale_multipliers"])
    posterior_multipliers = len(policy["sampling"]["full_data_posterior_scale_multipliers"])
    validation_multipliers = len(policy["sampling"]["grouped_validation_scale_multipliers"])

    admissible_candidate_records = [
        candidate_record
        for candidate_record in candidate_records
        if candidate_record["admissibility_status"] == "admissible"
    ]

    validation_feasible_identities = {
        candidate_record["candidate_identity"]
        for candidate_record in admissible_candidate_records
        if candidate_record["grouped_validation_status"] == "available"
    }

    admissible_candidate_count = len(admissible_candidate_records)

    prior_predictive_call_count = admissible_candidate_count * prior_multipliers
    full_posterior_call_count = admissible_candidate_count * posterior_multipliers

    validation_fold_count = sum(
        1
        for fold_record in fold_records
        if (
            fold_record["candidate_identity"] in validation_feasible_identities
            and fold_record["fold_status"] == "available"
        )
    )

    grouped_validation_posterior_call_count = validation_fold_count * validation_multipliers

    nuts_upper_bound = full_posterior_call_count + grouped_validation_posterior_call_count
    posterior_predictive_forward_upper_bound = nuts_upper_bound

    return {
        "admissible_candidate_count": admissible_candidate_count,
        "validation_feasible_candidate_count": len(validation_feasible_identities),
        "prior_predictive_call_count": prior_predictive_call_count,
        "full_data_posterior_call_count": full_posterior_call_count,
        "grouped_validation_posterior_call_count": grouped_validation_posterior_call_count,
        "nuts_sampler_call_upper_bound": nuts_upper_bound,
        "posterior_predictive_forward_call_upper_bound": posterior_predictive_forward_upper_bound,
        "all_stochastic_call_upper_bound": (
            prior_predictive_call_count
            + nuts_upper_bound
            + posterior_predictive_forward_upper_bound
        )
    }


# Serializing one CSV value
def serialize_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return canonical_json(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("Cannot serialize non-finite numerical value.")
        return repr(value)
    return value


# Serializing expected records exactly as deterministic CSV rows
def serialize_expected_csv_rows(columns: list[str], records: list[dict]) -> list[dict]:
    serialized_records = []

    for record in records:
        if set(record) != set(columns):
            raise RuntimeError("Expected record does not match its declared schema.")

        serialized_records.append(
            {
                column_name: str(
                    serialize_csv_value(
                        record[column_name]
                    )
                )
                for column_name in columns
            }
        )

    return serialized_records


# Writing one deterministic CSV table
def write_csv_table(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open(mode = "w", encoding = "utf-8", newline = "") as output_stream:
        writer = csv.DictWriter(output_stream, fieldnames = columns, lineterminator = "\n")
        writer.writeheader()

        for row in rows:
            if set(row) != set(columns):
                raise RuntimeError(f"Row does not match its schema : {path.name}")

            writer.writerow({
                column_name: serialize_csv_value(row[column_name])
                for column_name in columns
            })


# Confirming a reason code emitted at runtime is registered in the policy
def _validate_registered_reason_code(reason_code, category_name: str, policy: dict) -> None:
    if reason_code is None:
        return

    registered_codes = set(
        policy["reason_code_contract"][category_name]
    )

    if reason_code not in registered_codes:
        raise RuntimeError(f"Unregistered {category_name} reason code : {reason_code}")


# Confirming every reason code in one candidate record is registered
def _validate_candidate_reason_codes(candidate_record: dict, policy: dict) -> None:
    if candidate_record["admissibility_status"] == "inadmissible":
        _validate_registered_reason_code(
            candidate_record["admissibility_reason_code"],
            "candidate_admissibility",
            policy
        )

    if candidate_record["grouped_validation_status"] == "unavailable":
        _validate_registered_reason_code(
            candidate_record["grouped_validation_reason_code"],
            "grouped_validation",
            policy
        )


# Confirming every emitted reason code across analytical outputs is registered
def _validate_all_reason_codes(fit_outputs: dict, policy: dict) -> None:
    record_families = [
        ("sampler_diagnostics", fit_outputs["sampler_diagnostics"], "diagnostic_reason_code"),
        ("posterior_predictive_execution", fit_outputs["posterior_predictive_diagnostics"], "diagnostic_reason_code"),
        ("grouped_validation_execution", fit_outputs["grouped_validation_predictions"], "prediction_reason_code"),
        ("grouped_validation_execution", fit_outputs["grouped_validation_metrics"], "validation_reason_code"),
        ("grouped_validation_execution", fit_outputs["grouped_validation_condition_calibration"], "calibration_reason_code"),
        ("trajectory_reporting", fit_outputs["continuous_trajectory_summaries"], "analytical_reason_code"),
        ("trajectory_reporting", fit_outputs["continuous_trajectory_difference_summaries"], "analytical_reason_code"),
        ("analytical_qualification", fit_outputs["prior_sensitivity_summaries"], "sensitivity_reason_code")
    ]

    for category_name, records, field_name in record_families:
        for record in records:
            reason_code = record.get(field_name)

            if reason_code is None:
                continue

            _validate_registered_reason_code(
                reason_code,
                category_name,
                policy
            )

    for posterior_summary in fit_outputs["posterior_summaries"]:
        reason_code = posterior_summary.get("summary_reason_code")

        if reason_code is None:
            continue

        if reason_code == "POSTERIOR_SAMPLING_COMPLETED":
            _validate_registered_reason_code(
                reason_code,
                "posterior_sampling",
                policy
            )
        elif reason_code == "SAMPLER_NONFINITE_DRAWS":
            _validate_registered_reason_code(
                reason_code,
                "sampler_diagnostics",
                policy
            )
        else:
            raise RuntimeError(f"Unregistered posterior summary reason code : {reason_code}")

    for posterior_record in fit_outputs["posterior_records"]:
        fit_status = posterior_record.get("fit_status")

        if fit_status == "completed":
            _validate_registered_reason_code(
                "POSTERIOR_SAMPLING_COMPLETED",
                "posterior_sampling",
                policy
            )
        elif fit_status == "fit_failed":
            _validate_registered_reason_code(
                "POSTERIOR_SAMPLING_FAILED",
                "posterior_sampling",
                policy
            )


# Confirming every fold reason code is registered
def _validate_fold_reason_codes(fold_records: list[dict], policy: dict) -> None:
    for fold_record in fold_records:
        _validate_registered_reason_code(
            fold_record["fold_reason_code"],
            "grouped_validation",
            policy
        )


# Building one table manifest record
def build_table_manifest(path: Path, columns: list[str], row_count: int) -> dict:
    return {
        "file_name": path.name,
        "row_count": row_count,
        "sha256": calculate_file_sha256(path),
        "schema": [{"name": column_name} for column_name in columns]
    }


# Building the validate-operation artifact set
def build_validation_workflow(validated_inputs: dict) -> dict:
    policy = validated_inputs["policy"]

    source_candidates = select_source_candidates(
        validated_inputs["descriptive_tables"]["candidate_registry"],
        policy
    )

    if not source_candidates:
        raise RuntimeError("No source candidates matched the registered selection filter.")

    reconstructed_sources = []

    for source_candidate in source_candidates:
        reconstruction = reconstruct_source_candidate(source_candidate, validated_inputs)

        is_reconciled, reason = verify_reconstruction_against_descriptive(
            reconstruction,
            validated_inputs["descriptive_tables"]["observation_accounting"]
        )

        if is_reconciled:
            reconstructed_sources.append(reconstruction)
        else:
            raise RuntimeError(
                f"Source-candidate accounting does not reconcile with descriptive output : "
                f"{source_candidate['candidate_identity']} : {reason}"
            )

    collapsed_candidates = collapse_duplicate_source_candidates(
        reconstructed_sources, policy
    )

    attach_policy_to_collapsed_candidates(collapsed_candidates, policy)

    identification_records = validated_inputs["identification_tables"]["estimand_identification"]

    component_candidates = []

    for collapsed_candidate in collapsed_candidates:
        component_candidates.extend(
            build_component_candidates(collapsed_candidate, identification_records)
        )

    candidate_records = []
    observation_accounting_records = []
    all_fold_records = []

    for candidate in component_candidates:
        admissibility = evaluate_candidate_admissibility(candidate, policy)
        candidate["admissibility"] = admissibility

        if admissibility["admissibility_status"] == "admissible":
            fold_records, fold_summary = build_grouped_validation_folds(candidate, policy)
        else:
            fold_records = []
            fold_summary = {
                "effective_fold_count": 0,
                "available_fold_count": 0,
                "unavailable_fold_count": 0
            }

        all_fold_records.extend(fold_records)

        candidate_records.append({
            "candidate_identity": candidate["candidate_identity"],
            "support_region_identity": candidate["collapsed_candidate"]["support_region_identity"],
            "design_scope": candidate["collapsed_candidate"]["design_scope"],
            "sampling_design_scope": candidate["collapsed_candidate"]["sampling_design_scope"],
            "aging_mode": candidate["collapsed_candidate"]["aging_mode"],
            "outcome": candidate["collapsed_candidate"]["outcome"],
            "cohort": candidate["collapsed_candidate"]["cohort"],
            "horizon_axis": candidate["collapsed_candidate"]["horizon_axis"],
            "support_interval_minimum": float(candidate["collapsed_candidate"]["support_interval_minimum"]),
            "support_interval_maximum": float(candidate["collapsed_candidate"]["support_interval_maximum"]),
            "support_interval_lower_inclusive": parse_boolean(
                candidate["collapsed_candidate"]["support_interval_lower_inclusive"], "lower_inclusive"
            ),
            "support_interval_upper_inclusive": parse_boolean(
                candidate["collapsed_candidate"]["support_interval_upper_inclusive"], "upper_inclusive"
            ),
            "sorted_condition_protocol_identities": candidate["condition_identities"],
            "sorted_estimand_decision_identities": candidate["estimand_decision_identities"],
            "source_candidate_identities": candidate["collapsed_candidate"]["source_candidate_identities"],
            "source_authorization_status": candidate["collapsed_candidate"]["most_restrictive_authorization"],
            "condition_count": len(candidate["condition_identities"]),
            "independent_cell_count": admissibility["diagnostics"]["independent_cell_count"],
            "observation_count": len(candidate["included_observations"]),
            "inside_evaluation_interval_count": candidate["inside_count"],
            "outside_evaluation_interval_count": candidate["outside_count"],
            "repeated_cell_count": admissibility["diagnostics"]["repeated_cell_count"],
            "minimum_condition_repeated_cell_count": admissibility["diagnostics"]["minimum_condition_repeated_cell_count"],
            "fixed_location_matrix_rank": admissibility["matrix_rank"],
            "fixed_location_matrix_column_count": admissibility["matrix_column_count"],
            "fixed_location_rank_tolerance": admissibility["rank_tolerance"],
            "admissibility_status": admissibility["admissibility_status"],
            "admissibility_reason_code": admissibility["admissibility_reason_code"],
            "candidate_execution_status": (
                "admissible"
                if admissibility["admissibility_status"] == "admissible"
                else "inadmissible"
            ),
            "posterior_analytical_status": "not_available",
            "posterior_analytical_reason_code": "ANALYTICAL_QUALIFICATION_NOT_AVAILABLE",
            "posterior_qualification_failed_gates": [],
            "posterior_status_by_prior_multiplier": {},
            "posterior_reason_codes_by_prior_multiplier": {},
            "effective_fold_count": fold_summary["effective_fold_count"],
            "available_fold_count": fold_summary["available_fold_count"],
            "unavailable_fold_count": fold_summary["unavailable_fold_count"],
            "grouped_validation_status": _compute_grouped_validation_status(fold_summary),
            "grouped_validation_reason_code": _compute_grouped_validation_reason(fold_summary)
        })

        if (
            candidate_records[-1][
                "source_authorization_status"
            ]
            not in set(
                policy[
                    "candidate_inventory"
                ][
                    "authorization_restriction_order"
                ]
            )
        ):
            raise RuntimeError(
                "Candidate contains an unregistered source authorization status : "
                f"{candidate_records[-1]['source_authorization_status']}"
            )

        _validate_candidate_reason_codes(
            candidate_records[-1],
            policy
        )

        observation_accounting_records.append({
            "candidate_identity": candidate["candidate_identity"],
            "source_record_count": sum(candidate["statuses"].values()),
            "included_record_count": candidate["statuses"]["included"],
            "excluded_record_count": sum(
                candidate["statuses"][status]
                for status in ACCOUNTING_STATUSES
                if status != "included"
            ),
            "missing_horizon_count": candidate["statuses"]["missing_horizon"],
            "nonpositive_horizon_count": candidate["statuses"]["nonpositive_horizon"],
            "missing_outcome_count": candidate["statuses"]["missing_outcome"],
            "cohort_ineligible_count": candidate["statuses"]["cohort_ineligible"],
            "incomplete_cumulative_fec_count": candidate["statuses"]["incomplete_cumulative_fec"],
            "inside_evaluation_interval_count": candidate["inside_count"],
            "outside_evaluation_interval_count": candidate["outside_count"]
        })

    _validate_fold_reason_codes(
        all_fold_records,
        policy
    )

    execution_plan = build_execution_plan(
        candidate_records = sorted(candidate_records, key = lambda record: record["candidate_identity"]),
        fold_records = all_fold_records,
        policy = policy
    )

    return {
        "validated_inputs": validated_inputs,
        "source_candidates": source_candidates,
        "reconstructed_sources": reconstructed_sources,
        "collapsed_candidates": collapsed_candidates,
        "component_candidates": component_candidates,
        "candidate_records": sorted(candidate_records, key = lambda record: record["candidate_identity"]),
        "observation_accounting_records": sorted(
            observation_accounting_records, key = lambda record: record["candidate_identity"]
        ),
        "fold_records": sorted(
            all_fold_records, key = lambda record: (record["candidate_identity"], record["fold_index"])
        ),
        "execution_plan": execution_plan
    }


# Building the validate-operation report
def build_validation_report(workflow: dict, produced_table_names: list[str]) -> dict:
    validated_inputs = workflow["validated_inputs"]
    policy = validated_inputs["policy"]

    candidate_records = workflow["candidate_records"]
    admissible = [record for record in candidate_records if record["admissibility_status"] == "admissible"]
    inadmissible = [record for record in candidate_records if record["admissibility_status"] == "inadmissible"]

    admissibility_reason_counts = {}
    for record in inadmissible:
        reason = record["admissibility_reason_code"]
        admissibility_reason_counts[reason] = admissibility_reason_counts.get(reason, 0) + 1

    all_declared_tables = list(policy["output_contract"]["tables"])

    return {
        "assessment_status": "completed",
        "operation": "validate",
        "policy_version": policy["policy_version"],
        "policy_effective_date": policy["effective_date"],
        "policy_sha256": validated_inputs["policy_sha256"],
        "implementation_sha256": calculate_file_sha256(
            Path(__file__).resolve()
        ),
        "policy": policy,
        "input_hashes": {
            "feature_manifest_sha256": validated_inputs["feature_manifest_sha256"],
            "design_report_sha256": validated_inputs["design_report_sha256"],
            "identification_report_sha256": validated_inputs["identification_report_sha256"],
            "descriptive_report_sha256": validated_inputs["descriptive_report_sha256"]
        },
        "candidate_count": len(candidate_records),
        "admissible_candidate_count": len(admissible),
        "inadmissible_candidate_count": len(inadmissible),
        "admissibility_reason_counts": admissibility_reason_counts,
        "prior_predictive_review_required": False,
        "posterior_sampling_performed": False,
        "pymc_runtime_loaded": False,
        "execution_plan": workflow["execution_plan"],
        "produced_tables": list(produced_table_names),
        "not_produced_tables": [
            table_name for table_name in all_declared_tables
            if table_name not in set(produced_table_names)
        ],
        "tables": {}
    }


# Validating one staged or published Bayesian-report artifact set
def validate_bayesian_report(report_path: Path, expected_policy: dict,
                             expected_policy_sha256: str, expected_input_hashes: dict,
                             expected_candidate_records: list[dict], expected_accounting_records: list[dict],
                             expected_fold_records: list[dict], expected_execution_plan: dict,
                             expected_operation: str, expected_additional_tables: dict | None = None,
                             expected_review_template: dict | None = None, expected_execution_accounting: dict | None = None,
                             expected_fit_report_fields: dict | None = None
) -> None:
    report = load_json_object(report_path)

    if report.get("assessment_status") != "completed":
        raise RuntimeError("Bayesian report is not completed.")

    if report.get("operation") != expected_operation:
        raise RuntimeError("Bayesian report has an inconsistent operation.")

    if report.get("policy") != expected_policy:
        raise RuntimeError("Bayesian report does not embed the registered policy.")

    if report.get("policy_sha256") != expected_policy_sha256:
        raise RuntimeError("Bayesian report has an inconsistent policy digest.")

    if report.get("implementation_sha256") != calculate_file_sha256(
        Path(__file__).resolve()
    ):
        raise RuntimeError("Bayesian report has an inconsistent implementation digest.")

    if report.get("input_hashes") != expected_input_hashes:
        raise RuntimeError("Bayesian report has inconsistent input hashes.")

    if report.get("execution_plan") != expected_execution_plan:
        raise RuntimeError("Bayesian report has an inconsistent execution plan.")

    expected_review_required = expected_operation in {"prior_predictive", "fit"}

    if report.get("prior_predictive_review_required") != expected_review_required:
        raise RuntimeError("Bayesian report has an inconsistent prior_predictive_review_required flag.")

    if expected_operation == "validate":
        if report.get("pymc_runtime_loaded") is not False:
            raise RuntimeError("Bayesian report claims PyMC runtime was loaded during validate.")
        if report.get("posterior_sampling_performed") is not False:
            raise RuntimeError("Bayesian report claims posterior sampling under validate.")

    if expected_operation == "prior_predictive":
        if report.get(
            "pymc_runtime_loaded"
        ) is not True:
            raise RuntimeError("Prior-predictive report does not record the loaded PyMC runtime.")

        if report.get(
            "posterior_sampling_performed"
        ) is not False:
            raise RuntimeError("Prior-predictive report claims posterior sampling.")

        if (
            report.get(
                "prior_predictive_execution_accounting"
            )
            != expected_execution_accounting
        ):
            raise RuntimeError("Prior-predictive execution accounting differs from the validated execution.")

    if expected_operation == "fit":
        if expected_fit_report_fields is None:
            raise RuntimeError("Fit-report validation requires expected fit fields.")

        for field_name, expected_value in expected_fit_report_fields.items():
            if field_name == "posterior_artifacts":
                continue

            if report.get(field_name) != expected_value:
                raise RuntimeError(f"Bayesian fit report has an inconsistent field : {field_name}")

        posterior_manifest = report.get("posterior_artifacts")

        if not isinstance(posterior_manifest, list):
            raise RuntimeError("Bayesian fit report has an invalid posterior manifest.")

        if "posterior_artifacts" not in expected_fit_report_fields:
            raise RuntimeError("Fit-report validation requires the expected posterior artifact manifest.")

        expected_artifacts = expected_fit_report_fields["posterior_artifacts"]

        if len(posterior_manifest) != len(expected_artifacts):
            raise RuntimeError("Bayesian posterior manifest does not contain every expected artifact.")

        if posterior_manifest != expected_artifacts:
            raise RuntimeError("Bayesian fit report posterior manifest does not match the expected manifest.")

        observed_artifact_keys = set()

        for artifact_record in posterior_manifest:
            _require_exact_fields(
                artifact_record,
                {
                    "candidate_identity",
                    "prior_scale_multiplier",
                    "file_name",
                    "sha256",
                    "size_bytes"
                },
                "Bayesian posterior artifact manifest"
            )
            artifact_key = (
                artifact_record["candidate_identity"],
                float(artifact_record["prior_scale_multiplier"])
            )

            if artifact_key in observed_artifact_keys:
                raise RuntimeError("Bayesian posterior manifest contains a duplicate artifact.")

            observed_artifact_keys.add(artifact_key)
            artifact_path = report_path.parent / artifact_record["file_name"]

            if not artifact_path.is_file():
                raise RuntimeError("Declared Bayesian posterior artifact is missing.")

            if artifact_path.stat().st_size != int(artifact_record["size_bytes"]):
                raise RuntimeError("Bayesian posterior artifact size mismatch.")

            if calculate_file_sha256(artifact_path) != artifact_record["sha256"]:
                raise RuntimeError("Bayesian posterior artifact SHA-256 mismatch.")

    table_manifest = report.get("tables", {})

    if set(table_manifest) != set(report.get("produced_tables", [])):
        raise RuntimeError("Bayesian report table manifest does not match produced_tables.")

    for table_name, table_record in table_manifest.items():
        table_path = report_path.parent / table_record["file_name"]

        if not table_path.is_file():
            raise RuntimeError(f"Declared table is missing : {table_name}")

        if calculate_file_sha256(table_path) != table_record["sha256"]:
            raise RuntimeError(f"Table SHA-256 mismatch : {table_name}")

        field_names, rows = read_csv_rows(table_path)

        if len(rows) != table_record["row_count"]:
            raise RuntimeError(f"Table row count mismatch : {table_name}")

        expected_schema_names = [entry["name"] for entry in table_record["schema"]]
        if field_names != expected_schema_names:
            raise RuntimeError(f"Table schema mismatch : {table_name}")

    expected_tables = {
        "candidate_registry": (
            CANDIDATE_REGISTRY_FIELDS,
            expected_candidate_records
        ),
        "observation_accounting": (
            OBSERVATION_ACCOUNTING_FIELDS,
            expected_accounting_records
        ),
        "grouped_validation_folds": (
            GROUPED_VALIDATION_FOLDS_FIELDS,
            expected_fold_records
        )
    }

    if expected_additional_tables is not None:
        unexpected_overlap = set(
            expected_tables
        ).intersection(
            expected_additional_tables
        )

        if unexpected_overlap:
            raise RuntimeError(f"Additional Bayesian tables overlap required base tables : {sorted(unexpected_overlap)}")

        expected_tables.update(
            expected_additional_tables
        )

    if set(table_manifest) != set(expected_tables):
        raise RuntimeError("Bayesian report contains an unexpected table inventory.")

    expected_produced_tables = list(expected_tables)

    if report.get("produced_tables") != expected_produced_tables:
        raise RuntimeError("Produced-table inventory differs from the operation contract.")

    expected_not_produced_tables = [
        table_name
        for table_name in expected_policy["output_contract"]["tables"]
        if table_name not in expected_tables
    ]

    if report.get("not_produced_tables") != expected_not_produced_tables:
        raise RuntimeError("Not-produced table inventory differs from the operation contract.")

    parsed_tables = {}

    for table_name, (expected_columns, expected_records) in expected_tables.items():
        table_record = table_manifest[table_name]

        _require_exact_fields(
            table_record,
            {"file_name", "row_count", "sha256", "schema"},
            f"Bayesian table manifest : {table_name}"
        )

        expected_file_name = f"{table_name}.csv"

        if table_record["file_name"] != expected_file_name:
            raise RuntimeError(f"Unexpected Bayesian table file name : {table_name}")

        expected_schema = [
            {"name": column_name} for column_name in expected_columns
        ]

        if table_record["schema"] != expected_schema:
            raise RuntimeError(f"Bayesian table manifest schema differs from its contract : {table_name}")

        table_path = report_path.parent / expected_file_name

        if not table_path.is_file():
            raise RuntimeError(f"Declared Bayesian table is missing : {table_name}")

        if calculate_file_sha256(table_path) != table_record["sha256"]:
            raise RuntimeError(f"Bayesian table SHA-256 mismatch : {table_name}")

        field_names, observed_records = read_csv_rows(table_path)

        if field_names != expected_columns:
            raise RuntimeError(f"Bayesian table columns differ from their contract : {table_name}")

        expected_serialized_records = serialize_expected_csv_rows(
            columns = expected_columns,
            records = expected_records
        )

        if observed_records != expected_serialized_records:
            raise RuntimeError(f"Bayesian table contents differ from the validated workflow : {table_name}")

        if table_record["row_count"] != len(expected_serialized_records):
            raise RuntimeError(f"Bayesian table row count differs from the validated workflow : {table_name}")

        parsed_tables[table_name] = observed_records

    candidate_rows = parsed_tables["candidate_registry"]
    accounting_rows = parsed_tables["observation_accounting"]
    fold_rows = parsed_tables["grouped_validation_folds"]

    candidate_identities = [row["candidate_identity"] for row in candidate_rows]

    if len(candidate_identities) != len(set(candidate_identities)):
        raise RuntimeError("Candidate registry contains duplicate candidate identities.")

    accounting_candidate_identities = [row["candidate_identity"] for row in accounting_rows]

    if set(accounting_candidate_identities) != set(candidate_identities):
        raise RuntimeError("Observation accounting does not reconcile to candidate identities.")

    fold_candidate_identities = {row["candidate_identity"] for row in fold_rows}

    if not fold_candidate_identities.issubset(set(candidate_identities)):
        raise RuntimeError("Grouped-validation folds reference unknown candidates.")

    if expected_operation == "prior_predictive":
        if expected_review_template is None:
            raise RuntimeError("Prior-predictive validation requires the expected review template.")

        review_manifest = report.get(
            "review_template"
        )

        _require_exact_fields(
            review_manifest,
            {
                "file_name",
                "sha256",
                "candidate_inventory_sha256",
                "prior_predictive_diagnostics_sha256",
                "review_record_count"
            },
            "Prior-predictive review-template manifest"
        )

        expected_review_file_name = (
            expected_policy[
                "output_contract"
            ][
                "prior_predictive_review_template_file"
            ]
        )

        if (
            review_manifest[
                "file_name"
            ]
            != expected_review_file_name
        ):
            raise RuntimeError("Prior-predictive review-template file name differs from its contract.")

        review_path = (
            report_path.parent
            / expected_review_file_name
        )

        if not review_path.is_file():
            raise RuntimeError("Prior-predictive review template is missing.")

        if (
            calculate_file_sha256(
                review_path
            )
            != review_manifest[
                "sha256"
            ]
        ):
            raise RuntimeError("Prior-predictive review-template SHA-256 mismatch.")

        observed_review_template = (
            load_json_object(
                review_path
            )
        )

        if (
            observed_review_template
            != expected_review_template
        ):
            raise RuntimeError("Prior-predictive review-template contents differ from the validated execution.")

        if (
            review_manifest[
                "review_record_count"
            ]
            != len(
                expected_review_template[
                    "review_records"
                ]
            )
        ):
            raise RuntimeError("Prior-predictive review-record count mismatch.")

    if report.get("candidate_count") != len(candidate_rows):
        raise RuntimeError("Candidate count does not reconcile to the candidate table.")

    admissible_from_table = sum(1 for row in candidate_rows if row["admissibility_status"] == "admissible")
    inadmissible_from_table = sum(1 for row in candidate_rows if row["admissibility_status"] == "inadmissible")

    if report.get("admissible_candidate_count") != admissible_from_table:
        raise RuntimeError("Admissible candidate count does not reconcile.")

    if report.get("inadmissible_candidate_count") != inadmissible_from_table:
        raise RuntimeError("Inadmissible candidate count does not reconcile.")

    expected_admissibility_reason_counts = Counter()

    for row in candidate_rows:
        if row["admissibility_status"] == "inadmissible":
            expected_admissibility_reason_counts[row["admissibility_reason_code"]] += 1

    if report.get("admissibility_reason_counts") != dict(
        sorted(expected_admissibility_reason_counts.items())
    ):
        raise RuntimeError("Bayesian report admissibility reason counts do not reconcile to the candidate table.")


# Publishing the complete validate-operation artifact set
def publish_validation_artifacts(workflow: dict, output_directory: Path) -> dict:
    output_directory.parent.mkdir(parents = True, exist_ok = True)
    staging_directory = Path(
        tempfile.mkdtemp(prefix = f".{output_directory.name}.", dir = output_directory.parent)
    )

    produced_table_names = [
        "candidate_registry", "observation_accounting", "grouped_validation_folds"
    ]

    try:
        candidate_registry_path = staging_directory / "candidate_registry.csv"
        observation_accounting_path = staging_directory / "observation_accounting.csv"
        grouped_validation_folds_path = staging_directory / "grouped_validation_folds.csv"
        report_path = staging_directory / "bayesian_modeling_report.json"

        write_csv_table(candidate_registry_path, CANDIDATE_REGISTRY_FIELDS, workflow["candidate_records"])
        write_csv_table(observation_accounting_path, OBSERVATION_ACCOUNTING_FIELDS, workflow["observation_accounting_records"])
        write_csv_table(grouped_validation_folds_path, GROUPED_VALIDATION_FOLDS_FIELDS, workflow["fold_records"])

        report = build_validation_report(workflow, produced_table_names)

        report["tables"] = {
            "candidate_registry": build_table_manifest(
                candidate_registry_path, CANDIDATE_REGISTRY_FIELDS,
                len(workflow["candidate_records"])
            ),
            "observation_accounting": build_table_manifest(
                observation_accounting_path, OBSERVATION_ACCOUNTING_FIELDS,
                len(workflow["observation_accounting_records"])
            ),
            "grouped_validation_folds": build_table_manifest(
                grouped_validation_folds_path, GROUPED_VALIDATION_FOLDS_FIELDS,
                len(workflow["fold_records"])
            )
        }

        report_path.write_text(
            json.dumps(report, indent = 2, sort_keys = True) + "\n",
            encoding = "utf-8"
        )

        validate_bayesian_report(
            report_path = report_path,
            expected_policy = workflow["validated_inputs"]["policy"],
            expected_policy_sha256 = workflow["validated_inputs"]["policy_sha256"],
            expected_input_hashes = report["input_hashes"],
            expected_candidate_records = workflow["candidate_records"],
            expected_accounting_records = workflow["observation_accounting_records"],
            expected_fold_records = workflow["fold_records"],
            expected_execution_plan = workflow["execution_plan"],
            expected_operation = "validate"
        )

        backup_directory = None

        if output_directory.exists():
            backup_directory = output_directory.parent / f".{output_directory.name}.backup"
            if backup_directory.exists():
                shutil.rmtree(backup_directory)
            output_directory.replace(backup_directory)

        try:
            staging_directory.replace(output_directory)
        except Exception:
            if backup_directory is not None and backup_directory.exists() and not output_directory.exists():
                backup_directory.replace(output_directory)
            raise

        try:
            validate_bayesian_report(
                report_path = output_directory / "bayesian_modeling_report.json",
                expected_policy = workflow["validated_inputs"]["policy"],
                expected_policy_sha256 = workflow["validated_inputs"]["policy_sha256"],
                expected_input_hashes = report["input_hashes"],
                expected_candidate_records = workflow["candidate_records"],
                expected_accounting_records = workflow["observation_accounting_records"],
                expected_fold_records = workflow["fold_records"],
                expected_execution_plan = workflow["execution_plan"],
                expected_operation = "validate"
            )
        except Exception:
            shutil.rmtree(output_directory, ignore_errors = True)
            if backup_directory is not None and backup_directory.exists():
                backup_directory.replace(output_directory)
            raise

        if backup_directory is not None and backup_directory.exists():
            shutil.rmtree(backup_directory)

        return report

    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory, ignore_errors = True)


# Building the prior-predictive report
def build_prior_predictive_report(workflow: dict, produced_table_names: list[str],
                                  review_template_sha256: str, candidate_inventory_sha256: str,
                                  diagnostics_sha256: str, review_record_count: int,
                                  execution_accounting: dict
) -> dict:
    validated_inputs = workflow["validated_inputs"]
    policy = validated_inputs["policy"]

    candidate_records = workflow["candidate_records"]
    admissible = [
        record
        for record in candidate_records
        if record["admissibility_status"] == "admissible"
    ]
    inadmissible = [
        record
        for record in candidate_records
        if record["admissibility_status"] == "inadmissible"
    ]

    admissibility_reason_counts = {}

    for record in inadmissible:
        reason = record["admissibility_reason_code"]
        admissibility_reason_counts[reason] = admissibility_reason_counts.get(reason, 0) + 1

    all_declared_tables = list(policy["output_contract"]["tables"])

    return {
        "assessment_status": "completed",
        "operation": "prior_predictive",
        "policy_version": policy["policy_version"],
        "policy_effective_date": policy["effective_date"],
        "policy_sha256": validated_inputs["policy_sha256"],
        "implementation_sha256": calculate_file_sha256(
            Path(__file__).resolve()
        ),
        "policy": policy,
        "input_hashes": {
            "feature_manifest_sha256": validated_inputs["feature_manifest_sha256"],
            "design_report_sha256": validated_inputs["design_report_sha256"],
            "identification_report_sha256": validated_inputs["identification_report_sha256"],
            "descriptive_report_sha256": validated_inputs["descriptive_report_sha256"]
        },
        "candidate_count": len(candidate_records),
        "admissible_candidate_count": len(admissible),
        "inadmissible_candidate_count": len(inadmissible),
        "admissibility_reason_counts": admissibility_reason_counts,
        "prior_predictive_review_required": True,
        "posterior_sampling_performed": False,
        "pymc_runtime_loaded": True,
        "prior_predictive_execution_accounting": execution_accounting,
        "execution_plan": workflow["execution_plan"],
        "produced_tables": list(produced_table_names),
        "not_produced_tables": [
            table_name
            for table_name in all_declared_tables
            if table_name not in set(produced_table_names)
        ],
        "review_template": {
            "file_name": "bayesian_prior_predictive_review_template.json",
            "sha256": review_template_sha256,
            "candidate_inventory_sha256": candidate_inventory_sha256,
            "prior_predictive_diagnostics_sha256": diagnostics_sha256,
            "review_record_count": review_record_count
        },
        "tables": {}
    }


# Publishing the complete prior-predictive artifact set
def publish_prior_predictive_artifacts(workflow: dict, output_directory: Path,
                                       cache_directory: Path, software_environment_record: dict,
                                       software_environment_sha256: str,
                                       shard_count: int = 1, shard_index: int = 0
) -> dict:
    if shard_count > 1:
        _, shard_execution_accounting = execute_prior_predictive_sampling(
            workflow = workflow,
            cache_directory = cache_directory,
            software_environment_record = software_environment_record,
            software_environment_sha256 = software_environment_sha256,
            shard_count = shard_count,
            shard_index = shard_index
        )
        return {
            "operation": "prior_predictive_shard",
            "shard_count": shard_count,
            "shard_index": shard_index,
            "prior_predictive_execution_accounting": shard_execution_accounting
        }

    output_directory.parent.mkdir(
        parents = True,
        exist_ok = True
    )

    staging_directory = Path(
        tempfile.mkdtemp(
            prefix = f".{output_directory.name}.",
            dir = output_directory.parent
        )
    )

    produced_table_names = [
        "candidate_registry",
        "observation_accounting",
        "grouped_validation_folds",
        "prior_predictive_diagnostics"
    ]

    try:
        candidate_registry_path = staging_directory / "candidate_registry.csv"
        observation_accounting_path = staging_directory / "observation_accounting.csv"
        grouped_validation_folds_path = staging_directory / "grouped_validation_folds.csv"
        prior_predictive_diagnostics_path = staging_directory / "prior_predictive_diagnostics.csv"
        review_template_path = staging_directory / "bayesian_prior_predictive_review_template.json"
        report_path = staging_directory / "bayesian_modeling_report.json"

        write_csv_table(
            candidate_registry_path,
            CANDIDATE_REGISTRY_FIELDS,
            workflow["candidate_records"]
        )
        write_csv_table(
            observation_accounting_path,
            OBSERVATION_ACCOUNTING_FIELDS,
            workflow["observation_accounting_records"]
        )
        write_csv_table(
            grouped_validation_folds_path,
            GROUPED_VALIDATION_FOLDS_FIELDS,
            workflow["fold_records"]
        )

        (
            diagnostic_records,
            execution_accounting
        ) = execute_prior_predictive_sampling(
            workflow = workflow,
            cache_directory = cache_directory,
            software_environment_record = software_environment_record,
            software_environment_sha256 = software_environment_sha256,
            shard_count = 1,
            shard_index = 0
        )

        write_csv_table(
            prior_predictive_diagnostics_path,
            PRIOR_PREDICTIVE_DIAGNOSTIC_FIELDS,
            diagnostic_records
        )

        candidate_inventory_sha256 = calculate_file_sha256(candidate_registry_path)
        diagnostics_sha256 = calculate_file_sha256(prior_predictive_diagnostics_path)

        review_template = build_prior_predictive_review_template(
            workflow = workflow,
            diagnostic_records = diagnostic_records,
            candidate_inventory_sha256 = candidate_inventory_sha256,
            diagnostics_sha256 = diagnostics_sha256
        )

        review_template_path.write_text(
            json.dumps(
                review_template,
                indent = 2,
                sort_keys = True,
                allow_nan = False
            )
            + "\n",
            encoding = "utf-8"
        )

        report = build_prior_predictive_report(
            workflow = workflow,
            produced_table_names = produced_table_names,
            review_template_sha256 = calculate_file_sha256(review_template_path),
            candidate_inventory_sha256 = candidate_inventory_sha256,
            diagnostics_sha256 = diagnostics_sha256,
            review_record_count = len(review_template["review_records"]),
            execution_accounting = execution_accounting
        )

        report["tables"] = {
            "candidate_registry": build_table_manifest(
                candidate_registry_path,
                CANDIDATE_REGISTRY_FIELDS,
                len(workflow["candidate_records"])
            ),
            "observation_accounting": build_table_manifest(
                observation_accounting_path,
                OBSERVATION_ACCOUNTING_FIELDS,
                len(workflow["observation_accounting_records"])
            ),
            "grouped_validation_folds": build_table_manifest(
                grouped_validation_folds_path,
                GROUPED_VALIDATION_FOLDS_FIELDS,
                len(workflow["fold_records"])
            ),
            "prior_predictive_diagnostics": build_table_manifest(
                prior_predictive_diagnostics_path,
                PRIOR_PREDICTIVE_DIAGNOSTIC_FIELDS,
                len(diagnostic_records)
            )
        }

        report_path.write_text(
            json.dumps(
                report,
                indent = 2,
                sort_keys = True,
                allow_nan = False
            )
            + "\n",
            encoding = "utf-8"
        )

        validation_arguments = {
            "expected_policy": (
                workflow[
                    "validated_inputs"
                ][
                    "policy"
                ]
            ),
            "expected_policy_sha256": (
                workflow[
                    "validated_inputs"
                ][
                    "policy_sha256"
                ]
            ),
            "expected_input_hashes": (
                report[
                    "input_hashes"
                ]
            ),
            "expected_candidate_records": (
                workflow[
                    "candidate_records"
                ]
            ),
            "expected_accounting_records": (
                workflow[
                    "observation_accounting_records"
                ]
            ),
            "expected_fold_records": (
                workflow[
                    "fold_records"
                ]
            ),
            "expected_execution_plan": (
                workflow[
                    "execution_plan"
                ]
            ),
            "expected_operation": (
                "prior_predictive"
            ),
            "expected_additional_tables": {
                "prior_predictive_diagnostics": (
                    PRIOR_PREDICTIVE_DIAGNOSTIC_FIELDS,
                    diagnostic_records
                )
            },
            "expected_review_template": (
                review_template
            ),
            "expected_execution_accounting": (
                execution_accounting
            )
        }

        validate_bayesian_report(
            report_path = report_path,
            **validation_arguments
        )

        backup_directory = None

        if output_directory.exists():
            backup_directory = (
                output_directory.parent
                / f".{output_directory.name}.backup"
            )

            if backup_directory.exists():
                shutil.rmtree(
                    backup_directory
                )

            output_directory.replace(
                backup_directory
            )

        try:
            staging_directory.replace(
                output_directory
            )

            validate_bayesian_report(
                report_path = (
                    output_directory
                    / "bayesian_modeling_report.json"
                ),
                **validation_arguments
            )

        except Exception:
            if output_directory.exists():
                shutil.rmtree(
                    output_directory,
                    ignore_errors = True
                )

            if (
                backup_directory is not None
                and backup_directory.exists()
            ):
                backup_directory.replace(
                    output_directory
                )

            raise

        if (
            backup_directory is not None
            and backup_directory.exists()
        ):
            shutil.rmtree(
                backup_directory
            )

        return report

    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory, ignore_errors = True)


# Linking one immutable cache artifact into the published posterior inventory
def link_or_copy_file(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents = True, exist_ok = True)

    try:
        os.link(source_path, destination_path)
    except OSError:
        shutil.copy2(source_path, destination_path)


# Building the complete fit-operation report
def build_fit_report(workflow: dict, fit_outputs: dict,
                     table_names: list[str], posterior_manifest: list[dict],
                     software_environment_record: dict, software_environment_sha256: str
) -> dict:
    policy = workflow["validated_inputs"]["policy"]
    qualification_counts = defaultdict(int)

    for record in fit_outputs["qualification"].values():
        qualification_counts[record["status"]] += 1

    admissibility_reason_counts = {}

    for record in workflow["candidate_records"]:
        if record["admissibility_status"] == "inadmissible":
            reason = record["admissibility_reason_code"]
            admissibility_reason_counts[reason] = admissibility_reason_counts.get(reason, 0) + 1

    return {
        "assessment_status": "completed",
        "operation": "fit",
        "policy_version": policy["policy_version"],
        "policy_effective_date": policy["effective_date"],
        "policy_sha256": workflow["validated_inputs"]["policy_sha256"],
        "implementation_sha256": calculate_file_sha256(
            Path(__file__).resolve()
        ),
        "policy": policy,
        "input_hashes": build_stochastic_input_hashes(workflow),
        "candidate_count": len(workflow["candidate_records"]),
        "admissible_candidate_count": sum(
            record["admissibility_status"] == "admissible"
            for record in workflow["candidate_records"]
        ),
        "inadmissible_candidate_count": sum(
            record["admissibility_status"] == "inadmissible"
            for record in workflow["candidate_records"]
        ),
        "admissibility_reason_counts": admissibility_reason_counts,
        "execution_plan": workflow["execution_plan"],
        "pymc_runtime_loaded": True,
        "posterior_sampling_performed": (
            fit_outputs["full_posterior_execution_accounting"]["new_cache_count"]
            + fit_outputs["grouped_validation_execution_accounting"]["new_cache_count"]
            > 0
        ),
        "posterior_sampling_available_in_artifact_set": True,
        "prior_predictive_review_required": True,
        "prior_predictive_review_sha256": calculate_file_sha256(
            PROJECT_ROOT / policy["prior_predictive_review"]["review_artifact_path"]
        ),
        "software_environment": software_environment_record,
        "software_environment_sha256": software_environment_sha256,
        "full_posterior_execution_accounting": fit_outputs["full_posterior_execution_accounting"],
        "grouped_validation_execution_accounting": fit_outputs["grouped_validation_execution_accounting"],
        "analytical_qualification_counts": dict(sorted(qualification_counts.items())),
        "qualification_gate_failure_counts": dict(
            sorted(
                Counter(
                    gate_name
                    for qualification_record in fit_outputs["qualification"].values()
                    for gate_name in qualification_record.get("failed_gates", [])
                ).items()
            )
        ),
        "posterior_artifacts": posterior_manifest,
        "produced_tables": table_names,
        "not_produced_tables": [],
        "tables": {}
    }


# Publishing one staged directory with rollback validation
def replace_directory_transactionally(staging_directory: Path, output_directory: Path, validate_published) -> None:
    backup_directory = None

    if output_directory.exists():
        backup_directory = (
            output_directory.parent
            / f".{output_directory.name}.backup"
        )

        if backup_directory.exists():
            shutil.rmtree(
                backup_directory
            )

        output_directory.replace(
            backup_directory
        )

    try:
        staging_directory.replace(
            output_directory
        )

        validate_published()

    except Exception:
        if output_directory.exists():
            shutil.rmtree(
                output_directory,
                ignore_errors = True
            )

        if (
            backup_directory is not None
            and backup_directory.exists()
        ):
            backup_directory.replace(
                output_directory
            )

        raise

    if (
        backup_directory is not None
        and backup_directory.exists()
    ):
        shutil.rmtree(
            backup_directory
        )


# Publishing every required fit artifact transactionally
def publish_fit_artifacts(workflow: dict, output_directory: Path, cache_directory: Path,
                          software_environment_record: dict,
                          software_environment_sha256: str
) -> dict:
    fit_outputs = build_fit_outputs(
        workflow = workflow,
        output_directory = output_directory,
        cache_directory = cache_directory,
        software_environment_record = software_environment_record,
        software_environment_sha256 = software_environment_sha256
    )
    output_directory.parent.mkdir(parents = True, exist_ok = True)
    staging_directory = Path(
        tempfile.mkdtemp(prefix = f".{output_directory.name}.fit.", dir = output_directory.parent)
    )
    table_records = {
        "candidate_registry": fit_outputs["candidate_records"],
        "observation_accounting": workflow["observation_accounting_records"],
        "grouped_validation_folds": workflow["fold_records"],
        "prior_predictive_diagnostics": fit_outputs["prior_diagnostics"],
        "posterior_summaries": fit_outputs["posterior_summaries"],
        "sampler_diagnostics": fit_outputs["sampler_diagnostics"],
        "posterior_predictive_diagnostics": fit_outputs["posterior_predictive_diagnostics"],
        "grouped_validation_predictions": fit_outputs["grouped_validation_predictions"],
        "grouped_validation_metrics": fit_outputs["grouped_validation_metrics"],
        "grouped_validation_condition_calibration": fit_outputs["grouped_validation_condition_calibration"],
        "continuous_trajectory_summaries": fit_outputs["continuous_trajectory_summaries"],
        "continuous_trajectory_difference_summaries": fit_outputs["continuous_trajectory_difference_summaries"],
        "prior_sensitivity_summaries": fit_outputs["prior_sensitivity_summaries"]
    }

    try:
        for table_name in workflow["validated_inputs"]["policy"]["output_contract"]["tables"]:
            write_csv_table(
                staging_directory / f"{table_name}.csv",
                OUTPUT_TABLE_FIELDS[table_name],
                table_records[table_name]
            )

        posterior_directory = staging_directory / workflow["validated_inputs"]["policy"]["output_contract"]["posterior_directory"]
        posterior_manifest = []

        for cache_record in fit_outputs["posterior_cache_records"]:
            source_directory = build_cache_entry_directory(
                cache_directory,
                cache_record["cache_identity"]
            )
            source_path = source_directory / workflow["validated_inputs"]["policy"]["cache_contract"]["cache_inference_data_file"]
            multiplier_key = prior_multiplier_key(cache_record["prior_scale_multiplier"]).replace(".", "p")
            file_name = f"{cache_record['candidate_identity']}__prior_{multiplier_key}.nc"
            destination_path = posterior_directory / file_name
            link_or_copy_file(source_path, destination_path)
            posterior_manifest.append({
                "candidate_identity": cache_record["candidate_identity"],
                "prior_scale_multiplier": cache_record["prior_scale_multiplier"],
                "file_name": f"{posterior_directory.name}/{file_name}",
                "sha256": calculate_file_sha256(destination_path),
                "size_bytes": destination_path.stat().st_size
            })

        posterior_manifest.sort(key = lambda record: (
            record["candidate_identity"],
            record["prior_scale_multiplier"]
        ))

        table_names = list(table_records)

        additional_tables = {
            table_name: (
                OUTPUT_TABLE_FIELDS[table_name],
                table_records[table_name]
            )
            for table_name in table_names
            if table_name not in {
                "candidate_registry",
                "observation_accounting",
                "grouped_validation_folds"
            }
        }

        report = build_fit_report(
            workflow,
            fit_outputs,
            table_names,
            posterior_manifest,
            software_environment_record,
            software_environment_sha256
        )

        report["tables"] = {
            table_name: build_table_manifest(
                staging_directory / f"{table_name}.csv",
                OUTPUT_TABLE_FIELDS[table_name],
                len(table_records[table_name])
            )
            for table_name in table_names
        }
        report_path = staging_directory / "bayesian_modeling_report.json"
        report_path.write_text(
            json.dumps(report, indent = 2, sort_keys = True) + "\n",
            encoding = "utf-8"
        )

        for posterior_record in posterior_manifest:
            artifact_path = staging_directory / posterior_record["file_name"]
            if calculate_file_sha256(artifact_path) != posterior_record["sha256"]:
                raise RuntimeError("Published posterior artifact hash mismatch.")

        def validate_published_fit() -> None:
            expected_qualification_counts = defaultdict(int)
            expected_gate_failure_counts = Counter()

            for qualification_record in fit_outputs["qualification"].values():
                expected_qualification_counts[
                    qualification_record["status"]
                ] += 1

                for gate_name in qualification_record.get("failed_gates", []):
                    expected_gate_failure_counts[gate_name] += 1

            validate_bayesian_report(
                report_path = (
                    output_directory
                    / "bayesian_modeling_report.json"
                ),
                expected_policy = workflow[
                    "validated_inputs"
                ][
                    "policy"
                ],
                expected_policy_sha256 = workflow[
                    "validated_inputs"
                ][
                    "policy_sha256"
                ],
                expected_input_hashes = report[
                    "input_hashes"
                ],
                expected_candidate_records = fit_outputs[
                    "candidate_records"
                ],
                expected_accounting_records = workflow[
                    "observation_accounting_records"
                ],
                expected_fold_records = workflow[
                    "fold_records"
                ],
                expected_execution_plan = workflow[
                    "execution_plan"
                ],
                expected_operation = "fit",
                expected_additional_tables = additional_tables,
                expected_fit_report_fields = {
                    "pymc_runtime_loaded": True,
                    "posterior_sampling_performed": (
                        fit_outputs["full_posterior_execution_accounting"]["new_cache_count"]
                        + fit_outputs["grouped_validation_execution_accounting"]["new_cache_count"]
                        > 0
                    ),
                    "posterior_sampling_available_in_artifact_set": True,
                    "prior_predictive_review_required": True,
                    "prior_predictive_review_sha256": calculate_file_sha256(
                        PROJECT_ROOT
                        / workflow["validated_inputs"]["policy"]["prior_predictive_review"]["review_artifact_path"]
                    ),
                    "software_environment": software_environment_record,
                    "software_environment_sha256": software_environment_sha256,
                    "full_posterior_execution_accounting": fit_outputs["full_posterior_execution_accounting"],
                    "grouped_validation_execution_accounting": fit_outputs["grouped_validation_execution_accounting"],
                    "analytical_qualification_counts": dict(
                        sorted(expected_qualification_counts.items())
                    ),
                    "qualification_gate_failure_counts": dict(
                        sorted(expected_gate_failure_counts.items())
                    ),
                    "posterior_artifacts": posterior_manifest
                }
            )

            if (
                len(posterior_manifest) != int(workflow["execution_plan"]["full_data_posterior_call_count"])
            ):
                raise RuntimeError("Bayesian posterior manifest does not contain every registered full-data posterior.")

        replace_directory_transactionally(
            staging_directory = staging_directory,
            output_directory = output_directory,
            validate_published = validate_published_fit
        )

        return report

    finally:
        if staging_directory.exists():
            shutil.rmtree(staging_directory, ignore_errors = True)


# Populating only this shard's cache entries without publishing artifacts
def run_populate_fit_cache_operation(workflow: dict, arguments: argparse.Namespace) -> dict:
    policy = workflow["validated_inputs"]["policy"]

    validate_cache_shard(
        shard_count = arguments.shard_count,
        shard_index = arguments.shard_index
    )

    validate_stochastic_process_environment(
        policy
    )

    environment_record = (
        build_software_environment_record(
            policy
        )
    )

    require_compiler_environment(
        environment_record
    )

    environment_sha256 = deterministic_identity(
        environment_record
    )

    output_directory = (
        arguments.output_directory
    )

    prior_predictive_review_path = (
        output_directory
        / "bayesian_prior_predictive_review_template.json"
    )

    diagnostics_path = (
        output_directory
        / "prior_predictive_diagnostics.csv"
    )

    if not (
        prior_predictive_review_path.is_file()
        and diagnostics_path.is_file()
    ):
        raise RuntimeError("populate_fit_cache requires completed prior-predictive artifacts.")

    diagnostic_records = list(
        read_csv_rows(
            diagnostics_path
        )[1]
    )

    candidate_registry_path = (
        output_directory
        / "candidate_registry.csv"
    )

    acknowledged_pairs = (
        validate_prior_predictive_review(
            review_path = Path(
                policy[
                    "prior_predictive_review"
                ][
                    "review_artifact_path"
                ]
            ),
            workflow = workflow,
            candidate_inventory_sha256 = (
                calculate_file_sha256(
                    candidate_registry_path
                )
            ),
            diagnostics_sha256 = (
                calculate_file_sha256(
                    diagnostics_path
                )
            ),
            diagnostic_records = diagnostic_records
        )
    )

    (
        posterior_records,
        sampler_diagnostic_records,
        posterior_cache_records,
        full_data_execution_accounting
    ) = execute_full_posterior_sampling(
        workflow = workflow,
        acknowledged_pairs = acknowledged_pairs,
        cache_directory = arguments.cache_directory,
        software_environment_record = environment_record,
        software_environment_sha256 = environment_sha256,
        shard_count = arguments.shard_count,
        shard_index = arguments.shard_index
    )

    (
        grouped_prediction_records,
        grouped_metric_records,
        grouped_calibration_records,
        grouped_sampler_records,
        grouped_validation_execution_accounting
    ) = execute_grouped_validation(
        workflow = workflow,
        cache_directory = arguments.cache_directory,
        software_environment_record = environment_record,
        software_environment_sha256 = environment_sha256,
        shard_count = arguments.shard_count,
        shard_index = arguments.shard_index
    )

    return {
        "operation": "populate_fit_cache",
        "shard_count": arguments.shard_count,
        "shard_index": arguments.shard_index,
        "posterior_cache_count": len(
            posterior_cache_records
        ),
        "full_data_execution_accounting": (
            full_data_execution_accounting
        ),
        "grouped_validation_execution_accounting": (
            grouped_validation_execution_accounting
        )
    }


# Dispatching the requested operation without loading PyMC during validate
def execute_operation(arguments: argparse.Namespace, validated_inputs: dict) -> dict | None:
    workflow = build_validation_workflow(validated_inputs)

    if arguments.operation == "validate":
        return publish_validation_artifacts(
            workflow = workflow,
            output_directory = arguments.output_directory
        )

    validate_registered_source_cache_inventory(
        cache_directory = arguments.cache_directory,
        policy = validated_inputs["policy"]
    )

    if arguments.operation == "prior_predictive":
        validate_stochastic_process_environment(
            validated_inputs["policy"]
        )

        environment_record = build_software_environment_record(
            validated_inputs["policy"]
        )

        require_compiler_environment(
            environment_record
        )

        environment_sha256 = deterministic_identity(
            environment_record
        )

        return publish_prior_predictive_artifacts(
            workflow = workflow,
            output_directory = arguments.output_directory,
            cache_directory = arguments.cache_directory,
            software_environment_record = environment_record,
            software_environment_sha256 = environment_sha256,
            shard_count = arguments.shard_count,
            shard_index = arguments.shard_index
        )

    if arguments.operation == "populate_fit_cache":
        return run_populate_fit_cache_operation(
            workflow = workflow,
            arguments = arguments
        )

    if arguments.operation == "fit":
        validate_stochastic_process_environment(
            validated_inputs["policy"]
        )
        environment_record = build_software_environment_record(
            validated_inputs["policy"]
        )
        require_compiler_environment(environment_record)
        environment_sha256 = deterministic_identity(environment_record)
        return publish_fit_artifacts(
            workflow = workflow,
            output_directory = arguments.output_directory,
            cache_directory = arguments.cache_directory,
            software_environment_record = environment_record,
            software_environment_sha256 = environment_sha256
        )

    raise ValueError(f"Unsupported operation : {arguments.operation}")


# Running the requested operation
def main() -> None:
    arguments = parse_args()

    validated_inputs = load_validated_inputs(
        feature_manifest_path = arguments.feature_manifest_path,
        design_report_path = arguments.design_report_path,
        identification_report_path = arguments.identification_report_path,
        descriptive_report_path = arguments.descriptive_report_path,
        modeling_policy_path = arguments.modeling_policy_path
    )

    report = execute_operation(arguments, validated_inputs)

    if arguments.operation == "populate_fit_cache":
        print("Bayesian fit cache population finished.")
        print(f"Operation : {arguments.operation}")
        print(f"Shard count : {report['shard_count']}")
        print(f"Shard index : {report['shard_index']}")
        print(
            f"Posterior cache count : {report['posterior_cache_count']}"
        )
        print(
            f"Full posterior execution accounting : {report['full_data_execution_accounting']}"
        )
        print(
            f"Grouped validation execution accounting : {report['grouped_validation_execution_accounting']}"
        )
        return

    if arguments.operation == "prior_predictive":
        if report.get("operation") == "prior_predictive_shard":
            print(f"Prior-predictive shard {report['shard_index']}/{report['shard_count']} finished.")
            print(f"Execution accounting : {report['prior_predictive_execution_accounting']}")
            return

        print("Bayesian prior-predictive sampling finished.")
        print(f"Operation : {arguments.operation}")
        print(f"Candidate count : {report['candidate_count']}")
        print(f"Admissible candidate count : {report['admissible_candidate_count']}")
        print(f"Inadmissible candidate count : {report['inadmissible_candidate_count']}")
        print(f"Review records produced : {report['review_template']['review_record_count']}")
        print(f"Execution accounting : {report['prior_predictive_execution_accounting']}")
        print(f"Report : {arguments.output_directory / 'bayesian_modeling_report.json'}")
        return


    if arguments.operation == "fit":
        print("Bayesian posterior modeling finished.")
        print(f"Operation : {arguments.operation}")
        print(f"Candidate count : {report['candidate_count']}")
        print(f"Analytical qualification counts : {report['analytical_qualification_counts']}")
        print(f"Full posterior execution accounting : {report['full_posterior_execution_accounting']}")
        print(f"Grouped validation execution accounting : {report['grouped_validation_execution_accounting']}")
        print(f"Report : {arguments.output_directory / 'bayesian_modeling_report.json'}")
        return

    print("Bayesian-modeling validation finished.")
    print(f"Operation : {arguments.operation}")
    print(f"Candidate count : {report['candidate_count']}")
    print(f"Admissible candidate count : {report['admissible_candidate_count']}")
    print(f"Inadmissible candidate count : {report['inadmissible_candidate_count']}")
    print(f"Prior-predictive call count : {report['execution_plan']['prior_predictive_call_count']}")
    print(f"Full-data posterior call count : {report['execution_plan']['full_data_posterior_call_count']}")
    print(f"Grouped-validation posterior call count : {report['execution_plan']['grouped_validation_posterior_call_count']}")
    print(f"Total stochastic call upper bound : {report['execution_plan']['all_stochastic_call_upper_bound']}")
    print(f"Report : {arguments.output_directory / 'bayesian_modeling_report.json'}")


if __name__ == "__main__":
    main()

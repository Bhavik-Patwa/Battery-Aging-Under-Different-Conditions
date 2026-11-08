import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile

from collections import Counter, defaultdict
from datetime import date as iso_date
from pathlib import Path

import numpy as np

from scripts.run_predictive_models import (
    EstimatorUnavailableError,
    build_cell_weights,
    build_estimator,
    build_feature_matrix,
    build_inner_assignments,
    build_selected_candidates,
    calculate_condition_benchmark,
    calculate_file_sha256,
    calculate_training_horizon_scaling,
    deterministic_identity,
    equal_cell_metrics,
    fit_estimator,
    fold_has_horizon_coverage,
    load_json_object,
    predict_original_scale,
    read_csv_table,
    validate_software_environment,
    validate_upstream_inputs,
    weighted_quantile,
    write_csv_table
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SENSITIVITY_REGISTRY_FIELDS = [
    "variant_identity",
    "candidate_identity",
    "model_family",
    "variant_type",
    "partition_identity",
    "configuration_inventory",
    "training_window_identity",
    "incident_reason_code",
    "ablation_group",
    "registration_order"
]

SENSITIVITY_VALIDATION_FOLD_FIELDS = [
    "variant_identity",
    "candidate_identity",
    "model_family",
    "fold_index",
    "training_cell_count",
    "test_cell_count",
    "training_cell_identities",
    "test_cell_identities",
    "training_observation_count",
    "test_observation_count",
    "excluded_test_observation_count",
    "training_condition_count",
    "fold_status",
    "fold_reason_code"
]

SENSITIVITY_VALIDATION_PREDICTION_FIELDS = [
    "variant_identity",
    "variant_type",
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

SENSITIVITY_VALIDATION_METRIC_FIELDS = [
    "variant_identity",
    "variant_type",
    "candidate_identity",
    "model_family",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "outer_fold_count",
    "test_cell_count",
    "test_observation_count",
    "equal_cell_weighted_root_mean_squared_error",
    "equal_cell_weighted_mean_absolute_error",
    "equal_cell_weighted_predictive_r_squared",
    "equal_cell_weighted_prediction_interval_coverage",
    "equal_cell_weighted_mean_prediction_interval_width",
    "validation_status",
    "validation_reason_code",
    "analytical_status",
    "analytical_reason_code",
    "estimate_label"
]

CONFIGURATION_SELECTION_FIELDS = [
    "variant_identity",
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

SENSITIVITY_COMPARISON_FIELDS = [
    "variant_identity",
    "variant_type",
    "candidate_identity",
    "model_family",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "shared_observation_count",
    "shared_independent_cell_count",
    "excluded_test_observation_count",
    "primary_equal_cell_weighted_root_mean_squared_error",
    "variant_equal_cell_weighted_root_mean_squared_error",
    "root_mean_squared_error_difference",
    "primary_equal_cell_weighted_mean_absolute_error",
    "variant_equal_cell_weighted_mean_absolute_error",
    "mean_absolute_error_difference",
    "primary_equal_cell_weighted_predictive_r_squared",
    "variant_equal_cell_weighted_predictive_r_squared",
    "predictive_r_squared_difference",
    "primary_inventory_count",
    "variant_inventory_count",
    "exact_inventory_match",
    "variant_analytical_status",
    "comparison_status",
    "comparison_reason_code",
    "estimate_label"
]

CONDITION_CALIBRATION_FIELDS = [
    "variant_identity",
    "variant_type",
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

PREDICTOR_ABLATION_FIELDS = [
    "variant_identity",
    "candidate_identity",
    "model_family",
    "ablation_group",
    "primary_root_mean_squared_error",
    "ablated_root_mean_squared_error",
    "root_mean_squared_error_difference",
    "primary_mean_absolute_error",
    "ablated_mean_absolute_error",
    "mean_absolute_error_difference",
    "primary_predictive_r_squared",
    "ablated_predictive_r_squared",
    "predictive_r_squared_difference",
    "variant_validation_status",
    "variant_analytical_status",
    "comparison_status",
    "comparison_reason_code",
    "estimate_label"
]

INCIDENT_EXCLUSION_FIELDS = [
    "variant_identity",
    "candidate_identity",
    "model_family",
    "incident_reason_code",
    "eligible_in_every_fold",
    "minimum_excluded_independent_cell_count",
    "minimum_retained_independent_cell_count",
    "primary_root_mean_squared_error",
    "excluded_root_mean_squared_error",
    "root_mean_squared_error_difference",
    "primary_mean_absolute_error",
    "excluded_mean_absolute_error",
    "mean_absolute_error_difference",
    "primary_predictive_r_squared",
    "excluded_predictive_r_squared",
    "predictive_r_squared_difference",
    "variant_validation_status",
    "variant_analytical_status",
    "comparison_status",
    "comparison_reason_code",
    "estimate_label"
]

PREDICTION_INTERVAL_DIAGNOSTIC_FIELDS = [
    "variant_identity",
    "variant_type",
    "candidate_identity",
    "model_family",
    "grouping_dimension",
    "grouping_value",
    "nominal_interval_probability",
    "prediction_count",
    "independent_cell_count",
    "covered_prediction_count",
    "equal_cell_weighted_interval_coverage",
    "equal_cell_weighted_mean_interval_width",
    "assessment_status",
    "assessment_reason_code",
    "estimate_label"
]

REQUIRED_POLICY_SECTIONS = {
    "policy_name",
    "policy_version",
    "effective_date",
    "registration_status",
    "scientific_scope",
    "inputs",
    "partition_sensitivity",
    "expanded_configuration_inventory",
    "training_window_sensitivity",
    "incident_sensitivity",
    "predictor_ablation",
    "metric_contract",
    "execution",
    "output_contract"
}


# Parsing validated input paths, policy location and output directory
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = (
            "Validate and evaluate one-factor-at-a-time predictive sensitivity variants against the registered"
            " primary benchmark."
        )
    )

    parser.add_argument("--feature-manifest-path", type = Path, required = True)
    parser.add_argument("--design-report-path", type = Path, required = True)
    parser.add_argument("--identification-report-path", type = Path, required = True)
    parser.add_argument("--descriptive-report-path", type = Path, required = True)
    parser.add_argument("--bayesian-policy-path", type = Path, required = True)
    parser.add_argument("--bayesian-report-path", type = Path, required = True)
    parser.add_argument("--predictive-policy-path", type = Path, required = True)
    parser.add_argument("--predictive-report-path", type = Path, required = True)
    parser.add_argument("--sensitivity-policy-path", type = Path, required = True)
    parser.add_argument("--data-quality-contract-path", type = Path, required = True)
    parser.add_argument("--output-directory", type = Path, required = True)
    parser.add_argument("--operation", choices = ("validate", "fit"), required = True)

    return parser.parse_args()


# Producing deterministic canonical JSON
def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii = False,
        separators = (",", ":"),
        sort_keys = True
    )


# Serializing one CSV value deterministically
def serialize_csv_value(value: object):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return canonical_json(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError("Cannot serialize a non-finite value.")
        return repr(value)
    return value


# Parsing one CSV value into its typed representation
def parse_csv_value(value: str):
    if value == "":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith("[") or value.startswith("{"):
        return json.loads(value)
    return value


# Reading one CSV table with typed values
def read_typed_csv_table(path: Path) -> list[dict]:
    with path.open(mode = "r", encoding = "utf-8", newline = "") as stream:
        return [
            {field_name: parse_csv_value(field_value) for field_name, field_value in row.items()}
            for row in csv.DictReader(stream)
        ]


# Verifying one report table against its declared manifest
def verify_report_tables(report_path: Path, report: dict, required_tables: list[str]) -> dict[str, list[dict]]:
    verified_tables = {}

    for table_name in required_tables:
        table_manifest = report.get("tables", {}).get(table_name)

        if table_manifest is None:
            raise RuntimeError(f"Required report table is unavailable : {table_name}")

        table_path = report_path.parent / table_manifest["file_name"]

        if calculate_file_sha256(table_path) != table_manifest["sha256"]:
            raise RuntimeError(f"Report table hash does not match : {table_name}")

        rows = read_typed_csv_table(table_path)

        if len(rows) != int(table_manifest["row_count"]):
            raise RuntimeError(f"Report table row count does not match : {table_name}")

        verified_tables[table_name] = rows

    return verified_tables


# Validating the registered predictive-sensitivity policy
def validate_policy(policy: dict) -> None:
    if set(policy) != REQUIRED_POLICY_SECTIONS:
        raise RuntimeError("Predictive-sensitivity policy has an unexpected top-level schema.")

    if policy["policy_name"] != "predictive_sensitivity_policy":
        raise RuntimeError("Predictive-sensitivity policy name is invalid.")

    if re.fullmatch(r"[1-9]\d*\.\d+", str(policy.get("policy_version", ""))) is None:
        raise RuntimeError("Predictive-sensitivity policy version is invalid.")

    try:
        iso_date.fromisoformat(str(policy["effective_date"]))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Predictive-sensitivity effective_date is invalid.") from exc

    if policy["registration_status"] != "result_informed_sensitivity_analysis":
        raise RuntimeError("Predictive-sensitivity registration_status is invalid.")

    scientific_scope = policy["scientific_scope"]

    for prohibited_flag in (
        "causal_interpretation_authorized",
        "policy_recommendation_authorized",
        "treatment_optimization_authorized",
        "performance_selected_model_winner_authorized",
        "primary_benchmark_replacement_authorized",
        "full_factorial_perturbation_crossing_authorized"
    ):
        if scientific_scope.get(prohibited_flag) is not False:
            raise RuntimeError(f"Predictive-sensitivity scope incorrectly enables : {prohibited_flag}")

    for separation_flag in (
        "dataset_stages_separated",
        "calendar_and_cycle_aging_separated",
        "elapsed_time_and_cumulative_fec_separated",
        "primary_and_sensitivity_cohorts_separated"
    ):
        if scientific_scope.get(separation_flag) is not True:
            raise RuntimeError(f"Predictive-sensitivity scope must separate : {separation_flag}")

    inputs = policy["inputs"]

    for digest_field in (
        "required_predictive_policy_sha256",
        "required_predictive_report_sha256",
        "required_feature_manifest_sha256",
        "required_data_quality_contract_sha256",
        "required_bayesian_policy_sha256",
        "required_bayesian_report_sha256",
        "required_design_report_sha256",
        "required_identification_report_sha256",
        "required_descriptive_report_sha256"
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(inputs.get(digest_field, ""))) is None:
            raise RuntimeError(f"Predictive-sensitivity input digest is invalid : {digest_field}")

    if inputs["required_predictive_tables"] != [
        "candidate_registry",
        "validation_folds",
        "hyperparameter_selection",
        "validation_metrics",
        "validation_predictions",
        "condition_calibration"
    ]:
        raise RuntimeError("Predictive-sensitivity predictive-tables inventory is invalid.")

    if inputs["required_feature_tables"] != ["reference_test_features"]:
        raise RuntimeError("Predictive-sensitivity feature-tables inventory is invalid.")

    partition_contract = policy["partition_sensitivity"]

    if partition_contract["fold_count"] < 2:
        raise RuntimeError("Predictive-sensitivity fold_count must be at least two.")

    if partition_contract.get("minimum_fold_count", 0) < 2:
        raise RuntimeError("Predictive-sensitivity minimum_fold_count must be at least two.")

    if partition_contract["minimum_fold_count"] > partition_contract["fold_count"]:
        raise RuntimeError("Predictive-sensitivity minimum_fold_count cannot exceed fold_count.")

    if partition_contract.get("duplicate_assignment_action") != "report_uninformative":
        raise RuntimeError("Predictive-sensitivity duplicate_assignment_action is invalid.")

    if len({method["hash_namespace"] for method in partition_contract["assignment_methods"]}) != len(
        partition_contract["assignment_methods"]
    ):
        raise RuntimeError("Predictive-sensitivity assignment_methods contain duplicate hash namespaces.")

    for assignment_method in partition_contract["assignment_methods"]:
        if set(assignment_method.keys()) != {"assignment_identity", "hash_namespace"}:
            raise RuntimeError("Partition assignment_method uses an invalid schema.")

        if (
            not isinstance(assignment_method["assignment_identity"], str)
            or not assignment_method["assignment_identity"].strip()
        ):
            raise RuntimeError("Partition assignment_identity is empty.")

        if (
            not isinstance(assignment_method["hash_namespace"], str)
            or not assignment_method["hash_namespace"].strip()
        ):
            raise RuntimeError("Partition hash_namespace is empty.")

    expanded_inventory = policy["expanded_configuration_inventory"]

    if set(expanded_inventory) != {"elastic_net", "random_forest", "gradient_boosting"}:
        raise RuntimeError("Predictive-sensitivity expanded inventory is invalid.")

    for family_name, family_policy in expanded_inventory.items():
        if len(family_policy["configurations"]) != 9:
            raise RuntimeError(f"Predictive-sensitivity expanded grid must contain 9 configurations : {family_name}")

    training_window = policy["training_window_sensitivity"]

    if training_window["retuning_authorized"] is not False:
        raise RuntimeError("Training-window sensitivity must not authorize retuning.")

    expected_training_window_variants = {
        "first_observation_excluded": "exclude_minimum_horizon_per_cell",
        "terminal_observation_excluded": "exclude_maximum_horizon_per_cell"
    }

    observed_training_window_variants = training_window.get("variants")

    if (
        not isinstance(observed_training_window_variants, list)
        or len(observed_training_window_variants) != len(expected_training_window_variants)
    ):
        raise RuntimeError("Training-window variant inventory has an invalid length.")

    for training_variant in observed_training_window_variants:
        if set(training_variant.keys()) != {"variant_identity", "operation"}:
            raise RuntimeError("Training-window variant uses an invalid schema.")

        variant_identity = training_variant["variant_identity"]
        operation = training_variant["operation"]

        expected_operation = expected_training_window_variants.get(
            variant_identity
        )

        if expected_operation is None:
            raise RuntimeError(f"Training-window variant_identity is unregistered : {variant_identity}")

        if operation != expected_operation:
            raise RuntimeError(f"Training-window operation does not match its registered variant : {variant_identity}")

    if training_window["fixed_primary_fold_configuration_required"] is not True:
        raise RuntimeError("Training-window sensitivity must fix the primary fold configuration.")

    if training_window.get("evaluation_population_rule") != (
        "same_condition_horizons_inside_restricted_training_support"
    ):
        raise RuntimeError("Training-window evaluation-population rule is invalid.")

    if training_window.get("comparison_inventory_rule") != "exact_shared_held_out_observations":
        raise RuntimeError("Training-window comparison-inventory rule is invalid.")

    if training_window.get("prediction_interval_calibration") != (
        "inner_grouped_out_of_fold_residuals"
    ):
        raise RuntimeError("Training-window interval-calibration rule is invalid.")

    incident_sensitivity = policy["incident_sensitivity"]

    if incident_sensitivity["retuning_authorized"] is not False:
        raise RuntimeError("Incident sensitivity must not authorize retuning.")

    if incident_sensitivity[
        "empty_or_inadequately_supported_exclusions_reported_unavailable"
    ] is not True:
        raise RuntimeError("Incident sensitivity must record inadequate exclusions as unavailable.")

    if incident_sensitivity.get("eligibility_required_in_every_outer_fold") is not True:
        raise RuntimeError("Incident sensitivity must require every-fold eligibility.")

    if incident_sensitivity.get("evaluation_population_rule") != "unchanged_primary_held_out_observations":
        raise RuntimeError("Incident-sensitivity evaluation-population rule is invalid.")

    if incident_sensitivity.get("comparison_inventory_rule") != "exact_shared_held_out_observations":
        raise RuntimeError("Incident-sensitivity comparison-inventory rule is invalid.")

    if incident_sensitivity.get("prediction_interval_calibration") != (
        "inner_grouped_out_of_fold_residuals"
    ):
        raise RuntimeError("Incident-sensitivity interval-calibration rule is invalid.")

    if incident_sensitivity["fixed_primary_fold_configuration_required"] is not True:
        raise RuntimeError("Incident sensitivity must fix the primary fold configuration.")

    if incident_sensitivity["reason_code_sources"] != [
        "archive_quality_reason_codes",
        "quality_reason_codes"
    ]:
        raise RuntimeError("Incident-sensitivity reason-code source inventory is invalid.")

    if incident_sensitivity["join_key"] != ["source_identity", "cell_identity"]:
        raise RuntimeError("Incident-sensitivity join key is invalid.")

    if incident_sensitivity["minimum_excluded_independent_cell_count"] < 1:
        raise RuntimeError("Incident-sensitivity minimum excluded cell count must be at least one.")

    if incident_sensitivity["minimum_retained_independent_cell_count"] < 1:
        raise RuntimeError("Incident-sensitivity minimum retained cell count must be at least one.")

    predictor_ablation = policy["predictor_ablation"]

    if predictor_ablation["groups"] != [
        "horizon_main_and_interactions",
        "condition_main_and_interactions",
        "horizon_condition_interactions"
    ]:
        raise RuntimeError("Predictor-ablation groups are invalid.")

    if predictor_ablation.get("unavailable_group_action") != "report_unavailable":
        raise RuntimeError("Predictor-ablation unavailable-group action is invalid.")

    if predictor_ablation.get("comparison_inventory_rule") != "exact_shared_held_out_observations":
        raise RuntimeError("Predictor-ablation comparison-inventory rule is invalid.")

    if predictor_ablation["retuning_authorized"] is not False:
        raise RuntimeError("Predictor ablation must not authorize retuning.")

    if predictor_ablation["fixed_primary_fold_configuration_required"] is not True:
        raise RuntimeError("Predictor ablation must fix the primary fold configuration.")

    if predictor_ablation["held_out_evaluation_required"] is not True:
        raise RuntimeError("Predictor ablation must require held-out evaluation.")

    if predictor_ablation["causal_interpretation_authorized"] is not False:
        raise RuntimeError("Predictor ablation must not authorize causal interpretation.")

    if predictor_ablation["cross_family_magnitude_comparison_authorized"] is not False:
        raise RuntimeError("Predictor ablation must not authorize cross-family magnitude comparison.")

    metric_contract = policy["metric_contract"]

    if metric_contract["independent_unit"] != "cell_identity":
        raise RuntimeError("Predictive-sensitivity metric independent_unit must be cell_identity.")

    if metric_contract["weighting"] != "equal_total_weight_per_cell":
        raise RuntimeError("Predictive-sensitivity metric weighting must be equal_total_weight_per_cell.")

    if metric_contract.get("additional_metrics") != [
        "equal_cell_weighted_prediction_interval_coverage",
        "equal_cell_weighted_mean_prediction_interval_width"
    ]:
        raise RuntimeError("Predictive-sensitivity additional-metrics inventory is invalid.")

    if metric_contract.get("condition_calibration_required") is not True:
        raise RuntimeError("Predictive-sensitivity must require condition calibration.")

    if metric_contract.get("primary_comparison_requires_exact_shared_inventory") is not True:
        raise RuntimeError("Predictive-sensitivity comparison must require exact shared inventory.")

    if metric_contract.get("partial_outer_predictions_publishable") is not False:
        raise RuntimeError("Predictive-sensitivity must prohibit partial outer predictions.")

    probability = float(metric_contract["interval_probability"])

    if not (0.0 < probability < 1.0):
        raise RuntimeError("Predictive-sensitivity interval_probability must be strictly between zero and one.")

    execution = policy["execution"]

    if execution["parallel_worker_count"] != 1:
        raise RuntimeError("Predictive-sensitivity execution must be single-worker.")

    if execution["byte_identical_artifacts_required"] is not True:
        raise RuntimeError("Predictive-sensitivity execution must require byte-identical artifacts.")

    if execution["transactional_publication_required"] is not True:
        raise RuntimeError("Predictive-sensitivity execution must require transactional publication.")

    if execution["operations"] != ["validate", "fit"]:
        raise RuntimeError("Predictive-sensitivity operations are invalid.")

    incident_sensitivity = policy["incident_sensitivity"]

    if incident_sensitivity.get(
        "candidate_reason_inventory_rule"
    ) != "all_reason_codes_observed_in_candidate":
        raise RuntimeError("Incident sensitivity must register the complete candidate reason inventory.")

    if incident_sensitivity.get(
        "inadequately_supported_reason_action"
    ) != "report_unavailable":
        raise RuntimeError("Incident sensitivity must report inadequate exclusions as unavailable.")

    predictor_ablation = policy["predictor_ablation"]

    if predictor_ablation.get(
        "inner_validation_uses_ablated_predictor_matrix"
    ) is not True:
        raise RuntimeError("Predictor ablation must validate intervals on the ablated predictor matrix.")

    metric_contract = policy["metric_contract"]

    if metric_contract.get(
        "minimum_interval_diagnostic_independent_cell_count"
    ) != 2:
        raise RuntimeError("Predictive sensitivity must require at least two independent cells for interval diagnostics.")

    qualification = metric_contract.get(
        "analytical_qualification"
    )

    required_qualification = {
        "predictive_r_squared_must_exceed_benchmark": True,
        "condition_calibration_complete_required": True,
        "qualified_status": "available",
        "failed_status": "diagnostic_only",
        "unavailable_status": "not_available"
    }

    if not isinstance(
        qualification,
        dict
    ) or qualification != required_qualification:
        raise RuntimeError("Predictive sensitivity analytical qualification contract is invalid.")

    for family_name, family_policy in policy["expanded_configuration_inventory"].items():
        if (
            not isinstance(
                family_policy.get("configuration_rationale"),
                str
            )
            or not family_policy["configuration_rationale"].strip()
        ):
            raise RuntimeError(f"Expanded configuration family requires a rationale : {family_name}")

    output_contract = policy["output_contract"]

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

    if set(output_contract["tables"]) != expected_tables:
        raise RuntimeError("Predictive-sensitivity output table inventory is invalid.")

    if not output_contract["report_file"].endswith(".json"):
        raise RuntimeError("Predictive-sensitivity report file name is invalid.")


# Confirming the expanded grid is a strict superset of the primary grid, is duplicate-free, uses the correct
#  estimator-specific schema, contains finite values within valid ranges and carries a nonempty rationale
def validate_expanded_grid_against_primary(sensitivity_policy: dict, predictive_policy: dict) -> None:
    expanded_inventory = sensitivity_policy["expanded_configuration_inventory"]
    primary_families = predictive_policy.get("model_families")

    if not isinstance(primary_families, dict):
        raise RuntimeError("Primary predictive policy is missing the model_families section.")

    if set(expanded_inventory) != set(primary_families):
        raise RuntimeError("Expanded configuration inventory does not cover exactly the primary model families.")

    for family_name, primary_family in primary_families.items():
        expanded_family = expanded_inventory[family_name]

        rationale = expanded_family.get("configuration_rationale")

        if not isinstance(rationale, str) or not rationale.strip():
            raise RuntimeError(f"Expanded configuration family requires a nonempty rationale : {family_name}")

        primary_configurations = primary_family.get("configurations", [])

        if not primary_configurations:
            raise RuntimeError(f"Primary configuration inventory is empty : {family_name}")

        primary_keys = set(primary_configurations[0].keys())

        expanded_configurations = expanded_family.get("configurations", [])

        if not expanded_configurations:
            raise RuntimeError(f"Expanded configuration inventory is empty : {family_name}")

        serialized_configurations = [
            canonical_json(configuration)
            for configuration in expanded_configurations
        ]

        if len(serialized_configurations) != len(set(serialized_configurations)):
            raise RuntimeError(f"Expanded configuration inventory contains duplicate configurations : {family_name}")

        for configuration in expanded_configurations:
            if set(configuration.keys()) != primary_keys:
                raise RuntimeError(
                    "Expanded configuration uses a schema inconsistent with the primary grid : "
                    f"{family_name} : {sorted(configuration.keys())}"
                )

        for primary_configuration in primary_configurations:
            if primary_configuration not in expanded_configurations:
                raise RuntimeError(
                    f"Expanded configuration inventory does not contain a primary configuration : {family_name}"
                )

        for configuration in expanded_configurations:
            if family_name == "elastic_net":
                alpha = configuration.get("alpha")
                l1_ratio = configuration.get("l1_ratio")

                if (
                    not isinstance(alpha, (int, float))
                    or isinstance(alpha, bool)
                    or not math.isfinite(float(alpha))
                    or float(alpha) <= 0
                ):
                    raise RuntimeError(f"Expanded elastic-net configuration has invalid alpha : {family_name}")

                if (
                    not isinstance(l1_ratio, (int, float))
                    or isinstance(l1_ratio, bool)
                    or not math.isfinite(float(l1_ratio))
                    or not (0.0 <= float(l1_ratio) <= 1.0)
                ):
                    raise RuntimeError(f"Expanded elastic-net configuration has invalid l1_ratio : {family_name}")

            elif family_name == "random_forest":
                max_depth = configuration.get("max_depth")
                min_samples_leaf = configuration.get("min_samples_leaf")

                if (
                    max_depth is not None
                    and (
                        not isinstance(max_depth, int)
                        or isinstance(max_depth, bool)
                        or max_depth <= 0
                    )
                ):
                    raise RuntimeError(f"Expanded random-forest configuration has invalid max_depth : {family_name}")

                if (
                    not isinstance(min_samples_leaf, int)
                    or isinstance(min_samples_leaf, bool)
                    or min_samples_leaf <= 0
                ):
                    raise RuntimeError(f"Expanded random-forest configuration has invalid min_samples_leaf : {family_name}")

            elif family_name == "gradient_boosting":
                n_estimators = configuration.get("n_estimators")
                learning_rate = configuration.get("learning_rate")
                max_depth = configuration.get("max_depth")
                min_samples_leaf = configuration.get("min_samples_leaf")

                if (
                    not isinstance(n_estimators, int)
                    or isinstance(n_estimators, bool)
                    or n_estimators <= 0
                ):
                    raise RuntimeError(f"Expanded gradient-boosting configuration has invalid n_estimators : {family_name}")

                if (
                    not isinstance(learning_rate, (int, float))
                    or isinstance(learning_rate, bool)
                    or not math.isfinite(float(learning_rate))
                    or float(learning_rate) <= 0
                ):
                    raise RuntimeError(f"Expanded gradient-boosting configuration has invalid learning_rate : {family_name}")

                if (
                    not isinstance(max_depth, int)
                    or isinstance(max_depth, bool)
                    or max_depth <= 0
                ):
                    raise RuntimeError(f"Expanded gradient-boosting configuration has invalid max_depth : {family_name}")

                if (
                    not isinstance(min_samples_leaf, int)
                    or isinstance(min_samples_leaf, bool)
                    or min_samples_leaf <= 0
                ):
                    raise RuntimeError(
                        f"Expanded gradient-boosting configuration has invalid min_samples_leaf : {family_name}"
                    )


# Mapping encoded predictors to hierarchical ablation groups
def feature_group_indices(feature_names: list[str]) -> dict[str, list[int]]:
    horizon_indices = []
    condition_indices = []
    interaction_indices = []

    for index, feature_name in enumerate(
        feature_names
    ):
        if feature_name == "scaled_horizon":
            horizon_indices.append(
                index
            )

        elif feature_name.startswith(
            "condition["
        ):
            condition_indices.append(
                index
            )

        elif feature_name.startswith(
            "scaled_horizon:condition["
        ):
            interaction_indices.append(
                index
            )

        else:
            raise RuntimeError("Predictor matrix contains an unsupported feature.")

    return {
        "horizon_main_and_interactions": sorted(
            horizon_indices
            + interaction_indices
        ),
        "condition_main_and_interactions": sorted(
            condition_indices
            + interaction_indices
        ),
        "horizon_condition_interactions": (
            interaction_indices
        )
    }


# Removing one hierarchical predictor group
def remove_feature_group(training_matrix: np.ndarray, test_matrix: np.ndarray,
                         feature_names: list[str], group_name: str
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str]
]:
    grouped_indices = feature_group_indices(
        feature_names
    )

    if group_name not in grouped_indices:
        raise RuntimeError(f"Predictor group is unsupported : {group_name}")

    removed_indices = set(
        grouped_indices[
            group_name
        ]
    )

    if not removed_indices:
        raise EstimatorUnavailableError("PREDICTOR_GROUP_NOT_APPLICABLE")

    retained_indices = [
        index
        for index in range(
            len(
                feature_names
            )
        )
        if index not in removed_indices
    ]

    if not retained_indices:
        raise EstimatorUnavailableError("ABLATION_REMOVES_COMPLETE_PREDICTOR_MATRIX")

    return (
        training_matrix[
            :,
            retained_indices
        ],
        test_matrix[
            :,
            retained_indices
        ],
        [
            feature_names[
                index
            ]
            for index in retained_indices
        ]
    )


# Evaluating a configuration on grouped inner folds with optional predictor ablation
def evaluate_sensitivity_configuration(candidate_identity: str, model_family: str,
                                       configuration_record: dict, training_observations: list[dict],
                                       condition_identities: list[str], fitting_policy: dict,
                                       ablation_group: str | None = None
) -> dict:
    assignments, fold_count = build_inner_assignments(
        candidate_identity,
        training_observations,
        fitting_policy
    )

    def unavailable(reason_code: str) -> dict:
        return {
            "status": "unavailable",
            "reason_code": reason_code,
            "fold_count": fold_count,
            "metric": None,
            "residuals": None,
            "residual_weights": None
        }

    if assignments is None:
        return unavailable("INSUFFICIENT_INNER_FOLDS")

    prediction_records = []
    predicted_values = []

    for fold_index in range(fold_count):
        inner_training = [
            observation
            for observation in training_observations
            if assignments[observation["cell_identity"]] != fold_index
        ]

        inner_test = [
            observation
            for observation in training_observations
            if (
                assignments[observation["cell_identity"]] == fold_index
                and observation["inside_evaluation_interval"]
            )
        ]

        if not inner_test:
            return unavailable("EMPTY_INNER_TEST_FOLD")

        if not fold_has_horizon_coverage(inner_training, inner_test):
            return unavailable("INNER_TEST_HORIZON_OUTSIDE_TRAINING_RANGE")

        try:
            midpoint, width = calculate_training_horizon_scaling(inner_training)

            training_matrix, feature_names = build_feature_matrix(
                inner_training,
                condition_identities,
                midpoint,
                width
            )

            test_matrix, _ = build_feature_matrix(
                inner_test,
                condition_identities,
                midpoint,
                width
            )

            if ablation_group is not None:
                training_matrix, test_matrix, feature_names = remove_feature_group(
                    training_matrix,
                    test_matrix,
                    feature_names,
                    ablation_group
                )

        except (
            RuntimeError,
            EstimatorUnavailableError
        ) as exc:
            reason_code = getattr(
                exc,
                "reason_code",
                "INSUFFICIENT_HORIZON_VARIATION"
            )

            return unavailable(reason_code)

        estimator = build_estimator(
            model_family,
            configuration_record["configuration"],
            {
                "candidate_identity": candidate_identity,
                "model_family": model_family,
                "configuration_identity": (
                    configuration_record["configuration_identity"]
                ),
                "inner_fold_index": fold_index,
                "ablation_group": ablation_group
            },
            fitting_policy
        )

        try:
            fit_estimator(
                estimator,
                model_family,
                training_matrix,
                np.log(
                    np.asarray(
                        [
                            observation["outcome_value"]
                            for observation in inner_training
                        ],
                        dtype = float
                    )
                ),
                build_cell_weights(inner_training)
            )

            _, predictions = predict_original_scale(estimator, test_matrix)

        except EstimatorUnavailableError as exc:
            return unavailable(exc.reason_code)

        if (
            not np.all(np.isfinite(predictions))
            or np.any(predictions <= 0)
        ):
            return unavailable("NONPOSITIVE_OR_NONFINITE_PREDICTION")

        prediction_records.extend(inner_test)
        predicted_values.extend(predictions.tolist())

    observed = np.asarray(
        [record["outcome_value"] for record in prediction_records],
        dtype = float
    )

    predicted = np.asarray(predicted_values, dtype = float)

    metrics = equal_cell_metrics(
        prediction_records,
        observed,
        predicted
    )

    return {
        "status": "completed",
        "reason_code": None,
        "fold_count": fold_count,
        "metric": metrics["equal_cell_weighted_root_mean_squared_error"],
        "residuals": np.abs(
            np.log(observed) - np.log(predicted)
        ),
        "residual_weights": build_cell_weights(prediction_records)
    }


# Verifying the comparison inventory for one sensitivity variant
def comparison_inventory_status(primary_inventory: dict, variant_inventory: dict,
                                variant_type: str
) -> tuple[bool, bool, str | None]:
    primary_identities = set(primary_inventory)
    variant_identities = set(variant_inventory)

    if not primary_identities:
        return (False, False, "PRIMARY_PREDICTION_INVENTORY_UNAVAILABLE")

    if not variant_identities:
        return (False, False, "VARIANT_PREDICTION_INVENTORY_UNAVAILABLE")

    if variant_identities == primary_identities:
        return (True, True, None)

    if variant_type == "training_window":
        if variant_identities.issubset(primary_identities):
            return (True, False, None)

        return (False, False, "PREDICTION_INVENTORY_MISMATCH")

    return (False, False, "PREDICTION_INVENTORY_MISMATCH")


# Applying first-or-terminal training-window exclusion independently per cell
def apply_training_window(observations: list[dict], variant_identity: str) -> list[dict]:
    horizons_by_cell = defaultdict(list)

    for observation in observations:
        horizons_by_cell[observation["cell_identity"]].append(
            float(observation["horizon"])
        )

    if variant_identity == "first_observation_excluded":
        excluded_horizon_by_cell = {
            cell_identity: min(horizons)
            for cell_identity, horizons in horizons_by_cell.items()
        }
    elif variant_identity == "terminal_observation_excluded":
        excluded_horizon_by_cell = {
            cell_identity: max(horizons)
            for cell_identity, horizons in horizons_by_cell.items()
        }
    else:
        raise RuntimeError("Training-window variant is unregistered.")

    return [
        observation
        for observation in observations
        if float(observation["horizon"])
        != excluded_horizon_by_cell[observation["cell_identity"]]
    ]


# Building the reference-feature reason-code lookup keyed by (source, cell)
def build_reason_lookup(feature_records: list[dict]) -> dict[tuple[str, str], frozenset[str]]:
    lookup = {}

    for record in feature_records:
        identity = (
            str(record["source_identity"]),
            str(record["cell_identity"])
        )

        if identity in lookup:
            raise RuntimeError("Feature provenance contains a duplicate identity.")

        reason_codes = set()

        for field_name in ("archive_quality_reason_codes", "quality_reason_codes"):
            values = record[field_name]

            if not isinstance(values, list):
                raise RuntimeError("Feature reason-code field is not a list.")

            reason_codes.update(values)

        lookup[identity] = frozenset(reason_codes)

    return lookup


# Excluding one documented reason code from observations while preserving eligibility accounting
def apply_incident_exclusion(observations: list[dict], reason_lookup: dict[tuple[str, str], frozenset[str]],
                             reason_code: str, minimum_excluded_cell_count: int,
                             minimum_retained_cell_count: int
) -> tuple[list[dict], dict]:
    excluded = []
    retained = []

    for observation in observations:
        provenance_identity = (
            observation["source_identity"],
            observation["cell_identity"]
        )

        if provenance_identity not in reason_lookup:
            raise RuntimeError("Observation does not map to feature provenance.")

        if reason_code in reason_lookup[provenance_identity]:
            excluded.append(observation)
        else:
            retained.append(observation)

    excluded_cells = {observation["cell_identity"] for observation in excluded}
    retained_cells = {observation["cell_identity"] for observation in retained}

    eligibility = {
        "excluded_observation_count": len(excluded),
        "retained_observation_count": len(retained),
        "excluded_independent_cell_count": len(excluded_cells),
        "retained_independent_cell_count": len(retained_cells),
        "eligible": (
            len(excluded_cells) >= minimum_excluded_cell_count
            and len(retained_cells) >= minimum_retained_cell_count
        )
    }

    return retained, eligibility


# Building deterministic condition-stratified grouped partitions
def build_partition_assignments(candidate_identity: str, observations: list[dict],
                                requested_fold_count: int, minimum_fold_count: int,
                                hash_namespace: str
) -> tuple[
    dict[str, int] | None,
    int
]:
    cells_by_condition = defaultdict(
        set
    )

    conditions_by_cell = defaultdict(
        set
    )

    for observation in observations:
        cell_identity = observation[
            "cell_identity"
        ]

        condition_identity = observation[
            "condition_protocol_identity"
        ]

        cells_by_condition[
            condition_identity
        ].add(
            cell_identity
        )

        conditions_by_cell[
            cell_identity
        ].add(
            condition_identity
        )

    if not cells_by_condition:
        return (
            None,
            0
        )

    if any(
        len(
            conditions
        )
        != 1
        for conditions in conditions_by_cell.values()
    ):
        raise RuntimeError("A cell maps to multiple condition protocols.")

    effective_fold_count = min(
        requested_fold_count,
        min(
            len(
                cell_identities
            )
            for cell_identities in cells_by_condition.values()
        )
    )

    if effective_fold_count < minimum_fold_count:
        return (
            None,
            0
        )

    assignments = {}

    for condition_identity in sorted(
        cells_by_condition
    ):
        ordered_cells = sorted(
            cells_by_condition[
                condition_identity
            ],
            key = lambda cell_identity: (
                hashlib.sha256(
                    (
                        hash_namespace + "|"
                        + candidate_identity + "|"
                        + condition_identity + "|"
                        + cell_identity
                    ).encode(
                        "utf-8"
                    )
                ).hexdigest(),
                cell_identity
            )
        )

        for index, cell_identity in enumerate(
            ordered_cells
        ):
            if cell_identity in assignments:
                raise RuntimeError("A cell received multiple partition assignments.")

            assignments[
                cell_identity
            ] = (
                index
                % effective_fold_count
            )

    if set(
        assignments
    ) != set(
        conditions_by_cell
    ):
        raise RuntimeError("Partition assignment does not cover every cell.")

    return (
        assignments,
        effective_fold_count
    )


# Restricting held-out observations to same-condition training support
def restrict_test_to_training_support(training_observations: list[dict], test_observations: list[dict]) -> tuple[
    list[dict],
    int
]:
    horizons_by_condition = defaultdict(
        list
    )

    for observation in training_observations:
        horizons_by_condition[
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

    retained = []

    for observation in test_observations:
        horizons = horizons_by_condition.get(
            observation[
                "condition_protocol_identity"
            ]
        )

        if not horizons:
            continue

        horizon = float(
            observation[
                "horizon"
            ]
        )

        if min(
            horizons
        ) <= horizon <= max(
            horizons
        ):
            retained.append(
                observation
            )

    return (
        retained,
        len(
            test_observations
        )
        - len(
            retained
        )
    )


# Composing one observation identity for exact shared-inventory comparison.
def prediction_identity(record: dict) -> tuple:
    return (
        str(
            record[
                "cell_identity"
            ]
        ),
        str(
            record[
                "source_identity"
            ]
        ),
        str(
            record[
                "condition_protocol_identity"
            ]
        ),
        float(
            record[
                "horizon"
            ]
        )
    )


# Building one configuration inventory with deterministic identities
def build_config_inventory(family_policy: dict, model_family: str) -> list[dict]:
    records = []

    for configuration in family_policy["configurations"]:
        records.append({
            "configuration_identity": deterministic_identity({
                "model_family": model_family,
                "configuration": configuration
            }),
            "configuration": configuration
        })

    records.sort(key = lambda record: record["configuration_identity"])

    return records


# Building the primary and expanded inventories keyed by family
def build_config_inventories(primary_policy: dict, sensitivity_policy: dict) -> dict[str, dict[str, list[dict]]]:
    inventories = {
        "primary": {},
        "expanded": {}
    }

    for family_name in primary_policy["model_families"]:
        inventories["primary"][family_name] = build_config_inventory(
            primary_policy["model_families"][family_name],
            family_name
        )

    for family_name in sensitivity_policy["expanded_configuration_inventory"]:
        inventories["expanded"][family_name] = build_config_inventory(
            sensitivity_policy["expanded_configuration_inventory"][family_name],
            family_name
        )

    return inventories


# Building the primary selected configuration per candidate, family, and fold
def build_primary_selected_lookup(primary_hyperparameter_selection: list[dict]) -> dict[tuple[str, str, int], str]:
    lookup = {}

    for record in primary_hyperparameter_selection:
        if record["selected"] is not True and record["selected"] != "true":
            continue

        key = (
            record["candidate_identity"],
            record["model_family"],
            int(record["outer_fold_index"])
        )

        if key in lookup:
            raise RuntimeError(f"Primary selection contains duplicate selected config : {key}")

        lookup[key] = record["configuration_identity"]

    return lookup


# Building the primary per-benchmark metrics for later ablation and incident reference
def build_primary_metric_lookup(primary_validation_metrics: list[dict]) -> dict[tuple[str, str], dict]:
    lookup = {}

    for record in primary_validation_metrics:
        if record["validation_status"] != "completed":
            continue

        key = (record["candidate_identity"], record["model_family"])

        if key in lookup:
            raise RuntimeError(f"Primary metrics contain a duplicate completed record : {key}")

        lookup[key] = record

    return lookup


# Merging the primary predictive policy's nested-validation settings and family defaults into the sensitivity
#  policy at runtime, without changing the registered policy file.
def build_sensitivity_fitting_policy(sensitivity_policy: dict, primary_predictive_policy: dict) -> dict:
    if "nested_validation" not in primary_predictive_policy:
        raise RuntimeError("Primary predictive policy does not register nested_validation; sensitivity fitting requires it.")

    if "model_families" not in primary_predictive_policy:
        raise RuntimeError("Primary predictive policy does not register model_families; sensitivity fitting requires it.")

    nested_validation = primary_predictive_policy["nested_validation"]

    required_nested_fields = {
        "inner_requested_fold_count",
        "inner_minimum_fold_count",
        "inner_fold_assignment_seed"
    }

    missing_fields = sorted(
        required_nested_fields.difference(nested_validation)
    )

    if missing_fields:
        raise RuntimeError(
            f"Primary nested_validation is missing fields required by the sensitivity fitter : {missing_fields}"
        )

    required_family_fields = {
        "elastic_net": {"maximum_iterations", "tolerance"},
        "random_forest": {"n_estimators", "max_features"},
        "gradient_boosting": {"loss"}
    }

    for family_name, family_fields in required_family_fields.items():
        if family_name not in primary_predictive_policy["model_families"]:
            raise RuntimeError(f"Primary predictive policy is missing a required model family : {family_name}")

        missing_family_fields = sorted(
            family_fields.difference(
                primary_predictive_policy["model_families"][family_name]
            )
        )

        if missing_family_fields:
            raise RuntimeError(
                f"Primary model family is missing fields required by the sensitivity fitter : "
                f"{family_name} : {missing_family_fields}"
            )

    if "nested_validation" in sensitivity_policy:
        raise RuntimeError(
            "Sensitivity policy must not define nested_validation; it is inherited from the primary predictive policy."
        )

    if "model_families" in sensitivity_policy:
        raise RuntimeError(
            "Sensitivity policy must not define model_families; they are inherited from the primary predictive policy."
        )

    return {
        **sensitivity_policy,
        "nested_validation": nested_validation,
        "model_families": primary_predictive_policy["model_families"]
    }


# Enumerating every documented reason code observed for one candidate
def reason_codes_for_candidate(candidate: dict, reason_lookup: dict[tuple[str, str], frozenset[str]]) -> list[str]:
    reason_codes = set()

    for observation in candidate["included_observations"]:
        identity = (
            observation["source_identity"],
            observation["cell_identity"]
        )

        if identity not in reason_lookup:
            raise RuntimeError("Candidate observation does not map to feature provenance.")

        reason_codes.update(reason_lookup[identity])

    return sorted(reason_codes)


# Building the deterministic one-factor-at-a-time sensitivity registry
def build_sensitivity_registry(candidates: list[dict], reason_lookup: dict[tuple[str, str], frozenset[str]],
                               sensitivity_policy: dict
) -> list[dict]:
    registry = []

    for candidate in candidates:
        candidate_identity = candidate["candidate_identity"]

        for model_family in sorted(sensitivity_policy["expanded_configuration_inventory"]):
            base = {
                "candidate_identity": candidate_identity,
                "model_family": model_family,
                "partition_identity": "primary",
                "configuration_inventory": "primary",
                "training_window_identity": "full",
                "incident_reason_code": None,
                "ablation_group": None
            }

            def emit(variant_type: str, partition_identity: str = "primary",
                     configuration_inventory: str = "primary",
                     training_window_identity: str = "full",
                     incident_reason_code: str | None = None,
                     ablation_group: str | None = None) -> None:
                variant_identity = deterministic_identity({
                    "candidate_identity": candidate_identity,
                    "model_family": model_family,
                    "variant_type": variant_type,
                    "partition_identity": partition_identity,
                    "configuration_inventory": configuration_inventory,
                    "training_window_identity": training_window_identity,
                    "incident_reason_code": incident_reason_code,
                    "ablation_group": ablation_group
                })

                registry.append({
                    "variant_identity": variant_identity,
                    "candidate_identity": candidate_identity,
                    "model_family": model_family,
                    "variant_type": variant_type,
                    "partition_identity": partition_identity,
                    "configuration_inventory": configuration_inventory,
                    "training_window_identity": training_window_identity,
                    "incident_reason_code": incident_reason_code,
                    "ablation_group": ablation_group,
                    "registration_order": len(registry)
                })

            for method in sensitivity_policy["partition_sensitivity"]["assignment_methods"]:
                emit(variant_type = "partition", partition_identity = method["hash_namespace"])

            emit(variant_type = "expanded_config", configuration_inventory = "expanded")

            for window_variant in sensitivity_policy["training_window_sensitivity"]["variants"]:
                emit(variant_type = "training_window", training_window_identity = window_variant["variant_identity"])

            for reason_code in reason_codes_for_candidate(candidate, reason_lookup):
                emit(variant_type = "incident", incident_reason_code = reason_code)

            for ablation_group in sensitivity_policy["predictor_ablation"]["groups"]:
                emit(variant_type = "ablation", ablation_group = ablation_group)

    return registry


# Building the fold records for a new partition assignment
def build_partition_fold_records(candidate: dict, assignments: dict[str, int], fold_count: int) -> list[dict]:
    fold_records = []

    for fold_index in range(fold_count):
        test_cell_identities = sorted(
            cell_identity
            for cell_identity, assigned_fold in assignments.items()
            if assigned_fold == fold_index
        )

        training_cell_identities = sorted(
            set(assignments) - set(test_cell_identities)
        )

        training_observations = [
            observation for observation in candidate["included_observations"]
            if observation["cell_identity"] in set(training_cell_identities)
        ]

        test_observations = [
            observation for observation in candidate["included_observations"]
            if (
                observation["cell_identity"] in set(test_cell_identities)
                and observation["inside_evaluation_interval"]
            )
        ]

        training_condition_count = len({
            observation["condition_protocol_identity"]
            for observation in training_observations
        })

        fold_records.append({
            "fold_index": fold_index,
            "training_cell_identities": training_cell_identities,
            "test_cell_identities": test_cell_identities,
            "training_observation_count": len(training_observations),
            "test_observation_count": len(test_observations),
            "training_condition_count": training_condition_count
        })

    return fold_records


# Fitting one variant for one candidate and returning folds, selection, predictions, and metric
def fit_sensitivity_variant_for_candidate(candidate: dict, variant: dict,
                                          primary_folds_by_candidate: dict[str, list[dict]],
                                          primary_selected_lookup: dict[tuple[str, str, int], str],
                                          config_inventories: dict[str, dict[str, list[dict]]],
                                          sensitivity_policy: dict, reason_lookup: dict[tuple[str, str], frozenset[str]]
) -> tuple[list[dict], list[dict], list[dict], dict]:
    candidate_identity = candidate["candidate_identity"]
    model_family = variant["model_family"]
    variant_type = variant["variant_type"]
    variant_identity = variant["variant_identity"]
    condition_identities = sorted(candidate["condition_identities"])

    partition_unavailable_reason = None
    if variant_type == "partition":
        assignments, effective_fold_count = build_partition_assignments(
            candidate_identity,
            candidate["included_observations"],
            sensitivity_policy["partition_sensitivity"]["fold_count"],
            sensitivity_policy["partition_sensitivity"]["minimum_fold_count"],
            variant["partition_identity"]
        )
        if assignments is None:
            partition_unavailable_reason = "INSUFFICIENT_GROUPED_PARTITION_SUPPORT"
            fold_records_input = []
            use_primary_config = False
            config_inventory = None
        else:
            primary_partition_signature = frozenset(
                frozenset(fold["test_cell_identities"])
                for fold in primary_folds_by_candidate[candidate_identity]
            )
            variant_partition_signature = frozenset(
                frozenset(
                    cell_identity
                    for cell_identity, assigned_fold in assignments.items()
                    if assigned_fold == fold_index
                )
                for fold_index in range(effective_fold_count)
            )

            if (
                primary_partition_signature
                == variant_partition_signature
                and sensitivity_policy["partition_sensitivity"]["duplicate_assignment_action"]
                == "report_uninformative"
            ):
                partition_unavailable_reason = "PARTITION_ASSIGNMENT_UNCHANGED"
                fold_records_input = []
                use_primary_config = False
                config_inventory = None
            else:
                fold_records_input = build_partition_fold_records(
                    candidate, assignments, effective_fold_count
                )
                use_primary_config = False
                config_inventory = config_inventories["primary"][model_family]

    elif variant_type == "expanded_config":
        fold_records_input = primary_folds_by_candidate[candidate_identity]
        use_primary_config = False
        config_inventory = config_inventories["expanded"][model_family]
    elif variant_type in {"training_window", "incident", "ablation"}:
        fold_records_input = primary_folds_by_candidate[candidate_identity]
        use_primary_config = True
        config_inventory = None
    else:
        raise RuntimeError(f"Unsupported sensitivity variant type : {variant_type}")

    fold_records_out = []
    selection_records_out = []
    all_predictions = []
    fold_failure_reason = partition_unavailable_reason

    interval_probability = float(sensitivity_policy["metric_contract"]["interval_probability"])

    incident_fold_eligibility = []

    for fold_record in fold_records_input:
        fold_index = int(fold_record["fold_index"])
        test_cells = set(fold_record["test_cell_identities"])

        training_observations = [
            observation for observation in candidate["included_observations"]
            if observation["cell_identity"] not in test_cells
        ]
        test_observations = [
            observation for observation in candidate["included_observations"]
            if (
                observation["cell_identity"] in test_cells
                and observation["inside_evaluation_interval"]
            )
        ]

        fold_reason_code = None
        excluded_test_observation_count = 0

        if variant_type == "training_window":
            training_observations = apply_training_window(
                training_observations, variant["training_window_identity"]
            )
            test_observations, excluded_test_observation_count = (
                restrict_test_to_training_support(
                    training_observations,
                    test_observations
                )
            )
        elif variant_type == "incident":
            training_observations, eligibility = apply_incident_exclusion(
                training_observations,
                reason_lookup,
                variant["incident_reason_code"],
                sensitivity_policy["incident_sensitivity"]["minimum_excluded_independent_cell_count"],
                sensitivity_policy["incident_sensitivity"]["minimum_retained_independent_cell_count"]
            )
            incident_fold_eligibility.append({
                "fold_index": fold_index,
                **eligibility
            })
            if not eligibility["eligible"]:
                fold_reason_code = "INCIDENT_EXCLUSION_INSUFFICIENT_CELLS"

        if fold_reason_code is None and not test_observations:
            fold_reason_code = "EMPTY_OUTER_TEST_FOLD"

        if fold_reason_code is None and not training_observations:
            fold_reason_code = "EMPTY_TRAINING_OBSERVATIONS"

        if fold_reason_code is None and not fold_has_horizon_coverage(training_observations, test_observations):
            fold_reason_code = "OUTER_TEST_HORIZON_OUTSIDE_TRAINING_RANGE"

        training_matrix = None
        test_matrix = None
        feature_names = None
        horizon_midpoint = None
        horizon_width = None

        if fold_reason_code is None:
            try:
                horizon_midpoint, horizon_width = calculate_training_horizon_scaling(training_observations)
            except RuntimeError:
                fold_reason_code = "INSUFFICIENT_HORIZON_VARIATION"

        if fold_reason_code is None:
            try:
                training_matrix, feature_names = build_feature_matrix(
                    training_observations, condition_identities, horizon_midpoint, horizon_width
                )
                test_matrix, _ = build_feature_matrix(
                    test_observations, condition_identities, horizon_midpoint, horizon_width
                )

                if variant_type == "ablation":
                    training_matrix, test_matrix, feature_names = remove_feature_group(
                        training_matrix, test_matrix, feature_names, variant["ablation_group"]
                    )
            except EstimatorUnavailableError as exc:
                fold_reason_code = exc.reason_code

        selected_config = None
        selected_config_identity = None
        fixed_configuration_evaluation = None
        fold_configuration_records = []

        if fold_reason_code is None:
            if use_primary_config:
                key = (candidate_identity, model_family, fold_index)
                if key not in primary_selected_lookup:
                    fold_reason_code = "PRIMARY_SELECTED_CONFIG_UNAVAILABLE"
                else:
                    selected_config_identity = primary_selected_lookup[key]
                    selected_config = next(
                        (
                            record["configuration"]
                            for record in config_inventories["primary"][model_family]
                            if record["configuration_identity"] == selected_config_identity
                        ),
                        None
                    )
                    if selected_config is None:
                        fold_reason_code = "PRIMARY_SELECTED_CONFIG_UNAVAILABLE"
                    else:
                        fixed_configuration_evaluation = evaluate_sensitivity_configuration(
                            candidate_identity,
                            model_family,
                            {
                                "configuration_identity": selected_config_identity,
                                "configuration": selected_config
                            },
                            training_observations,
                            condition_identities,
                            sensitivity_policy,
                            (
                                variant["ablation_group"]
                                if variant_type == "ablation"
                                else None
                            )
                        )

                        if fixed_configuration_evaluation["status"] != "completed":
                            fold_reason_code = (
                                fixed_configuration_evaluation["reason_code"]
                                or "FIXED_CONFIGURATION_INNER_VALIDATION_UNAVAILABLE"
                            )
                        elif (
                            fixed_configuration_evaluation["residuals"] is None
                            or len(fixed_configuration_evaluation["residuals"]) == 0
                        ):
                            fold_reason_code = "FIXED_CONFIGURATION_RESIDUALS_UNAVAILABLE"
                        else:
                            fold_configuration_records.append({
                                "variant_identity": variant_identity,
                                "candidate_identity": candidate_identity,
                                "model_family": model_family,
                                "outer_fold_index": fold_index,
                                "configuration_identity": selected_config_identity,
                                "configuration": selected_config,
                                "inner_fold_count": fixed_configuration_evaluation["fold_count"],
                                "inner_validation_status": fixed_configuration_evaluation["status"],
                                "inner_validation_reason_code": fixed_configuration_evaluation["reason_code"],
                                "equal_cell_weighted_root_mean_squared_error": fixed_configuration_evaluation["metric"],
                                "selected": True
                            })
            else:
                evaluated = []
                for config_record in config_inventory:
                    evaluation = evaluate_sensitivity_configuration(
                        candidate_identity,
                        model_family,
                        config_record,
                        training_observations,
                        condition_identities,
                        sensitivity_policy,
                        (
                            variant["ablation_group"]
                            if variant_type == "ablation"
                            else None
                        )
                    )

                    fold_configuration_records.append({
                        "variant_identity": variant_identity,
                        "candidate_identity": candidate_identity,
                        "model_family": model_family,
                        "outer_fold_index": fold_index,
                        "configuration_identity": config_record["configuration_identity"],
                        "configuration": config_record["configuration"],
                        "inner_fold_count": evaluation["fold_count"],
                        "inner_validation_status": evaluation["status"],
                        "inner_validation_reason_code": evaluation["reason_code"],
                        "equal_cell_weighted_root_mean_squared_error": evaluation["metric"],
                        "selected": False
                    })

                    if evaluation["status"] == "completed":
                        evaluated.append((
                            evaluation["metric"],
                            config_record["configuration_identity"],
                            config_record,
                            evaluation
                        ))

                if not evaluated:
                    fold_reason_code = "NO_INNER_VALIDATION_CONFIGURATION_AVAILABLE"
                else:
                    best = min(evaluated, key = lambda item: (item[0], item[1]))
                    selected_config_identity = best[1]
                    selected_config = best[2]["configuration"]
                    fixed_configuration_evaluation = best[3]

                    for record in fold_configuration_records:
                        if record["configuration_identity"] == selected_config_identity:
                            record["selected"] = True

                    if (
                        fixed_configuration_evaluation["residuals"] is None
                        or len(fixed_configuration_evaluation["residuals"]) == 0
                    ):
                        fold_reason_code = "FIXED_CONFIGURATION_RESIDUALS_UNAVAILABLE"

        selection_records_out.extend(fold_configuration_records)

        if fold_reason_code is not None:
            fold_records_out.append({
                "variant_identity": variant_identity,
                "candidate_identity": candidate_identity,
                "model_family": model_family,
                "fold_index": fold_index,
                "training_cell_count": len(set(o["cell_identity"] for o in training_observations)),
                "test_cell_count": len(test_cells),
                "training_cell_identities": sorted(set(o["cell_identity"] for o in training_observations)),
                "test_cell_identities": sorted(test_cells),
                "training_observation_count": len(training_observations),
                "test_observation_count": len(test_observations),
                "excluded_test_observation_count": excluded_test_observation_count,
                "training_condition_count": len(set(o["condition_protocol_identity"] for o in training_observations)),
                "fold_status": "unavailable",
                "fold_reason_code": fold_reason_code
            })
            fold_failure_reason = fold_failure_reason or fold_reason_code
            continue

        log_residual_quantile = weighted_quantile(
            fixed_configuration_evaluation["residuals"],
            fixed_configuration_evaluation["residual_weights"],
            interval_probability
        )

        training_target = np.log(np.asarray(
            [observation["outcome_value"] for observation in training_observations], dtype = float
        ))
        training_weights = build_cell_weights(training_observations)

        estimator = build_estimator(
            model_family,
            selected_config,
            {
                "benchmark_identity": f"{candidate_identity}::{model_family}",
                "outer_fold_index": fold_index,
                "configuration_identity": selected_config_identity,
                "variant_identity": variant_identity
            },
            sensitivity_policy
        )

        try:
            fit_estimator(
                estimator, model_family, training_matrix, training_target, training_weights
            )
            predicted_log, predicted = predict_original_scale(estimator, test_matrix)
        except EstimatorUnavailableError as exc:
            fold_records_out.append({
                "variant_identity": variant_identity,
                "candidate_identity": candidate_identity,
                "model_family": model_family,
                "fold_index": fold_index,
                "training_cell_count": len(set(o["cell_identity"] for o in training_observations)),
                "test_cell_count": len(test_cells),
                "training_cell_identities": sorted(set(o["cell_identity"] for o in training_observations)),
                "test_cell_identities": sorted(test_cells),
                "training_observation_count": len(training_observations),
                "test_observation_count": len(test_observations),
                "excluded_test_observation_count": excluded_test_observation_count,
                "training_condition_count": len(set(o["condition_protocol_identity"] for o in training_observations)),
                "fold_status": "unavailable",
                "fold_reason_code": exc.reason_code
            })
            fold_failure_reason = fold_failure_reason or exc.reason_code
            continue

        with np.errstate(over = "ignore", invalid = "ignore"):
            lower = np.exp(predicted_log - log_residual_quantile)
            upper = np.exp(predicted_log + log_residual_quantile)

        benchmark = calculate_condition_benchmark(training_observations, test_observations)

        # Fail-closed validation of the outer predictions and derived intervals before publishing any prediction rows for
        #  this fold. This preserves the invariant that an unavailable metric never carries published predictions.
        if (
            not np.all(np.isfinite(predicted))
            or np.any(predicted <= 0)
        ):
            fold_reason_code = "NONPOSITIVE_OR_NONFINITE_PREDICTION"

        elif (
            not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower <= 0)
            or np.any(lower > upper)
        ):
            fold_reason_code = "INVALID_PREDICTION_INTERVAL"

        elif (
            not np.all(np.isfinite(benchmark))
            or np.any(benchmark <= 0)
        ):
            fold_reason_code = "INVALID_BENCHMARK_PREDICTION"

        if fold_reason_code is not None:
            fold_records_out.append({
                "variant_identity": variant_identity,
                "candidate_identity": candidate_identity,
                "model_family": model_family,
                "fold_index": fold_index,
                "training_cell_count": len(set(o["cell_identity"] for o in training_observations)),
                "test_cell_count": len(test_cells),
                "training_cell_identities": sorted(set(o["cell_identity"] for o in training_observations)),
                "test_cell_identities": sorted(test_cells),
                "training_observation_count": len(training_observations),
                "test_observation_count": len(test_observations),
                "excluded_test_observation_count": excluded_test_observation_count,
                "training_condition_count": len(set(o["condition_protocol_identity"] for o in training_observations)),
                "fold_status": "unavailable",
                "fold_reason_code": fold_reason_code
            })
            fold_failure_reason = fold_failure_reason or fold_reason_code
            continue

        for observation, pred, lo, hi, bench in zip(
            test_observations, predicted, lower, upper, benchmark, strict = True
        ):
            all_predictions.append({
                "variant_identity": variant_identity,
                "variant_type": variant_type,
                "candidate_identity": candidate_identity,
                "model_family": model_family,
                "outer_fold_index": fold_index,
                "cell_identity": observation["cell_identity"],
                "source_identity": observation["source_identity"],
                "condition_protocol_identity": observation["condition_protocol_identity"],
                "horizon": float(observation["horizon"]),
                "observed_outcome": float(observation["outcome_value"]),
                "predicted_outcome": float(pred),
                "prediction_interval_lower": float(lo),
                "prediction_interval_upper": float(hi),
                "benchmark_prediction": float(bench),
                "prediction_status": "completed",
                "prediction_reason_code": None,
                "estimate_label": "predictive_sensitivity"
            })

        fold_records_out.append({
            "variant_identity": variant_identity,
            "candidate_identity": candidate_identity,
            "model_family": model_family,
            "fold_index": fold_index,
            "training_cell_count": len(set(o["cell_identity"] for o in training_observations)),
            "test_cell_count": len(test_cells),
            "training_cell_identities": sorted(set(o["cell_identity"] for o in training_observations)),
            "test_cell_identities": sorted(test_cells),
            "training_observation_count": len(training_observations),
            "test_observation_count": len(test_observations),
            "excluded_test_observation_count": excluded_test_observation_count,
            "training_condition_count": len(set(o["condition_protocol_identity"] for o in training_observations)),
            "fold_status": "completed",
            "fold_reason_code": None
        })

    prediction_keys = [
        (
            record["outer_fold_index"],
            prediction_identity(record)
        )
        for record in all_predictions
    ]

    if len(prediction_keys) != len(set(prediction_keys)):
        raise RuntimeError("Sensitivity predictions contain duplicate observations.")

    if fold_failure_reason is not None:
        all_predictions = []

    metric_record = build_metric_record(
        candidate, variant, sensitivity_policy, all_predictions, fold_failure_reason
    )

    if incident_fold_eligibility:
        metric_record["incident_eligibility"] = {
            "eligible_in_every_fold": all(
                record["eligible"] for record in incident_fold_eligibility
            ),
            "minimum_excluded_independent_cell_count": min(
                record["excluded_independent_cell_count"] for record in incident_fold_eligibility
            ),
            "minimum_retained_independent_cell_count": min(
                record["retained_independent_cell_count"] for record in incident_fold_eligibility
            ),
            "fold_eligibility": incident_fold_eligibility
        }

    return (
        fold_records_out,
        selection_records_out,
        all_predictions,
        metric_record
    )


# Aggregating one variant's predictions into a validation metric record
def build_metric_record(candidate: dict, variant: dict,
                        sensitivity_policy: dict, predictions: list[dict],
                        failure_reason: str | None
) -> dict:
    candidate_identity = candidate["candidate_identity"]
    model_family = variant["model_family"]

    base = {
        "variant_identity": variant["variant_identity"],
        "variant_type": variant["variant_type"],
        "candidate_identity": candidate_identity,
        "model_family": model_family,
        "design_scope": candidate["collapsed_candidate"]["design_scope"],
        "sampling_design_scope": candidate["collapsed_candidate"]["sampling_design_scope"],
        "aging_mode": candidate["collapsed_candidate"]["aging_mode"],
        "outcome": candidate["collapsed_candidate"]["outcome"],
        "cohort": candidate["collapsed_candidate"]["cohort"],
        "horizon_axis": candidate["collapsed_candidate"]["horizon_axis"],
        "outer_fold_count": 0,
        "test_cell_count": 0,
        "test_observation_count": 0,
        "equal_cell_weighted_root_mean_squared_error": None,
        "equal_cell_weighted_mean_absolute_error": None,
        "equal_cell_weighted_predictive_r_squared": None,
        "equal_cell_weighted_prediction_interval_coverage": None,
        "equal_cell_weighted_mean_prediction_interval_width": None,
        "validation_status": "unavailable",
        "validation_reason_code": failure_reason or "NO_ELIGIBLE_OBSERVATIONS",
        "analytical_status": "not_available",
        "analytical_reason_code": failure_reason or "NO_ELIGIBLE_OBSERVATIONS",
        "estimate_label": "predictive_sensitivity"
    }

    if failure_reason is not None or not predictions:
        return base

    intervals_complete = all(
        record["prediction_interval_lower"] is not None
        and record["prediction_interval_upper"] is not None
        for record in predictions
    )

    if not intervals_complete:
        base["validation_reason_code"] = "INCOMPLETE_PREDICTION_INTERVALS"
        base["analytical_reason_code"] = "INCOMPLETE_PREDICTION_INTERVALS"
        return base

    observation_records = [
        {"cell_identity": record["cell_identity"]} for record in predictions
    ]
    observed = np.asarray(
        [record["observed_outcome"] for record in predictions], dtype = float
    )
    predicted = np.asarray(
        [record["predicted_outcome"] for record in predictions], dtype = float
    )
    lower = np.asarray(
        [record["prediction_interval_lower"] for record in predictions], dtype = float
    )
    upper = np.asarray(
        [record["prediction_interval_upper"] for record in predictions], dtype = float
    )
    benchmark = np.asarray(
        [record["benchmark_prediction"] for record in predictions], dtype = float
    )

    if (
        not np.all(np.isfinite(predicted))
        or np.any(predicted <= 0)
    ):
        base["validation_reason_code"] = "NONPOSITIVE_OR_NONFINITE_PREDICTION"
        base["analytical_reason_code"] = "NONPOSITIVE_OR_NONFINITE_PREDICTION"
        return base

    metrics = equal_cell_metrics(
        observation_records, observed, predicted, lower, upper, benchmark
    )

    base["outer_fold_count"] = len({record["outer_fold_index"] for record in predictions})
    base["test_cell_count"] = len({record["cell_identity"] for record in predictions})
    base["test_observation_count"] = len(predictions)
    base["equal_cell_weighted_root_mean_squared_error"] = metrics["equal_cell_weighted_root_mean_squared_error"]
    base["equal_cell_weighted_mean_absolute_error"] = metrics["equal_cell_weighted_mean_absolute_error"]
    base["equal_cell_weighted_predictive_r_squared"] = metrics["equal_cell_weighted_predictive_r_squared"]
    base["equal_cell_weighted_prediction_interval_coverage"] = metrics["prediction_interval_coverage"]
    base["equal_cell_weighted_mean_prediction_interval_width"] = metrics["mean_prediction_interval_width"]
    base["validation_status"] = "completed"
    base["validation_reason_code"] = None

    return base


# Qualifying every fitted variant from grouped predictive performance and calibration completeness
def qualify_sensitivity_metrics(metrics: list[dict], calibration_records: list[dict],
                                candidate_conditions: dict[str, list[str]], policy: dict
) -> None:
    calibration_by_variant_condition = {
        (
            record["variant_identity"],
            record["condition_protocol_identity"]
        ): record
        for record in calibration_records
        if record["condition_protocol_identity"] is not None
    }

    qualification = policy["metric_contract"]["analytical_qualification"]

    for metric in metrics:
        if metric["validation_status"] != "completed":
            metric["analytical_status"] = qualification["unavailable_status"]
            metric["analytical_reason_code"] = metric["validation_reason_code"]
            continue

        predictive_r_squared = metric["equal_cell_weighted_predictive_r_squared"]

        if (
            qualification["predictive_r_squared_must_exceed_benchmark"]
            and (
                predictive_r_squared is None
                or predictive_r_squared <= 0
            )
        ):
            metric["analytical_status"] = qualification["failed_status"]
            metric["analytical_reason_code"] = "PREDICTIVE_BENCHMARK_NOT_MET"
            continue

        incomplete_calibration = any(
            calibration_by_variant_condition.get(
                (
                    metric["variant_identity"],
                    condition_identity
                ),
                {}
            ).get("calibration_status") != "completed"
            for condition_identity in candidate_conditions.get(
                metric["candidate_identity"],
                []
            )
        )

        if (
            qualification["condition_calibration_complete_required"]
            and incomplete_calibration
        ):
            metric["analytical_status"] = qualification["failed_status"]
            metric["analytical_reason_code"] = "INCOMPLETE_CONDITION_CALIBRATION"
            continue

        metric["analytical_status"] = qualification["qualified_status"]
        metric["analytical_reason_code"] = None


# Building ablation comparison rows against exact shared primary prediction inventory
def build_ablation_rows(metrics: list[dict], sensitivity_predictions: list[dict],
                        primary_predictions_by_benchmark: dict[tuple[str, str], dict],
                        primary_metric_lookup: dict[tuple[str, str], dict], registry_by_variant: dict[str, dict]
) -> list[dict]:
    predictions_by_variant = defaultdict(dict)

    for record in sensitivity_predictions:
        predictions_by_variant[
            record["variant_identity"]
        ][
            prediction_identity(record)
        ] = record

    rows = []

    for metric in metrics:
        if metric["variant_type"] != "ablation":
            continue

        registry_record = registry_by_variant.get(metric["variant_identity"])

        if registry_record is None:
            raise RuntimeError(f"Ablation metric has no matching registry record : {metric['variant_identity']}")

        key = (
            metric["candidate_identity"], metric["model_family"]
        )
        primary_inventory = primary_predictions_by_benchmark.get(
            key, {}
        )
        variant_inventory = predictions_by_variant.get(
            metric["variant_identity"], {}
        )

        row = {
            "variant_identity": metric["variant_identity"],
            "candidate_identity": metric["candidate_identity"],
            "model_family": metric["model_family"],
            "ablation_group": registry_record["ablation_group"],
            "primary_root_mean_squared_error": None,
            "ablated_root_mean_squared_error": None,
            "root_mean_squared_error_difference": None,
            "primary_mean_absolute_error": None,
            "ablated_mean_absolute_error": None,
            "mean_absolute_error_difference": None,
            "primary_predictive_r_squared": None,
            "ablated_predictive_r_squared": None,
            "predictive_r_squared_difference": None,
            "variant_validation_status": metric["validation_status"],
            "variant_analytical_status": metric.get("analytical_status", "not_available"),
            "comparison_status": "unavailable",
            "comparison_reason_code": None,
            "estimate_label": "predictive_sensitivity"
        }

        primary_metric_record = primary_metric_lookup.get(key)

        primary_is_available = (
            primary_metric_record is not None
            and primary_metric_record["analytical_status"] == "available"
        )

        variant_is_available = metric.get("analytical_status") == "available"

        is_valid, exact_match, inventory_reason = comparison_inventory_status(
            primary_inventory, variant_inventory, metric["variant_type"]
        )

        if not primary_is_available:
            row["comparison_reason_code"] = "PRIMARY_BENCHMARK_NOT_ANALYTICALLY_AVAILABLE"

        elif metric["validation_status"] != "completed":
            row["comparison_reason_code"] = "VARIANT_VALIDATION_UNAVAILABLE"

        elif not is_valid:
            row["comparison_reason_code"] = inventory_reason

        elif not variant_is_available:
            row["comparison_status"] = "diagnostic_only"
            row["comparison_reason_code"] = metric.get(
                "analytical_reason_code",
                "VARIANT_BENCHMARK_NOT_ANALYTICALLY_AVAILABLE"
            )

        else:
            shared = sorted(set(primary_inventory).intersection(variant_inventory))

            primary_metric = compute_shared_metric_from_predictions(
                [primary_inventory[i] for i in shared],
                predicted_field = "predicted_outcome"
            )
            variant_metric = compute_shared_metric_from_predictions(
                [variant_inventory[i] for i in shared],
                predicted_field = "predicted_outcome"
            )

            if primary_metric is None or variant_metric is None:
                row["comparison_reason_code"] = "SHARED_METRIC_UNAVAILABLE"

            else:
                row["primary_root_mean_squared_error"] = primary_metric["rmse"]
                row["ablated_root_mean_squared_error"] = variant_metric["rmse"]
                row["root_mean_squared_error_difference"] = variant_metric["rmse"] - primary_metric["rmse"]
                row["primary_mean_absolute_error"] = primary_metric["mae"]
                row["ablated_mean_absolute_error"] = variant_metric["mae"]
                row["mean_absolute_error_difference"] = variant_metric["mae"] - primary_metric["mae"]
                row["primary_predictive_r_squared"] = primary_metric["r_squared"]
                row["ablated_predictive_r_squared"] = variant_metric["r_squared"]

                if (
                    primary_metric["r_squared"] is not None
                    and variant_metric["r_squared"] is not None
                ):
                    row["predictive_r_squared_difference"] = (
                        variant_metric["r_squared"] - primary_metric["r_squared"]
                    )

                row["comparison_status"] = "completed"
                row["comparison_reason_code"] = None

        rows.append(row)

    return rows


# Building incident-exclusion comparison rows against exact shared primary prediction inventory
def build_incident_rows(metrics: list[dict], sensitivity_predictions: list[dict],
                        primary_predictions_by_benchmark: dict[tuple[str, str], dict],
                        primary_metric_lookup: dict[tuple[str, str], dict], registry_by_variant: dict[str, dict]
) -> list[dict]:
    predictions_by_variant = defaultdict(dict)

    for record in sensitivity_predictions:
        predictions_by_variant[
            record["variant_identity"]
        ][
            prediction_identity(record)
        ] = record

    rows = []

    for metric in metrics:
        if metric["variant_type"] != "incident":
            continue

        registry_record = registry_by_variant.get(metric["variant_identity"])

        if registry_record is None:
            raise RuntimeError(f"Incident metric has no matching registry record : {metric['variant_identity']}")

        eligibility = metric.get("incident_eligibility") or {}

        key = (metric["candidate_identity"], metric["model_family"])
        primary_inventory = primary_predictions_by_benchmark.get(key, {})
        variant_inventory = predictions_by_variant.get(metric["variant_identity"], {})

        row = {
            "variant_identity": metric["variant_identity"],
            "candidate_identity": metric["candidate_identity"],
            "model_family": metric["model_family"],
            "incident_reason_code": registry_record["incident_reason_code"],
            "eligible_in_every_fold": eligibility.get("eligible_in_every_fold"),
            "minimum_excluded_independent_cell_count": eligibility.get(
                "minimum_excluded_independent_cell_count"
            ),
            "minimum_retained_independent_cell_count": eligibility.get(
                "minimum_retained_independent_cell_count"
            ),
            "primary_root_mean_squared_error": None,
            "excluded_root_mean_squared_error": None,
            "root_mean_squared_error_difference": None,
            "primary_mean_absolute_error": None,
            "excluded_mean_absolute_error": None,
            "mean_absolute_error_difference": None,
            "primary_predictive_r_squared": None,
            "excluded_predictive_r_squared": None,
            "predictive_r_squared_difference": None,
            "variant_validation_status": metric["validation_status"],
            "variant_analytical_status": metric.get("analytical_status", "not_available"),
            "comparison_status": "unavailable",
            "comparison_reason_code": None,
            "estimate_label": "predictive_sensitivity"
        }

        primary_metric_record = primary_metric_lookup.get(key)

        primary_is_available = (
            primary_metric_record is not None
            and primary_metric_record["analytical_status"] == "available"
        )

        variant_is_available = metric.get("analytical_status") == "available"

        is_valid, exact_match, inventory_reason = comparison_inventory_status(
            primary_inventory, variant_inventory, metric["variant_type"]
        )

        if eligibility.get("eligible_in_every_fold") is False:
            row["comparison_reason_code"] = "INCIDENT_EXCLUSION_INSUFFICIENT_CELLS"

        elif not primary_is_available:
            row["comparison_reason_code"] = "PRIMARY_BENCHMARK_NOT_ANALYTICALLY_AVAILABLE"

        elif metric["validation_status"] != "completed":
            row["comparison_reason_code"] = "VARIANT_VALIDATION_UNAVAILABLE"

        elif not is_valid:
            row["comparison_reason_code"] = inventory_reason

        elif not variant_is_available:
            row["comparison_status"] = "diagnostic_only"
            row["comparison_reason_code"] = metric.get(
                "analytical_reason_code",
                "VARIANT_BENCHMARK_NOT_ANALYTICALLY_AVAILABLE"
            )

        else:
            shared = sorted(set(primary_inventory).intersection(variant_inventory))

            primary_metric = compute_shared_metric_from_predictions(
                [primary_inventory[i] for i in shared],
                predicted_field = "predicted_outcome"
            )
            variant_metric = compute_shared_metric_from_predictions(
                [variant_inventory[i] for i in shared],
                predicted_field = "predicted_outcome"
            )

            if primary_metric is None or variant_metric is None:
                row["comparison_reason_code"] = "SHARED_METRIC_UNAVAILABLE"

            else:
                row["primary_root_mean_squared_error"] = primary_metric["rmse"]
                row["excluded_root_mean_squared_error"] = variant_metric["rmse"]
                row["root_mean_squared_error_difference"] = variant_metric["rmse"] - primary_metric["rmse"]
                row["primary_mean_absolute_error"] = primary_metric["mae"]
                row["excluded_mean_absolute_error"] = variant_metric["mae"]
                row["mean_absolute_error_difference"] = variant_metric["mae"] - primary_metric["mae"]
                row["primary_predictive_r_squared"] = primary_metric["r_squared"]
                row["excluded_predictive_r_squared"] = variant_metric["r_squared"]

                if (
                    primary_metric["r_squared"] is not None
                    and variant_metric["r_squared"] is not None
                ):
                    row["predictive_r_squared_difference"] = (
                        variant_metric["r_squared"] - primary_metric["r_squared"]
                    )

                row["comparison_status"] = "completed"
                row["comparison_reason_code"] = None

        rows.append(row)

    return rows


# Computing equal-cell-weighted metrics from one prediction inventory
def compute_shared_metric_from_predictions(records: list[dict], predicted_field: str) -> dict | None:
    if not records:
        return None

    observation_records = [
        {"cell_identity": record["cell_identity"]} for record in records
    ]
    observed = np.asarray(
        [record["observed_outcome"] for record in records], dtype = float
    )
    predicted = np.asarray(
        [record[predicted_field] for record in records], dtype = float
    )
    benchmark = np.asarray(
        [record["benchmark_prediction"] for record in records], dtype = float
    )

    if (
        not np.all(np.isfinite(predicted))
        or np.any(predicted <= 0)
    ):
        return None

    metrics = equal_cell_metrics(
        observation_records, observed, predicted, benchmark = benchmark
    )

    return {
        "rmse": metrics["equal_cell_weighted_root_mean_squared_error"],
        "mae": metrics["equal_cell_weighted_mean_absolute_error"],
        "r_squared": metrics["equal_cell_weighted_predictive_r_squared"]
    }


# Building the exact-shared-inventory comparison table for every variant
def build_sensitivity_comparisons(metrics: list[dict], sensitivity_predictions: list[dict],
                                  primary_predictions_by_benchmark: dict[tuple[str, str], dict],
                                  primary_metric_lookup: dict[tuple[str, str], dict],
                                  registry_by_variant: dict[str, dict], excluded_test_observation_counts: dict[str, int]
) -> list[dict]:
    predictions_by_variant = defaultdict(dict)

    for record in sensitivity_predictions:
        predictions_by_variant[
            record["variant_identity"]
        ][
            prediction_identity(record)
        ] = record

    rows = []

    for metric in metrics:
        registry_record = registry_by_variant.get(metric["variant_identity"])

        if registry_record is None:
            raise RuntimeError(f"Metric has no matching registry record : {metric['variant_identity']}")

        key = (
            metric["candidate_identity"], metric["model_family"]
        )
        primary_inventory = primary_predictions_by_benchmark.get(
            key, {}
        )
        variant_inventory = predictions_by_variant.get(
            metric["variant_identity"], {}
        )

        row = {
            "variant_identity": metric["variant_identity"],
            "variant_type": metric["variant_type"],
            "candidate_identity": metric["candidate_identity"],
            "model_family": metric["model_family"],
            "design_scope": metric["design_scope"],
            "sampling_design_scope": metric["sampling_design_scope"],
            "aging_mode": metric["aging_mode"],
            "outcome": metric["outcome"],
            "cohort": metric["cohort"],
            "horizon_axis": metric["horizon_axis"],
            "shared_observation_count": 0,
            "shared_independent_cell_count": 0,
            "excluded_test_observation_count": excluded_test_observation_counts.get(
                metric["variant_identity"], 0
            ),
            "primary_equal_cell_weighted_root_mean_squared_error": None,
            "variant_equal_cell_weighted_root_mean_squared_error": None,
            "root_mean_squared_error_difference": None,
            "primary_equal_cell_weighted_mean_absolute_error": None,
            "variant_equal_cell_weighted_mean_absolute_error": None,
            "mean_absolute_error_difference": None,
            "primary_equal_cell_weighted_predictive_r_squared": None,
            "variant_equal_cell_weighted_predictive_r_squared": None,
            "predictive_r_squared_difference": None,
            "primary_inventory_count": len(primary_inventory),
            "variant_inventory_count": len(variant_inventory),
            "exact_inventory_match": False,
            "variant_analytical_status": metric.get("analytical_status", "not_available"),
            "comparison_status": "unavailable",
            "comparison_reason_code": None,
            "estimate_label": "predictive_sensitivity"
        }

        is_valid, exact_match, inventory_reason = comparison_inventory_status(
            primary_inventory, variant_inventory, metric["variant_type"]
        )

        row["exact_inventory_match"] = exact_match

        if not primary_inventory:
            row["comparison_reason_code"] = "PRIMARY_PREDICTION_INVENTORY_UNAVAILABLE"
            rows.append(row)
            continue

        primary_metric_record = primary_metric_lookup.get(key)

        if primary_metric_record is None:
            row["comparison_reason_code"] = "PRIMARY_METRIC_UNAVAILABLE"
            rows.append(row)
            continue

        if primary_metric_record["analytical_status"] != "available":
            row["comparison_reason_code"] = "PRIMARY_BENCHMARK_NOT_ANALYTICALLY_AVAILABLE"
            rows.append(row)
            continue

        if metric["validation_status"] != "completed":
            row["comparison_reason_code"] = "VARIANT_VALIDATION_UNAVAILABLE"
            rows.append(row)
            continue

        if not is_valid:
            row["comparison_reason_code"] = inventory_reason
            rows.append(row)
            continue

        shared = sorted(set(primary_inventory).intersection(variant_inventory))

        if not shared:
            row["comparison_reason_code"] = "NO_SHARED_HELD_OUT_OBSERVATIONS"
            rows.append(row)
            continue

        row["shared_observation_count"] = len(shared)
        row["shared_independent_cell_count"] = len(
            {primary_inventory[i]["cell_identity"] for i in shared}
        )

        primary_metric = compute_shared_metric_from_predictions(
            [primary_inventory[i] for i in shared],
            predicted_field = "predicted_outcome"
        )
        variant_metric = compute_shared_metric_from_predictions(
            [variant_inventory[i] for i in shared],
            predicted_field = "predicted_outcome"
        )

        if primary_metric is None or variant_metric is None:
            row["comparison_reason_code"] = "SHARED_METRIC_UNAVAILABLE"
            rows.append(row)
            continue

        row["primary_equal_cell_weighted_root_mean_squared_error"] = primary_metric["rmse"]
        row["variant_equal_cell_weighted_root_mean_squared_error"] = variant_metric["rmse"]
        row["root_mean_squared_error_difference"] = variant_metric["rmse"] - primary_metric["rmse"]
        row["primary_equal_cell_weighted_mean_absolute_error"] = primary_metric["mae"]
        row["variant_equal_cell_weighted_mean_absolute_error"] = variant_metric["mae"]
        row["mean_absolute_error_difference"] = variant_metric["mae"] - primary_metric["mae"]
        row["primary_equal_cell_weighted_predictive_r_squared"] = primary_metric["r_squared"]
        row["variant_equal_cell_weighted_predictive_r_squared"] = variant_metric["r_squared"]

        if (
            primary_metric["r_squared"] is not None
            and variant_metric["r_squared"] is not None
        ):
            row["predictive_r_squared_difference"] = (
                variant_metric["r_squared"] - primary_metric["r_squared"]
            )

        variant_status = metric.get("analytical_status", "not_available")

        if variant_status == "available":
            row["comparison_status"] = "completed"
            row["comparison_reason_code"] = None

        elif variant_status == "diagnostic_only":
            row["comparison_status"] = "diagnostic_only"
            row["comparison_reason_code"] = metric.get(
                "analytical_reason_code",
                "VARIANT_BENCHMARK_DIAGNOSTIC_ONLY"
            )

        else:
            row["comparison_status"] = "unavailable"
            row["comparison_reason_code"] = metric.get(
                "analytical_reason_code",
                "VARIANT_BENCHMARK_NOT_ANALYTICALLY_AVAILABLE"
            )

        rows.append(row)

    return rows


# Summarizing prediction-interval coverage and width across registered groupings
def build_interval_diagnostics(metrics: list[dict], sensitivity_predictions: list[dict],
                               sensitivity_policy: dict, candidate_condition_identities: dict[str, list[str]]
) -> list[dict]:
    minimum_cells = int(
        sensitivity_policy["metric_contract"]["minimum_interval_diagnostic_independent_cell_count"]
    )

    nominal_probability = float(
        sensitivity_policy["metric_contract"]["interval_probability"]
    )

    predictions_by_variant = defaultdict(list)

    for record in sensitivity_predictions:
        predictions_by_variant[record["variant_identity"]].append(record)

    metrics_by_variant = {
        record["variant_identity"]: record for record in metrics
    }

    rows = []

    for variant_identity in sorted(metrics_by_variant):
        metric = metrics_by_variant[variant_identity]
        variant_records = predictions_by_variant.get(variant_identity, [])

        # Unavailable variants still receive one overall interval record so the interval table reconciles
        #  to the registry and to the metric counts.
        if metric["validation_status"] != "completed":
            rows.append({
                "variant_identity": variant_identity,
                "variant_type": metric["variant_type"],
                "candidate_identity": metric["candidate_identity"],
                "model_family": metric["model_family"],
                "grouping_dimension": "overall",
                "grouping_value": "all",
                "nominal_interval_probability": nominal_probability,
                "prediction_count": 0,
                "independent_cell_count": 0,
                "covered_prediction_count": 0,
                "equal_cell_weighted_interval_coverage": None,
                "equal_cell_weighted_mean_interval_width": None,
                "assessment_status": "unavailable",
                "assessment_reason_code": (
                    metric.get("validation_reason_code")
                    or "VARIANT_VALIDATION_UNAVAILABLE"
                ),
                "estimate_label": "predictive_sensitivity"
            })
            continue

        # Completed variants receive one record for the overall population, one record for every registered outer fold,
        #  and one record for every registered candidate condition. Folds and conditions that produced no
        #  predictions still receive explicit records.
        fold_groups = {
            fold_index: [
                record
                for record in variant_records
                if record["outer_fold_index"] == fold_index
            ]
            for fold_index in range(int(metric["outer_fold_count"]))
        }

        condition_groups = {
            condition_identity: [
                record
                for record in variant_records
                if record["condition_protocol_identity"] == condition_identity
            ]
            for condition_identity in candidate_condition_identities.get(
                metric["candidate_identity"],
                []
            )
        }

        groupings = [("overall", "all", variant_records)]

        for fold_index in sorted(fold_groups):
            groupings.append((
                "validation_fold",
                str(fold_index),
                fold_groups[fold_index]
            ))

        for condition_identity in sorted(condition_groups):
            groupings.append((
                "condition_protocol",
                condition_identity,
                condition_groups[condition_identity]
            ))

        for grouping_dimension, grouping_value, records in groupings:
            available_records = [
                record
                for record in records
                if record["prediction_interval_lower"] is not None
                and record["prediction_interval_upper"] is not None
                and record["prediction_interval_lower"] <= record["prediction_interval_upper"]
            ]

            cell_count = len({record["cell_identity"] for record in available_records})

            coverage = None
            mean_width = None
            covered_count = 0
            assessment_status = "unavailable"
            assessment_reason_code = None

            if not records:
                assessment_reason_code = "NO_PREDICTIONS"

            elif cell_count < minimum_cells:
                assessment_status = "diagnostic_only"
                assessment_reason_code = "INSUFFICIENT_INDEPENDENT_CELLS"

            elif len(available_records) != len(records):
                assessment_status = "diagnostic_only"
                assessment_reason_code = "INCOMPLETE_INTERVALS"

            else:
                assessment_status = "completed"
                assessment_reason_code = None

            # Preserve interval-specific sample-size restrictions before applying the source model's analytical availability.
            if assessment_status == "completed":
                source_analytical_status = metric.get(
                    "analytical_status",
                    "not_available"
                )

                if source_analytical_status == "diagnostic_only":
                    assessment_status = "diagnostic_only"
                    assessment_reason_code = (
                        metric.get("analytical_reason_code")
                        or "SOURCE_VARIANT_DIAGNOSTIC_ONLY"
                    )

                elif source_analytical_status != "available":
                    assessment_status = "unavailable"
                    assessment_reason_code = (
                        metric.get("analytical_reason_code")
                        or "SOURCE_VARIANT_NOT_ANALYTICALLY_AVAILABLE"
                    )

            if available_records:
                observation_records = [
                    {"cell_identity": record["cell_identity"]} for record in available_records
                ]

                weights = build_cell_weights(observation_records)
                weight_sum = sum(weights)

                covered_flags = [
                    (
                        1.0
                        if (
                            record["prediction_interval_lower"]
                            <= record["observed_outcome"]
                            <= record["prediction_interval_upper"]
                        )
                        else 0.0
                    )
                    for record in available_records
                ]

                covered_count = int(sum(covered_flags))

                coverage = sum(
                    weight * flag
                    for weight, flag in zip(weights, covered_flags, strict = True)
                ) / weight_sum

                mean_width = sum(
                    weight * (
                        record["prediction_interval_upper"]
                        - record["prediction_interval_lower"]
                    )
                    for weight, record in zip(weights, available_records, strict = True)
                ) / weight_sum

            rows.append({
                "variant_identity": variant_identity,
                "variant_type": metric["variant_type"],
                "candidate_identity": metric["candidate_identity"],
                "model_family": metric["model_family"],
                "grouping_dimension": grouping_dimension,
                "grouping_value": grouping_value,
                "nominal_interval_probability": nominal_probability,
                "prediction_count": len(records),
                "independent_cell_count": cell_count,
                "covered_prediction_count": covered_count,
                "equal_cell_weighted_interval_coverage": coverage,
                "equal_cell_weighted_mean_interval_width": mean_width,
                "assessment_status": assessment_status,
                "assessment_reason_code": assessment_reason_code,
                "estimate_label": "predictive_sensitivity"
            })

    return rows


# Building per-condition calibration records for every completed variant,
# covering every candidate condition including those with no variant predictions
def build_condition_calibration(metrics: list[dict], sensitivity_predictions: list[dict],
                                candidate_condition_identities: dict[str, list[str]]
) -> list[dict]:
    predictions_by_variant = defaultdict(list)

    for record in sensitivity_predictions:
        predictions_by_variant[record["variant_identity"]].append(record)

    rows = []

    for metric in metrics:
        candidate_identity = metric["candidate_identity"]
        model_family = metric["model_family"]
        all_conditions = candidate_condition_identities.get(
            candidate_identity,
            []
        )

        if metric["validation_status"] != "completed":
            rows.append({
                "variant_identity": metric["variant_identity"],
                "variant_type": metric["variant_type"],
                "candidate_identity": candidate_identity,
                "model_family": model_family,
                "condition_protocol_identity": None,
                "independent_cell_count": 0,
                "observation_count": 0,
                "equal_cell_weighted_mean_observed": None,
                "equal_cell_weighted_mean_predicted": None,
                "equal_cell_weighted_mean_error": None,
                "equal_cell_weighted_mean_absolute_error": None,
                "equal_cell_weighted_root_mean_squared_error": None,
                "calibration_status": "unavailable",
                "calibration_reason_code": "VARIANT_VALIDATION_UNAVAILABLE",
                "estimate_label": "predictive_sensitivity"
            })
            continue

        variant_predictions = predictions_by_variant[metric["variant_identity"]]
        conditions_with_predictions = {
            record["condition_protocol_identity"]
            for record in variant_predictions
        }

        for condition_identity in sorted(all_conditions):
            if condition_identity not in conditions_with_predictions:
                rows.append({
                    "variant_identity": metric["variant_identity"],
                    "variant_type": metric["variant_type"],
                    "candidate_identity": candidate_identity,
                    "model_family": model_family,
                    "condition_protocol_identity": condition_identity,
                    "independent_cell_count": 0,
                    "observation_count": 0,
                    "equal_cell_weighted_mean_observed": None,
                    "equal_cell_weighted_mean_predicted": None,
                    "equal_cell_weighted_mean_error": None,
                    "equal_cell_weighted_mean_absolute_error": None,
                    "equal_cell_weighted_root_mean_squared_error": None,
                    "calibration_status": "unavailable",
                    "calibration_reason_code": "EMPTY_CONDITION_PREDICTION_INVENTORY",
                    "estimate_label": "predictive_sensitivity"
                })
                continue

            condition_records = [
                record for record in variant_predictions
                if record["condition_protocol_identity"] == condition_identity
            ]

            observation_records = [
                {"cell_identity": record["cell_identity"]} for record in condition_records
            ]
            observed = np.asarray(
                [record["observed_outcome"] for record in condition_records], dtype = float
            )
            predicted = np.asarray(
                [record["predicted_outcome"] for record in condition_records], dtype = float
            )

            weights = build_cell_weights(observation_records)

            rows.append({
                "variant_identity": metric["variant_identity"],
                "variant_type": metric["variant_type"],
                "candidate_identity": candidate_identity,
                "model_family": model_family,
                "condition_protocol_identity": condition_identity,
                "independent_cell_count": len({record["cell_identity"] for record in condition_records}),
                "observation_count": len(condition_records),
                "equal_cell_weighted_mean_observed": float(
                    np.average(observed, weights = weights)
                ),
                "equal_cell_weighted_mean_predicted": float(
                    np.average(predicted, weights = weights)
                ),
                "equal_cell_weighted_mean_error": float(
                    np.average(predicted - observed, weights = weights)
                ),
                "equal_cell_weighted_mean_absolute_error": float(
                    np.average(np.abs(predicted - observed), weights = weights)
                ),
                "equal_cell_weighted_root_mean_squared_error": float(
                    math.sqrt(
                        np.average(np.square(predicted - observed), weights = weights)
                    )
                ),
                "calibration_status": "completed",
                "calibration_reason_code": None,
                "estimate_label": "predictive_sensitivity"
            })

    return rows


# Running the complete fitting workflow across the registered sensitivity registry
def run_sensitivity_fitting(candidates: list[dict], registry: list[dict],
                            primary_folds_by_candidate: dict[str, list[dict]],
                            primary_selected_lookup: dict[tuple[str, str, int], str],
                            primary_predictions_by_benchmark: dict[tuple[str, str], dict],
                            primary_metric_lookup: dict[tuple[str, str], dict],
                            config_inventories: dict[str, dict[str, list[dict]]],
                            sensitivity_policy: dict, reason_lookup: dict[tuple[str, str], frozenset[str]]
) -> dict:
    candidate_by_identity = {c["candidate_identity"]: c for c in candidates}

    fold_records_all = []
    selection_records_all = []
    predictions_all = []
    metrics_all = []

    for variant in registry:
        candidate = candidate_by_identity[variant["candidate_identity"]]

        (
            fold_records,
            selection_records,
            predictions,
            metric
        ) = fit_sensitivity_variant_for_candidate(
            candidate,
            variant,
            primary_folds_by_candidate,
            primary_selected_lookup,
            config_inventories,
            sensitivity_policy,
            reason_lookup
        )

        fold_records_all.extend(fold_records)
        selection_records_all.extend(selection_records)
        predictions_all.extend(predictions)
        metrics_all.append(metric)

    registry_by_variant = {
        record["variant_identity"]: record for record in registry
    }

    excluded_test_observation_counts = defaultdict(int)

    for fold_record in fold_records_all:
        excluded_test_observation_counts[
            fold_record["variant_identity"]
        ] += fold_record.get("excluded_test_observation_count", 0)

    candidate_condition_identities = {
        candidate["candidate_identity"]: sorted(candidate["condition_identities"])
        for candidate in candidates
    }

    calibration_rows = build_condition_calibration(
        metrics_all,
        predictions_all,
        candidate_condition_identities
    )

    qualify_sensitivity_metrics(
        metrics_all,
        calibration_rows,
        candidate_condition_identities,
        sensitivity_policy
    )

    ablation_rows = build_ablation_rows(
        metrics_all,
        predictions_all,
        primary_predictions_by_benchmark,
        primary_metric_lookup,
        registry_by_variant
    )

    incident_rows = build_incident_rows(
        metrics_all,
        predictions_all,
        primary_predictions_by_benchmark,
        primary_metric_lookup,
        registry_by_variant
    )

    comparison_rows = build_sensitivity_comparisons(
        metrics_all,
        predictions_all,
        primary_predictions_by_benchmark,
        primary_metric_lookup,
        registry_by_variant,
        excluded_test_observation_counts
    )

    interval_diagnostic_rows = build_interval_diagnostics(
        metrics_all,
        predictions_all,
        sensitivity_policy,
        candidate_condition_identities
    )

    for record in metrics_all:
        record.pop("incident_eligibility", None)

    return {
        "sensitivity_registry": sorted(
            registry,
            key = lambda record: record["variant_identity"]
        ),
        "validation_folds": sorted(
            fold_records_all,
            key = lambda record: (record["variant_identity"], record["fold_index"])
        ),
        "configuration_selection": sorted(
            selection_records_all,
            key = lambda record: (
                record["variant_identity"], record["outer_fold_index"],
                record["configuration_identity"]
            )
        ),
        "validation_predictions": sorted(
            predictions_all,
            key = lambda record: (
                record["variant_identity"], record["outer_fold_index"],
                record["cell_identity"], record["source_identity"]
            )
        ),
        "validation_metrics": sorted(
            metrics_all,
            key = lambda record: record["variant_identity"]
        ),
        "sensitivity_comparisons": sorted(
            comparison_rows, key = lambda record: record["variant_identity"]
        ),
        "condition_calibration": sorted(
            calibration_rows,
            key = lambda record: (record["variant_identity"], record["condition_protocol_identity"] or "")
        ),
        "predictor_ablation": sorted(
            ablation_rows,
            key = lambda record: (
                record["candidate_identity"], record["model_family"], record["ablation_group"]
            )
        ),
        "incident_exclusions": sorted(
            incident_rows,
            key = lambda record: (
                record["candidate_identity"], record["model_family"], record["incident_reason_code"]
            )
        ),
        "prediction_interval_diagnostics": sorted(
            interval_diagnostic_rows,
            key = lambda record: (
                record["variant_identity"], record["grouping_dimension"], record["grouping_value"]
            )
        )
    }


# Building one output table manifest record
def build_table_manifest(path: Path, records: list[dict], fields: list[str]) -> dict:
    return {
        "file_name": path.name,
        "row_count": len(records),
        "sha256": calculate_file_sha256(path),
        "schema": [{"name": field_name} for field_name in fields]
    }


# Publishing the complete artifact set transactionally through a staged directory
def publish_outputs(args: argparse.Namespace, policy: dict,
                    policy_sha256: str, input_hashes: dict,
                    tables: dict, software_environment: dict
) -> None:
    staging_root = Path(tempfile.mkdtemp(
        prefix = ".predictive_sensitivity_",
        dir = args.output_directory.parent
    ))

    try:
        table_definitions = {
            "sensitivity_registry": (
                SENSITIVITY_REGISTRY_FIELDS,
                tables["sensitivity_registry"]
            ),
            "validation_folds": (
                SENSITIVITY_VALIDATION_FOLD_FIELDS,
                tables["validation_folds"]
            ),
            "configuration_selection": (
                CONFIGURATION_SELECTION_FIELDS,
                tables["configuration_selection"]
            ),
            "validation_predictions": (
                SENSITIVITY_VALIDATION_PREDICTION_FIELDS,
                tables["validation_predictions"]
            ),
            "validation_metrics": (
                SENSITIVITY_VALIDATION_METRIC_FIELDS,
                tables["validation_metrics"]
            ),
            "sensitivity_comparisons": (
                SENSITIVITY_COMPARISON_FIELDS,
                tables["sensitivity_comparisons"]
            ),
            "condition_calibration": (
                CONDITION_CALIBRATION_FIELDS,
                tables["condition_calibration"]
            ),
            "predictor_ablation": (
                PREDICTOR_ABLATION_FIELDS,
                tables["predictor_ablation"]
            ),
            "incident_exclusions": (
                INCIDENT_EXCLUSION_FIELDS,
                tables["incident_exclusions"]
            ),
            "prediction_interval_diagnostics": (
                PREDICTION_INTERVAL_DIAGNOSTIC_FIELDS,
                tables["prediction_interval_diagnostics"]
            )
        }

        table_manifests = {}

        for table_name, (fields, records) in table_definitions.items():
            table_path = staging_root / f"{table_name}.csv"
            write_csv_table(table_path, fields, records)
            table_manifests[table_name] = build_table_manifest(table_path, records, fields)

        report = {
            "execution_status": "completed",
            "operation": "fit",
            "policy_name": policy["policy_name"],
            "policy_version": policy["policy_version"],
            "policy_sha256": policy_sha256,
            "implementation_sha256": calculate_file_sha256(Path(__file__).resolve()),
            "input_hashes": input_hashes,
            "software_environment": software_environment,
            "causal_interpretation_authorized": False,
            "policy_recommendation_authorized": False,
            "performance_selected_model_winner_declared": False,
            "primary_benchmark_replacement_declared": False,
            "full_factorial_perturbation_crossing_declared": False,
            "sensitivity_variant_count": len(tables["sensitivity_registry"]),
            "sensitivity_metric_status_counts": dict(sorted(Counter(
                record["validation_status"] for record in tables["validation_metrics"]
            ).items())),
            "validation_status_counts": dict(sorted(Counter(
                record["validation_status"] for record in tables["validation_metrics"]
            ).items())),
            "analytical_status_counts": dict(sorted(Counter(
                record["analytical_status"] for record in tables["validation_metrics"]
            ).items())),
            "comparison_status_counts": dict(sorted(Counter(
                record["comparison_status"] for record in tables["sensitivity_comparisons"]
            ).items())),
            "variant_counts_by_type": dict(sorted(Counter(
                record["variant_type"] for record in tables["sensitivity_registry"]
            ).items())),
            "tables": table_manifests
        }

        report_path = staging_root / policy["output_contract"]["report_file"]
        report_path.write_text(
            json.dumps(report, indent = 2, sort_keys = True) + "\n",
            encoding = "utf-8"
        )

        backup_directory = args.output_directory.with_name(f"{args.output_directory.name}.previous")

        if backup_directory.exists():
            raise RuntimeError("A prior predictive-sensitivity backup already exists.")

        if args.output_directory.exists():
            args.output_directory.rename(backup_directory)

        staging_root.rename(args.output_directory)

        try:
            validate_published_directory(args.output_directory, policy, policy_sha256, report, tables)
        except Exception:
            if args.output_directory.exists():
                shutil.rmtree(args.output_directory, ignore_errors = True)
            if backup_directory.exists():
                backup_directory.rename(args.output_directory)
            raise

        if backup_directory.exists():
            shutil.rmtree(backup_directory)

    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise


# Confirming that every published sensitivity table internally reconciles with every other table it references.
def _validate_cross_table_invariants(tables: dict[str, list[dict]]) -> None:
    registry = tables.get("sensitivity_registry", [])
    metrics = tables.get("validation_metrics", [])
    folds = tables.get("validation_folds", [])
    selections = tables.get("configuration_selection", [])
    predictions = tables.get("validation_predictions", [])
    comparisons = tables.get("sensitivity_comparisons", [])
    intervals = tables.get("prediction_interval_diagnostics", [])
    ablations = tables.get("predictor_ablation", [])
    incidents = tables.get("incident_exclusions", [])

    registry_identities = [r["variant_identity"] for r in registry]
    if len(registry_identities) != len(set(registry_identities)):
        raise RuntimeError("Sensitivity registry contains duplicate variant identities.")

    metric_identities = [r["variant_identity"] for r in metrics]
    if len(metric_identities) != len(set(metric_identities)):
        raise RuntimeError("Sensitivity metrics contain duplicate variant identities.")

    registry_ids = set(registry_identities)
    metrics_by_id = {r["variant_identity"]: r for r in metrics}

    completed_folds_by_variant = defaultdict(set)

    for fold in folds:
        training_cells = set(fold["training_cell_identities"])
        test_cells = set(fold["test_cell_identities"])

        if training_cells.intersection(test_cells):
            raise RuntimeError(f"Validation fold contains cell leakage : {fold['variant_identity']} : {fold['fold_index']}")

        if fold["fold_status"] == "completed":
            completed_folds_by_variant[fold["variant_identity"]].add(int(fold["fold_index"]))

    for prediction in predictions:
        variant_identity = prediction["variant_identity"]

        if variant_identity not in registry_ids:
            raise RuntimeError(f"Prediction references an unregistered variant : {variant_identity}")
        fold_index = int(prediction["outer_fold_index"])

        if fold_index not in completed_folds_by_variant.get(variant_identity, set()):
            raise RuntimeError(f"Prediction references a missing or unavailable fold : {variant_identity} : {fold_index}")

    predictions_by_variant = defaultdict(list)
    for record in predictions:
        predictions_by_variant[record["variant_identity"]].append(record)

    for variant_identity, metric in metrics_by_id.items():
        if metric["validation_status"] != "completed":
            if predictions_by_variant.get(variant_identity):
                raise RuntimeError(f"Unavailable metric must not publish predictions : {variant_identity}")
            continue

        variant_predictions = predictions_by_variant.get(variant_identity, [])

        if len(variant_predictions) != int(metric["test_observation_count"]):
            raise RuntimeError(f"Metric observation count does not match its published predictions : {variant_identity}")

        if len({r["cell_identity"] for r in variant_predictions}) != int(metric["test_cell_count"]):
            raise RuntimeError(f"Metric cell count does not match its published predictions : {variant_identity}")

        if len({int(r["outer_fold_index"]) for r in variant_predictions}) != int(metric["outer_fold_count"]):
            raise RuntimeError(f"Metric fold count does not match its published predictions : {variant_identity}")

    selections_by_fold = defaultdict(list)

    for selection in selections:
        selections_by_fold[
            (selection["variant_identity"], int(selection["outer_fold_index"]))
        ].append(selection)

    for fold in folds:
        if fold["fold_status"] != "completed":
            continue
        key = (fold["variant_identity"], int(fold["fold_index"]))
        fold_selections = selections_by_fold.get(key, [])
        selected_count = sum(
            1 for s in fold_selections
            if s["selected"] is True or s["selected"] == "true"
        )
        if selected_count != 1:
            raise RuntimeError(f"Completed fold does not have exactly one selected configuration : {key[0]} : {key[1]}")

    # Exactly one comparison per registry variant
    comparisons_by_variant = defaultdict(list)
    for comparison in comparisons:
        comparisons_by_variant[comparison["variant_identity"]].append(comparison)

    for variant_identity in registry_ids:
        if len(comparisons_by_variant.get(variant_identity, [])) != 1:
            raise RuntimeError(f"Each variant must have exactly one comparison record : {variant_identity}")

    # Exactly one overall interval record per registry variant
    overall_intervals_by_variant = defaultdict(list)
    for interval in intervals:
        if interval["grouping_dimension"] == "overall":
            overall_intervals_by_variant[interval["variant_identity"]].append(interval)

    for variant_identity in registry_ids:
        if len(overall_intervals_by_variant.get(variant_identity, [])) != 1:
            raise RuntimeError(f"Each variant must have exactly one overall interval record : {variant_identity}")

    # Completed variants must also provide interval records for every fold and condition observed in their predictions
    interval_folds_by_variant = defaultdict(set)

    interval_conditions_by_variant = defaultdict(set)
    for interval in intervals:
        if interval["grouping_dimension"] == "validation_fold":
            interval_folds_by_variant[interval["variant_identity"]].add(int(interval["grouping_value"]))

        elif interval["grouping_dimension"] == "condition_protocol":
            interval_conditions_by_variant[interval["variant_identity"]].add(interval["grouping_value"])

    for variant_identity, metric in metrics_by_id.items():
        if metric["validation_status"] != "completed":
            continue
        variant_predictions = predictions_by_variant.get(variant_identity, [])

        if not variant_predictions:
            continue
        observed_folds = {int(p["outer_fold_index"]) for p in variant_predictions}

        observed_conditions = {p["condition_protocol_identity"] for p in variant_predictions}

        if not observed_folds.issubset(interval_folds_by_variant.get(variant_identity, set())):
            raise RuntimeError(f"Completed variant is missing fold-level interval records : {variant_identity}")

        if not observed_conditions.issubset(interval_conditions_by_variant.get(variant_identity, set())):
            raise RuntimeError(f"Completed variant is missing condition-level interval records : {variant_identity}")

    # Exact ablation and incident summary coverage
    ablation_variants = {
        r["variant_identity"] for r in registry if r["variant_type"] == "ablation"
    }
    ablation_summary_variants = {r["variant_identity"] for r in ablations}
    if ablation_variants != ablation_summary_variants:
        raise RuntimeError("Ablation summary coverage does not match the ablation variant inventory.")

    incident_variants = {
        r["variant_identity"] for r in registry if r["variant_type"] == "incident"
    }
    incident_summary_variants = {r["variant_identity"] for r in incidents}
    if incident_variants != incident_summary_variants:
        raise RuntimeError("Incident summary coverage does not match the incident variant inventory.")


# Re-validating a published predictive-sensitivity directory against its computed records
def validate_published_directory(published_directory: Path, policy: dict,
                                 policy_sha256: str, expected_report: dict,
                                 expected_tables: dict[str, list[dict]]
) -> None:
    report_path = published_directory / policy["output_contract"]["report_file"]

    if not report_path.is_file():
        raise RuntimeError("Published predictive-sensitivity report is absent.")

    loaded = json.loads(report_path.read_text(encoding = "utf-8"))

    if loaded != expected_report:
        raise RuntimeError("Published sensitivity report content does not match.")

    if loaded.get("policy_sha256") != policy_sha256:
        raise RuntimeError("Published predictive-sensitivity policy_sha256 is invalid.")

    if loaded.get("implementation_sha256") != calculate_file_sha256(Path(__file__).resolve()):
        raise RuntimeError("Published predictive-sensitivity implementation_sha256 is invalid.")

    if loaded.get("causal_interpretation_authorized") is not False:
        raise RuntimeError("Published predictive-sensitivity report incorrectly authorizes causal interpretation.")

    if loaded.get("primary_benchmark_replacement_declared") is not False:
        raise RuntimeError("Published predictive-sensitivity report incorrectly declares primary replacement.")

    for metric_record in expected_tables["validation_metrics"]:
        if metric_record["validation_status"] == "completed":
            if metric_record["analytical_status"] not in {"available", "diagnostic_only"}:
                raise RuntimeError(
                    f"Completed metric must carry a qualified analytical status : {metric_record['variant_identity']}"
                )
        else:
            if metric_record["analytical_status"] != "not_available":
                raise RuntimeError(
                    f"Unavailable metric must carry the not_available analytical status : {metric_record['variant_identity']}"
                )

    if set(loaded.get("tables", {})) != set(policy["output_contract"]["tables"]):
        raise RuntimeError("Published predictive-sensitivity table inventory is invalid.")

    for table_name, table_record in loaded["tables"].items():
        table_path = published_directory / table_record["file_name"]

        if calculate_file_sha256(table_path) != table_record["sha256"]:
            raise RuntimeError(f"Published predictive-sensitivity table hash mismatch : {table_name}")

        loaded_records = read_typed_csv_table(table_path)

        if len(loaded_records) != int(table_record["row_count"]):
            raise RuntimeError(f"Published predictive-sensitivity table row count mismatch : {table_name}")

        expected_records = expected_tables[table_name]

        if len(loaded_records) != len(expected_records):
            raise RuntimeError(f"Published table row count does not match expected : {table_name}")

        loaded_serialized = [
            tuple(
                str(serialize_csv_value(record[field_name]))
                for field_name in sorted(record)
            )
            for record in loaded_records
        ]
        expected_serialized = [
            tuple(
                str(serialize_csv_value(record[field_name]))
                for field_name in sorted(record)
            )
            for record in expected_records
        ]

        if loaded_serialized != expected_serialized:
            raise RuntimeError(f"Published table content does not match : {table_name}")

    metric_variants = {
        record["variant_identity"] for record in expected_tables["validation_metrics"]
    }
    registry_variants = {
        record["variant_identity"] for record in expected_tables["sensitivity_registry"]
    }

    if metric_variants != registry_variants:
        raise RuntimeError("Every registry variant must have exactly one validation metric.")

    metrics_by_variant = {
        record["variant_identity"]: record for record in expected_tables["validation_metrics"]
    }

    predictions_by_variant = defaultdict(list)
    for record in expected_tables["validation_predictions"]:
        predictions_by_variant[record["variant_identity"]].append(record)

    for variant_identity, metric_record in metrics_by_variant.items():
        if metric_record["validation_status"] != "completed":
            if predictions_by_variant.get(variant_identity):
                raise RuntimeError(f"Unavailable metric must not publish predictions : {variant_identity}")

    _validate_cross_table_invariants(
        tables = expected_tables
    )


# Running the requested predictive-sensitivity operation
def main() -> None:
    args = parse_args()

    policy = load_json_object(args.sensitivity_policy_path)
    validate_policy(policy)

    policy_sha256 = calculate_file_sha256(args.sensitivity_policy_path)
    predictive_policy = load_json_object(args.predictive_policy_path)

    validate_expanded_grid_against_primary(
        sensitivity_policy = policy,
        predictive_policy = predictive_policy
    )

    input_paths = {
        "predictive_policy": args.predictive_policy_path,
        "predictive_report": args.predictive_report_path,
        "feature_manifest": args.feature_manifest_path,
        "data_quality_contract": args.data_quality_contract_path,
        "bayesian_policy": args.bayesian_policy_path,
        "bayesian_report": args.bayesian_report_path,
        "design_report": args.design_report_path,
        "identification_report": args.identification_report_path,
        "descriptive_report": args.descriptive_report_path
    }

    input_hashes = {
        input_name: calculate_file_sha256(input_path)
        for input_name, input_path in input_paths.items()
    }

    expected_hash_fields = {
        "predictive_policy": "required_predictive_policy_sha256",
        "predictive_report": "required_predictive_report_sha256",
        "feature_manifest": "required_feature_manifest_sha256",
        "data_quality_contract": "required_data_quality_contract_sha256",
        "bayesian_policy": "required_bayesian_policy_sha256",
        "bayesian_report": "required_bayesian_report_sha256",
        "design_report": "required_design_report_sha256",
        "identification_report": "required_identification_report_sha256",
        "descriptive_report": "required_descriptive_report_sha256"
    }

    for input_name, field_name in expected_hash_fields.items():
        if input_hashes[input_name] != policy["inputs"][field_name]:
            raise RuntimeError(f"Input does not match the sensitivity registration : {input_name}")

    software_environment = validate_software_environment(predictive_policy)

    predictive_report = load_json_object(args.predictive_report_path)
    predictive_tables = verify_report_tables(
        args.predictive_report_path,
        predictive_report,
        policy["inputs"]["required_predictive_tables"]
    )

    feature_manifest = load_json_object(args.feature_manifest_path)
    feature_tables = verify_report_tables(
        args.feature_manifest_path,
        feature_manifest,
        policy["inputs"]["required_feature_tables"]
    )

    (
        validated_inputs,
        workflow,
        bayesian_report
    ) = validate_upstream_inputs(
        args,
        predictive_policy
    )

    selected_candidates = build_selected_candidates(
        workflow,
        predictive_policy
    )

    reported_candidate_identities = {
        record["candidate_identity"]
        for record in predictive_tables["candidate_registry"]
    }

    selected_candidate_identities = {
        candidate["candidate_identity"]
        for candidate in selected_candidates
    }

    if selected_candidate_identities != reported_candidate_identities:
        raise RuntimeError("Sensitivity and primary candidate inventories differ.")

    reason_lookup = build_reason_lookup(feature_tables["reference_test_features"])

    primary_folds_by_candidate = defaultdict(list)
    for fold_record in predictive_tables["validation_folds"]:
        primary_folds_by_candidate[
            fold_record["candidate_identity"]
        ].append(fold_record)

    primary_selected_lookup = build_primary_selected_lookup(
        predictive_tables["hyperparameter_selection"]
    )

    primary_metric_lookup = build_primary_metric_lookup(
        predictive_tables["validation_metrics"]
    )

    primary_predictions_by_benchmark = defaultdict(dict)
    for record in predictive_tables["validation_predictions"]:
        key = (
            record["candidate_identity"], record["model_family"]
        )
        identity = prediction_identity(record)

        if identity in primary_predictions_by_benchmark[key]:
            raise RuntimeError("Primary predictions contain a duplicate observation.")

        primary_predictions_by_benchmark[key][identity] = record

    config_inventories = build_config_inventories(predictive_policy, policy)

    fitting_policy = build_sensitivity_fitting_policy(policy, predictive_policy)

    registry = build_sensitivity_registry(selected_candidates, reason_lookup, fitting_policy)

    if args.operation == "validate":
        print("Predictive-sensitivity inputs validated.")
        print("Selected candidate count :", len(selected_candidates))
        print("Sensitivity variant count :", len(registry))
        return

    tables = run_sensitivity_fitting(
        selected_candidates,
        registry,
        primary_folds_by_candidate,
        primary_selected_lookup,
        primary_predictions_by_benchmark,
        primary_metric_lookup,
        config_inventories,
        fitting_policy,
        reason_lookup
    )

    publish_outputs(
        args,
        policy,
        policy_sha256,
        input_hashes,
        tables,
        software_environment
    )

    print("Predictive-sensitivity assessment finished.")
    print(f"Sensitivity variant count : {len(tables['sensitivity_registry'])}")
    print(f"Validation metrics : {len(tables['validation_metrics'])}")
    print(f"Report : {args.output_directory / policy['output_contract']['report_file']}")


if __name__ == "__main__":
    main()
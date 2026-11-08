import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
import datetime
import re

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_SUMMARY_FIELDS = [
    "source_analysis",
    "model_name",
    "dataset_stage",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "record_count",
    "available_count",
    "diagnostic_only_count",
    "not_available_count",
    "validation_completed_count",
    "available_validation_completed_count",
    "metric_population",
    "median_mean_absolute_error",
    "median_root_mean_squared_error",
    "median_predictive_r_squared",
    "median_interval_coverage",
    "available_median_mean_absolute_error",
    "available_median_root_mean_squared_error",
    "available_median_predictive_r_squared",
    "available_median_interval_coverage",
    "assessment_status",
    "assessment_reason_code"
]

MODEL_FAMILY_COMPARISON_FIELDS = [
    "candidate_identity",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "first_model_family",
    "second_model_family",
    "first_analytical_status",
    "second_analytical_status",
    "first_validation_status",
    "second_validation_status",
    "first_root_mean_squared_error",
    "second_root_mean_squared_error",
    "root_mean_squared_error_difference",
    "first_mean_absolute_error",
    "second_mean_absolute_error",
    "mean_absolute_error_difference",
    "first_predictive_r_squared",
    "second_predictive_r_squared",
    "predictive_r_squared_difference",
    "comparison_status",
    "comparison_reason_code",
    "estimate_label"
]

PRIOR_SENSITIVITY_FIELDS = [
    "candidate_identity",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "reference_prior_scale_multiplier",
    "sensitivity_prior_scale_multiplier",
    "comparison_record_count",
    "completed_comparison_count",
    "unavailable_comparison_count",
    "median_absolute_median_difference",
    "maximum_absolute_median_difference",
    "median_absolute_relative_difference",
    "maximum_absolute_relative_difference",
    "assessment_status",
    "assessment_reason_code",
    "estimate_label"
]

INFLUENCE_SUMMARY_FIELDS = [
    "candidate_identity",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "template_name",
    "analytical_result_status",
    "deleted_cell_count",
    "completed_deleted_cell_count",
    "failed_deleted_cell_count",
    "maximum_absolute_standardized_coefficient_change",
    "assessment_status",
    "assessment_reason_code",
    "estimate_label"
]

PREDICTION_AGREEMENT_FIELDS = [
    "candidate_identity",
    "model_family",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "bayesian_analytical_status",
    "predictive_analytical_status",
    "bayesian_prediction_count",
    "predictive_prediction_count",
    "shared_prediction_count",
    "shared_inventory_fraction_bayesian",
    "shared_inventory_fraction_predictive",
    "exact_inventory_match",
    "shared_cell_count",
    "shared_observation_count",
    "equal_cell_weighted_mean_absolute_prediction_difference",
    "equal_cell_weighted_root_mean_squared_prediction_difference",
    "equal_cell_weighted_maximum_absolute_prediction_difference",
    "equal_cell_weighted_prediction_correlation",
    "assessment_status",
    "assessment_reason_code",
    "estimate_label"
]

ASSESSMENT_COVERAGE_FIELDS = [
    "assessment_identity",
    "evidence_source",
    "evidence_unit",
    "total_evidence_count",
    "completed_evidence_count",
    "restricted_evidence_count",
    "unavailable_evidence_count",
    "assessment_status",
    "assessment_reason_code",
    "interpretation"
]

EXTERNAL_VALIDITY_FIELDS = [
    "domain_identity",
    "applicability_status",
    "boundary"
]

COHORT_SENSITIVITY_FIELDS = [
    "matching_key_identity",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "horizon_axis",
    "model_name",
    "condition_count",
    "primary_analytical_status",
    "sensitivity_analytical_status",
    "primary_validation_status",
    "sensitivity_validation_status",
    "primary_record_count",
    "sensitivity_record_count",
    "primary_validation_cell_count",
    "sensitivity_validation_cell_count",
    "primary_validation_observation_count",
    "sensitivity_validation_observation_count",
    "validation_population_count_relationship",
    "primary_root_mean_squared_error",
    "sensitivity_root_mean_squared_error",
    "root_mean_squared_error_difference",
    "primary_mean_absolute_error",
    "sensitivity_mean_absolute_error",
    "mean_absolute_error_difference",
    "comparison_status",
    "comparison_reason_code",
    "estimate_label"
]

STRUCTURAL_LIMITATION_FIELDS = [
    "limitation_identity",
    "limitation_scope",
    "evidence_source",
    "evidence_record_count",
    "applicability_status",
    "reason_code",
    "interpretation"
]

CONFIGURATION_BOUNDARY_FIELDS = [
    "model_family",
    "configuration_identity",
    "configuration",
    "selection_count",
    "total_outer_folds",
    "selection_fraction",
    "boundary_classification",
    "boundary_assessment_status",
    "selected_on_any_outer_fold",
    "boundary_parameter_names",
    "boundary_parameter_count",
    "informative_parameter_names",
    "uninformative_parameter_names",
    "on_registered_search_boundary",
    "estimate_label"
]

DESCRIPTIVE_SPECIFICATION_COMPARISON_FIELDS = [
    "comparison_identity",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "specification",
    "context_family",
    "first_candidate_identity",
    "second_candidate_identity",
    "first_template_name",
    "second_template_name",
    "first_analytical_status",
    "second_analytical_status",
    "first_mean_absolute_error",
    "second_mean_absolute_error",
    "mean_absolute_error_difference",
    "first_root_mean_squared_error",
    "second_root_mean_squared_error",
    "root_mean_squared_error_difference",
    "comparison_status",
    "comparison_reason_code",
    "estimate_label"
]

FRAGILITY_REGISTRY_FIELDS = [
    "fragility_scope_identity",
    "source_analysis",
    "candidate_or_benchmark_identity",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "validation_available",
    "analytical_status",
    "prior_sensitivity_complete",
    "cell_deletion_refit_complete",
    "condition_calibration_complete",
    "predictive_benchmark_met",
    "cross_framework_inventory_complete",
    "cross_framework_source_models_available",
    "equipment_restriction_present",
    "sensitivity_cohort",
    "applicable_indicators",
    "fragility_indicators",
    "fragility_indicator_count",
    "estimate_label"
]

UNSUPPORTED_CONDITION_FIELDS = [
    "benchmark_identity",
    "candidate_identity",
    "model_family",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "condition_protocol_identity",
    "calibration_status",
    "calibration_reason_code",
    "estimate_label"
]

PREDICTION_INTERVAL_DIAGNOSTIC_FIELDS = [
    "source_analysis",
    "candidate_or_benchmark_identity",
    "model_name",
    "dataset_stage",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "grouping_dimension",
    "grouping_value",
    "nominal_interval_probability",
    "prediction_count",
    "independent_cell_count",
    "interval_available_count",
    "interval_covered_count",
    "equal_cell_weighted_interval_coverage",
    "equal_cell_weighted_mean_interval_width",
    "assessment_status",
    "assessment_reason_code",
    "estimate_label"
]

HORIZON_EVALUATION_SENSITIVITY_FIELDS = [
    "source_analysis",
    "candidate_or_benchmark_identity",
    "model_name",
    "dataset_stage",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "evaluation_window_identity",
    "prediction_count",
    "independent_cell_count",
    "minimum_horizon",
    "maximum_horizon",
    "equal_cell_weighted_mean_absolute_error",
    "equal_cell_weighted_root_mean_squared_error",
    "equal_cell_weighted_predictive_r_squared",
    "equal_cell_weighted_interval_coverage",
    "equal_cell_weighted_mean_interval_width",
    "root_mean_squared_error_difference_from_full_inventory",
    "assessment_status",
    "assessment_reason_code",
    "estimate_label"
]

INCIDENT_EVALUATION_SENSITIVITY_FIELDS = [
    "source_analysis",
    "candidate_or_benchmark_identity",
    "model_name",
    "dataset_stage",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "incident_reason_code",
    "full_prediction_count",
    "excluded_prediction_count",
    "retained_prediction_count",
    "full_independent_cell_count",
    "retained_independent_cell_count",
    "full_equal_cell_weighted_mean_absolute_error",
    "retained_equal_cell_weighted_mean_absolute_error",
    "mean_absolute_error_difference",
    "full_equal_cell_weighted_root_mean_squared_error",
    "retained_equal_cell_weighted_root_mean_squared_error",
    "root_mean_squared_error_difference",
    "full_equal_cell_weighted_predictive_r_squared",
    "retained_equal_cell_weighted_predictive_r_squared",
    "model_refitted",
    "assessment_status",
    "assessment_reason_code",
    "estimate_label"
]

PREDICTIVE_SENSITIVITY_SUMMARY_FIELDS = [
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
    "partition_identity",
    "configuration_inventory",
    "training_window_identity",
    "incident_reason_code",
    "ablation_group",
    "validation_status",
    "validation_reason_code",
    "analytical_status",
    "analytical_reason_code",
    "comparison_status",
    "comparison_reason_code",
    "exact_inventory_match",
    "shared_observation_count",
    "shared_independent_cell_count",
    "excluded_test_observation_count",
    "root_mean_squared_error_difference",
    "mean_absolute_error_difference",
    "predictive_r_squared_difference",
    "overall_interval_status",
    "overall_interval_reason_code",
    "overall_interval_prediction_count",
    "overall_interval_independent_cell_count",
    "overall_interval_coverage",
    "overall_interval_width",
    "assessment_status",
    "assessment_reason_code",
    "estimate_label"
]


# Parsing validated input paths, policy location and output directory
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = (
            "Reconcile robustness, model stability and external-validity evidence from completed analytical artifacts."
        )
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
        "--bayesian-report-path",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--predictive-report-path",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--data-quality-contract-path",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--feature-manifest-path",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--predictive-policy-path",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--policy-path",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--output-directory",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--predictive-sensitivity-report-path",
        type = Path,
        required = True
    )

    parser.add_argument(
        "--predictive-sensitivity-policy-path",
        type = Path,
        required = True
    )

    return parser.parse_args()


# Producing deterministic canonical JSON
def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        separators = (
            ",",
            ":"
        ),
        sort_keys = True
    )


# Calculating one file SHA-256 digest
def calculate_file_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


# Rejecting duplicate JSON object fields
def reject_duplicate_json_fields(field_pairs: list[tuple[str, object]]) -> dict:
    parsed_object = {}

    for field_name, field_value in field_pairs:
        if field_name in parsed_object:
            raise RuntimeError(f"JSON object contains a duplicate field : {field_name}")

        parsed_object[
            field_name
        ] = field_value

    return parsed_object


# Loading one strict JSON object
def load_json_object(path: Path) -> dict:
    with path.open(
        mode = "r",
        encoding = "utf-8"
    ) as input_stream:
        value = json.load(
            input_stream,
            object_pairs_hook = (
                reject_duplicate_json_fields
            )
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
        try:
            return json.loads(
                value,
                object_pairs_hook = (
                    reject_duplicate_json_fields
                )
            )

        except json.JSONDecodeError as exc:
            raise RuntimeError("CSV field begins with a JSON delimiter but is not valid JSON.") from exc

    return value


# Reading one CSV table with typed values
def read_csv_table(path: Path) -> list[dict]:
    with path.open(
        mode = "r",
        encoding = "utf-8",
        newline = ""
    ) as table_stream:
        reader = csv.DictReader(
            table_stream
        )

        if reader.fieldnames is None:
            raise RuntimeError(f"CSV table does not contain a header : {path}")

        if len(
            reader.fieldnames
        ) != len(
            set(
                reader.fieldnames
            )
        ):
            raise RuntimeError(f"CSV table contains duplicate header fields : {path}")

        records = []

        for row_number, row in enumerate(
            reader,
            start = 2
        ):
            if None in row:
                raise RuntimeError(f"CSV row contains extra fields : {path} : {row_number}")

            if any(
                field_value is None
                for field_value in row.values()
            ):
                raise RuntimeError(f"CSV row contains missing fields : {path} : {row_number}")

            records.append(
                {
                    field_name: parse_csv_value(
                        field_value
                    )
                    for field_name, field_value in row.items()
                }
            )

        return records


# Confirming that one table contains unique, non-empty record identities
def validate_unique_record_identities(records: list[dict], identity_fields: tuple[str, ...], table_name: str) -> None:
    observed_identities = set()

    for row_number, record in enumerate(
        records,
        start = 2
    ):
        identity_values = []

        for field_name in identity_fields:
            if field_name not in record:
                raise RuntimeError(f"Required identity field is absent : {table_name} : {field_name}")

            field_value = record[
                field_name
            ]

            if field_value in {
                None,
                ""
            }:
                raise RuntimeError(f"Required identity field is empty : {table_name} : {field_name} : row {row_number}")

            if isinstance(
                field_value,
                (
                    dict,
                    list
                )
            ):
                identity_values.append(
                    canonical_json(
                        field_value
                    )
                )

            else:
                identity_values.append(
                    str(
                        field_value
                    )
                )

        identity = tuple(
            identity_values
        )

        if identity in observed_identities:
            raise RuntimeError(f"Table contains a duplicate record identity : {table_name} : {identity}")

        observed_identities.add(
            identity
        )


# Validating identities used by downstream lookups and reconciliations
def validate_input_identity_constraints(input_tables: dict) -> None:
    identity_contracts = [
        (
            "causal_identification",
            "estimand_identification",
            (
                "estimand_decision_identity",
            )
        ),
        (
            "descriptive_modeling",
            "candidate_registry",
            (
                "candidate_identity",
            )
        ),
        (
            "descriptive_modeling",
            "validation_metrics",
            (
                "candidate_identity",
            )
        ),
        (
            "bayesian_modeling",
            "candidate_registry",
            (
                "candidate_identity",
            )
        ),
        (
            "bayesian_modeling",
            "grouped_validation_metrics",
            (
                "candidate_identity",
            )
        ),
        (
            "bayesian_modeling",
            "prior_sensitivity_summaries",
            (
                "candidate_identity",
                "parameter_or_trajectory_identity",
                "reference_prior_scale_multiplier",
                "sensitivity_prior_scale_multiplier"
            )
        ),
        (
            "predictive_modeling",
            "candidate_registry",
            (
                "benchmark_identity",
            )
        ),
        (
            "predictive_modeling",
            "validation_metrics",
            (
                "benchmark_identity",
            )
        ),
        (
            "predictive_modeling",
            "condition_calibration",
            (
                "benchmark_identity",
                "condition_protocol_identity"
            )
        ),
        (
            "predictive_modeling",
            "hyperparameter_selection",
            (
                "benchmark_identity",
                "outer_fold_index",
                "configuration_identity"
            )
        ),
        (
            "predictive_sensitivity",
            "sensitivity_registry",
            (
                "variant_identity",
            )
        ),
        (
            "predictive_sensitivity",
            "validation_metrics",
            (
                "variant_identity",
            )
        ),
        (
            "predictive_sensitivity",
            "sensitivity_comparisons",
            (
                "variant_identity",
            )
        ),
        (
            "predictive_sensitivity",
            "prediction_interval_diagnostics",
            (
                "variant_identity",
                "grouping_dimension",
                "grouping_value"
            )
        ),
        (
            "predictive_sensitivity",
            "predictor_ablation",
            (
                "variant_identity",
            )
        ),
        (
            "predictive_sensitivity",
            "incident_exclusions",
            (
                "variant_identity",
            )
        )
    ]

    for (
        analysis_name,
        table_name,
        identity_fields
    ) in identity_contracts:
        validate_unique_record_identities(
            records = input_tables[
                analysis_name
            ][
                table_name
            ],
            identity_fields = identity_fields,
            table_name = (
                f"{analysis_name}.{table_name}"
            )
        )

    validate_unique_record_identities(
        records = input_tables[
            "predictive_modeling"
        ][
            "candidate_registry"
        ],
        identity_fields = (
            "candidate_identity",
            "model_family"
        ),
        table_name = (
            "predictive_modeling.candidate_registry_candidate_family"
        )
    )


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
            raise RuntimeError("Cannot serialize a non-finite value.")

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


# Parsing one finite float or returning None for unusable values
def finite_float(value: object) -> float | None:
    if value in {
        None,
        ""
    }:
        return None

    try:
        numerical_value = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):
        return None

    if not math.isfinite(
        numerical_value
    ):
        return None

    return numerical_value


# Calculating the median of finite values only
def finite_median(values: list[object]) -> float | None:
    finite_values = [
        numerical_value
        for value in values
        if (
            numerical_value := finite_float(
                value
            )
        )
        is not None
    ]

    if not finite_values:
        return None

    return float(
        median(
            finite_values
        )
    )


# Validating the registered robustness-assessment policy
def validate_policy(policy: dict) -> None:
    required_fields = {
        "policy_name",
        "policy_version",
        "effective_date",
        "previous_policy_sha256",
        "registration_status",
        "scientific_scope",
        "inputs",
        "assessment_contract",
        "assessment_domains",
        "external_validity_domains",
        "execution",
        "output_contract",
        "amendment"
    }

    if set(policy) != required_fields:
        raise RuntimeError("Robustness policy has an unexpected schema.")

    if policy["policy_name"] != "robustness_assessment_policy":
        raise RuntimeError("Robustness policy name is invalid.")

    version_pattern = re.compile(r"^[1-9]\d*\.\d+$")

    if not isinstance(policy["policy_version"], str) or not version_pattern.fullmatch(policy["policy_version"]):
        raise RuntimeError("Robustness policy version is not a semantic version.")

    try:
        datetime.date.fromisoformat(policy["effective_date"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Robustness policy effective_date is not an ISO date.") from exc

    previous_sha = policy["previous_policy_sha256"]

    if not isinstance(previous_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", previous_sha):
        raise RuntimeError("Robustness policy previous_policy_sha256 is not a 64-character lowercase hex digest.")

    registration_status = policy["registration_status"]

    if registration_status not in {
        "prospective_for_robustness_classification",
        "prospective_core_with_result_informed_evidence_accounting_amendment",
        "result_informed_after_evidence_review"
    }:
        raise RuntimeError("Robustness policy registration_status is invalid.")

    amendment = policy["amendment"]

    required_amendment_fields = {
        "reason",
        "result_informed",
        "result_informed_scope",
        "affected_analyses",
        "complete_rerun_required"
    }

    if not isinstance(amendment, dict) or set(amendment) != required_amendment_fields:
        raise RuntimeError("Robustness policy amendment schema is invalid.")

    if not isinstance(amendment["reason"], str) or not amendment["reason"].strip():
        raise RuntimeError("Robustness policy amendment reason is empty.")

    if not isinstance(amendment["result_informed"], bool):
        raise RuntimeError("Robustness policy amendment result_informed is not Boolean.")

    if not isinstance(amendment["result_informed_scope"], str):
        raise RuntimeError("Robustness policy amendment result_informed_scope is not a string.")

    if amendment["result_informed"] and not amendment["result_informed_scope"].strip():
        raise RuntimeError("Result-informed amendment requires a non-empty result_informed_scope.")

    if not isinstance(amendment["affected_analyses"], list) or not amendment["affected_analyses"]:
        raise RuntimeError("Robustness policy amendment affected_analyses is invalid.")

    if len(amendment["affected_analyses"]) != len(set(amendment["affected_analyses"])):
        raise RuntimeError("Robustness policy amendment affected_analyses contains duplicates.")

    if amendment["complete_rerun_required"] is not True:
        raise RuntimeError("Robustness policy amendment must require a complete rerun.")

    if registration_status == "prospective_for_robustness_classification" and amendment["result_informed"] is not False:
        raise RuntimeError("Prospective registration_status cannot accompany a result-informed amendment.")

    if (
        registration_status
        in {
            "prospective_core_with_result_informed_evidence_accounting_amendment",
            "result_informed_after_evidence_review"
        }
        and amendment["result_informed"] is not True
    ):
        raise RuntimeError("Result-informed registration requires amendment.result_informed = True.")

    scientific_scope = policy["scientific_scope"]

    prohibited_authorizations = [
        "causal_interpretation_authorized",
        "policy_recommendation_authorized",
        "treatment_optimization_authorized",
        "novel_condition_prediction_authorized",
        "cross_stage_pooling_authorized",
        "outcome_magnitude_used_for_record_selection",
        "statistical_significance_used_for_classification",
        "performance_selected_model_winner_authorized"
    ]

    for field_name in prohibited_authorizations:
        if scientific_scope.get(field_name) is not False:
            raise RuntimeError("Robustness policy authorizes an unsupported interpretation.")

    inputs = policy["inputs"]

    required_file_sha = inputs.get("required_file_sha256")

    if not isinstance(required_file_sha, dict) or set(required_file_sha) != {
        "data_quality_contract",
        "feature_manifest",
        "predictive_modeling_policy",
        "predictive_sensitivity_policy"
    }:
        raise RuntimeError("Robustness policy required_file_sha256 schema is invalid.")

    required_feature_tables = inputs.get("required_feature_tables")

    if (
        not isinstance(required_feature_tables, list)
        or required_feature_tables != ["reference_test_features"]
    ):
        raise RuntimeError("Robustness policy required_feature_tables is invalid.")

    for name, digest in required_file_sha.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError(f"Robustness policy required_file_sha256 digest is invalid : {name}")

    if inputs.get("table_hash_verification_required") is not True:
        raise RuntimeError("Robustness policy table_hash_verification_required must be True.")

    if inputs.get("schema_verification_required") is not True:
        raise RuntimeError("Robustness policy schema_verification_required must be True.")

    assessment_contract = policy["assessment_contract"]

    held_out_contract = assessment_contract.get(
        "held_out_evaluation"
    )

    if held_out_contract != {
        "minimum_independent_cell_count": 2,
        "prediction_inventory_rule": "completed_out_of_fold_predictions_only",
        "metric_weighting": "each_cell_has_equal_total_weight",
        "predictive_r_squared_reference": "equal_cell_weighted_observed_outcome_mean",
        "training_data_or_fitted_model_changes_permitted": False
    }:
        raise RuntimeError("Robustness held-out evaluation contract is invalid.")

    expected_windows = {
        "full_evaluation_inventory",
        "first_follow_up_excluded",
        "terminal_observation_excluded"
    }

    observed_windows = {
        record.get("window_identity")
        for record in assessment_contract.get("evaluation_windows", [])
    }

    if observed_windows != expected_windows:
        raise RuntimeError("Robustness evaluation-window inventory is invalid.")

    if assessment_contract.get("prediction_interval_groupings") != [
        "overall",
        "validation_fold",
        "condition_protocol"
    ]:
        raise RuntimeError("Robustness prediction-interval groupings are invalid.")

    incident_contract = assessment_contract.get(
        "incident_evaluation"
    )

    if (
        not isinstance(incident_contract, dict)
        or incident_contract.get("join_key") != ["source_identity", "cell_identity"]
        or incident_contract.get("model_refitting_permitted") is not False
    ):
        raise RuntimeError("Robustness incident-evaluation contract is invalid.")

    sensitivity_contract = assessment_contract.get(
        "predictive_sensitivity_evaluation"
    )

    expected_sensitivity_contract = {
        "required_variant_types": [
            "partition",
            "expanded_config",
            "training_window",
            "incident",
            "ablation"
        ],
        "one_factor_at_a_time_required": True,
        "analytical_qualification_required": True,
        "primary_comparison_required": True,
        "causal_interpretation_authorized": False,
        "primary_benchmark_replacement_authorized": False
    }

    if sensitivity_contract != expected_sensitivity_contract:
        raise RuntimeError("Robustness predictive-sensitivity contract is invalid.")

    evidence_count_semantics = assessment_contract.get(
        "evidence_count_semantics"
    )

    required_evidence_semantics = {
        "total_evidence_count",
        "completed_evidence_count",
        "restricted_evidence_count",
        "unavailable_evidence_count",
        "evidence_unit"
    }

    if (
        not isinstance(
            evidence_count_semantics,
            dict
        )
        or set(
            evidence_count_semantics
        )
        != required_evidence_semantics
    ):
        raise RuntimeError("Robustness assessment evidence-count semantics are invalid.")

    for field_name, field_definition in (
        evidence_count_semantics.items()
    ):
        if (
            not isinstance(
                field_definition,
                str
            )
            or not field_definition.strip()
        ):
            raise RuntimeError(f"Robustness evidence-count definition is empty : {field_name}")

    if assessment_contract.get("cell_equal_weighting_required") is not True:
        raise RuntimeError("Robustness assessment contract is invalid.")

    if assessment_contract.get("unmatched_prediction_inventory_accounting_required") is not True:
        raise RuntimeError("Robustness assessment contract is invalid.")

    if assessment_contract.get("magnitude_dependent_robustness_label_authorized") is not False:
        raise RuntimeError("Robustness assessment contract is invalid.")

    if assessment_contract.get("unavailable_evidence_must_remain_explicit") is not True:
        raise RuntimeError("Robustness assessment contract is invalid.")

    terminal_statuses = assessment_contract.get("terminal_domain_statuses")

    if not isinstance(terminal_statuses, list) or not terminal_statuses:
        raise RuntimeError("Robustness assessment contract terminal_domain_statuses is invalid.")

    if len(terminal_statuses) != len(set(terminal_statuses)):
        raise RuntimeError("Robustness assessment contract terminal_domain_statuses contains duplicates.")

    matching_fields = assessment_contract.get("cohort_sensitivity_matching_fields")

    if not isinstance(matching_fields, list) or not matching_fields:
        raise RuntimeError("Robustness assessment contract cohort_sensitivity_matching_fields is invalid.")

    if len(matching_fields) != len(set(matching_fields)):
        raise RuntimeError("Robustness assessment contract cohort_sensitivity_matching_fields contains duplicates.")

    status_vocabularies = assessment_contract.get("registered_status_vocabularies")

    if not isinstance(status_vocabularies, dict) or not status_vocabularies:
        raise RuntimeError("Robustness assessment contract registered_status_vocabularies is invalid.")

    for status_name, allowed_values in status_vocabularies.items():
        if (
            not isinstance(allowed_values, list)
            or not allowed_values
            or any(not isinstance(v, str) or not v for v in allowed_values)
            or len(allowed_values) != len(set(allowed_values))
        ):
            raise RuntimeError(f"Robustness assessment contract status vocabulary is invalid : {status_name}")

    if (
        "boundary_assessment_status" not in status_vocabularies
        or "boundary_classification" not in status_vocabularies
    ):
        raise RuntimeError(
            "Robustness assessment contract registered_status_vocabularies must include "
            "boundary_assessment_status and boundary_classification."
        )

    reason_codes = assessment_contract.get("registered_reason_codes")

    if (
        not isinstance(reason_codes, list)
        or not reason_codes
        or any(not isinstance(v, str) or not v for v in reason_codes)
        or len(reason_codes) != len(set(reason_codes))
    ):
        raise RuntimeError("Robustness assessment contract registered_reason_codes is invalid.")

    domain_identities = [d["assessment_identity"] for d in policy["assessment_domains"]]

    if len(domain_identities) != len(set(domain_identities)):
        raise RuntimeError("Robustness policy assessment_domains contains duplicate identities.")

    for domain in policy["assessment_domains"]:
        required_domain_fields = {
            "assessment_identity",
            "evidence_source",
            "evidence_unit",
            "evaluation_rule",
            "interpretation"
        }

        if set(domain) != required_domain_fields:
            raise RuntimeError("Robustness policy assessment_domain schema is invalid.")

        for field_name in required_domain_fields:
            field_value = domain[
                field_name
            ]

            if (
                not isinstance(
                    field_value,
                    str
                )
                or not field_value.strip()
            ):
                raise RuntimeError(
                    f"Robustness assessment domain field is empty : {domain.get('assessment_identity')} : {field_name}"
                )

    external_validity_identities = [d["domain_identity"] for d in policy["external_validity_domains"]]

    if len(external_validity_identities) != len(set(external_validity_identities)):
        raise RuntimeError("Robustness policy external_validity_domains contains duplicate identities.")

    for external_domain in policy["external_validity_domains"]:
        if set(external_domain) != {"domain_identity", "applicability_status", "boundary"}:
            raise RuntimeError("Robustness policy external_validity_domain schema is invalid.")

    if policy["execution"] != {
        "byte_identical_artifacts_required": True,
        "transactional_publication_required": True
    }:
        raise RuntimeError("Robustness execution contract is invalid.")

    output_contract = policy["output_contract"]
    declared_tables = output_contract.get("tables")

    if not isinstance(declared_tables, list) or not declared_tables:
        raise RuntimeError("Robustness output contract tables is invalid.")

    expected_tables = {
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

    if set(declared_tables) != expected_tables:
        raise RuntimeError("Robustness output contract tables inventory is invalid.")

    if output_contract.get("schemas_required") is not True:
        raise RuntimeError("Robustness output contract must require schemas.")

    if (
        output_contract.get("input_hashes_required") is not True
        or output_contract.get("implementation_hash_required") is not True
        or output_contract.get("table_hashes_required") is not True
        or output_contract.get("row_counts_required") is not True
    ):
        raise RuntimeError(
            "Robustness output contract must require input hashes, implementation hash, table hashes and row counts."
        )

    report_file = output_contract.get("report_file")

    if (
        not isinstance(report_file, str)
        or not report_file
        or not report_file.endswith(".json")
    ):
        raise RuntimeError("Robustness output contract report_file is invalid.")

    required_report_sha = inputs.get("required_report_sha256")

    if not isinstance(required_report_sha, dict):
        raise RuntimeError("Robustness policy required_report_sha256 is invalid.")

    required_report_keys = {
        "experimental_design",
        "causal_identification",
        "descriptive_modeling",
        "bayesian_modeling",
        "predictive_modeling",
        "predictive_sensitivity"
    }

    if set(required_report_sha) != required_report_keys:
        raise RuntimeError("Robustness policy required_report_sha256 key inventory is invalid.")

    for name, digest in required_report_sha.items():
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise RuntimeError(f"Robustness policy required_report_sha256 digest is invalid : {name}")

    required_tables = inputs.get("required_tables")

    if not isinstance(required_tables, dict):
        raise RuntimeError("Robustness policy required_tables is invalid.")

    if set(required_tables) != required_report_keys:
        raise RuntimeError("Robustness policy required_tables key inventory is invalid.")

    for input_name, table_names in required_tables.items():
        if (
            not isinstance(table_names, list)
            or not table_names
            or any(not isinstance(name, str) or not name for name in table_names)
            or len(table_names) != len(set(table_names))
        ):
            raise RuntimeError(f"Robustness policy required_tables entry is invalid : {input_name}")

    for domain in policy["assessment_domains"]:
        for field_name in (
            "assessment_identity",
            "evidence_source",
            "evidence_unit",
            "evaluation_rule",
            "interpretation"
        ):
            field_value = domain.get(field_name)

            if not isinstance(field_value, str) or not field_value.strip():
                raise RuntimeError(
                    f"Robustness assessment domain field is empty : {domain.get('assessment_identity')} : {field_name}"
                )

    allowed_external_statuses = {
        "supported_with_restrictions",
        "unsupported",
        "not_authorized"
    }

    for external_domain in policy["external_validity_domains"]:
        if external_domain.get("applicability_status") not in allowed_external_statuses:
            raise RuntimeError(
                "Robustness external-validity domain has an unsupported applicability_status : "
                f"{external_domain.get('domain_identity')} : {external_domain.get('applicability_status')}"
            )

    if amendment.get("affected_analyses") != ["robustness_assessment"]:
        raise RuntimeError(
            "Robustness amendment affected_analyses must be exactly ['robustness_assessment'] unless "
            "a broader amendment is registered."
        )


# Verifying input report hashes and required table manifests
def verify_report_tables(report_path: Path, expected_sha256: str,
                         required_tables: list[str]
) -> tuple[dict, dict[str, list[dict]]]:
    observed_sha256 = calculate_file_sha256(report_path)

    if observed_sha256 != expected_sha256:
        raise RuntimeError(f"Input report hash does not match the policy : {report_path}")

    report = load_json_object(report_path)
    table_manifests = report.get("tables", {})
    tables = {}

    for table_name in required_tables:
        if table_name not in table_manifests:
            raise RuntimeError(f"Required input table is absent : {table_name}")

        table_manifest = table_manifests[table_name]
        table_path = report_path.parent / table_manifest["file_name"]

        if not table_path.is_file():
            raise RuntimeError(f"Required input table is unavailable : {table_path}")

        if calculate_file_sha256(table_path) != table_manifest["sha256"]:
            raise RuntimeError(f"Input table hash does not match its report : {table_name}")

        declared_fields = [field["name"] for field in table_manifest.get("schema", [])]

        if len(
            declared_fields
        ) != len(
            set(
                declared_fields
            )
        ):
            raise RuntimeError(f"Input table manifest contains duplicate schema fields : {table_name}")

        with table_path.open(mode = "r", encoding = "utf-8", newline = "") as table_stream:
            observed_fields = csv.DictReader(table_stream).fieldnames

        if observed_fields != declared_fields:
            raise RuntimeError(f"Input table schema does not match its report : {table_name}")

        rows = read_csv_table(table_path)

        if len(rows) != int(table_manifest["row_count"]):
            raise RuntimeError(f"Input table row count does not match its report : {table_name}")

        tables[table_name] = rows

    return (report, tables)


# Extracting the dataset stage from a design scope
def dataset_stage(design_scope: object) -> str:
    normalized = str(
        design_scope
        or ""
    )

    return normalized.split(
        "/",
        maxsplit = 1
    )[0]


# Normalizing an upstream analytical status to the local vocabulary
def normalize_status(status: object) -> str:
    normalized = str(
        status
        or ""
    )

    if normalized == "available":
        return "available"

    if normalized in {
        "diagnostic_only",
        "fitted"
    }:
        return "diagnostic_only"

    return "not_available"


# Building the cross-analysis validation summary
def build_validation_summary(descriptive_tables: dict,
                             bayesian_tables: dict,
                             predictive_tables: dict
) -> list[dict]:
    normalized_records = []

    descriptive_registry = {
        record[
            "candidate_identity"
        ]: record
        for record in descriptive_tables[
            "candidate_registry"
        ]
    }

    for metric in descriptive_tables[
        "validation_metrics"
    ]:
        candidate = descriptive_registry[
            metric[
                "candidate_identity"
            ]
        ]

        normalized_records.append(
            {
                "source_analysis": "descriptive_modeling",
                "model_name": candidate[
                    "template_name"
                ],
                "dataset_stage": dataset_stage(
                    candidate[
                        "design_scope"
                    ]
                ),
                "design_scope": candidate[
                    "design_scope"
                ],
                "sampling_design_scope": candidate[
                    "sampling_design_scope"
                ],
                "aging_mode": candidate[
                    "aging_mode"
                ],
                "outcome": candidate[
                    "outcome"
                ],
                "cohort": candidate[
                    "cohort"
                ],
                "horizon_axis": candidate[
                    "horizon_axis"
                ],
                "identity": candidate[
                    "candidate_identity"
                ],
                "analytical_status": normalize_status(
                    candidate[
                        "analytical_result_status"
                    ]
                ),
                "validation_status": metric[
                    "validation_status"
                ],
                "mean_absolute_error": metric[
                    "mean_absolute_error"
                ],
                "root_mean_squared_error": metric[
                    "root_mean_squared_error"
                ],
                "predictive_r_squared": metric[
                    "predictive_r_squared"
                ],
                "interval_coverage": metric[
                    "prediction_interval_coverage"
                ]
            }
        )

    bayesian_registry = {
        record[
            "candidate_identity"
        ]: record
        for record in bayesian_tables[
            "candidate_registry"
        ]
    }

    for metric in bayesian_tables[
        "grouped_validation_metrics"
    ]:
        candidate = bayesian_registry[
            metric[
                "candidate_identity"
            ]
        ]

        normalized_records.append(
            {
                "source_analysis": "bayesian_modeling",
                "model_name": "hierarchical_bayesian_trajectory",
                "dataset_stage": dataset_stage(
                    candidate[
                        "design_scope"
                    ]
                ),
                "design_scope": candidate[
                    "design_scope"
                ],
                "sampling_design_scope": candidate[
                    "sampling_design_scope"
                ],
                "aging_mode": candidate[
                    "aging_mode"
                ],
                "outcome": candidate[
                    "outcome"
                ],
                "cohort": candidate[
                    "cohort"
                ],
                "horizon_axis": candidate[
                    "horizon_axis"
                ],
                "identity": candidate[
                    "candidate_identity"
                ],
                "analytical_status": normalize_status(
                    candidate[
                        "posterior_analytical_status"
                    ]
                ),
                "validation_status": metric[
                    "validation_status"
                ],
                "mean_absolute_error": metric[
                    "equal_cell_weighted_mean_absolute_error"
                ],
                "root_mean_squared_error": metric[
                    "equal_cell_weighted_root_mean_squared_error"
                ],
                "predictive_r_squared": metric[
                    "equal_cell_weighted_predictive_r_squared"
                ],
                "interval_coverage": metric[
                    "posterior_predictive_interval_coverage"
                ]
            }
        )

    predictive_registry = {
        record[
            "benchmark_identity"
        ]: record
        for record in predictive_tables[
            "candidate_registry"
        ]
    }

    for metric in predictive_tables[
        "validation_metrics"
    ]:
        candidate = predictive_registry[
            metric[
                "benchmark_identity"
            ]
        ]

        normalized_records.append(
            {
                "source_analysis": "predictive_modeling",
                "model_name": candidate[
                    "model_family"
                ],
                "dataset_stage": dataset_stage(
                    candidate[
                        "design_scope"
                    ]
                ),
                "design_scope": candidate[
                    "design_scope"
                ],
                "sampling_design_scope": candidate[
                    "sampling_design_scope"
                ],
                "aging_mode": candidate[
                    "aging_mode"
                ],
                "outcome": candidate[
                    "outcome"
                ],
                "cohort": candidate[
                    "cohort"
                ],
                "horizon_axis": candidate[
                    "horizon_axis"
                ],
                "identity": candidate[
                    "benchmark_identity"
                ],
                "analytical_status": normalize_status(
                    candidate[
                        "analytical_status"
                    ]
                ),
                "validation_status": metric[
                    "validation_status"
                ],
                "mean_absolute_error": metric[
                    "equal_cell_weighted_mean_absolute_error"
                ],
                "root_mean_squared_error": metric[
                    "equal_cell_weighted_root_mean_squared_error"
                ],
                "predictive_r_squared": metric[
                    "equal_cell_weighted_predictive_r_squared"
                ],
                "interval_coverage": metric[
                    "prediction_interval_coverage"
                ]
            }
        )

    grouping_fields = [
        "source_analysis",
        "model_name",
        "dataset_stage",
        "design_scope",
        "sampling_design_scope",
        "aging_mode",
        "outcome",
        "cohort",
        "horizon_axis"
    ]

    grouped_records = defaultdict(
        list
    )

    for record in normalized_records:
        grouped_records[
            tuple(
                str(
                    record[
                        field_name
                    ]
                    or ""
                )
                for field_name in grouping_fields
            )
        ].append(
            record
        )

    summaries = []

    for grouping_key, records in sorted(
        grouped_records.items()
    ):
        grouping_values = dict(
            zip(
                grouping_fields,
                grouping_key,
                strict = True
            )
        )

        status_counts = Counter(
            record[
                "analytical_status"
            ]
            for record in records
        )

        completed_records = [
            record
            for record in records
            if record[
                "validation_status"
            ] == "completed"
        ]

        completed_count = len(
            completed_records
        )

        available_completed_records = [
            record
            for record in completed_records
            if record["analytical_status"] == "available"
        ]

        total_count = len(
            records
        )

        available_count = status_counts[
            "available"
        ]

        all_validation_completed = (
            completed_count == total_count
            and total_count > 0
        )

        if completed_count == 0:
            assessment_status = "unavailable"
            assessment_reason_code = "VALIDATION_METRICS_UNAVAILABLE"

        elif available_count == 0:
            assessment_status = "diagnostic_only"
            assessment_reason_code = (
                "NO_ANALYTICALLY_AVAILABLE_VALIDATION_RECORDS"
            )

        elif (
            not all_validation_completed
            or available_count != total_count
        ):
            assessment_status = "completed_with_restrictions"
            assessment_reason_code = "PARTIAL_VALIDATION_COVERAGE"

        else:
            assessment_status = "completed"
            assessment_reason_code = None

        summaries.append(
            {
                **grouping_values,
                "record_count": total_count,
                "available_count": available_count,
                "diagnostic_only_count": status_counts["diagnostic_only"],
                "not_available_count": status_counts["not_available"],
                "validation_completed_count": completed_count,
                "available_validation_completed_count": len(
                    available_completed_records
                ),
                "metric_population": "all_validation_completed_records",
                "median_mean_absolute_error": finite_median(
                    [record["mean_absolute_error"] for record in completed_records]
                ),
                "median_root_mean_squared_error": finite_median(
                    [record["root_mean_squared_error"] for record in completed_records]
                ),
                "median_predictive_r_squared": finite_median(
                    [record["predictive_r_squared"] for record in completed_records]
                ),
                "median_interval_coverage": finite_median(
                    [record["interval_coverage"] for record in completed_records]
                ),
                "available_median_mean_absolute_error": finite_median(
                    [record["mean_absolute_error"] for record in available_completed_records]
                ),
                "available_median_root_mean_squared_error": finite_median(
                    [record["root_mean_squared_error"] for record in available_completed_records]
                ),
                "available_median_predictive_r_squared": finite_median(
                    [record["predictive_r_squared"] for record in available_completed_records]
                ),
                "available_median_interval_coverage": finite_median(
                    [record["interval_coverage"] for record in available_completed_records]
                ),
                "assessment_status": assessment_status,
                "assessment_reason_code": assessment_reason_code
            }
        )

    return summaries


# Building shared-candidate model-family comparisons with analytical-status propagation
def build_model_family_comparisons(predictive_tables: dict) -> list[dict]:
    registry = {
        record["benchmark_identity"]: record
        for record in predictive_tables["candidate_registry"]
    }

    metrics_by_candidate = defaultdict(dict)

    for metric in predictive_tables["validation_metrics"]:
        candidate = registry[metric["benchmark_identity"]]
        metrics_by_candidate[candidate["candidate_identity"]][candidate["model_family"]] = (
            candidate, metric
        )

    records = []

    for candidate_identity, family_records in sorted(metrics_by_candidate.items()):
        for first_family, second_family in combinations(sorted(family_records), 2):
            first_candidate, first_metric = family_records[first_family]
            second_candidate, second_metric = family_records[second_family]

            first_rmse = finite_float(first_metric["equal_cell_weighted_root_mean_squared_error"])
            second_rmse = finite_float(second_metric["equal_cell_weighted_root_mean_squared_error"])
            first_mae = finite_float(first_metric["equal_cell_weighted_mean_absolute_error"])
            second_mae = finite_float(second_metric["equal_cell_weighted_mean_absolute_error"])
            first_r_squared = finite_float(first_metric["equal_cell_weighted_predictive_r_squared"])
            second_r_squared = finite_float(second_metric["equal_cell_weighted_predictive_r_squared"])

            metrics_finite = all(
                value is not None
                for value in (
                    first_rmse, second_rmse,
                    first_mae, second_mae,
                    first_r_squared, second_r_squared
                )
            )

            first_validation_status = first_metric[
                "validation_status"
            ]

            second_validation_status = second_metric[
                "validation_status"
            ]

            both_validated = (
                first_validation_status == "completed"
                and second_validation_status == "completed"
            )

            both_available = (
                first_candidate["analytical_status"] == "available"
                and second_candidate["analytical_status"] == "available"
            )

            if not both_validated:
                comparison_status = "unavailable"
                comparison_reason_code = "VALIDATION_METRICS_UNAVAILABLE"
            elif not metrics_finite:
                comparison_status = "unavailable"
                comparison_reason_code = "VALIDATION_METRICS_UNAVAILABLE"
            elif not both_available:
                comparison_status = "diagnostic_only"
                comparison_reason_code = "SOURCE_BENCHMARK_NOT_ANALYTICALLY_AVAILABLE"
            else:
                comparison_status = "completed"
                comparison_reason_code = None

            records.append({
                "candidate_identity": candidate_identity,
                "design_scope": first_candidate["design_scope"],
                "sampling_design_scope": first_candidate["sampling_design_scope"],
                "aging_mode": first_candidate["aging_mode"],
                "outcome": first_candidate["outcome"],
                "cohort": first_candidate["cohort"],
                "horizon_axis": first_candidate["horizon_axis"],
                "first_model_family": first_family,
                "second_model_family": second_family,
                "first_analytical_status": first_candidate["analytical_status"],
                "second_analytical_status": second_candidate["analytical_status"],
                "first_validation_status": first_validation_status,
                "second_validation_status": second_validation_status,
                "first_root_mean_squared_error": first_rmse,
                "second_root_mean_squared_error": second_rmse,
                "root_mean_squared_error_difference": (
                    None if not metrics_finite else second_rmse - first_rmse
                ),
                "first_mean_absolute_error": first_mae,
                "second_mean_absolute_error": second_mae,
                "mean_absolute_error_difference": (
                    None if not metrics_finite else second_mae - first_mae
                ),
                "first_predictive_r_squared": first_r_squared,
                "second_predictive_r_squared": second_r_squared,
                "predictive_r_squared_difference": (
                    None if not metrics_finite else second_r_squared - first_r_squared
                ),
                "comparison_status": comparison_status,
                "comparison_reason_code": comparison_reason_code,
                "estimate_label": "predictive_robustness"
            })

    return records


# Aggregating prior-scale sensitivity comparisons per candidate and multiplier pair
def build_prior_sensitivity(bayesian_tables: dict) -> list[dict]:
    registry = {
        record[
            "candidate_identity"
        ]: record
        for record in bayesian_tables[
            "candidate_registry"
        ]
    }

    grouped_records = defaultdict(
        list
    )

    for record in bayesian_tables[
        "prior_sensitivity_summaries"
    ]:
        grouped_records[
            (
                record[
                    "candidate_identity"
                ],
                str(
                    record[
                        "reference_prior_scale_multiplier"
                    ]
                ),
                str(
                    record[
                        "sensitivity_prior_scale_multiplier"
                    ]
                )
            )
        ].append(
            record
        )

    summaries = []

    for (
        candidate_identity,
        reference_multiplier,
        sensitivity_multiplier
    ), records in sorted(
        grouped_records.items()
    ):
        candidate = registry[
            candidate_identity
        ]

        completed_records = [
            record
            for record in records
            if str(
                record[
                    "sensitivity_comparison_status"
                ]
            ).startswith(
                "completed"
            )
        ]

        absolute_differences = [
            value
            for record in completed_records
            if (
                value := finite_float(
                    record[
                        "absolute_median_difference"
                    ]
                )
            )
            is not None
        ]

        relative_differences = [
            abs(
                value
            )
            for record in completed_records
            if (
                value := finite_float(
                    record[
                        "relative_median_difference"
                    ]
                )
            )
            is not None
        ]

        assessment_complete = (
            len(
                completed_records
            )
            == len(
                records
            )
            and bool(
                completed_records
            )
        )

        summaries.append(
            {
                "candidate_identity": candidate_identity,
                "design_scope": candidate[
                    "design_scope"
                ],
                "sampling_design_scope": candidate[
                    "sampling_design_scope"
                ],
                "aging_mode": candidate[
                    "aging_mode"
                ],
                "outcome": candidate[
                    "outcome"
                ],
                "cohort": candidate[
                    "cohort"
                ],
                "horizon_axis": candidate[
                    "horizon_axis"
                ],
                "reference_prior_scale_multiplier": (
                    reference_multiplier
                ),
                "sensitivity_prior_scale_multiplier": (
                    sensitivity_multiplier
                ),
                "comparison_record_count": len(
                    records
                ),
                "completed_comparison_count": len(
                    completed_records
                ),
                "unavailable_comparison_count": (
                    len(
                        records
                    )
                    - len(
                        completed_records
                    )
                ),
                "median_absolute_median_difference": (
                    None
                    if not absolute_differences
                    else float(
                        median(
                            absolute_differences
                        )
                    )
                ),
                "maximum_absolute_median_difference": (
                    None
                    if not absolute_differences
                    else max(
                        absolute_differences
                    )
                ),
                "median_absolute_relative_difference": (
                    None
                    if not relative_differences
                    else float(
                        median(
                            relative_differences
                        )
                    )
                ),
                "maximum_absolute_relative_difference": (
                    None
                    if not relative_differences
                    else max(
                        relative_differences
                    )
                ),
                "assessment_status": (
                    "completed"
                    if assessment_complete
                    else "incomplete"
                ),
                "assessment_reason_code": (
                    None
                    if assessment_complete
                    else "PRIOR_SENSITIVITY_COMPARISON_INCOMPLETE"
                ),
                "estimate_label": "associational_sensitivity"
            }
        )

    return summaries


# Aggregating delete-one-cell influence diagnostics per candidate
def build_influence_summary(descriptive_tables: dict) -> list[dict]:
    registry = {
        record[
            "candidate_identity"
        ]: record
        for record in descriptive_tables[
            "candidate_registry"
        ]
    }

    grouped_records = defaultdict(
        list
    )

    for record in descriptive_tables[
        "influence_diagnostics"
    ]:
        grouped_records[
            record[
                "candidate_identity"
            ]
        ].append(
            record
        )

    summaries = []

    for candidate_identity, records in sorted(
        grouped_records.items()
    ):
        candidate = registry[
            candidate_identity
        ]

        cells = {
            record[
                "deleted_cell_identity"
            ]
            for record in records
        }

        failed_cells = {
            record[
                "deleted_cell_identity"
            ]
            for record in records
            if record[
                "fit_status"
            ] != "completed"
        }

        standardized_changes = [
            abs(
                value
            )
            for record in records
            if (
                value := finite_float(
                    record[
                        "standardized_coefficient_change"
                    ]
                )
            )
            is not None
        ]

        summaries.append(
            {
                "candidate_identity": candidate_identity,
                "design_scope": candidate[
                    "design_scope"
                ],
                "sampling_design_scope": candidate[
                    "sampling_design_scope"
                ],
                "aging_mode": candidate[
                    "aging_mode"
                ],
                "outcome": candidate[
                    "outcome"
                ],
                "cohort": candidate[
                    "cohort"
                ],
                "horizon_axis": candidate[
                    "horizon_axis"
                ],
                "template_name": candidate[
                    "template_name"
                ],
                "analytical_result_status": candidate[
                    "analytical_result_status"
                ],
                "deleted_cell_count": len(
                    cells
                ),
                "completed_deleted_cell_count": (
                    len(
                        cells
                    )
                    - len(
                        failed_cells
                    )
                ),
                "failed_deleted_cell_count": len(
                    failed_cells
                ),
                "maximum_absolute_standardized_coefficient_change": (
                    None
                    if not standardized_changes
                    else max(
                        standardized_changes
                    )
                ),
                "assessment_status": (
                    "completed"
                    if cells and not failed_cells
                    else "incomplete"
                ),
                "assessment_reason_code": (
                    None
                    if cells and not failed_cells
                    else "CELL_DELETION_REFIT_INCOMPLETE"
                ),
                "estimate_label": "associational_influence"
            }
        )

    return summaries


# Building one deterministic prediction identity
def prediction_identity(record: dict) -> tuple:
    return (
        record[
            "cell_identity"
        ],
        record[
            "source_identity"
        ],
        record[
            "condition_protocol_identity"
        ],
        float(
            record[
                "horizon"
            ]
        )
    )


# Building normalized equal-cell prediction weights
def equal_cell_weights(records: list[dict]) -> list[float]:
    counts = Counter(
        record[
            "cell_identity"
        ]
        for record in records
    )

    raw_weights = [
        1.0
        / counts[
            record[
                "cell_identity"
            ]
        ]
        for record in records
    ]

    mean_weight = sum(
        raw_weights
    ) / len(
        raw_weights
    )

    return [
        weight
        / mean_weight
        for weight in raw_weights
    ]


# Calculating one weighted Pearson correlation
def weighted_correlation(first: list[float], second: list[float], weights: list[float]) -> float | None:
    total_weight = sum(
        weights
    )

    first_mean = sum(
        weight * value
        for value, weight in zip(
            first,
            weights,
            strict = True
        )
    ) / total_weight

    second_mean = sum(
        weight * value
        for value, weight in zip(
            second,
            weights,
            strict = True
        )
    ) / total_weight

    covariance = sum(
        weight
        * (
            first_value
            - first_mean
        )
        * (
            second_value
            - second_mean
        )
        for first_value, second_value, weight in zip(
            first,
            second,
            weights,
            strict = True
        )
    )

    first_variance = sum(
        weight
        * (
            value
            - first_mean
        ) ** 2
        for value, weight in zip(
            first,
            weights,
            strict = True
        )
    )

    second_variance = sum(
        weight
        * (
            value
            - second_mean
        ) ** 2
        for value, weight in zip(
            second,
            weights,
            strict = True
        )
    )

    if (
        first_variance <= 0
        or second_variance <= 0
    ):
        return None

    return covariance / math.sqrt(
        first_variance
        * second_variance
    )


# Registering one completed prediction in a group, rejecting duplicate identities
def register_prediction(destination: dict, group_identity: str, record: dict) -> None:
    identity = prediction_identity(
        record
    )

    if identity in destination[group_identity]:
        raise RuntimeError(f"Prediction inventory contains a duplicate identity : {group_identity} : {identity}")

    destination[
        group_identity
    ][
        identity
    ] = record


# Building cross-framework prediction agreement with exact-inventory requirement
def build_prediction_agreement(bayesian_tables: dict, predictive_tables: dict) -> list[dict]:
    bayesian_registry = {
        record[
            "candidate_identity"
        ]: record
        for record in bayesian_tables[
            "candidate_registry"
        ]
    }

    predictive_registry = {
        record[
            "benchmark_identity"
        ]: record
        for record in predictive_tables[
            "candidate_registry"
        ]
    }

    bayesian_predictions = defaultdict(
        dict
    )

    for record in bayesian_tables[
        "grouped_validation_predictions"
    ]:
        if record[
            "prediction_status"
        ] == "completed":
            register_prediction(
                destination = bayesian_predictions,
                group_identity = record[
                    "candidate_identity"
                ],
                record = record
            )

    predictive_predictions = defaultdict(
        dict
    )

    for record in predictive_tables[
        "validation_predictions"
    ]:
        if record[
            "prediction_status"
        ] == "completed":
            register_prediction(
                destination = predictive_predictions,
                group_identity = record[
                    "benchmark_identity"
                ],
                record = record
            )

    summaries = []

    for benchmark_identity, candidate in sorted(
        predictive_registry.items()
    ):
        candidate_identity = candidate[
            "candidate_identity"
        ]

        bayesian_candidate = bayesian_registry[
            candidate_identity
        ]

        first_predictions = bayesian_predictions[
            candidate_identity
        ]

        second_predictions = predictive_predictions[
            benchmark_identity
        ]

        shared_identities = sorted(
            set(
                first_predictions
            ).intersection(
                second_predictions
            )
        )

        paired_records = []
        first_values = []
        second_values = []

        for identity in shared_identities:
            first_record = first_predictions[
                identity
            ]

            second_record = second_predictions[
                identity
            ]

            first_observed = finite_float(
                first_record[
                    "observed_outcome"
                ]
            )

            second_observed = finite_float(
                second_record[
                    "observed_outcome"
                ]
            )

            if (
                first_observed is None
                or second_observed is None
                or not math.isclose(
                    first_observed,
                    second_observed,
                    rel_tol = 0,
                    abs_tol = 1e-12
                )
            ):
                raise RuntimeError("Shared prediction records disagree on the observed outcome.")

            first_prediction = finite_float(
                first_record[
                    "posterior_predictive_median"
                ]
            )

            second_prediction = finite_float(
                second_record[
                    "predicted_outcome"
                ]
            )

            if (
                first_prediction is None
                or second_prediction is None
            ):
                continue

            paired_records.append(
                first_record
            )

            first_values.append(
                first_prediction
            )

            second_values.append(
                second_prediction
            )

        first_inventory_size = len(first_predictions)
        second_inventory_size = len(second_predictions)
        shared_size = len(shared_identities)

        exact_match = (
            first_inventory_size > 0
            and second_inventory_size > 0
            and first_inventory_size == second_inventory_size == shared_size
        )

        both_analytically_available = (
            bayesian_candidate["posterior_analytical_status"] == "available"
            and candidate["analytical_status"] == "available"
        )

        finite_inventory_complete = (
            len(
                paired_records
            )
            == shared_size
        )

        if not paired_records:
            assessment_status = "unavailable"
            assessment_reason_code = "NO_SHARED_VALIDATION_PREDICTIONS"
        elif not finite_inventory_complete:
            assessment_status = "unavailable"
            assessment_reason_code = "SHARED_PREDICTION_VALUE_UNAVAILABLE"
        elif not exact_match:
            assessment_status = "diagnostic_only"
            assessment_reason_code = "PARTIAL_INVENTORY_OVERLAP"
        elif not both_analytically_available:
            assessment_status = "diagnostic_only"
            assessment_reason_code = "SOURCE_MODEL_NOT_ANALYTICALLY_AVAILABLE"
        else:
            assessment_status = "completed"
            assessment_reason_code = None

        weights = (
            equal_cell_weights(paired_records)
            if paired_records
            else []
        )

        differences = [
            second_value - first_value
            for first_value, second_value in zip(
                first_values, second_values, strict = True
            )
        ]

        total_weight = sum(weights)

        def safe_mean_abs():
            if not paired_records or not weights or total_weight == 0:
                return None
            return sum(
                weight * abs(difference)
                for difference, weight in zip(differences, weights, strict = True)
            ) / total_weight

        def safe_rmse():
            if not paired_records or not weights or total_weight == 0:
                return None
            return math.sqrt(
                sum(
                    weight * difference ** 2
                    for difference, weight in zip(differences, weights, strict = True)
                ) / total_weight
            )

        def safe_max_abs():
            if not paired_records:
                return None
            return max(abs(difference) for difference in differences)

        def safe_corr():
            if not paired_records or not weights:
                return None
            return weighted_correlation(first_values, second_values, weights)

        summaries.append({
            "candidate_identity": candidate_identity,
            "model_family": candidate["model_family"],
            "design_scope": candidate["design_scope"],
            "sampling_design_scope": candidate["sampling_design_scope"],
            "aging_mode": candidate["aging_mode"],
            "outcome": candidate["outcome"],
            "cohort": candidate["cohort"],
            "horizon_axis": candidate["horizon_axis"],
            "bayesian_analytical_status": bayesian_candidate["posterior_analytical_status"],
            "predictive_analytical_status": candidate["analytical_status"],
            "bayesian_prediction_count": first_inventory_size,
            "predictive_prediction_count": second_inventory_size,
            "shared_prediction_count": shared_size,
            "shared_inventory_fraction_bayesian": (
                float(shared_size) / float(first_inventory_size)
                if first_inventory_size > 0
                else 0.0
            ),
            "shared_inventory_fraction_predictive": (
                float(shared_size) / float(second_inventory_size)
                if second_inventory_size > 0
                else 0.0
            ),
            "exact_inventory_match": exact_match,
            "shared_cell_count": len({record["cell_identity"] for record in paired_records}),
            "shared_observation_count": len(paired_records),
            "equal_cell_weighted_mean_absolute_prediction_difference": safe_mean_abs(),
            "equal_cell_weighted_root_mean_squared_prediction_difference": safe_rmse(),
            "equal_cell_weighted_maximum_absolute_prediction_difference": safe_max_abs(),
            "equal_cell_weighted_prediction_correlation": safe_corr(),
            "assessment_status": assessment_status,
            "assessment_reason_code": assessment_reason_code,
            "estimate_label": "predictive_agreement"
        })

    return summaries


# Building one reconciliation-checked domain evidence-count record
def domain_evaluation(status: str, total_count: int,
                      completed_count: int, restricted_count: int,
                      unavailable_count: int, reason_code: str | None
) -> dict:
    counts = {
        "total_count": int(total_count),
        "completed_count": int(completed_count),
        "restricted_count": int(restricted_count),
        "unavailable_count": int(unavailable_count)
    }

    if any(
        count < 0
        for count in counts.values()
    ):
        raise RuntimeError("Domain evidence count is negative.")

    if (
        counts["completed_count"]
        + counts["restricted_count"]
        + counts["unavailable_count"]
        != counts["total_count"]
    ):
        raise RuntimeError("Domain evidence counts do not reconcile.")

    return {
        "status": status,
        **counts,
        "reason_code": reason_code
    }


# Deriving a domain status and reason from reconciled evidence counts
def derive_domain_status_from_counts(total_count: int, completed_count: int,
                                     restricted_count: int, unavailable_count: int,
                                     no_evidence_reason_code: str, partial_reason_code: str,
                                     all_diagnostic_reason_code: str, all_unavailable_reason_code: str
) -> tuple[str, str | None]:
    if total_count <= 0:
        return ("unavailable", no_evidence_reason_code)

    if completed_count == total_count:
        return ("completed", None)

    if completed_count > 0:
        return ("completed_with_restrictions", partial_reason_code)

    if unavailable_count == total_count:
        return ("unavailable", all_unavailable_reason_code)

    return ("diagnostic_only", all_diagnostic_reason_code)


# Building evidence-backed assessment-domain coverage records
def build_assessment_coverage(policy: dict, tables: dict, domain_evaluations: dict) -> list[dict]:
    records = []

    terminal_statuses = set(
        policy["assessment_contract"]["terminal_domain_statuses"]
    )

    registered_reason_codes = set(
        policy["assessment_contract"]["registered_reason_codes"]
    )

    restricted_statuses = terminal_statuses.difference(
        {"completed"}
    )

    for assessment in policy["assessment_domains"]:
        identity = assessment["assessment_identity"]

        if identity not in domain_evaluations:
            raise RuntimeError(f"Assessment domain was not evaluated : {identity}")

        evaluation = domain_evaluations[identity]

        status = evaluation["status"]
        reason_code = evaluation["reason_code"]

        if status not in terminal_statuses:
            raise RuntimeError(f"Assessment domain has non-terminal status : {identity} : {status}")

        if status in restricted_statuses and not reason_code:
            raise RuntimeError(f"Restricted assessment domain does not provide a reason code : {identity} : {status}")

        if reason_code and reason_code not in registered_reason_codes:
            raise RuntimeError(f"Assessment domain uses an unregistered reason code : {identity} : {reason_code}")

        if status == "completed" and reason_code:
            raise RuntimeError(
                f"Completed assessment domain unexpectedly provides a restriction reason : {identity} : {reason_code}"
            )

        records.append({
            "assessment_identity": identity,
            "evidence_source": assessment["evidence_source"],
            "evidence_unit": assessment["evidence_unit"],
            "total_evidence_count": evaluation["total_count"],
            "completed_evidence_count": evaluation["completed_count"],
            "restricted_evidence_count": evaluation["restricted_count"],
            "unavailable_evidence_count": evaluation["unavailable_count"],
            "assessment_status": status,
            "assessment_reason_code": reason_code,
            "interpretation": assessment["interpretation"]
        })

    return records


# Building the descriptive model identity used for cohort matching
def compute_descriptive_model_identity(candidate: dict) -> str:
    return "|".join([
        str(candidate.get("template_name", "")),
        str(candidate.get("estimator", "")),
        str(candidate.get("specification", "")),
        str(candidate.get("context_family", ""))
    ])


# Building exact out-of-fold validation prediction inventories keyed by record identity
def build_validation_prediction_inventories(records: list[dict], identity_field: str,
                                            prediction_status_field: str | None = None
) -> dict[str, frozenset[tuple]]:
    inventories = defaultdict(set)

    for record in records:
        if (
            prediction_status_field is not None
            and record.get(prediction_status_field) != "completed"
        ):
            continue

        record_identity = str(record[identity_field])

        horizon_value = finite_float(
            record["horizon"]
        )

        if horizon_value is None:
            raise RuntimeError(f"Validation prediction contains an invalid horizon : {record_identity}")

        inventory_identity = (
            str(record["cell_identity"]),
            str(record["source_identity"]),
            str(record["condition_protocol_identity"]),
            horizon_value
        )

        if inventory_identity in inventories[
            record_identity
        ]:
            raise RuntimeError(
                f"Validation prediction inventory contains a duplicate record : {record_identity} : {inventory_identity}"
            )

        inventories[
            record_identity
        ].add(
            inventory_identity
        )

    return {
        record_identity: frozenset(inventory)
        for record_identity, inventory in inventories.items()
    }


# Parsing one optional non-negative integral count
def parse_optional_count(value: object, identity: str, field_name: str) -> int | None:
    if value in {
        None,
        ""
    }:
        return None

    numerical_value = finite_float(
        value
    )

    if (
        numerical_value is None
        or numerical_value < 0
        or not numerical_value.is_integer()
    ):
        raise RuntimeError(f"Validation metric contains an invalid count : {identity} : {field_name} : {value}")

    return int(
        numerical_value
    )


# Confirming one metric reconciles exactly with its prediction inventory
def validate_prediction_inventory_counts(identity: str, inventory: frozenset[tuple] | None,
                                         declared_cell_count: object, declared_observation_count: object,
                                         metric_status: object
) -> None:
    normalized_status = str(
        metric_status
        or ""
    )

    if normalized_status not in {
        "completed",
        "unavailable"
    }:
        raise RuntimeError(f"Validation metric has an unsupported status : {identity} : {normalized_status}")

    expected_cell_count = parse_optional_count(
        declared_cell_count,
        identity,
        "cell_count"
    )

    expected_observation_count = parse_optional_count(
        declared_observation_count,
        identity,
        "observation_count"
    )

    observed_inventory = (
        inventory
        if inventory is not None
        else frozenset()
    )

    observed_observation_count = len(
        observed_inventory
    )

    observed_cell_count = len({
        inventory_record[0]
        for inventory_record in observed_inventory
    })

    if normalized_status == "completed":
        if (
            expected_cell_count is None
            or expected_observation_count is None
        ):
            raise RuntimeError(f"Completed validation metric is missing required counts : {identity}")

        if (
            expected_observation_count > 0
            and inventory is None
        ):
            raise RuntimeError(f"Completed validation metric has no prediction inventory : {identity}")

        if (
            observed_observation_count
            != expected_observation_count
        ):
            raise RuntimeError(f"Validation prediction count does not match its metric record : {identity}")

        if (
            observed_cell_count
            != expected_cell_count
        ):
            raise RuntimeError(f"Validation cell count does not match its metric record : {identity}")

        return

    if observed_inventory:
        raise RuntimeError(f"Unavailable validation metric has completed prediction records : {identity}")

    if (
        expected_cell_count not in {
            None,
            0
        }
        or expected_observation_count not in {
            None,
            0
        }
    ):
        raise RuntimeError(f"Unavailable validation metric declares non-zero validation counts : {identity}")


# Confirming all prediction inventories reconcile with exactly one metric
def reconcile_prediction_inventories(prediction_records: list[dict], metric_records: list[dict],
                                     identity_field: str, prediction_status_field: str | None,
                                     metric_status_field: str, cell_count_field: str,
                                     observation_count_field: str
) -> None:
    inventories = build_validation_prediction_inventories(
        prediction_records,
        identity_field,
        prediction_status_field
    )

    metrics_by_identity = {}

    for metric in metric_records:
        identity = str(
            metric[
                identity_field
            ]
        )

        if identity in metrics_by_identity:
            raise RuntimeError(f"Validation metrics contain a duplicate identity : {identity}")

        metrics_by_identity[
            identity
        ] = metric

    orphan_inventory_identities = (
        set(
            inventories
        )
        - set(
            metrics_by_identity
        )
    )

    if orphan_inventory_identities:
        raise RuntimeError(
            f"Validation prediction inventories do not map to metric records : {sorted(orphan_inventory_identities)}"
        )

    for identity, metric in (
        metrics_by_identity.items()
    ):
        validate_prediction_inventory_counts(
            identity = identity,
            inventory = inventories.get(
                identity
            ),
            declared_cell_count = metric.get(
                cell_count_field
            ),
            declared_observation_count = metric.get(
                observation_count_field
            ),
            metric_status = metric.get(
                metric_status_field
            )
        )


# Normalizing completed held-out predictions from each analytical framework
def build_held_out_prediction_inventory(descriptive_tables: dict, bayesian_tables: dict,
                                        predictive_tables: dict
) -> list[dict]:
    records = []

    descriptive_candidates = {
        record["candidate_identity"]: record
        for record in descriptive_tables["candidate_registry"]
    }

    for prediction in descriptive_tables["validation_predictions"]:
        candidate = descriptive_candidates[
            prediction["candidate_identity"]
        ]

        records.append(
            {
                "source_analysis": "descriptive_modeling",
                "candidate_or_benchmark_identity": prediction["candidate_identity"],
                "model_name": candidate["template_name"],
                "dataset_stage": dataset_stage(candidate["design_scope"]),
                "design_scope": candidate["design_scope"],
                "sampling_design_scope": candidate["sampling_design_scope"],
                "aging_mode": candidate["aging_mode"],
                "outcome": candidate["outcome"],
                "cohort": candidate["cohort"],
                "horizon_axis": candidate["horizon_axis"],
                "analytical_status": candidate["analytical_result_status"],
                "fold_identity": prediction["fold_identity"],
                "cell_identity": prediction["cell_identity"],
                "source_identity": prediction["source_identity"],
                "condition_protocol_identity": prediction["condition_protocol_identity"],
                "horizon": finite_float(prediction["horizon"]),
                "observed_outcome": finite_float(prediction["observed_outcome"]),
                "predicted_outcome": finite_float(prediction["predicted_outcome"]),
                "interval_status": prediction["prediction_interval_status"],
                "interval_lower": finite_float(prediction["prediction_interval_lower"]),
                "interval_upper": finite_float(prediction["prediction_interval_upper"]),
                "nominal_interval_probability": 0.95
            }
        )

    bayesian_candidates = {
        record["candidate_identity"]: record
        for record in bayesian_tables["candidate_registry"]
    }

    for prediction in bayesian_tables["grouped_validation_predictions"]:
        if prediction["prediction_status"] != "completed":
            continue

        candidate = bayesian_candidates[
            prediction["candidate_identity"]
        ]

        records.append(
            {
                "source_analysis": "bayesian_modeling",
                "candidate_or_benchmark_identity": prediction["candidate_identity"],
                "model_name": "hierarchical_bayesian_trajectory",
                "dataset_stage": dataset_stage(candidate["design_scope"]),
                "design_scope": candidate["design_scope"],
                "sampling_design_scope": candidate["sampling_design_scope"],
                "aging_mode": candidate["aging_mode"],
                "outcome": candidate["outcome"],
                "cohort": candidate["cohort"],
                "horizon_axis": candidate["horizon_axis"],
                "analytical_status": candidate["posterior_analytical_status"],
                "fold_identity": str(prediction["fold_index"]),
                "cell_identity": prediction["cell_identity"],
                "source_identity": prediction["source_identity"],
                "condition_protocol_identity": prediction["condition_protocol_identity"],
                "horizon": finite_float(prediction["horizon"]),
                "observed_outcome": finite_float(prediction["observed_outcome"]),
                "predicted_outcome": finite_float(prediction["posterior_predictive_median"]),
                "interval_status": "completed",
                "interval_lower": finite_float(prediction["posterior_predictive_interval_lower"]),
                "interval_upper": finite_float(prediction["posterior_predictive_interval_upper"]),
                "nominal_interval_probability": 0.95
            }
        )

    predictive_candidates = {
        record["benchmark_identity"]: record
        for record in predictive_tables["candidate_registry"]
    }

    for prediction in predictive_tables["validation_predictions"]:
        if prediction["prediction_status"] != "completed":
            continue

        candidate = predictive_candidates[
            prediction["benchmark_identity"]
        ]

        records.append(
            {
                "source_analysis": "predictive_modeling",
                "candidate_or_benchmark_identity": prediction["benchmark_identity"],
                "model_name": candidate["model_family"],
                "dataset_stage": dataset_stage(candidate["design_scope"]),
                "design_scope": candidate["design_scope"],
                "sampling_design_scope": candidate["sampling_design_scope"],
                "aging_mode": candidate["aging_mode"],
                "outcome": candidate["outcome"],
                "cohort": candidate["cohort"],
                "horizon_axis": candidate["horizon_axis"],
                "analytical_status": candidate["analytical_status"],
                "fold_identity": str(prediction["outer_fold_index"]),
                "cell_identity": prediction["cell_identity"],
                "source_identity": prediction["source_identity"],
                "condition_protocol_identity": prediction["condition_protocol_identity"],
                "horizon": finite_float(prediction["horizon"]),
                "observed_outcome": finite_float(prediction["observed_outcome"]),
                "predicted_outcome": finite_float(prediction["predicted_outcome"]),
                "interval_status": "completed",
                "interval_lower": finite_float(prediction["prediction_interval_lower"]),
                "interval_upper": finite_float(prediction["prediction_interval_upper"]),
                "nominal_interval_probability": 0.9
            }
        )

    required_numeric_fields = (
        "horizon",
        "observed_outcome",
        "predicted_outcome"
    )

    for record in records:
        if any(
            record[field_name] is None
            for field_name in required_numeric_fields
        ):
            raise RuntimeError("Completed held-out prediction contains a missing or non-finite value.")

    records.sort(
        key = lambda record: (
            record["source_analysis"],
            record["candidate_or_benchmark_identity"],
            record["fold_identity"],
            record["cell_identity"],
            record["source_identity"]
        )
    )

    return records


# Calculating equal-cell-weighted held-out metrics
def calculate_held_out_metrics(records: list[dict]) -> dict | None:
    if not records:
        return None

    weights = equal_cell_weights(
        records
    )

    weight_sum = sum(
        weights
    )

    if weight_sum <= 0:
        return None

    observed = [
        float(record["observed_outcome"])
        for record in records
    ]

    predicted = [
        float(record["predicted_outcome"])
        for record in records
    ]

    absolute_errors = [
        abs(observed_value - predicted_value)
        for observed_value, predicted_value in zip(
            observed,
            predicted,
            strict = True
        )
    ]

    squared_errors = [
        (observed_value - predicted_value) ** 2
        for observed_value, predicted_value in zip(
            observed,
            predicted,
            strict = True
        )
    ]

    weighted_mean_observed = sum(
        weight * observed_value
        for weight, observed_value in zip(
            weights,
            observed,
            strict = True
        )
    ) / weight_sum

    weighted_total_sum_of_squares = sum(
        weight * (observed_value - weighted_mean_observed) ** 2
        for weight, observed_value in zip(
            weights,
            observed,
            strict = True
        )
    )

    weighted_residual_sum_of_squares = sum(
        weight * squared_error
        for weight, squared_error in zip(
            weights,
            squared_errors,
            strict = True
        )
    )

    predictive_r_squared = None

    if weighted_total_sum_of_squares > 0:
        predictive_r_squared = (
            1
            - (
                weighted_residual_sum_of_squares
                / weighted_total_sum_of_squares
            )
        )

    interval_records = [
        record
        for record in records
        if (
            record["interval_status"] == "completed"
            and record["interval_lower"] is not None
            and record["interval_upper"] is not None
            and record["interval_lower"] <= record["interval_upper"]
        )
    ]

    interval_coverage = None
    mean_interval_width = None

    if len(interval_records) == len(records):
        interval_weights = equal_cell_weights(
            interval_records
        )

        interval_weight_sum = sum(
            interval_weights
        )

        interval_coverage = sum(
            weight * float(
                record["interval_lower"]
                <= record["observed_outcome"]
                <= record["interval_upper"]
            )
            for weight, record in zip(
                interval_weights,
                interval_records,
                strict = True
            )
        ) / interval_weight_sum

        mean_interval_width = sum(
            weight * (
                record["interval_upper"]
                - record["interval_lower"]
            )
            for weight, record in zip(
                interval_weights,
                interval_records,
                strict = True
            )
        ) / interval_weight_sum

    return {
        "prediction_count": len(records),
        "independent_cell_count": len(
            {
                record["cell_identity"]
                for record in records
            }
        ),
        "minimum_horizon": min(
            record["horizon"]
            for record in records
        ),
        "maximum_horizon": max(
            record["horizon"]
            for record in records
        ),
        "mean_absolute_error": sum(
            weight * error
            for weight, error in zip(
                weights,
                absolute_errors,
                strict = True
            )
        ) / weight_sum,
        "root_mean_squared_error": math.sqrt(
            weighted_residual_sum_of_squares
            / weight_sum
        ),
        "predictive_r_squared": predictive_r_squared,
        "interval_coverage": interval_coverage,
        "mean_interval_width": mean_interval_width
    }


# Assigning evidence status without changing fitted models
def held_out_assessment_status( records: list[dict], minimum_cell_count: int) -> tuple[str, str | None]:
    if not records:
        return (
            "unavailable",
            "HORIZON_EVALUATION_INSUFFICIENT_OBSERVATIONS"
        )

    cell_count = len(
        {
            record["cell_identity"]
            for record in records
        }
    )

    if cell_count < minimum_cell_count:
        return (
            "unavailable",
            "HORIZON_EVALUATION_INSUFFICIENT_CELLS"
        )

    if any(
        record["analytical_status"] != "available"
        for record in records
    ):
        return (
            "completed_with_restrictions",
            "SOURCE_MODEL_NOT_ANALYTICALLY_AVAILABLE"
        )

    return ("completed", None)


# Summarizing interval coverage and width by registered held-out groupings
def build_prediction_interval_diagnostics(prediction_inventory: list[dict]) -> list[dict]:
    grouped_records = defaultdict(
        list
    )

    for record in prediction_inventory:
        base_key = (
            record["source_analysis"],
            record["candidate_or_benchmark_identity"]
        )

        grouped_records[base_key + ("overall", "all")].append(record)
        grouped_records[base_key + ("validation_fold", record["fold_identity"])].append(record)
        grouped_records[base_key + ("condition_protocol", record["condition_protocol_identity"])].append(record)

    output_records = []

    for (
        source_analysis,
        identity,
        grouping_dimension,
        grouping_value
    ), records in sorted(
        grouped_records.items()
    ):
        first = records[0]

        available_intervals = [
            record
            for record in records
            if (
                record["interval_status"] == "completed"
                and record["interval_lower"] is not None
                and record["interval_upper"] is not None
                and record["interval_lower"] <= record["interval_upper"]
            )
        ]

        interval_coverage = None
        mean_interval_width = None
        covered_count = 0

        if available_intervals:
            interval_weights = equal_cell_weights(available_intervals)
            interval_weight_sum = sum(interval_weights)

            covered_count = sum(
                int(
                    record["interval_lower"]
                    <= record["observed_outcome"]
                    <= record["interval_upper"]
                )
                for record in available_intervals
            )

            interval_coverage = sum(
                weight * float(
                    record["interval_lower"]
                    <= record["observed_outcome"]
                    <= record["interval_upper"]
                )
                for weight, record in zip(
                    interval_weights,
                    available_intervals,
                    strict = True
                )
            ) / interval_weight_sum

            mean_interval_width = sum(
                weight * (record["interval_upper"] - record["interval_lower"])
                for weight, record in zip(
                    interval_weights,
                    available_intervals,
                    strict = True
                )
            ) / interval_weight_sum

        if not available_intervals:
            assessment_status = "unavailable"
            assessment_reason_code = "PREDICTION_INTERVALS_UNAVAILABLE"

        elif len(available_intervals) != len(records):
            assessment_status = "completed_with_restrictions"
            assessment_reason_code = "PREDICTION_INTERVALS_PARTIALLY_AVAILABLE"

        elif first["analytical_status"] != "available":
            assessment_status = "completed_with_restrictions"
            assessment_reason_code = "SOURCE_MODEL_NOT_ANALYTICALLY_AVAILABLE"

        else:
            assessment_status = "completed"
            assessment_reason_code = None

        output_records.append(
            {
                "source_analysis": source_analysis,
                "candidate_or_benchmark_identity": identity,
                "model_name": first["model_name"],
                "dataset_stage": first["dataset_stage"],
                "design_scope": first["design_scope"],
                "sampling_design_scope": first["sampling_design_scope"],
                "aging_mode": first["aging_mode"],
                "outcome": first["outcome"],
                "cohort": first["cohort"],
                "horizon_axis": first["horizon_axis"],
                "grouping_dimension": grouping_dimension,
                "grouping_value": grouping_value,
                "nominal_interval_probability": first["nominal_interval_probability"],
                "prediction_count": len(records),
                "independent_cell_count": len({
                    record["cell_identity"]
                    for record in records
                }),
                "interval_available_count": len(available_intervals),
                "interval_covered_count": covered_count,
                "equal_cell_weighted_interval_coverage": interval_coverage,
                "equal_cell_weighted_mean_interval_width": mean_interval_width,
                "assessment_status": assessment_status,
                "assessment_reason_code": assessment_reason_code,
                "estimate_label": "predictive"
            }
        )

    return output_records


# Recomputing held-out metrics under registered evaluation windows
def build_horizon_evaluation_sensitivity(prediction_inventory: list[dict], minimum_cell_count: int) -> list[dict]:
    grouped_records = defaultdict(list)

    for record in prediction_inventory:
        grouped_records[
            (
                record["source_analysis"],
                record["candidate_or_benchmark_identity"]
            )
        ].append(record)

    output_records = []

    for (
        source_analysis,
        identity
    ), full_records in sorted(
        grouped_records.items()
    ):
        first = full_records[0]

        minimum_horizon_by_cell = {}
        maximum_horizon_by_cell = {}

        for record in full_records:
            cell_identity = record["cell_identity"]

            minimum_horizon_by_cell[cell_identity] = min(
                minimum_horizon_by_cell.get(cell_identity, record["horizon"]),
                record["horizon"]
            )

            maximum_horizon_by_cell[cell_identity] = max(
                maximum_horizon_by_cell.get(cell_identity, record["horizon"]),
                record["horizon"]
            )

        evaluation_windows = {
            "full_evaluation_inventory": full_records,
            "first_follow_up_excluded": [
                record
                for record in full_records
                if record["horizon"] > minimum_horizon_by_cell[record["cell_identity"]]
            ],
            "terminal_observation_excluded": [
                record
                for record in full_records
                if record["horizon"] < maximum_horizon_by_cell[record["cell_identity"]]
            ]
        }

        full_metrics = calculate_held_out_metrics(full_records)

        for window_identity, records in evaluation_windows.items():
            metrics = calculate_held_out_metrics(records)

            assessment_status, assessment_reason_code = held_out_assessment_status(
                records,
                minimum_cell_count
            )

            output_records.append(
                {
                    "source_analysis": source_analysis,
                    "candidate_or_benchmark_identity": identity,
                    "model_name": first["model_name"],
                    "dataset_stage": first["dataset_stage"],
                    "design_scope": first["design_scope"],
                    "sampling_design_scope": first["sampling_design_scope"],
                    "aging_mode": first["aging_mode"],
                    "outcome": first["outcome"],
                    "cohort": first["cohort"],
                    "horizon_axis": first["horizon_axis"],
                    "evaluation_window_identity": window_identity,
                    "prediction_count": (metrics["prediction_count"] if metrics is not None else 0),
                    "independent_cell_count": (metrics["independent_cell_count"] if metrics is not None else 0),
                    "minimum_horizon": (metrics["minimum_horizon"] if metrics is not None else None),
                    "maximum_horizon": (metrics["maximum_horizon"] if metrics is not None else None),
                    "equal_cell_weighted_mean_absolute_error": (metrics["mean_absolute_error"] if metrics is not None else None),
                    "equal_cell_weighted_root_mean_squared_error": (metrics["root_mean_squared_error"] if metrics is not None else None),
                    "equal_cell_weighted_predictive_r_squared": (metrics["predictive_r_squared"] if metrics is not None else None),
                    "equal_cell_weighted_interval_coverage": (metrics["interval_coverage"] if metrics is not None else None),
                    "equal_cell_weighted_mean_interval_width": (metrics["mean_interval_width"] if metrics is not None else None),
                    "root_mean_squared_error_difference_from_full_inventory": (
                        (metrics["root_mean_squared_error"] - full_metrics["root_mean_squared_error"])
                        if (metrics is not None and full_metrics is not None)
                        else None
                    ),
                    "assessment_status": assessment_status,
                    "assessment_reason_code": assessment_reason_code,
                    "estimate_label": "predictive"
                }
            )

    return output_records


# Joining documented reason codes to prediction-source identities
def build_feature_reason_lookup(reference_feature_records: list[dict]
) -> tuple[dict[tuple[str, str], frozenset[str]], list[str]]:
    lookup = {}
    reason_inventory = set()

    for record in reference_feature_records:
        identity = (
            str(record["source_identity"]),
            str(record["cell_identity"])
        )

        if identity in lookup:
            raise RuntimeError(f"Reference feature provenance contains a duplicate identity : {identity}")

        reasons = set()

        for field_name in (
            "archive_quality_reason_codes",
            "quality_reason_codes"
        ):
            field_value = record[field_name]

            if not isinstance(field_value, list):
                raise RuntimeError(f"Reference feature reason-code field is not a list : {field_name}")

            for reason_code in field_value:
                if not isinstance(reason_code, str) or not reason_code:
                    raise RuntimeError(f"Reference feature reason code is invalid : {identity}")

                reasons.add(reason_code)
                reason_inventory.add(reason_code)

        lookup[identity] = frozenset(reasons)

    return (lookup, sorted(reason_inventory))


# Recomputing held-out metrics after one documented reason-code exclusion
def build_incident_evaluation_sensitivity(prediction_inventory: list[dict], reference_feature_records: list[dict],
                                          minimum_cell_count: int
) -> list[dict]:
    reason_lookup, reason_inventory = build_feature_reason_lookup(
        reference_feature_records
    )

    enriched_records = []

    for record in prediction_inventory:
        provenance_identity = (
            record["source_identity"],
            record["cell_identity"]
        )

        if provenance_identity not in reason_lookup:
            raise RuntimeError(f"Held-out prediction does not map to reference feature provenance : {provenance_identity}")

        enriched_record = dict(record)
        enriched_record["incident_reason_codes"] = reason_lookup[provenance_identity]

        enriched_records.append(enriched_record)

    grouped_records = defaultdict(list)

    for record in enriched_records:
        grouped_records[
            (
                record["source_analysis"],
                record["candidate_or_benchmark_identity"]
            )
        ].append(record)

    output_records = []

    for (
        source_analysis,
        identity
    ), full_records in sorted(
        grouped_records.items()
    ):
        first = full_records[0]
        full_metrics = calculate_held_out_metrics(full_records)

        for reason_code in reason_inventory:
            excluded_records = [
                record
                for record in full_records
                if reason_code in record["incident_reason_codes"]
            ]

            retained_records = [
                record
                for record in full_records
                if reason_code not in record["incident_reason_codes"]
            ]

            retained_metrics = calculate_held_out_metrics(retained_records)

            if not excluded_records:
                assessment_status = "unavailable"
                assessment_reason_code = "INCIDENT_REASON_NOT_PRESENT"

            elif not retained_records:
                assessment_status = "unavailable"
                assessment_reason_code = "INCIDENT_EXCLUSION_EMPTY"

            elif len({
                record["cell_identity"]
                for record in retained_records
            }) < minimum_cell_count:
                assessment_status = "unavailable"
                assessment_reason_code = "INCIDENT_EXCLUSION_INSUFFICIENT_CELLS"

            elif first["analytical_status"] != "available":
                assessment_status = "completed_with_restrictions"
                assessment_reason_code = "SOURCE_MODEL_NOT_ANALYTICALLY_AVAILABLE"

            else:
                assessment_status = "completed"
                assessment_reason_code = None

            output_records.append(
                {
                    "source_analysis": source_analysis,
                    "candidate_or_benchmark_identity": identity,
                    "model_name": first["model_name"],
                    "dataset_stage": first["dataset_stage"],
                    "design_scope": first["design_scope"],
                    "sampling_design_scope": first["sampling_design_scope"],
                    "aging_mode": first["aging_mode"],
                    "outcome": first["outcome"],
                    "cohort": first["cohort"],
                    "horizon_axis": first["horizon_axis"],
                    "incident_reason_code": reason_code,
                    "full_prediction_count": len(full_records),
                    "excluded_prediction_count": len(excluded_records),
                    "retained_prediction_count": len(retained_records),
                    "full_independent_cell_count": len({
                        record["cell_identity"]
                        for record in full_records
                    }),
                    "retained_independent_cell_count": len({
                        record["cell_identity"]
                        for record in retained_records
                    }),
                    "full_equal_cell_weighted_mean_absolute_error": (full_metrics["mean_absolute_error"] if full_metrics is not None else None),
                    "retained_equal_cell_weighted_mean_absolute_error": (retained_metrics["mean_absolute_error"] if retained_metrics is not None else None),
                    "mean_absolute_error_difference": (
                        (retained_metrics["mean_absolute_error"] - full_metrics["mean_absolute_error"])
                        if (retained_metrics is not None and full_metrics is not None)
                        else None
                    ),
                    "full_equal_cell_weighted_root_mean_squared_error": (full_metrics["root_mean_squared_error"] if full_metrics is not None else None),
                    "retained_equal_cell_weighted_root_mean_squared_error": (retained_metrics["root_mean_squared_error"] if retained_metrics is not None else None),
                    "root_mean_squared_error_difference": (
                        (retained_metrics["root_mean_squared_error"] - full_metrics["root_mean_squared_error"])
                        if (retained_metrics is not None and full_metrics is not None)
                        else None
                    ),
                    "full_equal_cell_weighted_predictive_r_squared": (full_metrics["predictive_r_squared"] if full_metrics is not None else None),
                    "retained_equal_cell_weighted_predictive_r_squared": (retained_metrics["predictive_r_squared"] if retained_metrics is not None else None),
                    "model_refitted": False,
                    "assessment_status": assessment_status,
                    "assessment_reason_code": assessment_reason_code,
                    "estimate_label": "predictive"
                }
            )

    return output_records


# Building exact-match primary-versus-sensitivity cohort comparisons
def build_cohort_sensitivity(descriptive_tables: dict, bayesian_tables: dict,
                             predictive_tables: dict, policy: dict
) -> list[dict]:
    matching_fields = policy["assessment_contract"]["cohort_sensitivity_matching_fields"]

    descriptive_prediction_inventories = (
        build_validation_prediction_inventories(
            descriptive_tables.get("validation_predictions", []),
            identity_field = "candidate_identity"
        )
    )

    bayesian_prediction_inventories = (
        build_validation_prediction_inventories(
            bayesian_tables.get("grouped_validation_predictions", []),
            identity_field = "candidate_identity",
            prediction_status_field = "prediction_status"
        )
    )

    predictive_prediction_inventories = (
        build_validation_prediction_inventories(
            predictive_tables.get("validation_predictions", []),
            identity_field = "benchmark_identity",
            prediction_status_field = "prediction_status"
        )
    )

    normalized_records = []

    descriptive_registry = {r["candidate_identity"]: r for r in descriptive_tables["candidate_registry"]}

    for metric in descriptive_tables["validation_metrics"]:
        candidate = descriptive_registry[metric["candidate_identity"]]
        normalized_records.append({
            "source_analysis": "descriptive_modeling",
            "model_identity": compute_descriptive_model_identity(candidate),
            "design_scope": candidate["design_scope"],
            "sampling_design_scope": candidate["sampling_design_scope"],
            "aging_mode": candidate["aging_mode"],
            "outcome": candidate["outcome"],
            "cohort": candidate["cohort"],
            "horizon_axis": candidate["horizon_axis"],
            "condition_identities": tuple(sorted(candidate.get("condition_protocol_identities", []))),
            "support_interval_minimum": candidate.get("support_interval_minimum"),
            "support_interval_maximum": candidate.get("support_interval_maximum"),
            "support_interval_lower_inclusive": candidate.get("support_interval_lower_inclusive"),
            "support_interval_upper_inclusive": candidate.get("support_interval_upper_inclusive"),
            "analytical_status": normalize_status(candidate["analytical_result_status"]),
            "validation_status": metric["validation_status"],
            "root_mean_squared_error": metric["root_mean_squared_error"],
            "mean_absolute_error": metric["mean_absolute_error"],
            "cell_count": metric.get("independent_cell_count"),
            "observation_count": metric.get("observation_count"),
            "validation_inventory": descriptive_prediction_inventories.get(
                metric["candidate_identity"]
            )
        })

    bayesian_registry = {r["candidate_identity"]: r for r in bayesian_tables["candidate_registry"]}

    for metric in bayesian_tables["grouped_validation_metrics"]:
        candidate = bayesian_registry[metric["candidate_identity"]]
        normalized_records.append({
            "source_analysis": "bayesian_modeling",
            "model_identity": "hierarchical_bayesian_trajectory",
            "design_scope": candidate["design_scope"],
            "sampling_design_scope": candidate["sampling_design_scope"],
            "aging_mode": candidate["aging_mode"],
            "outcome": candidate["outcome"],
            "cohort": candidate["cohort"],
            "horizon_axis": candidate["horizon_axis"],
            "condition_identities": tuple(sorted(candidate.get("sorted_condition_protocol_identities", []))),
            "support_interval_minimum": candidate.get("support_interval_minimum"),
            "support_interval_maximum": candidate.get("support_interval_maximum"),
            "support_interval_lower_inclusive": candidate.get("support_interval_lower_inclusive"),
            "support_interval_upper_inclusive": candidate.get("support_interval_upper_inclusive"),
            "analytical_status": normalize_status(candidate["posterior_analytical_status"]),
            "validation_status": metric["validation_status"],
            "root_mean_squared_error": metric.get("equal_cell_weighted_root_mean_squared_error"),
            "mean_absolute_error": metric.get("equal_cell_weighted_mean_absolute_error"),
            "cell_count": metric.get("test_cell_count"),
            "observation_count": metric.get("test_observation_count"),
            "validation_inventory": bayesian_prediction_inventories.get(
                metric["candidate_identity"]
            )
        })

    predictive_registry = {r["benchmark_identity"]: r for r in predictive_tables["candidate_registry"]}

    for metric in predictive_tables["validation_metrics"]:
        candidate = predictive_registry[metric["benchmark_identity"]]
        normalized_records.append({
            "source_analysis": "predictive_modeling",
            "model_identity": candidate["model_family"],
            "design_scope": candidate["design_scope"],
            "sampling_design_scope": candidate["sampling_design_scope"],
            "aging_mode": candidate["aging_mode"],
            "outcome": candidate["outcome"],
            "cohort": candidate["cohort"],
            "horizon_axis": candidate["horizon_axis"],
            "condition_identities": tuple(sorted(candidate.get("condition_protocol_identities", []))),
            "support_interval_minimum": candidate.get("support_interval_minimum"),
            "support_interval_maximum": candidate.get("support_interval_maximum"),
            "support_interval_lower_inclusive": candidate.get("support_interval_lower_inclusive"),
            "support_interval_upper_inclusive": candidate.get("support_interval_upper_inclusive"),
            "analytical_status": normalize_status(candidate["analytical_status"]),
            "validation_status": metric["validation_status"],
            "root_mean_squared_error": metric.get("equal_cell_weighted_root_mean_squared_error"),
            "mean_absolute_error": metric.get("equal_cell_weighted_mean_absolute_error"),
            "cell_count": metric.get("test_cell_count"),
            "observation_count": metric.get("test_observation_count"),
            "validation_inventory": predictive_prediction_inventories.get(
                metric["benchmark_identity"]
            )
        })

    grouped_by_key = defaultdict(lambda: {"primary": [], "sensitivity": []})

    for record in normalized_records:
        if record["cohort"] not in {"primary", "sensitivity"}:
            continue

        key_values = tuple(str(record.get(f) or "") for f in matching_fields)
        key = (record["source_analysis"],) + key_values
        grouped_by_key[key][record["cohort"]].append(record)

    records = []

    for key, cohorts in sorted(grouped_by_key.items()):
        primary_records = cohorts["primary"]
        sensitivity_records = cohorts["sensitivity"]
        primary_count = len(primary_records)
        sensitivity_count = len(sensitivity_records)

        matching_key_identity = hashlib.sha256(
            canonical_json({
                "source_analysis": key[0],
                "matching_fields": dict(zip(matching_fields, key[1:], strict = True))
            }).encode("utf-8")
        ).hexdigest()

        matching_fields_from_key = dict(zip(matching_fields, key[1:], strict = True))

        if primary_count == 0 or sensitivity_count == 0:
            status = "unavailable"
            reason_code = "COHORT_COUNTERPART_UNAVAILABLE"
            primary = primary_records[0] if primary_count == 1 else None
            sensitivity = sensitivity_records[0] if sensitivity_count == 1 else None
        elif primary_count != 1 or sensitivity_count != 1:
            status = "unavailable"
            reason_code = "COHORT_MATCHING_KEY_AMBIGUOUS"
            primary = None
            sensitivity = None
        else:
            primary = primary_records[0]
            sensitivity = sensitivity_records[0]

            primary_rmse = finite_float(primary["root_mean_squared_error"])
            sensitivity_rmse = finite_float(sensitivity["root_mean_squared_error"])
            primary_mae = finite_float(primary["mean_absolute_error"])
            sensitivity_mae = finite_float(sensitivity["mean_absolute_error"])

            metrics_finite = all(
                v is not None for v in (primary_rmse, sensitivity_rmse, primary_mae, sensitivity_mae)
            )
            both_available = (
                primary["analytical_status"] == "available"
                and sensitivity["analytical_status"] == "available"
            )
            both_validated = (
                primary["validation_status"] == "completed"
                and sensitivity["validation_status"] == "completed"
            )

            primary_inventory = primary.get("validation_inventory")
            sensitivity_inventory = sensitivity.get("validation_inventory")

            if not metrics_finite or not both_validated:
                status = "unavailable"
                reason_code = "VALIDATION_METRICS_UNAVAILABLE"
            elif primary_inventory is None or sensitivity_inventory is None:
                status = "unavailable"
                reason_code = "VALIDATION_POPULATION_INVENTORY_UNAVAILABLE"
            elif primary_inventory != sensitivity_inventory:
                status = "diagnostic_only"
                reason_code = "OUT_OF_FOLD_POPULATIONS_DIFFER"
            elif not both_available:
                status = "diagnostic_only"
                reason_code = "SOURCE_MODEL_NOT_ANALYTICALLY_AVAILABLE"
            else:
                status = "completed"
                reason_code = None

        primary_rmse_val = finite_float(primary["root_mean_squared_error"]) if primary else None
        sensitivity_rmse_val = finite_float(sensitivity["root_mean_squared_error"]) if sensitivity else None
        primary_mae_val = finite_float(primary["mean_absolute_error"]) if primary else None
        sensitivity_mae_val = finite_float(sensitivity["mean_absolute_error"]) if sensitivity else None

        metrics_comparable = all(
            v is not None for v in (primary_rmse_val, sensitivity_rmse_val, primary_mae_val, sensitivity_mae_val)
        )

        def parse_optional_int(value):
            if value is None or value == "":
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        primary_cell_count = parse_optional_int(
            primary.get("cell_count") if primary else None
        )
        sensitivity_cell_count = parse_optional_int(
            sensitivity.get("cell_count") if sensitivity else None
        )
        primary_observation_count = parse_optional_int(
            primary.get("observation_count") if primary else None
        )
        sensitivity_observation_count = parse_optional_int(
            sensitivity.get("observation_count") if sensitivity else None
        )

        primary_inventory_for_relationship = (
            primary.get("validation_inventory") if primary else None
        )
        sensitivity_inventory_for_relationship = (
            sensitivity.get("validation_inventory") if sensitivity else None
        )

        if (
            primary_cell_count is None
            or sensitivity_cell_count is None
            or primary_observation_count is None
            or sensitivity_observation_count is None
        ):
            population_relationship = "unavailable"
        elif (
            primary_inventory_for_relationship is not None
            and sensitivity_inventory_for_relationship is not None
            and primary_inventory_for_relationship == sensitivity_inventory_for_relationship
        ):
            population_relationship = "identical_inventory"
        elif (
            primary_inventory_for_relationship is not None
            and sensitivity_inventory_for_relationship is not None
        ):
            population_relationship = "different_inventory"
        else:
            population_relationship = "unavailable"

        records.append({
            "matching_key_identity": matching_key_identity,
            "design_scope": matching_fields_from_key.get("design_scope", ""),
            "sampling_design_scope": matching_fields_from_key.get("sampling_design_scope", ""),
            "aging_mode": matching_fields_from_key.get("aging_mode", ""),
            "outcome": matching_fields_from_key.get("outcome", ""),
            "horizon_axis": matching_fields_from_key.get("horizon_axis", ""),
            "model_name": matching_fields_from_key.get("model_identity", ""),
            "condition_count": len(primary.get("condition_identities", ())) if primary else (
                len(sensitivity.get("condition_identities", ())) if sensitivity else 0
            ),
            "primary_analytical_status": primary["analytical_status"] if primary else "not_available",
            "sensitivity_analytical_status": sensitivity["analytical_status"] if sensitivity else "not_available",
            "primary_validation_status": primary["validation_status"] if primary else "unavailable",
            "sensitivity_validation_status": sensitivity["validation_status"] if sensitivity else "unavailable",
            "primary_record_count": primary_count,
            "sensitivity_record_count": sensitivity_count,
            "primary_validation_cell_count": primary_cell_count,
            "sensitivity_validation_cell_count": sensitivity_cell_count,
            "primary_validation_observation_count": primary_observation_count,
            "sensitivity_validation_observation_count": sensitivity_observation_count,
            "validation_population_count_relationship": population_relationship,
            "primary_root_mean_squared_error": primary_rmse_val,
            "sensitivity_root_mean_squared_error": sensitivity_rmse_val,
            "root_mean_squared_error_difference": (
                sensitivity_rmse_val - primary_rmse_val if metrics_comparable else None
            ),
            "primary_mean_absolute_error": primary_mae_val,
            "sensitivity_mean_absolute_error": sensitivity_mae_val,
            "mean_absolute_error_difference": (
                sensitivity_mae_val - primary_mae_val if metrics_comparable else None
            ),
            "comparison_status": status,
            "comparison_reason_code": reason_code,
            "estimate_label": "descriptive_cohort_sensitivity"
        })

    return records


# Building evidence-backed structural limitation records
def build_structural_limitations(design_tables: dict, data_quality_contract: dict,
                                 estimand_identification_rows: list[dict]
) -> list[dict]:
    analytical_scope = design_tables.get("analytical_scope", [])
    stage_overlap = design_tables.get("stage_overlap", [])
    estimand_support = design_tables.get("estimand_support", [])
    identifiability = design_tables.get("design_identifiability", [])

    def count_stage_batch_aliasing_records() -> int:
        return sum(
            1
            for record in analytical_scope
            if str(
                record.get("stage_batch_relationship_status", "")
            ) == "stage_2_used_a_new_battery_cell_batch_relative_to_stage_1"
        )

    def count_stage_overlap_with_no_cross_stage():
        return sum(
            1 for record in stage_overlap
            if finite_float(record.get("exact_overlap_count")) == 0
        )

    laboratory_matrix_records = [
        record
        for record in identifiability
        if (
            record.get("record_type") == "matrix_summary"
            and record.get("context_family") == "laboratory"
        )
    ]

    laboratory_rank_deficient_matrices = [
        record
        for record in laboratory_matrix_records
        if str(record.get("full_column_rank", "")).lower() == "false"
    ]

    laboratory_term_records = [
        record
        for record in identifiability
        if (
            record.get("record_type") == "term"
            and record.get("context_family") == "laboratory"
        )
    ]

    laboratory_non_estimable_terms = [
        record
        for record in laboratory_term_records
        if str(record.get("structurally_estimable", "")).lower() == "false"
    ]

    def count_unavailable_net_exposure():
        return sum(
            1 for record in estimand_support
            if str(record.get("net_calendar_aging_exposure_status", "")) == "unavailable"
        )

    def count_distinct_exposure_interpretations():
        return len({
            str(record.get("exposure_time_interpretation", ""))
            for record in analytical_scope
            if record.get("exposure_time_interpretation")
        })

    def count_equipment_restricted_estimands():
        return sum(
            1 for record in estimand_identification_rows
            if str(record.get("equipment_restriction_present", "")).strip().lower() == "true"
        )

    ocv_feature_contract = data_quality_contract.get("feature_contract", {}).get("dc_internal_resistance", {})
    ocv_status = str(ocv_feature_contract.get("status", ""))

    limitation_specs = [
        {
            "limitation_identity": "stage_batch_aliasing",
            "limitation_scope": "global_design",
            "declared_evidence_source": "analytical_scope",
            "evidence_record_count": count_stage_batch_aliasing_records(),
            "applicability_status": "unsupported",
            "reason_code": "CROSS_STAGE_TRANSPORTABILITY_NOT_ESTABLISHED",
            "interpretation": "Dataset stages use different cell batches; exchangeability is not established."
        },
        {
            "limitation_identity": "no_exact_cross_stage_combinations",
            "limitation_scope": "global_design",
            "declared_evidence_source": "stage_overlap",
            "evidence_record_count": count_stage_overlap_with_no_cross_stage(),
            "applicability_status": "unsupported",
            "reason_code": "NO_CROSS_STAGE_COMBINATION_OVERLAP",
            "interpretation": "No exact treatment combination occurs in both stages."
        },
        {
            "limitation_identity": "laboratory_rank_deficient_matrices",
            "limitation_scope": "global_design",
            "declared_evidence_source": "design_identifiability",
            "evidence_record_count": len(laboratory_rank_deficient_matrices),
            "applicability_status": "unsupported",
            "reason_code": "LABORATORY_TEST_POINT_ALIASING",
            "interpretation": "Laboratory-context model matrices that are not full column rank cannot separate all laboratory and design terms."
        },
        {
            "limitation_identity": "laboratory_non_estimable_terms",
            "limitation_scope": "global_design",
            "declared_evidence_source": "design_identifiability",
            "evidence_record_count": len(laboratory_non_estimable_terms),
            "applicability_status": "unsupported",
            "reason_code": "LABORATORY_TEST_POINT_ALIASING",
            "interpretation": "Laboratory-context terms that are structurally non-estimable cannot be interpreted as separately identified effects."
        },
        {
            "limitation_identity": "equipment_allocation_restrictions",
            "limitation_scope": "global_design",
            "declared_evidence_source": "estimand_identification",
            "evidence_record_count": count_equipment_restricted_estimands(),
            "applicability_status": "unsupported",
            "reason_code": "RECORDED_EQUIPMENT_ALLOCATION_RESTRICTION",
            "interpretation": "Equipment allocation restrictions are recorded and prevent universal adjustment."
        },
        {
            "limitation_identity": "ocv_corrected_resistance_not_approved",
            "limitation_scope": "global_feature_contract",
            "declared_evidence_source": "data_quality_contract",
            "evidence_record_count": 1 if ocv_status != "approved" else 0,
            "applicability_status": "not_authorized",
            "reason_code": "OCV_CORRECTED_RESISTANCE_PROVISIONAL",
            "interpretation": "OCV-corrected resistance requires capacity-based normalization before use as a primary outcome."
        },
        {
            "limitation_identity": "net_storage_exposure_unavailable",
            "limitation_scope": "calendar_estimands",
            "declared_evidence_source": "estimand_support",
            "evidence_record_count": count_unavailable_net_exposure(),
            "applicability_status": "unavailable",
            "reason_code": "NET_STORAGE_EXPOSURE_UNAVAILABLE",
            "interpretation": "Gross whole-protocol elapsed time is available; net storage exposure is unavailable."
        },
        {
            "limitation_identity": "elapsed_and_fec_axes_distinct",
            "limitation_scope": "global_design",
            "declared_evidence_source": "analytical_scope",
            "evidence_record_count": count_distinct_exposure_interpretations(),
            "applicability_status": "unsupported",
            "reason_code": "HORIZON_AXES_NOT_INTERCHANGEABLE",
            "interpretation": "Elapsed-time and cumulative-FEC axes are distinct and not interchangeable."
        }
    ]

    records = []

    for spec in limitation_specs:
        evidence_count = spec.pop("evidence_record_count")
        declared_source = spec.pop("declared_evidence_source")

        if evidence_count > 0:
            evidence_source = declared_source
        else:
            evidence_source = "registered_policy_restriction"

        records.append({
            **spec,
            "evidence_source": evidence_source,
            "evidence_record_count": evidence_count
        })

    return records


# Summarizing registered hyperparameter-selection frequencies per configuration
def build_configuration_boundary_summary(predictive_tables: dict, predictive_policy: dict) -> list[dict]:
    selection_records = predictive_tables["hyperparameter_selection"]

    registered_identities_by_family: dict[str, set[str]] = {}

    for family_name, family_policy in predictive_policy["model_families"].items():
        identities = set()

        for configuration in family_policy["configurations"]:
            identity = hashlib.sha256(
                canonical_json({
                    "model_family": family_name,
                    "configuration": configuration
                }).encode("utf-8")
            ).hexdigest()

            if identity in identities:
                raise RuntimeError(
                    f"Predictive policy contains a duplicate configuration identity : {family_name} : {identity}"
                )

            identities.add(identity)

        registered_identities_by_family[family_name] = identities

    for record in selection_records:
        family = record["model_family"]

        identity = record["configuration_identity"]

        if family not in registered_identities_by_family:
            raise RuntimeError(f"Observed model family is not registered in the predictive policy : {family}")

        if identity not in registered_identities_by_family[family]:
            raise RuntimeError(
                f"Observed configuration identity is not registered in the predictive policy : {family} : {identity}"
            )

        observed_configuration = record["configuration"]

        recomputed_identity = hashlib.sha256(
            canonical_json({
                "model_family": family,
                "configuration": observed_configuration
            }).encode("utf-8")
        ).hexdigest()

        if recomputed_identity != identity:
            raise RuntimeError(f"Observed configuration value does not match its registered identity : {family} : {identity}")

    selection_counts = Counter()
    model_family_by_configuration = {}
    configuration_by_identity = {}
    distinct_outer_folds_by_family = defaultdict(set)
    selected_configurations_by_family = defaultdict(set)

    for record in selection_records:
        model_family = record["model_family"]
        configuration_identity = record["configuration_identity"]

        configuration_by_identity[configuration_identity] = record["configuration"]
        model_family_by_configuration[configuration_identity] = model_family

        if str(record["selected"]).lower() == "true":
            selection_counts[(model_family, configuration_identity)] += 1
            selected_configurations_by_family[model_family].add(configuration_identity)

        candidate_identity = str(record.get("candidate_identity", ""))
        outer_fold_index = str(record.get("outer_fold_index", ""))

        if candidate_identity and outer_fold_index != "":
            distinct_outer_folds_by_family[model_family].add(
                (candidate_identity, outer_fold_index)
            )

    total_folds_by_family = {family: len(folds) for family, folds in distinct_outer_folds_by_family.items()}

    registered_configurations_by_family = {}
    registered_parameters_by_family = {}

    for family_name, family_policy in predictive_policy["model_families"].items():
        registrations = []

        for config_dict in family_policy["configurations"]:
            identity = hashlib.sha256(
                canonical_json({"model_family": family_name, "configuration": config_dict}).encode("utf-8")
            ).hexdigest()

            registrations.append({
                "configuration_identity": identity,
                "configuration": config_dict
            })

        registered_configurations_by_family[family_name] = registrations

        parameter_values = defaultdict(set)

        for config_dict in family_policy["configurations"]:
            for parameter_name, parameter_value in config_dict.items():
                try:
                    parameter_values[parameter_name].add((str(parameter_value), parameter_value))
                except TypeError:
                    parameter_values[parameter_name].add((str(parameter_value), repr(parameter_value)))

        registered_parameters_by_family[family_name] = {
            name: sorted(entries, key = lambda item: item[0])
            for name, entries in parameter_values.items()
        }

    records = []

    for family_name in sorted(registered_configurations_by_family):
        registrations = registered_configurations_by_family[family_name]
        total = total_folds_by_family.get(family_name, 0)

        for registration in registrations:
            identity = registration["configuration_identity"]
            configuration = registration["configuration"]

            count = selection_counts.get((family_name, identity), 0)
            fraction = (count / total) if total else 0.0
            selected_any = identity in selected_configurations_by_family.get(family_name, set())

            boundary_parameter_names = []
            informative_parameter_names = []
            uninformative_parameter_names = []

            for parameter_name, selected_value in configuration.items():
                registered_entries = registered_parameters_by_family[family_name].get(parameter_name, [])

                registered_values = [entry[1] for entry in registered_entries]

                if len(registered_values) < 2:
                    continue

                try:
                    sorted_values = sorted(registered_values)
                except TypeError:
                    uninformative_parameter_names.append(parameter_name)
                    continue

                if len(sorted_values) < 3:
                    uninformative_parameter_names.append(parameter_name)
                    continue

                informative_parameter_names.append(parameter_name)

                if selected_value in {sorted_values[0], sorted_values[-1]}:
                    boundary_parameter_names.append(parameter_name)

            boundary_parameter_names.sort()
            informative_parameter_names.sort()
            uninformative_parameter_names.sort()

            if not informative_parameter_names:
                boundary_assessment_status = "uninformative"
                boundary_classification = "boundary_assessment_uninformative"
                on_registered_search_boundary = None
            elif not selected_any:
                boundary_assessment_status = "not_applicable"
                boundary_classification = "configuration_not_selected"
                on_registered_search_boundary = None
            elif boundary_parameter_names:
                boundary_assessment_status = "completed"
                boundary_classification = "selected_registered_search_boundary"
                on_registered_search_boundary = True
            else:
                boundary_assessment_status = "completed"
                boundary_classification = "selected_interior"
                on_registered_search_boundary = False

            records.append({
                "model_family": family_name,
                "configuration_identity": identity,
                "configuration": configuration,
                "selection_count": count,
                "total_outer_folds": total,
                "selection_fraction": float(fraction),
                "boundary_classification": boundary_classification,
                "boundary_assessment_status": boundary_assessment_status,
                "selected_on_any_outer_fold": selected_any,
                "boundary_parameter_names": boundary_parameter_names,
                "boundary_parameter_count": len(boundary_parameter_names),
                "informative_parameter_names": informative_parameter_names,
                "uninformative_parameter_names": uninformative_parameter_names,
                "on_registered_search_boundary": on_registered_search_boundary,
                "estimate_label": "configuration_selection_summary"
            })

    records.sort(key = lambda r: (r["model_family"], r["configuration_identity"]))

    return records


# Building descriptive-template specification comparisons with analytical-status propagation
def build_descriptive_specification_comparisons(descriptive_tables: dict) -> list[dict]:
    model_comparisons = descriptive_tables.get("model_comparisons", [])

    registry = {
        record["candidate_identity"]: record
        for record in descriptive_tables["candidate_registry"]
    }

    records = []

    for comparison in model_comparisons:
        first_identity = comparison["first_candidate_identity"]
        second_identity = comparison["second_candidate_identity"]

        first_candidate = registry.get(first_identity)
        second_candidate = registry.get(second_identity)

        first_analytical = (
            normalize_status(first_candidate["analytical_result_status"])
            if first_candidate is not None
            else "not_available"
        )

        second_analytical = (
            normalize_status(second_candidate["analytical_result_status"])
            if second_candidate is not None
            else "not_available"
        )

        upstream_status = comparison["comparison_status"]

        first_rmse = finite_float(comparison.get("first_root_mean_squared_error"))
        second_rmse = finite_float(comparison.get("second_root_mean_squared_error"))
        first_mae = finite_float(comparison.get("first_mean_absolute_error"))
        second_mae = finite_float(comparison.get("second_mean_absolute_error"))

        metrics_finite = all(
            value is not None
            for value in (first_rmse, second_rmse, first_mae, second_mae)
        )

        if upstream_status != "completed" or not metrics_finite:
            comparison_status = "unavailable"
            comparison_reason_code = (
                comparison.get("comparison_reason_code")
                or "VALIDATION_METRICS_UNAVAILABLE"
            )
        elif first_analytical != "available" or second_analytical != "available":
            comparison_status = "diagnostic_only"
            comparison_reason_code = "SOURCE_MODEL_NOT_ANALYTICALLY_AVAILABLE"
        else:
            comparison_status = "completed"
            comparison_reason_code = None

        comparison_identity = hashlib.sha256(
            canonical_json({
                "first_candidate_identity": first_identity,
                "second_candidate_identity": second_identity,
                "specification": comparison.get("specification", ""),
                "context_family": comparison.get("context_family", "")
            }).encode("utf-8")
        ).hexdigest()

        records.append({
            "comparison_identity": comparison_identity,
            "design_scope": comparison["design_scope"],
            "sampling_design_scope": comparison["sampling_design_scope"],
            "aging_mode": comparison["aging_mode"],
            "outcome": comparison["outcome"],
            "cohort": comparison["cohort"],
            "horizon_axis": comparison["horizon_axis"],
            "specification": comparison.get("specification", ""),
            "context_family": comparison.get("context_family", ""),
            "first_candidate_identity": first_identity,
            "second_candidate_identity": second_identity,
            "first_template_name": comparison["first_template_name"],
            "second_template_name": comparison["second_template_name"],
            "first_analytical_status": first_analytical,
            "second_analytical_status": second_analytical,
            "first_mean_absolute_error": first_mae,
            "second_mean_absolute_error": second_mae,
            "mean_absolute_error_difference": (
                None if not metrics_finite else second_mae - first_mae
            ),
            "first_root_mean_squared_error": first_rmse,
            "second_root_mean_squared_error": second_rmse,
            "root_mean_squared_error_difference": (
                None if not metrics_finite else second_rmse - first_rmse
            ),
            "comparison_status": comparison_status,
            "comparison_reason_code": comparison_reason_code,
            "estimate_label": "descriptive_specification_robustness"
        })

    return records


# Extracting a list of estimand decision identities from a candidate field
def _extract_estimand_identities(value: object) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v) for v in value]

    if isinstance(value, str):
        if not value.strip():
            return []

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []

        if isinstance(parsed, list):
            return [str(v) for v in parsed]

    return []


# Building an exact estimand-decision → equipment-restriction lookup
def _build_equipment_restriction_lookup(estimand_identification_rows: list[dict]) -> dict[str, bool]:
    lookup: dict[str, bool] = {}

    for row in estimand_identification_rows:
        estimand_identity = row.get("estimand_decision_identity")

        if not isinstance(estimand_identity, str) or not estimand_identity:
            continue

        observed_value = str(row.get("equipment_restriction_present", "")).strip().lower()

        if observed_value == "true":
            lookup[estimand_identity] = True
        elif observed_value == "false":
            lookup[estimand_identity] = False

    return lookup


# Resolving a candidate's equipment restriction state from its estimand identities
def _resolve_equipment_state(estimand_identities: list[str], equipment_lookup: dict[str, bool]) -> bool | None:
    observed_states = [equipment_lookup[i] for i in estimand_identities if i in equipment_lookup]

    if not observed_states:
        return None

    return any(observed_states)


# Deriving direct condition-calibration state from its table
def build_condition_calibration_states(predictive_tables: dict) -> dict[str, bool | None]:
    candidates = {
        record[
            "benchmark_identity"
        ]: record
        for record in predictive_tables[
            "candidate_registry"
        ]
    }

    records_by_benchmark = defaultdict(
        dict
    )

    for record in predictive_tables.get(
        "condition_calibration",
        []
    ):
        benchmark_identity = record[
            "benchmark_identity"
        ]

        condition_identity = record[
            "condition_protocol_identity"
        ]

        if condition_identity in records_by_benchmark[
            benchmark_identity
        ]:
            raise RuntimeError(
                "Condition calibration contains a duplicate benchmark-condition record : " \
                f"{benchmark_identity} : {condition_identity}"
            )

        records_by_benchmark[
            benchmark_identity
        ][
            condition_identity
        ] = record

    states = {}

    for benchmark_identity, candidate in candidates.items():
        expected_conditions = set(
            candidate[
                "condition_protocol_identities"
            ]
        )

        observed_records = records_by_benchmark.get(
            benchmark_identity,
            {}
        )

        observed_conditions = set(
            observed_records
        )

        if observed_conditions != expected_conditions:
            raise RuntimeError(
                f"Condition calibration inventory does not match the benchmark conditions : {benchmark_identity}"
            )

        statuses = {
            record[
                "calibration_status"
            ]
            for record in observed_records.values()
        }

        if not statuses:
            states[
                benchmark_identity
            ] = None

        elif statuses == {
            "completed"
        }:
            states[
                benchmark_identity
            ] = True

        elif statuses.issubset({
            "completed",
            "incomplete",
            "unavailable"
        }):
            states[
                benchmark_identity
            ] = False

        else:
            raise RuntimeError(
                f"Condition calibration contains an unsupported status : {benchmark_identity} : {sorted(statuses)}"
            )

    return states


# Building per-candidate and per-benchmark fragility indicators without binary classification
def build_fragility_registry(descriptive_tables: dict, bayesian_tables: dict,
                             predictive_tables: dict, estimand_identification_rows: list[dict],
                             tables: dict
) -> list[dict]:
    equipment_lookup = _build_equipment_restriction_lookup(estimand_identification_rows)

    calibration_state_by_benchmark = (
        build_condition_calibration_states(
            predictive_tables
        )
    )

    prior_complete_by_candidate = {}

    for row in tables.get("prior_sensitivity", []):
        candidate_identity = row["candidate_identity"]
        this_complete = row["assessment_status"] == "completed"
        current = prior_complete_by_candidate.get(candidate_identity)

        if current is None:
            prior_complete_by_candidate[candidate_identity] = this_complete
        else:
            prior_complete_by_candidate[candidate_identity] = current and this_complete

    cell_deletion_by_candidate = {
        row["candidate_identity"]: (row["assessment_status"] == "completed")
        for row in tables.get("influence_summary", [])
    }

    predictive_registry_by_identity = {
        record["benchmark_identity"]: record
        for record in predictive_tables["candidate_registry"]
    }

    benchmark_by_candidate_and_family = {}

    for benchmark_identity, candidate in predictive_registry_by_identity.items():
        lookup_key = (
            candidate["candidate_identity"],
            candidate["model_family"]
        )

        if lookup_key in benchmark_by_candidate_and_family:
            raise RuntimeError(f"Predictive registry contains a duplicate candidate-family mapping : {lookup_key}")

        benchmark_by_candidate_and_family[lookup_key] = benchmark_identity

    cross_framework_by_benchmark = {}

    for row in tables.get("prediction_agreement", []):
        lookup_key = (
            row["candidate_identity"],
            row["model_family"]
        )

        benchmark_identity = benchmark_by_candidate_and_family.get(lookup_key)

        if benchmark_identity is None:
            raise RuntimeError(f"Prediction-agreement record does not map to a predictive benchmark : {lookup_key}")

        cross_framework_by_benchmark[benchmark_identity] = {
            "inventory_complete": row["exact_inventory_match"] is True,
            "source_models_available": (
                row["bayesian_analytical_status"] == "available"
                and row["predictive_analytical_status"] == "available"
            )
        }

    predictive_metrics_by_benchmark = {
        record["benchmark_identity"]: record
        for record in predictive_tables["validation_metrics"]
    }

    predictive_benchmark_met_by_identity = {}

    for benchmark_identity, candidate in predictive_registry_by_identity.items():
        metric = predictive_metrics_by_benchmark.get(benchmark_identity)

        if metric is None or metric["validation_status"] != "completed":
            predictive_benchmark_met_by_identity[benchmark_identity] = None
            continue

        r_squared = finite_float(
            metric.get("equal_cell_weighted_predictive_r_squared")
        )

        predictive_benchmark_met_by_identity[benchmark_identity] = (
            r_squared is not None and r_squared > 0
        )

    descriptive_validation_by_candidate = defaultdict(bool)

    for metric in descriptive_tables["validation_metrics"]:
        if metric["validation_status"] == "completed":
            descriptive_validation_by_candidate[metric["candidate_identity"]] = True

    bayesian_validation_by_candidate = defaultdict(list)

    for metric in bayesian_tables["grouped_validation_metrics"]:
        bayesian_validation_by_candidate[metric["candidate_identity"]].append(
            metric["validation_status"]
        )

    bayesian_equipment_state_by_candidate_identity = {}

    for candidate in bayesian_tables["candidate_registry"]:
        if candidate.get("admissibility_status") != "admissible":
            continue

        estimand_identities = _extract_estimand_identities(
            candidate.get("sorted_estimand_decision_identities")
        )

        bayesian_equipment_state_by_candidate_identity[candidate["candidate_identity"]] = (
            _resolve_equipment_state(estimand_identities, equipment_lookup)
        )

    records = []

    def finalize_row(row: dict) -> dict:
        applicable_indicators = row["applicable_indicators"]
        fragility_indicators = row["fragility_indicators"]

        if len(applicable_indicators) != len(set(applicable_indicators)):
            raise RuntimeError(
                "Fragility record contains duplicate applicable indicators : "
                f"{row['source_analysis']} : {row['candidate_or_benchmark_identity']}"
            )

        if len(fragility_indicators) != len(set(fragility_indicators)):
            raise RuntimeError(
                "Fragility record contains duplicate fragility indicators : "
                f"{row['source_analysis']} : {row['candidate_or_benchmark_identity']}"
            )

        row["applicable_indicators"] = sorted(applicable_indicators)
        row["fragility_indicators"] = sorted(fragility_indicators)
        row["fragility_indicator_count"] = len(fragility_indicators)
        return row

    for candidate in descriptive_tables["candidate_registry"]:
        if candidate.get("final_disposition") != "fitted":
            continue

        candidate_identity = candidate["candidate_identity"]
        analytical_status = normalize_status(candidate.get("analytical_result_status"))
        validation_completed = descriptive_validation_by_candidate.get(candidate_identity, False)
        cell_deletion_complete = cell_deletion_by_candidate.get(candidate_identity)

        applicable = [
            "validation_available",
            "analytical_status",
            "cell_deletion_refit_complete"
        ]
        indicators = []

        if not validation_completed:
            indicators.append("validation_unavailable")

        if analytical_status == "diagnostic_only":
            indicators.append("analytical_diagnostic_only")
        elif analytical_status == "not_available":
            indicators.append("analytical_not_available")

        if cell_deletion_complete is False:
            indicators.append("cell_deletion_refit_incomplete")

        fragility_scope_identity = hashlib.sha256(
            canonical_json({
                "source_analysis": "descriptive_modeling",
                "candidate_identity": candidate_identity
            }).encode("utf-8")
        ).hexdigest()

        records.append(finalize_row({
            "fragility_scope_identity": fragility_scope_identity,
            "source_analysis": "descriptive_modeling",
            "candidate_or_benchmark_identity": candidate_identity,
            "design_scope": candidate.get("design_scope", ""),
            "sampling_design_scope": candidate.get("sampling_design_scope", ""),
            "aging_mode": candidate.get("aging_mode", ""),
            "outcome": candidate.get("outcome", ""),
            "cohort": candidate.get("cohort", ""),
            "horizon_axis": candidate.get("horizon_axis", ""),
            "validation_available": validation_completed,
            "analytical_status": analytical_status,
            "prior_sensitivity_complete": None,
            "cell_deletion_refit_complete": cell_deletion_complete,
            "condition_calibration_complete": None,
            "predictive_benchmark_met": None,
            "cross_framework_inventory_complete": None,
            "cross_framework_source_models_available": None,
            "equipment_restriction_present": None,
            "sensitivity_cohort": candidate.get("cohort") == "sensitivity",
            "applicable_indicators": applicable,
            "fragility_indicators": indicators,
            "fragility_indicator_count": 0,
            "estimate_label": "fragility_inspection"
        }))

    for candidate in bayesian_tables["candidate_registry"]:
        if candidate.get("admissibility_status") != "admissible":
            continue

        candidate_identity = candidate["candidate_identity"]
        analytical_status = normalize_status(candidate.get("posterior_analytical_status"))
        validation_statuses = bayesian_validation_by_candidate.get(candidate_identity, [])
        validation_completed = (
            len(validation_statuses) > 0
            and all(status == "completed" for status in validation_statuses)
        )
        prior_complete = prior_complete_by_candidate.get(candidate_identity)
        equipment_state = bayesian_equipment_state_by_candidate_identity.get(candidate_identity)

        applicable = [
            "validation_available",
            "analytical_status",
            "prior_sensitivity_complete"
        ]
        indicators = []

        if not validation_completed:
            indicators.append("validation_unavailable")

        if analytical_status == "diagnostic_only":
            indicators.append("analytical_diagnostic_only")
        elif analytical_status == "not_available":
            indicators.append("analytical_not_available")

        if prior_complete is False:
            indicators.append("prior_sensitivity_incomplete")

        if equipment_state is not None:
            applicable.append("equipment_restriction_present")
            if equipment_state:
                indicators.append("equipment_restriction_present")

        fragility_scope_identity = hashlib.sha256(
            canonical_json({
                "source_analysis": "bayesian_modeling",
                "candidate_identity": candidate_identity
            }).encode("utf-8")
        ).hexdigest()

        records.append(finalize_row({
            "fragility_scope_identity": fragility_scope_identity,
            "source_analysis": "bayesian_modeling",
            "candidate_or_benchmark_identity": candidate_identity,
            "design_scope": candidate.get("design_scope", ""),
            "sampling_design_scope": candidate.get("sampling_design_scope", ""),
            "aging_mode": candidate.get("aging_mode", ""),
            "outcome": candidate.get("outcome", ""),
            "cohort": candidate.get("cohort", ""),
            "horizon_axis": candidate.get("horizon_axis", ""),
            "validation_available": validation_completed,
            "analytical_status": analytical_status,
            "prior_sensitivity_complete": prior_complete,
            "cell_deletion_refit_complete": None,
            "condition_calibration_complete": None,
            "predictive_benchmark_met": None,
            "cross_framework_inventory_complete": None,
            "cross_framework_source_models_available": None,
            "equipment_restriction_present": equipment_state,
            "sensitivity_cohort": candidate.get("cohort") == "sensitivity",
            "applicable_indicators": applicable,
            "fragility_indicators": indicators,
            "fragility_indicator_count": 0,
            "estimate_label": "fragility_inspection"
        }))

    for candidate in predictive_tables["candidate_registry"]:
        benchmark_identity = candidate["benchmark_identity"]
        analytical_status = normalize_status(candidate.get(
            "analytical_status"
        ))
        metric = predictive_metrics_by_benchmark.get(
            benchmark_identity
        )
        validation_completed = (
            metric is not None
            and metric["validation_status"] == "completed"
        )
        predictive_met = predictive_benchmark_met_by_identity.get(
            benchmark_identity
        )
        cross_framework_state = cross_framework_by_benchmark.get(
            benchmark_identity
        )

        cross_framework_inventory_complete = (
            cross_framework_state["inventory_complete"]
            if cross_framework_state is not None
            else None
        )

        cross_framework_source_models_available = (
            cross_framework_state["source_models_available"]
            if cross_framework_state is not None
            else None
        )

        condition_calibration_complete = (
            calibration_state_by_benchmark.get(
                benchmark_identity
            )
        )
        equipment_state = bayesian_equipment_state_by_candidate_identity.get(
            candidate.get(
                "candidate_identity"
            )
        )

        applicable = [
            "validation_available",
            "analytical_status",
            "condition_calibration_complete",
            "predictive_benchmark_met"
        ]
        indicators = []

        if not validation_completed:
            indicators.append("validation_unavailable")

        if analytical_status == "diagnostic_only":
            indicators.append("analytical_diagnostic_only")
        elif analytical_status == "not_available":
            indicators.append("analytical_not_available")

        if condition_calibration_complete is False:
            indicators.append("condition_calibration_incomplete")

        if predictive_met is False:
            indicators.append("predictive_benchmark_not_met")

        if cross_framework_state is not None:
            applicable.extend([
                "cross_framework_inventory_complete",
                "cross_framework_source_models_available"
            ])

        if cross_framework_inventory_complete is False:
            indicators.append("cross_framework_inventory_incomplete")

        if cross_framework_source_models_available is False:
            indicators.append("cross_framework_source_model_unavailable")

        if equipment_state is not None:
            applicable.append("equipment_restriction_present")
            if equipment_state:
                indicators.append("equipment_restriction_present")

        fragility_scope_identity = hashlib.sha256(
            canonical_json({
                "source_analysis": "predictive_modeling",
                "benchmark_identity": benchmark_identity
            }).encode("utf-8")
        ).hexdigest()

        records.append(finalize_row({
            "fragility_scope_identity": fragility_scope_identity,
            "source_analysis": "predictive_modeling",
            "candidate_or_benchmark_identity": benchmark_identity,
            "design_scope": candidate.get("design_scope", ""),
            "sampling_design_scope": candidate.get("sampling_design_scope", ""),
            "aging_mode": candidate.get("aging_mode", ""),
            "outcome": candidate.get("outcome", ""),
            "cohort": candidate.get("cohort", ""),
            "horizon_axis": candidate.get("horizon_axis", ""),
            "validation_available": validation_completed,
            "analytical_status": analytical_status,
            "prior_sensitivity_complete": None,
            "cell_deletion_refit_complete": None,
            "condition_calibration_complete": condition_calibration_complete,
            "predictive_benchmark_met": predictive_met,
            "cross_framework_inventory_complete": cross_framework_inventory_complete,
            "cross_framework_source_models_available": cross_framework_source_models_available,
            "equipment_restriction_present": equipment_state,
            "sensitivity_cohort": candidate.get("cohort") == "sensitivity",
            "applicable_indicators": applicable,
            "fragility_indicators": indicators,
            "fragility_indicator_count": 0,
            "estimate_label": "fragility_inspection"
        }))

    records.sort(key = lambda record: (
        record["source_analysis"],
        record["candidate_or_benchmark_identity"]
    ))

    return records


# Building an inventory of unsupported condition-benchmark combinations
def build_unsupported_condition_inventory(predictive_tables: dict) -> list[dict]:
    registry = {
        record[
            "benchmark_identity"
        ]: record
        for record in predictive_tables[
            "candidate_registry"
        ]
    }

    records = []

    for calibration in predictive_tables[
        "condition_calibration"
    ]:
        if calibration[
            "calibration_status"
        ] != "unavailable":
            continue

        benchmark_identity = calibration[
            "benchmark_identity"
        ]

        candidate = registry[
            benchmark_identity
        ]

        records.append({
            "benchmark_identity": benchmark_identity,
            "candidate_identity": candidate[
                "candidate_identity"
            ],
            "model_family": candidate[
                "model_family"
            ],
            "design_scope": candidate[
                "design_scope"
            ],
            "sampling_design_scope": candidate[
                "sampling_design_scope"
            ],
            "aging_mode": candidate[
                "aging_mode"
            ],
            "outcome": candidate[
                "outcome"
            ],
            "cohort": candidate[
                "cohort"
            ],
            "horizon_axis": candidate[
                "horizon_axis"
            ],
            "condition_protocol_identity": calibration[
                "condition_protocol_identity"
            ],
            "calibration_status": calibration[
                "calibration_status"
            ],
            "calibration_reason_code": calibration[
                "calibration_reason_code"
            ],
            "estimate_label": "predictive_support_restriction"
        })

    records.sort(
        key = lambda record: (
            record[
                "benchmark_identity"
            ],
            record[
                "condition_protocol_identity"
            ]
        )
    )

    return records


# Building the reconciled predictive-sensitivity summary from the registered sensitivity tables
def build_predictive_sensitivity_summary(sensitivity_tables: dict) -> list[dict]:
    def unique_lookup(records: list[dict], table_name: str) -> dict[str, dict]:
        lookup = {}

        for record in records:
            variant_identity = str(
                record["variant_identity"]
            )

            if variant_identity in lookup:
                raise RuntimeError(
                    f"Predictive-sensitivity table contains a duplicate variant identity : {table_name} : {variant_identity}"
                )

            lookup[variant_identity] = record

        return lookup

    registry = unique_lookup(
        sensitivity_tables["sensitivity_registry"],
        "sensitivity_registry"
    )

    metrics = unique_lookup(
        sensitivity_tables["validation_metrics"],
        "validation_metrics"
    )

    comparisons = unique_lookup(
        sensitivity_tables["sensitivity_comparisons"],
        "sensitivity_comparisons"
    )

    overall_interval_records = [
        record
        for record in sensitivity_tables[
            "prediction_interval_diagnostics"
        ]
        if record["grouping_dimension"] == "overall"
    ]

    intervals = unique_lookup(
        overall_interval_records,
        "prediction_interval_diagnostics.overall"
    )

    identity_sets = {
        "registry": set(registry),
        "metrics": set(metrics),
        "comparisons": set(comparisons),
        "intervals": set(intervals)
    }

    if len({
        frozenset(identities)
        for identities in identity_sets.values()
    }) != 1:
        raise RuntimeError(
            "Predictive-sensitivity registry, metric, comparison and overall-interval inventories do not reconcile."
        )

    ablation_summaries = unique_lookup(
        sensitivity_tables["predictor_ablation"],
        "predictor_ablation"
    )

    incident_summaries = unique_lookup(
        sensitivity_tables["incident_exclusions"],
        "incident_exclusions"
    )

    expected_ablation_identities = {
        variant_identity
        for variant_identity, record in registry.items()
        if record["variant_type"] == "ablation"
    }

    expected_incident_identities = {
        variant_identity
        for variant_identity, record in registry.items()
        if record["variant_type"] == "incident"
    }

    if set(ablation_summaries) != expected_ablation_identities:
        raise RuntimeError(f"Predictor-ablation summary inventory does not reconcile with the registered ablation variants.")

    if set(incident_summaries) != expected_incident_identities:
        raise RuntimeError("Incident-exclusion summary inventory does not reconcile with the registered incident variants.")

    for variant_identity, source_record in (
        ablation_summaries.items()
    ):
        if (
            source_record["comparison_status"]
            != comparisons[variant_identity]["comparison_status"]
            or source_record["variant_analytical_status"]
            != metrics[variant_identity]["analytical_status"]
        ):
            raise RuntimeError(f"Predictor-ablation status does not reconcile : {variant_identity}")

    for variant_identity, source_record in (
        incident_summaries.items()
    ):
        if (
            source_record["comparison_status"]
            != comparisons[variant_identity]["comparison_status"]
            or source_record["variant_analytical_status"]
            != metrics[variant_identity]["analytical_status"]
        ):
            raise RuntimeError(f"Incident-exclusion status does not reconcile : {variant_identity}")

        if (
            source_record["eligible_in_every_fold"] is False
            and source_record["comparison_status"] != "unavailable"
        ):
            raise RuntimeError(f"Ineligible incident exclusion must remain unavailable : {variant_identity}")

    rows = []

    for variant_identity in sorted(registry):
        registry_record = registry[variant_identity]
        metric_record = metrics[variant_identity]
        comparison_record = comparisons[variant_identity]
        interval_record = intervals[variant_identity]

        if (
            registry_record["variant_type"]
            != metric_record["variant_type"]
            or registry_record["variant_type"]
            != comparison_record["variant_type"]
        ):
            raise RuntimeError(f"Predictive-sensitivity variant type does not reconcile : {variant_identity}")

        comparison_status = comparison_record[
            "comparison_status"
        ]

        if comparison_status == "completed":
            assessment_status = "completed"
            assessment_reason_code = None

        elif comparison_status == "diagnostic_only":
            assessment_status = "diagnostic_only"
            assessment_reason_code = (
                "SENSITIVITY_VARIANT_DIAGNOSTIC_ONLY"
            )

        elif comparison_status == "unavailable":
            assessment_status = "unavailable"
            assessment_reason_code = (
                "SENSITIVITY_VARIANT_UNAVAILABLE"
            )

        else:
            raise RuntimeError(
                f"Predictive-sensitivity comparison uses an unsupported status : {variant_identity} : {comparison_status}"
            )

        rows.append({
            "variant_identity": variant_identity,
            "variant_type": registry_record["variant_type"],
            "candidate_identity": registry_record["candidate_identity"],
            "model_family": registry_record["model_family"],
            "design_scope": metric_record["design_scope"],
            "sampling_design_scope": metric_record[
                "sampling_design_scope"
            ],
            "aging_mode": metric_record["aging_mode"],
            "outcome": metric_record["outcome"],
            "cohort": metric_record["cohort"],
            "horizon_axis": metric_record["horizon_axis"],
            "partition_identity": registry_record[
                "partition_identity"
            ],
            "configuration_inventory": registry_record[
                "configuration_inventory"
            ],
            "training_window_identity": registry_record[
                "training_window_identity"
            ],
            "incident_reason_code": registry_record[
                "incident_reason_code"
            ],
            "ablation_group": registry_record["ablation_group"],
            "validation_status": metric_record["validation_status"],
            "validation_reason_code": metric_record[
                "validation_reason_code"
            ],
            "analytical_status": metric_record["analytical_status"],
            "analytical_reason_code": metric_record[
                "analytical_reason_code"
            ],
            "comparison_status": comparison_status,
            "comparison_reason_code": comparison_record[
                "comparison_reason_code"
            ],
            "exact_inventory_match": comparison_record[
                "exact_inventory_match"
            ],
            "shared_observation_count": comparison_record[
                "shared_observation_count"
            ],
            "shared_independent_cell_count": comparison_record[
                "shared_independent_cell_count"
            ],
            "excluded_test_observation_count": comparison_record[
                "excluded_test_observation_count"
            ],
            "root_mean_squared_error_difference": comparison_record[
                "root_mean_squared_error_difference"
            ],
            "mean_absolute_error_difference": comparison_record[
                "mean_absolute_error_difference"
            ],
            "predictive_r_squared_difference": comparison_record[
                "predictive_r_squared_difference"
            ],
            "overall_interval_status": interval_record[
                "assessment_status"
            ],
            "overall_interval_reason_code": interval_record[
                "assessment_reason_code"
            ],
            "overall_interval_prediction_count": interval_record[
                "prediction_count"
            ],
            "overall_interval_independent_cell_count": interval_record[
                "independent_cell_count"
            ],
            "overall_interval_coverage": interval_record[
                "equal_cell_weighted_interval_coverage"
            ],
            "overall_interval_width": interval_record[
                "equal_cell_weighted_mean_interval_width"
            ],
            "assessment_status": assessment_status,
            "assessment_reason_code": assessment_reason_code,
            "estimate_label": "predictive_sensitivity"
        })

    return rows


# Building one output table manifest record
def table_manifest(path: Path, records: list[dict], fields: list[str]) -> dict:
    return {
        "file_name": path.name,
        "row_count": len(
            records
        ),
        "sha256": calculate_file_sha256(
            path
        ),
        "schema": [
            {
                "name": field_name
            }
            for field_name in fields
        ]
    }


# Canonicalizing loaded CSV rows for content comparison against the computed records
def csv_canonical_rows(rows: list[dict]) -> list[dict]:
    canonicalized = []

    for row in rows:
        canonicalized.append(
            {
                field_name: str(
                    serialize_csv_value(
                        row[
                            field_name
                        ]
                    )
                )
                for field_name in sorted(
                    row
                )
            }
        )

    return canonicalized


# Validating a published robustness artifact directory against the registered policy and computed records
def validate_published_robustness_directory(published_directory: Path, expected_policy: dict,
                                            expected_policy_sha256: str, expected_input_hashes: dict,
                                            expected_tables: dict[str, list[dict]], expected_assessment_status: str,
                                            expected_assessment_status_counts: dict,
                                            expected_prediction_inventory_reconciliation: dict
) -> None:
    report_path = published_directory / expected_policy["output_contract"]["report_file"]

    if not report_path.is_file():
        raise RuntimeError("Published robustness report is absent.")

    report = load_json_object(report_path)

    required_report_fields = {
        "execution_status",
        "scientific_assessment_status",
        "policy_name",
        "policy_version",
        "policy_sha256",
        "implementation_sha256",
        "input_hashes",
        "causal_interpretation_authorized",
        "policy_recommendation_authorized",
        "performance_selected_model_winner_declared",
        "magnitude_dependent_robustness_classification_used",
        "registration_status",
        "amendment",
        "domain_status_counts",
        "prediction_inventory_reconciliation",
        "analytical_handoff",
        "tables"
    }

    if set(report) != required_report_fields:
        raise RuntimeError("Published robustness report has an invalid schema.")

    if report["policy_name"] != expected_policy["policy_name"]:
        raise RuntimeError("Published robustness report policy_name is invalid.")

    if report["policy_version"] != expected_policy["policy_version"]:
        raise RuntimeError("Published robustness report policy_version is invalid.")

    if report["policy_sha256"] != expected_policy_sha256:
        raise RuntimeError("Published robustness report policy_sha256 is invalid.")

    if report["input_hashes"] != expected_input_hashes:
        raise RuntimeError("Published robustness report input_hashes are invalid.")

    if (
        report["implementation_sha256"]
        != calculate_file_sha256(Path(__file__).resolve())
    ):
        raise RuntimeError("Published robustness report implementation_sha256 is invalid.")

    for boolean_field in (
        "causal_interpretation_authorized",
        "policy_recommendation_authorized",
        "performance_selected_model_winner_declared",
        "magnitude_dependent_robustness_classification_used"
    ):
        if report[boolean_field] is not False:
            raise RuntimeError(f"Published robustness report incorrectly sets : {boolean_field}")

    if report["execution_status"] != "completed":
        raise RuntimeError("Published robustness report has an inconsistent execution_status.")

    if report["scientific_assessment_status"] not in {
        "completed",
        "restricted_evidence_summary_only"
    }:
        raise RuntimeError("Published robustness report has an unsupported scientific_assessment_status.")

    if report["scientific_assessment_status"] != expected_assessment_status:
        raise RuntimeError("Published robustness report scientific_assessment_status does not match expected.")

    if report["domain_status_counts"] != expected_assessment_status_counts:
        raise RuntimeError("Published robustness report domain_status_counts do not match expected.")

    if report["registration_status"] != expected_policy["registration_status"]:
        raise RuntimeError("Published robustness report registration_status is invalid.")

    if report["amendment"] != expected_policy["amendment"]:
        raise RuntimeError("Published robustness report amendment is invalid.")

    if (
        not isinstance(
            report[
                "prediction_inventory_reconciliation"
            ],
            dict
        )
    ):
        raise RuntimeError("Published robustness report prediction inventory reconciliation is not an object.")

    if (
        report[
            "prediction_inventory_reconciliation"
        ]
        != expected_prediction_inventory_reconciliation
    ):
        raise RuntimeError("Published robustness report prediction inventory reconciliation is invalid.")

    handoff = report["analytical_handoff"]

    if not isinstance(handoff, dict):
        raise RuntimeError("Published robustness report analytical_handoff is not an object.")

    for required_handoff_field in (
        "status",
        "fully_assessed_domain_count",
        "restricted_or_unavailable_domain_count",
        "causal_interpretation_authorized",
        "policy_recommendation_authorized",
        "treatment_optimization_authorized",
        "cross_stage_transport_authorized"
    ):
        if required_handoff_field not in handoff:
            raise RuntimeError(f"Published robustness report analytical_handoff is missing : {required_handoff_field}")

    for authorization_field in (
        "causal_interpretation_authorized",
        "policy_recommendation_authorized",
        "treatment_optimization_authorized",
        "cross_stage_transport_authorized"
    ):
        if handoff[authorization_field] is not False:
            raise RuntimeError(f"Published robustness report analytical_handoff incorrectly enables : {authorization_field}")

    if handoff["status"] != report["scientific_assessment_status"]:
        raise RuntimeError("Published robustness handoff status does not match the scientific assessment status.")

    declared_tables = report.get("tables", {})
    expected_table_names = set(expected_policy["output_contract"]["tables"])

    if set(declared_tables) != expected_table_names:
        raise RuntimeError("Published robustness report has an unexpected table inventory.")

    for table_name in expected_table_names:
        table_manifest_record = declared_tables[table_name]
        table_path = published_directory / table_manifest_record["file_name"]

        if not table_path.is_file():
            raise RuntimeError(f"Published robustness table is missing : {table_name}")

        if calculate_file_sha256(table_path) != table_manifest_record["sha256"]:
            raise RuntimeError(f"Published robustness table hash does not match : {table_name}")

        loaded_rows = read_csv_table(table_path)

        if len(loaded_rows) != int(table_manifest_record["row_count"]):
            raise RuntimeError(f"Published robustness table row count does not match : {table_name}")

        if len(loaded_rows) != len(expected_tables[table_name]):
            raise RuntimeError(f"Published robustness table has unexpected row count : {table_name}")

        if csv_canonical_rows(loaded_rows) != csv_canonical_rows(expected_tables[table_name]):
            raise RuntimeError(f"Published robustness table content does not match the computed records : {table_name}")

    coverage_rows = read_csv_table(
        published_directory / "assessment_coverage.csv"
    )
    registered_domains = {
        record["assessment_identity"]
        for record in expected_policy["assessment_domains"]
    }
    covered_domains = [row["assessment_identity"] for row in coverage_rows]

    if len(covered_domains) != len(set(covered_domains)):
        raise RuntimeError("Published robustness coverage contains duplicate domains.")

    if set(covered_domains) != registered_domains:
        missing = registered_domains - set(covered_domains)
        extra = set(covered_domains) - registered_domains
        raise RuntimeError(
            "Published robustness coverage does not reconcile with registered domains : " \
            f"missing {sorted(missing)} : extra {sorted(extra)}"
        )

    terminal_statuses = set(
        expected_policy["assessment_contract"]["terminal_domain_statuses"]
    )

    for row in coverage_rows:
        if row["assessment_status"] not in terminal_statuses:
            raise RuntimeError(
                f"Published coverage domain has non-terminal status : {row['assessment_identity']} : {row['assessment_status']}"
            )

    observed_statuses = {row["assessment_status"] for row in coverage_rows}

    if not observed_statuses:
        raise RuntimeError("Published robustness report contains no domain coverage.")

    if not observed_statuses.issubset(terminal_statuses):
        raise RuntimeError("Published robustness coverage contains a non-terminal status.")

    if observed_statuses == {"completed"}:
        recomputed_scientific_status = "completed"
    else:
        recomputed_scientific_status = "restricted_evidence_summary_only"

    if report["scientific_assessment_status"] != recomputed_scientific_status:
        raise RuntimeError("Published robustness scientific_assessment_status does not recompute from coverage.")

    for row in coverage_rows:
        completed = int(row["completed_evidence_count"])
        restricted = int(row["restricted_evidence_count"])
        unavailable = int(row["unavailable_evidence_count"])
        total = int(row["total_evidence_count"])

        if completed + restricted + unavailable != total:
            raise RuntimeError(f"Published robustness coverage evidence counts do not reconcile : {row['assessment_identity']}")

    expected_domain_status_counts = dict(
        sorted(
            Counter(
                row["assessment_status"] for row in coverage_rows
            ).items()
        )
    )

    if report["domain_status_counts"] != expected_domain_status_counts:
        raise RuntimeError("Published robustness report domain_status_counts do not reconcile.")

    expected_fully_assessed = expected_domain_status_counts.get("completed", 0)
    expected_restricted = sum(
        count
        for status, count in expected_domain_status_counts.items()
        if status != "completed"
    )

    if handoff["fully_assessed_domain_count"] != expected_fully_assessed:
        raise RuntimeError("Published robustness report fully_assessed_domain_count does not reconcile.")

    if handoff["restricted_or_unavailable_domain_count"] != expected_restricted:
        raise RuntimeError("Published robustness report restricted_or_unavailable_domain_count does not reconcile.")

    external_validity_rows = read_csv_table(
        published_directory / "external_validity.csv"
    )
    expected_external = [
        {
            "domain_identity": record["domain_identity"],
            "applicability_status": record["applicability_status"],
            "boundary": record["boundary"]
        }
        for record in expected_policy["external_validity_domains"]
    ]

    if len(external_validity_rows) != len(expected_external):
        raise RuntimeError("Published external-validity row count does not match policy.")

    for published_row, expected_row in zip(
        external_validity_rows, expected_external, strict = True
    ):
        for field_name in ("domain_identity", "applicability_status", "boundary"):
            if str(published_row[field_name]) != str(expected_row[field_name]):
                raise RuntimeError(f"Published external-validity row does not match policy : {field_name}")


# Confirming every emitted status and reason code belongs to the registered vocabularies
def validate_emitted_codes_and_statuses(tables: dict, policy: dict) -> None:
    vocabularies = policy["assessment_contract"]["registered_status_vocabularies"]
    reason_codes = set(policy["assessment_contract"]["registered_reason_codes"])

    def check_status(status_value, vocabulary_name, table_name, field_name):
        if status_value is None or status_value == "":
            return
        allowed = set(vocabularies[vocabulary_name])
        if status_value not in allowed:
            raise RuntimeError(
                f"Robustness table contains an unregistered status : {table_name} : {field_name} : {status_value}"
            )

    def check_reason(reason_value, table_name, field_name):
        if reason_value is None or reason_value == "":
            return
        if reason_value not in reason_codes:
            raise RuntimeError(
                f"Robustness table contains an unregistered reason code : {table_name} : {field_name} : {reason_value}"
            )

    for row in tables["model_family_comparisons"]:
        check_status(row["comparison_status"], "comparison_status", "model_family_comparisons", "comparison_status")
        check_reason(row["comparison_reason_code"], "model_family_comparisons", "comparison_reason_code")

    for row in tables["prior_sensitivity"]:
        check_status(row["assessment_status"], "calibration_status", "prior_sensitivity", "assessment_status")
        check_reason(row["assessment_reason_code"], "prior_sensitivity", "assessment_reason_code")

    for row in tables["influence_summary"]:
        check_status(row["assessment_status"], "calibration_status", "influence_summary", "assessment_status")
        check_reason(row["assessment_reason_code"], "influence_summary", "assessment_reason_code")

    for row in tables["prediction_agreement"]:
        check_status(row["assessment_status"], "prediction_agreement_status", "prediction_agreement", "assessment_status")
        check_reason(row["assessment_reason_code"], "prediction_agreement", "assessment_reason_code")

    for row in tables["cohort_sensitivity"]:
        check_status(row["comparison_status"], "cohort_comparison_status", "cohort_sensitivity", "comparison_status")
        check_reason(row["comparison_reason_code"], "cohort_sensitivity", "comparison_reason_code")

    for row in tables["descriptive_specification_comparisons"]:
        check_status(row["comparison_status"], "comparison_status", "descriptive_specification_comparisons", "comparison_status")
        check_reason(row["comparison_reason_code"], "descriptive_specification_comparisons", "comparison_reason_code")

    for row in tables["structural_limitations"]:
        if row["applicability_status"] not in {"unsupported", "unavailable", "not_authorized"}:
            raise RuntimeError(
                f"Structural limitation has non-terminal status : {row['limitation_identity']} : {row['applicability_status']}"
            )
        check_reason(row["reason_code"], "structural_limitations", "reason_code")

    for row in tables["validation_summary"]:
        check_status(
            row["assessment_status"],
            "domain_status",
            "validation_summary",
            "assessment_status"
        )

        check_reason(
            row["assessment_reason_code"],
            "validation_summary",
            "assessment_reason_code"
        )

    if "configuration_boundary_summary" in tables:
        for row in tables["configuration_boundary_summary"]:
            boundary_status = row["boundary_assessment_status"]
            boundary_class = row["boundary_classification"]

            if boundary_status not in set(vocabularies["boundary_assessment_status"]):
                raise RuntimeError(
                    f"Robustness table contains an unregistered boundary status : "
                    f"configuration_boundary_summary : boundary_assessment_status : {boundary_status}"
                )

            if boundary_class not in set(vocabularies["boundary_classification"]):
                raise RuntimeError(
                    f"Robustness table contains an unregistered boundary classification : "
                    f"configuration_boundary_summary : boundary_classification : {boundary_class}"
                )

    for row in tables[
        "unsupported_condition_inventory"
    ]:
        check_status(
            row[
                "calibration_status"
            ],
            "calibration_status",
            "unsupported_condition_inventory",
            "calibration_status"
        )

        check_reason(
            row[
                "calibration_reason_code"
            ],
            "unsupported_condition_inventory",
            "calibration_reason_code"
        )

    for table_name in (
        "prediction_interval_diagnostics",
        "horizon_evaluation_sensitivity",
        "incident_evaluation_sensitivity"
    ):
        for row in tables[table_name]:
            check_status(
                row["assessment_status"],
                "domain_status",
                table_name,
                "assessment_status"
            )

            check_reason(
                row["assessment_reason_code"],
                table_name,
                "assessment_reason_code"
            )

    if "assessment_coverage" in tables:
        for row in tables["assessment_coverage"]:
            check_status(
                row["assessment_status"],
                "domain_status",
                "assessment_coverage",
                "assessment_status"
            )

            check_reason(
                row["assessment_reason_code"],
                "assessment_coverage",
                "assessment_reason_code"
            )

    if "predictive_sensitivity_summary" in tables:
        for row in tables["predictive_sensitivity_summary"]:
            check_status(
                row["assessment_status"],
                "domain_status",
                "predictive_sensitivity_summary",
                "assessment_status"
            )

            check_reason(
                row["assessment_reason_code"],
                "predictive_sensitivity_summary",
                "assessment_reason_code"
            )


# Publishing the complete artifact set transactionally
def publish_outputs(args: argparse.Namespace, policy: dict,
                    policy_sha256: str, input_hashes: dict,
                    tables: dict, prediction_inventory_reconciliation: dict
) -> None:
    output_directory = args.output_directory.resolve()

    output_directory.parent.mkdir(
        parents = True,
        exist_ok = True
    )

    staging_root = Path(
        tempfile.mkdtemp(
            prefix = ".robustness_assessment_",
            dir = output_directory.parent
        )
    )

    try:
        table_contracts = {
            "validation_summary": (
                VALIDATION_SUMMARY_FIELDS,
                tables["validation_summary"]
            ),
            "model_family_comparisons": (
                MODEL_FAMILY_COMPARISON_FIELDS,
                tables["model_family_comparisons"]
            ),
            "prior_sensitivity": (
                PRIOR_SENSITIVITY_FIELDS,
                tables["prior_sensitivity"]
            ),
            "influence_summary": (
                INFLUENCE_SUMMARY_FIELDS,
                tables["influence_summary"]
            ),
            "prediction_agreement": (
                PREDICTION_AGREEMENT_FIELDS,
                tables["prediction_agreement"]
            ),
            "cohort_sensitivity": (
                COHORT_SENSITIVITY_FIELDS,
                tables["cohort_sensitivity"]
            ),
            "descriptive_specification_comparisons": (
                DESCRIPTIVE_SPECIFICATION_COMPARISON_FIELDS,
                tables["descriptive_specification_comparisons"]
            ),
            "structural_limitations": (
                STRUCTURAL_LIMITATION_FIELDS,
                tables["structural_limitations"]
            ),
            "configuration_boundary_summary": (
                CONFIGURATION_BOUNDARY_FIELDS,
                tables["configuration_boundary_summary"]
            ),
            "fragility_registry": (
                FRAGILITY_REGISTRY_FIELDS,
                tables["fragility_registry"]
            ),
            "unsupported_condition_inventory": (
                UNSUPPORTED_CONDITION_FIELDS,
                tables["unsupported_condition_inventory"]
            ),
            "assessment_coverage": (
                ASSESSMENT_COVERAGE_FIELDS,
                tables["assessment_coverage"]
            ),
            "external_validity": (
                EXTERNAL_VALIDITY_FIELDS,
                tables["external_validity"]
            ),
            "prediction_interval_diagnostics": (
                PREDICTION_INTERVAL_DIAGNOSTIC_FIELDS,
                tables["prediction_interval_diagnostics"]
            ),
            "horizon_evaluation_sensitivity": (
                HORIZON_EVALUATION_SENSITIVITY_FIELDS,
                tables["horizon_evaluation_sensitivity"]
            ),
            "incident_evaluation_sensitivity": (
                INCIDENT_EVALUATION_SENSITIVITY_FIELDS,
                tables["incident_evaluation_sensitivity"]
            ),
            "predictive_sensitivity_summary": (
                PREDICTIVE_SENSITIVITY_SUMMARY_FIELDS,
                tables["predictive_sensitivity_summary"]
            )
        }

        report_tables = {}

        for table_name, (
            fields,
            records
        ) in table_contracts.items():
            table_path = (
                staging_root
                / f"{table_name}.csv"
            )

            write_csv_table(
                table_path,
                fields,
                records
            )

            report_tables[
                table_name
            ] = table_manifest(
                table_path,
                records,
                fields
            )

        coverage_status_counts = Counter(
            record[
                "assessment_status"
            ]
            for record in tables[
                "assessment_coverage"
            ]
        )

        restricted_domain_count = sum(
            count
            for status, count in coverage_status_counts.items()
            if status != "completed"
        )

        coverage_statuses = set(
            coverage_status_counts.keys()
        )

        if coverage_statuses == {"completed"}:
            scientific_assessment_status = "completed"
        else:
            scientific_assessment_status = "restricted_evidence_summary_only"

        analytical_handoff = {
            "status": scientific_assessment_status,
            "fully_assessed_domain_count": coverage_status_counts.get(
                "completed", 0
            ),
            "restricted_or_unavailable_domain_count": restricted_domain_count,
            "causal_interpretation_authorized": False,
            "policy_recommendation_authorized": False,
            "treatment_optimization_authorized": False,
            "cross_stage_transport_authorized": False
        }

        report = {
            "execution_status": "completed",
            "scientific_assessment_status": (
                scientific_assessment_status
            ),
            "policy_name": policy[
                "policy_name"
            ],
            "policy_version": policy[
                "policy_version"
            ],
            "policy_sha256": policy_sha256,
            "implementation_sha256": calculate_file_sha256(
                Path(
                    __file__
                ).resolve()
            ),
            "input_hashes": input_hashes,
            "causal_interpretation_authorized": False,
            "policy_recommendation_authorized": False,
            "performance_selected_model_winner_declared": False,
            "magnitude_dependent_robustness_classification_used": False,
            "registration_status": policy["registration_status"],
            "amendment": policy["amendment"],
            "domain_status_counts": dict(
                sorted(coverage_status_counts.items())
            ),
            "prediction_inventory_reconciliation": (
                prediction_inventory_reconciliation
            ),
            "analytical_handoff": analytical_handoff,
            "tables": report_tables
        }

        report_path = (
            staging_root
            / policy[
                "output_contract"
            ][
                "report_file"
            ]
        )

        report_path.write_text(
            json.dumps(
                report,
                indent = 2,
                sort_keys = True
            )
            + "\n",
            encoding = "utf-8"
        )

        backup_directory = output_directory.with_name(
            f"{output_directory.name}.previous"
        )

        if backup_directory.exists():
            raise RuntimeError("A prior publication backup already exists.")

        if output_directory.exists():
            output_directory.rename(
                backup_directory
            )

        staging_root.rename(
            output_directory
        )

        try:
            validate_published_robustness_directory(
                published_directory = output_directory,
                expected_policy = policy,
                expected_policy_sha256 = policy_sha256,
                expected_input_hashes = input_hashes,
                expected_tables = tables,
                expected_assessment_status = scientific_assessment_status,
                expected_assessment_status_counts = dict(
                    sorted(coverage_status_counts.items())
                ),
                expected_prediction_inventory_reconciliation = (
                    prediction_inventory_reconciliation
                )
            )
        except Exception:
            if output_directory.exists():
                shutil.rmtree(output_directory, ignore_errors = True)

            if backup_directory.exists():
                backup_directory.rename(output_directory)

            raise

        if backup_directory.exists():
            shutil.rmtree(
                backup_directory
            )

    except Exception:
        if staging_root.exists():
            shutil.rmtree(
                staging_root
            )

        raise


# Converting table-level empirical statuses into one domain evaluation
def empirical_domain_evaluation(records: list[dict], no_evidence_reason_code: str) -> dict:
    status_counts = Counter(
        record["assessment_status"]
        for record in records
    )

    completed_count = status_counts.get(
        "completed",
        0
    )

    restricted_count = (
        status_counts.get(
            "completed_with_restrictions",
            0
        )
        + status_counts.get(
            "incomplete",
            0
        )
        + status_counts.get(
            "diagnostic_only",
            0
        )
    )

    unavailable_count = status_counts.get(
        "unavailable",
        0
    )

    total_count = len(
        records
    )

    status, reason_code = derive_domain_status_from_counts(
        total_count = total_count,
        completed_count = completed_count,
        restricted_count = restricted_count,
        unavailable_count = unavailable_count,
        no_evidence_reason_code = no_evidence_reason_code,
        partial_reason_code = "PARTIAL_EVIDENCE_COVERAGE",
        all_diagnostic_reason_code = "ALL_EVIDENCE_DIAGNOSTIC_ONLY",
        all_unavailable_reason_code = "VALIDATION_METRICS_UNAVAILABLE"
    )

    return domain_evaluation(
        status = status,
        total_count = total_count,
        completed_count = completed_count,
        restricted_count = restricted_count,
        unavailable_count = unavailable_count,
        reason_code = reason_code
    )


# Converting sensitivity-variant statuses into one domain evaluation for the supplied variant types
def sensitivity_domain_evaluation(records: list[dict], variant_types: set[str]) -> dict:
    selected_records = [
        record
        for record in records
        if record["variant_type"] in variant_types
    ]

    status_counts = Counter(
        record["assessment_status"]
        for record in selected_records
    )

    completed_count = status_counts.get(
        "completed",
        0
    )

    restricted_count = status_counts.get(
        "diagnostic_only",
        0
    )

    unavailable_count = status_counts.get(
        "unavailable",
        0
    )

    status, reason_code = derive_domain_status_from_counts(
        total_count = len(selected_records),
        completed_count = completed_count,
        restricted_count = restricted_count,
        unavailable_count = unavailable_count,
        no_evidence_reason_code = "NO_EVIDENCE",
        partial_reason_code = "PARTIAL_EVIDENCE_COVERAGE",
        all_diagnostic_reason_code = "ALL_EVIDENCE_DIAGNOSTIC_ONLY",
        all_unavailable_reason_code = "SENSITIVITY_VARIANT_UNAVAILABLE"
    )

    return domain_evaluation(
        status = status,
        total_count = len(selected_records),
        completed_count = completed_count,
        restricted_count = restricted_count,
        unavailable_count = unavailable_count,
        reason_code = reason_code
    )


# Removing held-out predictions whose candidate or benchmark identity has no metric record
def exclude_unmatched_predictions(input_tables: dict) -> dict:
    framework_contracts = [
        (
            "descriptive_modeling",
            "validation_predictions",
            "validation_metrics",
            "candidate_identity"
        ),
        (
            "bayesian_modeling",
            "grouped_validation_predictions",
            "grouped_validation_metrics",
            "candidate_identity"
        ),
        (
            "predictive_modeling",
            "validation_predictions",
            "validation_metrics",
            "benchmark_identity"
        )
    ]

    reconciliation = {}

    for (
        framework_name,
        prediction_table_name,
        metric_table_name,
        identity_field
    ) in framework_contracts:
        metric_identities = {
            str(metric[identity_field])
            for metric in input_tables[
                framework_name
            ][
                metric_table_name
            ]
        }

        raw_predictions = input_tables[
            framework_name
        ].get(
            prediction_table_name,
            []
        )

        orphan_predictions = [
            prediction
            for prediction in raw_predictions
            if str(
                prediction[
                    identity_field
                ]
            ) not in metric_identities
        ]

        orphan_identities = sorted({
            str(
                prediction[
                    identity_field
                ]
            )
            for prediction in orphan_predictions
        })

        reconciliation[
            framework_name
        ] = {
            "identity_field": identity_field,
            "excluded_identity_count": len(
                orphan_identities
            ),
            "excluded_prediction_count": len(
                orphan_predictions
            ),
            "excluded_identities": orphan_identities,
            "reason_code": (
                "PREDICTION_INVENTORY_WITHOUT_METRIC_RECORD"
                if orphan_identities
                else None
            )
        }

        if orphan_identities:
            print(
                f"WARNING : {len(orphan_identities)} {framework_name} identities have held-out predictions without matching"
                " metric records; their predictions are excluded from the robustness assessment."
            )

            orphan_identity_set = set(
                orphan_identities
            )

            input_tables[
                framework_name
            ][
                prediction_table_name
            ] = [
                prediction
                for prediction in raw_predictions
                if str(
                    prediction[
                        identity_field
                    ]
                ) not in orphan_identity_set
            ]

    return reconciliation


def main() -> None:
    args = parse_args()

    policy = load_json_object(args.policy_path)
    validate_policy(policy)

    policy_sha256 = calculate_file_sha256(args.policy_path)

    data_quality_contract_sha256 = calculate_file_sha256(args.data_quality_contract_path)

    if data_quality_contract_sha256 != policy["inputs"]["required_file_sha256"]["data_quality_contract"]:
        raise RuntimeError("Data-quality contract does not match the registered robustness policy.")

    predictive_policy_sha256 = calculate_file_sha256(args.predictive_policy_path)

    if predictive_policy_sha256 != policy["inputs"]["required_file_sha256"]["predictive_modeling_policy"]:
        raise RuntimeError("Predictive modeling policy does not match the registered robustness policy.")

    predictive_sensitivity_policy_sha256 = calculate_file_sha256(
        args.predictive_sensitivity_policy_path
    )

    if (
        predictive_sensitivity_policy_sha256
        != policy["inputs"]["required_file_sha256"][
            "predictive_sensitivity_policy"
        ]
    ):
        raise RuntimeError("Predictive-sensitivity policy does not match the registered robustness policy.")

    data_quality_contract = load_json_object(args.data_quality_contract_path)
    predictive_policy = load_json_object(args.predictive_policy_path)

    feature_manifest_sha256 = calculate_file_sha256(
        args.feature_manifest_path
    )

    if (
        feature_manifest_sha256
        != policy["inputs"]["required_file_sha256"]["feature_manifest"]
    ):
        raise RuntimeError("Feature manifest does not match the registered robustness policy.")

    feature_manifest, feature_tables = verify_report_tables(
        args.feature_manifest_path,
        policy["inputs"]["required_file_sha256"]["feature_manifest"],
        policy["inputs"]["required_feature_tables"]
    )

    report_paths = {
        "experimental_design": args.design_report_path,
        "causal_identification": args.identification_report_path,
        "descriptive_modeling": args.descriptive_report_path,
        "bayesian_modeling": args.bayesian_report_path,
        "predictive_modeling": args.predictive_report_path,
        "predictive_sensitivity": args.predictive_sensitivity_report_path
    }

    reports = {}
    input_tables = {}
    input_hashes = {}

    for input_name, report_path in report_paths.items():
        input_hashes[input_name] = calculate_file_sha256(report_path)
        report, report_tables = verify_report_tables(
            report_path,
            policy["inputs"]["required_report_sha256"][input_name],
            policy["inputs"]["required_tables"][input_name]
        )
        reports[input_name] = report
        input_tables[input_name] = report_tables

    input_hashes["data_quality_contract"] = data_quality_contract_sha256
    input_hashes["predictive_modeling_policy"] = predictive_policy_sha256
    input_hashes["feature_manifest"] = feature_manifest_sha256
    input_hashes["predictive_sensitivity_policy"] = predictive_sensitivity_policy_sha256

    sensitivity_report = reports["predictive_sensitivity"]

    if sensitivity_report.get("execution_status") != "completed":
        raise RuntimeError("Predictive-sensitivity execution is not complete.")

    if (
        sensitivity_report.get("policy_sha256")
        != predictive_sensitivity_policy_sha256
    ):
        raise RuntimeError("Predictive-sensitivity report and policy do not match.")

    if (
        sensitivity_report.get("causal_interpretation_authorized")
        is not False
    ):
        raise RuntimeError("Predictive-sensitivity report incorrectly authorizes causal interpretation.")

    if (
        sensitivity_report.get("primary_benchmark_replacement_declared")
        is not False
    ):
        raise RuntimeError("Predictive-sensitivity report incorrectly replaces the primary benchmark.")

    validate_input_identity_constraints(
        input_tables
    )

    if reports["causal_identification"].get("causal_modeling_status") != "not_authorized":
        raise RuntimeError("Causal-identification status does not match the registered scope.")

    prediction_inventory_reconciliation = (
        exclude_unmatched_predictions(
            input_tables
        )
    )

    reconcile_prediction_inventories(
        prediction_records = input_tables["descriptive_modeling"].get("validation_predictions", []),
        metric_records = input_tables["descriptive_modeling"]["validation_metrics"],
        identity_field = "candidate_identity",
        prediction_status_field = None,
        metric_status_field = "validation_status",
        cell_count_field = "independent_cell_count",
        observation_count_field = "observation_count"
    )

    reconcile_prediction_inventories(
        prediction_records = input_tables["bayesian_modeling"].get("grouped_validation_predictions", []),
        metric_records = input_tables["bayesian_modeling"]["grouped_validation_metrics"],
        identity_field = "candidate_identity",
        prediction_status_field = "prediction_status",
        metric_status_field = "validation_status",
        cell_count_field = "test_cell_count",
        observation_count_field = "test_observation_count"
    )

    reconcile_prediction_inventories(
        prediction_records = input_tables["predictive_modeling"].get("validation_predictions", []),
        metric_records = input_tables["predictive_modeling"]["validation_metrics"],
        identity_field = "benchmark_identity",
        prediction_status_field = "prediction_status",
        metric_status_field = "validation_status",
        cell_count_field = "test_cell_count",
        observation_count_field = "test_observation_count"
    )

    prediction_inventory = build_held_out_prediction_inventory(
        input_tables["descriptive_modeling"],
        input_tables["bayesian_modeling"],
        input_tables["predictive_modeling"]
    )

    minimum_cell_count = policy["assessment_contract"]["held_out_evaluation"]["minimum_independent_cell_count"]

    tables = {
        "validation_summary": build_validation_summary(
            input_tables["descriptive_modeling"],
            input_tables["bayesian_modeling"],
            input_tables["predictive_modeling"]
        ),
        "model_family_comparisons": build_model_family_comparisons(
            input_tables["predictive_modeling"]
        ),
        "prior_sensitivity": build_prior_sensitivity(
            input_tables["bayesian_modeling"]
        ),
        "influence_summary": build_influence_summary(
            input_tables["descriptive_modeling"]
        ),
        "prediction_agreement": build_prediction_agreement(
            input_tables["bayesian_modeling"],
            input_tables["predictive_modeling"]
        ),
        "cohort_sensitivity": build_cohort_sensitivity(
            input_tables["descriptive_modeling"],
            input_tables["bayesian_modeling"],
            input_tables["predictive_modeling"],
            policy
        ),
        "descriptive_specification_comparisons": (
            build_descriptive_specification_comparisons(
                input_tables["descriptive_modeling"]
            )
        ),
        "structural_limitations": build_structural_limitations(
            input_tables["experimental_design"],
            data_quality_contract,
            input_tables["causal_identification"]["estimand_identification"]
        ),
        "configuration_boundary_summary": build_configuration_boundary_summary(
            input_tables["predictive_modeling"],
            predictive_policy
        ),
        "external_validity": [dict(r) for r in policy["external_validity_domains"]],
        "unsupported_condition_inventory": build_unsupported_condition_inventory(
            input_tables["predictive_modeling"]
        ),
        "prediction_interval_diagnostics": build_prediction_interval_diagnostics(
            prediction_inventory
        ),
        "horizon_evaluation_sensitivity": build_horizon_evaluation_sensitivity(
            prediction_inventory,
            minimum_cell_count
        ),
        "incident_evaluation_sensitivity": build_incident_evaluation_sensitivity(
            prediction_inventory,
            feature_tables["reference_test_features"],
            minimum_cell_count
        ),
        "predictive_sensitivity_summary": build_predictive_sensitivity_summary(
            input_tables["predictive_sensitivity"]
        )
    }

    tables["fragility_registry"] = build_fragility_registry(
        input_tables["descriptive_modeling"],
        input_tables["bayesian_modeling"],
        input_tables["predictive_modeling"],
        input_tables["causal_identification"]["estimand_identification"],
        tables
    )

    domain_evaluations = {}

    def condition_stage(condition_identity: object) -> str:
        return str(condition_identity or "").split("/", maxsplit = 1)[0]

    stage_comparisons = [
        r for r in input_tables["experimental_design"]["estimand_support"]
        if condition_stage(r.get("first_condition_protocol_identity", ""))
        != condition_stage(r.get("second_condition_protocol_identity", ""))
    ]

    stage_summary_records = tables["validation_summary"]
    stage_summary_status_counts = Counter(
        record["assessment_status"]
        for record in stage_summary_records
    )
    stage_summary_total = len(stage_summary_records)
    stage_summary_completed = stage_summary_status_counts.get("completed", 0)
    stage_summary_restricted = (
        stage_summary_status_counts.get("completed_with_restrictions", 0)
        + stage_summary_status_counts.get("diagnostic_only", 0)
    )
    stage_summary_unavailable = stage_summary_status_counts.get("unavailable", 0)

    stage_performance_status, stage_performance_reason = derive_domain_status_from_counts(
        total_count = stage_summary_total,
        completed_count = stage_summary_completed,
        restricted_count = stage_summary_restricted,
        unavailable_count = stage_summary_unavailable,
        no_evidence_reason_code = "NO_EVIDENCE",
        partial_reason_code = "PARTIAL_VALIDATION_COVERAGE",
        all_diagnostic_reason_code = "ALL_EVIDENCE_DIAGNOSTIC_ONLY",
        all_unavailable_reason_code = "VALIDATION_METRICS_UNAVAILABLE"
    )

    domain_evaluations["stage_stratified_performance_comparison"] = domain_evaluation(
        status = stage_performance_status,
        total_count = stage_summary_total,
        completed_count = stage_summary_completed,
        restricted_count = stage_summary_restricted,
        unavailable_count = stage_summary_unavailable,
        reason_code = stage_performance_reason
    )

    domain_evaluations["cross_stage_transportability"] = domain_evaluation(
        status = ("not_authorized" if stage_comparisons else "unavailable"),
        total_count = len(stage_comparisons),
        completed_count = 0,
        restricted_count = (len(stage_comparisons) if stage_comparisons else 0),
        unavailable_count = 0,
        reason_code = (
            "CROSS_STAGE_COMPARISONS_NOT_AUTHORIZED"
            if stage_comparisons
            else "NO_CROSS_STAGE_COMPARISONS_PRESENT"
        )
    )

    cohort_status_counts = Counter(
        record["comparison_status"] for record in tables["cohort_sensitivity"]
    )
    cohort_total = len(tables["cohort_sensitivity"])

    cohort_domain_status, cohort_domain_reason = derive_domain_status_from_counts(
        total_count = cohort_total,
        completed_count = cohort_status_counts.get("completed", 0),
        restricted_count = cohort_status_counts.get("diagnostic_only", 0),
        unavailable_count = cohort_status_counts.get("unavailable", 0),
        no_evidence_reason_code = "NO_PRIMARY_SENSITIVITY_PAIRS",
        partial_reason_code = "PARTIAL_EVIDENCE_COVERAGE",
        all_diagnostic_reason_code = "ALL_EVIDENCE_DIAGNOSTIC_ONLY",
        all_unavailable_reason_code = "VALIDATION_METRICS_UNAVAILABLE"
    )

    domain_evaluations["cohort_sensitivity_comparison"] = domain_evaluation(
        status = cohort_domain_status,
        total_count = cohort_total,
        completed_count = cohort_status_counts.get("completed", 0),
        restricted_count = cohort_status_counts.get("diagnostic_only", 0),
        unavailable_count = cohort_status_counts.get("unavailable", 0),
        reason_code = cohort_domain_reason
    )

    domain_evaluations["cohort_inclusion_sensitivity"] = domain_evaluation(
        status = cohort_domain_status,
        total_count = cohort_total,
        completed_count = cohort_status_counts.get("completed", 0),
        restricted_count = cohort_status_counts.get("diagnostic_only", 0),
        unavailable_count = cohort_status_counts.get("unavailable", 0),
        reason_code = cohort_domain_reason
    )

    model_spec_status_counts = Counter()

    for record in tables["model_family_comparisons"]:
        model_spec_status_counts[record["comparison_status"]] += 1

    for record in tables["descriptive_specification_comparisons"]:
        model_spec_status_counts[record["comparison_status"]] += 1

    model_spec_total = sum(model_spec_status_counts.values())

    model_spec_domain_status, model_spec_domain_reason = derive_domain_status_from_counts(
        total_count = model_spec_total,
        completed_count = model_spec_status_counts.get("completed", 0),
        restricted_count = model_spec_status_counts.get("diagnostic_only", 0),
        unavailable_count = model_spec_status_counts.get("unavailable", 0),
        no_evidence_reason_code = "NO_MODEL_COMPARISONS",
        partial_reason_code = "PARTIAL_EVIDENCE_COVERAGE",
        all_diagnostic_reason_code = "ALL_EVIDENCE_DIAGNOSTIC_ONLY",
        all_unavailable_reason_code = "VALIDATION_METRICS_UNAVAILABLE"
    )

    domain_evaluations["model_specification_comparison"] = domain_evaluation(
        status = model_spec_domain_status,
        total_count = model_spec_total,
        completed_count = model_spec_status_counts.get("completed", 0),
        restricted_count = model_spec_status_counts.get("diagnostic_only", 0),
        unavailable_count = model_spec_status_counts.get("unavailable", 0),
        reason_code = model_spec_domain_reason
    )

    prior_status_counts = Counter(record["assessment_status"] for record in tables["prior_sensitivity"])
    prior_total = len(tables["prior_sensitivity"])

    prior_domain_status, prior_domain_reason = derive_domain_status_from_counts(
        total_count = prior_total,
        completed_count = prior_status_counts.get("completed", 0),
        restricted_count = prior_status_counts.get("incomplete", 0),
        unavailable_count = 0,
        no_evidence_reason_code = "NO_PRIOR_SENSITIVITY_PAIRS",
        partial_reason_code = "PARTIAL_EVIDENCE_COVERAGE",
        all_diagnostic_reason_code = "ALL_EVIDENCE_DIAGNOSTIC_ONLY",
        all_unavailable_reason_code = "VALIDATION_METRICS_UNAVAILABLE"
    )

    domain_evaluations["prior_scale_sensitivity_comparison"] = domain_evaluation(
        status = prior_domain_status,
        total_count = prior_total,
        completed_count = prior_status_counts.get("completed", 0),
        restricted_count = prior_status_counts.get("incomplete", 0),
        unavailable_count = 0,
        reason_code = prior_domain_reason
    )

    influence_status_counts = Counter(record["assessment_status"] for record in tables["influence_summary"])
    influence_total = len(tables["influence_summary"])

    influence_domain_status, influence_domain_reason = derive_domain_status_from_counts(
        total_count = influence_total,
        completed_count = influence_status_counts.get("completed", 0),
        restricted_count = influence_status_counts.get("incomplete", 0),
        unavailable_count = 0,
        no_evidence_reason_code = "NO_INFLUENCE_RECORDS",
        partial_reason_code = "PARTIAL_EVIDENCE_COVERAGE",
        all_diagnostic_reason_code = "ALL_EVIDENCE_DIAGNOSTIC_ONLY",
        all_unavailable_reason_code = "VALIDATION_METRICS_UNAVAILABLE"
    )

    domain_evaluations["cell_influence_assessment"] = domain_evaluation(
        status = influence_domain_status,
        total_count = influence_total,
        completed_count = influence_status_counts.get("completed", 0),
        restricted_count = influence_status_counts.get("incomplete", 0),
        unavailable_count = 0,
        reason_code = influence_domain_reason
    )

    agreement_status_counts = Counter(record["assessment_status"] for record in tables["prediction_agreement"])
    agreement_total = len(tables["prediction_agreement"])

    agreement_domain_status, agreement_domain_reason = derive_domain_status_from_counts(
        total_count = agreement_total,
        completed_count = agreement_status_counts.get("completed", 0),
        restricted_count = agreement_status_counts.get("diagnostic_only", 0),
        unavailable_count = agreement_status_counts.get("unavailable", 0),
        no_evidence_reason_code = "NO_PREDICTION_PAIRS",
        partial_reason_code = "PARTIAL_EVIDENCE_COVERAGE",
        all_diagnostic_reason_code = "ALL_EVIDENCE_DIAGNOSTIC_ONLY",
        all_unavailable_reason_code = "VALIDATION_METRICS_UNAVAILABLE"
    )

    domain_evaluations["cross_framework_prediction_comparison"] = domain_evaluation(
        status = agreement_domain_status,
        total_count = agreement_total,
        completed_count = agreement_status_counts.get("completed", 0),
        restricted_count = agreement_status_counts.get("diagnostic_only", 0),
        unavailable_count = agreement_status_counts.get("unavailable", 0),
        reason_code = agreement_domain_reason
    )

    sensitivity_summary = tables[
        "predictive_sensitivity_summary"
    ]

    sensitivity_interval_records = [
        {
            "assessment_status": record[
                "assessment_status"
            ]
        }
        for record in input_tables[
            "predictive_sensitivity"
        ][
            "prediction_interval_diagnostics"
        ]
    ]

    domain_evaluations[
        "prediction_interval_diagnostics"
    ] = empirical_domain_evaluation(
        tables[
            "prediction_interval_diagnostics"
        ]
        + sensitivity_interval_records,
        "PREDICTION_INTERVALS_UNAVAILABLE"
    )

    training_window_records = [
        record
        for record in sensitivity_summary
        if record["variant_type"] == "training_window"
    ]

    domain_evaluations[
        "horizon_evaluation_sensitivity"
    ] = empirical_domain_evaluation(
        tables[
            "horizon_evaluation_sensitivity"
        ]
        + training_window_records,
        "HORIZON_EVALUATION_INSUFFICIENT_OBSERVATIONS"
    )

    incident_refit_records = [
        record
        for record in sensitivity_summary
        if record["variant_type"] == "incident"
    ]

    domain_evaluations[
        "incident_evaluation_sensitivity"
    ] = empirical_domain_evaluation(
        tables[
            "incident_evaluation_sensitivity"
        ]
        + incident_refit_records,
        "INCIDENT_REASON_NOT_PRESENT"
    )

    domain_evaluations[
        "validation_partition_refit_sensitivity"
    ] = sensitivity_domain_evaluation(
        sensitivity_summary,
        {"partition"}
    )

    domain_evaluations[
        "held_out_feature_ablation"
    ] = sensitivity_domain_evaluation(
        sensitivity_summary,
        {"ablation"}
    )

    domain_evaluations[
        "expanded_hyperparameter_search"
    ] = sensitivity_domain_evaluation(
        sensitivity_summary,
        {"expanded_config"}
    )

    structural_evidence_counts = {
        record["limitation_identity"]: int(record["evidence_record_count"])
        for record in tables["structural_limitations"]
    }

    laboratory_equipment_category_count = sum(
        1
        for limitation_identity in (
            "laboratory_rank_deficient_matrices",
            "laboratory_non_estimable_terms",
            "equipment_allocation_restrictions"
        )
        if structural_evidence_counts.get(limitation_identity, 0) > 0
    )

    if laboratory_equipment_category_count > 0:
        laboratory_equipment_status = "unsupported"
        laboratory_equipment_reason = "EQUIPMENT_LAB_ALIASED"
    else:
        laboratory_equipment_status = "unavailable"
        laboratory_equipment_reason = "NO_EVIDENCE"

    domain_evaluations["laboratory_and_equipment_separability"] = domain_evaluation(
        status = laboratory_equipment_status,
        total_count = laboratory_equipment_category_count,
        completed_count = 0,
        restricted_count = laboratory_equipment_category_count,
        unavailable_count = 0,
        reason_code = laboratory_equipment_reason
    )

    alternative_outcome_evidence_count = structural_evidence_counts.get("ocv_corrected_resistance_not_approved", 0)

    if alternative_outcome_evidence_count > 0:
        alternative_outcome_status = "not_authorized"
        alternative_outcome_reason = "OCV_CORRECTED_RESISTANCE_PROVISIONAL"
    else:
        alternative_outcome_status = "unavailable"
        alternative_outcome_reason = "NO_EVIDENCE"

    domain_evaluations["alternative_outcome_definition"] = domain_evaluation(
        status = alternative_outcome_status,
        total_count = alternative_outcome_evidence_count,
        completed_count = 0,
        restricted_count = alternative_outcome_evidence_count,
        unavailable_count = 0,
        reason_code = alternative_outcome_reason
    )

    net_exposure_evidence_count = structural_evidence_counts.get("net_storage_exposure_unavailable", 0)

    if net_exposure_evidence_count > 0:
        net_exposure_status = "unavailable"
        net_exposure_reason = "NET_STORAGE_EXPOSURE_UNAVAILABLE"
    else:
        net_exposure_status = "unavailable"
        net_exposure_reason = "NO_EVIDENCE"

    domain_evaluations["net_calendar_exposure"] = domain_evaluation(
        status = net_exposure_status,
        total_count = net_exposure_evidence_count,
        completed_count = 0,
        restricted_count = 0,
        unavailable_count = net_exposure_evidence_count,
        reason_code = net_exposure_reason
    )

    batch_evidence_count = structural_evidence_counts.get("stage_batch_aliasing", 0)

    if batch_evidence_count > 0:
        batch_status = "unsupported"
        batch_reason = "STAGE_BATCH_ALIASED"
    else:
        batch_status = "unavailable"
        batch_reason = "NO_EVIDENCE"

    domain_evaluations["batch_stability"] = domain_evaluation(
        status = batch_status,
        total_count = batch_evidence_count,
        completed_count = 0,
        restricted_count = batch_evidence_count,
        unavailable_count = 0,
        reason_code = batch_reason
    )

    domain_evaluations["bayesian_model_form_sensitivity"] = domain_evaluation(
        status = "unavailable",
        total_count = 1,
        completed_count = 0,
        restricted_count = 0,
        unavailable_count = 1,
        reason_code = "BAYESIAN_MODEL_FORM_SENSITIVITY_NOT_PERFORMED"
    )

    domain_evaluations["causal_ml_sensitivity"] = domain_evaluation(
        status = "not_authorized",
        total_count = 1,
        completed_count = 0,
        restricted_count = 1,
        unavailable_count = 0,
        reason_code = "CAUSAL_ML_SENSITIVITY_NOT_AUTHORIZED"
    )

    calibration_status_counts = Counter(
        record["calibration_status"]
        for record in input_tables["predictive_modeling"]["condition_calibration"]
    )

    calibration_total = sum(calibration_status_counts.values())
    calibration_completed = calibration_status_counts.get("completed", 0)
    calibration_unavailable = calibration_status_counts.get("unavailable", 0)
    calibration_incomplete = calibration_status_counts.get("incomplete", 0)

    if calibration_completed + calibration_unavailable + calibration_incomplete != calibration_total:
        raise RuntimeError("Condition calibration contains an unrecognized calibration status.")

    domain_evaluations["condition_calibration_coverage"] = domain_evaluation(
        status = (
            "completed"
            if calibration_unavailable == 0 and calibration_incomplete == 0
            else "completed_with_restrictions"
        ),
        total_count = calibration_total,
        completed_count = calibration_completed,
        restricted_count = calibration_incomplete,
        unavailable_count = calibration_unavailable,
        reason_code = (
            None
            if calibration_unavailable == 0 and calibration_incomplete == 0
            else "CONDITION_CALIBRATION_INCOMPLETE"
        )
    )

    tables["assessment_coverage"] = build_assessment_coverage(
        policy,
        tables,
        domain_evaluations
    )

    validate_emitted_codes_and_statuses(tables, policy)

    publish_outputs(
        args,
        policy,
        policy_sha256,
        input_hashes,
        tables,
        prediction_inventory_reconciliation
    )

    print("Robustness assessment finished.")
    print("Validation summary count :", len(tables["validation_summary"]))
    print("Model-family comparison count :", len(tables["model_family_comparisons"]))
    print("Descriptive specification comparison count :", len(tables["descriptive_specification_comparisons"]))
    print("Fragility registry count :", len(tables["fragility_registry"]))
    print("Assessment report :", args.output_directory / policy["output_contract"]["report_file"])


if __name__ == "__main__":
    main()
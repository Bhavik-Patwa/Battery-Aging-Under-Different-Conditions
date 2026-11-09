import argparse
import csv
import hashlib
import json

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Specifying report-level fields required by the application
REPORT_REQUIRED_FIELDS = {
    "feature_extraction": {
        "extraction_status": "completed",
        "summary": {
            "cell_count": 3,
            "condition_protocol_count": 2
        }
    },
    "experimental_design": {
        "audit_status": "completed",
        "discrete_comparison_horizon_status": "not_authorized_without_outcome_blind_pre_specification",
        "trajectory_estimand_status": "defined_over_recorded_common_support_ranges"
    },
    "causal_identification": {
        "assessment_status": "completed",
        "causal_modeling_status": "not_authorized",
        "associational_hierarchical_modeling_status": "partially_authorized",
        "estimand_count": 4
    },
    "descriptive_modeling": {
        "analysis_status": "completed",
        "causal_interpretation_authorized": False,
        "policy_recommendation_authorized": False
    },
    "bayesian_modeling": {
        "assessment_status": "completed",
        "operation": "fit"
    },
    "predictive_modeling": {
        "analysis_status": "completed_with_candidate_restrictions",
        "causal_interpretation_authorized": False,
        "policy_recommendation_authorized": False,
        "novel_condition_prediction_authorized": False
    },
    "predictive_sensitivity": {
        "execution_status": "completed",
        "causal_interpretation_authorized": False,
        "policy_recommendation_authorized": False
    },
    "robustness_assessment": {
        "execution_status": "completed",
        "scientific_assessment_status": "restricted_evidence_summary_only",
        "causal_interpretation_authorized": False,
        "policy_recommendation_authorized": False
    }
}

# Defining synthetic per-table schemas and deterministic rows
TABLE_FIXTURES = {
    "feature_extraction": {},
    "experimental_design": {
        "treatment_combination_support": {
            "columns": [
                "treatment_combination_identity",
                "aging_mode",
                "ambient_temperature_celsius",
                "maximum_state_of_charge_fraction",
                "depth_of_discharge_fraction",
                "charge_rate_c",
                "discharge_rate_c",
                "reference_test_schedule",
                "condition_protocol_count",
                "independent_cell_count",
                "stage_counts",
                "laboratory_counts"
            ],
            "rows": [
                {
                    "treatment_combination_identity": "fixture_combo_a",
                    "aging_mode": "calendar_aging",
                    "ambient_temperature_celsius": "23.0",
                    "maximum_state_of_charge_fraction": "1.0",
                    "depth_of_discharge_fraction": "1.0",
                    "charge_rate_c": "0.5",
                    "discharge_rate_c": "0.5",
                    "reference_test_schedule": "periodic",
                    "condition_protocol_count": "2",
                    "independent_cell_count": "3",
                    "stage_counts": "{\"Stage_1\": 3}",
                    "laboratory_counts": "{\"HM\": 3}"
                }
            ]
        }
    },
    "causal_identification": {
        "estimand_identification": {
            "columns": [
                "estimand_decision_identity",
                "aging_mode",
                "first_condition_protocol_identity",
                "second_condition_protocol_identity",
                "outcome",
                "cohort",
                "horizon_axis",
                "final_support_classification",
                "point_identification_status",
                "associational_model_status",
                "causal_identification_reason_codes"
            ],
            "rows": [
                {
                    "estimand_decision_identity": "fixture_estimand_a",
                    "aging_mode": "calendar_aging",
                    "first_condition_protocol_identity": "fixture_cond_a",
                    "second_condition_protocol_identity": "fixture_cond_b",
                    "outcome": "relative_capacity",
                    "cohort": "primary",
                    "horizon_axis": "elapsed_calendar_days",
                    "final_support_classification": "supported_with_restrictions",
                    "point_identification_status": "not_identified_from_observed_design",
                    "associational_model_status": "authorized_with_recorded_restrictions",
                    "causal_identification_reason_codes": "[]"
                }
            ]
        },
        "assumption_inventory": {
            "columns": [
                "assumption_name",
                "assumption_scope",
                "evidence_status",
                "required_for_point_identification",
                "evidence_source",
                "observed_evidence",
                "analytical_implication",
                "reason_code"
            ],
            "rows": [
                {
                    "assumption_name": "treatment_assignment",
                    "assumption_scope": "all_estimands",
                    "evidence_status": "not_established",
                    "required_for_point_identification": "true",
                    "evidence_source": "experimental_design_report",
                    "observed_evidence": "not_established",
                    "analytical_implication": "causal_point_identification_not_available",
                    "reason_code": "ASSIGNMENT_MECHANISM_NOT_ESTABLISHED"
                }
            ]
        }
    },
    "descriptive_modeling": {
        "trajectory_summaries": {
            "columns": [
                "record_type",
                "aging_mode",
                "dataset_stage",
                "outcome",
                "cohort",
                "horizon_axis",
                "condition_protocol_identity",
                "horizon",
                "independent_cell_count",
                "observation_count",
                "mean_outcome",
                "standard_deviation_outcome",
                "estimate_label"
            ],
            "rows": [
                {
                    "record_type": "condition",
                    "aging_mode": "calendar_aging",
                    "dataset_stage": "Stage_1",
                    "outcome": "relative_capacity",
                    "cohort": "primary",
                    "horizon_axis": "elapsed_calendar_days",
                    "condition_protocol_identity": "fixture_cond_a",
                    "horizon": "10.0",
                    "independent_cell_count": "3",
                    "observation_count": "3",
                    "mean_outcome": "0.99",
                    "standard_deviation_outcome": "0.005",
                    "estimate_label": "descriptive"
                },
                {
                    "record_type": "condition",
                    "aging_mode": "calendar_aging",
                    "dataset_stage": "Stage_1",
                    "outcome": "relative_capacity",
                    "cohort": "primary",
                    "horizon_axis": "elapsed_calendar_days",
                    "condition_protocol_identity": "fixture_cond_a",
                    "horizon": "20.0",
                    "independent_cell_count": "3",
                    "observation_count": "3",
                    "mean_outcome": "0.98",
                    "standard_deviation_outcome": "0.006",
                    "estimate_label": "descriptive"
                }
            ]
        }
    },
    "bayesian_modeling": {
        "candidate_registry": {
            "columns": [
                "candidate_identity",
                "design_scope",
                "aging_mode",
                "sampling_design_scope"
            ],
            "rows": [
                {
                    "candidate_identity": "fixture_bayes_candidate_a",
                    "design_scope": "Stage_1",
                    "aging_mode": "calendar_aging",
                    "sampling_design_scope": "combined_within_scope"
                }
            ]
        },
        "continuous_trajectory_summaries": {
            "columns": [
                "candidate_identity",
                "outcome",
                "cohort",
                "horizon_axis",
                "condition_protocol_identity",
                "horizon",
                "posterior_median",
                "credible_interval_lower",
                "credible_interval_upper",
                "predictive_interval_lower",
                "predictive_interval_upper",
                "analytical_status",
                "trajectory_target",
                "estimate_label",
                "causal_interpretation"
            ],
            "rows": [
                {
                    "candidate_identity": "fixture_bayes_candidate_a",
                    "outcome": "relative_capacity",
                    "cohort": "primary",
                    "horizon_axis": "elapsed_calendar_days",
                    "condition_protocol_identity": "fixture_cond_a",
                    "horizon": "10.0",
                    "posterior_median": "0.99",
                    "credible_interval_lower": "0.98",
                    "credible_interval_upper": "1.00",
                    "predictive_interval_lower": "0.97",
                    "predictive_interval_upper": "1.01",
                    "analytical_status": "available",
                    "trajectory_target": "latent_reference_cell",
                    "estimate_label": "associational",
                    "causal_interpretation": "not_authorized"
                }
            ]
        },
        "grouped_validation_metrics": {
            "columns": [
                "candidate_identity",
                "outcome",
                "cohort",
                "fold_count",
                "test_cell_count",
                "test_observation_count",
                "equal_cell_weighted_mean_absolute_error",
                "equal_cell_weighted_root_mean_squared_error",
                "equal_cell_weighted_predictive_r_squared",
                "posterior_predictive_interval_coverage",
                "validation_status",
                "validation_reason_code",
                "estimate_label"
            ],
            "rows": [
                {
                    "candidate_identity": "fixture_bayes_candidate_a",
                    "outcome": "relative_capacity",
                    "cohort": "primary",
                    "fold_count": "3",
                    "test_cell_count": "3",
                    "test_observation_count": "6",
                    "equal_cell_weighted_mean_absolute_error": "0.005",
                    "equal_cell_weighted_root_mean_squared_error": "0.006",
                    "equal_cell_weighted_predictive_r_squared": "0.90",
                    "posterior_predictive_interval_coverage": "0.95",
                    "validation_status": "completed",
                    "validation_reason_code": "",
                    "estimate_label": "associational"
                }
            ]
        }
    },
    "predictive_modeling": {
        "candidate_registry": {
            "columns": [
                "benchmark_identity",
                "candidate_identity",
                "model_family",
                "design_scope",
                "aging_mode",
                "outcome",
                "cohort",
                "horizon_axis",
                "analytical_status",
                "analytical_reason_code"
            ],
            "rows": [
                {
                    "benchmark_identity": "fixture_benchmark_a",
                    "candidate_identity": "fixture_bayes_candidate_a",
                    "model_family": "elastic_net",
                    "design_scope": "Stage_1",
                    "aging_mode": "calendar_aging",
                    "outcome": "relative_capacity",
                    "cohort": "primary",
                    "horizon_axis": "elapsed_calendar_days",
                    "analytical_status": "available",
                    "analytical_reason_code": ""
                }
            ]
        },
        "validation_metrics": {
            "columns": [
                "benchmark_identity",
                "model_family",
                "test_cell_count",
                "test_observation_count",
                "equal_cell_weighted_mean_absolute_error",
                "equal_cell_weighted_root_mean_squared_error",
                "equal_cell_weighted_predictive_r_squared",
                "prediction_interval_coverage",
                "analytical_status",
                "analytical_reason_code",
                "estimate_label"
            ],
            "rows": [
                {
                    "benchmark_identity": "fixture_benchmark_a",
                    "model_family": "elastic_net",
                    "test_cell_count": "3",
                    "test_observation_count": "6",
                    "equal_cell_weighted_mean_absolute_error": "0.005",
                    "equal_cell_weighted_root_mean_squared_error": "0.006",
                    "equal_cell_weighted_predictive_r_squared": "0.90",
                    "prediction_interval_coverage": "0.90",
                    "analytical_status": "available",
                    "analytical_reason_code": "",
                    "estimate_label": "predictive"
                }
            ]
        },
        "feature_importance": {
            "columns": [
                "benchmark_identity",
                "model_family",
                "configuration_identity",
                "encoded_feature",
                "feature_group",
                "importance_value",
                "importance_method",
                "importance_status",
                "importance_reason_code",
                "estimate_label"
            ],
            "rows": [
                {
                    "benchmark_identity": "fixture_benchmark_a",
                    "model_family": "elastic_net",
                    "configuration_identity": "fixture_config_a",
                    "encoded_feature": "scaled_horizon",
                    "feature_group": "horizon",
                    "importance_value": "0.01",
                    "importance_method": "absolute_standardized_coefficient",
                    "importance_status": "completed",
                    "importance_reason_code": "",
                    "estimate_label": "predictive_model_inspection"
                }
            ]
        }
    },
    "predictive_sensitivity": {
        "sensitivity_comparisons": {
            "columns": [
                "variant_identity",
                "variant_type",
                "model_family",
                "design_scope",
                "aging_mode",
                "outcome",
                "cohort",
                "horizon_axis",
                "shared_observation_count",
                "shared_independent_cell_count",
                "root_mean_squared_error_difference",
                "mean_absolute_error_difference",
                "predictive_r_squared_difference",
                "exact_inventory_match",
                "variant_analytical_status",
                "comparison_status",
                "comparison_reason_code",
                "estimate_label"
            ],
            "rows": [
                {
                    "variant_identity": "fixture_variant_a",
                    "variant_type": "partition",
                    "model_family": "elastic_net",
                    "design_scope": "Stage_1",
                    "aging_mode": "calendar_aging",
                    "outcome": "relative_capacity",
                    "cohort": "primary",
                    "horizon_axis": "elapsed_calendar_days",
                    "shared_observation_count": "6",
                    "shared_independent_cell_count": "3",
                    "root_mean_squared_error_difference": "0.001",
                    "mean_absolute_error_difference": "0.001",
                    "predictive_r_squared_difference": "0.01",
                    "exact_inventory_match": "true",
                    "variant_analytical_status": "available",
                    "comparison_status": "completed",
                    "comparison_reason_code": "",
                    "estimate_label": "predictive_sensitivity"
                }
            ]
        }
    },
    "robustness_assessment": {
        "assessment_coverage": {
            "columns": [
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
            ],
            "rows": [
                {
                    "assessment_identity": "fixture_assessment",
                    "evidence_source": "validation_summary",
                    "evidence_unit": "validation_summary_record",
                    "total_evidence_count": "1",
                    "completed_evidence_count": "0",
                    "restricted_evidence_count": "1",
                    "unavailable_evidence_count": "0",
                    "assessment_status": "completed_with_restrictions",
                    "assessment_reason_code": "PARTIAL_EVIDENCE_COVERAGE",
                    "interpretation": "Fixture restricted evidence."
                }
            ]
        },
        "structural_limitations": {
            "columns": [
                "limitation_identity",
                "limitation_scope",
                "evidence_source",
                "evidence_record_count",
                "applicability_status",
                "reason_code",
                "interpretation"
            ],
            "rows": [
                {
                    "limitation_identity": "fixture_limitation",
                    "limitation_scope": "global_design",
                    "evidence_source": "registered_policy_restriction",
                    "evidence_record_count": "0",
                    "applicability_status": "unsupported",
                    "reason_code": "CROSS_STAGE_TRANSPORTABILITY_NOT_ESTABLISHED",
                    "interpretation": "Fixture limitation."
                }
            ]
        },
        "external_validity": {
            "columns": [
                "domain_identity",
                "applicability_status",
                "boundary"
            ],
            "rows": [
                {
                    "domain_identity": "fixture_domain",
                    "applicability_status": "supported_with_restrictions",
                    "boundary": "Fixture boundary."
                }
            ]
        },
        "fragility_registry": {
            "columns": [
                "fragility_scope_identity",
                "source_analysis",
                "analytical_status",
                "fragility_indicator_count"
            ],
            "rows": [
                {
                    "fragility_scope_identity": "fixture_fragility_a",
                    "source_analysis": "predictive_modeling",
                    "analytical_status": "available",
                    "fragility_indicator_count": "0"
                }
            ]
        }
    }
}


# Writing one deterministic CSV table
def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)

    with path.open(mode = "w", encoding = "utf-8", newline = "") as handle:
        writer = csv.DictWriter(handle, fieldnames = columns, lineterminator = "\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


# Calculating the SHA-256 digest of one file
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Building one deterministic table manifest
def table_manifest(path: Path, columns: list[str], rows: list[dict]) -> dict:
    return {
        "file_name": path.name,
        "row_count": len(rows),
        "sha256": sha256(path),
        "schema": [{"name": column} for column in columns]
    }


# Building a complete synthetic evidence inventory
def build_evidence_dashboard_fixture(output_directory: Path, source_policy_path: Path) -> tuple[Path, Path]:
    source_policy = json.loads(source_policy_path.read_text(encoding = "utf-8"))
    artifact_root = (output_directory / "artifacts").resolve()
    artifact_root.mkdir(parents = True, exist_ok = True)

    fixture_policy = json.loads(json.dumps(source_policy))

    for source_name, source_contract in source_policy["inputs"].items():
        report_path = (artifact_root / source_contract["report_path"]).resolve()
        report_path.parent.mkdir(parents = True, exist_ok = True)

        table_manifests: dict = {}

        for table_name in source_contract["tables"]:
            spec = TABLE_FIXTURES[source_name][table_name]
            table_path = report_path.parent / f"{table_name}.csv"
            write_csv(table_path, spec["columns"], spec["rows"])
            table_manifests[table_name] = table_manifest(table_path, spec["columns"], spec["rows"])

        report = {
            "tables": table_manifests,
            **REPORT_REQUIRED_FIELDS[source_name]
        }

        report_path.write_text(
            json.dumps(report, indent = 2, sort_keys = True) + "\n",
            encoding = "utf-8"
        )

        fixture_policy["inputs"][source_name]["report_sha256"] = sha256(report_path)

    fixture_policy_path = output_directory / "evidence_dashboard_policy.json"
    fixture_policy_path.write_text(
        json.dumps(fixture_policy, indent = 2, sort_keys = True) + "\n",
        encoding = "utf-8"
    )

    return artifact_root, fixture_policy_path


# Parsing fixture command-line arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Build a deterministic synthetic evidence-dashboard fixture."
    )
    parser.add_argument(
        "--output-directory",
        type = Path,
        required = True
    )
    parser.add_argument(
        "--policy-path",
        type = Path,
        required = True
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()

    build_evidence_dashboard_fixture(
        output_directory = arguments.output_directory,
        source_policy_path = arguments.policy_path
    )
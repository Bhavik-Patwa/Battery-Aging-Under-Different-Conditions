import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
import numpy as np

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_table_manifest.json"
)

DEFAULT_AUDIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dataset_quality_audit.json"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "experimental_design"
)

DEFAULT_ANALYSIS_POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experimental_design_policy.json"
)

OUTCOME_DEFINITIONS = {
    "relative_capacity": {
        "value_field": "relative_capacity",
        "primary_eligibility_field": (
            "relative_capacity_primary_eligible"
        ),
        "sensitivity_eligibility_field": (
            "relative_capacity_sensitivity_eligible"
        )
    },
    "relative_dc_internal_resistance": {
        "value_field": (
            "relative_dc_internal_resistance"
        ),
        "primary_eligibility_field": (
            "relative_dc_internal_resistance_primary_eligible"
        ),
        "sensitivity_eligibility_field": (
            "relative_dc_internal_resistance_sensitivity_eligible"
        )
    }
}

TREATMENT_FIELDS = {
    "calendar_aging": [
        "ambient_temperature_celsius",
        "maximum_state_of_charge_fraction",
        "reference_test_schedule"
    ],
    "cycle_aging": [
        "ambient_temperature_celsius",
        "maximum_state_of_charge_fraction",
        "depth_of_discharge_fraction",
        "charge_rate_c",
        "discharge_rate_c"
    ]
}

ALL_TREATMENT_FIELDS = [
    "ambient_temperature_celsius",
    "maximum_state_of_charge_fraction",
    "depth_of_discharge_fraction",
    "charge_rate_c",
    "discharge_rate_c",
    "reference_test_schedule"
]

QUALITY_DEPENDENCE_FIELDS = [
    "manual_review_reference_present",
    "excluded_reference_present",
    "capacity_unavailable_present",
    "resistance_unavailable_present",
    "elapsed_time_unavailable_present",
    "incomplete_fec_present",
    "primary_capacity_followup_available",
    "primary_resistance_followup_available",
    "sensitivity_eligibility_unresolved"
]

CONTEXT_FIELDS = [
    "dataset_stage",
    "laboratory"
]

DESIGN_PROVENANCE_FIELDS = [
    "sampling_method"
]

EQUIPMENT_FIELDS = {
    "measurement_devices": "measurement_device",
    "test_channels": "test_channel",
    "climate_chambers": "climate_chamber",
    "climate_chamber_positions": (
        "climate_chamber_position"
    )
}

CONTEXT_FAMILY_NONE = "no_additional_context"

CONTEXT_FAMILY_LABORATORY = "laboratory"

TARGET_DESIGN_TERM_TYPES = {
    "main_effect",
    "pairwise_interaction",
    "target_treatment"
}

BASELINE_FIELDS = [
    "mean_capacity_ah",
    "uncorrected_dc_internal_resistance_milliohms",
    "baseline_voltage_volts"
]

CELL_LEVEL_BASELINE_FIELDS = [
    "initial_cell_mass_grams",
    "initial_1khz_resistance_source_value",
    "initial_voltage_volts"
]

CELL_LEVEL_BASELINE_EQUIPMENT_FIELDS = {
    "initial_cell_mass_grams": "initial_cell_mass_scale",
    "initial_1khz_resistance_source_value": (
        "initial_1khz_resistance_meter"
    ),
    "initial_voltage_volts": "initial_voltage_meter"
}

BASELINE_FIELD_ELIGIBILITY_FIELDS = {
    "mean_capacity_ah": {
        "primary": (
            "relative_capacity_primary_eligible"
        ),
        "sensitivity": (
            "relative_capacity_sensitivity_eligible"
        )
    },
    "uncorrected_dc_internal_resistance_milliohms": {
        "primary": (
            "relative_dc_internal_resistance_primary_eligible"
        ),
        "sensitivity": (
            "relative_dc_internal_resistance_sensitivity_eligible"
        )
    },
    "baseline_voltage_volts": {
        "primary": "primary_cohort_eligible",
        "sensitivity": "sensitivity_cohort_eligible"
    },
    "initial_cell_mass_grams": {
        "primary": "primary_cohort_eligible",
        "sensitivity": "sensitivity_cohort_eligible"
    },
    "initial_1khz_resistance_source_value": {
        "primary": "primary_cohort_eligible",
        "sensitivity": "sensitivity_cohort_eligible"
    },
    "initial_voltage_volts": {
        "primary": "primary_cohort_eligible",
        "sensitivity": "sensitivity_cohort_eligible"
    }
}

BASELINE_ANALYSIS_POPULATIONS = [
    "assigned",
    "primary",
    "sensitivity"
]


MINIMUM_REPLICATED_CELL_COUNT = 2

NUMERICAL_TREATMENT_FIELDS = {
    "calendar_aging": [
        "ambient_temperature_celsius",
        "maximum_state_of_charge_fraction"
    ],
    "cycle_aging": [
        "ambient_temperature_celsius",
        "maximum_state_of_charge_fraction",
        "depth_of_discharge_fraction",
        "charge_rate_c",
        "discharge_rate_c"
    ]
}

DESIGN_SPECIFICATIONS = [
    "main_effects",
    "main_effects_and_pairwise_interactions"
]


# Listing context families without combining unrelated allocation dimensions
def build_context_families(aging_mode: str) -> list[str]:
    context_families = [
        CONTEXT_FAMILY_NONE,
        CONTEXT_FAMILY_LABORATORY,
        *(
            f"reference_measurement_{output_field}"
            for output_field in EQUIPMENT_FIELDS
        )
    ]

    if aging_mode == "cycle_aging":
        context_families.extend(
            f"cycle_aging_exposure_{output_field}"
            for output_field in EQUIPMENT_FIELDS
        )

    return context_families


def context_family_is_equipment(context_family: str) -> bool:
    return context_family not in {
        CONTEXT_FAMILY_NONE,
        CONTEXT_FAMILY_LABORATORY
    }


# Parsing experimental-design audit arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = (
            "Auditing experimental support without estimating treatment effects."
        )
    )

    parser.add_argument(
        "--manifest-path",
        type = Path,
        default = DEFAULT_MANIFEST_PATH
    )

    parser.add_argument(
        "--audit-path",
        type = Path,
        default = DEFAULT_AUDIT_PATH
    )

    parser.add_argument(
        "--output-directory",
        type = Path,
        default = DEFAULT_OUTPUT_DIRECTORY
    )

    parser.add_argument(
        "--analysis-policy-path",
        type = Path,
        default = DEFAULT_ANALYSIS_POLICY_PATH,
        help = (
            "JSON policy defining continuous-support or approved-horizon rules without examining outcome direction."
        )
    )

    return parser.parse_args()


# Calculating one file digest without loading it into memory
def calculate_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with file_path.open(
        mode = "rb"
    ) as file_stream:
        while True:
            file_chunk = file_stream.read(
                chunk_size
            )

            if not file_chunk:
                break

            digest.update(
                file_chunk
            )

    return digest.hexdigest()


# Rejecting duplicate keys while loading a JSON document
def reject_duplicate_json_fields(field_pairs: list[tuple[str, object]]) -> dict:
    result = {}

    for field_name, field_value in field_pairs:
        if field_name in result:
            raise RuntimeError(f"JSON document contains a duplicate field : {field_name}")

        result[field_name] = field_value

    return result


# Loading a JSON object while rejecting duplicate keys and non-object roots
def load_strict_json_object(path: Path) -> dict:
    document = json.loads(
        path.read_text(
            encoding = "utf-8"
        ),
        object_pairs_hook = reject_duplicate_json_fields
    )

    if not isinstance(
        document,
        dict
    ):
        raise RuntimeError(f"JSON document must contain an object : {path}")

    return document


# Parsing one nullable serialized boolean
def parse_nullable_boolean(value: str) -> bool | None:
    if value == "true":
        return True

    if value == "false":
        return False

    if value == "":
        return None

    raise RuntimeError(f"Table contains an invalid boolean value : {value!r}")


# Parsing one finite numerical value
def parse_optional_number(value: str) -> float | None:
    if value == "":
        return None

    numerical_value = float(
        value
    )

    if not math.isfinite(
        numerical_value
    ):
        raise RuntimeError(f"Table contains a non-finite value : {value!r}")

    return numerical_value


# Parsing one serialized JSON value
def parse_json_value(value: str) -> object:
    if value == "":
        return None

    return json.loads(
        value
    )


# Loading and validating the feature manifest
def load_feature_manifest(manifest_path: Path) -> dict:
    manifest = load_strict_json_object(
        manifest_path
    )

    if manifest["extraction_status"] != "completed":
        raise RuntimeError("Experimental-design auditing requires a completed feature extraction.")

    required_tables = {
        "cell_features",
        "condition_features",
        "reference_test_features",
        "cycling_exposure_features"
    }

    observed_tables = set(
        manifest["tables"]
    )

    if observed_tables != required_tables:
        raise RuntimeError(f"Feature manifest contains an unexpected table inventory : {sorted(observed_tables)}")

    return manifest


# Loading one manifest-bound analytical table
def load_validated_table(manifest_path: Path, table_name: str,
                         table_manifest: dict
) -> list[dict]:
    table_path = (
        manifest_path.parent
        / table_manifest["file_name"]
    )

    observed_sha256 = calculate_sha256(
        table_path
    )

    if observed_sha256 != table_manifest["sha256"]:
        raise RuntimeError(
            "Analytical-table SHA-256 does not match the manifest : "
            f"{table_name} : {observed_sha256} : {table_manifest['sha256']}"
        )

    with table_path.open(
        mode = "r",
        encoding = "utf-8",
        newline = ""
    ) as table_stream:
        table_reader = csv.DictReader(
            table_stream
        )

        records = list(
            table_reader
        )

        field_names = (
            table_reader.fieldnames
            or []
        )

    expected_field_names = [
        field_schema["name"]
        for field_schema in table_manifest[
            "schema"
        ]
    ]

    if field_names != expected_field_names:
        raise RuntimeError(
            f"Analytical-table schema does not match the manifest : {table_name}"
        )

    if len(records) != table_manifest["row_count"]:
        raise RuntimeError(
            "Analytical-table row count does not match the manifest : "
            f"{table_name} : {len(records)} : {table_manifest['row_count']}"
        )

    return records


# Loading and validating the experimental-design analysis policy
def load_analysis_policy(policy_path: Path) -> dict:
    policy = load_strict_json_object(
        policy_path
    )

    expected_fields = {
        "policy_version",
        "support_mode",
        "coverage_rules",
        "horizon_definitions",
        "calendar_exposure_policy",
        "longitudinal_observation_policy",
        "treatment_evaluation_policy"
    }

    if set(policy) != expected_fields:
        raise RuntimeError(
            "Analysis policy contains missing or unexpected fields : "
            f"{sorted(set(policy).symmetric_difference(expected_fields))}"
        )

    if (
        not isinstance(
            policy["policy_version"],
            str
        )
        or not policy["policy_version"].strip()
    ):
        raise RuntimeError("Analysis policy must contain a nonempty policy_version.")

    support_mode = policy[
        "support_mode"
    ]

    if support_mode not in {
        "within_cell_bracketing_ranges",
        "approved_discrete_horizons"
    }:
        raise RuntimeError(f"Unknown analysis-policy support mode : {support_mode}")

    coverage_rules = policy[
        "coverage_rules"
    ]

    expected_cohorts = {
        "primary",
        "sensitivity"
    }

    if set(coverage_rules) != expected_cohorts:
        raise RuntimeError("Analysis policy must define primary and sensitivity coverage rules.")

    for cohort, rule in coverage_rules.items():
        if set(rule) != {
            "minimum_assigned_cell_coverage_fraction",
            "rationale"
        }:
            raise RuntimeError(f"Coverage rule contains invalid fields : {cohort}")

        raw_fraction = rule[
            "minimum_assigned_cell_coverage_fraction"
        ]

        if isinstance(
            raw_fraction,
            bool
        ):
            raise RuntimeError(f"Coverage fraction must not be boolean : {cohort}")

        minimum_fraction = float(
            raw_fraction
        )

        if not math.isfinite(
            minimum_fraction
        ):
            raise RuntimeError(f"Coverage fraction must be finite : {cohort}")

        if not 0 < minimum_fraction <= 1:
            raise RuntimeError(f"Coverage fraction must be greater than zero and no greater than one : {cohort}")

        if (
            not isinstance(
                rule.get(
                    "rationale"
                ),
                str
            )
            or not rule["rationale"].strip()
        ):
            raise RuntimeError(f"Coverage rule requires a rationale : {cohort}")

    valid_axes = {
        (
            "calendar_aging",
            "elapsed_calendar_days"
        ),
        (
            "cycle_aging",
            "elapsed_calendar_days"
        ),
        (
            "cycle_aging",
            "cumulative_full_equivalent_cycles_observed"
        )
    }

    seen_horizon_keys = set()

    for definition in policy["horizon_definitions"]:
        if set(definition) != {
            "aging_mode",
            "horizon_axis",
            "horizon",
            "analysis_role",
            "interpolation_rule",
            "maximum_bracketing_span",
            "rationale",
            "source"
        }:
            raise RuntimeError("Horizon definition contains invalid fields.")

        key = (
            definition["aging_mode"],
            definition["horizon_axis"],
            float(definition["horizon"]),
            definition["analysis_role"]
        )

        if key in seen_horizon_keys:
            raise RuntimeError(f"Duplicate horizon definition : {key}")

        seen_horizon_keys.add(
            key
        )

    for definition in policy[
        "horizon_definitions"
    ]:
        axis_identity = (
            definition["aging_mode"],
            definition["horizon_axis"]
        )

        if axis_identity not in valid_axes:
            raise RuntimeError(f"Invalid aging-mode and horizon-axis combination : {axis_identity}")

        horizon = float(
            definition["horizon"]
        )

        if not math.isfinite(horizon) or horizon <= 0:
            raise RuntimeError(f"Invalid comparison horizon : {horizon!r}")

        if definition["analysis_role"] not in {
            "primary",
            "secondary",
            "sensitivity"
        }:
            raise RuntimeError(f"Unknown horizon analysis role : {definition['analysis_role']}")

        if definition["interpolation_rule"] not in {
            "exact_observation_only",
            "within_cell_bracketing"
        }:
            raise RuntimeError(f"Unknown interpolation rule : {definition['interpolation_rule']}")

        maximum_bracketing_span = definition[
            "maximum_bracketing_span"
        ]

        if maximum_bracketing_span is not None:
            maximum_bracketing_span = float(
                maximum_bracketing_span
            )

            if (
                not math.isfinite(
                    maximum_bracketing_span
                )
                or maximum_bracketing_span <= 0
            ):
                raise RuntimeError("Maximum bracketing span must be positive and finite.")

        if (
            not isinstance(
                definition["rationale"],
                str
            )
            or not definition["rationale"].strip()
        ):
            raise RuntimeError("Every comparison horizon requires a rationale.")

        if (
            not isinstance(
                definition["source"],
                str
            )
            or not definition["source"].strip()
        ):
            raise RuntimeError("Every comparison horizon requires a source.")

    if (
        support_mode
        == "approved_discrete_horizons"
        and not policy["horizon_definitions"]
    ):
        raise RuntimeError("Discrete-horizon support mode requires at least one approved comparison horizon.")

    if (
        support_mode
        == "within_cell_bracketing_ranges"
        and policy["horizon_definitions"]
    ):
        raise RuntimeError("Continuous-support mode must not contain separately selected numerical horizons.")

    calendar_exposure_policy = policy[
        "calendar_exposure_policy"
    ]

    expected_calendar_policy = {
        "primary_time_axis": "elapsed_calendar_days",
        "primary_estimand": (
            "whole_assigned_protocol_outcome_difference_"
            "at_common_elapsed_calendar_time"
        ),
        "primary_estimand_availability": "available",
        "primary_model_authorization": (
            "permitted_with_restrictions"
        ),
        "net_storage_exposure_estimand": (
            "outcome_difference_at_common_"
            "net_calendar_aging_time"
        ),
        "net_storage_exposure_availability": (
            "unavailable"
        ),
        "net_storage_exposure_model_authorization": (
            "not_authorized"
        ),
        "claim_boundary": (
            "Gross elapsed calendar days support whole-protocol comparisons only. They do not identify time under "
            "storage, isolate calendar-aging exposure, or exclude time spent in reference tests and documented pauses."
        )
    }

    if (
        not isinstance(calendar_exposure_policy, dict)
        or set(calendar_exposure_policy)
        != {
            *expected_calendar_policy,
            "rationale"
        }
    ):
        raise RuntimeError("calendar_exposure_policy contains invalid fields.")

    for field_name, expected_value in (
        expected_calendar_policy.items()
    ):
        if (
            calendar_exposure_policy[field_name]
            != expected_value
        ):
            raise RuntimeError(
                f"Unsupported calendar-exposure policy value : {field_name} : {calendar_exposure_policy[field_name]!r}"
            )

    if (
        not isinstance(
            calendar_exposure_policy["rationale"],
            str
        )
        or not calendar_exposure_policy[
            "rationale"
        ].strip()
    ):
        raise RuntimeError("Calendar-exposure policy requires a nonempty rationale.")

    longitudinal_observation_policy = policy[
        "longitudinal_observation_policy"
    ]

    expected_longitudinal_policy = {
        "support_horizon_requirement": (
            "strictly_positive"
        ),
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
        )
    }

    expected_longitudinal_fields = {
        *expected_longitudinal_policy,
        "rationale"
    }

    if (
        not isinstance(
            longitudinal_observation_policy,
            dict
        )
        or set(longitudinal_observation_policy)
        != expected_longitudinal_fields
    ):
        raise RuntimeError("longitudinal_observation_policy contains invalid fields.")

    for field_name, expected_value in (
        expected_longitudinal_policy.items()
    ):
        if (
            longitudinal_observation_policy[
                field_name
            ]
            != expected_value
        ):
            raise RuntimeError(
                "Unsupported longitudinal-observation policy value : "
                f"{field_name} : {longitudinal_observation_policy[field_name]!r}"
            )

    if (
        not isinstance(
            longitudinal_observation_policy["rationale"],
            str
        )
        or not longitudinal_observation_policy[
            "rationale"
        ].strip()
    ):
        raise RuntimeError("Longitudinal-observation policy requires a nonempty rationale.")

    treatment_evaluation_policy = policy[
        "treatment_evaluation_policy"
    ]

    expected_treatment_evaluation_policy = {
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
        )
    }

    if (
        not isinstance(treatment_evaluation_policy, dict)
        or set(treatment_evaluation_policy)
        != {
            *expected_treatment_evaluation_policy,
            "rationale"
        }
    ):
        raise RuntimeError("treatment_evaluation_policy contains invalid fields.")

    for field_name, expected_value in (
        expected_treatment_evaluation_policy.items()
    ):
        if (
            treatment_evaluation_policy[field_name]
            != expected_value
        ):
            raise RuntimeError(
                "Unsupported treatment-evaluation policy value : "
                f"{field_name} : {treatment_evaluation_policy[field_name]!r}"
            )

    if (
        not isinstance(
            treatment_evaluation_policy["rationale"],
            str
        )
        or not treatment_evaluation_policy[
            "rationale"
        ].strip()
    ):
        raise RuntimeError("Treatment-evaluation policy requires a nonempty rationale.")

    primary_coverage_fraction = float(
        coverage_rules["primary"][
            "minimum_assigned_cell_coverage_fraction"
        ]
    )

    sensitivity_coverage_fraction = float(
        coverage_rules["sensitivity"][
            "minimum_assigned_cell_coverage_fraction"
        ]
    )

    if (
        sensitivity_coverage_fraction
        > primary_coverage_fraction
    ):
        raise RuntimeError("Sensitivity coverage cannot be stricter than primary coverage.")

    policy[
        "support_mode"
    ] = support_mode

    return policy


# Loading every validated input
def load_validated_inputs(manifest_path: Path, audit_path: Path) -> tuple[dict, dict[str, list[dict]], dict]:
    manifest = load_feature_manifest(
        manifest_path
    )

    observed_audit_sha256 = calculate_sha256(
        audit_path
    )

    if observed_audit_sha256 != manifest["audit_sha256"]:
        raise RuntimeError(
            "Data-quality audit SHA-256 does not match the feature manifest : "
            f"{observed_audit_sha256} : {manifest['audit_sha256']}"
        )

    audit_result = load_strict_json_object(
        audit_path
    )

    if (
        audit_result["dataset"]
        != manifest["dataset"]
    ):
        raise RuntimeError("Feature manifest and data-quality audit identify different datasets.")

    if (
        manifest.get("quality_contract")
        != audit_result.get("quality_contract")
    ):
        raise RuntimeError("Feature manifest and quality audit have different exact quality-contract bindings.")

    if (
        manifest.get("data_quality_contract_version")
        != audit_result.get("contract_version")
    ):
        raise RuntimeError("Feature manifest and quality audit have different contract versions.")

    reconciliation = manifest.get(
        "source_measurement_reconciliation"
    )

    if (
        not isinstance(
            reconciliation,
            dict
        )
        or reconciliation.get(
            "reconciliation_complete"
        )
        is not True
    ):
        raise RuntimeError("Feature manifest does not contain completed source-measurement reconciliation.")

    if (
        reconciliation["reconciled_measurement_count"]
        != reconciliation["audited_measurement_count"]
    ):
        raise RuntimeError("Feature manifest source-measurement counts do not reconcile.")

    if (
        audit_result["archive_audit_failure_count"] != 0
        or audit_result["measurement_audit_failure_count"] != 0
        or audit_result.get(
            "incident_rule_coverage_error"
        )
        is not None
    ):
        raise RuntimeError("Experimental-design auditing requires a completed quality audit without execution failures.")

    tables = {
        table_name: load_validated_table(
            manifest_path = manifest_path,
            table_name = table_name,
            table_manifest = table_manifest
        )
        for table_name, table_manifest in (
            manifest["tables"].items()
        )
    }

    return manifest, tables, audit_result


# Confirming finite feature availability without using outcome direction
def feature_available(value: str) -> bool:
    numerical_value = parse_optional_number(
        value
    )

    return numerical_value is not None


# Confirming positive finite exposure availability
def positive_feature_available(value: str) -> bool:
    numerical_value = parse_optional_number(
        value
    )

    return (
        numerical_value is not None
        and numerical_value > 0
    )


# Determining positive follow-up horizon availability
def record_has_positive_horizon(record: dict) -> bool:
    elapsed_calendar_days = parse_optional_number(
        record[
            "elapsed_calendar_days"
        ]
    )

    cumulative_fec = parse_optional_number(
        record[
            "cumulative_full_equivalent_cycles_observed"
        ]
    )

    elapsed_support_available = (
        record[
            "elapsed_calendar_time_status"
        ]
        == "completed"
        and elapsed_calendar_days is not None
        and elapsed_calendar_days > 0
    )

    fec_support_available = (
        parse_nullable_boolean(
            record[
                "cumulative_fec_observation_complete"
            ]
        )
        is True
        and cumulative_fec is not None
        and cumulative_fec > 0
    )

    return (
        elapsed_support_available
        or fec_support_available
    )


# Grouping records by physical cell
def group_records_by_cell(records: list[dict]) -> dict[str, list[dict]]:
    records_by_cell = defaultdict(
        list
    )

    for record in records:
        records_by_cell[
            record["cell_identity"]
        ].append(
            record
        )

    return dict(
        records_by_cell
    )


# Validating cell, condition and schedule relationships
def validate_design_dimensions(cell_records: list[dict], condition_records: list[dict],
                               reference_records: list[dict]
) -> None:
    if len(cell_records) != 279:
        raise RuntimeError(f"Experimental-design audit requires 279 cells : {len(cell_records)}")

    if len(condition_records) != 93:
        raise RuntimeError(f"Experimental-design audit requires 93 condition protocols : {len(condition_records)}")

    cell_identities = [
        record["cell_identity"]
        for record in cell_records
    ]

    if len(cell_identities) != len(
        set(
            cell_identities
        )
    ):
        raise RuntimeError("Cell feature table contains duplicate identities.")

    condition_identities = {
        record["condition_protocol_identity"]
        for record in condition_records
    }

    if any(
        record["condition_protocol_identity"]
        not in condition_identities
        for record in cell_records
    ):
        raise RuntimeError("Cell feature table references an unknown condition protocol.")

    represented_cell_count = sum(
        int(
            record["cell_count"]
        )
        for record in condition_records
    )

    if represented_cell_count != len(
        cell_records
    ):
        raise RuntimeError(
            "Condition protocols do not reconcile to the cell inventory : "
            f"{represented_cell_count} : {len(cell_records)}"
        )

    reference_records_by_cell = group_records_by_cell(
        reference_records
    )

    schedule_conflicts = []

    for cell_record in cell_records:
        cell_identity = cell_record[
            "cell_identity"
        ]

        cell_reference_records = (
            reference_records_by_cell.get(
                cell_identity,
                []
            )
        )

        observed_protocols = {
            record["reference_test_protocol"]
            for record in cell_reference_records
        }

        has_intermediate_reference_test = bool(
            observed_protocols.intersection(
                {
                    "CU",
                    "exCU"
                }
            )
        )

        assigned_schedule = cell_record[
            "reference_test_schedule"
        ]

        if (
            assigned_schedule
            == "periodic_reference_tests"
            and not has_intermediate_reference_test
        ):
            schedule_conflicts.append(
                cell_identity
            )

        if (
            assigned_schedule
            == "initial_and_final_only"
            and has_intermediate_reference_test
        ):
            schedule_conflicts.append(
                cell_identity
            )

    if schedule_conflicts:
        raise RuntimeError(f"Reference-test schedule conflicts with the observed inventory : {schedule_conflicts[:20]}")


# Collecting validated and identity-conflicted equipment separately
def build_equipment_inventory(audit_result: dict) -> dict:
    equipment_by_cell = defaultdict(
        lambda: {
            equipment_role: {
                evidence_status: {
                    output_field: set()
                    for output_field in EQUIPMENT_FIELDS
                }
                for evidence_status in [
                    "validated",
                    "identity_conflicted"
                ]
            }
            for equipment_role in [
                "reference_measurement",
                "cycle_aging_exposure"
            ]
        }
    )

    for archive_record in audit_result[
        "archives"
    ]:
        for measurement_record in archive_record[
            "measurements"
        ]:
            companion_metadata = measurement_record.get(
                "companion_metadata"
            )

            if companion_metadata is None:
                continue

            source_identity = measurement_record[
                "source_identity"
            ]

            cell_identity = source_identity.rsplit(
                "/",
                1
            )[0]

            experimental_role = (
                "cycle_aging_exposure"
                if source_identity.endswith(
                    "_ZYK.csv"
                )
                else "reference_measurement"
            )

            identity_conflicted = any(
                quality_reason["reason_code"]
                == "COMPANION_METADATA_IDENTITY_MISMATCH"
                for quality_reason in measurement_record[
                    "quality_reasons"
                ]
            )

            evidence_status = (
                "identity_conflicted"
                if identity_conflicted
                else "validated"
            )

            for (
                output_field,
                metadata_field
            ) in EQUIPMENT_FIELDS.items():
                equipment_value = companion_metadata.get(
                    metadata_field
                )

                if equipment_value:
                    equipment_by_cell[
                        cell_identity
                    ][
                        experimental_role
                    ][
                        evidence_status
                    ][
                        output_field
                    ].add(
                        equipment_value
                    )

    return equipment_by_cell


# Summarizing validated and conflicted equipment evidence separately
def summarize_role_separated_equipment(cell_identities: set[str], equipment_by_cell: dict,
                                       aging_mode: str
) -> dict:
    equipment_summary = {}

    for equipment_role in [
        "reference_measurement",
        "cycle_aging_exposure"
    ]:
        for evidence_status in [
            "validated",
            "identity_conflicted"
        ]:
            for output_field in EQUIPMENT_FIELDS:
                equipment_values_by_cell = {
                    cell_identity: set(
                        equipment_by_cell.get(
                            cell_identity,
                            {}
                        ).get(
                            equipment_role,
                            {}
                        ).get(
                            evidence_status,
                            {}
                        ).get(
                            output_field,
                            set()
                        )
                    )
                    for cell_identity in cell_identities
                }

                equipment_values = sorted(
                    {
                        equipment_value
                        for cell_values in (
                            equipment_values_by_cell.values()
                        )
                        for equipment_value in cell_values
                    }
                )

                equipment_cell_counts = Counter(
                    equipment_value
                    for cell_values in (
                        equipment_values_by_cell.values()
                    )
                    for equipment_value in cell_values
                )

                field_prefix = (
                    equipment_role
                    if evidence_status == "validated"
                    else (
                        f"{equipment_role}_identity_conflicted"
                    )
                )

                equipment_summary[
                    f"{field_prefix}_{output_field}"
                ] = equipment_values

                equipment_summary[
                    f"{field_prefix}_{output_field}_cell_counts"
                ] = dict(
                    sorted(
                        equipment_cell_counts.items()
                    )
                )

    equipment_summary[
        "calendar_aging_exposure_equipment_status"
    ] = (
        "unavailable_from_measurement_files"
        if aging_mode == "calendar_aging"
        else "not_applicable"
    )

    return equipment_summary


# Selecting one record for an outcome, cohort and horizon axis
def record_supports_horizon(record: dict, outcome_definition: dict,
                            cohort_name: str, horizon_axis: str
) -> bool:
    if not feature_available(
        record[
            outcome_definition[
                "value_field"
            ]
        ]
    ):
        return False

    eligibility_field = (
        outcome_definition[
            "primary_eligibility_field"
        ]
        if cohort_name == "primary"
        else outcome_definition[
            "sensitivity_eligibility_field"
        ]
    )

    if parse_nullable_boolean(
        record[
            eligibility_field
        ]
    ) is not True:
        return False

    if horizon_axis == "elapsed_calendar_days":
        if (
            record["elapsed_calendar_time_status"]
            != "completed"
        ):
            return False

        horizon_value = parse_optional_number(
            record["elapsed_calendar_days"]
        )

    elif (
        horizon_axis
        == "cumulative_full_equivalent_cycles_observed"
    ):
        cumulative_eligibility_field = (
            "cumulative_fec_primary_eligible"
            if cohort_name == "primary"
            else "cumulative_fec_sensitivity_eligible"
        )

        if (
            parse_nullable_boolean(
                record[
                    "cumulative_fec_observation_complete"
                ]
            )
            is not True
            or parse_nullable_boolean(
                record[
                    cumulative_eligibility_field
                ]
            )
            is not True
        ):
            return False

        horizon_value = parse_optional_number(
            record[
                "cumulative_full_equivalent_cycles_observed"
            ]
        )

    else:
        raise RuntimeError(
            f"Unknown horizon axis : {horizon_axis}"
        )

    return (
        horizon_value is not None
        and horizon_value > 0
    )


# Calculating the minimum supporting-cell count implied by one coverage rule
def calculate_required_support_cell_count(assigned_cell_count: int, minimum_assigned_cell_coverage_fraction: float) -> int:
    if assigned_cell_count <= 0:
        return 0

    return max(
        1,
        math.ceil(
            minimum_assigned_cell_coverage_fraction
            * assigned_cell_count
            - 1e-12
        )
    )


# Calculating continuous support intervals from per-cell horizon ranges
def calculate_supported_intervals(horizon_values_by_cell: dict[str, list[float]], assigned_cell_count: int,
                                  minimum_assigned_cell_coverage_fraction: float
) -> list[dict]:
    required_support_cell_count = (
        calculate_required_support_cell_count(
            assigned_cell_count = assigned_cell_count,
            minimum_assigned_cell_coverage_fraction = (
                minimum_assigned_cell_coverage_fraction
            )
        )
    )

    cell_ranges = [
        (
            min(
                horizon_values
            ),
            max(
                horizon_values
            )
        )
        for horizon_values in horizon_values_by_cell.values()
        if horizon_values
    ]

    if (
        not cell_ranges
        or required_support_cell_count <= 0
    ):
        return []

    boundaries = sorted(
        {
            boundary
            for lower_horizon, upper_horizon in cell_ranges
            for boundary in (
                lower_horizon,
                upper_horizon
            )
        }
    )

    def supporting_cell_count(horizon: float) -> int:
        return sum(
            lower_horizon
            <= horizon
            <= upper_horizon
            for lower_horizon, upper_horizon in cell_ranges
        )

    supported_pieces = []

    for boundary in boundaries:
        if (
            supporting_cell_count(
                boundary
            )
            >= required_support_cell_count
        ):
            supported_pieces.append(
                (
                    boundary,
                    boundary
                )
            )

    for lower_boundary, upper_boundary in zip(
        boundaries,
        boundaries[1:]
    ):
        midpoint = (
            lower_boundary
            + upper_boundary
        ) / 2

        if (
            supporting_cell_count(
                midpoint
            )
            >= required_support_cell_count
        ):
            supported_pieces.append(
                (
                    lower_boundary,
                    upper_boundary
                )
            )

    if not supported_pieces:
        return []

    merged_intervals = []

    for lower_horizon, upper_horizon in sorted(
        supported_pieces
    ):
        if (
            not merged_intervals
            or lower_horizon
            > merged_intervals[-1][1]
        ):
            merged_intervals.append(
                [
                    lower_horizon,
                    upper_horizon
                ]
            )

        else:
            merged_intervals[-1][1] = max(
                merged_intervals[-1][1],
                upper_horizon
            )

    return [
        {
            "minimum": lower_horizon,
            "maximum": upper_horizon,
            "required_support_cell_count": (
                required_support_cell_count
            ),
            "assigned_cell_count": assigned_cell_count,
            "minimum_assigned_cell_coverage_fraction": (
                minimum_assigned_cell_coverage_fraction
            ),
            "support_interpretation": (
                "within_observed_cell_ranges_without_extrapolation"
            )
        }
        for lower_horizon, upper_horizon in merged_intervals
    ]


# Calculating observed ranges and assigned-cell horizon coverage
def summarize_common_horizon_support(horizon_values_by_cell: dict[str, list[float]], assigned_cell_identities: set[str],
                                     candidate_horizons: list[float], minimum_assigned_cell_coverage_fraction: float
) -> dict:
    assigned_cell_count = len(
        assigned_cell_identities
    )

    eligible_cell_count = len(
        horizon_values_by_cell
    )

    eligible_cell_fraction = (
        eligible_cell_count
        / assigned_cell_count
        if assigned_cell_count > 0
        else 0.0
    )

    if eligible_cell_count == 0:
        return {
            "assigned_cell_count": assigned_cell_count,
            "eligible_cell_count": 0,
            "eligible_assigned_cell_fraction": 0.0,
            "observation_count": 0,
            "observed_horizon_minimum": None,
            "observed_horizon_maximum": None,
            "complete_eligible_cell_bracketing_minimum": None,
            "complete_eligible_cell_bracketing_maximum": None,
            "exact_common_horizons": [],
            "observed_horizon_coverage_inventory": [],
            "coverage_eligible_observed_horizons": [],
            "required_support_cell_count": (
                calculate_required_support_cell_count(
                    assigned_cell_count = assigned_cell_count,
                    minimum_assigned_cell_coverage_fraction = (
                        minimum_assigned_cell_coverage_fraction
                    )
                )
            ),
            "supported_bracketing_ranges": []
        }

    ordered_values_by_cell = {
        cell_identity: sorted(
            set(
                horizon_values
            )
        )
        for cell_identity, horizon_values in (
            horizon_values_by_cell.items()
        )
    }

    observed_horizon_values = sorted(
        {
            horizon_value
            for horizon_values in (
                ordered_values_by_cell.values()
            )
            for horizon_value in horizon_values
        }
    )

    complete_eligible_minimum = max(
        horizon_values[0]
        for horizon_values in (
            ordered_values_by_cell.values()
        )
    )

    complete_eligible_maximum = min(
        horizon_values[-1]
        for horizon_values in (
            ordered_values_by_cell.values()
        )
    )

    if (
        complete_eligible_minimum
        > complete_eligible_maximum
    ):
        complete_eligible_minimum = None
        complete_eligible_maximum = None

    exact_common_horizons = sorted(
        set.intersection(
            *(
                set(
                    horizon_values
                )
                for horizon_values in (
                    ordered_values_by_cell.values()
                )
            )
        )
    )

    required_support_cell_count = (
        calculate_required_support_cell_count(
            assigned_cell_count = assigned_cell_count,
            minimum_assigned_cell_coverage_fraction = (
                minimum_assigned_cell_coverage_fraction
            )
        )
    )

    observed_horizon_coverage_inventory = []

    for candidate_horizon in sorted(
        set(
            candidate_horizons
        )
    ):
        bracketing_cell_count = sum(
            horizon_values[0]
            <= candidate_horizon
            <= horizon_values[-1]
            for horizon_values in (
                ordered_values_by_cell.values()
            )
        )

        assigned_cell_coverage_fraction = (
            bracketing_cell_count
            / assigned_cell_count
            if assigned_cell_count > 0
            else 0.0
        )

        observed_horizon_coverage_inventory.append(
            {
                "horizon": candidate_horizon,
                "bracketing_assigned_cell_count": (
                    bracketing_cell_count
                ),
                "assigned_cell_count": (
                    assigned_cell_count
                ),
                "assigned_cell_coverage_fraction": (
                    assigned_cell_coverage_fraction
                ),
                "required_support_cell_count": (
                    required_support_cell_count
                ),
                "coverage_threshold_met": (
                    bracketing_cell_count
                    >= required_support_cell_count
                )
            }
        )

    coverage_eligible_observed_horizons = [
        record["horizon"]
        for record in observed_horizon_coverage_inventory
        if record["coverage_threshold_met"]
    ]

    supported_bracketing_ranges = (
        calculate_supported_intervals(
            horizon_values_by_cell = (
                ordered_values_by_cell
            ),
            assigned_cell_count = assigned_cell_count,
            minimum_assigned_cell_coverage_fraction = (
                minimum_assigned_cell_coverage_fraction
            )
        )
    )

    return {
        "assigned_cell_count": assigned_cell_count,
        "eligible_cell_count": eligible_cell_count,
        "eligible_assigned_cell_fraction": (
            eligible_cell_fraction
        ),
        "observation_count": sum(
            len(
                horizon_values
            )
            for horizon_values in (
                ordered_values_by_cell.values()
            )
        ),
        "observed_horizon_minimum": (
            observed_horizon_values[0]
        ),
        "observed_horizon_maximum": (
            observed_horizon_values[-1]
        ),
        "complete_eligible_cell_bracketing_minimum": (
            complete_eligible_minimum
        ),
        "complete_eligible_cell_bracketing_maximum": (
            complete_eligible_maximum
        ),
        "exact_common_horizons": (
            exact_common_horizons
        ),
        "observed_horizon_coverage_inventory": (
            observed_horizon_coverage_inventory
        ),
        "coverage_eligible_observed_horizons": (
            coverage_eligible_observed_horizons
        ),
        "required_support_cell_count": (
            required_support_cell_count
        ),
        "supported_bracketing_ranges": (
            supported_bracketing_ranges
        )
    }


# Intersecting two continuous support interval inventories
def intersect_supported_intervals(first_intervals: list[dict], second_intervals: list[dict]) -> list[dict]:
    intersections = []

    for first_interval in first_intervals:
        for second_interval in second_intervals:
            lower_horizon = max(
                first_interval["minimum"],
                second_interval["minimum"]
            )

            upper_horizon = min(
                first_interval["maximum"],
                second_interval["maximum"]
            )

            if lower_horizon > upper_horizon:
                continue

            intersections.append(
                {
                    "minimum": lower_horizon,
                    "maximum": upper_horizon,
                    "first_required_support_cell_count": (
                        first_interval[
                            "required_support_cell_count"
                        ]
                    ),
                    "second_required_support_cell_count": (
                        second_interval[
                            "required_support_cell_count"
                        ]
                    ),
                    "support_interpretation": (
                        "shared_continuous_observed_support_without_"
                        "extrapolation"
                    )
                }
            )

    merged_intersections = []

    for interval in sorted(
        intersections,
        key = lambda record: (
            record["minimum"],
            record["maximum"]
        )
    ):
        if (
            not merged_intersections
            or interval["minimum"]
            > merged_intersections[-1]["maximum"]
        ):
            merged_intersections.append(
                dict(
                    interval
                )
            )

        else:
            merged_intersections[-1][
                "maximum"
            ] = max(
                merged_intersections[-1]["maximum"],
                interval["maximum"]
            )

    return merged_intersections


# Collecting positive observed horizons without using outcome values
def build_observed_horizon_inventory(cell_records: list[dict], reference_records: list[dict]) -> dict[tuple[str, str], list[float]]:
    aging_mode_by_cell = {
        record["cell_identity"]: record["aging_mode"]
        for record in cell_records
    }

    candidates = defaultdict(
        set
    )

    for record in reference_records:
        aging_mode = aging_mode_by_cell.get(
            record["cell_identity"]
        )

        if aging_mode is None:
            continue

        if (
            record["elapsed_calendar_time_status"]
            == "completed"
        ):
            elapsed_days = parse_optional_number(
                record["elapsed_calendar_days"]
            )

            if (
                elapsed_days is not None
                and elapsed_days > 0
            ):
                candidates[
                    (
                        aging_mode,
                        "elapsed_calendar_days"
                    )
                ].add(
                    elapsed_days
                )

        if (
            aging_mode == "cycle_aging"
            and parse_nullable_boolean(
                record[
                    "cumulative_fec_observation_complete"
                ]
            )
            is True
        ):
            cumulative_fec = parse_optional_number(
                record[
                    "cumulative_full_equivalent_cycles_observed"
                ]
            )

            if (
                cumulative_fec is not None
                and cumulative_fec > 0
            ):
                candidates[
                    (
                        aging_mode,
                        "cumulative_full_equivalent_cycles_observed"
                    )
                ].add(
                    cumulative_fec
                )

    return {
        key: sorted(
            values
        )
        for key, values in candidates.items()
    }


# Indexing approved discrete horizons declared by the analysis policy
def build_approved_horizon_inventory(analysis_policy: dict) -> dict[tuple[str, str], list[dict]]:
    inventory = defaultdict(
        list
    )

    for definition in analysis_policy[
        "horizon_definitions"
    ]:
        inventory[
            (
                definition["aging_mode"],
                definition["horizon_axis"]
            )
        ].append(
            {
                **definition,
                "horizon": float(
                    definition["horizon"]
                )
            }
        )

    return {
        key: sorted(
            definitions,
            key = lambda record: (
                record["horizon"],
                record["analysis_role"]
            )
        )
        for key, definitions in inventory.items()
    }


# Evaluating one cell's support for one approved comparison horizon
def evaluate_cell_horizon_support(horizon_values: list[float], horizon: float,
                                  interpolation_rule: str, maximum_bracketing_span: float | None
) -> dict:
    ordered_values = sorted(
        set(
            horizon_values
        )
    )

    if horizon in ordered_values:
        return {
            "supported": True,
            "support_type": "exact_observation",
            "lower_horizon": horizon,
            "upper_horizon": horizon,
            "bracketing_span": 0.0
        }

    if interpolation_rule == "exact_observation_only":
        return {
            "supported": False,
            "support_type": "no_exact_observation",
            "lower_horizon": None,
            "upper_horizon": None,
            "bracketing_span": None
        }

    lower_values = [
        value
        for value in ordered_values
        if value < horizon
    ]

    upper_values = [
        value
        for value in ordered_values
        if value > horizon
    ]

    if not lower_values or not upper_values:
        return {
            "supported": False,
            "support_type": "not_bracketed",
            "lower_horizon": None,
            "upper_horizon": None,
            "bracketing_span": None
        }

    lower_horizon = max(
        lower_values
    )

    upper_horizon = min(
        upper_values
    )

    bracketing_span = (
        upper_horizon
        - lower_horizon
    )

    if (
        maximum_bracketing_span is not None
        and bracketing_span
        > maximum_bracketing_span
    ):
        return {
            "supported": False,
            "support_type": "bracketing_span_exceeds_policy",
            "lower_horizon": lower_horizon,
            "upper_horizon": upper_horizon,
            "bracketing_span": bracketing_span
        }

    return {
        "supported": True,
        "support_type": "within_cell_bracketing",
        "lower_horizon": lower_horizon,
        "upper_horizon": upper_horizon,
        "bracketing_span": bracketing_span
    }


# Building condition-specific horizon support
def build_horizon_support(cell_records: list[dict], reference_records: list[dict],
                          observed_horizon_inventory: dict[tuple[str, str], list[float]],
                          coverage_rules: dict
) -> list[dict]:
    cells_by_condition = defaultdict(
        list
    )

    for cell_record in cell_records:
        cells_by_condition[
            cell_record[
                "condition_protocol_identity"
            ]
        ].append(
            cell_record
        )

    reference_records_by_cell = group_records_by_cell(
        reference_records
    )

    horizon_records = []

    for (
        condition_identity,
        condition_cells
    ) in cells_by_condition.items():
        aging_mode = condition_cells[0][
            "aging_mode"
        ]

        horizon_axes = [
            "elapsed_calendar_days"
        ]

        if aging_mode == "cycle_aging":
            horizon_axes.append(
                "cumulative_full_equivalent_cycles_observed"
            )

        assigned_cell_identities = {
            record["cell_identity"]
            for record in condition_cells
        }

        for (
            outcome_name,
            outcome_definition
        ) in OUTCOME_DEFINITIONS.items():
            for cohort_name in [
                "primary",
                "sensitivity"
            ]:
                for horizon_axis in horizon_axes:
                    horizon_values_by_cell = {}

                    for cell_record in condition_cells:
                        cell_identity = cell_record[
                            "cell_identity"
                        ]

                        horizon_values = []

                        for reference_record in (
                            reference_records_by_cell.get(
                                cell_identity,
                                []
                            )
                        ):
                            if not record_supports_horizon(
                                record = reference_record,
                                outcome_definition = (
                                    outcome_definition
                                ),
                                cohort_name = cohort_name,
                                horizon_axis = horizon_axis
                            ):
                                continue

                            horizon_values.append(
                                float(
                                    reference_record[
                                        horizon_axis
                                    ]
                                )
                            )

                        if horizon_values:
                            horizon_values_by_cell[
                                cell_identity
                            ] = horizon_values

                    horizon_summary = (
                        summarize_common_horizon_support(
                            horizon_values_by_cell = (
                                horizon_values_by_cell
                            ),
                            assigned_cell_identities = (
                                assigned_cell_identities
                            ),
                            candidate_horizons = (
                                observed_horizon_inventory.get(
                                    (
                                        aging_mode,
                                        horizon_axis
                                    ),
                                    []
                                )
                            ),
                            minimum_assigned_cell_coverage_fraction = (
                                coverage_rules[
                                    cohort_name
                                ][
                                    "minimum_assigned_cell_coverage_fraction"
                                ]
                            )
                        )
                    )

                    horizon_records.append(
                        {
                            "condition_protocol_identity": (
                                condition_identity
                            ),
                            "aging_mode": aging_mode,
                            "outcome": outcome_name,
                            "cohort": cohort_name,
                            "horizon_axis": horizon_axis,
                            "assigned_cell_count": len(
                                condition_cells
                            ),
                            **horizon_summary
                        }
                    )

    return sorted(
        horizon_records,
        key = lambda record: (
            record["aging_mode"],
            record["condition_protocol_identity"],
            record["outcome"],
            record["cohort"],
            record["horizon_axis"]
        )
    )


# Building treatment replication and cohort accounting
def build_treatment_support(cell_records: list[dict], condition_records: list[dict],
                            reference_records: list[dict], equipment_by_cell: dict
) -> list[dict]:
    cells_by_condition = defaultdict(
        list
    )

    for cell_record in cell_records:
        cells_by_condition[
            cell_record[
                "condition_protocol_identity"
            ]
        ].append(
            cell_record
        )

    reference_records_by_cell = group_records_by_cell(
        reference_records
    )

    treatment_records = []

    for condition_record in condition_records:
        condition_identity = condition_record[
            "condition_protocol_identity"
        ]

        condition_cells = cells_by_condition[
            condition_identity
        ]

        condition_cell_identities = {
            cell_record["cell_identity"]
            for cell_record in condition_cells
        }

        condition_reference_records = [
            reference_record
            for cell_identity in condition_cell_identities
            for reference_record in (
                reference_records_by_cell.get(
                    cell_identity,
                    []
                )
            )
        ]

        condition_followup_records = [
            record
            for record in condition_reference_records
            if record_has_positive_horizon(
                record
            )
        ]

        disposition_counts = Counter(
            reference_record["disposition"]
            for reference_record in (
                condition_reference_records
            )
        )

        reason_counts = Counter()

        for reference_record in condition_reference_records:
            for reason_code in (
                parse_json_value(
                    reference_record[
                        "quality_reason_codes"
                    ]
                )
                or []
            ):
                reason_counts[
                    reason_code
                ] += 1

        equipment_summary = summarize_role_separated_equipment(
            cell_identities = condition_cell_identities,
            equipment_by_cell = equipment_by_cell,
            aging_mode = condition_record["aging_mode"]
        )

        treatment_records.append(
            {
                **condition_record,
                "assigned_cell_count": len(
                    condition_cells
                ),
                "reference_observation_count": len(
                    condition_reference_records
                ),
                "primary_capacity_cell_count": len(
                    {
                        record["cell_identity"]
                        for record in condition_followup_records
                        if (
                            feature_available(
                                record["relative_capacity"]
                            )
                            and parse_nullable_boolean(
                                record[
                                    "relative_capacity_primary_eligible"
                                ]
                            )
                            is True
                        )
                    }
                ),
                "primary_resistance_cell_count": len(
                    {
                        record["cell_identity"]
                        for record in condition_followup_records
                        if (
                            feature_available(
                                record[
                                    "relative_dc_internal_resistance"
                                ]
                            )
                            and parse_nullable_boolean(
                                record[
                                    "relative_dc_internal_resistance_primary_eligible"
                                ]
                            )
                            is True
                        )
                    }
                ),
                "complete_fec_cell_count": len(
                    {
                        record["cell_identity"]
                        for record in condition_reference_records
                        if (
                            parse_nullable_boolean(
                                record[
                                    "cumulative_fec_observation_complete"
                                ]
                            )
                            is True
                            and positive_feature_available(
                                record[
                                    "cumulative_full_equivalent_cycles_observed"
                                ]
                            )
                        )
                    }
                ),
                "disposition_counts": dict(
                    sorted(
                        disposition_counts.items()
                    )
                ),
                "quality_reason_counts": dict(
                    sorted(
                        reason_counts.items()
                    )
                ),
                **equipment_summary
            }
        )

    return sorted(
        treatment_records,
        key = lambda record: (
            record["aging_mode"],
            record["condition_protocol_identity"]
        )
    )


# Building one deterministic treatment-combination identity
def build_treatment_combination_identity(record: dict) -> str:
    aging_mode = record[
        "aging_mode"
    ]

    identity_fields = list(
        TREATMENT_FIELDS[
            aging_mode
        ]
    )

    if (
        "reference_test_schedule"
        not in identity_fields
    ):
        identity_fields.append(
            "reference_test_schedule"
        )

    identity_components = [
        aging_mode,
        *(
            f"{field_name}={record[field_name]}"
            for field_name in identity_fields
        )
    ]

    return "/".join(
        identity_components
    )


# Summarizing outcome availability and cohort eligibility
def summarize_outcome_accounting(reference_records: list[dict], outcome_name: str,
                                 outcome_definition: dict
) -> dict:
    reference_records = [
        record
        for record in reference_records
        if record_has_positive_horizon(
            record
        )
    ]

    available_records = [
        record
        for record in reference_records
        if feature_available(
            record[
                outcome_definition[
                    "value_field"
                ]
            ]
        )
    ]

    primary_records = [
        record
        for record in available_records
        if parse_nullable_boolean(
            record[
                outcome_definition[
                    "primary_eligibility_field"
                ]
            ]
        )
        is True
    ]

    sensitivity_true_records = [
        record
        for record in available_records
        if parse_nullable_boolean(
            record[
                outcome_definition[
                    "sensitivity_eligibility_field"
                ]
            ]
        )
        is True
    ]

    sensitivity_false_records = [
        record
        for record in available_records
        if parse_nullable_boolean(
            record[
                outcome_definition[
                    "sensitivity_eligibility_field"
                ]
            ]
        )
        is False
    ]

    sensitivity_unresolved_records = [
        record
        for record in available_records
        if parse_nullable_boolean(
            record[
                outcome_definition[
                    "sensitivity_eligibility_field"
                ]
            ]
        )
        is None
    ]

    return {
        f"{outcome_name}_available_observation_count": len(
            available_records
        ),
        f"{outcome_name}_unavailable_observation_count": (
            len(reference_records)
            - len(available_records)
        ),
        f"{outcome_name}_primary_observation_count": len(
            primary_records
        ),
        f"{outcome_name}_primary_cell_count": len(
            {
                record["cell_identity"]
                for record in primary_records
            }
        ),
        f"{outcome_name}_sensitivity_observation_count": len(
            sensitivity_true_records
        ),
        f"{outcome_name}_sensitivity_cell_count": len(
            {
                record["cell_identity"]
                for record in sensitivity_true_records
            }
        ),
        f"{outcome_name}_sensitivity_false_observation_count": len(
            sensitivity_false_records
        ),
        f"{outcome_name}_sensitivity_unresolved_observation_count": len(
            sensitivity_unresolved_records
        ),
        f"{outcome_name}_sensitivity_unresolved_cell_count": len(
            {
                record["cell_identity"]
                for record in sensitivity_unresolved_records
            }
        )
    }


# Building support for unique treatment combinations
def build_treatment_combination_support(cell_records: list[dict], reference_records: list[dict],
                                        equipment_by_cell: dict
) -> list[dict]:
    cells_by_treatment_combination = defaultdict(
        list
    )

    for cell_record in cell_records:
        treatment_combination_identity = (
            build_treatment_combination_identity(
                cell_record
            )
        )

        cells_by_treatment_combination[
            treatment_combination_identity
        ].append(
            cell_record
        )

    reference_records_by_cell = group_records_by_cell(
        reference_records
    )

    support_records = []

    for (
        treatment_combination_identity,
        combination_cells
    ) in cells_by_treatment_combination.items():
        first_cell = combination_cells[0]

        cell_identities = {
            cell_record["cell_identity"]
            for cell_record in combination_cells
        }

        condition_identities = sorted(
            {
                cell_record[
                    "condition_protocol_identity"
                ]
                for cell_record in combination_cells
            }
        )

        combination_reference_records = [
            reference_record
            for cell_identity in sorted(
                cell_identities
            )
            for reference_record in (
                reference_records_by_cell.get(
                    cell_identity,
                    []
                )
            )
        ]

        disposition_counts = Counter(
            record["disposition"]
            for record in combination_reference_records
        )

        quality_reason_counts = Counter()

        for reference_record in combination_reference_records:
            for reason_code in (
                parse_json_value(
                    reference_record[
                        "quality_reason_codes"
                    ]
                )
                or []
            ):
                quality_reason_counts[
                    reason_code
                ] += 1

        equipment_summary = summarize_role_separated_equipment(
            cell_identities = cell_identities,
            equipment_by_cell = equipment_by_cell,
            aging_mode = first_cell["aging_mode"]
        )

        outcome_accounting = {}

        for (
            outcome_name,
            outcome_definition
        ) in OUTCOME_DEFINITIONS.items():
            outcome_accounting.update(
                summarize_outcome_accounting(
                    reference_records = (
                        combination_reference_records
                    ),
                    outcome_name = outcome_name,
                    outcome_definition = (
                        outcome_definition
                    )
                )
            )

        positive_complete_fec_records = [
            record
            for record in combination_reference_records
            if (
                parse_nullable_boolean(
                    record[
                        "cumulative_fec_observation_complete"
                    ]
                )
                is True
                and positive_feature_available(
                    record[
                        "cumulative_full_equivalent_cycles_observed"
                    ]
                )
            )
        ]

        incomplete_fec_records = [
            record
            for record in combination_reference_records
            if record[
                "cumulative_fec_status"
            ]
            == "incomplete"
        ]

        support_records.append(
            {
                "treatment_combination_identity": (
                    treatment_combination_identity
                ),
                "aging_mode": first_cell[
                    "aging_mode"
                ],
                **{
                    field_name: (
                        first_cell[field_name]
                        if field_name in (
                            TREATMENT_FIELDS[
                                first_cell[
                                    "aging_mode"
                                ]
                            ]
                            + [
                                "reference_test_schedule"
                            ]
                        )
                        else None
                    )
                    for field_name in (
                        ALL_TREATMENT_FIELDS
                    )
                },
                "condition_protocol_identities": (
                    condition_identities
                ),
                "condition_protocol_count": len(
                    condition_identities
                ),
                "independent_cell_count": len(
                    cell_identities
                ),
                "stage_counts": dict(
                    sorted(
                        Counter(
                            cell_record[
                                "dataset_stage"
                            ]
                            for cell_record in combination_cells
                        ).items()
                    )
                ),
                "laboratory_counts": dict(
                    sorted(
                        Counter(
                            cell_record[
                                "laboratory"
                            ]
                            for cell_record in combination_cells
                        ).items()
                    )
                ),
                "sampling_method_counts": dict(
                    sorted(
                        Counter(
                            cell_record[
                                "sampling_method"
                            ]
                            for cell_record in combination_cells
                        ).items()
                    )
                ),
                "reference_observation_count": len(
                    combination_reference_records
                ),
                "elapsed_time_unavailable_observation_count": sum(
                    record[
                        "elapsed_calendar_time_status"
                    ]
                    != "completed"
                    for record in combination_reference_records
                ),
                "positive_complete_fec_observation_count": len(
                    positive_complete_fec_records
                ),
                "positive_complete_fec_cell_count": len(
                    {
                        record["cell_identity"]
                        for record in positive_complete_fec_records
                    }
                ),
                "incomplete_fec_observation_count": len(
                    incomplete_fec_records
                ),
                "incomplete_fec_cell_count": len(
                    {
                        record["cell_identity"]
                        for record in incomplete_fec_records
                    }
                ),
                "disposition_counts": dict(
                    sorted(
                        disposition_counts.items()
                    )
                ),
                "quality_reason_counts": dict(
                    sorted(
                        quality_reason_counts.items()
                    )
                ),
                **outcome_accounting,
                **equipment_summary
            }
        )

    return sorted(
        support_records,
        key = lambda record: (
            record["aging_mode"],
            record[
                "treatment_combination_identity"
            ]
        )
    )


# Building cell-level quality and availability indicators
def build_cell_design_records(cell_records: list[dict], reference_records: list[dict]) -> list[dict]:
    reference_records_by_cell = group_records_by_cell(
        reference_records
    )

    design_records = []

    for cell_record in cell_records:
        cell_reference_records = (
            reference_records_by_cell.get(
                cell_record[
                    "cell_identity"
                ],
                []
            )
        )

        followup_records = [
            record
            for record in cell_reference_records
            if int(
                record["sequence_number"]
            ) > 1
        ]

        design_records.append(
            {
                **cell_record,
                "manual_review_reference_present": str(
                    any(
                        record["disposition"]
                        == "manual_review"
                        for record in cell_reference_records
                    )
                ).lower(),
                "excluded_reference_present": str(
                    any(
                        record["disposition"]
                        == "exclude"
                        for record in cell_reference_records
                    )
                ).lower(),
                "capacity_unavailable_present": str(
                    any(
                        not feature_available(
                            record[
                                "relative_capacity"
                            ]
                        )
                        for record in followup_records
                    )
                ).lower(),
                "resistance_unavailable_present": str(
                    any(
                        not feature_available(
                            record[
                                "relative_dc_internal_resistance"
                            ]
                        )
                        for record in followup_records
                    )
                ).lower(),
                "elapsed_time_unavailable_present": str(
                    any(
                        record[
                            "elapsed_calendar_time_status"
                        ]
                        != "completed"
                        for record in followup_records
                    )
                ).lower(),
                "incomplete_fec_present": str(
                    any(
                        record[
                            "cumulative_fec_status"
                        ]
                        == "incomplete"
                        for record in followup_records
                    )
                ).lower(),
                "primary_capacity_followup_available": str(
                    any(
                        feature_available(
                            record[
                                "relative_capacity"
                            ]
                        )
                        and parse_nullable_boolean(
                            record[
                                "relative_capacity_primary_eligible"
                            ]
                        )
                        is True
                        for record in followup_records
                    )
                ).lower(),
                "primary_resistance_followup_available": str(
                    any(
                        feature_available(
                            record[
                                "relative_dc_internal_resistance"
                            ]
                        )
                        and parse_nullable_boolean(
                            record[
                                "relative_dc_internal_resistance_primary_eligible"
                            ]
                        )
                        is True
                        for record in followup_records
                    )
                ).lower(),
                "sensitivity_eligibility_unresolved": str(
                    any(
                        (
                            feature_available(
                                record[
                                    "relative_capacity"
                                ]
                            )
                            and parse_nullable_boolean(
                                record[
                                    "relative_capacity_sensitivity_eligible"
                                ]
                            )
                            is None
                        )
                        or (
                            feature_available(
                                record[
                                    "relative_dc_internal_resistance"
                                ]
                            )
                            and parse_nullable_boolean(
                                record[
                                    "relative_dc_internal_resistance_sensitivity_eligible"
                                ]
                            )
                            is None
                        )
                        for record in followup_records
                    )
                ).lower()
            }
        )

    return design_records


# Calculating a pooled standardized mean difference between two groups
def calculate_standardized_mean_difference(first_values: list[float], second_values: list[float]) -> float | None:
    if len(first_values) < 2 or len(second_values) < 2:
        return None

    first_variance = float(
        np.var(
            first_values,
            ddof = 1
        )
    )

    second_variance = float(
        np.var(
            second_values,
            ddof = 1
        )
    )

    pooled_variance = (
        (
            (len(first_values) - 1)
            * first_variance
            + (len(second_values) - 1)
            * second_variance
        )
        / (
            len(first_values)
            + len(second_values)
            - 2
        )
    )

    if pooled_variance <= 0:
        return None

    return (
        float(np.mean(first_values))
        - float(np.mean(second_values))
    ) / math.sqrt(
        pooled_variance
    )


# Building baseline comparability separately for assigned and eligible populations
def build_baseline_balance_diagnostics(cell_records: list[dict], reference_records: list[dict]) -> list[dict]:
    baseline_records = [
        record
        for record in reference_records
        if int(
            record["sequence_number"]
        ) == 1
    ]

    baseline_cell_identities = [
        record["cell_identity"]
        for record in baseline_records
    ]

    if len(
        baseline_cell_identities
    ) != len(
        set(
            baseline_cell_identities
        )
    ):
        raise RuntimeError("Baseline inventory contains duplicate cell identities.")

    baseline_by_cell = {
        record["cell_identity"]: record
        for record in baseline_records
    }

    eligibility_fields = sorted(
        {
            eligibility_field
            for population_fields in (
                BASELINE_FIELD_ELIGIBILITY_FIELDS.values()
            )
            for eligibility_field in (
                population_fields.values()
            )
        }
    )

    cell_records_with_baseline = [
        {
            **cell_record,
            **{
                field_name: (
                    baseline_by_cell.get(
                        cell_record["cell_identity"],
                        {}
                    ).get(
                        field_name
                    )
                )
                for field_name in [
                    *BASELINE_FIELDS,
                    *eligibility_fields
                ]
            }
        }
        for cell_record in cell_records
    ]

    grouping_specifications = [
        (
            "condition_protocol",
            [
                "condition_protocol_identity"
            ]
        ),
        (
            "test_point",
            [
                "dataset_stage",
                "aging_type",
                "test_point_number"
            ]
        ),
        (
            "stage",
            [
                "dataset_stage"
            ]
        ),
        (
            "laboratory",
            [
                "laboratory"
            ]
        ),
        (
            "sampling_method",
            [
                "sampling_method"
            ]
        ),
        (
            "stage_sampling_method",
            [
                "dataset_stage",
                "sampling_method"
            ]
        )
    ]

    def record_is_population_eligible(record: dict, baseline_field: str,
                                      analysis_population: str
    ) -> bool:
        if analysis_population == "assigned":
            return True

        eligibility_field = (
            BASELINE_FIELD_ELIGIBILITY_FIELDS[
                baseline_field
            ][
                analysis_population
            ]
        )

        serialized_value = record.get(
            eligibility_field
        )

        if serialized_value in {
            None,
            ""
        }:
            return False

        return (
            parse_nullable_boolean(
                serialized_value
            )
            is True
        )

    def parse_baseline_value(record: dict, baseline_field: str) -> float | None:
        serialized_value = record.get(
            baseline_field
        )

        if serialized_value in {
            None,
            ""
        }:
            return None

        return parse_optional_number(
            serialized_value
        )

    def collect_measurement_equipment_values(records: list[dict], baseline_field: str) -> tuple[str | None, list[str]]:
        equipment_field = (
            CELL_LEVEL_BASELINE_EQUIPMENT_FIELDS.get(
                baseline_field
            )
        )

        if equipment_field is None:
            return None, []

        equipment_values = sorted(
            {
                str(record[equipment_field])
                for record in records
                if record.get(equipment_field) not in {
                    None,
                    ""
                }
            }
        )

        return equipment_field, equipment_values

    def classify_measurement_equipment_relationship(first_values: list[str], second_values: list[str]) -> str:
        if not first_values or not second_values:
            return "unavailable"

        first_set = set(first_values)
        second_set = set(second_values)

        if first_set == second_set:
            return "identical"

        if first_set.intersection(second_set):
            return "overlapping"

        return "disjoint"

    diagnostic_records = []

    for (
        grouping_name,
        grouping_fields
    ) in grouping_specifications:
        records_by_group = defaultdict(
            list
        )

        for cell_record in cell_records_with_baseline:
            group_key = tuple(
                cell_record[
                    field_name
                ]
                for field_name in grouping_fields
            )

            records_by_group[
                group_key
            ].append(
                cell_record
            )

        for group_key, group_records in sorted(
            records_by_group.items()
        ):
            for baseline_field in [
                *BASELINE_FIELDS,
                *CELL_LEVEL_BASELINE_FIELDS
            ]:
                for analysis_population in (
                    BASELINE_ANALYSIS_POPULATIONS
                ):
                    population_records = [
                        record
                        for record in group_records
                        if record_is_population_eligible(
                            record = record,
                            baseline_field = baseline_field,
                            analysis_population = (
                                analysis_population
                            )
                        )
                    ]

                    observed_values = [
                        numerical_value
                        for numerical_value in (
                            parse_baseline_value(
                                record = record,
                                baseline_field = (
                                    baseline_field
                                )
                            )
                            for record in (
                                population_records
                            )
                        )
                        if numerical_value is not None
                    ]

                    if not observed_values:
                        continue

                    eligibility_field = (
                        None
                        if analysis_population
                        == "assigned"
                        else (
                            BASELINE_FIELD_ELIGIBILITY_FIELDS[
                                baseline_field
                            ][
                                analysis_population
                            ]
                        )
                    )

                    (
                        measurement_equipment_field,
                        group_measurement_equipment_values
                    ) = collect_measurement_equipment_values(
                        records = population_records,
                        baseline_field = baseline_field
                    )

                    diagnostic_records.append(
                        {
                            "record_type": (
                                "baseline_group_summary"
                            ),
                            "analysis_population": (
                                analysis_population
                            ),
                            "eligibility_field": (
                                eligibility_field
                            ),
                            "grouping_name": (
                                grouping_name
                            ),
                            "group_identity": "/".join(
                                (
                                    f"{field_name}="
                                    f"{value}"
                                )
                                for field_name, value in zip(
                                    grouping_fields,
                                    group_key
                                )
                            ),
                            "baseline_field": (
                                baseline_field
                            ),
                            "available_cell_count": len(
                                observed_values
                            ),
                            "population_eligible_cell_count": (
                                len(
                                    population_records
                                )
                            ),
                            "assigned_cell_count": len(
                                group_records
                            ),
                            "mean_value": float(
                                np.mean(
                                    observed_values
                                )
                            ),
                            "standard_deviation": (
                                float(
                                    np.std(
                                        observed_values,
                                        ddof = 1
                                    )
                                )
                                if len(
                                    observed_values
                                ) > 1
                                else None
                            ),
                            "minimum_value": float(
                                np.min(
                                    observed_values
                                )
                            ),
                            "maximum_value": float(
                                np.max(
                                    observed_values
                                )
                            ),
                            "first_condition_mean": None,
                            "second_condition_mean": None,
                            "mean_difference": None,
                            "absolute_difference": None,
                            "standardized_mean_difference": (
                                None
                            ),
                            "first_aging_mode": None,
                            "second_aging_mode": None,
                            "first_dataset_stage": None,
                            "second_dataset_stage": None,
                            "comparison_scope_status": (
                                "not_applicable_group_summary"
                            ),
                            "comparison_scope_authorization": (
                                "descriptive_group_summary"
                            ),
                            "measurement_equipment_field": (
                                measurement_equipment_field
                            ),
                            "group_measurement_equipment_values": (
                                group_measurement_equipment_values
                            ),
                            "first_measurement_equipment_values": [],
                            "second_measurement_equipment_values": [],
                            "measurement_equipment_relationship": (
                                "descriptive_group_summary"
                                if measurement_equipment_field
                                is not None
                                else "not_applicable"
                            )
                        }
                    )

    condition_scope_by_identity = {}

    for record in cell_records_with_baseline:
        condition_identity = record[
            "condition_protocol_identity"
        ]

        observed_scope = (
            record["aging_mode"],
            record["dataset_stage"]
        )

        existing_scope = condition_scope_by_identity.get(
            condition_identity
        )

        if (
            existing_scope is not None
            and existing_scope != observed_scope
        ):
            raise RuntimeError(
                f"Condition protocol spans inconsistent aging-mode or stage assignments : {condition_identity}"
            )

        condition_scope_by_identity[
            condition_identity
        ] = observed_scope

    condition_identities = sorted(
        {
            record[
                "condition_protocol_identity"
            ]
            for record in (
                cell_records_with_baseline
            )
        }
    )

    for first_condition, second_condition in combinations(
        condition_identities,
        2
    ):
        (
            first_aging_mode,
            first_dataset_stage
        ) = condition_scope_by_identity[
            first_condition
        ]

        (
            second_aging_mode,
            second_dataset_stage
        ) = condition_scope_by_identity[
            second_condition
        ]

        same_aging_mode = (
            first_aging_mode
            == second_aging_mode
        )

        same_dataset_stage = (
            first_dataset_stage
            == second_dataset_stage
        )

        if same_aging_mode and same_dataset_stage:
            comparison_scope_status = (
                "eligible_within_aging_mode_and_stage"
            )
            comparison_scope_authorization = (
                "eligible_for_baseline_balance_interpretation"
            )

        elif not same_aging_mode and not same_dataset_stage:
            comparison_scope_status = (
                "cross_aging_mode_and_stage"
            )
            comparison_scope_authorization = (
                "descriptive_only_prohibited_for_treatment_support"
            )

        elif not same_aging_mode:
            comparison_scope_status = (
                "cross_aging_mode"
            )
            comparison_scope_authorization = (
                "descriptive_only_prohibited_for_treatment_support"
            )

        else:
            comparison_scope_status = (
                "cross_stage"
            )
            comparison_scope_authorization = (
                "descriptive_only_prohibited_for_treatment_support"
            )
        first_assigned_records = [
            record
            for record in cell_records_with_baseline
            if record[
                "condition_protocol_identity"
            ]
            == first_condition
        ]

        second_assigned_records = [
            record
            for record in cell_records_with_baseline
            if record[
                "condition_protocol_identity"
            ]
            == second_condition
        ]

        for baseline_field in [
            *BASELINE_FIELDS,
            *CELL_LEVEL_BASELINE_FIELDS
        ]:
            for analysis_population in (
                BASELINE_ANALYSIS_POPULATIONS
            ):
                first_population_records = [
                    record
                    for record in (
                        first_assigned_records
                    )
                    if record_is_population_eligible(
                        record = record,
                        baseline_field = (
                            baseline_field
                        ),
                        analysis_population = (
                            analysis_population
                        )
                    )
                ]

                second_population_records = [
                    record
                    for record in (
                        second_assigned_records
                    )
                    if record_is_population_eligible(
                        record = record,
                        baseline_field = (
                            baseline_field
                        ),
                        analysis_population = (
                            analysis_population
                        )
                    )
                ]

                first_values = [
                    numerical_value
                    for numerical_value in (
                        parse_baseline_value(
                            record = record,
                            baseline_field = (
                                baseline_field
                            )
                        )
                        for record in (
                            first_population_records
                        )
                    )
                    if numerical_value is not None
                ]

                second_values = [
                    numerical_value
                    for numerical_value in (
                        parse_baseline_value(
                            record = record,
                            baseline_field = (
                                baseline_field
                            )
                        )
                        for record in (
                            second_population_records
                        )
                    )
                    if numerical_value is not None
                ]

                if (
                    not first_values
                    or not second_values
                ):
                    continue

                mean_difference = (
                    float(
                        np.mean(
                            first_values
                        )
                    )
                    - float(
                        np.mean(
                            second_values
                        )
                    )
                )

                absolute_difference = abs(
                    mean_difference
                )

                standardized_difference = (
                    calculate_standardized_mean_difference(
                        first_values = first_values,
                        second_values = (
                            second_values
                        )
                    )
                )

                eligibility_field = (
                    None
                    if analysis_population
                    == "assigned"
                    else (
                        BASELINE_FIELD_ELIGIBILITY_FIELDS[
                            baseline_field
                        ][
                            analysis_population
                        ]
                    )
                )

                (
                    measurement_equipment_field,
                    first_measurement_equipment_values
                ) = collect_measurement_equipment_values(
                    records = first_population_records,
                    baseline_field = baseline_field
                )

                (
                    _,
                    second_measurement_equipment_values
                ) = collect_measurement_equipment_values(
                    records = second_population_records,
                    baseline_field = baseline_field
                )

                measurement_equipment_relationship = (
                    classify_measurement_equipment_relationship(
                        first_values = (
                            first_measurement_equipment_values
                        ),
                        second_values = (
                            second_measurement_equipment_values
                        )
                    )
                    if measurement_equipment_field is not None
                    else "not_applicable"
                )

                diagnostic_records.append(
                    {
                        "record_type": (
                            "condition_pair_difference"
                        ),
                        "analysis_population": (
                            analysis_population
                        ),
                        "eligibility_field": (
                            eligibility_field
                        ),
                        "grouping_name": (
                            "condition_protocol_pair"
                        ),
                        "group_identity": (
                            f"{first_condition}|"
                            f"{second_condition}"
                        ),
                        "baseline_field": (
                            baseline_field
                        ),
                        "available_cell_count": (
                            len(
                                first_values
                            )
                            + len(
                                second_values
                            )
                        ),
                        "population_eligible_cell_count": (
                            len(
                                first_population_records
                            )
                            + len(
                                second_population_records
                            )
                        ),
                        "assigned_cell_count": (
                            len(
                                first_assigned_records
                            )
                            + len(
                                second_assigned_records
                            )
                        ),
                        "mean_value": None,
                        "standard_deviation": None,
                        "minimum_value": None,
                        "maximum_value": None,
                        "first_condition_mean": float(
                            np.mean(
                                first_values
                            )
                        ),
                        "second_condition_mean": float(
                            np.mean(
                                second_values
                            )
                        ),
                        "mean_difference": (
                            mean_difference
                        ),
                        "absolute_difference": (
                            absolute_difference
                        ),
                        "standardized_mean_difference": (
                            standardized_difference
                        ),
                        "first_aging_mode": (
                            first_aging_mode
                        ),
                        "second_aging_mode": (
                            second_aging_mode
                        ),
                        "first_dataset_stage": (
                            first_dataset_stage
                        ),
                        "second_dataset_stage": (
                            second_dataset_stage
                        ),
                        "comparison_scope_status": (
                            comparison_scope_status
                        ),
                        "comparison_scope_authorization": (
                            comparison_scope_authorization
                        ),
                        "measurement_equipment_field": (
                            measurement_equipment_field
                        ),
                        "group_measurement_equipment_values": [],
                        "first_measurement_equipment_values": (
                            first_measurement_equipment_values
                        ),
                        "second_measurement_equipment_values": (
                            second_measurement_equipment_values
                        ),
                        "measurement_equipment_relationship": (
                            measurement_equipment_relationship
                        )
                    }
                )

    return sorted(
        diagnostic_records,
        key = lambda record: (
            record["analysis_population"],
            record["record_type"],
            record["grouping_name"],
            record["group_identity"],
            record["baseline_field"]
        )
    )


# Measuring deterministic categorical dependence
def measure_categorical_dependence(records: list[dict], first_field: str, second_field: str) -> dict:
    first_to_second = defaultdict(
        Counter
    )

    second_to_first = defaultdict(
        Counter
    )

    for record in records:
        first_value = record[
            first_field
        ]

        second_value = record[
            second_field
        ]

        first_to_second[
            first_value
        ][
            second_value
        ] += 1

        second_to_first[
            second_value
        ][
            first_value
        ] += 1

    record_count = len(
        records
    )

    first_to_second_purity = (
        sum(
            max(
                value_counts.values()
            )
            for value_counts in (
                first_to_second.values()
            )
        )
        / record_count
    )

    second_to_first_purity = (
        sum(
            max(
                value_counts.values()
            )
            for value_counts in (
                second_to_first.values()
            )
        )
        / record_count
    )

    return {
        "first_field": first_field,
        "second_field": second_field,
        "first_level_count": len(
            first_to_second
        ),
        "second_level_count": len(
            second_to_first
        ),
        "first_determines_second": all(
            len(
                value_counts
            )
            == 1
            for value_counts in (
                first_to_second.values()
            )
        ),
        "second_determines_first": all(
            len(
                value_counts
            )
            == 1
            for value_counts in (
                second_to_first.values()
            )
        ),
        "first_to_second_purity": (
            first_to_second_purity
        ),
        "second_to_first_purity": (
            second_to_first_purity
        )
    }


# Building treatment dependence with design-quality indicators
def build_treatment_dependence(cell_records: list[dict], reference_records: list[dict]) -> list[dict]:
    design_records = build_cell_design_records(
        cell_records = cell_records,
        reference_records = reference_records
    )

    dependence_records = []

    for aging_mode in [
        "calendar_aging",
        "cycle_aging"
    ]:
        mode_records = [
            record
            for record in design_records
            if record[
                "aging_mode"
            ]
            == aging_mode
        ]

        treatment_fields = TREATMENT_FIELDS[
            aging_mode
        ]

        dependence_pairs = {
            tuple(
                sorted(
                    (
                        first_field,
                        second_field
                    )
                )
            )
            for first_field, second_field in combinations(
                treatment_fields,
                2
            )
        }

        dependence_pairs.update(
            tuple(
                sorted(
                    (
                        treatment_field,
                        comparison_field
                    )
                )
            )
            for treatment_field in treatment_fields
            for comparison_field in [
                *CONTEXT_FIELDS,
                *DESIGN_PROVENANCE_FIELDS,
                *QUALITY_DEPENDENCE_FIELDS
            ]
        )

        for first_field, second_field in sorted(
            dependence_pairs
        ):
            dependence_records.append(
                {
                    "aging_mode": aging_mode,
                    **measure_categorical_dependence(
                        records = mode_records,
                        first_field = first_field,
                        second_field = second_field
                    )
                }
            )

    return sorted(
        dependence_records,
        key = lambda record: (
            record["aging_mode"],
            record["first_field"],
            record["second_field"]
        )
    )


# Classifying analytical support independently of causal identification
def classify_estimand_support(changed_treatment_count: int, context_difference_count: int,
                              equipment_disjoint_dimension_count: int, equipment_identity_conflicted_dimension_count: int,
                              minimum_eligible_cell_count: int, observed_horizon_overlap_available: bool,
                              common_support_interval_available: bool,
                              target_estimand_available: bool
) -> tuple[str, str]:
    if changed_treatment_count < 1:
        raise ValueError("A treatment contrast must change at least one treatment.")

    if not observed_horizon_overlap_available:
        return (
            "unsupported",
            "no_observed_condition_level_horizon_overlap"
        )

    if not common_support_interval_available:
        return (
            "unsupported",
            "no_common_assigned_cell_support_interval"
        )

    if minimum_eligible_cell_count < MINIMUM_REPLICATED_CELL_COUNT:
        return (
            "descriptive_only",
            "unreplicated_observed_contrast"
        )

    if not target_estimand_available:
        return (
            "unsupported",
            "target_estimand_unavailable"
        )

    if context_difference_count > 0:
        return (
            "descriptive_only",
            "replicated_contrast_with_structural_design_context_difference"
        )

    equipment_restrictions_present = (
        equipment_disjoint_dimension_count > 0
        or equipment_identity_conflicted_dimension_count > 0
    )

    if changed_treatment_count == 1:
        detail = (
            "replicated_one_factor_analytical_contrast_with_equipment_allocation_restrictions"
            if equipment_restrictions_present
            else
            "replicated_one_factor_analytical_contrast_with_restrictions"
        )

    else:
        detail = (
            "replicated_joint_condition_analytical_contrast_with_equipment_allocation_restrictions"
            if equipment_restrictions_present
            else
            "replicated_joint_condition_analytical_contrast_with_restrictions"
        )

    return (
        "supported_with_restrictions",
        detail
    )


# Comparing equipment overlap without equating imbalance with aliasing
def compare_equipment_allocation(first_support: dict, second_support: dict,
                                 equipment_fields: list[tuple[str, str]]
) -> list[dict]:
    allocation_records = []

    for equipment_field, difference_code in equipment_fields:
        first_validated = set(
            first_support[
                equipment_field
            ]
        )

        second_validated = set(
            second_support[
                equipment_field
            ]
        )

        conflicted_field = equipment_field.replace(
            "_measurement_devices",
            "_identity_conflicted_measurement_devices"
        ).replace(
            "_test_channels",
            "_identity_conflicted_test_channels"
        ).replace(
            "_climate_chambers",
            "_identity_conflicted_climate_chambers"
        ).replace(
            "_climate_chamber_positions",
            "_identity_conflicted_climate_chamber_positions"
        )

        first_conflicted = set(
            first_support[
                conflicted_field
            ]
        )

        second_conflicted = set(
            second_support[
                conflicted_field
            ]
        )

        if first_conflicted or second_conflicted:
            relationship = "identity_conflicted"

        elif (
            not first_validated
            or not second_validated
        ):
            relationship = "unavailable"

        elif not first_validated.intersection(
            second_validated
        ):
            relationship = "disjoint"

        elif first_validated == second_validated:
            relationship = "identical_or_shared"

        else:
            relationship = "overlapping"

        allocation_records.append(
            {
                "equipment_field": equipment_field,
                "difference_code": difference_code,
                "relationship": relationship,
                "first_values": sorted(
                    first_validated
                ),
                "second_values": sorted(
                    second_validated
                )
            }
        )

    return allocation_records


# Separating descriptive, one-factor-structure and joint-condition permissions
def build_question_disposition(record: dict) -> dict:
    bracketing_support_available = bool(
        record[
            "permitted_bracketing_ranges"
        ]
    )

    replicated = min(
        record["first_eligible_cell_count"],
        record["second_eligible_cell_count"]
    ) >= MINIMUM_REPLICATED_CELL_COUNT

    analytically_supported = (
        record["direct_contrast_support_status"]
        == "supported_with_restrictions"
        and record.get(
            "target_estimand_available",
            False
        )
    )

    one_factor_contrast_supported = (
        analytically_supported
        and len(record["changed_treatments"]) == 1
        and record.get(
            "equipment_disjoint_dimension_count",
            0
        ) == 0
        and record.get(
            "equipment_identity_conflicted_dimension_count",
            0
        ) == 0
    )

    joint_condition_supported = (
        analytically_supported
        and len(record["changed_treatments"]) >= 1
    )

    return {
        "descriptive_analysis_status": (
            "permitted_with_restrictions"
            if bracketing_support_available
            and replicated
            else "blocked"
        ),
        "condition_comparison_interpretation": (
            "one_factor_structural_contrast"
            if len(record["changed_treatments"]) == 1
            else "joint_treatment_combination_contrast"
        ),
        "joint_condition_comparison_status": (
            "permitted_with_restrictions"
            if joint_condition_supported
            else "not_supported"
        ),
        "one_factor_contrast_structure_status": (
            "permitted_with_restrictions"
            if one_factor_contrast_supported
            else "not_supported"
        ),
        "one_factor_contrast_identification_status": (
            "not_established_assignment_mechanism_undocumented"
        ),
        "causal_identification_status": (
            "not_established"
        ),
        "cross_stage_transportability_status": (
            "not_applicable"
        )
    }


# Selecting supported cohort-specific condition-pair decisions for one aging mode
def direct_supported_records(records: list[dict], aging_mode: str) -> list[dict]:
    return [
        record
        for record in records
        if (
            record["aging_mode"] == aging_mode
            and record["final_support_classification"]
            in {
                "supported_with_restrictions",
                "sensitivity_only"
            }
        )
    ]


# Counting distinct supported condition-pair questions across both cohorts
def direct_supported_question_count(records: list[dict]) -> int:
    return len(
        {
            (
                record["first_condition_protocol_identity"],
                record["second_condition_protocol_identity"],
                record["outcome"],
                record["horizon_axis"]
            )
            for record in records
        }
    )


# Building explicit support decisions for every condition comparison
def build_estimand_support(condition_records: list[dict], treatment_support_records: list[dict],
                           horizon_support_records: list[dict], coverage_rules: dict,
                           analysis_policy: dict
) -> list[dict]:
    if analysis_policy is None:
        raise RuntimeError("Estimand support requires the analysis policy.")

    if coverage_rules is None:
        raise RuntimeError("Estimand support requires cohort-specific coverage rules.")

    treatment_support_by_condition = {
        record[
            "condition_protocol_identity"
        ]: record
        for record in treatment_support_records
    }

    horizon_support_by_key = {
        (
            record["condition_protocol_identity"],
            record["outcome"],
            record["cohort"],
            record["horizon_axis"]
        ): record
        for record in horizon_support_records
    }

    conditions_by_mode = defaultdict(
        list
    )

    for condition_record in condition_records:
        conditions_by_mode[
            condition_record[
                "aging_mode"
            ]
        ].append(
            condition_record
        )

    estimand_records = []

    for (
        aging_mode,
        mode_conditions
    ) in conditions_by_mode.items():
        treatment_fields = TREATMENT_FIELDS[
            aging_mode
        ]

        horizon_axes = [
            "elapsed_calendar_days"
        ]

        if aging_mode == "cycle_aging":
            horizon_axes.append(
                "cumulative_full_equivalent_cycles_observed"
            )

        for (
            first_condition,
            second_condition
        ) in combinations(
            sorted(
                mode_conditions,
                key = lambda record: record[
                    "condition_protocol_identity"
                ]
            ),
            2
        ):
            changed_treatments = [
                field_name
                for field_name in treatment_fields
                if (
                    first_condition[field_name]
                    != second_condition[field_name]
                )
            ]

            if not changed_treatments:
                continue

            context_differences = [
                field_name
                for field_name in CONTEXT_FIELDS
                if (
                    first_condition[field_name]
                    != second_condition[field_name]
                )
            ]

            # Recording design provenance separately from outcome-generating context
            design_provenance_differences = [
                field_name
                for field_name in (
                    DESIGN_PROVENANCE_FIELDS
                )
                if (
                    first_condition[field_name]
                    != second_condition[field_name]
                )
            ]

            first_support = (
                treatment_support_by_condition[
                    first_condition[
                        "condition_protocol_identity"
                    ]
                ]
            )

            second_support = (
                treatment_support_by_condition[
                    second_condition[
                        "condition_protocol_identity"
                    ]
                ]
            )

            recorded_equipment_differences = []

            # Comparing every recorded equipment-allocation dimension
            equipment_comparison_fields = [
                (
                    "reference_measurement_measurement_devices",
                    "REFERENCE_MEASUREMENT_DEVICE_ALLOCATION_DIFFERS"
                ),
                (
                    "reference_measurement_test_channels",
                    "REFERENCE_MEASUREMENT_CHANNEL_ALLOCATION_DIFFERS"
                ),
                (
                    "reference_measurement_climate_chambers",
                    "REFERENCE_MEASUREMENT_CHAMBER_ALLOCATION_DIFFERS"
                ),
                (
                    "reference_measurement_climate_chamber_positions",
                    "REFERENCE_MEASUREMENT_CHAMBER_POSITION_DIFFERS"
                )
            ]

            if aging_mode == "cycle_aging":
                equipment_comparison_fields.extend(
                    [
                        (
                            "cycle_aging_exposure_measurement_devices",
                            "CYCLE_EXPOSURE_DEVICE_ALLOCATION_DIFFERS"
                        ),
                        (
                            "cycle_aging_exposure_test_channels",
                            "CYCLE_EXPOSURE_CHANNEL_ALLOCATION_DIFFERS"
                        ),
                        (
                            "cycle_aging_exposure_climate_chambers",
                            "CYCLE_EXPOSURE_CHAMBER_ALLOCATION_DIFFERS"
                        ),
                        (
                            "cycle_aging_exposure_climate_chamber_positions",
                            "CYCLE_EXPOSURE_CHAMBER_POSITION_DIFFERS"
                        )
                    ]
                )

            equipment_allocation_relationships = (
                compare_equipment_allocation(
                    first_support = first_support,
                    second_support = second_support,
                    equipment_fields = (
                        equipment_comparison_fields
                    )
                )
            )

            equipment_disjoint_dimension_count = sum(
                relationship_record[
                    "relationship"
                ]
                == "disjoint"
                for relationship_record in (
                    equipment_allocation_relationships
                )
            )

            equipment_identity_conflicted_dimension_count = sum(
                relationship_record[
                    "relationship"
                ]
                == "identity_conflicted"
                for relationship_record in (
                    equipment_allocation_relationships
                )
            )

            recorded_equipment_differences = [
                relationship_record[
                    "difference_code"
                ]
                for relationship_record in (
                    equipment_allocation_relationships
                )
                if relationship_record[
                    "relationship"
                ]
                != "identical_or_shared"
            ]

            # Confirming that every equipment restriction is represented once
            if len(
                recorded_equipment_differences
            ) != len(
                set(
                    recorded_equipment_differences
                )
            ):
                raise RuntimeError(
                    "Condition comparison contains duplicate equipment-allocation restrictions : "
                    f"{first_condition['condition_protocol_identity']} : {second_condition['condition_protocol_identity']}"
                )

            if aging_mode == "calendar_aging":
                exposure_context_available = False

            else:
                cycle_exposure_fields = [
                    "cycle_aging_exposure_measurement_devices",
                    "cycle_aging_exposure_test_channels",
                    "cycle_aging_exposure_climate_chambers",
                    "cycle_aging_exposure_climate_chamber_positions"
                ]

                exposure_context_available = all(
                    bool(
                        first_support[
                            equipment_field
                        ]
                    )
                    and bool(
                        second_support[
                            equipment_field
                        ]
                    )
                    for equipment_field in (
                        cycle_exposure_fields
                    )
                )

            for outcome_name in OUTCOME_DEFINITIONS:
                for cohort_name in [
                    "primary",
                    "sensitivity"
                ]:
                    cohort_coverage_fraction = float(
                        coverage_rules[
                            cohort_name
                        ][
                            "minimum_assigned_cell_coverage_fraction"
                        ]
                    )

                    for horizon_axis in horizon_axes:
                        first_horizon = (
                            horizon_support_by_key[
                                (
                                    first_condition[
                                        "condition_protocol_identity"
                                    ],
                                    outcome_name,
                                    cohort_name,
                                    horizon_axis
                                )
                            ]
                        )

                        second_horizon = (
                            horizon_support_by_key[
                                (
                                    second_condition[
                                        "condition_protocol_identity"
                                    ],
                                    outcome_name,
                                    cohort_name,
                                    horizon_axis
                                )
                            ]
                        )

                        first_selected_horizons = set(
                            first_horizon[
                                "coverage_eligible_observed_horizons"
                            ]
                        )

                        second_selected_horizons = set(
                            second_horizon[
                                "coverage_eligible_observed_horizons"
                            ]
                        )

                        coverage_eligible_observed_horizons = sorted(
                            first_selected_horizons.intersection(
                                second_selected_horizons
                            )
                        )

                        permitted_bracketing_ranges = (
                            intersect_supported_intervals(
                                first_intervals = first_horizon[
                                    "supported_bracketing_ranges"
                                ],
                                second_intervals = second_horizon[
                                    "supported_bracketing_ranges"
                                ]
                            )
                        )

                        first_observed_minimum = (
                            first_horizon[
                                "observed_horizon_minimum"
                            ]
                        )

                        first_observed_maximum = (
                            first_horizon[
                                "observed_horizon_maximum"
                            ]
                        )

                        second_observed_minimum = (
                            second_horizon[
                                "observed_horizon_minimum"
                            ]
                        )

                        second_observed_maximum = (
                            second_horizon[
                                "observed_horizon_maximum"
                            ]
                        )

                        if (
                            first_observed_minimum is None
                            or first_observed_maximum is None
                            or second_observed_minimum is None
                            or second_observed_maximum is None
                        ):
                            observed_overlap_minimum = None
                            observed_overlap_maximum = None

                        else:
                            observed_overlap_minimum = max(
                                first_observed_minimum,
                                second_observed_minimum
                            )

                            observed_overlap_maximum = min(
                                first_observed_maximum,
                                second_observed_maximum
                            )

                            if (
                                observed_overlap_minimum
                                > observed_overlap_maximum
                            ):
                                observed_overlap_minimum = None
                                observed_overlap_maximum = None

                        first_complete_minimum = (
                            first_horizon[
                                "complete_eligible_cell_bracketing_minimum"
                            ]
                        )

                        first_complete_maximum = (
                            first_horizon[
                                "complete_eligible_cell_bracketing_maximum"
                            ]
                        )

                        second_complete_minimum = (
                            second_horizon[
                                "complete_eligible_cell_bracketing_minimum"
                            ]
                        )

                        second_complete_maximum = (
                            second_horizon[
                                "complete_eligible_cell_bracketing_maximum"
                            ]
                        )

                        if (
                            first_complete_minimum is None
                            or first_complete_maximum is None
                            or second_complete_minimum is None
                            or second_complete_maximum is None
                        ):
                            complete_overlap_minimum = None
                            complete_overlap_maximum = None

                        else:
                            complete_overlap_minimum = max(
                                first_complete_minimum,
                                second_complete_minimum
                            )

                            complete_overlap_maximum = min(
                                first_complete_maximum,
                                second_complete_maximum
                            )

                            if (
                                complete_overlap_minimum
                                > complete_overlap_maximum
                            ):
                                complete_overlap_minimum = None
                                complete_overlap_maximum = None

                        exact_common_horizons = sorted(
                            set(
                                first_horizon[
                                    "exact_common_horizons"
                                ]
                            ).intersection(
                                second_horizon[
                                    "exact_common_horizons"
                                ]
                            )
                        )

                        minimum_eligible_cell_count = min(
                            first_horizon[
                                "eligible_cell_count"
                            ],
                            second_horizon[
                                "eligible_cell_count"
                            ]
                        )

                        is_calendar_elapsed_time_estimand = (
                            aging_mode == "calendar_aging"
                            and horizon_axis == "elapsed_calendar_days"
                        )

                        if is_calendar_elapsed_time_estimand:
                            target_estimand_available = (
                                analysis_policy[
                                    "calendar_exposure_policy"
                                ][
                                    "primary_estimand_availability"
                                ]
                                == "available"
                            )

                            target_estimand_identity = (
                                analysis_policy[
                                    "calendar_exposure_policy"
                                ][
                                    "primary_estimand"
                                ]
                            )

                            target_estimand_status = (
                                "available_whole_assigned_protocol_"
                                "elapsed_time_estimand"
                                if target_estimand_available
                                else "unavailable"
                            )

                        else:
                            target_estimand_available = True

                            target_estimand_identity = (
                                "outcome_difference_at_common_"
                                f"{horizon_axis}"
                            )

                            target_estimand_status = "available"

                        (
                            support_status,
                            support_detail
                        ) = classify_estimand_support(
                            changed_treatment_count = len(
                                changed_treatments
                            ),
                            context_difference_count = len(
                                context_differences
                            ),
                            equipment_disjoint_dimension_count = (
                                equipment_disjoint_dimension_count
                            ),
                            equipment_identity_conflicted_dimension_count = (
                                equipment_identity_conflicted_dimension_count
                            ),
                            minimum_eligible_cell_count = (
                                minimum_eligible_cell_count
                            ),
                            observed_horizon_overlap_available = (
                                observed_overlap_minimum
                                is not None
                            ),
                            common_support_interval_available = bool(
                                permitted_bracketing_ranges
                            ),
                            target_estimand_available = (
                                target_estimand_available
                            )
                        )

                        restriction_reasons = []

                        if len(changed_treatments) != 1:
                            restriction_reasons.append(
                                "JOINT_CONDITION_CONTRAST_NOT_AN_ISOLATED_TREATMENT_EFFECT"
                            )

                        if context_differences:
                            restriction_reasons.append(
                                "DESIGN_CONTEXT_DIFFERS"
                            )

                        if design_provenance_differences:
                            restriction_reasons.append(
                                "DESIGN_PROVENANCE_DIFFERS"
                            )

                        restriction_reasons.extend(
                            recorded_equipment_differences
                        )

                        if not exposure_context_available:
                            if aging_mode == "calendar_aging":
                                restriction_reasons.append(
                                    "CALENDAR_AGING_EXPOSURE_EQUIPMENT_UNAVAILABLE"
                                )

                            else:
                                restriction_reasons.append(
                                    "CYCLE_AGING_EXPOSURE_EQUIPMENT_INCOMPLETE"
                                )

                        if (
                            minimum_eligible_cell_count
                            < MINIMUM_REPLICATED_CELL_COUNT
                        ):
                            restriction_reasons.append(
                                "INSUFFICIENT_INDEPENDENT_REPLICATION"
                            )

                        if observed_overlap_minimum is None:
                            restriction_reasons.append(
                                "NO_OBSERVED_CONDITION_LEVEL_HORIZON_OVERLAP"
                            )

                        elif not permitted_bracketing_ranges:
                            restriction_reasons.append(
                                "NO_COMMON_ASSIGNED_CELL_SUPPORT_INTERVAL"
                            )

                        if (
                            permitted_bracketing_ranges
                            and not exact_common_horizons
                        ):
                            restriction_reasons.append(
                                "CONTINUOUS_SUPPORT_REQUIRES_MODEL_BASED_"
                                "WITHIN_RANGE_ESTIMATION"
                            )

                        restriction_reasons.append(
                            "ASSIGNMENT_MECHANISM_NOT_ESTABLISHED"
                        )

                        estimand_record = {
                            "aging_mode": aging_mode,
                            "decision_scope": "condition_protocol_pair",
                            "first_condition_protocol_identity": (
                                first_condition["condition_protocol_identity"]
                            ),
                            "second_condition_protocol_identity": (
                                second_condition["condition_protocol_identity"]
                            ),
                            "outcome": outcome_name,
                            "cohort": cohort_name,
                            "horizon_axis": horizon_axis,
                            "changed_treatments": changed_treatments,
                            "context_differences": context_differences,
                            "recorded_equipment_differences": (
                                recorded_equipment_differences
                            ),
                            "equipment_allocation_relationships": (
                                equipment_allocation_relationships
                            ),
                            "equipment_disjoint_dimension_count": (
                                equipment_disjoint_dimension_count
                            ),
                            "equipment_identity_conflicted_dimension_count": (
                                equipment_identity_conflicted_dimension_count
                            ),
                            "equipment_allocation_support_role": (
                                "descriptive_restriction_not_support_veto"
                            ),
                            "equipment_adjustment_authorization": (
                                "not_authorized_from_aggregated_allocation_history"
                            ),
                            "first_eligible_cell_count": (
                                first_horizon["eligible_cell_count"]
                            ),
                            "second_eligible_cell_count": (
                                second_horizon["eligible_cell_count"]
                            ),
                            "first_required_support_cell_count": (
                                first_horizon[
                                    "required_support_cell_count"
                                ]
                            ),
                            "second_required_support_cell_count": (
                                second_horizon[
                                    "required_support_cell_count"
                                ]
                            ),
                            "first_supported_continuous_intervals": (
                                first_horizon[
                                    "supported_bracketing_ranges"
                                ]
                            ),
                            "second_supported_continuous_intervals": (
                                second_horizon[
                                    "supported_bracketing_ranges"
                                ]
                            ),
                            "permitted_bracketing_ranges": (
                                permitted_bracketing_ranges
                            ),
                            "observed_horizon_overlap_minimum": (
                                observed_overlap_minimum
                            ),
                            "observed_horizon_overlap_maximum": (
                                observed_overlap_maximum
                            ),
                            "complete_cell_bracketing_minimum": (
                                complete_overlap_minimum
                            ),
                            "complete_cell_bracketing_maximum": (
                                complete_overlap_maximum
                            ),
                            "exact_common_horizons": exact_common_horizons,
                            "model_identifiability_source": (
                                "design_identifiability.csv"
                            ),
                            "direct_contrast_support_status": support_status,
                            "direct_contrast_support_detail": support_detail,
                            "restriction_reasons": sorted(
                                set(restriction_reasons)
                            ),
                            "causal_interpretation": "not_established",
                            "design_provenance_differences": (
                                design_provenance_differences
                            ),
                            "first_assigned_cell_count": (
                                first_horizon["assigned_cell_count"]
                            ),
                            "second_assigned_cell_count": (
                                second_horizon["assigned_cell_count"]
                            ),
                            "first_eligible_assigned_cell_fraction": (
                                first_horizon[
                                    "eligible_assigned_cell_fraction"
                                ]
                            ),
                            "second_eligible_assigned_cell_fraction": (
                                second_horizon[
                                    "eligible_assigned_cell_fraction"
                                ]
                            ),
                            "minimum_assigned_cell_horizon_coverage_fraction": (
                                cohort_coverage_fraction
                            ),
                            "coverage_eligible_observed_horizons": (
                                coverage_eligible_observed_horizons
                            ),
                            "exposure_context_status": (
                                "available"
                                if exposure_context_available
                                else "unavailable"
                            ),
                            "causal_identification_status": (
                                "not_established"
                            ),
                            "target_estimand_available": (
                                target_estimand_available
                            ),
                            "target_estimand_identity": (
                                target_estimand_identity
                            ),
                            "target_estimand_status": (
                                target_estimand_status
                            ),
                            "calendar_time_measurement": (
                                "gross_elapsed_time_between_reference_test_start_dates"
                                if is_calendar_elapsed_time_estimand
                                else None
                            ),
                            "net_calendar_aging_exposure_status": (
                                analysis_policy[
                                    "calendar_exposure_policy"
                                ][
                                    "net_storage_exposure_availability"
                                ]
                                if is_calendar_elapsed_time_estimand
                                else None
                            ),
                            "net_calendar_aging_model_authorization": (
                                analysis_policy[
                                    "calendar_exposure_policy"
                                ][
                                    "net_storage_exposure_model_authorization"
                                ]
                                if is_calendar_elapsed_time_estimand
                                else None
                            ),
                            "gross_elapsed_time_analysis_status": (
                                "candidate_whole_protocol_target_with_restrictions"
                                if is_calendar_elapsed_time_estimand
                                else None
                            ),
                            "calendar_claim_boundary": (
                                analysis_policy[
                                    "calendar_exposure_policy"
                                ][
                                    "claim_boundary"
                                ]
                                if is_calendar_elapsed_time_estimand
                                else None
                            )
                        }

                        estimand_record.update(
                            build_question_disposition(
                                estimand_record
                            )
                        )

                        estimand_records.append(
                            estimand_record
                        )

    return sorted(
        estimand_records,
        key = lambda record: (
            record["aging_mode"],
            record[
                "first_condition_protocol_identity"
            ],
            record[
                "second_condition_protocol_identity"
            ],
            record["outcome"],
            record["cohort"],
            record["horizon_axis"]
        )
    )


# Reconciling primary and sensitivity decisions without elevating descriptive support
def apply_final_estimand_classifications(estimand_records: list[dict]) -> list[dict]:
    primary_status_by_question = {
        (
            record["aging_mode"],
            record[
                "first_condition_protocol_identity"
            ],
            record[
                "second_condition_protocol_identity"
            ],
            record["outcome"],
            record["horizon_axis"]
        ): record[
            "direct_contrast_support_status"
        ]
        for record in estimand_records
        if record["cohort"] == "primary"
    }

    classified_records = []

    for record in estimand_records:
        question_key = (
            record["aging_mode"],
            record[
                "first_condition_protocol_identity"
            ],
            record[
                "second_condition_protocol_identity"
            ],
            record["outcome"],
            record["horizon_axis"]
        )

        underlying_status = record[
            "direct_contrast_support_status"
        ]

        primary_status = (
            underlying_status
            if record["cohort"] == "primary"
            else primary_status_by_question.get(
                question_key
            )
        )

        final_classification = underlying_status
        classification_basis = (
            "underlying_direct_contrast_classification"
        )

        if record["cohort"] == "primary":
            cohort_comparison_status = (
                "primary_cohort_classification"
            )

        elif primary_status is None:
            cohort_comparison_status = (
                "primary_counterpart_missing"
            )

        elif (
            primary_status == "unsupported"
            and underlying_status != "unsupported"
        ):
            cohort_comparison_status = (
                "available_only_in_sensitivity_cohort"
            )

        else:
            cohort_comparison_status = (
                "primary_counterpart_available"
            )

        if (
            record["cohort"] == "sensitivity"
            and underlying_status
            == "supported_with_restrictions"
            and primary_status
            != "supported_with_restrictions"
        ):
            final_classification = (
                "sensitivity_only"
            )

            classification_basis = (
                "analytically_supported_only_in_"
                "sensitivity_cohort"
            )

        elif underlying_status == "descriptive_only":
            final_classification = (
                "descriptive_only"
            )

            classification_basis = (
                "descriptive_limit_retained_"
                "regardless_of_cohort_availability"
            )

        classified_records.append(
            {
                **record,
                "primary_direct_contrast_support_status": (
                    primary_status
                ),
                "underlying_direct_contrast_support_status": (
                    underlying_status
                ),
                "cohort_comparison_status": (
                    cohort_comparison_status
                ),
                "final_support_classification": (
                    final_classification
                ),
                "final_support_classification_basis": (
                    classification_basis
                )
            }
        )

    return classified_records


# Building one eligible horizon range for every cell and analytical population
def build_cell_horizon_ranges(cell_records: list[dict], reference_records: list[dict]) -> list[dict]:
    reference_records_by_cell = group_records_by_cell(
        reference_records
    )

    range_records = []

    for cell_record in cell_records:
        aging_mode = cell_record[
            "aging_mode"
        ]

        horizon_axes = [
            "elapsed_calendar_days"
        ]

        if aging_mode == "cycle_aging":
            horizon_axes.append(
                "cumulative_full_equivalent_cycles_observed"
            )

        cell_identity = cell_record[
            "cell_identity"
        ]

        for outcome_name, outcome_definition in (
            OUTCOME_DEFINITIONS.items()
        ):
            for cohort_name in [
                "primary",
                "sensitivity"
            ]:
                for horizon_axis in horizon_axes:
                    eligible_horizons = sorted(
                        {
                            float(
                                reference_record[
                                    horizon_axis
                                ]
                            )
                            for reference_record in (
                                reference_records_by_cell.get(
                                    cell_identity,
                                    []
                                )
                            )
                            if record_supports_horizon(
                                record = reference_record,
                                outcome_definition = (
                                    outcome_definition
                                ),
                                cohort_name = cohort_name,
                                horizon_axis = horizon_axis
                            )
                        }
                    )

                    if not eligible_horizons:
                        continue

                    range_records.append(
                        {
                            "cell_identity": cell_identity,
                            "condition_protocol_identity": (
                                cell_record[
                                    "condition_protocol_identity"
                                ]
                            ),
                            "aging_mode": aging_mode,
                            "dataset_stage": cell_record[
                                "dataset_stage"
                            ],
                            "laboratory": cell_record[
                                "laboratory"
                            ],
                            "sampling_method": cell_record[
                                "sampling_method"
                            ],
                            "outcome": outcome_name,
                            "cohort": cohort_name,
                            "horizon_axis": horizon_axis,
                            "supporting_cell_eligible_observation_count": len(
                                eligible_horizons
                            ),
                            "eligible_horizon_minimum": (
                                eligible_horizons[0]
                            ),
                            "eligible_horizon_maximum": (
                                eligible_horizons[-1]
                            ),
                            "eligible_horizons": (
                                eligible_horizons
                            )
                        }
                    )

    return sorted(
        range_records,
        key = lambda record: (
            record["aging_mode"],
            record["outcome"],
            record["cohort"],
            record["horizon_axis"],
            record["condition_protocol_identity"],
            record["cell_identity"]
        )
    )


# Building atomic regions whose supporting-cell membership cannot change internally
def build_continuous_support_regions(cell_records: list[dict], cell_horizon_range_records: list[dict],
                                     coverage_rules: dict
) -> tuple[list[dict], list[dict]]:
    assigned_cells_by_condition = defaultdict(
        set
    )

    for cell_record in cell_records:
        assigned_cells_by_condition[
            cell_record[
                "condition_protocol_identity"
            ]
        ].add(
            cell_record[
                "cell_identity"
            ]
        )

    ranges_by_population = defaultdict(
        list
    )

    boundaries_by_base_population = defaultdict(
        set
    )

    for range_record in cell_horizon_range_records:
        population_key = (
            range_record["aging_mode"],
            range_record["outcome"],
            range_record["cohort"],
            range_record["horizon_axis"]
        )

        base_population_key = (
            range_record["aging_mode"],
            range_record["outcome"],
            range_record["horizon_axis"]
        )

        ranges_by_population[
            population_key
        ].append(
            range_record
        )

        boundaries_by_base_population[
            base_population_key
        ].update(
            {
                float(
                    range_record[
                        "eligible_horizon_minimum"
                    ]
                ),
                float(
                    range_record[
                        "eligible_horizon_maximum"
                    ]
                )
            }
        )

    region_records = []
    membership_records = []

    for population_key, population_ranges in sorted(
        ranges_by_population.items()
    ):
        (
            aging_mode,
            outcome,
            cohort,
            horizon_axis
        ) = population_key

        boundaries = sorted(
            boundaries_by_base_population[
                (
                    aging_mode,
                    outcome,
                    horizon_axis
                )
            ]
        )

        atomic_regions = [
            (
                "exact_point",
                boundary,
                boundary,
                True,
                True
            )
            for boundary in boundaries
        ]

        atomic_regions.extend(
            (
                "open_interval",
                lower_boundary,
                upper_boundary,
                False,
                False
            )
            for lower_boundary, upper_boundary in zip(
                boundaries,
                boundaries[1:]
            )
            if lower_boundary < upper_boundary
        )

        for (
            region_kind,
            lower_horizon,
            upper_horizon,
            lower_inclusive,
            upper_inclusive
        ) in sorted(
            atomic_regions,
            key = lambda item: (
                item[1],
                item[2],
                item[0]
            )
        ):
            supporting_ranges = [
                range_record
                for range_record in population_ranges
                if (
                    float(
                        range_record[
                            "eligible_horizon_minimum"
                        ]
                    )
                    <= lower_horizon
                    and float(
                        range_record[
                            "eligible_horizon_maximum"
                        ]
                    )
                    >= upper_horizon
                )
            ]

            supporting_ranges_by_condition = defaultdict(
                list
            )

            for range_record in supporting_ranges:
                supporting_ranges_by_condition[
                    range_record[
                        "condition_protocol_identity"
                    ]
                ].append(
                    range_record
                )

            qualifying_conditions = set()

            for (
                condition_identity,
                condition_supporting_ranges
            ) in supporting_ranges_by_condition.items():
                assigned_cell_count = len(
                    assigned_cells_by_condition[
                        condition_identity
                    ]
                )

                required_support_cell_count = (
                    calculate_required_support_cell_count(
                        assigned_cell_count = (
                            assigned_cell_count
                        ),
                        minimum_assigned_cell_coverage_fraction = (
                            coverage_rules[
                                cohort
                            ][
                                "minimum_assigned_cell_coverage_fraction"
                            ]
                        )
                    )
                )

                if (
                    len(
                        condition_supporting_ranges
                    )
                    >= required_support_cell_count
                ):
                    qualifying_conditions.add(
                        condition_identity
                    )

            qualifying_ranges = [
                range_record
                for range_record in supporting_ranges
                if (
                    range_record[
                        "condition_protocol_identity"
                    ]
                    in qualifying_conditions
                )
            ]

            if not qualifying_ranges:
                continue

            identity_payload = {
                "aging_mode": aging_mode,
                "outcome": outcome,
                "cohort": cohort,
                "horizon_axis": horizon_axis,
                "region_kind": region_kind,
                "lower_horizon": lower_horizon,
                "upper_horizon": upper_horizon
            }

            support_region_identity = hashlib.sha256(
                json.dumps(
                    identity_payload,
                    sort_keys = True,
                    separators = (
                        ",",
                        ":"
                    )
                ).encode(
                    "utf-8"
                )
            ).hexdigest()

            region_records.append(
                {
                    "support_region_identity": (
                        support_region_identity
                    ),
                    "aging_mode": aging_mode,
                    "outcome": outcome,
                    "cohort": cohort,
                    "horizon_axis": horizon_axis,
                    "region_kind": region_kind,
                    "lower_horizon": lower_horizon,
                    "upper_horizon": upper_horizon,
                    "lower_inclusive": lower_inclusive,
                    "upper_inclusive": upper_inclusive,
                    "qualifying_condition_count": len(
                        qualifying_conditions
                    ),
                    "supporting_cell_count": len(
                        qualifying_ranges
                    ),
                    "minimum_assigned_cell_coverage_fraction": (
                        coverage_rules[
                            cohort
                        ][
                            "minimum_assigned_cell_coverage_fraction"
                        ]
                    ),
                    "extrapolation_permitted": False,
                    "support_geometry": (
                        "within_cell_minimum_maximum_bracketing_range"
                    ),
                    "continuous_observation_status": (
                        "not_established_between_observed_horizons"
                    ),
                    "interpolation_authorization_status": (
                        "requires_model_specific_validation"
                    )
                }
            )

            for range_record in qualifying_ranges:
                membership_records.append(
                    {
                        "support_region_identity": (
                            support_region_identity
                        ),
                        "aging_mode": aging_mode,
                        "outcome": outcome,
                        "cohort": cohort,
                        "horizon_axis": horizon_axis,
                        "region_kind": region_kind,
                        "lower_horizon": lower_horizon,
                        "upper_horizon": upper_horizon,
                        "lower_inclusive": lower_inclusive,
                        "upper_inclusive": upper_inclusive,
                        "cell_identity": range_record[
                            "cell_identity"
                        ],
                        "condition_protocol_identity": (
                            range_record[
                                "condition_protocol_identity"
                            ]
                        ),
                        "dataset_stage": range_record[
                            "dataset_stage"
                        ],
                        "laboratory": range_record[
                            "laboratory"
                        ],
                        "sampling_method": range_record[
                            "sampling_method"
                        ],
                        "supporting_cell_eligible_observation_count": (
                            range_record[
                                "supporting_cell_eligible_observation_count"
                            ]
                        ),
                        "cell_horizon_minimum": (
                            range_record[
                                "eligible_horizon_minimum"
                            ]
                        ),
                        "cell_horizon_maximum": (
                            range_record[
                                "eligible_horizon_maximum"
                            ]
                        ),
                        "support_geometry": (
                            "within_cell_minimum_maximum_bracketing_range"
                        ),
                        "continuous_observation_status": (
                            "not_established_between_observed_horizons"
                        ),
                        "interpolation_authorization_status": (
                            "requires_model_specific_validation"
                        )
                    }
                )

    return (
        sorted(
            region_records,
            key = lambda record: (
                record["aging_mode"],
                record["outcome"],
                record["cohort"],
                record["horizon_axis"],
                record["lower_horizon"],
                record["upper_horizon"],
                record["region_kind"]
            )
        ),
        sorted(
            membership_records,
            key = lambda record: (
                record["support_region_identity"],
                record["condition_protocol_identity"],
                record["cell_identity"]
            )
        )
    )


# Confirming that one context family varies within a region's supporting cells
def context_family_varies(context_family: str, region_cell_records: list[dict],
                          equipment_by_cell: dict
) -> bool:
    if context_family == CONTEXT_FAMILY_NONE:
        return True

    if context_family == CONTEXT_FAMILY_LABORATORY:
        return len(
            {
                record["laboratory"]
                for record in region_cell_records
            }
        ) > 1

    for equipment_role in [
        "reference_measurement",
        "cycle_aging_exposure"
    ]:
        for output_field in EQUIPMENT_FIELDS:
            if context_family != f"{equipment_role}_{output_field}":
                continue

            signatures = {
                build_equipment_signature(
                    cell_identities=[
                        record["cell_identity"]
                    ],
                    equipment_by_cell=equipment_by_cell,
                    equipment_role=equipment_role,
                    output_field=output_field
                )
                for record in region_cell_records
            }

            return len(signatures) > 1

    raise RuntimeError(f"Unknown context family : {context_family}")


# Jointly compacting atomic support regions across cohort signatures
def jointly_compact_support_regions(cell_horizon_range_records: list[dict], atomic_region_records: list[dict],
                                    atomic_membership_records: list[dict]
) -> tuple[list[dict], list[dict]]:
    memberships_by_region = defaultdict(
        list
    )

    for membership_record in atomic_membership_records:
        memberships_by_region[
            membership_record[
                "support_region_identity"
            ]
        ].append(
            membership_record
        )

    boundaries_by_base_population = defaultdict(
        set
    )

    for range_record in cell_horizon_range_records:
        base_population_key = (
            range_record["aging_mode"],
            range_record["outcome"],
            range_record["horizon_axis"]
        )

        boundaries_by_base_population[
            base_population_key
        ].update(
            {
                float(
                    range_record[
                        "eligible_horizon_minimum"
                    ]
                ),
                float(
                    range_record[
                        "eligible_horizon_maximum"
                    ]
                )
            }
        )

    atomic_by_base_and_cohort = defaultdict(
        lambda: defaultdict(
            dict
        )
    )

    for region_record in atomic_region_records:
        base_population_key = (
            region_record["aging_mode"],
            region_record["outcome"],
            region_record["horizon_axis"]
        )

        geometry = (
            region_record["region_kind"],
            region_record["lower_horizon"],
            region_record["upper_horizon"],
            region_record["lower_inclusive"],
            region_record["upper_inclusive"]
        )

        atomic_by_base_and_cohort[
            base_population_key
        ][
            region_record["cohort"]
        ][
            geometry
        ] = region_record

    compact_region_records = []
    compact_membership_records = []

    for (
        base_population_key,
        boundaries
    ) in sorted(
        boundaries_by_base_population.items()
    ):
        (
            aging_mode,
            outcome,
            horizon_axis
        ) = base_population_key

        ordered_boundaries = sorted(
            boundaries
        )

        atomic_geometries = [
            (
                "exact_point",
                boundary,
                boundary,
                True,
                True
            )
            for boundary in ordered_boundaries
        ]

        atomic_geometries.extend(
            (
                "open_interval",
                lower_boundary,
                upper_boundary,
                False,
                False
            )
            for lower_boundary, upper_boundary in zip(
                ordered_boundaries,
                ordered_boundaries[1:]
            )
            if lower_boundary < upper_boundary
        )

        atomic_geometries.sort(
            key = lambda geometry: (
                geometry[1],
                geometry[2],
                geometry[0]
            )
        )

        def geometry_signature(cohort_name: str, geometry: tuple) -> frozenset | None:
            region_record = atomic_by_base_and_cohort[
                base_population_key
            ][
                cohort_name
            ].get(
                geometry
            )

            if region_record is None:
                return None

            return frozenset(
                (
                    membership_record["cell_identity"],
                    membership_record[
                        "condition_protocol_identity"
                    ]
                )
                for membership_record in memberships_by_region[
                    region_record[
                        "support_region_identity"
                    ]
                ]
            )

        combined_signatures = {
            geometry: (
                geometry_signature(
                    "primary",
                    geometry
                ),
                geometry_signature(
                    "sensitivity",
                    geometry
                )
            )
            for geometry in atomic_geometries
        }

        merged_groups = []

        for geometry in atomic_geometries:
            combined_signature = combined_signatures[
                geometry
            ]

            if (
                merged_groups
                and merged_groups[-1][
                    "upper_horizon"
                ]
                == geometry[1]
                and merged_groups[-1][
                    "combined_signature"
                ]
                == combined_signature
            ):
                merged_groups[-1][
                    "upper_horizon"
                ] = geometry[2]

                merged_groups[-1][
                    "upper_inclusive"
                ] = geometry[4]

                merged_groups[-1][
                    "atomic_geometries"
                ].append(
                    geometry
                )

            else:
                merged_groups.append(
                    {
                        "lower_horizon": geometry[1],
                        "upper_horizon": geometry[2],
                        "lower_inclusive": geometry[3],
                        "upper_inclusive": geometry[4],
                        "combined_signature": (
                            combined_signature
                        ),
                        "atomic_geometries": [
                            geometry
                        ]
                    }
                )

        for merged_group in merged_groups:
            (
                primary_signature,
                sensitivity_signature
            ) = merged_group[
                "combined_signature"
            ]

            if (
                primary_signature is None
                and sensitivity_signature is None
            ):
                continue

            region_kind = (
                "exact_point"
                if (
                    merged_group["lower_horizon"]
                    == merged_group["upper_horizon"]
                )
                else "interval"
            )

            combined_identity_payload = {
                "aging_mode": aging_mode,
                "outcome": outcome,
                "horizon_axis": horizon_axis,
                "region_kind": region_kind,
                "lower_horizon": (
                    merged_group["lower_horizon"]
                ),
                "upper_horizon": (
                    merged_group["upper_horizon"]
                ),
                "lower_inclusive": (
                    merged_group["lower_inclusive"]
                ),
                "upper_inclusive": (
                    merged_group["upper_inclusive"]
                ),
                "primary_signature": sorted(
                    primary_signature
                    if primary_signature is not None
                    else []
                ),
                "sensitivity_signature": sorted(
                    sensitivity_signature
                    if sensitivity_signature is not None
                    else []
                )
            }

            combined_region_identity = hashlib.sha256(
                json.dumps(
                    combined_identity_payload,
                    sort_keys = True,
                    separators = (",", ":")
                ).encode(
                    "utf-8"
                )
            ).hexdigest()

            for (
                cohort_name,
                cohort_signature
            ) in (
                (
                    "primary",
                    primary_signature
                ),
                (
                    "sensitivity",
                    sensitivity_signature
                )
            ):
                if cohort_signature is None:
                    continue

                sample_region = None

                for geometry in merged_group[
                    "atomic_geometries"
                ]:
                    candidate_region = (
                        atomic_by_base_and_cohort[
                            base_population_key
                        ][
                            cohort_name
                        ].get(
                            geometry
                        )
                    )

                    if candidate_region is not None:
                        sample_region = candidate_region
                        break

                if sample_region is None:
                    raise RuntimeError(
                        "Merged cohort region does not contain any atomic source region : "
                        f"{combined_region_identity} : {cohort_name}"
                    )

                sample_memberships = (
                    memberships_by_region[
                        sample_region[
                            "support_region_identity"
                        ]
                    ]
                )

                sample_signature = frozenset(
                    (
                        membership_record[
                            "cell_identity"
                        ],
                        membership_record[
                            "condition_protocol_identity"
                        ]
                    )
                    for membership_record in (
                        sample_memberships
                    )
                )

                if sample_signature != cohort_signature:
                    raise RuntimeError(
                        "Representative atomic membership does not match the compact cohort signature : "
                        f"{combined_region_identity} : {cohort_name}"
                    )

                if (
                    len(sample_memberships)
                    != len(sample_signature)
                ):
                    raise RuntimeError(
                        f"Atomic support region contains duplicate cell memberships : {combined_region_identity}"
                    )

                cohort_region_identity = hashlib.sha256(
                    json.dumps(
                        {
                            "combined_region_identity": (
                                combined_region_identity
                            ),
                            "cohort": cohort_name
                        },
                        sort_keys = True,
                        separators = (",", ":")
                    ).encode(
                        "utf-8"
                    )
                ).hexdigest()

                compact_region_records.append(
                    {
                        **sample_region,
                        "support_region_identity": (
                            cohort_region_identity
                        ),
                        "combined_region_identity": (
                            combined_region_identity
                        ),
                        "region_kind": region_kind,
                        "lower_horizon": (
                            merged_group[
                                "lower_horizon"
                            ]
                        ),
                        "upper_horizon": (
                            merged_group[
                                "upper_horizon"
                            ]
                        ),
                        "lower_inclusive": (
                            merged_group[
                                "lower_inclusive"
                            ]
                        ),
                        "upper_inclusive": (
                            merged_group[
                                "upper_inclusive"
                            ]
                        ),
                        "compacted_atomic_region_count": len(
                            merged_group[
                                "atomic_geometries"
                            ]
                        )
                    }
                )

                for membership_record in sample_memberships:
                    compact_membership_records.append(
                        {
                            **membership_record,
                            "support_region_identity": (
                                cohort_region_identity
                            ),
                            "combined_region_identity": (
                                combined_region_identity
                            ),
                            "region_kind": region_kind,
                            "lower_horizon": (
                                merged_group[
                                    "lower_horizon"
                                ]
                            ),
                            "upper_horizon": (
                                merged_group[
                                    "upper_horizon"
                                ]
                            ),
                            "lower_inclusive": (
                                merged_group[
                                    "lower_inclusive"
                                ]
                            ),
                            "upper_inclusive": (
                                merged_group[
                                    "upper_inclusive"
                                ]
                            )
                        }
                    )

    return (
        sorted(
            compact_region_records,
            key = lambda record: (
                record["aging_mode"],
                record["outcome"],
                record["cohort"],
                record["horizon_axis"],
                record["lower_horizon"],
                record["upper_horizon"],
                record["region_kind"]
            )
        ),
        sorted(
            compact_membership_records,
            key = lambda record: (
                record["support_region_identity"],
                record["condition_protocol_identity"],
                record["cell_identity"]
            )
        )
    )


# Classifying one interval term as excluded or candidate
def interval_term_exclusion_reason(record: dict) -> str | None:
    if record["structurally_estimable"] is not True:
        return "structurally_nonestimable"

    if record["design_scope"] == "pooled_stages":
        return (
            "cross_stage_pooling_not_authorized_by_design_audit"
        )

    if context_family_is_equipment(
        record["context_family"]
    ):
        return (
            "equipment_adjustment_not_outcome_horizon_specific"
        )

    return None


# Recomputing identifiability using only cells supporting each region
def build_continuous_support_identifiability(support_region_records: list[dict], support_membership_records: list[dict],
                                             cell_records: list[dict], equipment_by_cell: dict
) -> tuple[list[dict], list[dict]]:
    cells_by_identity = {
        record["cell_identity"]: record
        for record in cell_records
    }

    memberships_by_region = defaultdict(
        list
    )

    for membership_record in support_membership_records:
        memberships_by_region[
            membership_record[
                "support_region_identity"
            ]
        ].append(
            membership_record
        )

    identifiability_records = []
    identifiability_cache = {}
    exclusion_counts = Counter()

    for region_record in support_region_records:
        region_identity = region_record[
            "support_region_identity"
        ]

        region_cell_records = []

        for membership_record in memberships_by_region[
            region_identity
        ]:
            cell_record = cells_by_identity[
                membership_record[
                    "cell_identity"
                ]
            ]

            region_cell_records.append(
                {
                    **cell_record,
                    "independent_cell_count": 1,
                    "analytical_independent_cell_count": 1,
                    "analytical_observation_count": (
                        membership_record[
                            "supporting_cell_eligible_observation_count"
                        ]
                    )
                }
            )

        cell_signature = tuple(
            sorted(
                (
                    record["cell_identity"],
                    record["condition_protocol_identity"],
                    record["dataset_stage"],
                    record["sampling_method"],
                    record["laboratory"],
                    int(
                        record[
                            "analytical_observation_count"
                        ]
                    )
                )
                for record in region_cell_records
            )
        )

        cache_key = (
            region_record["aging_mode"],
            cell_signature
        )

        if cache_key not in identifiability_cache:
            cached_records = []

            for context_family in build_context_families(
                region_record["aging_mode"]
            ):
                if not context_family_varies(
                    context_family = context_family,
                    region_cell_records = region_cell_records,
                    equipment_by_cell = equipment_by_cell
                ):
                    continue

                context_records = (
                    build_population_identifiability(
                        records = region_cell_records,
                        aging_mode = region_record[
                            "aging_mode"
                        ],
                        analysis_population = (
                            "bracketing_range_cell_design"
                        ),
                        outcome = None,
                        cohort = None,
                        horizon_axis = None,
                        equipment_by_cell = equipment_by_cell,
                        context_adjusted = True,
                        context_family = context_family,
                        analysis_role = (
                            "within_cell_bracketing_support"
                        )
                    )
                )

                for base_record in context_records:
                    if (
                        base_record["record_type"] == "term"
                        and base_record["term_type"]
                        in TARGET_DESIGN_TERM_TYPES
                    ):
                        exclusion_reason = (
                            interval_term_exclusion_reason(
                                base_record
                            )
                        )

                        cached_records.append(
                            (
                                base_record,
                                exclusion_reason
                            )
                        )

            identifiability_cache[
                cache_key
            ] = cached_records

        for (
            base_record,
            exclusion_reason
        ) in identifiability_cache[
            cache_key
        ]:
            if exclusion_reason is not None:
                exclusion_key = (
                    exclusion_reason,
                    region_record["aging_mode"],
                    region_record["outcome"],
                    region_record["cohort"],
                    region_record["horizon_axis"],
                    base_record["design_scope"],
                    base_record["sampling_design_scope"],
                    base_record["specification"],
                    base_record["context_family"],
                    base_record["term_type"]
                )

                exclusion_counts[
                    exclusion_key
                ] += 1

                continue

            identifiability_records.append(
                {
                    **base_record,
                    "outcome": region_record["outcome"],
                    "cohort": region_record["cohort"],
                    "horizon_axis": region_record[
                        "horizon_axis"
                    ],
                    "support_region_identity": (
                        region_identity
                    ),
                    "combined_region_identity": (
                        region_record.get(
                            "combined_region_identity"
                        )
                    ),
                    "support_region_kind": (
                        region_record["region_kind"]
                    ),
                    "support_interval_minimum": (
                        region_record["lower_horizon"]
                    ),
                    "support_interval_maximum": (
                        region_record["upper_horizon"]
                    ),
                    "support_interval_lower_inclusive": (
                        region_record["lower_inclusive"]
                    ),
                    "support_interval_upper_inclusive": (
                        region_record["upper_inclusive"]
                    )
                }
            )

    exclusion_records = [
        {
            "exclusion_reason": key[0],
            "aging_mode": key[1],
            "outcome": key[2],
            "cohort": key[3],
            "horizon_axis": key[4],
            "design_scope": key[5],
            "sampling_design_scope": key[6],
            "specification": key[7],
            "context_family": key[8],
            "term_type": key[9],
            "excluded_term_region_count": count
        }
        for key, count in sorted(
            exclusion_counts.items()
        )
    ]

    return (
        identifiability_records,
        exclusion_records
    )


# Confirming that one cohort support region includes one horizon value
def support_interval_contains_value(region_record: dict, value: float) -> bool:
    lower_value = float(
        region_record["lower_horizon"]
    )

    upper_value = float(
        region_record["upper_horizon"]
    )

    lower_supported = (
        value > lower_value
        or (
            value == lower_value
            and region_record["lower_inclusive"] is True
        )
    )

    upper_supported = (
        value < upper_value
        or (
            value == upper_value
            and region_record["upper_inclusive"] is True
        )
    )

    return lower_supported and upper_supported


# Confirming that two cohort support regions share at least one horizon
def support_intervals_overlap(first_record: dict, second_record: dict) -> bool:
    intersection_minimum = max(
        float(first_record["lower_horizon"]),
        float(second_record["lower_horizon"])
    )

    intersection_maximum = min(
        float(first_record["upper_horizon"]),
        float(second_record["upper_horizon"])
    )

    if intersection_minimum < intersection_maximum:
        return True

    if intersection_minimum > intersection_maximum:
        return False

    return (
        support_interval_contains_value(
            first_record,
            intersection_minimum
        )
        and support_interval_contains_value(
            second_record,
            intersection_minimum
        )
    )


# Confirming that cohort support regions are jointly aligned
def validate_cohort_aligned_support_regions(support_region_records: list[dict]) -> None:
    regions_by_base_and_cohort = defaultdict(
        lambda: {
            "primary": [],
            "sensitivity": []
        }
    )

    for region_record in support_region_records:
        base_population_key = (
            region_record["aging_mode"],
            region_record["outcome"],
            region_record["horizon_axis"]
        )

        regions_by_base_and_cohort[
            base_population_key
        ][
            region_record["cohort"]
        ].append(
            region_record
        )

    for (
        base_population_key,
        cohort_regions
    ) in regions_by_base_and_cohort.items():
        for primary_record in cohort_regions["primary"]:
            for sensitivity_record in cohort_regions["sensitivity"]:
                if not support_intervals_overlap(
                    primary_record,
                    sensitivity_record
                ):
                    continue

                primary_geometry = (
                    primary_record["region_kind"],
                    float(primary_record["lower_horizon"]),
                    float(primary_record["upper_horizon"]),
                    primary_record["lower_inclusive"],
                    primary_record["upper_inclusive"]
                )

                sensitivity_geometry = (
                    sensitivity_record["region_kind"],
                    float(sensitivity_record["lower_horizon"]),
                    float(sensitivity_record["upper_horizon"]),
                    sensitivity_record["lower_inclusive"],
                    sensitivity_record["upper_inclusive"]
                )

                if primary_geometry != sensitivity_geometry:
                    raise RuntimeError(
                        "Cohort support regions overlap without joint geometry alignment : "
                        f"{base_population_key} : {primary_geometry} : {sensitivity_geometry}"
                    )


# Building the interval-specific analytical-scope handoff table
def build_analytical_scope_table(identifiability_records: list[dict]) -> list[dict]:
    analytical_scope_records = []

    for record in identifiability_records:
        if (
            record["analysis_population"]
            != "bracketing_range_cell_design"
            or record["record_type"] != "term"
            or record["term_type"]
            not in TARGET_DESIGN_TERM_TYPES
        ):
            continue

        if (
            record["context_family"]
            != CONTEXT_FAMILY_NONE
            and record["context_term_count"] == 0
        ):
            continue

        context_family = record[
            "context_family"
        ]

        if record["structurally_estimable"] is not True:
            authorization_status = (
                "unsupported_structurally_nonestimable"
            )
            question_classification = "unsupported"
            inferential_adequacy_status = (
                "not_applicable_structurally_nonestimable"
            )
            model_evaluation_role = "not_applicable"

        elif record["design_scope"] == "pooled_stages":
            authorization_status = (
                "unsupported_cross_stage_pooling_not_established"
            )
            question_classification = "unsupported"
            inferential_adequacy_status = (
                "not_established"
            )
            model_evaluation_role = "not_applicable"

        elif context_family_is_equipment(
            context_family
        ):
            authorization_status = (
                "unsupported_equipment_adjustment_not_outcome_horizon_specific"
            )
            question_classification = "unsupported"
            inferential_adequacy_status = (
                "not_established"
            )
            model_evaluation_role = "not_applicable"

        else:
            authorization_status = (
                "permitted_for_model_evaluation_with_restrictions"
            )
            question_classification = (
                "supported_with_restrictions"
            )
            inferential_adequacy_status = (
                "requires_model_specific_validation"
            )
            model_evaluation_role = (
                "candidate_for_model_evaluation"
            )

        if context_family == CONTEXT_FAMILY_NONE:
            permitted_adjustment_variables = []
            context_evidence_role = (
                "no_additional_context"
            )

        elif context_family == CONTEXT_FAMILY_LABORATORY:
            permitted_adjustment_variables = [
                "laboratory"
            ]
            context_evidence_role = (
                "permitted_stage_specific_adjustment"
            )

        else:
            permitted_adjustment_variables = []
            context_evidence_role = (
                "descriptive_allocation_diagnostic_only"
            )

        is_calendar_elapsed_time_estimand = (
            record["aging_mode"] == "calendar_aging"
            and record["horizon_axis"] == "elapsed_calendar_days"
        )

        is_cycle_elapsed_time_estimand = (
            record["aging_mode"] == "cycle_aging"
            and record["horizon_axis"] == "elapsed_calendar_days"
        )

        is_cycle_fec_estimand = (
            record["aging_mode"] == "cycle_aging"
            and record["horizon_axis"]
            == "cumulative_full_equivalent_cycles_observed"
        )

        if is_calendar_elapsed_time_estimand:
            exposure_time_interpretation = (
                "whole_assigned_protocol_gross_elapsed_calendar_time_not_net_storage_exposure"
            )

        elif is_cycle_elapsed_time_estimand:
            exposure_time_interpretation = (
                "cycle_aging_elapsed_calendar_days_not_cumulative_fec"
            )

        elif is_cycle_fec_estimand:
            exposure_time_interpretation = (
                "complete_cumulative_fec_only"
            )

        else:
            raise RuntimeError(
                "Analytical scope record has an unsupported aging-mode and horizon-axis combination : "
                f"{record['aging_mode']} : {record['horizon_axis']}"
            )

        target_estimand_available = True

        target_estimand_status = (
            "available_whole_assigned_protocol_elapsed_time_estimand"
            if is_calendar_elapsed_time_estimand
            else "available"
        )

        analytical_scope_records.append(
            {
                "support_region_identity": record[
                    "support_region_identity"
                ],
                "support_region_kind": record[
                    "support_region_kind"
                ],
                "support_interval_minimum": record[
                    "support_interval_minimum"
                ],
                "support_interval_maximum": record[
                    "support_interval_maximum"
                ],
                "support_interval_lower_inclusive": (
                    record[
                        "support_interval_lower_inclusive"
                    ]
                ),
                "support_interval_upper_inclusive": (
                    record[
                        "support_interval_upper_inclusive"
                    ]
                ),
                "aging_mode": record["aging_mode"],
                "design_scope": record["design_scope"],
                "sampling_design_scope": record[
                    "sampling_design_scope"
                ],
                "outcome": record["outcome"],
                "cohort": record["cohort"],
                "horizon_axis": record["horizon_axis"],
                "analysis_role": record[
                    "analysis_role"
                ],
                "analysis_population": record[
                    "analysis_population"
                ],
                "condition_protocol_count": record[
                    "condition_protocol_count"
                ],
                "independent_cell_count": record[
                    "independent_cell_count"
                ],
                "analytical_observation_count": record[
                    "analytical_observation_count"
                ],
                "specification": record["specification"],
                "context_family": context_family,
                "context_term_count": record[
                    "context_term_count"
                ],
                "permitted_adjustment_variables": (
                    permitted_adjustment_variables
                ),
                "context_evidence_role": (
                    context_evidence_role
                ),
                "target_estimand_available": (
                    target_estimand_available
                ),
                "target_estimand_status": (
                    target_estimand_status
                ),
                "model_evaluation_role": (
                    model_evaluation_role
                ),
                "inferential_adequacy_status": (
                    inferential_adequacy_status
                ),
                "inferential_claim_authorization": (
                    "not_authorized_by_design_audit"
                ),
                "term_name": record["term_name"],
                "term_type": record["term_type"],
                "structurally_estimable": record[
                    "structurally_estimable"
                ],
                "bracketing_support_source": (
                    "bracketing_support_regions.csv"
                ),
                "supporting_cell_membership_source": (
                    "bracketing_support_membership.csv"
                ),
                "exposure_time_interpretation": (
                    exposure_time_interpretation
                ),
                "required_validation_grouping": [
                    "cell_identity"
                ],
                "stage_pooling_status": (
                    "not_authorized_by_design_audit"
                    if record["design_scope"]
                    == "pooled_stages"
                    else "stage_specific"
                ),
                "stage_batch_relationship_status": (
                    "stage_2_used_a_new_battery_cell_"
                    "batch_relative_to_stage_1"
                ),
                "stage_batch_interpretation": (
                    "cross_stage_effects_cannot_separate_"
                    "stage_from_batch"
                    if record["design_scope"]
                    == "pooled_stages"
                    else (
                        "stage_specific_results_are_"
                        "conditional_on_that_stage_batch"
                    )
                ),
                "extrapolation_status": "prohibited",
                "causal_identification_status": (
                    "not_established"
                ),
                "question_classification": (
                    question_classification
                ),
                "authorization_status": (
                    authorization_status
                ),
                "residual_independent_cell_degrees_of_freedom": (
                    record[
                        "residual_independent_cell_degrees_of_freedom"
                    ]
                ),
                "smallest_rank_singular_value": (
                    record[
                        "smallest_rank_singular_value"
                    ]
                ),
                "effective_condition_number": (
                    record[
                        "effective_condition_number"
                    ]
                ),
                "analytical_observation_count_scope": (
                    "all_eligible_positive_horizon_observations_"
                    "from_support_region_member_cells"
                )
            }
        )

    primary_authorization = {
        (
            record["support_region_kind"],
            record["support_interval_minimum"],
            record["support_interval_maximum"],
            record[
                "support_interval_lower_inclusive"
            ],
            record[
                "support_interval_upper_inclusive"
            ],
            record["aging_mode"],
            record["design_scope"],
            record["sampling_design_scope"],
            record["outcome"],
            record["horizon_axis"],
            record["specification"],
            record["context_family"],
            record["term_name"]
        ): record["authorization_status"]
        for record in analytical_scope_records
        if record["cohort"] == "primary"
    }

    for record in analytical_scope_records:
        if (
            record["cohort"] == "sensitivity"
            and record["authorization_status"]
            == "permitted_for_model_evaluation_with_restrictions"
        ):
            primary_status = primary_authorization.get(
                (
                    record["support_region_kind"],
                    record["support_interval_minimum"],
                    record["support_interval_maximum"],
                    record[
                        "support_interval_lower_inclusive"
                    ],
                    record[
                        "support_interval_upper_inclusive"
                    ],
                    record["aging_mode"],
                    record["design_scope"],
                    record["sampling_design_scope"],
                    record["outcome"],
                    record["horizon_axis"],
                    record["specification"],
                    record["context_family"],
                    record["term_name"]
                )
            )

            if primary_status != (
                "permitted_for_model_evaluation_with_restrictions"
            ):
                record[
                    "authorization_status"
                ] = (
                    "permitted_for_model_evaluation_"
                    "sensitivity_only"
                )

                record[
                    "question_classification"
                ] = "sensitivity_only"

    return sorted(
        analytical_scope_records,
        key = lambda record: (
            record["aging_mode"],
            record["design_scope"],
            record["sampling_design_scope"],
            record["outcome"],
            record["horizon_axis"],
            record["support_interval_minimum"],
            record["support_interval_maximum"],
            record["support_region_kind"],
            record["cohort"],
            record["specification"],
            record["context_family"],
            record["term_name"]
        )
    )


# Serializing one deterministic CSV value
def serialize_value(value: object) -> object:
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
        (list, dict)
    ):
        return json.dumps(
            value,
            sort_keys = True,
            separators = (
                ",",
                ":"
            )
        )

    return value


# Classifying one experimental-design table value for its schema
def classify_report_value(value: object) -> str:
    if isinstance(
        value,
        bool
    ):
        return "boolean"

    if isinstance(
        value,
        int
    ):
        return "integer"

    if isinstance(
        value,
        float
    ):
        if not math.isfinite(
            value
        ):
            raise RuntimeError("Experimental-design tables cannot contain non-finite values.")

        return "number"

    if isinstance(
        value,
        str
    ):
        return "string"

    if isinstance(
        value,
        (list, dict)
    ):
        return "json"

    raise RuntimeError(f"Unsupported experimental-design value type : {type(value).__name__}")


# Building one machine-readable schema for an experimental-design table
def build_report_table_schema(records: list[dict]) -> list[dict]:
    if not records:
        return []

    field_names = list(
        records[0]
    )

    table_schema = []

    for field_name in field_names:
        observed_types = {
            classify_report_value(
                record[field_name]
            )
            for record in records
            if record[field_name] is not None
        }

        if observed_types.issubset(
            {
                "integer",
                "number"
            }
        ) and observed_types:
            data_type = (
                "number"
                if "number" in observed_types
                else "integer"
            )

        elif len(observed_types) == 1:
            data_type = next(
                iter(observed_types)
            )

        elif not observed_types:
            data_type = "null"

        else:
            raise RuntimeError(
                f"Experimental-design column contains incompatible value types : {field_name} : {sorted(observed_types)}"
            )

        table_schema.append(
            {
                "name": field_name,
                "data_type": data_type,
                "nullable": any(
                    record[field_name] is None
                    for record in records
                )
            }
        )

    return table_schema


# Writing one deterministic report table
def write_report_table(output_path: Path, records: list[dict]) -> None:
    if not records:
        raise RuntimeError(f"Experimental-design table is empty : {output_path.name}")

    field_names = list(
        records[0]
    )

    if any(
        list(
            record
        )
        != field_names
        for record in records
    ):
        raise RuntimeError(f"Experimental-design table contains inconsistent columns : {output_path.name}")

    with output_path.open(
        mode = "w",
        encoding = "utf-8",
        newline = ""
    ) as output_stream:
        table_writer = csv.DictWriter(
            output_stream,
            fieldnames = field_names,
            extrasaction = "raise"
        )

        table_writer.writeheader()

        for record in records:
            table_writer.writerow(
                {
                    field_name: serialize_value(
                        record[field_name]
                    )
                    for field_name in field_names
                }
            )


# Building one test-point identity
def build_test_point_identity(cell_record: dict) -> str:
    return (
        f"{cell_record['dataset_stage']}/"
        f"TP_{cell_record['aging_type']}"
        f"{int(cell_record['test_point_number']):02d}"
    )


# Building the unique experimental test-point design
def build_test_point_design(cell_records: list[dict]) -> list[dict]:
    cells_by_test_point = defaultdict(
        list
    )

    for cell_record in cell_records:
        test_point_identity = (
            build_test_point_identity(
                cell_record
            )
        )

        cells_by_test_point[
            test_point_identity
        ].append(
            cell_record
        )

    test_point_records = []

    invariant_fields = [
        "dataset_stage",
        "aging_type",
        "aging_mode",
        "test_point_number",
        "laboratory",
        "sampling_method",
        "ambient_temperature_celsius",
        "maximum_state_of_charge_fraction",
        "depth_of_discharge_fraction",
        "charge_rate_c",
        "discharge_rate_c"
    ]

    for (
        test_point_identity,
        test_point_cells
    ) in cells_by_test_point.items():
        first_cell = test_point_cells[0]

        for field_name in invariant_fields:
            observed_values = {
                cell_record[field_name]
                for cell_record in test_point_cells
            }

            if len(observed_values) != 1:
                raise RuntimeError(
                    "Test point contains conflicting design values : "
                    f"{test_point_identity} : {field_name} : {sorted(observed_values)}"
                )

        test_point_records.append(
            {
                "test_point_identity": (
                    test_point_identity
                ),
                **{
                    field_name: first_cell[
                        field_name
                    ]
                    for field_name in invariant_fields
                },
                "independent_cell_count": len(
                    test_point_cells
                ),
                "reference_test_schedules": sorted(
                    {
                        cell_record[
                            "reference_test_schedule"
                        ]
                        for cell_record in test_point_cells
                    }
                ),
                "condition_protocol_identities": sorted(
                    {
                        cell_record[
                            "condition_protocol_identity"
                        ]
                        for cell_record in test_point_cells
                    }
                )
            }
        )

    test_point_records.sort(
        key = lambda record: (
            record["dataset_stage"],
            record["aging_type"],
            int(
                record["test_point_number"]
            )
        )
    )

    if sum(
        int(
            record["independent_cell_count"]
        )
        for record in test_point_records
    ) != len(
        cell_records
    ):
        raise RuntimeError("Test-point design does not reconcile to the cell inventory.")

    return test_point_records


# Building numerical treatment correlations on unique test points
def build_treatment_correlations(test_point_records: list[dict]) -> list[dict]:
    correlation_records = []

    for aging_mode in [
        "calendar_aging",
        "cycle_aging"
    ]:
        mode_records = [
            record
            for record in test_point_records
            if record[
                "aging_mode"
            ]
            == aging_mode
        ]

        scopes = {
            "pooled_stages": mode_records,
            "Stage_1": [
                record
                for record in mode_records
                if record["dataset_stage"] == "Stage_1"
            ],
            "Stage_2": [
                record
                for record in mode_records
                if record["dataset_stage"] == "Stage_2"
            ]
        }

        treatment_fields = (
            NUMERICAL_TREATMENT_FIELDS[
                aging_mode
            ]
        )

        for scope_name, scope_records in scopes.items():
            for (
                first_field,
                second_field
            ) in combinations(
                treatment_fields,
                2
            ):
                first_values = np.asarray(
                    [
                        float(
                            record[first_field]
                        )
                        for record in scope_records
                    ],
                    dtype = float
                )

                second_values = np.asarray(
                    [
                        float(
                            record[second_field]
                        )
                        for record in scope_records
                    ],
                    dtype = float
                )

                first_standard_deviation = float(
                    np.std(
                        first_values
                    )
                )

                second_standard_deviation = float(
                    np.std(
                        second_values
                    )
                )

                if (
                    first_standard_deviation == 0
                    or second_standard_deviation == 0
                ):
                    correlation_value = None
                    correlation_status = (
                        "insufficient_variation"
                    )

                else:
                    correlation_value = float(
                        np.corrcoef(
                            first_values,
                            second_values
                        )[0, 1]
                    )

                    correlation_status = "completed"

                correlation_records.append(
                    {
                        "aging_mode": aging_mode,
                        "design_scope": scope_name,
                        "test_point_count": len(
                            scope_records
                        ),
                        "first_treatment": first_field,
                        "second_treatment": second_field,
                        "first_minimum": float(
                            np.min(
                                first_values
                            )
                        ),
                        "first_maximum": float(
                            np.max(
                                first_values
                            )
                        ),
                        "second_minimum": float(
                            np.min(
                                second_values
                            )
                        ),
                        "second_maximum": float(
                            np.max(
                                second_values
                            )
                        ),
                        "pearson_correlation": (
                            correlation_value
                        ),
                        "correlation_status": (
                            correlation_status
                        ),
                        "replication_weighting": (
                            "one_row_per_unique_test_point"
                        )
                    }
                )

    return sorted(
        correlation_records,
        key = lambda record: (
            record["aging_mode"],
            record["design_scope"],
            record["first_treatment"],
            record["second_treatment"]
        )
    )


# Standardizing one numerical design column
def standardize_design_column(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean_value = float(
        np.mean(
            values
        )
    )

    standard_deviation = float(
        np.std(
            values
        )
    )

    if standard_deviation == 0:
        return (
            np.zeros_like(
                values,
                dtype = float
            ),
            mean_value,
            standard_deviation
        )

    return (
        (
            values
            - mean_value
        )
        / standard_deviation,
        mean_value,
        standard_deviation
    )


# Building one numerical design matrix
def build_design_matrix(records: list[dict], aging_mode: str,
                        design_scope: str, specification: str
) -> tuple[np.ndarray, list[dict]]:
    treatment_fields = (
        NUMERICAL_TREATMENT_FIELDS[
            aging_mode
        ]
    )

    standardized_columns = {}
    term_metadata = []

    for field_name in treatment_fields:
        source_values = np.asarray(
            [
                float(
                    record[field_name]
                )
                for record in records
            ],
            dtype = float
        )

        (
            standardized_values,
            mean_value,
            standard_deviation
        ) = standardize_design_column(
            source_values
        )

        standardized_columns[
            field_name
        ] = standardized_values

        term_metadata.append(
            {
                "term_name": field_name,
                "term_type": "main_effect",
                "source_mean": mean_value,
                "source_standard_deviation": (
                    standard_deviation
                )
            }
        )

    columns = [
        np.ones(
            len(
                records
            ),
            dtype = float
        )
    ]

    matrix_terms = [
        {
            "term_name": "intercept",
            "term_type": "intercept",
            "source_mean": None,
            "source_standard_deviation": None
        }
    ]

    for term_record in term_metadata:
        columns.append(
            standardized_columns[
                term_record[
                    "term_name"
                ]
            ]
        )

        matrix_terms.append(
            term_record
        )

    if design_scope == "pooled_stages":
        stage_values = np.asarray(
            [
                1.0
                if record[
                    "dataset_stage"
                ]
                == "Stage_2"
                else 0.0
                for record in records
            ],
            dtype = float
        )

        (
            standardized_stage,
            stage_mean,
            stage_standard_deviation
        ) = standardize_design_column(
            stage_values
        )

        columns.append(
            standardized_stage
        )

        matrix_terms.append(
            {
                "term_name": "dataset_stage",
                "term_type": "stage_indicator",
                "source_mean": stage_mean,
                "source_standard_deviation": (
                    stage_standard_deviation
                )
            }
        )

    if (
        specification
        == "main_effects_and_pairwise_interactions"
    ):
        for (
            first_field,
            second_field
        ) in combinations(
            treatment_fields,
            2
        ):
            columns.append(
                standardized_columns[
                    first_field
                ]
                * standardized_columns[
                    second_field
                ]
            )

            matrix_terms.append(
                {
                    "term_name": (
                        f"{first_field}:"
                        f"{second_field}"
                    ),
                    "term_type": (
                        "pairwise_interaction"
                    ),
                    "source_mean": None,
                    "source_standard_deviation": None
                }
            )

    return (
        np.column_stack(
            columns
        ),
        matrix_terms
    )


# Building the realized test-point design for one analytical cohort
def build_analytical_test_point_design(test_point_records: list[dict], cell_records: list[dict],
                                       reference_records: list[dict], aging_mode: str,
                                       outcome_definition: dict, cohort_name: str,
                                       horizon_axis: str
) -> list[dict]:
    cell_records_by_identity = {
        cell_record["cell_identity"]: cell_record
        for cell_record in cell_records
        if cell_record["aging_mode"] == aging_mode
    }

    eligible_cells_by_test_point = defaultdict(
        set
    )

    eligible_observation_counts = Counter()

    for reference_record in reference_records:
        cell_identity = reference_record[
            "cell_identity"
        ]

        cell_record = cell_records_by_identity.get(
            cell_identity
        )

        if cell_record is None:
            continue

        if not record_supports_horizon(
            record = reference_record,
            outcome_definition = outcome_definition,
            cohort_name = cohort_name,
            horizon_axis = horizon_axis
        ):
            continue

        test_point_identity = build_test_point_identity(
            cell_record
        )

        eligible_cells_by_test_point[
            test_point_identity
        ].add(
            cell_identity
        )

        eligible_observation_counts[
            test_point_identity
        ] += 1

    analytical_test_point_records = []

    for test_point_record in test_point_records:
        if test_point_record["aging_mode"] != aging_mode:
            continue

        test_point_identity = test_point_record[
            "test_point_identity"
        ]

        eligible_cells = eligible_cells_by_test_point.get(
            test_point_identity,
            set()
        )

        if not eligible_cells:
            continue

        analytical_test_point_records.append(
            {
                **test_point_record,
                "analytical_independent_cell_count": len(
                    eligible_cells
                ),
                "eligible_cell_identities": sorted(
                    eligible_cells
                ),
                "analytical_observation_count": (
                    eligible_observation_counts[
                        test_point_identity
                    ]
                )
            }
        )

    return analytical_test_point_records


# Building the realized cell-level design for one analytical cohort
def build_analytical_cell_design(cell_records: list[dict], reference_records: list[dict],
                                 aging_mode: str, outcome_definition: dict,
                                 cohort_name: str, horizon_axis: str
) -> list[dict]:
    eligible_observation_counts = Counter()

    for reference_record in reference_records:
        if record_supports_horizon(
            record = reference_record,
            outcome_definition = outcome_definition,
            cohort_name = cohort_name,
            horizon_axis = horizon_axis
        ):
            eligible_observation_counts[
                reference_record["cell_identity"]
            ] += 1

    return [
        {
            **cell_record,
            "independent_cell_count": 1,
            "analytical_independent_cell_count": 1,
            "analytical_observation_count": (
                eligible_observation_counts[
                    cell_record["cell_identity"]
                ]
            )
        }
        for cell_record in cell_records
        if (
            cell_record["aging_mode"] == aging_mode
            and eligible_observation_counts[
                cell_record["cell_identity"]
            ] > 0
        )
    ]


# Selecting the cells that support one approved comparison horizon
def build_horizon_specific_cell_design(cell_records: list[dict], reference_records: list[dict],
                                       aging_mode: str, outcome_definition: dict,
                                       cohort_name: str, horizon_axis: str,
                                       horizon_definition: dict
) -> tuple[list[dict], list[dict]]:
    reference_records_by_cell = group_records_by_cell(
        reference_records
    )

    supported_records = []
    support_audit_records = []

    for cell_record in cell_records:
        if cell_record["aging_mode"] != aging_mode:
            continue

        cell_identity = cell_record[
            "cell_identity"
        ]

        eligible_horizons = []

        for reference_record in (
            reference_records_by_cell.get(
                cell_identity,
                []
            )
        ):
            if record_supports_horizon(
                record = reference_record,
                outcome_definition = outcome_definition,
                cohort_name = cohort_name,
                horizon_axis = horizon_axis
            ):
                eligible_horizons.append(
                    float(
                        reference_record[
                            horizon_axis
                        ]
                    )
                )

        support = evaluate_cell_horizon_support(
            horizon_values = eligible_horizons,
            horizon = float(
                horizon_definition["horizon"]
            ),
            interpolation_rule = horizon_definition[
                "interpolation_rule"
            ],
            maximum_bracketing_span = (
                horizon_definition[
                    "maximum_bracketing_span"
                ]
            )
        )

        support_audit_records.append(
            {
                "cell_identity": cell_identity,
                "condition_protocol_identity": (
                    cell_record[
                        "condition_protocol_identity"
                    ]
                ),
                "aging_mode": aging_mode,
                "dataset_stage": cell_record[
                    "dataset_stage"
                ],
                "sampling_method": cell_record[
                    "sampling_method"
                ],
                "outcome": next(
                    name
                    for name, definition in (
                        OUTCOME_DEFINITIONS.items()
                    )
                    if definition
                    is outcome_definition
                ),
                "cohort": cohort_name,
                "horizon_axis": horizon_axis,
                "horizon": float(
                    horizon_definition["horizon"]
                ),
                **support
            }
        )

        if support["supported"]:
            supported_records.append(
                {
                    **cell_record,
                    "independent_cell_count": 1,
                    "analytical_independent_cell_count": 1,
                    "analytical_observation_count": len(
                        eligible_horizons
                    ),
                    "comparison_horizon": float(
                        horizon_definition[
                            "horizon"
                        ]
                    ),
                    "horizon_support_type": support[
                        "support_type"
                    ]
                }
            )

    return (
        supported_records,
        support_audit_records
    )


# Adding one deterministic reference-coded categorical variable
def append_reference_coded_category(columns: list[np.ndarray], matrix_terms: list[dict],
                                    values: list[str], field_name: str,
                                    term_type: str
) -> None:
    levels = sorted(
        set(
            values
        )
    )

    for level in levels[1:]:
        columns.append(
            np.asarray(
                [
                    1.0
                    if value == level
                    else 0.0
                    for value in values
                ],
                dtype = float
            )
        )

        matrix_terms.append(
            {
                "term_name": (
                    f"{field_name}={level}"
                ),
                "term_type": term_type,
                "source_mean": None,
                "source_standard_deviation": None
            }
        )


# Adding calendar reference-schedule treatment terms
def append_calendar_schedule_terms(columns: list[np.ndarray], matrix_terms: list[dict],
                                   records: list[dict], specification: str
) -> None:
    schedule_levels = sorted(
        {
            record["reference_test_schedule"]
            for record in records
        }
    )

    for schedule_level in schedule_levels[1:]:
        schedule_indicator = np.asarray(
            [
                1.0
                if record["reference_test_schedule"]
                == schedule_level
                else 0.0
                for record in records
            ],
            dtype = float
        )

        schedule_term_name = (
            "reference_test_schedule="
            f"{schedule_level}"
        )

        columns.append(
            schedule_indicator
        )

        matrix_terms.append(
            {
                "term_name": schedule_term_name,
                "term_type": "target_treatment",
                "source_mean": None,
                "source_standard_deviation": None
            }
        )

        if (
            specification
            != "main_effects_and_pairwise_interactions"
        ):
            continue

        for field_name in (
            NUMERICAL_TREATMENT_FIELDS[
                "calendar_aging"
            ]
        ):
            field_values = np.asarray(
                [
                    float(record[field_name])
                    for record in records
                ],
                dtype = float
            )

            (
                standardized_values,
                _,
                _
            ) = standardize_design_column(
                field_values
            )

            columns.append(
                standardized_values
                * schedule_indicator
            )

            matrix_terms.append(
                {
                    "term_name": (
                        f"{field_name}:"
                        f"{schedule_term_name}"
                    ),
                    "term_type": "pairwise_interaction",
                    "source_mean": None,
                    "source_standard_deviation": None
                }
            )


# Building a validated equipment-allocation signature
def build_equipment_signature(cell_identities: list[str], equipment_by_cell: dict,
                              equipment_role: str, output_field: str
) -> str:
    conflicted_any = False
    validated_values = set()

    for cell_identity in cell_identities:
        role_inventory = equipment_by_cell.get(
            cell_identity,
            {}
        ).get(
            equipment_role,
            {}
        )

        if role_inventory.get(
            "identity_conflicted",
            {}
        ).get(
            output_field,
            set()
        ):
            conflicted_any = True

        validated_values.update(
            role_inventory.get(
                "validated",
                {}
            ).get(
                output_field,
                set()
            )
        )

    if conflicted_any:
        return "__identity_conflicted__"

    if not validated_values:
        return "__unavailable__"

    return json.dumps(
        sorted(validated_values),
        separators = (
            ",",
            ":"
        )
    )


# Building one cell-level matrix for one defined context family
def build_context_adjusted_design_matrix(records: list[dict], aging_mode: str,
                                         design_scope: str, specification: str,
                                         equipment_by_cell: dict, context_family: str
) -> tuple[np.ndarray, list[dict]]:
    (
        treatment_matrix,
        matrix_terms
    ) = build_design_matrix(
        records = records,
        aging_mode = aging_mode,
        design_scope = design_scope,
        specification = specification
    )

    columns = [
        treatment_matrix[
            :,
            column_index
        ]
        for column_index in range(
            treatment_matrix.shape[1]
        )
    ]

    # Retaining reference-test schedule as a calendar-aging target treatment
    if aging_mode == "calendar_aging":
        append_calendar_schedule_terms(
            columns = columns,
            matrix_terms = matrix_terms,
            records = records,
            specification = specification
        )

    if context_family == CONTEXT_FAMILY_NONE:
        return (
            np.column_stack(
                columns
            ),
            matrix_terms
        )

    if context_family == CONTEXT_FAMILY_LABORATORY:
        append_reference_coded_category(
            columns = columns,
            matrix_terms = matrix_terms,
            values = [
                record["laboratory"]
                for record in records
            ],
            field_name = "laboratory",
            term_type = "design_context"
        )

        return (
            np.column_stack(
                columns
            ),
            matrix_terms
        )

    matching_equipment_families = [
        (
            equipment_role,
            output_field
        )
        for equipment_role in [
            "reference_measurement",
            "cycle_aging_exposure"
        ]
        for output_field in EQUIPMENT_FIELDS
        if context_family
        == f"{equipment_role}_{output_field}"
    ]

    if len(matching_equipment_families) != 1:
        raise RuntimeError(f"Unknown context family : {context_family}")

    (
        equipment_role,
        output_field
    ) = matching_equipment_families[0]

    if (
        equipment_role
        == "cycle_aging_exposure"
        and aging_mode != "cycle_aging"
    ):
        raise RuntimeError("Cycle-aging exposure equipment cannot enter a calendar-aging design matrix.")

    append_reference_coded_category(
        columns = columns,
        matrix_terms = matrix_terms,
        values = [
            build_equipment_signature(
                cell_identities = [
                    record["cell_identity"]
                ],
                equipment_by_cell = equipment_by_cell,
                equipment_role = equipment_role,
                output_field = output_field
            )
            for record in records
        ],
        field_name = context_family,
        term_type = "design_context"
    )

    return (
        np.column_stack(
            columns
        ),
        matrix_terms
    )


# Splitting one record population into pooled, stage and sampling-method scopes
def build_design_scopes(records: list[dict]) -> dict[str, list[dict]]:
    scopes = {
        "pooled_stages": records
    }

    for stage_name in sorted(
        {
            record["dataset_stage"]
            for record in records
        }
    ):
        stage_records = [
            record
            for record in records
            if record["dataset_stage"]
            == stage_name
        ]

        scopes[
            stage_name
        ] = stage_records

        for sampling_method in sorted(
            {
                record["sampling_method"]
                for record in stage_records
            }
        ):
            scopes[
                (
                    f"{stage_name}/"
                    f"sampling_method={sampling_method}"
                )
            ] = [
                record
                for record in stage_records
                if record["sampling_method"]
                == sampling_method
            ]

    return scopes


# Collecting condition-protocol identities from singular and plural fields
def collect_condition_protocol_identities(records: list[dict]) -> set[str]:
    identities = set()

    for record in records:
        singular_identity = record.get(
            "condition_protocol_identity"
        )

        if singular_identity is not None:
            identities.add(
                singular_identity
            )

        plural_identities = record.get(
            "condition_protocol_identities"
        )

        if plural_identities is None:
            continue

        if (
            not isinstance(plural_identities, list)
            or any(
                not isinstance(identity, str)
                or not identity
                for identity in plural_identities
            )
        ):
            raise RuntimeError("Design record contains invalid condition_protocol_identities.")

        identities.update(
            plural_identities
        )

    return identities


# Building diagnostics for one assigned or realized design population
def build_population_identifiability(records: list[dict], aging_mode: str,
                                     analysis_population: str, outcome: str | None,
                                     cohort: str | None, horizon_axis: str | None,
                                     equipment_by_cell: dict, context_adjusted: bool,
                                     context_family: str | None, comparison_horizon: float | None = None,
                                     analysis_role: str | None = None, support_region_identity: str | None = None,
                                     support_region_kind: str | None = None, support_interval_minimum: float | None = None,
                                     support_interval_maximum: float | None = None, support_interval_lower_inclusive: bool | None = None, 
                                     support_interval_upper_inclusive: bool | None = None
) -> list[dict]:
    diagnostic_records = []

    scopes = build_design_scopes(
        records
    )

    for scope_name, scope_records in scopes.items():
        if not scope_records:
            raise RuntimeError(
                "Design-identifiability population does not contain records : "
                f"{aging_mode} : {analysis_population} : {scope_name} : {outcome} : {cohort} : {horizon_axis}"
            )

        if context_adjusted:
            test_point_count = len(
                {
                    build_test_point_identity(
                        record
                    )
                    for record in scope_records
                }
            )

            independent_cell_count = len(
                scope_records
            )

            analytical_observation_count = (
                sum(
                    int(
                        record[
                            "analytical_observation_count"
                        ]
                    )
                    for record in scope_records
                )
                if all(
                    "analytical_observation_count"
                    in record
                    for record in scope_records
                )
                else None
            )

        else:
            test_point_count = len(
                scope_records
            )

            independent_cell_count = sum(
                int(
                    record.get(
                        "analytical_independent_cell_count",
                        record["independent_cell_count"]
                    )
                )
                for record in scope_records
            )

            analytical_observation_count = (
                sum(
                    int(record["analytical_observation_count"])
                    for record in scope_records
                )
                if all(
                    "analytical_observation_count"
                    in record
                    for record in scope_records
                )
                else None
            )

        condition_protocol_count = len(
            collect_condition_protocol_identities(
                scope_records
            )
        )

        for specification in DESIGN_SPECIFICATIONS:
            if context_adjusted:
                if context_family is None:
                    raise RuntimeError("Context-adjusted identifiability requires one explicit context family.")

                (
                    design_matrix,
                    matrix_terms
                ) = build_context_adjusted_design_matrix(
                    records = scope_records,
                    aging_mode = aging_mode,
                    design_scope = scope_name,
                    specification = specification,
                    equipment_by_cell = equipment_by_cell,
                    context_family = context_family
                )

                matrix_scope = (
                    "context_family_cell_allocation"
                )

            else:
                if context_family is not None:
                    raise RuntimeError("Test-point treatment matrices cannot contain a cell-level context family.")

                (
                    design_matrix,
                    matrix_terms
                ) = build_design_matrix(
                    records = scope_records,
                    aging_mode = aging_mode,
                    design_scope = scope_name,
                    specification = specification
                )

                matrix_scope = (
                    "standardized_treatments_plus_stage_indicator"
                    if scope_name == "pooled_stages"
                    else "standardized_treatments_only"
                )

            context_term_count = sum(
                term_record[
                    "term_type"
                ]
                == "design_context"
                for term_record in matrix_terms
            )

            singular_values = np.linalg.svd(
                design_matrix,
                compute_uv = False
            )

            matrix_rank = int(
                np.linalg.matrix_rank(
                    design_matrix
                )
            )

            column_count = int(
                design_matrix.shape[1]
            )

            full_column_rank = (
                matrix_rank
                == column_count
            )

            residual_independent_cell_degrees_of_freedom = (
                independent_cell_count
                - matrix_rank
            )

            effective_condition_number = None
            smallest_rank_singular_value = None

            if matrix_rank > 0:
                smallest_rank_singular_value = float(
                    singular_values[
                        matrix_rank - 1
                    ]
                )

                if smallest_rank_singular_value > 0:
                    effective_condition_number = float(
                        singular_values[0]
                        / smallest_rank_singular_value
                    )

            if (
                full_column_rank
                and singular_values[-1] > 0
            ):
                condition_number = float(
                    singular_values[0]
                    / singular_values[-1]
                )

            else:
                condition_number = None

            diagnostic_records.append(
                {
                    "record_type": "matrix_summary",
                    "analysis_population": (
                        analysis_population
                    ),
                    "aging_mode": aging_mode,
                    "design_scope": scope_name,
                    "outcome": outcome,
                    "cohort": cohort,
                    "horizon_axis": horizon_axis,
                    "comparison_horizon": (
                        comparison_horizon
                    ),
                    "analysis_role": (
                        analysis_role
                    ),
                    "support_region_identity": (
                        support_region_identity
                    ),
                    "support_region_kind": (
                        support_region_kind
                    ),
                    "support_interval_minimum": (
                        support_interval_minimum
                    ),
                    "support_interval_maximum": (
                        support_interval_maximum
                    ),
                    "support_interval_lower_inclusive": (
                        support_interval_lower_inclusive
                    ),
                    "support_interval_upper_inclusive": (
                        support_interval_upper_inclusive
                    ),
                    "specification": specification,
                    "matrix_scope": matrix_scope,
                    "sampling_design_scope": (
                        scope_name
                        if "sampling_method=" in scope_name
                        else "combined_within_scope"
                    ),
                    "context_family": (
                        context_family
                    ),
                    "context_term_count": (
                        context_term_count
                    ),
                    "test_point_count": test_point_count,
                    "independent_cell_count": (
                        independent_cell_count
                    ),
                    "condition_protocol_count": (
                        condition_protocol_count
                    ),
                    "analytical_observation_count": (
                        analytical_observation_count
                    ),
                    "term_name": None,
                    "term_type": None,
                    "column_count": column_count,
                    "matrix_rank": matrix_rank,
                    "full_column_rank": (
                        full_column_rank
                    ),
                    "largest_singular_value": float(
                        singular_values[0]
                    ),
                    "smallest_singular_value": float(
                        singular_values[-1]
                    ),
                    "condition_number": (
                        condition_number
                    ),
                    "residual_independent_cell_degrees_of_freedom": (
                        residual_independent_cell_degrees_of_freedom
                    ),
                    "smallest_rank_singular_value": (
                        smallest_rank_singular_value
                    ),
                    "effective_condition_number": (
                        effective_condition_number
                    ),
                    "structurally_estimable": None,
                    "source_standard_deviation": None,
                    "causal_interpretation": (
                        "not_established"
                    )
                }
            )

            for column_index, term_record in enumerate(
                matrix_terms
            ):
                reduced_matrix = np.delete(
                    design_matrix,
                    column_index,
                    axis = 1
                )

                reduced_rank = int(
                    np.linalg.matrix_rank(
                        reduced_matrix
                    )
                )

                structurally_estimable = (
                    matrix_rank
                    > reduced_rank
                )

                diagnostic_records.append(
                    {
                        "record_type": "term",
                        "analysis_population": (
                            analysis_population
                        ),
                        "aging_mode": aging_mode,
                        "design_scope": scope_name,
                        "outcome": outcome,
                        "cohort": cohort,
                        "horizon_axis": horizon_axis,
                        "comparison_horizon": (
                            comparison_horizon
                        ),
                        "analysis_role": (
                            analysis_role
                        ),
                        "support_region_identity": (
                            support_region_identity
                        ),
                        "support_region_kind": (
                            support_region_kind
                        ),
                        "support_interval_minimum": (
                            support_interval_minimum
                        ),
                        "support_interval_maximum": (
                            support_interval_maximum
                        ),
                        "support_interval_lower_inclusive": (
                            support_interval_lower_inclusive
                        ),
                        "support_interval_upper_inclusive": (
                            support_interval_upper_inclusive
                        ),
                        "specification": specification,
                        "matrix_scope": matrix_scope,
                        "sampling_design_scope": (
                            scope_name
                            if "sampling_method=" in scope_name
                            else "combined_within_scope"
                        ),
                        "context_family": (
                            context_family
                        ),
                        "context_term_count": (
                            context_term_count
                        ),
                        "test_point_count": test_point_count,
                        "independent_cell_count": (
                            independent_cell_count
                        ),
                        "condition_protocol_count": (
                            condition_protocol_count
                        ),
                        "analytical_observation_count": (
                            analytical_observation_count
                        ),
                        "term_name": term_record[
                            "term_name"
                        ],
                        "term_type": term_record[
                            "term_type"
                        ],
                        "column_count": column_count,
                        "matrix_rank": matrix_rank,
                        "full_column_rank": (
                            full_column_rank
                        ),
                        "largest_singular_value": float(
                            singular_values[0]
                        ),
                        "smallest_singular_value": float(
                            singular_values[-1]
                        ),
                        "condition_number": (
                            condition_number
                        ),
                        "residual_independent_cell_degrees_of_freedom": (
                            residual_independent_cell_degrees_of_freedom
                        ),
                        "smallest_rank_singular_value": (
                            smallest_rank_singular_value
                        ),
                        "effective_condition_number": (
                            effective_condition_number
                        ),
                        "structurally_estimable": (
                            structurally_estimable
                        ),
                        "source_standard_deviation": (
                            term_record[
                                "source_standard_deviation"
                            ]
                        ),
                        "causal_interpretation": (
                            "not_established"
                        )
                    }
                )

    return diagnostic_records


# Building structural identifiability for assigned and realized designs
def build_design_identifiability(test_point_records: list[dict], cell_records: list[dict],
                                 reference_records: list[dict], equipment_by_cell: dict,
                                 approved_horizon_inventory: dict[tuple[str, str], list[dict]]
) -> tuple[list[dict], list[dict]]:
    diagnostic_records = []
    horizon_support_audit_records = []

    for aging_mode in [
        "calendar_aging",
        "cycle_aging"
    ]:
        assigned_test_point_records = [
            record
            for record in test_point_records
            if record["aging_mode"] == aging_mode
        ]

        diagnostic_records.extend(
            build_population_identifiability(
                records = assigned_test_point_records,
                aging_mode = aging_mode,
                analysis_population = (
                    "assigned_test_point_design"
                ),
                outcome = None,
                cohort = None,
                horizon_axis = None,
                equipment_by_cell = equipment_by_cell,
                context_adjusted = False,
                context_family = None
            )
        )

        assigned_cell_records = [
            record
            for record in cell_records
            if record["aging_mode"] == aging_mode
        ]

        for context_family in build_context_families(
            aging_mode
        ):
            diagnostic_records.extend(
                build_population_identifiability(
                    records = assigned_cell_records,
                    aging_mode = aging_mode,
                    analysis_population = (
                        "assigned_cell_design"
                    ),
                    outcome = None,
                    cohort = None,
                    horizon_axis = None,
                    equipment_by_cell = equipment_by_cell,
                    context_adjusted = True,
                    context_family = context_family
                )
            )

        horizon_axes = [
            "elapsed_calendar_days"
        ]

        if aging_mode == "cycle_aging":
            horizon_axes.append(
                "cumulative_full_equivalent_cycles_observed"
            )

        for (
            outcome_name,
            outcome_definition
        ) in OUTCOME_DEFINITIONS.items():
            for cohort_name in [
                "primary",
                "sensitivity"
            ]:
                for horizon_axis in horizon_axes:
                    analytical_test_point_records = (
                        build_analytical_test_point_design(
                            test_point_records = (
                                test_point_records
                            ),
                            cell_records = cell_records,
                            reference_records = (
                                reference_records
                            ),
                            aging_mode = aging_mode,
                            outcome_definition = (
                                outcome_definition
                            ),
                            cohort_name = cohort_name,
                            horizon_axis = horizon_axis
                        )
                    )

                    diagnostic_records.extend(
                        build_population_identifiability(
                            records = (
                                analytical_test_point_records
                            ),
                            aging_mode = aging_mode,
                            analysis_population = (
                                "eligible_positive_horizon_test_point_design"
                            ),
                            outcome = outcome_name,
                            cohort = cohort_name,
                            horizon_axis = horizon_axis,
                            equipment_by_cell = equipment_by_cell,
                            context_adjusted = False,
                            context_family = None
                        )
                    )

                    analytical_cell_records = (
                        build_analytical_cell_design(
                            cell_records = cell_records,
                            reference_records = (
                                reference_records
                            ),
                            aging_mode = aging_mode,
                            outcome_definition = (
                                outcome_definition
                            ),
                            cohort_name = cohort_name,
                            horizon_axis = horizon_axis
                        )
                    )

                    if not analytical_cell_records:
                        continue

                    for context_family in (
                        build_context_families(
                            aging_mode
                        )
                    ):
                        diagnostic_records.extend(
                            build_population_identifiability(
                                records = (
                                    analytical_cell_records
                                ),
                                aging_mode = aging_mode,
                                analysis_population = (
                                    "eligible_positive_horizon_cell_design"
                                ),
                                outcome = outcome_name,
                                cohort = cohort_name,
                                horizon_axis = horizon_axis,
                                equipment_by_cell = (
                                    equipment_by_cell
                                ),
                                context_adjusted = True,
                                context_family = (
                                    context_family
                                )
                            )
                        )

                    for horizon_definition in (
                        approved_horizon_inventory.get(
                            (
                                aging_mode,
                                horizon_axis
                            ),
                            []
                        )
                    ):
                        (
                            horizon_specific_cell_records,
                            horizon_support_audit_records_for_definition
                        ) = build_horizon_specific_cell_design(
                            cell_records = cell_records,
                            reference_records = (
                                reference_records
                            ),
                            aging_mode = aging_mode,
                            outcome_definition = (
                                outcome_definition
                            ),
                            cohort_name = cohort_name,
                            horizon_axis = horizon_axis,
                            horizon_definition = (
                                horizon_definition
                            )
                        )

                        horizon_support_audit_records.extend(
                            horizon_support_audit_records_for_definition
                        )

                        if not horizon_specific_cell_records:
                            continue

                        for context_family in (
                            build_context_families(
                                aging_mode
                            )
                        ):
                            diagnostic_records.extend(
                                build_population_identifiability(
                                    records = (
                                        horizon_specific_cell_records
                                    ),
                                    aging_mode = aging_mode,
                                    analysis_population = (
                                        "approved_horizon_cell_design"
                                    ),
                                    outcome = outcome_name,
                                    cohort = cohort_name,
                                    horizon_axis = horizon_axis,
                                    equipment_by_cell = (
                                        equipment_by_cell
                                    ),
                                    context_adjusted = True,
                                    context_family = (
                                        context_family
                                    ),
                                    comparison_horizon = float(
                                        horizon_definition[
                                            "horizon"
                                        ]
                                    ),
                                    analysis_role = (
                                        horizon_definition[
                                            "analysis_role"
                                        ]
                                    )
                                )
                            )

    return (
        sorted(
            diagnostic_records,
            key = lambda record: (
                record["analysis_population"],
                record["aging_mode"],
                record["design_scope"],
                record["outcome"]
                or "",
                record["cohort"]
                or "",
                record["horizon_axis"]
                or "",
                record["specification"],
                record["record_type"],
                record["context_family"]
                or "",
                record["term_name"]
                or ""
            )
        ),
        horizon_support_audit_records
    )


# Building exact, marginal and nearest-distance stage overlap
def build_stage_overlap(test_point_records: list[dict]) -> list[dict]:
    overlap_records = []

    for aging_mode in [
        "calendar_aging",
        "cycle_aging"
    ]:
        treatment_fields = (
            NUMERICAL_TREATMENT_FIELDS[
                aging_mode
            ]
        )

        mode_records = [
            record
            for record in test_point_records
            if record[
                "aging_mode"
            ]
            == aging_mode
        ]

        if not mode_records:
            continue

        stage_records = {
            stage_name: [
                record
                for record in mode_records
                if record[
                    "dataset_stage"
                ]
                == stage_name
            ]
            for stage_name in [
                "Stage_1",
                "Stage_2"
            ]
        }

        stage_treatment_vectors = {
            stage_name: {
                tuple(
                    float(
                        record[field_name]
                    )
                    for field_name in treatment_fields
                )
                for record in records
            }
            for stage_name, records in (
                stage_records.items()
            )
        }

        exact_overlap_count = len(
            stage_treatment_vectors[
                "Stage_1"
            ].intersection(
                stage_treatment_vectors[
                    "Stage_2"
                ]
            )
        )

        overlap_records.append(
            {
                "record_type": (
                    "exact_combination_summary"
                ),
                "aging_mode": aging_mode,
                "overlap_basis": "numerical_treatments",
                "source_stage": None,
                "target_stage": None,
                "test_point_identity": None,
                "nearest_test_point_identity": None,
                "treatment_field": None,
                "stage_1_minimum": None,
                "stage_1_maximum": None,
                "stage_2_minimum": None,
                "stage_2_maximum": None,
                "overlap_minimum": None,
                "overlap_maximum": None,
                "overlap_available": None,
                "exact_overlap_count": (
                    exact_overlap_count
                ),
                "range_scaled_nearest_distance": None
            }
        )

        if aging_mode == "calendar_aging":
            stage_protocol_vectors = {
                stage_name: {
                    (
                        *(
                            float(record[field_name])
                            for field_name in treatment_fields
                        ),
                        schedule
                    )
                    for record in records
                    for schedule in record[
                        "reference_test_schedules"
                    ]
                }
                for stage_name, records in (
                    stage_records.items()
                )
            }

            protocol_overlap_count = len(
                stage_protocol_vectors[
                    "Stage_1"
                ].intersection(
                    stage_protocol_vectors[
                        "Stage_2"
                    ]
                )
            )

            overlap_records.append(
                {
                    "record_type": (
                        "exact_condition_protocol_summary"
                    ),
                    "aging_mode": aging_mode,
                    "overlap_basis": (
                        "numerical_treatments_and_reference_test_schedule"
                    ),
                    "source_stage": None,
                    "target_stage": None,
                    "test_point_identity": None,
                    "nearest_test_point_identity": None,
                    "treatment_field": None,
                    "stage_1_minimum": None,
                    "stage_1_maximum": None,
                    "stage_2_minimum": None,
                    "stage_2_maximum": None,
                    "overlap_minimum": None,
                    "overlap_maximum": None,
                    "overlap_available": None,
                    "exact_overlap_count": (
                        protocol_overlap_count
                    ),
                    "range_scaled_nearest_distance": None
                }
            )

        treatment_ranges = {}

        for field_name in treatment_fields:
            stage_1_values = [
                float(
                    record[field_name]
                )
                for record in stage_records[
                    "Stage_1"
                ]
            ]

            stage_2_values = [
                float(
                    record[field_name]
                )
                for record in stage_records[
                    "Stage_2"
                ]
            ]

            pooled_values = [
                *stage_1_values,
                *stage_2_values
            ]

            treatment_ranges[
                field_name
            ] = (
                min(
                    pooled_values
                ),
                max(
                    pooled_values
                )
            )

            overlap_minimum = max(
                min(
                    stage_1_values
                ),
                min(
                    stage_2_values
                )
            )

            overlap_maximum = min(
                max(
                    stage_1_values
                ),
                max(
                    stage_2_values
                )
            )

            overlap_available = (
                overlap_minimum
                <= overlap_maximum
            )

            overlap_records.append(
                {
                    "record_type": "marginal_range",
                    "aging_mode": aging_mode,
                    "overlap_basis": "numerical_treatments",
                    "source_stage": None,
                    "target_stage": None,
                    "test_point_identity": None,
                    "nearest_test_point_identity": None,
                    "treatment_field": field_name,
                    "stage_1_minimum": min(
                        stage_1_values
                    ),
                    "stage_1_maximum": max(
                        stage_1_values
                    ),
                    "stage_2_minimum": min(
                        stage_2_values
                    ),
                    "stage_2_maximum": max(
                        stage_2_values
                    ),
                    "overlap_minimum": (
                        overlap_minimum
                        if overlap_available
                        else None
                    ),
                    "overlap_maximum": (
                        overlap_maximum
                        if overlap_available
                        else None
                    ),
                    "overlap_available": (
                        overlap_available
                    ),
                    "exact_overlap_count": None,
                    "range_scaled_nearest_distance": None
                }
            )

        for source_stage, target_stage in [
            (
                "Stage_1",
                "Stage_2"
            ),
            (
                "Stage_2",
                "Stage_1"
            )
        ]:
            for source_record in stage_records[
                source_stage
            ]:
                nearest_record = None
                nearest_distance = None

                for target_record in stage_records[
                    target_stage
                ]:
                    squared_distance = 0.0

                    for field_name in treatment_fields:
                        (
                            pooled_minimum,
                            pooled_maximum
                        ) = treatment_ranges[
                            field_name
                        ]

                        pooled_range = (
                            pooled_maximum
                            - pooled_minimum
                        )

                        if pooled_range == 0:
                            standardized_difference = 0.0

                        else:
                            standardized_difference = (
                                float(
                                    source_record[
                                        field_name
                                    ]
                                )
                                - float(
                                    target_record[
                                        field_name
                                    ]
                                )
                            ) / pooled_range

                        squared_distance += (
                            standardized_difference
                            ** 2
                        )

                    distance = math.sqrt(
                        squared_distance
                    )

                    candidate_identity = (
                        target_record[
                            "test_point_identity"
                        ]
                    )

                    if (
                        nearest_distance is None
                        or distance < nearest_distance
                        or (
                            distance == nearest_distance
                            and candidate_identity
                            < nearest_record[
                                "test_point_identity"
                            ]
                        )
                    ):
                        nearest_distance = distance
                        nearest_record = target_record

                overlap_records.append(
                    {
                        "record_type": (
                            "nearest_test_point"
                        ),
                        "aging_mode": aging_mode,
                        "overlap_basis": "numerical_treatments",
                        "source_stage": source_stage,
                        "target_stage": target_stage,
                        "test_point_identity": (
                            source_record[
                                "test_point_identity"
                            ]
                        ),
                        "nearest_test_point_identity": (
                            nearest_record[
                                "test_point_identity"
                            ]
                        ),
                        "treatment_field": None,
                        "stage_1_minimum": None,
                        "stage_1_maximum": None,
                        "stage_2_minimum": None,
                        "stage_2_maximum": None,
                        "overlap_minimum": None,
                        "overlap_maximum": None,
                        "overlap_available": None,
                        "exact_overlap_count": None,
                        "range_scaled_nearest_distance": (
                            nearest_distance
                        )
                    }
                )

    return sorted(
        overlap_records,
        key = lambda record: (
            record["aging_mode"],
            record["record_type"],
            record["source_stage"]
            or "",
            record["test_point_identity"]
            or "",
            record["treatment_field"]
            or ""
        )
    )


# Summarizing the findings that determine downstream analytical scope
def build_design_findings(estimand_records: list[dict], identifiability_records: list[dict],
                          correlation_records: list[dict], stage_overlap_records: list[dict]
) -> dict:
    support_counts = Counter(
        (
            record["aging_mode"],
            record["final_support_classification"]
        )
        for record in estimand_records
    )

    support_detail_counts = Counter(
        (
            record["aging_mode"],
            record["direct_contrast_support_detail"]
        )
        for record in estimand_records
    )

    support_details_by_mode = [
        {
            "aging_mode": aging_mode,
            "support_detail": support_detail,
            "decision_count": decision_count
        }
        for (
            aging_mode,
            support_detail
        ), decision_count in sorted(
            support_detail_counts.items()
        )
    ]

    support_counts_by_mode = [
        {
            "aging_mode": aging_mode,
            "support_status": support_status,
            "decision_count": decision_count
        }
        for (
            aging_mode,
            support_status
        ), decision_count in sorted(
            support_counts.items()
        )
    ]

    matrix_summaries = [
        {
            "analysis_population": record[
                "analysis_population"
            ],
            "aging_mode": record["aging_mode"],
            "design_scope": record["design_scope"],
            "outcome": record["outcome"],
            "cohort": record["cohort"],
            "horizon_axis": record["horizon_axis"],
            "specification": record["specification"],
            "context_family": record[
                "context_family"
            ],
            "test_point_count": record[
                "test_point_count"
            ],
            "column_count": record["column_count"],
            "matrix_rank": record["matrix_rank"],
            "full_column_rank": record[
                "full_column_rank"
            ],
            "condition_number": record[
                "condition_number"
            ]
        }
        for record in identifiability_records
        if record["record_type"] == "matrix_summary"
    ]

    term_estimability_counts = Counter(
        (
            record["analysis_population"],
            record["aging_mode"],
            record["design_scope"],
            record["outcome"] or "",
            record["cohort"] or "",
            record["horizon_axis"] or "",
            record["specification"],
            record["context_family"] or "",
            bool(record["structurally_estimable"])
        )
        for record in identifiability_records
        if (
            record["record_type"] == "term"
            and record["term_type"]
            in TARGET_DESIGN_TERM_TYPES
        )
    )

    term_estimability_aggregates = [
        {
            "analysis_population": key[0],
            "aging_mode": key[1],
            "design_scope": key[2],
            "outcome": key[3],
            "cohort": key[4],
            "horizon_axis": key[5],
            "specification": key[6],
            "context_family": key[7],
            "structurally_estimable": key[8],
            "term_count": count
        }
        for key, count in sorted(
            term_estimability_counts.items()
        )
    ]

    strongest_correlations = []

    correlation_groups = defaultdict(
        list
    )

    for record in correlation_records:
        if (
            record["correlation_status"]
            != "completed"
            or record["pearson_correlation"]
            is None
        ):
            continue

        correlation_groups[
            (
                record["aging_mode"],
                record["design_scope"]
            )
        ].append(
            record
        )

    for (
        aging_mode,
        design_scope
    ), group_records in sorted(
        correlation_groups.items()
    ):
        strongest_record = max(
            group_records,
            key = lambda record: (
                abs(
                    record[
                        "pearson_correlation"
                    ]
                ),
                record["first_treatment"],
                record["second_treatment"]
            )
        )

        strongest_correlations.append(
            {
                "aging_mode": aging_mode,
                "design_scope": design_scope,
                "first_treatment": strongest_record[
                    "first_treatment"
                ],
                "second_treatment": strongest_record[
                    "second_treatment"
                ],
                "pearson_correlation": strongest_record[
                    "pearson_correlation"
                ]
            }
        )

    exact_stage_overlap = [
        {
            "aging_mode": record["aging_mode"],
            "overlap_basis": record[
                "overlap_basis"
            ],
            "exact_overlap_count": record[
                "exact_overlap_count"
            ]
        }
        for record in stage_overlap_records
        if record["record_type"] in {
            "exact_combination_summary",
            "exact_condition_protocol_summary"
        }
    ]

    return {
        "support_counts_by_mode": (
            support_counts_by_mode
        ),
        "support_details_by_mode": (
            support_details_by_mode
        ),
        "matrix_summaries": matrix_summaries,
        "term_estimability_aggregates": (
            term_estimability_aggregates
        ),
        "strongest_treatment_correlations": (
            strongest_correlations
        ),
        "exact_stage_overlap": (
            exact_stage_overlap
        )
    }


# Creating one unique staging output directory on the same filesystem
def create_staging_output_directory(output_directory: Path) -> Path:
    output_directory.parent.mkdir(
        parents = True,
        exist_ok = True
    )

    return Path(
        tempfile.mkdtemp(
            prefix = (
                f"{output_directory.name}.staging."
            ),
            dir = output_directory.parent
        )
    )


# Publishing staging output transactionally
def publish_staging_output_directory(staging_directory: Path, output_directory: Path) -> None:
    backup_directory = output_directory.with_name(
        output_directory.name + ".backup"
    )

    if backup_directory.exists():
        raise RuntimeError(f"Stale backup directory exists : {backup_directory}")

    if output_directory.exists():
        output_directory.rename(
            backup_directory
        )

    try:
        staging_directory.rename(
            output_directory
        )

    except Exception:
        if (
            not output_directory.exists()
            and backup_directory.exists()
        ):
            backup_directory.rename(
                output_directory
            )

        raise

    if backup_directory.exists():
        shutil.rmtree(
            backup_directory
        )


# Hashing one canonical JSON value deterministically
def canonical_value_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys = True,
            separators = (",", ":")
        ).encode(
            "utf-8"
        )
    ).hexdigest()


# Projecting large horizon-support fields to counts and digests
def compact_horizon_support_records(records: list[dict]) -> list[dict]:
    compact_records = []

    large_fields = {
        "exact_common_horizons",
        "observed_horizon_coverage_inventory",
        "coverage_eligible_observed_horizons"
    }

    for record in records:
        compact_record = {
            field_name: field_value
            for field_name, field_value in record.items()
            if field_name not in large_fields
        }

        for field_name in sorted(large_fields):
            values = record[field_name]

            compact_record[
                f"{field_name}_count"
            ] = len(values)

            compact_record[
                f"{field_name}_sha256"
            ] = canonical_value_sha256(
                values
            )

        compact_records.append(
            compact_record
        )

    return compact_records


# Projecting large estimand-support fields to counts and digests
def compact_estimand_support_records(records: list[dict]) -> list[dict]:
    omitted_fields = {
        "equipment_allocation_relationships",
        "first_supported_continuous_intervals",
        "second_supported_continuous_intervals",
        "coverage_eligible_observed_horizons",
        "exact_common_horizons"
    }

    compact_records = []

    for record in records:
        compact_record = {
            field_name: field_value
            for field_name, field_value in record.items()
            if field_name not in omitted_fields
        }

        for field_name in sorted(omitted_fields):
            value = record[field_name]

            compact_record[
                f"{field_name}_count"
            ] = len(value)

            compact_record[
                f"{field_name}_sha256"
            ] = canonical_value_sha256(
                value
            )

        compact_record[
            "equipment_allocation_source"
        ] = "treatment_support.csv"

        compact_record[
            "condition_horizon_support_source"
        ] = "horizon_support.csv"

        compact_records.append(
            compact_record
        )

    return compact_records


# Writing every output table and the report, then publishing atomically
def write_and_publish_design_artifacts(output_directory: Path, output_tables: dict[str, list[dict]],
                                       report_without_tables: dict
) -> Path:
    if "tables" in report_without_tables:
        raise RuntimeError("Report must not contain table metadata before tables are written.")

    staging_directory = (
        create_staging_output_directory(
            output_directory
        )
    )

    try:
        for table_name, records in (
            output_tables.items()
        ):
            write_report_table(
                output_path = (
                    staging_directory
                    / f"{table_name}.csv"
                ),
                records = records
            )

        completed_report = {
            **report_without_tables,
            "tables": {
                table_name: {
                    "file_name": (
                        f"{table_name}.csv"
                    ),
                    "row_count": len(
                        records
                    ),
                    "sha256": calculate_sha256(
                        staging_directory
                        / f"{table_name}.csv"
                    ),
                    "schema": (
                        build_report_table_schema(
                            records
                        )
                    )
                }
                for table_name, records in (
                    output_tables.items()
                )
            }
        }

        staging_report_path = (
            staging_directory
            / "experimental_design_report.json"
        )

        staging_report_path.write_text(
            (
                json.dumps(
                    completed_report,
                    indent = 2,
                    sort_keys = True
                )
                + "\n"
            ),
            encoding = "utf-8"
        )

        publish_staging_output_directory(
            staging_directory = staging_directory,
            output_directory = output_directory
        )

    finally:
        if staging_directory.exists():
            shutil.rmtree(
                staging_directory
            )

    return (
        output_directory
        / "experimental_design_report.json"
    )


# Running the complete experimental-design audit
def main() -> None:
    args = parse_args()

    analysis_policy = load_analysis_policy(
        args.analysis_policy_path
    )

    (
        feature_manifest,
        feature_tables,
        audit_result
    ) = load_validated_inputs(
        manifest_path = args.manifest_path,
        audit_path = args.audit_path
    )

    cell_records = feature_tables[
        "cell_features"
    ]

    condition_records = feature_tables[
        "condition_features"
    ]

    reference_records = feature_tables[
        "reference_test_features"
    ]

    test_point_records = build_test_point_design(
        cell_records
    )

    if len(test_point_records) != 71:
        raise RuntimeError(
            f"Experimental design does not reconcile to 71 unique test points : {len(test_point_records)}"
        )

    treatment_correlation_records = (
        build_treatment_correlations(
            test_point_records
        )
    )

    equipment_by_cell = build_equipment_inventory(
        audit_result
    )

    approved_horizon_inventory = (
        build_approved_horizon_inventory(
            analysis_policy = analysis_policy
        )
    )

    (
        design_identifiability_records,
        _
    ) = build_design_identifiability(
        test_point_records = test_point_records,
        cell_records = cell_records,
        reference_records = reference_records,
        equipment_by_cell = equipment_by_cell,
        approved_horizon_inventory = (
            approved_horizon_inventory
        )
    )

    cell_horizon_support_records = (
        build_cell_horizon_ranges(
            cell_records = cell_records,
            reference_records = reference_records
        )
    )

    (
        atomic_region_records,
        atomic_membership_records
    ) = build_continuous_support_regions(
        cell_records = cell_records,
        cell_horizon_range_records = (
            cell_horizon_support_records
        ),
        coverage_rules = analysis_policy[
            "coverage_rules"
        ]
    )

    (
        continuous_support_region_records,
        continuous_support_membership_records
    ) = jointly_compact_support_regions(
        cell_horizon_range_records = (
            cell_horizon_support_records
        ),
        atomic_region_records = (
            atomic_region_records
        ),
        atomic_membership_records = (
            atomic_membership_records
        )
    )

    (
        interval_identifiability_records,
        analytical_scope_exclusion_records
    ) = build_continuous_support_identifiability(
        support_region_records = (
            continuous_support_region_records
        ),
        support_membership_records = (
            continuous_support_membership_records
        ),
        cell_records = cell_records,
        equipment_by_cell = equipment_by_cell
    )

    design_identifiability_records = sorted(
        design_identifiability_records,
        key = lambda record: (
            record["analysis_population"],
            record["aging_mode"],
            record["design_scope"],
            record["outcome"] or "",
            record["cohort"] or "",
            record["horizon_axis"] or "",
            record["specification"],
            record["record_type"],
            record["context_family"] or "",
            record["term_name"] or ""
        )
    )

    assigned_calendar_schedule_terms = [
        record
        for record in design_identifiability_records
        if (
            record["record_type"] == "term"
            and record["analysis_population"]
            == "assigned_cell_design"
            and record["aging_mode"] == "calendar_aging"
            and record["context_family"]
            == CONTEXT_FAMILY_NONE
            and record["term_name"].startswith(
                "reference_test_schedule="
            )
        )
    ]

    if not assigned_calendar_schedule_terms:
        raise RuntimeError("Assigned calendar design does not contain a reference_test_schedule treatment term.")

    stage_overlap_records = (
        build_stage_overlap(
            test_point_records
        )
    )

    nonzero_exact_stage_overlap_records = [
        record
        for record in stage_overlap_records
        if (
            record["record_type"] in {
                "exact_combination_summary",
                "exact_condition_protocol_summary"
            }
            and int(record.get("exact_overlap_count") or 0) > 0
        )
    ]

    if nonzero_exact_stage_overlap_records:
        raise RuntimeError(
            "Cross-stage exact treatment overlap is nonzero; the absent-replication status is not valid for this dataset : "
            f"{nonzero_exact_stage_overlap_records}"
        )

    validate_design_dimensions(
        cell_records = cell_records,
        condition_records = condition_records,
        reference_records = reference_records
    )

    treatment_support_records = (
        build_treatment_support(
            cell_records = cell_records,
            condition_records = condition_records,
            reference_records = reference_records,
            equipment_by_cell = equipment_by_cell
        )
    )

    treatment_combination_support_records = (
        build_treatment_combination_support(
            cell_records = cell_records,
            reference_records = reference_records,
            equipment_by_cell = equipment_by_cell
        )
    )

    observed_horizon_inventory = build_observed_horizon_inventory(
        cell_records = cell_records,
        reference_records = reference_records
    )

    horizon_support_records = (
        build_horizon_support(
            cell_records = cell_records,
            reference_records = reference_records,
            observed_horizon_inventory = observed_horizon_inventory,
            coverage_rules = analysis_policy[
                "coverage_rules"
            ]
        )
    )

    treatment_dependence_records = (
        build_treatment_dependence(
            cell_records = cell_records,
            reference_records = reference_records
        )
    )

    estimand_support_records = (
        apply_final_estimand_classifications(
            build_estimand_support(
                condition_records = condition_records,
                treatment_support_records = (
                    treatment_support_records
                ),
                horizon_support_records = (
                    horizon_support_records
                ),
                coverage_rules = analysis_policy[
                    "coverage_rules"
                ],
                analysis_policy = analysis_policy
            )
        )
    )

    invalid_sensitivity_only_records = [
        record
        for record in estimand_support_records
        if (
            record["final_support_classification"]
            == "sensitivity_only"
            and record[
                "underlying_direct_contrast_support_status"
            ]
            != "supported_with_restrictions"
        )
    ]

    if invalid_sensitivity_only_records:
        raise RuntimeError("Sensitivity-only classification contains a contrast without underlying analytical support.")

    analytical_scope_records = (
        build_analytical_scope_table(
            identifiability_records = (
                interval_identifiability_records
            )
        )
    )

    context_family_no_term_dropped_count = sum(
        1
        for record in interval_identifiability_records
        if (
            record["context_family"]
            != CONTEXT_FAMILY_NONE
            and record["context_term_count"] == 0
        )
    )

    if (
        len(analytical_scope_records)
        + context_family_no_term_dropped_count
        != len(interval_identifiability_records)
    ):
        raise RuntimeError(
            "Candidate interval terms were lost while building the analytical-scope handoff : "
            f"{len(interval_identifiability_records)} : {len(analytical_scope_records)} : "
            f"context_family_dropped={context_family_no_term_dropped_count}"
        )

    excluded_scope_decision_count = sum(
        int(
            record[
                "excluded_term_region_count"
            ]
        )
        for record in (
            analytical_scope_exclusion_records
        )
    )

    analytical_scope_reconciliation = {
        "candidate_term_region_count": len(
            analytical_scope_records
        ),
        "context_family_no_term_dropped_count": (
            context_family_no_term_dropped_count
        ),
        "excluded_term_region_count": (
            excluded_scope_decision_count
        ),
        "complete_term_region_count": (
            len(analytical_scope_records)
            + context_family_no_term_dropped_count
            + excluded_scope_decision_count
        ),
        "reconciliation_complete": True
    }

    identifiability_populations = sorted(
        {
            record["analysis_population"]
            for record in [
                *design_identifiability_records,
                *interval_identifiability_records
            ]
        }
    )

    del interval_identifiability_records

    validate_cohort_aligned_support_regions(
        continuous_support_region_records
    )

    permitted_primary_geometries = {
        (
            record["support_region_kind"],
            record["support_interval_minimum"],
            record["support_interval_maximum"],
            record["support_interval_lower_inclusive"],
            record["support_interval_upper_inclusive"],
            record["aging_mode"],
            record["design_scope"],
            record["sampling_design_scope"],
            record["outcome"],
            record["horizon_axis"],
            record["analysis_role"],
            record["analysis_population"],
            record["specification"],
            record["context_family"],
            record["term_name"],
            record["term_type"]
        )
        for record in analytical_scope_records
        if (
            record["cohort"] == "primary"
            and record["authorization_status"]
            == "permitted_for_model_evaluation_with_restrictions"
        )
    }

    leaked_sensitivity_only_records = [
        record
        for record in analytical_scope_records
        if (
            record["question_classification"]
            == "sensitivity_only"
            and (
                record["support_region_kind"],
                record["support_interval_minimum"],
                record["support_interval_maximum"],
                record["support_interval_lower_inclusive"],
                record["support_interval_upper_inclusive"],
                record["aging_mode"],
                record["design_scope"],
                record["sampling_design_scope"],
                record["outcome"],
                record["horizon_axis"],
                record["analysis_role"],
                record["analysis_population"],
                record["specification"],
                record["context_family"],
                record["term_name"],
                record["term_type"]
            )
            in permitted_primary_geometries
        )
    ]

    if leaked_sensitivity_only_records:
        raise RuntimeError("Sensitivity-only scope contains a record with a permitted primary counterpart.")

    invalid_sensitivity_scope_records = [
        record
        for record in analytical_scope_records
        if (
            record["question_classification"]
            == "sensitivity_only"
            and record["cohort"]
            != "sensitivity"
        )
    ]

    if invalid_sensitivity_scope_records:
        raise RuntimeError("Sensitivity-only analytical scope contains a non-sensitivity record.")

    if any(
        record["region_kind"]
        not in {
            "exact_point",
            "interval"
        }
        for record in (
            continuous_support_region_records
        )
    ):
        raise RuntimeError("Authoritative support inventory contains an unexpected region kind.")

    if any(
        (
            record["region_kind"]
            == "exact_point"
            and (
                record["lower_horizon"]
                != record["upper_horizon"]
                or record["lower_inclusive"]
                is not True
                or record["upper_inclusive"]
                is not True
            )
        )
        or (
            record["region_kind"]
            == "interval"
            and record["lower_horizon"]
            >= record["upper_horizon"]
        )
        for record in (
            continuous_support_region_records
        )
    ):
        raise RuntimeError("Authoritative support-region geometry is inconsistent.")

    invalid_equipment_authorizations = [
        record
        for record in analytical_scope_records
        if (
            context_family_is_equipment(
                record["context_family"]
            )
            and record["authorization_status"].startswith(
                "permitted"
            )
        )
    ]

    if invalid_equipment_authorizations:
        raise RuntimeError("Time-collapsed equipment context was incorrectly authorized as a longitudinal adjustment.")

    invalid_inferential_authorizations = [
        record
        for record in analytical_scope_records
        if record["inferential_claim_authorization"]
        != "not_authorized_by_design_audit"
    ]

    if invalid_inferential_authorizations:
        raise RuntimeError("Design audit incorrectly authorized an inferential claim.")

    if any(
        (
            record["authorization_status"].startswith(
                "permitted"
            )
            and not record[
                "support_region_identity"
            ]
        )
        for record in analytical_scope_records
    ):
        raise RuntimeError("Permitted analytical scope is missing a support-region identity.")

    if any(
        (
            record["authorization_status"].startswith(
                "permitted"
            )
            and record["design_scope"]
            == "pooled_stages"
        )
        for record in analytical_scope_records
    ):
        raise RuntimeError("Cross-stage pooled analysis was incorrectly authorized.")

    if any(
        record["analysis_population"]
        != "bracketing_range_cell_design"
        for record in analytical_scope_records
    ):
        raise RuntimeError("Analytical scope contains a non-interval-specific population.")

    if not analytical_scope_records:
        raise RuntimeError("Analytical-scope handoff is empty.")

    if any(
        not record["authorization_status"].startswith(
            "permitted"
        )
        for record in analytical_scope_records
    ):
        raise RuntimeError("Analytical-scope handoff contains a non-permitted record.")

    output_tables = {
        "treatment_support": (
            treatment_support_records
        ),
        "treatment_dependence": (
            treatment_dependence_records
        ),
        "estimand_support": (
            compact_estimand_support_records(
                estimand_support_records
            )
        ),
        "horizon_support": (
            compact_horizon_support_records(
                horizon_support_records
            )
        ),
        "analytical_scope": (
            analytical_scope_records
        ),
        "analytical_scope_exclusions": (
            analytical_scope_exclusion_records
        ),
        "cell_horizon_support": (
            cell_horizon_support_records
        ),
        "bracketing_support_regions": (
            continuous_support_region_records
        ),
        "bracketing_support_membership": (
            continuous_support_membership_records
        ),
        "baseline_balance": (
            build_baseline_balance_diagnostics(
                cell_records = cell_records,
                reference_records = reference_records
            )
        ),
        "treatment_combination_support": (
            treatment_combination_support_records
        ),
        "test_point_design": test_point_records,
        "treatment_correlation": (
            treatment_correlation_records
        ),
        "design_identifiability": (
            design_identifiability_records
        ),
        "stage_overlap": (
            stage_overlap_records
        )
    }

    support_status_counts = Counter(
        record["final_support_classification"]
        for record in estimand_support_records
    )

    design_findings = build_design_findings(
        estimand_records = estimand_support_records,
        identifiability_records = (
            design_identifiability_records
        ),
        correlation_records = (
            treatment_correlation_records
        ),
        stage_overlap_records = (
            stage_overlap_records
        )
    )

    supported_direct_calendar_records = (
        direct_supported_records(
            records = estimand_support_records,
            aging_mode = "calendar_aging"
        )
    )

    supported_direct_cycle_records = (
        direct_supported_records(
            records = estimand_support_records,
            aging_mode = "cycle_aging"
        )
    )

    direct_calendar_condition_pair_support_status = (
        "none"
        if not supported_direct_calendar_records
        else "present_with_restrictions"
    )

    direct_cycle_condition_pair_support_status = (
        "none"
        if not supported_direct_cycle_records
        else "present_with_restrictions"
    )

    report = {
        "audit_status": "completed",
        "dataset": feature_manifest["dataset"],
        "data_quality_contract_version": (
            feature_manifest[
                "data_quality_contract_version"
            ]
        ),
        "feature_manifest_sha256": (
            calculate_sha256(
                args.manifest_path
            )
        ),
        "audit_sha256": (
            feature_manifest[
                "audit_sha256"
            ]
        ),
        "identifiability_unit": (
            "population_specific_test_point_or_independent_cell"
        ),
        "identifiability_units": [
            "unique_test_point",
            "independent_cell"
        ],
        "identifiability_populations": (
            identifiability_populations
        ),
        "support_construction_status": (
            "atomic_boundaries_jointly_compacted_across_cohorts"
        ),
        "findings": design_findings,
        "context_identifiability_strategy": (
            "separate_context_family_diagnostics"
        ),
        "support_status_scope": (
            "analytical_design_support_not_causal_identification"
        ),
        "joint_equipment_fixed_effect_matrix_used_for_support": False,
        "full_matrix_rank_required_for_target_term_estimability": False,
        "direct_contrast_inventory_role": (
            "observed_condition_pair_support_inventory_without_causal_authorization"
        ),
        "experimental_design_method_status": (
            "test_point_design_methods_documented"
        ),
        "cell_to_condition_randomization_status": (
            "not_established_in_validated_inputs_or_primary_publication"
        ),
        "experimental_design_source": (
            "https://doi.org/10.1038/s41597-024-03859-z"
        ),
        "analysis_policy_sha256": (
            calculate_sha256(
                args.analysis_policy_path
            )
        ),
        "analysis_policy_version": (
            analysis_policy["policy_version"]
        ),
        "production_batch_identifier_status": (
            "unavailable_as_explicit_validated_field"
        ),
        "stage_batch_relationship_status": (
            "stage_2_used_a_new_battery_cell_batch_relative_to_stage_1"
        ),
        "stage_batch_relationship_source": (
            "https://doi.org/10.1038/s41597-024-03859-z"
        ),
        "stage_batch_separability_status": (
            "batch_change_is_aligned_with_stage_and_cannot_be_separately_identified"
        ),
        "publication_initial_cell_comparability_evidence": {
            "reported_result": (
                "all_initial_dva_comparisons_r_squared_greater_than_0_99"
            ),
            "publication_interpretation": (
                "cells_described_as_one_statistical_ensemble"
            ),
            "project_interpretation": (
                "supports_initial_electrochemical_comparability_but_does_not_establish_"
                "cross_stage_degradation_exchangeability"
            ),
            "source": (
                "https://doi.org/10.1038/s41597-024-03859-z"
            )
        },
        "horizon_selection_policy": {
            "support_mode": analysis_policy[
                "support_mode"
            ],
            "observed_horizon_inventory_source": (
                "positive_observed_horizons_without_using_outcome_magnitude_or_direction"
            ),
            "coverage_denominator": "assigned_cells",
            "primary_coverage_rule": (
                analysis_policy[
                    "coverage_rules"
                ][
                    "primary"
                ]
            ),
            "sensitivity_coverage_rule": (
                analysis_policy[
                    "coverage_rules"
                ][
                    "sensitivity"
                ]
            ),
            "approved_discrete_horizons": (
                analysis_policy[
                    "horizon_definitions"
                ]
            ),
            "bracketing_support_region_source": (
                "bracketing_support_regions.csv:"
                "per_condition_coverage_applied_to_"
                "bracketing_support_membership.csv"
            ),
            "extrapolation_permitted": False
        },
        "formal_handoff": {
            "permitted_analysis_inventory": (
                "analytical_scope.csv"
            ),
            "candidate_term_inventory": (
                "analytical_scope.csv"
            ),
            "excluded_term_region_summary": (
                "analytical_scope_exclusions.csv"
            ),
            "global_design_identifiability_inventory": (
                "design_identifiability.csv"
            ),
            "detailed_identifiability_table": (
                "design_identifiability.csv"
            ),
            "model_class_evaluation_status": (
                "deferred_until_model_specification_and_validation"
            ),
            "condition_pair_decisions": (
                "estimand_support.csv"
            ),
            "condition_level_longitudinal_support": (
                "horizon_support.csv"
            ),
            "required_validation_grouping": [
                "cell_identity"
            ],
            "cross_stage_pooling_status": (
                "not_authorized_by_design_audit"
            ),
            "cross_stage_partial_pooling_status": (
                "model_dependent_sensitivity_evaluation_only_under_explicit_exchangeability_and_"
                "transportability_assumptions"
            ),
            "cross_stage_treatment_effect_claim_status": (
                "not_identified_because_stage_and_batch_are_aligned_and_exact_treatment_overlap_is_absent"
            ),
            "baseline_balance_diagnostics": (
                "baseline_balance.csv"
            ),
            "baseline_covariate_contract": {
                "source_table": "cell_features.csv",
                "join_key": "cell_identity",
                "interpretation": (
                    "descriptive_only_model_adjustment_requires_explicit_model_specific_justification"
                ),
                "fields": {
                    "initial_cell_mass_grams": {
                        "unit": "gram",
                        "equipment_field": (
                            "initial_cell_mass_scale"
                        )
                    },
                    "initial_1khz_resistance_source_value": {
                        "unit": None,
                        "source_unit_label": "ohm",
                        "unit_status": (
                            "unresolved_source_label_conflicts_with_observed_scale_and_cell_specification"
                        ),
                        "equipment_field": (
                            "initial_1khz_resistance_meter"
                        )
                    },
                    "initial_voltage_volts": {
                        "unit": "volt",
                        "equipment_field": (
                            "initial_voltage_meter"
                        )
                    }
                }
            },
            "cross_stage_batch_restriction": (
                "stage_2_used_a_new_batch_and_stage_cannot_be_separated_from_batch"
            ),
            "causal_interpretation": (
                "not_established"
            ),
            "condition_pair_sensitivity_only_definition": (
                "underlying_supported_with_restrictions_available_only_in_sensitivity_cohort"
            ),
            "bracketing_support_regions": (
                "bracketing_support_regions.csv"
            ),
            "bracketing_support_cell_membership": (
                "bracketing_support_membership.csv"
            ),
            "cell_level_horizon_ranges": (
                "cell_horizon_support.csv"
            ),
            "authorization_basis": (
                "interval_specific_bracketing_support_using_only_cells_supporting_each_region"
            ),
            "observation_selection_contract": {
                "source_table": "reference_test_features.csv",
                "record_key": "source_identity",
                "independent_unit_key": "cell_identity",
                "condition_key": "condition_protocol_identity",
                "support_region_key": "support_region_identity",
                "support_membership_table": (
                    "bracketing_support_membership.csv"
                ),
                "outcome_definitions": OUTCOME_DEFINITIONS,
                "elapsed_time_requirements": {
                    "status_field": "elapsed_calendar_time_status",
                    "required_status": "completed",
                    "value_field": "elapsed_calendar_days",
                    "minimum_exclusive": 0
                },
                "fec_requirements": {
                    "value_field": (
                        "cumulative_full_equivalent_cycles_observed"
                    ),
                    "observation_complete_field": (
                        "cumulative_fec_observation_complete"
                    ),
                    "primary_eligibility_field": (
                        "cumulative_fec_primary_eligible"
                    ),
                    "sensitivity_eligibility_field": (
                        "cumulative_fec_sensitivity_eligible"
                    ),
                    "minimum_exclusive": 0
                },
                "source_observation_rule": (
                    "all_eligible_positive_horizon_observations_"
                    "from_support_region_member_cells"
                ),
                "support_interval_role": (
                    "prediction_and_estimand_evaluation_domain"
                ),
                "observations_outside_support_interval": (
                    "permitted_only_for_support_region_member_"
                    "cells_to_establish_within_cell_bracketing"
                ),
                "baseline_horizon_role": (
                    "excluded_from_support_determination_"
                    "and_primary_longitudinal_response"
                ),
                "model_specific_baseline_use": (
                    "requires_explicit_normalization_aware_"
                    "justification"
                )
            },
            "variable_units": {
                "relative_capacity": "fraction",
                "relative_dc_internal_resistance": "fraction",
                "ambient_temperature_celsius": "degree_Celsius",
                "maximum_state_of_charge_fraction": "fraction",
                "depth_of_discharge_fraction": "fraction",
                "charge_rate_c": "C_rate",
                "discharge_rate_c": "C_rate",
                "elapsed_calendar_days": "day",
                "cumulative_full_equivalent_cycles_observed": (
                    "full_equivalent_cycle"
                ),
                "mean_capacity_ah": "ampere_hour",
                "uncorrected_dc_internal_resistance_milliohms": (
                    "milliohm"
                ),
                "baseline_voltage_volts": "volt",
                "initial_cell_mass_grams": "gram",
                "initial_voltage_volts": "volt",
                "standardized_mean_difference": "dimensionless",
                "pearson_correlation": "dimensionless"
            },
            "calendar_whole_protocol_modeling_status": (
                "candidate_with_restrictions_using_gross_elapsed_calendar_days"
            ),
            "calendar_net_storage_exposure_status": (
                "unavailable_not_authorized"
            ),
            "calendar_claim_boundary": (
                analysis_policy[
                    "calendar_exposure_policy"
                ][
                    "claim_boundary"
                ]
            ),
            "cycle_model_evaluation_status": (
                "limited_to_stage_specific_structurally_estimable_terms_"
                "with_model_specific_validation"
            ),
            "direct_calendar_condition_pair_support_status": (
                direct_calendar_condition_pair_support_status
            ),
            "direct_calendar_condition_pair_supported_decision_count": (
                len(
                    supported_direct_calendar_records
                )
            ),
            "direct_calendar_condition_pair_supported_question_count": (
                direct_supported_question_count(
                    supported_direct_calendar_records
                )
            ),
            "direct_cycle_condition_pair_support_status": (
                direct_cycle_condition_pair_support_status
            ),
            "direct_cycle_condition_pair_supported_decision_count": (
                len(
                    supported_direct_cycle_records
                )
            ),
            "direct_cycle_condition_pair_supported_question_count": (
                direct_supported_question_count(
                    supported_direct_cycle_records
                )
            ),
            "interpolation_authorization_status": (
                "requires_model_specific_validation"
            ),
            "longitudinal_observation_policy": (
                analysis_policy[
                    "longitudinal_observation_policy"
                ]
            ),
            "treatment_evaluation_policy": (
                analysis_policy[
                    "treatment_evaluation_policy"
                ]
            ),
            "equipment_allocation_interpretation": {
                "support_role": (
                    "descriptive_restriction_not_condition_pair_support_veto"
                ),
                "adjustment_status": (
                    "not_authorized_from_aggregated_measurement_history"
                ),
                "causal_interpretation": (
                    "not_established"
                )
            }
        },
        "treatment_evaluation_domain": {
            "authorization": (
                "observed_condition_protocol_combinations_only"
            ),
            "source_table": (
                "treatment_combination_support.csv"
            ),
            "condition_key": (
                "condition_protocol_identity"
            ),
            "unobserved_combination_status": (
                "not_authorized_by_design_audit"
            ),
            "continuous_treatment_interpolation": (
                "requires_separate_model_specific_validation"
            )
        },
        "trajectory_estimand_status": (
            "defined_over_recorded_common_support_ranges"
        ),
        "discrete_comparison_horizon_status": (
            analysis_policy[
                "longitudinal_observation_policy"
            ][
                "discrete_horizon_status"
            ]
        ),
        "point_contrast_requirement": (
            analysis_policy[
                "longitudinal_observation_policy"
            ][
                "point_contrast_requirement"
            ]
        ),
        "analytical_scope_exclusions": (
            "analytical_scope_exclusions.csv"
        ),
        "analytical_scope_reconciliation": (
            analytical_scope_reconciliation
        ),
        "cross_stage_transportability_status": (
            "not_established_requires_model_dependent_joint_support_and_exchangeability_assessment"
        ),
        "independent_experimental_unit": "cell",
        "calendar_and_cycle_designs_separated": True,
        "outcome_availability_used_for_support_selection": True,
        "outcome_magnitude_or_direction_used_for_support_selection": False,
        "treatment_surface_rank_scope": (
            "standardized_treatment_terms_with_an_optional_stage_indicator"
        ),
        "longitudinal_model_matrix_status": (
            "final_outcome_model_matrix_must_be_validated_after_model_specification"
        ),
        "minimum_replicated_cell_count": (
            MINIMUM_REPLICATED_CELL_COUNT
        ),
        "summary": {
            "cell_count": len(
                cell_records
            ),
            "condition_protocol_count": len(
                condition_records
            ),
            "calendar_condition_protocol_count": sum(
                record["aging_mode"]
                == "calendar_aging"
                for record in condition_records
            ),
            "cycle_condition_protocol_count": sum(
                record["aging_mode"]
                == "cycle_aging"
                for record in condition_records
            ),
            "estimand_decision_count": len(
                estimand_support_records
            ),
            "treatment_combination_count": len(
                treatment_combination_support_records
            ),
            "calendar_treatment_combination_count": sum(
                record["aging_mode"]
                == "calendar_aging"
                for record in (
                    treatment_combination_support_records
                )
            ),
            "cycle_treatment_combination_count": sum(
                record["aging_mode"]
                == "cycle_aging"
                for record in (
                    treatment_combination_support_records
                )
            ),
            "support_status_counts": dict(
                sorted(
                    support_status_counts.items()
                )
            ),
            "test_point_count": len(
                test_point_records
            ),
            "distinct_condition_pair_count": len(
                {
                    (
                        record[
                            "first_condition_protocol_identity"
                        ],
                        record[
                            "second_condition_protocol_identity"
                        ]
                    )
                    for record in estimand_support_records
                }
            ),
            "distinct_outcome_axis_question_count": len(
                {
                    (
                        record["outcome"],
                        record["horizon_axis"]
                    )
                    for record in estimand_support_records
                }
            ),
            "final_classification_row_counts": dict(
                sorted(
                    support_status_counts.items()
                )
            )
        },
        "limitations": [
            (
                "Gross elapsed calendar days are not net aging exposure."
            ),
            (
                "Complete cumulative FEC uses the audited incident registry and observed measurement availability."
            ),
            (
                "Exact source equipment labels are preserved without undocumented normalization."
            ),
            (
                "Randomization and treatment-assignment mechanisms are not documented in the validated inputs or "
                "primary publication."
            ),
            (
                "Structural identifiability is reported separately for the assigned design and each realized "
                "outcome, cohort and positive-horizon design."
            ),
            (
                "A bracketing horizon does not authorize extrapolation, establish interpolation validity or "
                "predetermine a longitudinal model."
            ),
            (
                "Context identifiability is evaluated separately for laboratory and each validated equipment "
                "allocation family. The audit does not treat a saturated joint fixed-effect equipment matrix as "
                "evidence that every treatment is confounded."
            ),
            (
                "Unequal equipment allocation is preserved as a warning. Only disjoint validated allocation is "
                "treated as structural aliasing, while identity-conflicted equipment remains unresolved."
            ),
            (
                "Calendar reference-test schedule is included as a target treatment in cell-level rank diagnostics."
            ),
            (
                "A context-family matrix may be rank deficient while an individual target treatment remains "
                "structurally estimable. Term-level estimability, rather than full-matrix rank alone, controls the "
                "identifiability finding."
            ),
            (
                "The publication documents test-point design methods but does not establish cell-to-condition "
                "randomization in the validated project inputs."
            ),
            (
                "Sampling method records how test points were selected. It is preserved as design provenance "
                "but does not by itself represent an outcome-generating treatment or context."
            ),
            (
                "The minimum replicated-cell count identifies whether a condition is repeated. It is not an "
                "inferential power or adequacy threshold."
            ),
            (
                "Treatment-term estimability is evaluated separately within each supported horizon region. "
                "Final outcome-model fixed-effect, nonlinear-time and repeated-measurement structures require"
                " specification-level validation before fitting."
            ),
            (
                "Full rank in a pooled-stage treatment matrix does not establish cross-stage exchangeability or "
                "transportability when exact treatment combinations do not overlap."
            )
        ],
        "horizon_axis_units": {
            "elapsed_calendar_days": "day",
            "cumulative_full_equivalent_cycles_observed": (
                "full_equivalent_cycle"
            )
        }
    }

    published_report_path = (
        write_and_publish_design_artifacts(
            output_directory = (
                args.output_directory
            ),
            output_tables = output_tables,
            report_without_tables = (
                report
            )
        )
    )

    print(
        "Experimental-design audit finished."
    )
    print(
        f"Cell count : {len(cell_records)}"
    )
    print(
        f"Condition-protocol count : {len(condition_records)}"
    )
    print(
        f"Estimand decision count : {len(estimand_support_records)}"
    )
    print(
        f"Design report : {published_report_path}"
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        raise SystemExit(
            f"Experimental-design audit failed : {exc}"
        ) from exc
import argparse
import csv
import io
import json
import math
import re
import sys
import zipfile

from collections import Counter
from pathlib import Path, PurePosixPath

from google.cloud import storage

from scripts.validate_ingested_dataset import (
    ARTICLE_METADATA_OBJECT,
    EXPECTED_CELL_COUNT,
    build_experiment_metadata_summary,
    build_landed_object_name,
    download_text_object,
    find_archive_file_record,
    find_dataset_root_file,
    inspect_archive_file,
    parse_elapsed_seconds,
    parse_measurement_start_date,
    resolve_cached_archive_path,
    validate_experiment_metadata_summary
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONTRACT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "data_quality_contract.json"
)

DEFAULT_ARCHIVE_CACHE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "archives"
)

MAXIMUM_GUARDRAIL_EXAMPLES = 20


# Parsing the cloud source and contract being audited
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = (
            "Auditing experiment metadata and landed battery archives against data-quality contracts and incident rules."
        )
    )

    parser.add_argument(
        "--project-id",
        required = True
    )

    parser.add_argument(
        "--bucket-name",
        required = True
    )

    parser.add_argument(
        "--selection-mode",
        choices = [
            "specified",
            "complete"
        ],
        default = "specified",
        help = (
            "Auditing explicitly supplied archives or every cell archive in the pinned dataset."
        )
    )

    parser.add_argument(
        "--archive-object",
        action = "append",
        default = [],
        help = (
            "Cloud Storage object containing one battery archive. Repeat this option when selection mode is specified."
        )
    )

    parser.add_argument(
        "--output-path",
        type = Path,
        required = True,
        help = (
            "Local path receiving the machine-readable audit result."
        )
    )

    parser.add_argument(
        "--contract-path",
        type = Path,
        default = DEFAULT_CONTRACT_PATH
    )

    parser.add_argument(
        "--archive-cache-directory",
        type = Path,
        default = DEFAULT_ARCHIVE_CACHE_DIRECTORY,
        help = (
            "Directory retaining integrity-validated compressed archives for audit and feature extraction."
        )
    )

    return parser.parse_args()


# Rejecting duplicate keys while constructing JSON objects
def build_unique_json_object(object_pairs: list[tuple[str, object]]) -> dict:
    unique_object = {}

    for field_name, field_value in object_pairs:
        if field_name in unique_object:
            raise RuntimeError(f"Data-quality contract contains a duplicate JSON field : {field_name}")

        unique_object[
            field_name
        ] = field_value

    return unique_object


# Loading and validating the machine-readable quality contract
def load_quality_contract(contract_path: Path) -> dict:
    with contract_path.open(
        mode = "r",
        encoding = "utf-8"
    ) as contract_stream:
        quality_contract = json.load(
            contract_stream,
            object_pairs_hook = (
                build_unique_json_object
            )
        )

    required_sections = {
        "contract_version",
        "dataset",
        "disposition_priority",
        "experiment_contract",
        "measurement_contract",
        "feature_contract",
        "downstream_incident_rules",
        "incident_rules"
    }

    missing_sections = required_sections.difference(
        quality_contract
    )

    if missing_sections:
        raise RuntimeError(f"Data-quality contract is missing required sections : {sorted(missing_sections)}")

    disposition_priority = quality_contract[
        "disposition_priority"
    ]

    expected_dispositions = {
        "include",
        "include_with_warning",
        "manual_review",
        "exclude"
    }

    if set(disposition_priority) != expected_dispositions:
        raise RuntimeError("Data-quality contract contains invalid dispositions.")

    direct_incident_rules = quality_contract[
        "incident_rules"
    ]

    downstream_incident_rules = quality_contract[
        "downstream_incident_rules"
    ]

    direct_reason_codes = {
        incident_rule["reason_code"]
        for incident_rule in direct_incident_rules
    }

    all_reason_codes = [
        incident_rule["reason_code"]
        for incident_rule in [
            *direct_incident_rules,
            *downstream_incident_rules
        ]
    ]

    duplicate_reason_codes = sorted(
        reason_code
        for reason_code, reason_count in Counter(
            all_reason_codes
        ).items()
        if reason_count > 1
    )

    if duplicate_reason_codes:
        raise RuntimeError(f"Data-quality contract contains duplicate reason codes : {duplicate_reason_codes}")

    for incident_rule in direct_incident_rules:
        if (
            incident_rule["target_scope"]
            not in {"archive", "measurement"}
        ):
            raise RuntimeError(f"Incident rule contains an invalid target scope : {incident_rule['reason_code']}")

        if (
            incident_rule["disposition"]
            not in expected_dispositions
        ):
            raise RuntimeError(f"Incident rule contains an invalid disposition : {incident_rule['reason_code']}")

        incident_details = incident_rule.get(
            "incident_details"
        )

        if incident_details is not None:
            required_incident_detail_fields = {
                "cycling_performed",
                "missing_sequence_numbers"
            }

            if (
                set(incident_details)
                != required_incident_detail_fields
            ):
                raise RuntimeError(
                    f"Incident details contain invalid fields : {incident_rule['reason_code']}"
                )

            if not isinstance(
                incident_details[
                    "cycling_performed"
                ],
                bool
            ):
                raise RuntimeError(
                    f"Incident cycling-performed status must be Boolean : {incident_rule['reason_code']}"
                )

            missing_sequence_numbers = (
                incident_details[
                    "missing_sequence_numbers"
                ]
            )

            if (
                not missing_sequence_numbers
                or any(
                    not isinstance(
                        sequence_number,
                        int
                    )
                    or sequence_number <= 0
                    for sequence_number in (
                        missing_sequence_numbers
                    )
                )
            ):
                raise RuntimeError(
                    f"Incident contains invalid missing sequence numbers : {incident_rule['reason_code']}"
                )

            if (
                len(
                    missing_sequence_numbers
                )
                != len(
                    set(
                        missing_sequence_numbers
                    )
                )
            ):
                raise RuntimeError(
                    f"Incident contains duplicate missing sequence numbers : {incident_rule['reason_code']}"
                )

        re.compile(
            incident_rule["target_pattern"]
        )

    required_downstream_fields = {
        "reason_code",
        "originating_reason_code",
        "source",
        "target_scope",
        "target_pattern",
        "disposition",
        "outcome_scope",
        "affected_exposure_interval",
        "primary_cohort_eligible",
        "sensitivity_cohort_eligible",
        "incident_contexts",
        "description"
    }

    for downstream_incident_rule in downstream_incident_rules:
        missing_downstream_fields = (
            required_downstream_fields.difference(
                downstream_incident_rule
            )
        )

        if missing_downstream_fields:
            raise RuntimeError(
                "Downstream incident rule is missing required fields : "
                f"{downstream_incident_rule.get('reason_code')} : {sorted(missing_downstream_fields)}"
            )

        if (
            downstream_incident_rule[
                "originating_reason_code"
            ]
            not in direct_reason_codes
        ):
            raise RuntimeError(
                "Downstream incident rule references an unknown originating reason : "
                f"{downstream_incident_rule['reason_code']}"
            )

        if (
            downstream_incident_rule["target_scope"]
            != "measurement"
        ):
            raise RuntimeError(
                f"Downstream incident rule must target measurements : {downstream_incident_rule['reason_code']}"
            )

        if (
            downstream_incident_rule["disposition"]
            not in expected_dispositions
        ):
            raise RuntimeError(
                f"Downstream incident rule contains an invalid disposition : {downstream_incident_rule['reason_code']}"
            )

        for eligibility_field in (
            "primary_cohort_eligible",
            "sensitivity_cohort_eligible"
        ):
            if not isinstance(
                downstream_incident_rule[
                    eligibility_field
                ],
                bool
            ):
                raise RuntimeError(
                    "Downstream incident eligibility must be Boolean : "
                    f"{downstream_incident_rule['reason_code']} : {eligibility_field}"
                )

        incident_contexts = downstream_incident_rule[
            "incident_contexts"
        ]

        if not incident_contexts:
            raise RuntimeError(
                f"Downstream incident rule does not contain incident contexts : {downstream_incident_rule['reason_code']}"
            )

        for incident_context in incident_contexts:
            if set(
                incident_context
            ) != {
                "cell_identity",
                "incident_sequence"
            }:
                raise RuntimeError(
                    f"Downstream incident context contains invalid fields : {downstream_incident_rule['reason_code']}"
                )

            if (
                not isinstance(
                    incident_context[
                        "incident_sequence"
                    ],
                    int
                )
                or incident_context[
                    "incident_sequence"
                ] <= 0
            ):
                raise RuntimeError(
                    f"Downstream incident context contains an invalid sequence : {downstream_incident_rule['reason_code']}"
                )

        re.compile(
            downstream_incident_rule[
                "target_pattern"
            ]
        )

    return quality_contract


# Creating one standardized quality reason
def build_quality_reason(reason_code: str, disposition: str,
                         source: str, description: str,
                         details: dict | None = None
) -> dict:
    quality_reason = {
        "reason_code": reason_code,
        "disposition": disposition,
        "source": source,
        "description": description
    }

    if details:
        quality_reason["details"] = details

    return quality_reason


# Resolving the strictest disposition from all applicable reasons
def resolve_disposition(quality_reasons: list[dict], quality_contract: dict) -> str:
    disposition_priority = quality_contract[
        "disposition_priority"
    ]

    if not quality_reasons:
        return "include"

    return max(
        (
            quality_reason["disposition"]
            for quality_reason in quality_reasons
        ),
        key = disposition_priority.index
    )


# Matching direct and downstream incidents to one source identity
def match_incident_rules(target_scope: str, target_identity: str,
                         quality_contract: dict
) -> list[dict]:
    matching_reasons = []

    for incident_rule in quality_contract[
        "incident_rules"
    ]:
        if (
            incident_rule["target_scope"]
            != target_scope
        ):
            continue

        if re.fullmatch(
            incident_rule["target_pattern"],
            target_identity
        ):
            matching_reasons.append(
                build_quality_reason(
                    reason_code = incident_rule[
                        "reason_code"
                    ],
                    disposition = incident_rule[
                        "disposition"
                    ],
                    source = incident_rule[
                        "source"
                    ],
                    description = incident_rule[
                        "description"
                    ],
                    details = incident_rule.get(
                        "incident_details"
                    )
                )
            )

    if target_scope != "measurement":
        return matching_reasons

    for downstream_incident_rule in quality_contract[
        "downstream_incident_rules"
    ]:
        if re.fullmatch(
            downstream_incident_rule[
                "target_pattern"
            ],
            target_identity
        ) is None:
            continue

        matching_incident_context = next(
            (
                incident_context
                for incident_context in downstream_incident_rule[
                    "incident_contexts"
                ]
                if target_identity.startswith(
                    f"{incident_context['cell_identity']}/"
                )
            ),
            None
        )

        if matching_incident_context is None:
            raise RuntimeError(
                "Downstream incident rule matched outside its configured cell contexts : "
                f"{downstream_incident_rule['reason_code']} : {target_identity}"
            )

        matching_reasons.append(
            build_quality_reason(
                reason_code = downstream_incident_rule[
                    "reason_code"
                ],
                disposition = downstream_incident_rule[
                    "disposition"
                ],
                source = downstream_incident_rule[
                    "source"
                ],
                description = downstream_incident_rule[
                    "description"
                ],
                details = {
                    "originating_reason_code": (
                        downstream_incident_rule[
                            "originating_reason_code"
                        ]
                    ),
                    "cell_identity": (
                        matching_incident_context[
                            "cell_identity"
                        ]
                    ),
                    "incident_sequence": (
                        matching_incident_context[
                            "incident_sequence"
                        ]
                    ),
                    "affected_exposure_interval": (
                        downstream_incident_rule[
                            "affected_exposure_interval"
                        ]
                    ),
                    "outcome_scope": (
                        downstream_incident_rule[
                            "outcome_scope"
                        ]
                    ),
                    "primary_cohort_eligible": (
                        downstream_incident_rule[
                            "primary_cohort_eligible"
                        ]
                    ),
                    "sensitivity_cohort_eligible": (
                        downstream_incident_rule[
                            "sensitivity_cohort_eligible"
                        ]
                    )
                }
            )
        )

    return matching_reasons


# Parsing one finite numerical contract value
def parse_finite_number(value: str, field_name: str, record_identity: str) -> float:
    try:
        numerical_value = float(
            value
        )

    except ValueError as exc:
        raise RuntimeError(f"Invalid numerical value for {field_name} in {record_identity} : {value}") from exc

    if not math.isfinite(
        numerical_value
    ):
        raise RuntimeError(f"Non-finite numerical value for {field_name} in {record_identity} : {value}")

    return numerical_value


# Parsing one measurement cell that may be recorded as an empty string
def parse_audited_measurement_value(value: str, field_name: str,
                            source_identity: str, row_number: int
) -> float:
    normalized_value = value.strip()

    if not normalized_value:
        return math.nan

    try:
        numerical_value = float(
            normalized_value
        )

    except ValueError as exc:
        raise RuntimeError(
            f"Invalid numerical value for {field_name} in {source_identity} : row {row_number} : {value!r}"
        ) from exc

    return numerical_value


# Parsing documented and numeric elapsed-time representations for auditing
def parse_audited_run_time(value: str, source_identity: str,
                           row_number: int
) -> tuple[float, str]:
    normalized_value = value.strip()

    if ":" in normalized_value:
        try:
            elapsed_value = parse_elapsed_seconds(
                normalized_value
            )

        except ValueError as exc:
            raise RuntimeError(
                f"Measurement contains an invalid clock-formatted run_time value : {source_identity} : row {row_number} : {value!r}"
            ) from exc

        return (
            elapsed_value,
            "clock_elapsed_time"
        )

    try:
        numeric_elapsed_value = float(
            normalized_value
        )

    except ValueError as exc:
        raise RuntimeError(
            f"Measurement contains an unrecognized run_time value : {source_identity} : row {row_number} : {value!r}"
        ) from exc

    if (
        not math.isfinite(
            numeric_elapsed_value
        )
        or numeric_elapsed_value < 0
    ):
        raise RuntimeError(
            f"Measurement contains an invalid numeric run_time value : {source_identity} : row {row_number} : {value!r}"
        )

    return (
        numeric_elapsed_value,
        "numeric_elapsed_value"
    )


# Parsing integer and integral-decimal step-type representations
def parse_audited_step_type(value: str, source_identity: str,
                            row_number: int
) -> tuple[str, str]:
    normalized_value = value.strip()

    if (
        normalized_value
        and normalized_value.isascii()
        and normalized_value.isdecimal()
    ):
        canonical_step_type = str(
            int(
                normalized_value
            )
        )

        return (
            canonical_step_type,
            (
                "integer_text"
                if normalized_value == canonical_step_type
                else "integral_numeric_text"
            )
        )

    try:
        numerical_step_type = float(
            normalized_value
        )

    except ValueError as exc:
        raise RuntimeError(
            f"Measurement contains a non-numerical step type : {source_identity} : row {row_number} : {value!r}"
        ) from exc

    if (
        not math.isfinite(
            numerical_step_type
        )
        or numerical_step_type < 0
        or not numerical_step_type.is_integer()
    ):
        raise RuntimeError(
            f"Measurement contains a non-integral step type : {source_identity} : row {row_number} : {value!r}"
        )

    return (
        str(
            int(
                numerical_step_type
            )
        ),
        "integral_numeric_text"
    )


# Validating experiment metadata against treatment contracts
def audit_experiment_metadata(experiment_metadata_content: str, quality_contract: dict) -> dict:
    metadata_summary = (
        build_experiment_metadata_summary(
            experiment_metadata_content = (
                experiment_metadata_content
            )
        )
    )

    validate_experiment_metadata_summary(
        metadata_summary = metadata_summary
    )

    metadata_reader = csv.DictReader(
        io.StringIO(
            experiment_metadata_content
        )
    )

    experiment_contract = quality_contract[
        "experiment_contract"
    ]

    contract_violations = []

    for row_number, experiment_row in enumerate(
        metadata_reader,
        start = 2
    ):
        record_identity = (
            f"metadata row {row_number}"
        )

        stage = experiment_row["stage"]
        aging_type = experiment_row["type"]
        laboratory = experiment_row["lab"]

        if (
            stage
            not in experiment_contract[
                "allowed_stages"
            ]
        ):
            contract_violations.append(
                f"{record_identity} has invalid stage : {stage}"
            )

        if (
            aging_type
            not in experiment_contract[
                "allowed_aging_types"
            ]
        ):
            contract_violations.append(
                f"{record_identity} has invalid aging type : {aging_type}"
            )

        if (
            laboratory
            not in experiment_contract[
                "allowed_laboratories"
            ]
        ):
            contract_violations.append(
                f"{record_identity} has invalid laboratory : {laboratory}"
            )

        ambient_temperature_celsius = (
            parse_finite_number(
                value = experiment_row[
                    "amb_temp_tp"
                ],
                field_name = "amb_temp_tp",
                record_identity = record_identity
            )
        )

        maximum_state_of_charge_fraction = (
            parse_finite_number(
                value = experiment_row[
                    "soc_max_tp"
                ],
                field_name = "soc_max_tp",
                record_identity = record_identity
            )
        )

        depth_of_discharge_fraction = (
            parse_finite_number(
                value = experiment_row[
                    "dod_tp"
                ],
                field_name = "dod_tp",
                record_identity = record_identity
            )
        )

        charge_rate_c = parse_finite_number(
            value = experiment_row["c_ch_tp"],
            field_name = "c_ch_tp",
            record_identity = record_identity
        )

        discharge_rate_c = parse_finite_number(
            value = experiment_row["c_dch_tp"],
            field_name = "c_dch_tp",
            record_identity = record_identity
        )

        if (
            ambient_temperature_celsius
            not in experiment_contract[
                "allowed_ambient_temperatures_celsius"
            ]
        ):
            contract_violations.append(
                f"{record_identity} has unsupported ambient temperature : {ambient_temperature_celsius}"
            )

        bounded_values = [
            (
                "soc_max_tp",
                maximum_state_of_charge_fraction,
                experiment_contract[
                    "minimum_state_of_charge_fraction"
                ],
                experiment_contract[
                    "maximum_state_of_charge_fraction"
                ]
            ),
            (
                "dod_tp",
                depth_of_discharge_fraction,
                experiment_contract[
                    "minimum_depth_of_discharge_fraction"
                ],
                experiment_contract[
                    "maximum_depth_of_discharge_fraction"
                ]
            ),
            (
                "c_ch_tp",
                charge_rate_c,
                experiment_contract[
                    "minimum_charge_rate_c"
                ],
                experiment_contract[
                    "maximum_charge_rate_c"
                ]
            ),
            (
                "c_dch_tp",
                discharge_rate_c,
                experiment_contract[
                    "minimum_discharge_rate_c"
                ],
                experiment_contract[
                    "maximum_discharge_rate_c"
                ]
            )
        ]

        for (
            field_name,
            numerical_value,
            minimum_value,
            maximum_value
        ) in bounded_values:
            if not (
                minimum_value
                <= numerical_value
                <= maximum_value
            ):
                contract_violations.append(
                    f"{record_identity} has out-of-range {field_name} : {numerical_value}"
                )

        if aging_type == "k":
            if any(
                numerical_value != 0
                for numerical_value in (
                    depth_of_discharge_fraction,
                    charge_rate_c,
                    discharge_rate_c
                )
            ):
                contract_violations.append(
                    f"{record_identity} has cycling treatments for a calendar-aging cell."
                )

        if aging_type == "z":
            minimum_cycle_state_of_charge = (
                maximum_state_of_charge_fraction
                - depth_of_discharge_fraction
            )

            if (
                minimum_cycle_state_of_charge
                < experiment_contract[
                    "cycle_minimum_state_of_charge_fraction"
                ]
                - 1e-9
                or maximum_state_of_charge_fraction
                > experiment_contract[
                    "cycle_maximum_state_of_charge_fraction"
                ]
                + 1e-9
            ):
                contract_violations.append(
                    f"{record_identity} has a cycling SOC window outside the supported range."
                )

        if not experiment_row["sampling"].strip():
            contract_violations.append(
                f"{record_identity} has an empty sampling method."
            )

    if contract_violations:
        raise RuntimeError(f"Experiment metadata violates its data contract : {contract_violations[:20]}")

    return {
        **metadata_summary,
        "contract_violation_count": 0
    }


# Parsing documented companion-metadata fields
def parse_companion_metadata(metadata_content: str, source_identity: str) -> dict:
    metadata_fields = {}

    metadata_field_aliases = {
        "Measurement deviceused": (
            "Measurement device used"
        )
    }

    for metadata_line in metadata_content.splitlines():
        field_name, separator, field_value = (
            metadata_line.partition(":")
        )

        if not separator:
            continue

        observed_field_name = field_name.strip()

        # Normalizing the documented source spelling variation
        normalized_field_name = (
            metadata_field_aliases.get(
                observed_field_name,
                observed_field_name
            )
        )

        normalized_field_value = field_value.strip()

        if normalized_field_name in metadata_fields:
            raise RuntimeError(f"Companion metadata contains a duplicate field : {source_identity} : {normalized_field_name}")

        metadata_fields[
            normalized_field_name
        ] = normalized_field_value

    required_fields = {
        "Laboratory identifier",
        "Internal cell serial",
        "Testpoint",
        "Measurement start date",
        "Measurement device used",
        "Test channel",
        "Climate chamber used",
        "Climate chamber temperature setpoint",
        "Position within climate chamber"
    }

    missing_fields = required_fields.difference(
        metadata_fields
    )

    if missing_fields:
        raise RuntimeError(f"Companion metadata is missing required fields : {source_identity} : {sorted(missing_fields)}")

    empty_fields = sorted(
        field_name
        for field_name in required_fields
        if not metadata_fields[field_name]
    )

    if empty_fields:
        raise RuntimeError(f"Companion metadata contains empty required fields : {source_identity} : {empty_fields}")

    return metadata_fields


# Parsing the type, test point and optional cell recorded in a companion label
def parse_companion_test_point_label(test_point_label: str) -> dict | None:
    normalized_label = test_point_label.strip()

    compact_label_match = re.fullmatch(
        r"(?P<type>[kz])"
        r"(?P<tp>\d{2})"
        r"(?:noCU)?",
        normalized_label
    )

    if compact_label_match is not None:
        return {
            "type": compact_label_match.group(
                "type"
            ),
            "tp": int(
                compact_label_match.group(
                    "tp"
                )
            ),
            "cell": None
        }

    extended_label_match = re.fullmatch(
        r"TP_(?P<type>[kz])"
        r"(?P<tp>\d{2})_"
        r"(?P<cell>\d{2})"
        r"_MKII",
        normalized_label
    )

    if extended_label_match is not None:
        return {
            "type": extended_label_match.group(
                "type"
            ),
            "tp": int(
                extended_label_match.group(
                    "tp"
                )
            ),
            "cell": int(
                extended_label_match.group(
                    "cell"
                )
            )
        }

    return None


# Comparing internal-cell serials while accepting numeric zero padding
def internal_cell_serials_match(expected_serial: str, observed_serial: str) -> bool:
    normalized_expected_serial = (
        expected_serial.strip()
    )

    normalized_observed_serial = (
        observed_serial.strip()
    )

    if (
        normalized_expected_serial
        == normalized_observed_serial
    ):
        return True

    numeric_serial_pattern = r"\d+"

    if (
        re.fullmatch(
            numeric_serial_pattern,
            normalized_expected_serial
        )
        is not None
        and re.fullmatch(
            numeric_serial_pattern,
            normalized_observed_serial
        )
        is not None
    ):
        return (
            int(
                normalized_expected_serial
            )
            == int(
                normalized_observed_serial
            )
        )

    return False


# Determining the expected chamber setpoint for one measurement
def determine_expected_measurement_temperature(measurement_name: str, experiment_record: dict) -> float:
    measurement_file_name = PurePosixPath(
        measurement_name
    ).name

    temperature_match = re.search(
        r"_(?:ET|AT)_T(-?\d+(?:\.\d+)?)\.csv$",
        measurement_file_name
    )

    if temperature_match is not None:
        return float(
            temperature_match.group(1)
        )

    if (
        measurement_file_name.endswith("_CU.csv")
        or measurement_file_name.endswith("_exCU.csv")
    ):
        return 23.0

    if measurement_file_name.endswith("_ZYK.csv"):
        return float(
            experiment_record["amb_temp_tp"]
        )

    raise RuntimeError(f"Cannot determine the expected measurement temperature : {measurement_name}")


# Validating one companion metadata record against its cell identity
def audit_companion_metadata(metadata_content: str, measurement_name: str,
                             source_identity: str, experiment_record: dict
) -> dict:
    metadata_fields = parse_companion_metadata(
        metadata_content = metadata_content,
        source_identity = source_identity
    )

    expected_values = {
        "Laboratory identifier": (
            experiment_record["lab"]
        )
    }

    mismatched_fields = {
        field_name: {
            "expected": expected_value,
            "observed": metadata_fields[field_name]
        }
        for field_name, expected_value in (
            expected_values.items()
        )
        if (
            metadata_fields[field_name]
            != expected_value
        )
    }

    # Comparing internal-cell serials while preserving both source values
    expected_internal_cell_serial = (
        experiment_record[
            "serial_internal"
        ]
    )

    observed_internal_cell_serial = (
        metadata_fields[
            "Internal cell serial"
        ]
    )

    if not internal_cell_serials_match(
        expected_serial = (
            expected_internal_cell_serial
        ),
        observed_serial = (
            observed_internal_cell_serial
        )
    ):
        mismatched_fields[
            "Internal cell serial"
        ] = {
            "expected": (
                expected_internal_cell_serial
            ),
            "observed": (
                observed_internal_cell_serial
            )
        }

    # Comparing the type, test point and optional cell in the companion label
    parsed_test_point_label = (
        parse_companion_test_point_label(
            metadata_fields[
                "Testpoint"
            ]
        )
    )

    expected_test_point_fields = {
        "type": experiment_record[
            "type"
        ],
        "tp": int(
            experiment_record[
                "tp"
            ]
        ),
        "cell": int(
            experiment_record[
                "cell"
            ]
        )
    }

    test_point_matches = (
        parsed_test_point_label is not None
        and parsed_test_point_label[
            "type"
        ] == expected_test_point_fields[
            "type"
        ]
        and parsed_test_point_label[
            "tp"
        ] == expected_test_point_fields[
            "tp"
        ]
        and (
            parsed_test_point_label[
                "cell"
            ] is None
            or parsed_test_point_label[
                "cell"
            ] == expected_test_point_fields[
                "cell"
            ]
        )
    )

    if not test_point_matches:
        mismatched_fields[
            "Testpoint"
        ] = {
            "expected": (
                expected_test_point_fields
            ),
            "observed": metadata_fields[
                "Testpoint"
            ],
            "parsed_observed": (
                parsed_test_point_label
            )
        }

    quality_reasons = []

    if mismatched_fields:
        quality_reasons.append(
            build_quality_reason(
                reason_code = (
                    "COMPANION_METADATA_IDENTITY_MISMATCH"
                ),
                disposition = "manual_review",
                source = "Automated data-quality audit",
                description = (
                    "Companion metadata conflicts with the published experiment metadata."
                ),
                details = {
                    "mismatched_fields": mismatched_fields
                }
            )
        )

    raw_measurement_start_date = metadata_fields[
        "Measurement start date"
    ]

    try:
        measurement_start_date = (
            parse_measurement_start_date(
                metadata_content
            )
        )

    except ValueError:
        measurement_start_date = None

        quality_reasons.append(
            build_quality_reason(
                reason_code = (
                    "COMPANION_METADATA_START_DATE_INVALID"
                ),
                disposition = "manual_review",
                source = "Automated data-quality audit",
                description = (
                    "The companion metadata contains an invalid measurement start date. No replacement date "
                    "has been inferred."
                ),
                details = {
                    "raw_measurement_start_date": (
                        raw_measurement_start_date
                    )
                }
            )
        )

    setpoint_match = re.fullmatch(
        (
            r"\s*(?:"
            r"\[\s*(?P<bracketed_temperature>-?\d+(?:\.\d+)?)\s*\]"
            r"|"
            r"(?P<plain_temperature>-?\d+(?:\.\d+)?)"
            r")\s*(?:°\s*C|C)?\s*"
        ),
        metadata_fields[
            "Climate chamber temperature setpoint"
        ]
    )

    raw_temperature_setpoint = metadata_fields[
        "Climate chamber temperature setpoint"
    ]

    observed_temperature_celsius = None

    if setpoint_match is None:
        quality_reasons.append(
            build_quality_reason(
                reason_code = (
                    "COMPANION_METADATA_TEMPERATURE_UNPARSABLE"
                ),
                disposition = "manual_review",
                source = "Automated data-quality audit",
                description = (
                    "The companion metadata chamber setpoint does not use a recognized temperature representation."
                ),
                details = {
                    "raw_temperature_setpoint": (
                        raw_temperature_setpoint
                    )
                }
            )
        )

    else:
        observed_temperature_value = (
            setpoint_match.group(
                "bracketed_temperature"
            )
            or setpoint_match.group(
                "plain_temperature"
            )
        )

        observed_temperature_celsius = float(
            observed_temperature_value
        )

    expected_temperature_celsius = (
        determine_expected_measurement_temperature(
            measurement_name = measurement_name,
            experiment_record = experiment_record
        )
    )

    if (
        observed_temperature_celsius is not None
        and abs(
            observed_temperature_celsius
            - expected_temperature_celsius
        )
        > 0.1
    ):
        quality_reasons.append(
            build_quality_reason(
                reason_code = (
                    "COMPANION_METADATA_TEMPERATURE_MISMATCH"
                ),
                disposition = "manual_review",
                source = "Automated data-quality audit",
                description = (
                    "Companion metadata chamber setpoint conflicts with the measurement protocol."
                ),
                details = {
                    "expected_temperature_celsius": (
                        expected_temperature_celsius
                    ),
                    "observed_temperature_celsius": (
                        observed_temperature_celsius
                    )
                }
            )
        )

    return {
        "laboratory": metadata_fields[
            "Laboratory identifier"
        ],
        "internal_cell_serial": metadata_fields[
            "Internal cell serial"
        ],
        "test_point": metadata_fields["Testpoint"],
        "parsed_test_point": (
            parsed_test_point_label
        ),
        "measurement_start_date": measurement_start_date,
        "measurement_device": metadata_fields[
            "Measurement device used"
        ],
        "test_channel": metadata_fields["Test channel"],
        "climate_chamber": metadata_fields[
            "Climate chamber used"
        ],
        "climate_chamber_temperature_setpoint_celsius": (
            observed_temperature_celsius
        ),
        "raw_measurement_start_date": (
            raw_measurement_start_date
        ),
        "raw_climate_chamber_temperature_setpoint": (
            raw_temperature_setpoint
        ),
        "climate_chamber_position": metadata_fields[
            "Position within climate chamber"
        ],
        "quality_reasons": quality_reasons
    }


# Auditing measurement values and timestamp behavior
def audit_measurement_stream(measurement_stream: io.TextIOBase, source_identity: str,
                             quality_contract: dict
) -> dict:
    measurement_reader = csv.reader(
        measurement_stream
    )

    measurement_contract = quality_contract[
        "measurement_contract"
    ]

    observed_columns = next(
        measurement_reader,
        []
    )

    expected_columns = measurement_contract[
        "columns"
    ]

    source_row_index_column = measurement_contract[
        "optional_leading_source_row_index_column"
    ]

    elapsed_seconds_column = measurement_contract[
        "optional_trailing_elapsed_seconds_column"
    ]

    source_row_index_present = (
        observed_columns
        == [
            source_row_index_column,
            *expected_columns
        ]
    )

    elapsed_seconds_column_present = (
        observed_columns
        == [
            *expected_columns,
            elapsed_seconds_column
        ]
    )

    canonical_columns_present = (
        observed_columns == expected_columns
    )

    if not (
        canonical_columns_present
        or source_row_index_present
        or elapsed_seconds_column_present
    ):
        raise RuntimeError(
            f"Measurement columns do not match the data contract : {source_identity} : {observed_columns}"
        )

    column_positions = {
        column_name: column_position
        for column_position, column_name in enumerate(
            observed_columns
        )
    }

    expected_row_length = len(observed_columns)

    run_time_position = column_positions["run_time"]
    voltage_position = column_positions["c_vol"]
    current_position = column_positions["c_cur"]
    surface_temperature_position = column_positions["c_surf_temp"]
    ambient_temperature_position = column_positions["amb_temp"]
    step_type_position = column_positions["step_type"]

    source_row_index_position = (
        column_positions[source_row_index_column]
        if source_row_index_present
        else None
    )

    elapsed_seconds_position = (
        column_positions[elapsed_seconds_column]
        if elapsed_seconds_column_present
        else None
    )

    minimum_voltage_volts = measurement_contract["minimum_voltage_volts"]
    maximum_voltage_volts = measurement_contract["maximum_voltage_volts"]
    minimum_current_amperes = measurement_contract["minimum_current_amperes"]
    maximum_current_amperes = measurement_contract["maximum_current_amperes"]
    minimum_surface_temperature_celsius = measurement_contract["minimum_surface_temperature_celsius"]
    maximum_surface_temperature_celsius = measurement_contract["maximum_surface_temperature_celsius"]

    allowed_elapsed_seconds_difference = (
        measurement_contract["elapsed_seconds_tolerance_seconds"]
        + measurement_contract["elapsed_seconds_floating_point_margin_seconds"]
    )

    direction_tolerance_amperes = measurement_contract["cycling_direction_tolerance_amperes"]

    cycling_measurement = source_identity.endswith("_ZYK.csv")

    is_finite = math.isfinite

    row_count = 0
    ambient_temperature_missing_count = 0
    surface_temperature_missing_count = 0
    out_of_range_field_counts = Counter()
    out_of_range_examples = []
    cycling_direction_mismatch_count = 0
    timestamp_reversal_count = 0
    timestamp_reversal_examples = []
    first_elapsed_seconds = None
    last_elapsed_seconds = None
    first_source_row_index = None
    last_source_row_index = None
    source_row_index_nonincreasing_count = 0
    source_row_index_discontinuity_examples = []
    maximum_elapsed_seconds_difference = None
    run_time_representations = set()
    step_type_representations = set()

    # Recording bounded examples for every guardrail violation
    def record_guardrail_violation(field_name: str, numerical_value: float,
                                   row_number: int, minimum_value: float, maximum_value: float
    ) -> None:
        out_of_range_field_counts[
            field_name
        ] += 1

        if (
            len(out_of_range_examples)
            < MAXIMUM_GUARDRAIL_EXAMPLES
        ):
            out_of_range_examples.append(
                {
                    "field_name": field_name,
                    "row_number": row_number,
                    "observed_value": numerical_value,
                    "minimum_value": minimum_value,
                    "maximum_value": maximum_value
                }
            )

    row_number = 1

    for measurement_row in measurement_reader:
        if not measurement_row:
            continue

        row_number += 1

        if len(measurement_row) > expected_row_length:
            raise RuntimeError(
                f"Measurement contains a row with extra columns : {source_identity} : {row_number}"
            )

        if len(measurement_row) < expected_row_length:
            raise RuntimeError(
                f"Measurement contains a row with missing columns : {source_identity} : {row_number}"
            )

        if source_row_index_present:
            source_row_index_value = measurement_row[
                source_row_index_position
            ].strip()

            try:
                source_row_index = int(
                    source_row_index_value
                )

            except ValueError as exc:
                raise RuntimeError(
                    "Measurement contains a non-integer source row index : "
                    f"{source_identity} : {row_number} : {source_row_index_value!r}"
                ) from exc

            if (
                last_source_row_index is not None
                and source_row_index
                <= last_source_row_index
            ):
                source_row_index_nonincreasing_count += 1

                if (
                    len(
                        source_row_index_discontinuity_examples
                    )
                    < MAXIMUM_GUARDRAIL_EXAMPLES
                ):
                    source_row_index_discontinuity_examples.append(
                        {
                            "row_number": row_number,
                            "previous_source_row_index": (
                                last_source_row_index
                            ),
                            "current_source_row_index": (
                                source_row_index
                            )
                        }
                    )

            if first_source_row_index is None:
                first_source_row_index = (
                    source_row_index
                )

            last_source_row_index = source_row_index

        (
            elapsed_seconds,
            run_time_representation
        ) = parse_audited_run_time(
            value = measurement_row[
                run_time_position
            ],
            source_identity = source_identity,
            row_number = row_number
        )

        run_time_representations.add(
            run_time_representation
        )

        if elapsed_seconds_column_present:
            observed_elapsed_seconds = (
                parse_audited_measurement_value(
                    value = measurement_row[
                        elapsed_seconds_position
                    ],
                    field_name = (
                        elapsed_seconds_column
                    ),
                    source_identity = source_identity,
                    row_number = row_number
                )
            )

            if not math.isfinite(
                observed_elapsed_seconds
            ):
                raise RuntimeError(
                    f"Measurement contains a non-finite derived elapsed-time value : {source_identity} : {row_number}"
                )

            elapsed_seconds_difference = abs(
                observed_elapsed_seconds
                - elapsed_seconds
            )

            if (
                elapsed_seconds_difference
                > allowed_elapsed_seconds_difference
            ):
                raise RuntimeError(
                    "Measurement derived elapsed time conflicts with run_time : "
                    f"{source_identity} : {row_number} : {elapsed_seconds_difference}"
                )

            if (
                maximum_elapsed_seconds_difference
                is None
            ):
                maximum_elapsed_seconds_difference = (
                    elapsed_seconds_difference
                )

            else:
                maximum_elapsed_seconds_difference = max(
                    maximum_elapsed_seconds_difference,
                    elapsed_seconds_difference
                )

        if (
            last_elapsed_seconds is not None
            and elapsed_seconds < last_elapsed_seconds
        ):
            timestamp_reversal_count += 1

            if (
                len(timestamp_reversal_examples)
                < MAXIMUM_GUARDRAIL_EXAMPLES
            ):
                timestamp_reversal_examples.append(
                    {
                        "row_number": row_number,
                        "previous_elapsed_seconds": (
                            last_elapsed_seconds
                        ),
                        "current_elapsed_seconds": (
                            elapsed_seconds
                        )
                    }
                )

        if first_elapsed_seconds is None:
            first_elapsed_seconds = elapsed_seconds

        last_elapsed_seconds = elapsed_seconds

        voltage_volts = parse_audited_measurement_value(
            value = measurement_row[voltage_position],
            field_name = "c_vol",
            source_identity = source_identity,
            row_number = row_number
        )
        current_amperes = parse_audited_measurement_value(
            value = measurement_row[current_position],
            field_name = "c_cur",
            source_identity = source_identity,
            row_number = row_number
        )
        surface_temperature_celsius = parse_audited_measurement_value(
            value = measurement_row[surface_temperature_position],
            field_name = "c_surf_temp",
            source_identity = source_identity,
            row_number = row_number
        )
        ambient_temperature_celsius = parse_audited_measurement_value(
            value = measurement_row[ambient_temperature_position],
            field_name = "amb_temp",
            source_identity = source_identity,
            row_number = row_number
        )

        if (
            not math.isfinite(voltage_volts)
            or not math.isfinite(current_amperes)
        ):
            raise RuntimeError(f"Measurement contains a non-finite voltage or current : {source_identity} : {row_number}")

        if not math.isfinite(
            surface_temperature_celsius
        ):
            surface_temperature_missing_count += 1

        if not math.isfinite(
            ambient_temperature_celsius
        ):
            ambient_temperature_missing_count += 1

        if not (
            minimum_voltage_volts
            <= voltage_volts
            <= maximum_voltage_volts
        ):
            record_guardrail_violation(
                field_name = "c_vol",
                numerical_value = voltage_volts,
                row_number = row_number,
                minimum_value = minimum_voltage_volts,
                maximum_value = maximum_voltage_volts
            )

        if not (
            minimum_current_amperes
            <= current_amperes
            <= maximum_current_amperes
        ):
            record_guardrail_violation(
                field_name = "c_cur",
                numerical_value = current_amperes,
                row_number = row_number,
                minimum_value = minimum_current_amperes,
                maximum_value = maximum_current_amperes
            )

        if (
            is_finite(
                surface_temperature_celsius
            )
            and not (
                minimum_surface_temperature_celsius
                <= surface_temperature_celsius
                <= maximum_surface_temperature_celsius
            )
        ):
            record_guardrail_violation(
                field_name = "c_surf_temp",
                numerical_value = surface_temperature_celsius,
                row_number = row_number,
                minimum_value = minimum_surface_temperature_celsius,
                maximum_value = maximum_surface_temperature_celsius
            )

        (
            step_type,
            step_type_representation
        ) = parse_audited_step_type(
            value = measurement_row[
                step_type_position
            ],
            source_identity = source_identity,
            row_number = row_number
        )

        step_type_representations.add(
            step_type_representation
        )

        if (
            cycling_measurement
            and (
                (
                    step_type == "41"
                    and current_amperes
                    < -direction_tolerance_amperes
                )
                or (
                    step_type == "42"
                    and current_amperes
                    > direction_tolerance_amperes
                )
            )
        ):
            cycling_direction_mismatch_count += 1

        row_count += 1

    if row_count == 0:
        quality_reasons = match_incident_rules(
            target_scope = "measurement",
            target_identity = source_identity,
            quality_contract = quality_contract
        )

        quality_reasons.append(
            build_quality_reason(
                reason_code = (
                    "MEASUREMENT_CONTAINS_NO_DATA_ROWS"
                ),
                disposition = "manual_review",
                source = "Automated data-quality audit",
                description = (
                    "The measurement file contains a header row and no data rows."
                )
            )
        )

        return {
            "source_identity": source_identity,
            "row_count": 0,
            "source_row_index_present": (
                source_row_index_present
            ),
            "source_row_index_nonincreasing_count": (
                source_row_index_nonincreasing_count
            ),
            "source_row_index_discontinuity_examples": (
                source_row_index_discontinuity_examples
            ),
            "first_source_row_index": (
                first_source_row_index
            ),
            "last_source_row_index": (
                last_source_row_index
            ),
            "first_elapsed_seconds": None,
            "last_elapsed_seconds": None,
            "elapsed_seconds_column_present": (
                elapsed_seconds_column_present
            ),
            "maximum_elapsed_seconds_difference": (
                maximum_elapsed_seconds_difference
            ),
            "timestamp_reversal_count": 0,
            "timestamp_reversal_examples": [],
            "ambient_temperature_missing_count": 0,
            "surface_temperature_missing_count": 0,
            "out_of_range_value_count": 0,
            "out_of_range_field_counts": {},
            "out_of_range_examples": [],
            "cycling_direction_mismatch_count": 0,
            "run_time_representations": sorted(
                run_time_representations
            ),
            "step_type_representations": sorted(
                step_type_representations
            ),
            "disposition": resolve_disposition(
                quality_reasons = quality_reasons,
                quality_contract = quality_contract
            ),
            "quality_reasons": quality_reasons
        }

    out_of_range_value_count = sum(
        out_of_range_field_counts.values()
    )

    quality_reasons = match_incident_rules(
        target_scope = "measurement",
        target_identity = source_identity,
        quality_contract = quality_contract
    )

    if source_row_index_present:
        quality_reasons.append(
            build_quality_reason(
                reason_code = (
                    "SOURCE_ROW_INDEX_COLUMN_PRESENT"
                ),
                disposition = "include",
                source = "Automated data-quality audit",
                description = (
                    "The source measurement contains an exported integer record-index column. Its values are "
                    "retained for provenance but are not used for measurement ordering or analytical calculations."
                ),
                details = {
                    "first_source_row_index": (
                        first_source_row_index
                    ),
                    "last_source_row_index": (
                        last_source_row_index
                    ),
                    "nonincreasing_transition_count": (
                        source_row_index_nonincreasing_count
                    ),
                    "discontinuity_examples": (
                        source_row_index_discontinuity_examples
                    )
                }
            )
        )

    if elapsed_seconds_column_present:
        quality_reasons.append(
            build_quality_reason(
                reason_code = (
                    "DERIVED_ELAPSED_SECONDS_COLUMN_PRESENT"
                ),
                disposition = "include",
                source = "Automated data-quality audit",
                description = (
                    "The source measurement contains a derived elapsed-seconds column that agrees with run_time "
                    "within the configured tolerance."
                ),
                details = {
                    "column_name": (
                        elapsed_seconds_column
                    ),
                    "maximum_difference_seconds": (
                        maximum_elapsed_seconds_difference
                    ),
                    "tolerance_seconds": (
                        measurement_contract[
                            "elapsed_seconds_tolerance_seconds"
                        ]
                    )
                }
            )
        )

    if ambient_temperature_missing_count == row_count:
        quality_reasons.append(
            build_quality_reason(
                reason_code = (
                    "AMBIENT_TEMPERATURE_NOT_RECORDED"
                ),
                disposition = "include",
                source = "Automated data-quality audit",
                description = (
                    "Every sampled ambient-temperature value is non-finite; assigned treatment temperature"
                    " must come from experiment metadata."
                )
            )
        )

    if surface_temperature_missing_count > 0:
        quality_reasons.append(
            build_quality_reason(
                reason_code = (
                    "SURFACE_TEMPERATURE_VALUE_MISSING"
                ),
                disposition = "manual_review",
                source = "Automated data-quality audit",
                description = (
                    "One or more cell-surface temperature values are non-finite."
                ),
                details = {
                    "missing_value_count": (
                        surface_temperature_missing_count
                    )
                }
            )
        )

    if out_of_range_value_count > 0:
        quality_reasons.append(
            build_quality_reason(
                reason_code = (
                    "MEASUREMENT_VALUE_OUTSIDE_GUARDRAIL"
                ),
                disposition = "manual_review",
                source = "Automated data-quality audit",
                description = (
                    "One or more measurement values fall outside the explicit validation guardrails."
                ),
                details = {
                    "value_count": out_of_range_value_count,
                    "field_counts": dict(
                        sorted(
                            out_of_range_field_counts.items()
                        )
                    ),
                    "examples": out_of_range_examples
                }
            )
        )

    if cycling_direction_mismatch_count > 0:
        quality_reasons.append(
            build_quality_reason(
                reason_code = (
                    "CYCLING_STEP_CURRENT_DIRECTION_MISMATCH"
                ),
                disposition = "include_with_warning",
                source = "Automated data-quality audit",
                description = (
                    "Measured current direction conflicts with one or more recorded cycling-step labels."
                ),
                details = {
                    "sample_count": (
                        cycling_direction_mismatch_count
                    )
                }
            )
        )

    if "numeric_elapsed_value" in run_time_representations:
        quality_reasons.append(
            build_quality_reason(
                reason_code = (
                    "NUMERIC_RUN_TIME_REPRESENTATION"
                ),
                disposition = measurement_contract[
                    "numeric_run_time_disposition"
                ],
                source = "Automated data-quality audit",
                description = (
                    "The source measurement records run_time as numeric values instead of the documented "
                    "clock representation. Values are being used provisionally for structural ordering only. "
                    "Their unit requires confirmation before feature extraction."
                ),
                details = {
                    "run_time_representations": sorted(
                        run_time_representations
                    ),
                    "step_type_representations": sorted(
                        step_type_representations
                    )
                }
            )
        )

    if timestamp_reversal_count > 0:
        quality_reasons.append(
            build_quality_reason(
                reason_code = (
                    "MEASUREMENT_TIMESTAMP_REVERSAL"
                ),
                disposition = "manual_review",
                source = "Automated data-quality audit",
                description = (
                    "One or more measurement timestamps move backward and require review before temporal "
                    "feature extraction."
                ),
                details = {
                    "reversal_count": (
                        timestamp_reversal_count
                    ),
                    "examples": (
                        timestamp_reversal_examples
                    )
                }
            )
        )

    return {
        "source_identity": source_identity,
        "row_count": row_count,
        "source_row_index_present": (
            source_row_index_present
        ),
        "source_row_index_nonincreasing_count": (
            source_row_index_nonincreasing_count
        ),
        "source_row_index_discontinuity_examples": (
            source_row_index_discontinuity_examples
        ),
        "first_source_row_index": (
            first_source_row_index
        ),
        "last_source_row_index": (
            last_source_row_index
        ),
        "first_elapsed_seconds": first_elapsed_seconds,
        "last_elapsed_seconds": last_elapsed_seconds,
        "elapsed_seconds_column_present": (
            elapsed_seconds_column_present
        ),
        "timestamp_reversal_count": (
            timestamp_reversal_count
        ),
        "timestamp_reversal_examples": (
            timestamp_reversal_examples
        ),

        "maximum_elapsed_seconds_difference": (
            maximum_elapsed_seconds_difference
        ),
        "ambient_temperature_missing_count": (
            ambient_temperature_missing_count
        ),
        "surface_temperature_missing_count": (
            surface_temperature_missing_count
        ),
        "out_of_range_value_count": (
            out_of_range_value_count
        ),
        "out_of_range_field_counts": dict(
            sorted(
                out_of_range_field_counts.items()
            )
        ),
        "out_of_range_examples": out_of_range_examples,
        "cycling_direction_mismatch_count": (
            cycling_direction_mismatch_count
        ),
        "disposition": resolve_disposition(
            quality_reasons = quality_reasons,
            quality_contract = quality_contract
        ),
        "run_time_representations": sorted(
            run_time_representations
        ),
        "step_type_representations": sorted(
            step_type_representations
        ),
        "quality_reasons": quality_reasons
    }


# Selecting the complete or explicitly requested archive inventory
def select_archive_objects(article_metadata: dict, selection_mode: str,
                           requested_archive_objects: list[str]
) -> list[str]:
    unique_requested_objects = list(
        dict.fromkeys(
            requested_archive_objects
        )
    )

    if selection_mode == "specified":
        if not unique_requested_objects:
            raise ValueError("At least one archive object is required when selection mode is specified.")

        return unique_requested_objects

    if unique_requested_objects:
        raise ValueError("Archive objects cannot be supplied when selection mode is complete.")

    archive_file_records = [
        file_record
        for file_record in article_metadata["files"]
        if file_record["name"].endswith(".zip")
    ]

    if len(archive_file_records) != EXPECTED_CELL_COUNT:
        raise RuntimeError(f"The complete pinned dataset has an unexpected archive count : {len(archive_file_records)}")

    return sorted(
        build_landed_object_name(
            file_record = file_record
        )
        for file_record in archive_file_records
    )


# Indexing experiment metadata by physical cell identity
def build_experiment_record_index(experiment_metadata_content: str) -> dict:
    metadata_reader = csv.DictReader(
        io.StringIO(
            experiment_metadata_content
        )
    )

    experiment_record_index = {}

    for experiment_record in metadata_reader:
        record_key = (
            experiment_record["stage"],
            experiment_record["type"],
            int(experiment_record["tp"]),
            int(experiment_record["cell"])
        )

        if record_key in experiment_record_index:
            raise RuntimeError(f"Experiment metadata contains a duplicate cell identity : {record_key}")

        experiment_record_index[
            record_key
        ] = experiment_record

    return experiment_record_index


# Resolving one archive to its experiment-metadata record
def find_archive_experiment_record(article_metadata: dict, archive_object_name: str,
                                   experiment_record_index: dict
) -> dict:
    file_record = find_archive_file_record(
        article_metadata = article_metadata,
        archive_object_name = archive_object_name
    )

    archive_name_match = re.fullmatch(
        r"TP_([kz])(\d{2})_(\d{2})\.zip",
        file_record["name"]
    )

    if archive_name_match is None:
        raise RuntimeError(f"Published archive has an unexpected name : {file_record['name']}")

    source_folder = article_metadata[
        "folder_structure"
    ][str(file_record["id"])]

    stage_name = PurePosixPath(
        source_folder
    ).name

    if not stage_name.startswith("Stage_"):
        raise RuntimeError(f"Archive has an unexpected stage folder : {source_folder}")

    record_key = (
        stage_name.removeprefix("Stage_"),
        archive_name_match.group(1),
        int(archive_name_match.group(2)),
        int(archive_name_match.group(3))
    )

    if record_key not in experiment_record_index:
        raise RuntimeError(f"Archive does not have a matching experiment-metadata record : {file_record['name']}")

    return experiment_record_index[
        record_key
    ]


# Confirming measurements documented as absent are not present
def validate_expected_absent_measurements(archive_identity: str, measurement_names: list[str],
                                          quality_contract: dict
) -> None:
    for incident_rule in quality_contract[
        "incident_rules"
    ]:
        if (
            incident_rule["target_scope"] != "archive"
            or re.fullmatch(
                incident_rule["target_pattern"],
                archive_identity
            )
            is None
        ):
            continue

        expected_absent_pattern = incident_rule.get(
            "expected_absent_measurement_pattern"
        )

        if expected_absent_pattern is None:
            continue

        unexpectedly_present_measurements = sorted(
            measurement_name
            for measurement_name in measurement_names
            if re.fullmatch(
                expected_absent_pattern,
                measurement_name
            )
            is not None
        )

        if unexpectedly_present_measurements:
            raise RuntimeError(
                f"Measurements documented as absent are present : {archive_identity} : {unexpectedly_present_measurements}"
            )


# Recording a measurement that cannot complete automated auditing
def build_measurement_audit_failure(source_identity: str, measurement_name: str,
                                    audit_exception: Exception, quality_contract: dict
) -> dict:
    quality_reasons = match_incident_rules(
        target_scope = "measurement",
        target_identity = source_identity,
        quality_contract = quality_contract
    )

    quality_reasons.append(
        build_quality_reason(
            reason_code = (
                "MEASUREMENT_AUDIT_FAILURE"
            ),
            disposition = "manual_review",
            source = "Automated data-quality audit",
            description = (
                "The measurement could not complete automated auditing and requires review."
            ),
            details = {
                "exception_type": (
                    type(audit_exception).__name__
                ),
                "exception_message": str(
                    audit_exception
                )
            }
        )
    )

    return {
        "measurement_file": measurement_name,
        "source_identity": source_identity,
        "audit_status": "failed",
        "disposition": resolve_disposition(
            quality_reasons = quality_reasons,
            quality_contract = quality_contract
        ),
        "quality_reasons": quality_reasons
    }


# Recording an archive that cannot complete automated auditing
def build_archive_audit_failure(article_metadata: dict, archive_object_name: str,
                                bucket_name: str, audit_exception: Exception,
                                quality_contract: dict
) -> dict:
    file_record = find_archive_file_record(
        article_metadata = article_metadata,
        archive_object_name = archive_object_name
    )

    source_folder = article_metadata[
        "folder_structure"
    ][str(file_record["id"])]

    archive_identity = (
        f"{PurePosixPath(source_folder).name}/"
        f"{file_record['name']}"
    )

    quality_reasons = match_incident_rules(
        target_scope = "archive",
        target_identity = archive_identity,
        quality_contract = quality_contract
    )

    quality_reasons.append(
        build_quality_reason(
            reason_code = "ARCHIVE_AUDIT_FAILURE",
            disposition = "manual_review",
            source = "Automated data-quality audit",
            description = (
                "The archive could not complete automated auditing and requires review."
            ),
            details = {
                "exception_type": (
                    type(audit_exception).__name__
                ),
                "exception_message": str(
                    audit_exception
                )
            }
        )
    )

    return {
        "archive_identity": archive_identity,
        "archive_structure": None,
        "cloud_object": (
            f"gs://{bucket_name}/{archive_object_name}"
        ),
        "audit_status": "failed",
        "measurement_count": 0,
        "measurement_disposition_counts": {},
        "disposition": "manual_review",
        "maximum_measurement_disposition": "manual_review",
        "quality_reasons": quality_reasons,
        "measurements": []
    }


# Confirming that every documented incident rule matches the dataset
def build_incident_rule_coverage(archive_audits: list[dict], quality_contract: dict) -> dict:
    observed_reason_counts = Counter()

    for archive_audit in archive_audits:
        for quality_reason in archive_audit[
            "quality_reasons"
        ]:
            observed_reason_counts[
                quality_reason["reason_code"]
            ] += 1

        for measurement_audit in archive_audit[
            "measurements"
        ]:
            for quality_reason in measurement_audit[
                "quality_reasons"
            ]:
                observed_reason_counts[
                    quality_reason["reason_code"]
                ] += 1

    configured_reason_codes = {
        incident_rule["reason_code"]
        for incident_rule in [
            *quality_contract[
                "incident_rules"
            ],
            *quality_contract[
                "downstream_incident_rules"
            ]
        ]
    }

    unmatched_reason_codes = sorted(
        reason_code
        for reason_code in configured_reason_codes
        if observed_reason_counts[reason_code] == 0
    )

    if unmatched_reason_codes:
        raise RuntimeError(f"Documented incident rules did not match the complete dataset : {unmatched_reason_codes}")

    return {
        reason_code: observed_reason_counts[
            reason_code
        ]
        for reason_code in sorted(
            configured_reason_codes
        )
    }


# Auditing one landed archive without retaining extracted files
def audit_cloud_archive(bucket: storage.Bucket, article_metadata: dict,
                        archive_object_name: str, quality_contract: dict,
                        experiment_record: dict, archive_cache_directory: Path
) -> dict:
    file_record = find_archive_file_record(
        article_metadata = article_metadata,
        archive_object_name = archive_object_name
    )

    source_folder = article_metadata[
        "folder_structure"
    ][str(file_record["id"])]

    stage_name = PurePosixPath(
        source_folder
    ).name

    archive_identity = (
        f"{stage_name}/{file_record['name']}"
    )

    archive_reasons = match_incident_rules(
        target_scope = "archive",
        target_identity = archive_identity,
        quality_contract = quality_contract
    )

    published_md5 = file_record.get(
        "computed_md5"
    )

    if not published_md5:
        archive_reasons.append(
            build_quality_reason(
                reason_code = "PUBLISHED_MD5_UNAVAILABLE",
                disposition = "include_with_warning",
                source = "Figshare article metadata",
                description = (
                    "Figshare does not provide an MD5 checksum. "
                    "The landed object matches the published size and its recorded destination SHA-256 baseline."
                )
            )
        )

    archive_path = resolve_cached_archive_path(
        bucket = bucket,
        article_metadata = article_metadata,
        archive_object_name = archive_object_name,
        archive_cache_directory = (
            archive_cache_directory
        )
    )

    # Validating archive structure before auditing measurement values
    archive_structure_summary = (
        inspect_archive_file(
            archive_path = archive_path,
            require_complete_metadata_pairs = False,
            validate_member_checksums = False
        )
    )

    missing_metadata_names = (
        archive_structure_summary[
            "missing_metadata_names"
        ]
    )

    unexpected_metadata_names = (
        archive_structure_summary[
            "unexpected_metadata_names"
        ]
    )

    if (
        missing_metadata_names
        or unexpected_metadata_names
    ):
        archive_reasons.append(
            build_quality_reason(
                reason_code = (
                    "ARCHIVE_METADATA_PAIR_MISMATCH"
                ),
                disposition = "manual_review",
                source = "Automated data-quality audit",
                description = (
                    "The archive contains missing or unmatched companion metadata files."
                ),
                details = {
                    "missing_metadata_names": (
                        missing_metadata_names
                    ),
                    "unexpected_metadata_names": (
                        unexpected_metadata_names
                    )
                }
            )
        )

    with zipfile.ZipFile(
        archive_path,
        mode = "r"
    ) as archive:
        archive_member_names = set(
            archive.namelist()
        )

        measurement_names = sorted(
            member_name
            for member_name in archive_member_names
            if member_name.endswith(".csv")
        )

        validate_expected_absent_measurements(
            archive_identity = archive_identity,
            measurement_names = measurement_names,
            quality_contract = quality_contract
        )

        measurement_audits = []

        for measurement_name in measurement_names:
            source_identity = (
                f"{stage_name}/{measurement_name}"
            )

            try:
                with archive.open(
                    measurement_name,
                    mode = "r"
                ) as binary_stream:
                    with io.TextIOWrapper(
                        binary_stream,
                        encoding = "utf-8-sig",
                        newline = ""
                    ) as text_stream:
                        measurement_audit = (
                            audit_measurement_stream(
                                measurement_stream = text_stream,
                                source_identity = source_identity,
                                quality_contract = quality_contract
                            )
                        )

                measurement_path = PurePosixPath(
                    measurement_name
                )

                metadata_name = str(
                    measurement_path.with_name(
                        f"{measurement_path.stem}_meta.txt"
                    )
                )

                if metadata_name not in archive_member_names:
                    measurement_audit[
                        "quality_reasons"
                    ].append(
                        build_quality_reason(
                            reason_code = (
                                "COMPANION_METADATA_MISSING"
                            ),
                            disposition = "manual_review",
                            source = (
                                "Automated data-quality audit"
                            ),
                            description = (
                                "The measurement does not have its expected companion metadata file."
                            ),
                            details = {
                                "expected_metadata_name": (
                                    metadata_name
                                )
                            }
                        )
                    )

                    measurement_audit[
                        "disposition"
                    ] = resolve_disposition(
                        quality_reasons = (
                            measurement_audit[
                                "quality_reasons"
                            ]
                        ),
                        quality_contract = (
                            quality_contract
                        )
                    )

                    measurement_audit[
                        "companion_metadata"
                    ] = None

                    measurement_audit[
                        "audit_status"
                    ] = "completed"

                    measurement_audits.append(
                        measurement_audit
                    )

                    continue

                with archive.open(
                    metadata_name,
                    mode = "r"
                ) as metadata_stream:
                    metadata_content = (
                        metadata_stream.read().decode(
                            "utf-8-sig"
                        )
                    )

                companion_metadata_audit = (
                    audit_companion_metadata(
                        metadata_content = metadata_content,
                        measurement_name = measurement_name,
                        source_identity = source_identity,
                        experiment_record = experiment_record
                    )
                )

                companion_quality_reasons = (
                    companion_metadata_audit.pop(
                        "quality_reasons"
                    )
                )

                measurement_audit[
                    "quality_reasons"
                ].extend(
                    companion_quality_reasons
                )

                measurement_audit["disposition"] = (
                    resolve_disposition(
                        quality_reasons = measurement_audit[
                            "quality_reasons"
                        ],
                        quality_contract = quality_contract
                    )
                )

                measurement_audit[
                    "companion_metadata"
                ] = companion_metadata_audit

                measurement_audit["audit_status"] = "completed"

                measurement_audits.append(
                    measurement_audit
                )

            except (
                RuntimeError,
                ValueError,
                UnicodeDecodeError
            ) as exc:
                measurement_audits.append(
                    build_measurement_audit_failure(
                        source_identity = (
                            source_identity
                        ),
                        measurement_name = (
                            measurement_name
                        ),
                        audit_exception = exc,
                        quality_contract = (
                            quality_contract
                        )
                    )
                )

                continue

    measurement_disposition_counts = Counter(
        measurement_audit["disposition"]
        for measurement_audit in measurement_audits
    )

    # Resolving archive findings independently from measurement findings
    archive_disposition = resolve_disposition(
        quality_reasons = archive_reasons,
        quality_contract = quality_contract
    )

    # Summarizing the strictest measurement disposition without changing archive scope
    maximum_measurement_disposition = (
        resolve_disposition(
            quality_reasons = [
                {
                    "disposition": (
                        measurement_audit[
                            "disposition"
                        ]
                    )
                }
                for measurement_audit in (
                    measurement_audits
                )
            ],
            quality_contract = quality_contract
        )
    )

    return {
        "archive_identity": archive_identity,
        "archive_structure": {
            "member_count": archive_structure_summary[
                "member_count"
            ],
            "missing_metadata_names": (
                archive_structure_summary[
                    "missing_metadata_names"
                ]
            ),
            "unexpected_metadata_names": (
                archive_structure_summary[
                    "unexpected_metadata_names"
                ]
            ),
            "measurement_file_count": (
                archive_structure_summary[
                    "measurement_file_count"
                ]
            ),
            "metadata_file_count": (
                archive_structure_summary[
                    "metadata_file_count"
                ]
            ),
            "uncompressed_size_bytes": (
                archive_structure_summary[
                    "uncompressed_size_bytes"
                ]
            ),
            "empty_member_count": (
                archive_structure_summary[
                    "empty_member_count"
                ]
            ),
            "extension_counts": (
                archive_structure_summary[
                    "extension_counts"
                ]
            )
        },
        "cloud_object": (
            f"gs://{bucket.name}/{archive_object_name}"
        ),
        "published_md5_available": bool(
            published_md5
        ),
        "measurement_count": len(
            measurement_audits
        ),
        "measurement_disposition_counts": dict(
            sorted(
                measurement_disposition_counts.items()
            )
        ),
        "disposition": archive_disposition,
        "maximum_measurement_disposition": (
            maximum_measurement_disposition
        ),
        "quality_reasons": archive_reasons,
        "measurements": measurement_audits,
        "audit_status": "completed"
    }


# Auditing metadata and selected landed archives
def main() -> None:
    args = parse_args()

    quality_contract = load_quality_contract(
        contract_path = args.contract_path
    )

    storage_client = storage.Client(
        project = args.project_id
    )

    bucket = storage_client.bucket(
        args.bucket_name
    )

    if not bucket.exists():
        raise RuntimeError(f"Cloud Storage bucket was not found : {args.bucket_name}")

    article_metadata = json.loads(
        download_text_object(
            bucket = bucket,
            object_name = ARTICLE_METADATA_OBJECT
        )
    )

    dataset_contract = quality_contract[
        "dataset"
    ]

    observed_dataset_identity = {
        "figshare_article_id": article_metadata["id"],
        "figshare_version": article_metadata["version"],
        "doi": article_metadata["doi"]
    }

    if observed_dataset_identity != dataset_contract:
        raise RuntimeError(
            f"Landed article metadata does not match the data-quality contract : {observed_dataset_identity}"
        )

    experiment_metadata_file = (
        find_dataset_root_file(
            article_metadata = article_metadata,
            file_name = "experiments_meta.csv"
        )
    )

    experiment_metadata_object = (
        build_landed_object_name(
            file_record = experiment_metadata_file
        )
    )

    experiment_metadata_content = (
        download_text_object(
            bucket = bucket,
            object_name = (
                experiment_metadata_object
            )
        )
    )

    experiment_record_index = (
        build_experiment_record_index(
            experiment_metadata_content = (
                experiment_metadata_content
            )
        )
    )

    experiment_metadata_audit = (
        audit_experiment_metadata(
            experiment_metadata_content = (
                experiment_metadata_content
            ),
            quality_contract = quality_contract
        )
    )

    selected_archive_objects = select_archive_objects(
        article_metadata = article_metadata,
        selection_mode = args.selection_mode,
        requested_archive_objects = (
            args.archive_object
        )
    )

    archive_audits = []

    for archive_object_name in selected_archive_objects:
        try:
            experiment_record = (
                find_archive_experiment_record(
                    article_metadata = article_metadata,
                    archive_object_name = (
                        archive_object_name
                    ),
                    experiment_record_index = (
                        experiment_record_index
                    )
                )
            )

            archive_audits.append(
                audit_cloud_archive(
                    bucket = bucket,
                    article_metadata = article_metadata,
                    archive_object_name = (
                        archive_object_name
                    ),
                    quality_contract = quality_contract,
                    experiment_record = experiment_record,
                    archive_cache_directory = (
                        args.archive_cache_directory
                    )
                )
            )

        except (
            RuntimeError,
            ValueError,
            FileNotFoundError,
            zipfile.BadZipFile
        ) as exc:
            archive_audits.append(
                build_archive_audit_failure(
                    article_metadata = article_metadata,
                    archive_object_name = (
                        archive_object_name
                    ),
                    bucket_name = args.bucket_name,
                    audit_exception = exc,
                    quality_contract = quality_contract
                )
            )

    archive_audit_failure_count = sum(
        archive_audit.get(
            "audit_status"
        ) == "failed"
        for archive_audit in archive_audits
    )

    measurement_audit_failure_count = sum(
        measurement_audit.get(
            "audit_status"
        ) == "failed"
        for archive_audit in archive_audits
        for measurement_audit in archive_audit[
            "measurements"
        ]
    )

    incident_rule_coverage = None
    incident_rule_coverage_error = None

    if (
        args.selection_mode == "complete"
        and archive_audit_failure_count == 0
        and measurement_audit_failure_count == 0
    ):
        try:
            incident_rule_coverage = (
                build_incident_rule_coverage(
                    archive_audits = archive_audits,
                    quality_contract = quality_contract
                )
            )

        except RuntimeError as exc:
            incident_rule_coverage_error = str(
                exc
            )

    else:
        incident_rule_coverage = None

    audit_summary = {
        "contract_version": quality_contract[
            "contract_version"
        ],
        "dataset": observed_dataset_identity,
        "experiment_metadata": (
            experiment_metadata_audit
        ),
        "feature_contract": quality_contract[
            "feature_contract"
        ],
        "incident_rule_coverage": (
            incident_rule_coverage
        ),
        "incident_rule_coverage_error": (
            incident_rule_coverage_error
        ),
        "measurement_audit_failure_count": (
            measurement_audit_failure_count
        ),
        "archives": archive_audits,
        "archive_audit_failure_count": (
            archive_audit_failure_count
        )
    }

    audit_content = (
        json.dumps(
            audit_summary,
            indent = 2,
            sort_keys = True
        )
        + "\n"
    )

    args.output_path.parent.mkdir(
        parents = True,
        exist_ok = True
    )

    args.output_path.write_text(
        audit_content,
        encoding = "utf-8"
    )

    print(
        "Complete data-quality scan finished."
    )
    print(
        f"Audited archive count : {len(archive_audits)}"
    )
    print(
        f"Audit artifact : {args.output_path}"
    )

    total_audit_failure_count = (
        archive_audit_failure_count
        + measurement_audit_failure_count
    )

    audit_has_unresolved_failures = (
        total_audit_failure_count > 0
        or incident_rule_coverage_error is not None
    )

    if audit_has_unresolved_failures:
        print(
            "Data-quality audit completed with unresolved failures.", file = sys.stderr
        )

        if incident_rule_coverage_error is not None:
            print(
                incident_rule_coverage_error,
                file = sys.stderr
            )

        sys.exit(2)

    print("Data-quality audit completed successfully.")


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            f"Data-quality audit failed : {exc}",
            file = sys.stderr
        )
        sys.exit(1)
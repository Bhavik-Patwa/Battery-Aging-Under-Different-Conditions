import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile

from datetime import date
from pathlib import Path
from collections.abc import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DESIGN_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "experimental_design"
    / "experimental_design_report.json"
)

DEFAULT_POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "causal_identification_policy.json"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "causal_identification"
)

ESTIMAND_IDENTITY_FIELDS = [
    "decision_scope",
    "first_condition_protocol_identity",
    "second_condition_protocol_identity",
    "outcome",
    "cohort",
    "horizon_axis"
]

ESTIMAND_OUTPUT_COLUMNS = [
    "estimand_decision_identity",
    "aging_mode",
    "decision_scope",
    "first_condition_protocol_identity",
    "second_condition_protocol_identity",
    "outcome",
    "cohort",
    "horizon_axis",
    "target_estimand_identity",
    "condition_comparison_interpretation",
    "final_support_classification",
    "target_estimand_available",
    "upstream_causal_identification_status",
    "point_identification_status",
    "identification_strategy_status",
    "causal_identification_reason_codes",
    "interpretation_restriction_codes",
    "associational_model_status",
    "causal_sensitivity_status",
    "exposure_time_interpretation",
    "horizon_conditioning_status",
    "cross_stage_comparison",
    "equipment_restriction_present"
]

ASSUMPTION_OUTPUT_COLUMNS = [
    "assumption_name",
    "assumption_scope",
    "evidence_status",
    "required_for_point_identification",
    "evidence_source",
    "observed_evidence",
    "analytical_implication",
    "reason_code"
]

POLICY_VERSION_PATTERN = re.compile(
    r"^[1-9]\d*\.\d+$"
)

SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)

REQUIREMENT_ASSUMPTION_NAMES = {
    "assignment_mechanism_required": (
        "treatment_assignment"
    ),
    "conditional_exchangeability_required_for_covariate_adjustment": (
        "conditional_exchangeability"
    ),
    "conditional_positivity_required": (
        "conditional_positivity"
    ),
    "consistency_and_well_defined_treatment_versions_required": (
        "consistency_and_treatment_versions"
    ),
    "no_interference_required": (
        "no_interference"
    ),
    "selection_process_identification_required": (
        "selection_and_outcome_availability"
    ),
    "outcome_measurement_assumptions_required": (
        "outcome_measurement"
    ),
    "cross_stage_transportability_requires_explicit_assumptions": (
        "cross_stage_transportability"
    )
}

REQUIRED_DESIGN_EVIDENCE_FIELDS = {
    "cell_to_condition_randomization_status",
    "causal_interpretation",
    "cross_stage_transportability_status",
    "calendar_net_storage_exposure_status",
    "equipment_adjustment_status",
    "baseline_adjustment_status"
}

EXPECTED_IDENTIFICATION_STRATEGIES = {
    "overall_status": "no_supported_identification_strategy",
    "documented_random_assignment": "not_established",
    "pretreatment_covariate_adjustment": (
        "not_authorized_no_sufficient_adjustment_set"
    ),
    "instrumental_variable": "not_established",
    "regression_discontinuity": "not_established",
    "difference_in_differences": "not_established",
    "other_natural_experiment": "not_established"
}

EXPECTED_CURRENT_IDENTIFICATION_STATUS = {
    "assignment_mechanism": "not_established",
    "conditional_exchangeability": "not_established",
    "sufficient_pretreatment_adjustment_set": "not_authorized",
    "conditional_positivity": "not_established",
    "consistency": "not_established_for_causal_interventions",
    "no_interference": "not_established",
    "selection_process": "not_established",
    "outcome_measurement_error_process": (
        "not_established_for_causal_identification"
    ),
    "cross_stage_transportability": "not_established",
    "calendar_net_storage_exposure": "unavailable",
    "causal_point_identification": (
        "not_identified_from_observed_design"
    )
}


EXPECTED_UPSTREAM_ESTIMAND_CONTRACT = {
    "aging_modes": [
        "calendar_aging",
        "cycle_aging"
    ],
    "decision_scopes": [
        "condition_protocol_pair"
    ],
    "outcomes": [
        "relative_capacity",
        "relative_dc_internal_resistance"
    ],
    "cohorts": [
        "primary",
        "sensitivity"
    ],
    "horizon_axes_by_aging_mode": {
        "calendar_aging": [
            "elapsed_calendar_days"
        ],
        "cycle_aging": [
            "elapsed_calendar_days",
            "cumulative_full_equivalent_cycles_observed"
        ]
    },
    "support_classifications": [
        "unsupported",
        "descriptive_only",
        "supported_with_restrictions",
        "sensitivity_only"
    ],
    "causal_identification_statuses": [
        "not_established"
    ],
    "cross_stage_transportability_statuses": [
        "not_applicable"
    ],
    "sensitivity_only_permitted_cohorts": [
        "sensitivity"
    ]
}

EXPECTED_ASSUMPTION_EVIDENCE_STATUSES = {
    "treatment_assignment": "not_established",
    "conditional_exchangeability": "not_established",
    "conditional_positivity": "not_established",
    "consistency_and_treatment_versions": (
        "not_established_for_causal_interventions"
    ),
    "no_interference": "not_established",
    "selection_and_outcome_availability": "not_established",
    "outcome_measurement": (
        "validated_feature_definitions_without_"
        "causal_measurement_error_identification"
    ),
    "achieved_fec_horizon": (
        "requires_explicit_causal_definition"
    ),
    "calendar_exposure_definition": (
        "net_storage_exposure_unavailable"
    ),
    "cross_stage_transportability": "not_established"
}

IDENTIFICATION_REPORT_FIELDS = {
    "assessment_status",
    "policy_version",
    "policy_effective_date",
    "policy_sha256",
    "policy",
    "design_report_sha256",
    "independent_unit",
    "identification_decision_basis",
    "outcome_values_used_for_identification_decisions",
    "predictive_performance_used_for_identification_decisions",
    "causal_modeling_status",
    "associational_hierarchical_modeling_status",
    "causal_sensitivity_analysis_status",
    "causal_sensitivity_activation_requirement",
    "estimand_count",
    "assumption_count",
    "point_identification_status_counts",
    "associational_model_status_counts",
    "tables"
}


# Parsing assessment arguments
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = (
            "Assessing causal-identification evidence for authorized battery-aging estimands."
        )
    )

    parser.add_argument(
        "--design-report-path",
        type = Path,
        default = DEFAULT_DESIGN_REPORT_PATH
    )

    parser.add_argument(
        "--policy-path",
        type = Path,
        default = DEFAULT_POLICY_PATH
    )

    parser.add_argument(
        "--output-directory",
        type = Path,
        default = DEFAULT_OUTPUT_DIRECTORY
    )

    return parser.parse_args()


# Producing deterministic canonical JSON
def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys = True,
        separators = (
            ",",
            ":"
        ),
        ensure_ascii = False
    )


# Calculating one file SHA-256 digest
def calculate_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open(
        mode = "rb"
    ) as input_stream:
        for data_block in iter(
            lambda: input_stream.read(
                1024 * 1024
            ),
            b""
        ):
            digest.update(
                data_block
            )

    return digest.hexdigest()


# Rejecting duplicate JSON object fields
def build_unique_json_object(object_pairs: list[tuple[str, object]]) -> dict:
    json_object = {}

    for field_name, field_value in object_pairs:
        if field_name in json_object:
            raise RuntimeError(f"JSON object contains a duplicate field : {field_name}")

        json_object[
            field_name
        ] = field_value

    return json_object


# Loading one strict JSON object
def load_json_object(path: Path) -> dict:
    try:
        with path.open(
            mode = "r",
            encoding = "utf-8"
        ) as input_stream:
            value = json.load(
                input_stream,
                object_pairs_hook = (
                    build_unique_json_object
                )
            )

    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON document is invalid : {path}") from exc

    if not isinstance(
        value,
        dict
    ):
        raise RuntimeError(f"JSON document is not an object : {path}")

    return value


# Parsing one nonnegative integer
def parse_nonnegative_integer(value: object, field_name: str) -> int:
    try:
        parsed_value = int(
            value
        )

    except (
        TypeError,
        ValueError
    ) as exc:
        raise RuntimeError(f"Invalid integer value : {field_name} : {value!r}") from exc

    if (
        isinstance(
            value,
            bool
        )
        or parsed_value < 0
        or str(
            parsed_value
        )
        != str(
            value
        ).strip()
    ):
        raise RuntimeError(f"Invalid nonnegative integer value : {field_name} : {value!r}")

    return parsed_value


# Parsing one Boolean value
def parse_boolean(value: object, field_name: str) -> bool:
    if isinstance(
        value,
        bool
    ):
        return value

    if isinstance(
        value,
        str
    ):
        normalized_value = value.strip().lower()

        if normalized_value == "true":
            return True

        if normalized_value == "false":
            return False

    raise RuntimeError(f"Invalid Boolean value : {field_name} : {value!r}")


# Reading one CSV table
def read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(
        mode = "r",
        encoding = "utf-8",
        newline = ""
    ) as input_stream:
        reader = csv.DictReader(
            input_stream
        )

        field_names = reader.fieldnames

        if field_names is None:
            raise RuntimeError(f"CSV table does not contain a header : {path}")

        rows = list(
            reader
        )

    return field_names, rows


# Verifying one declared report table
def verify_report_table(report_path: Path, report: dict, table_name: str) -> tuple[Path, list[dict]]:
    table_manifest = report.get(
        "tables"
    )

    if not isinstance(
        table_manifest,
        dict
    ):
        raise RuntimeError(f"Report does not contain a table manifest : {report_path}")

    table_record = table_manifest.get(
        table_name
    )

    if not isinstance(
        table_record,
        dict
    ):
        raise RuntimeError(f"Report is missing required table : {table_name}")

    file_name = table_record.get(
        "file_name"
    )

    expected_sha256 = table_record.get(
        "sha256"
    )

    expected_row_count = table_record.get(
        "row_count"
    )

    if (
        not isinstance(
            file_name,
            str
        )
        or not file_name
        or not isinstance(
            expected_sha256,
            str
        )
        or len(
            expected_sha256
        ) != 64
        or not isinstance(
            expected_row_count,
            int
        )
        or isinstance(
            expected_row_count,
            bool
        )
        or expected_row_count < 0
    ):
        raise RuntimeError(f"Invalid table manifest record : {table_name}")

    table_path = (
        report_path.parent
        / file_name
    )

    if not table_path.is_file():
        raise RuntimeError(f"Required table is unavailable : {table_path}")

    observed_sha256 = calculate_file_sha256(
        table_path
    )

    if observed_sha256 != expected_sha256:
        raise RuntimeError(f"Table SHA-256 does not match its report : {table_name}")

    field_names, rows = read_csv_rows(
        table_path
    )

    if len(
        rows
    ) != expected_row_count:
        raise RuntimeError(f"Table row count does not match its report : {table_name}")

    schema = table_record.get(
        "schema"
    )

    if not isinstance(
        schema,
        list
    ):
        raise RuntimeError(f"Table manifest does not contain an exact schema : {table_name}")

    schema_names = [
        field_record[
            "name"
        ]
        for field_record in schema
    ]

    if field_names != schema_names:
        raise RuntimeError(f"Table schema does not match its report : {table_name}")

    return table_path, rows


# Requiring an object to contain exactly the declared fields
def require_exact_fields(value: object, required_fields: set[str], description: str) -> dict:
    if not isinstance(
        value,
        dict
    ):
        raise RuntimeError(f"{description} is not an object.")

    observed_fields = set(
        value
    )

    if observed_fields != required_fields:
        raise RuntimeError(
            f"{description} fields differ from the contract : missing {sorted(required_fields - observed_fields)} : "
            f"unexpected {sorted(observed_fields - required_fields)}"
        )

    return value


# Validating one registered reason-code sequence
def validate_reason_code_sequence(reason_codes: object, registered_reason_codes: set[str], description: str) -> None:
    if (
        not isinstance(
            reason_codes,
            list
        )
        or any(
            not isinstance(
                reason_code,
                str
            )
            or not reason_code
            for reason_code in reason_codes
        )
        or len(
            reason_codes
        )
        != len(
            set(
                reason_codes
            )
        )
    ):
        raise RuntimeError(f"{description} is invalid.")

    unknown_reason_codes = (
        set(
            reason_codes
        )
        - registered_reason_codes
    )

    if unknown_reason_codes:
        raise RuntimeError(f"{description} contains unregistered reason codes : {sorted(unknown_reason_codes)}")


# Reconciling registered reason rules with their assumptions
def validate_reason_rule_consistency(reason_code_rules: dict, assumptions_by_name: dict[str, dict]) -> None:
    universal_assumption_codes = {
        assumption_record["reason_code"]
        for assumption_record in assumptions_by_name.values()
        if (
            assumption_record["assumption_scope"]
            == "all_estimands"
            and assumption_record[
                "required_for_point_identification"
            ]
            is True
        )
    }

    expected_universal_codes = {
        "CAUSAL_IDENTIFICATION_NOT_ESTABLISHED_UPSTREAM",
        "ALTERNATIVE_IDENTIFICATION_STRATEGY_NOT_ESTABLISHED",
        *universal_assumption_codes
    }

    if set(
        reason_code_rules[
            "universal_point_identification"
        ]
    ) != expected_universal_codes:
        raise RuntimeError("Universal identification reasons do not reconcile with the registered assumptions.")

    expected_scope_rules = {
        "calendar_elapsed_time": {
            "causal_reason_codes": set(),
            "interpretation_restriction_codes": {
                assumptions_by_name[
                    "calendar_exposure_definition"
                ][
                    "reason_code"
                ]
            }
        },
        "cycle_elapsed_time": {
            "causal_reason_codes": set(),
            "interpretation_restriction_codes": set()
        },
        "cycle_fec": {
            "causal_reason_codes": {
                assumptions_by_name[
                    "achieved_fec_horizon"
                ][
                    "reason_code"
                ]
            },
            "interpretation_restriction_codes": set()
        },
        "cross_stage_comparison": {
            "causal_reason_codes": {
                assumptions_by_name[
                    "cross_stage_transportability"
                ][
                    "reason_code"
                ]
            },
            "interpretation_restriction_codes": set()
        },
        "equipment_restriction": {
            "causal_reason_codes": set(),
            "interpretation_restriction_codes": {
                "RECORDED_EQUIPMENT_ALLOCATION_RESTRICTION"
            }
        },
        "associational_model_not_authorized": {
            "causal_reason_codes": set(),
            "interpretation_restriction_codes": {
                "DESIGN_SUPPORT_NOT_AUTHORIZED"
            }
        }
    }

    for rule_name, expected_rule in (
        expected_scope_rules.items()
    ):
        observed_rule = reason_code_rules[
            rule_name
        ]

        for code_family in [
            "causal_reason_codes",
            "interpretation_restriction_codes"
        ]:
            if set(
                observed_rule[
                    code_family
                ]
            ) != expected_rule[
                code_family
            ]:
                raise RuntimeError(
                    f"Identification reason rule conflicts with its registered assumption : {rule_name} : {code_family}"
                )


# Validating the identification policy
def validate_policy(policy: dict) -> None:
    required_top_level_fields = {
        "policy_version",
        "effective_date",
        "previous_policy_sha256",
        "registration",
        "amendment",
        "independent_unit",
        "decision_identity_fields",
        "upstream_estimand_contract",
        "required_design_evidence",
        "identification_requirements",
        "identification_strategies_assessed",
        "estimand_interpretations",
        "current_identification_status",
        "reason_code_rules",
        "assumption_inventory",
        "model_authorization",
        "reason_codes",
        "output_contract"
    }

    observed_top_level_fields = set(
        policy
    )

    missing_fields = (
        required_top_level_fields
        - observed_top_level_fields
    )

    unexpected_fields = (
        observed_top_level_fields
        - required_top_level_fields
    )

    if (
        missing_fields
        or unexpected_fields
    ):
        raise RuntimeError(
            f"Identification-policy fields differ from the contract : missing {sorted(missing_fields)} : "
            f"unexpected {sorted(unexpected_fields)}"
        )

    policy_version = policy[
        "policy_version"
    ]

    if (
        not isinstance(
            policy_version,
            str
        )
        or POLICY_VERSION_PATTERN.fullmatch(
            policy_version
        )
        is None
    ):
        raise RuntimeError("Identification policy has an invalid version.")

    try:
        date.fromisoformat(
            policy[
                "effective_date"
            ]
        )

    except (
        TypeError,
        ValueError
    ) as exc:
        raise RuntimeError("Identification policy has an invalid effective date.") from exc

    previous_policy_sha256 = policy[
        "previous_policy_sha256"
    ]

    if (
        not isinstance(
            previous_policy_sha256,
            str
        )
        or SHA256_PATTERN.fullmatch(
            previous_policy_sha256
        )
        is None
    ):
        raise RuntimeError("Identification policy has an invalid previous-policy digest.")

    registration = require_exact_fields(
        value = policy[
            "registration"
        ],
        required_fields = {
            "status",
            "descriptive_results_available_at_registration",
            "causal_model_results_available_at_registration",
            "outcome_values_used_for_identification_decisions",
            "predictive_performance_used_for_identification_decisions",
            "result_informed"
        },
        description = "Identification-policy registration"
    )

    if (
        not isinstance(
            registration,
            dict
        )
        or registration.get(
            "status"
        )
        != "registered_before_bayesian_model_fitting"
        or registration.get(
            "descriptive_results_available_at_registration"
        )
        is not True
        or registration.get(
            "causal_model_results_available_at_registration"
        )
        is not False
        or registration.get(
            "outcome_values_used_for_identification_decisions"
        )
        is not False
        or registration.get(
            "predictive_performance_used_for_identification_decisions"
        )
        is not False
    ):
        raise RuntimeError("Identification-policy registration context is invalid.")

    if policy[
        "independent_unit"
    ] != "cell_identity":
        raise RuntimeError("Identification policy does not use the cell as the independent unit.")

    if policy[
        "decision_identity_fields"
    ] != ESTIMAND_IDENTITY_FIELDS:
        raise RuntimeError("Identification policy contains an unexpected decision identity.")


    upstream_estimand_contract = require_exact_fields(
        value = policy[
            "upstream_estimand_contract"
        ],
        required_fields = set(
            EXPECTED_UPSTREAM_ESTIMAND_CONTRACT
        ),
        description = (
            "Upstream estimand contract"
        )
    )

    if (
        upstream_estimand_contract
        != EXPECTED_UPSTREAM_ESTIMAND_CONTRACT
    ):
        raise RuntimeError("Upstream estimand contract differs from the registered supported vocabulary.")

    required_design_evidence = require_exact_fields(
        value = policy[
            "required_design_evidence"
        ],
        required_fields = (
            REQUIRED_DESIGN_EVIDENCE_FIELDS
        ),
        description = (
            "Required design evidence"
        )
    )

    if any(
        not isinstance(
            evidence_value,
            str
        )
        or not evidence_value.strip()
        for evidence_value in (
            required_design_evidence.values()
        )
    ):
        raise RuntimeError("Required design evidence contains an invalid value.")

    model_authorization = require_exact_fields(
        value = policy[
            "model_authorization"
        ],
        required_fields = {
            "associational_hierarchical_modeling",
            "causal_modeling",
            "causal_sensitivity_analysis"
        },
        description = "Model authorization"
    )

    causal_authorization = require_exact_fields(
        value = model_authorization[
            "causal_modeling"
        ],
        required_fields = {
            "authorization_status",
            "reason",
            "bayesian_estimation_alone_establishes_identification"
        },
        description = "Causal-model authorization"
    )

    if (
        causal_authorization[
            "authorization_status"
        ]
        != "not_authorized"
        or not isinstance(
            causal_authorization[
                "reason"
            ],
            str
        )
        or not causal_authorization[
            "reason"
        ].strip()
        or causal_authorization[
            "bayesian_estimation_alone_establishes_identification"
        ]
        is not False
    ):
        raise RuntimeError("Causal-model authorization is inconsistent.")

    reason_codes = policy[
        "reason_codes"
    ]

    if (
        not isinstance(
            reason_codes,
            dict
        )
        or not reason_codes
        or any(
            not isinstance(
                reason_code,
                str
            )
            or not reason_code
            or not isinstance(
                description,
                str
            )
            or not description
            for reason_code, description in reason_codes.items()
        )
    ):
        raise RuntimeError("Identification policy contains invalid reason-code definitions.")

    for assumption_record in policy[
        "assumption_inventory"
    ]:
        if assumption_record[
            "reason_code"
        ] not in reason_codes:
            raise RuntimeError(
                f"Identification assumption uses an unregistered reason code : {assumption_record['reason_code']}"
            )

    amendment = require_exact_fields(
        value = policy[
            "amendment"
        ],
        required_fields = {
            "reason",
            "result_informed",
            "affected_analyses",
            "complete_reassessment_required"
        },
        description = "Identification-policy amendment"
    )

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
        or amendment[
            "result_informed"
        ] is not False
        or not isinstance(
            amendment[
                "result_informed"
            ],
            bool
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
            "complete_reassessment_required"
        ]
        is not True
    ):
        raise RuntimeError("Identification-policy amendment is invalid.")

    if policy[
        "registration"
    ].get(
        "result_informed"
    ) is not False:
        raise RuntimeError("Identification decisions must not be result-informed.")

    reason_code_rules = require_exact_fields(
        value = policy[
            "reason_code_rules"
        ],
        required_fields = {
            "universal_point_identification",
            "calendar_elapsed_time",
            "cycle_elapsed_time",
            "cycle_fec",
            "cross_stage_comparison",
            "equipment_restriction",
            "associational_model_not_authorized"
        },
        description = "Identification reason-code rules"
    )

    registered_reason_codes = set(
        reason_codes
    )

    validate_reason_code_sequence(
        reason_codes = reason_code_rules[
            "universal_point_identification"
        ],
        registered_reason_codes = (
            registered_reason_codes
        ),
        description = (
            "Universal point-identification reasons"
        )
    )

    for rule_name in [
        "calendar_elapsed_time",
        "cycle_elapsed_time",
        "cycle_fec",
        "cross_stage_comparison",
        "equipment_restriction",
        "associational_model_not_authorized"
    ]:
        scope_rule = require_exact_fields(
            value = reason_code_rules[
                rule_name
            ],
            required_fields = {
                "causal_reason_codes",
                "interpretation_restriction_codes"
            },
            description = (
                f"Identification reason rule {rule_name}"
            )
        )

        validate_reason_code_sequence(
            reason_codes = scope_rule[
                "causal_reason_codes"
            ],
            registered_reason_codes = (
                registered_reason_codes
            ),
            description = (
                f"Causal reasons for {rule_name}"
            )
        )

        validate_reason_code_sequence(
            reason_codes = scope_rule[
                "interpretation_restriction_codes"
            ],
            registered_reason_codes = (
                registered_reason_codes
            ),
            description = (
                f"Interpretation restrictions for {rule_name}"
            )
        )

    output_contract = require_exact_fields(
        value = policy[
            "output_contract"
        ],
        required_fields = {
            "report_file",
            "estimand_table",
            "assumption_table",
            "deterministic_ordering_required",
            "input_hashes_required",
            "table_hashes_required",
            "exact_table_schemas_required",
            "embedded_policy_required",
            "post_publication_validation_required",
            "design_evidence_only_for_identification_decisions"
        },
        description = "Identification output contract"
    )

    if (
        output_contract[
            "report_file"
        ]
        != "causal_identification_report.json"
        or output_contract[
            "estimand_table"
        ]
        != "estimand_identification.csv"
        or output_contract[
            "assumption_table"
        ]
        != "assumption_inventory.csv"
        or output_contract[
            "deterministic_ordering_required"
        ]
        is not True
        or output_contract[
            "input_hashes_required"
        ]
        is not True
        or output_contract[
            "table_hashes_required"
        ]
        is not True
        or output_contract[
            "exact_table_schemas_required"
        ]
        is not True
        or output_contract[
            "embedded_policy_required"
        ]
        is not True
        or output_contract[
            "post_publication_validation_required"
        ]
        is not True
        or output_contract[
            "design_evidence_only_for_identification_decisions"
        ]
        is not True
    ):
        raise RuntimeError("Identification output contract is invalid.")

    validated_assumption_rows = build_assumption_inventory(
        policy
    )

    for assumption_row in validated_assumption_rows:
        if not isinstance(
            assumption_row[
                "required_for_point_identification"
            ],
            bool
        ):
            raise RuntimeError("Identification assumption contains an invalid requirement flag.")

    identification_requirements = require_exact_fields(
        value = policy[
            "identification_requirements"
        ],
        required_fields = {
            "assignment_mechanism_required",
            "conditional_exchangeability_required_for_covariate_adjustment",
            "conditional_positivity_required",
            "consistency_and_well_defined_treatment_versions_required",
            "no_interference_required",
            "selection_process_identification_required",
            "outcome_measurement_assumptions_required",
            "cross_stage_transportability_requires_explicit_assumptions",
            "absence_of_evidence_never_satisfies_an_assumption",
            "bayesian_estimation_alone_establishes_identification"
        },
        description = "Identification requirements"
    )

    for requirement_name, requirement_value in identification_requirements.items():
        if not isinstance(
            requirement_value,
            bool
        ):
            raise RuntimeError(f"Identification requirement is not Boolean : {requirement_name}")

    if identification_requirements[
        "bayesian_estimation_alone_establishes_identification"
    ] is not False:
        raise RuntimeError("Identification requirements treat Bayesian estimation as identification.")

    assumptions_by_name = {
        assumption_row[
            "assumption_name"
        ]: assumption_row
        for assumption_row in (
            validated_assumption_rows
        )
    }

    if set(
        assumptions_by_name
    ) != set(
        EXPECTED_ASSUMPTION_EVIDENCE_STATUSES
    ):
        raise RuntimeError("Identification policy contains an unexpected assumption inventory.")

    for (
        assumption_name,
        expected_evidence_status
    ) in EXPECTED_ASSUMPTION_EVIDENCE_STATUSES.items():
        if (
            assumptions_by_name[
                assumption_name
            ][
                "evidence_status"
            ]
            != expected_evidence_status
        ):
            raise RuntimeError(f"Identification assumption has an inconsistent evidence status : {assumption_name}")

    validate_reason_rule_consistency(
        reason_code_rules = reason_code_rules,
        assumptions_by_name = assumptions_by_name
    )

    for (
        requirement_name,
        assumption_name
    ) in REQUIREMENT_ASSUMPTION_NAMES.items():
        requirement_enabled = (
            identification_requirements[
                requirement_name
            ]
        )

        assumption_record = (
            assumptions_by_name.get(
                assumption_name
            )
        )

        if assumption_record is None:
            raise RuntimeError(f"Identification requirement has no corresponding assumption record : {requirement_name}")

        if (
            requirement_enabled
            and assumption_record[
                "required_for_point_identification"
            ]
            is not True
        ):
            raise RuntimeError(f"Identification requirement conflicts with its assumption record : {requirement_name}")

    scope_specific_assumption_requirements = {
        "achieved_fec_horizon": True,
        "calendar_exposure_definition": False
    }

    for (
        assumption_name,
        expected_requirement
    ) in scope_specific_assumption_requirements.items():
        assumption_record = assumptions_by_name.get(
            assumption_name
        )

        if assumption_record is None:
            raise RuntimeError(f"Identification policy is missing a scope-specific assumption : {assumption_name}")

        if (
            assumption_record[
                "required_for_point_identification"
            ]
            is not expected_requirement
        ):
            raise RuntimeError(f"Scope-specific point-identification requirement is inconsistent : {assumption_name}")

    identification_strategies = require_exact_fields(
        value = policy[
            "identification_strategies_assessed"
        ],
        required_fields = set(
            EXPECTED_IDENTIFICATION_STRATEGIES
        ),
        description = (
            "Identification strategies assessed"
        )
    )

    if (
        identification_strategies
        != EXPECTED_IDENTIFICATION_STRATEGIES
    ):
        raise RuntimeError(
            "Identification-strategy declarations are inconsistent with the registered non-identification conclusion."
        )

    estimand_interpretations = require_exact_fields(
        value = policy[
            "estimand_interpretations"
        ],
        required_fields = {
            "calendar_elapsed_time",
            "cycle_elapsed_time",
            "cycle_fec"
        },
        description = "Estimand interpretations"
    )

    require_exact_fields(
        value = estimand_interpretations[
            "calendar_elapsed_time"
        ],
        required_fields = {
            "horizon_axis",
            "authorized_target",
            "exposure_time_interpretation",
            "horizon_conditioning_status",
            "net_storage_effect_authorized",
            "interpretation"
        },
        description = "Calendar estimand interpretation"
    )

    require_exact_fields(
        value = estimand_interpretations[
            "cycle_elapsed_time"
        ],
        required_fields = {
            "horizon_axis",
            "authorized_target",
            "exposure_time_interpretation",
            "horizon_conditioning_status",
            "interpretation"
        },
        description = "Cycle elapsed-time estimand interpretation"
    )

    require_exact_fields(
        value = estimand_interpretations[
            "cycle_fec"
        ],
        required_fields = {
            "horizon_axis",
            "authorized_target",
            "exposure_time_interpretation",
            "horizon_conditioning_status",
            "interpretation"
        },
        description = "Cycle FEC estimand interpretation"
    )

    expected_targets = {
        "calendar_elapsed_time": (
            "elapsed_calendar_days",
            "whole_assigned_protocol_outcome_difference_at_common_elapsed_calendar_time"
        ),
        "cycle_elapsed_time": (
            "elapsed_calendar_days",
            "outcome_difference_at_common_elapsed_calendar_days"
        ),
        "cycle_fec": (
            "cumulative_full_equivalent_cycles_observed",
            "outcome_difference_at_common_cumulative_full_equivalent_cycles_observed"
        )
    }

    for scope_name, expected_pair in expected_targets.items():
        observed_pair = (
            estimand_interpretations[
                scope_name
            ][
                "horizon_axis"
            ],
            estimand_interpretations[
                scope_name
            ][
                "authorized_target"
            ]
        )

        if observed_pair != expected_pair:
            raise RuntimeError(f"Estimand interpretation does not match the registered target : {scope_name}")

    if estimand_interpretations[
        "calendar_elapsed_time"
    ][
        "net_storage_effect_authorized"
    ] is not False:
        raise RuntimeError("Calendar estimand interpretation incorrectly authorizes net storage effect.")

    current_identification_status = require_exact_fields(
        value = policy[
            "current_identification_status"
        ],
        required_fields = set(
            EXPECTED_CURRENT_IDENTIFICATION_STATUS
        ),
        description = "Current identification status"
    )

    if (
        current_identification_status
        != EXPECTED_CURRENT_IDENTIFICATION_STATUS
    ):
        raise RuntimeError("Current identification declarations are internally inconsistent.")

    associational_authorization = require_exact_fields(
        value = policy[
            "model_authorization"
        ][
            "associational_hierarchical_modeling"
        ],
        required_fields = {
            "authorization_status",
            "authorized_estimand_status",
            "unauthorized_estimand_status",
            "permitted_support_classifications",
            "causal_language_permitted",
            "policy_language_permitted",
            "unobserved_condition_combinations_permitted",
            "cross_stage_pooling_permitted",
            "primary_and_sensitivity_cohorts_pooled"
        },
        description = "Associational hierarchical modeling authorization"
    )

    if associational_authorization[
        "authorization_status"
    ] != "partially_authorized":
        raise RuntimeError("Associational authorization status is unexpected.")

    if associational_authorization[
        "authorized_estimand_status"
    ] != "authorized_with_recorded_restrictions":
        raise RuntimeError("Associational authorized estimand status is unexpected.")

    if associational_authorization[
        "unauthorized_estimand_status"
    ] != "not_authorized":
        raise RuntimeError("Associational unauthorized estimand status is unexpected.")

    if associational_authorization[
        "permitted_support_classifications"
    ] != [
        "supported_with_restrictions",
        "sensitivity_only"
    ]:
        raise RuntimeError("Associational permitted support classifications are unexpected.")

    for forbidden_flag in [
        "causal_language_permitted",
        "policy_language_permitted",
        "unobserved_condition_combinations_permitted",
        "cross_stage_pooling_permitted",
        "primary_and_sensitivity_cohorts_pooled"
    ]:
        if associational_authorization[
            forbidden_flag
        ] is not False:
            raise RuntimeError(f"Associational authorization incorrectly enables : {forbidden_flag}")

    sensitivity_authorization = require_exact_fields(
        value = policy[
            "model_authorization"
        ][
            "causal_sensitivity_analysis"
        ],
        required_fields = {
            "authorization_status",
            "estimand_status",
            "activation_requirement",
            "result_label"
        },
        description = "Causal sensitivity analysis authorization"
    )

    if sensitivity_authorization[
        "authorization_status"
    ] != "not_authorized":
        raise RuntimeError("Causal sensitivity analysis incorrectly authorized.")

    if sensitivity_authorization[
        "estimand_status"
    ] != "not_authorized_without_registered_assumption_set":
        raise RuntimeError("Causal sensitivity estimand status is unexpected.")


# Validating the design report against the policy
def validate_design_report(design_report: dict, policy: dict) -> None:
    if design_report.get(
        "audit_status"
    ) != "completed":
        raise RuntimeError("Experimental-design report is not completed.")

    if design_report.get(
        "independent_experimental_unit"
    ) != "cell":
        raise RuntimeError("Experimental-design report does not identify the cell as the independent unit.")

    required_evidence = policy[
        "required_design_evidence"
    ]

    if design_report.get(
        "cell_to_condition_randomization_status"
    ) != required_evidence[
        "cell_to_condition_randomization_status"
    ]:
        raise RuntimeError("Cell-assignment evidence differs from the registered policy.")

    if design_report.get(
        "cross_stage_transportability_status"
    ) != required_evidence[
        "cross_stage_transportability_status"
    ]:
        raise RuntimeError("Cross-stage transportability evidence differs from the registered policy.")

    formal_handoff = design_report.get(
        "formal_handoff"
    )

    if not isinstance(
        formal_handoff,
        dict
    ):
        raise RuntimeError("Experimental-design report does not contain its formal handoff.")

    if formal_handoff.get(
        "causal_interpretation"
    ) != required_evidence[
        "causal_interpretation"
    ]:
        raise RuntimeError("Causal interpretation differs from the registered policy.")

    if formal_handoff.get(
        "calendar_net_storage_exposure_status"
    ) != required_evidence[
        "calendar_net_storage_exposure_status"
    ]:
        raise RuntimeError("Calendar-exposure evidence differs from the registered policy.")

    equipment_interpretation = formal_handoff.get(
        "equipment_allocation_interpretation"
    )

    if (
        not isinstance(
            equipment_interpretation,
            dict
        )
        or equipment_interpretation.get(
            "adjustment_status"
        )
        != required_evidence[
            "equipment_adjustment_status"
        ]
    ):
        raise RuntimeError("Equipment-adjustment evidence differs from the registered policy.")

    baseline_contract = formal_handoff.get(
        "baseline_covariate_contract"
    )

    if (
        not isinstance(
            baseline_contract,
            dict
        )
        or baseline_contract.get(
            "interpretation"
        )
        != required_evidence[
            "baseline_adjustment_status"
        ]
    ):
        raise RuntimeError("Baseline-adjustment evidence differs from the registered policy.")


# Building one deterministic estimand identity
def build_estimand_decision_identity(estimand_record: dict) -> str:
    identity_values = {
        field_name: estimand_record[
            field_name
        ]
        for field_name in ESTIMAND_IDENTITY_FIELDS
    }

    return hashlib.sha256(
        canonical_json(
            identity_values
        ).encode(
            "utf-8"
        )
    ).hexdigest()


# Reading the dataset stage from one condition identity
def condition_stage(condition_identity: str) -> str:
    stage_name = condition_identity.split(
        "/",
        maxsplit = 1
    )[0]

    if stage_name not in {
        "Stage_1",
        "Stage_2"
    }:
        raise RuntimeError(f"Condition has an invalid dataset stage : {condition_identity}")

    return stage_name


# Validating decision-driving upstream estimand fields
def validate_upstream_estimand_record(estimand_record: dict, policy: dict) -> None:
    contract = policy[
        "upstream_estimand_contract"
    ]

    categorical_fields = {
        "aging_mode": "aging_modes",
        "decision_scope": "decision_scopes",
        "outcome": "outcomes",
        "cohort": "cohorts",
        "final_support_classification": (
            "support_classifications"
        ),
        "causal_identification_status": (
            "causal_identification_statuses"
        ),
        "cross_stage_transportability_status": (
            "cross_stage_transportability_statuses"
        )
    }

    for (
        field_name,
        contract_field
    ) in categorical_fields.items():
        field_value = estimand_record.get(
            field_name
        )

        if field_value not in contract[
            contract_field
        ]:
            raise RuntimeError(f"Estimand contains an unsupported categorical value : {field_name} : {field_value!r}")

    aging_mode = estimand_record[
        "aging_mode"
    ]

    horizon_axis = estimand_record.get(
        "horizon_axis"
    )

    if horizon_axis not in contract[
        "horizon_axes_by_aging_mode"
    ][
        aging_mode
    ]:
        raise RuntimeError(
            f"Estimand contains an unsupported aging-mode and horizon-axis combination : {aging_mode} : {horizon_axis!r}"
        )

    if (
        estimand_record[
            "final_support_classification"
        ]
        == "sensitivity_only"
        and estimand_record[
            "cohort"
        ]
        not in contract[
            "sensitivity_only_permitted_cohorts"
        ]
    ):
        raise RuntimeError("Sensitivity-only support is assigned to an unsupported cohort.")


# Assessing identification for one estimand
def assess_estimand_record(estimand_record: dict, policy: dict) -> dict:
    validate_upstream_estimand_record(
        estimand_record = estimand_record,
        policy = policy
    )

    upstream_causal_status = estimand_record[
        "causal_identification_status"
    ]

    if upstream_causal_status != "not_established":
        raise RuntimeError(f"Estimand contains an unexpected causal-identification status : {upstream_causal_status}")

    target_estimand_available = parse_boolean(
        estimand_record[
            "target_estimand_available"
        ],
        "target_estimand_available"
    )

    final_support_classification = estimand_record[
        "final_support_classification"
    ]

    associational_policy = policy[
        "model_authorization"
    ][
        "associational_hierarchical_modeling"
    ]

    permitted_support = set(
        associational_policy[
            "permitted_support_classifications"
        ]
    )

    first_condition_identity = estimand_record[
        "first_condition_protocol_identity"
    ]

    second_condition_identity = estimand_record[
        "second_condition_protocol_identity"
    ]

    cross_stage_comparison = (
        condition_stage(
            first_condition_identity
        )
        != condition_stage(
            second_condition_identity
        )
    )

    support_authorized = (
        target_estimand_available
        and final_support_classification
        in permitted_support
    )

    stage_authorized = (
        not cross_stage_comparison
        or associational_policy[
            "cross_stage_pooling_permitted"
        ]
    )

    associational_model_status = (
        associational_policy[
            "authorized_estimand_status"
        ]
        if (
            support_authorized
            and stage_authorized
        )
        else associational_policy[
            "unauthorized_estimand_status"
        ]
    )

    upstream_transportability_status = estimand_record[
        "cross_stage_transportability_status"
    ]

    if not upstream_transportability_status:
        raise RuntimeError("Estimand does not record a cross-stage transportability status.")

    equipment_disjoint_count = parse_nonnegative_integer(
        estimand_record[
            "equipment_disjoint_dimension_count"
        ],
        "equipment_disjoint_dimension_count"
    )

    equipment_conflicted_count = parse_nonnegative_integer(
        estimand_record[
            "equipment_identity_conflicted_dimension_count"
        ],
        "equipment_identity_conflicted_dimension_count"
    )

    equipment_restriction_present = (
        equipment_disjoint_count > 0
        or equipment_conflicted_count > 0
    )

    reason_code_rules = policy[
        "reason_code_rules"
    ]

    identification_reason_codes = list(
        reason_code_rules[
            "universal_point_identification"
        ]
    )

    interpretation_restriction_codes = []

    aging_mode = estimand_record[
        "aging_mode"
    ]

    horizon_axis = estimand_record[
        "horizon_axis"
    ]

    if aging_mode == "calendar_aging":
        scope_rule_name = "calendar_elapsed_time"

    elif (
        aging_mode == "cycle_aging"
        and horizon_axis == "elapsed_calendar_days"
    ):
        scope_rule_name = "cycle_elapsed_time"

    elif (
        aging_mode == "cycle_aging"
        and horizon_axis
        == "cumulative_full_equivalent_cycles_observed"
    ):
        scope_rule_name = "cycle_fec"

    else:
        raise RuntimeError("Estimand contains an unsupported aging-mode and horizon-axis combination.")

    estimand_interpretation = policy[
        "estimand_interpretations"
    ][
        scope_rule_name
    ]

    if (
        horizon_axis
        != estimand_interpretation[
            "horizon_axis"
        ]
    ):
        raise RuntimeError("Estimand horizon axis differs from the registered interpretation.")

    if (
        estimand_record[
            "target_estimand_identity"
        ]
        != estimand_interpretation[
            "authorized_target"
        ]
    ):
        raise RuntimeError("Estimand identity differs from the registered interpretation.")

    exposure_time_interpretation = (
        estimand_interpretation[
            "exposure_time_interpretation"
        ]
    )

    horizon_conditioning_status = (
        estimand_interpretation[
            "horizon_conditioning_status"
        ]
    )

    scope_rule = reason_code_rules[
        scope_rule_name
    ]

    identification_reason_codes.extend(
        scope_rule[
            "causal_reason_codes"
        ]
    )

    interpretation_restriction_codes.extend(
        scope_rule[
            "interpretation_restriction_codes"
        ]
    )

    if cross_stage_comparison:
        cross_stage_rule = reason_code_rules[
            "cross_stage_comparison"
        ]

        identification_reason_codes.extend(
            cross_stage_rule[
                "causal_reason_codes"
            ]
        )

        interpretation_restriction_codes.extend(
            cross_stage_rule[
                "interpretation_restriction_codes"
            ]
        )

    if equipment_restriction_present:
        equipment_rule = reason_code_rules[
            "equipment_restriction"
        ]

        identification_reason_codes.extend(
            equipment_rule[
                "causal_reason_codes"
            ]
        )

        interpretation_restriction_codes.extend(
            equipment_rule[
                "interpretation_restriction_codes"
            ]
        )

    if associational_model_status == "not_authorized":
        support_rule = reason_code_rules[
            "associational_model_not_authorized"
        ]

        identification_reason_codes.extend(
            support_rule[
                "causal_reason_codes"
            ]
        )

        interpretation_restriction_codes.extend(
            support_rule[
                "interpretation_restriction_codes"
            ]
        )

    registered_reason_codes = set(
        policy[
            "reason_codes"
        ]
    )

    emitted_reason_codes = {
        *identification_reason_codes,
        *interpretation_restriction_codes
    }

    unknown_reason_codes = (
        emitted_reason_codes
        - registered_reason_codes
    )

    if unknown_reason_codes:
        raise RuntimeError(f"Identification assessment emitted unregistered reason codes : {sorted(unknown_reason_codes)}")

    return {
        "estimand_decision_identity": (
            build_estimand_decision_identity(
                estimand_record
            )
        ),
        "aging_mode": aging_mode,
        "decision_scope": estimand_record[
            "decision_scope"
        ],
        "first_condition_protocol_identity": (
            first_condition_identity
        ),
        "second_condition_protocol_identity": (
            second_condition_identity
        ),
        "outcome": estimand_record[
            "outcome"
        ],
        "cohort": estimand_record[
            "cohort"
        ],
        "horizon_axis": horizon_axis,
        "target_estimand_identity": estimand_record[
            "target_estimand_identity"
        ],
        "condition_comparison_interpretation": (
            estimand_record[
                "condition_comparison_interpretation"
            ]
        ),
        "final_support_classification": (
            final_support_classification
        ),
        "target_estimand_available": (
            target_estimand_available
        ),
        "upstream_causal_identification_status": (
            upstream_causal_status
        ),
        "point_identification_status": (
            policy[
                "current_identification_status"
            ][
                "causal_point_identification"
            ]
        ),
        "identification_strategy_status": (
            policy[
                "identification_strategies_assessed"
            ][
                "overall_status"
            ]
        ),
        "causal_identification_reason_codes": sorted(
            set(
                identification_reason_codes
            )
        ),
        "interpretation_restriction_codes": sorted(
            set(
                interpretation_restriction_codes
            )
        ),
        "associational_model_status": (
            associational_model_status
        ),
        "causal_sensitivity_status": (
            policy[
                "model_authorization"
            ][
                "causal_sensitivity_analysis"
            ][
                "estimand_status"
            ]
        ),
        "exposure_time_interpretation": (
            exposure_time_interpretation
        ),
        "horizon_conditioning_status": (
            horizon_conditioning_status
        ),
        "cross_stage_comparison": (
            cross_stage_comparison
        ),
        "equipment_restriction_present": (
            equipment_restriction_present
        )
    }


# Building the registered identification-assumption inventory
def build_assumption_inventory(policy: dict) -> list[dict]:
    assumption_rows = policy[
        "assumption_inventory"
    ]

    if (
        not isinstance(
            assumption_rows,
            list
        )
        or not assumption_rows
    ):
        raise RuntimeError("Identification policy does not contain an assumption inventory.")

    observed_assumption_names = set()

    validated_rows = []

    for assumption_row in assumption_rows:
        if (
            not isinstance(
                assumption_row,
                dict
            )
            or set(
                assumption_row
            )
            != set(
                ASSUMPTION_OUTPUT_COLUMNS
            )
        ):
            raise RuntimeError("Identification-policy assumption does not match the output schema.")

        assumption_name = assumption_row[
            "assumption_name"
        ]

        if assumption_name in observed_assumption_names:
            raise RuntimeError(f"Identification policy contains a duplicate assumption : {assumption_name}")

        observed_assumption_names.add(
            assumption_name
        )

        validated_rows.append(
            dict(
                assumption_row
            )
        )

    return sorted(
        validated_rows,
        key = lambda record: (
            record[
                "assumption_scope"
            ],
            record[
                "assumption_name"
            ]
        )
    )


# Serializing one CSV value
def serialize_csv_value(value: object) -> object:
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
def write_csv_table(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open(
        mode = "w",
        encoding = "utf-8",
        newline = ""
    ) as output_stream:
        writer = csv.DictWriter(
            output_stream,
            fieldnames = columns,
            lineterminator = "\n"
        )

        writer.writeheader()

        for row in rows:
            if set(
                row
            ) != set(
                columns
            ):
                raise RuntimeError(f"Output row does not match its schema : {path.name}")

            writer.writerow(
                {
                    column_name: serialize_csv_value(
                        row[
                            column_name
                        ]
                    )
                    for column_name in columns
                }
            )


# Building one table manifest record
def build_table_manifest(path: Path, columns: list[str], row_count: int) -> dict:
    return {
        "file_name": path.name,
        "row_count": row_count,
        "sha256": calculate_file_sha256(
            path
        ),
        "schema": [
            {
                "name": column_name
            }
            for column_name in columns
        ]
    }


# Publishing staged artifacts transactionally with optional post-publication validation
def publish_outputs(output_directory: Path, files: dict[str, Path],
                    final_validator: Callable[[Path], None] | None = None
) -> None:
    output_directory.parent.mkdir(
        parents = True,
        exist_ok = True
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
        files[
            "staging_directory"
        ].replace(
            output_directory
        )

    except Exception:
        if (
            backup_directory is not None
            and backup_directory.exists()
            and not output_directory.exists()
        ):
            backup_directory.replace(
                output_directory
            )

        raise

    if final_validator is not None:
        try:
            final_validator(
                output_directory
            )

        except Exception:
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


# Validating one completed identification report
def validate_identification_report(report_path: Path, expected_policy: dict,
                                   expected_policy_sha256: str, expected_design_report_sha256: str,
                                   expected_estimand_rows: list[dict], expected_assumption_rows: list[dict]
) -> None:
    report = load_json_object(
        report_path
    )

    observed_report_fields = set(
        report
    )

    if observed_report_fields != (
        IDENTIFICATION_REPORT_FIELDS
    ):
        raise RuntimeError(
            "Identification-report fields differ from the contract : "
            f"missing {sorted(IDENTIFICATION_REPORT_FIELDS - observed_report_fields)} : "
            f"unexpected {sorted(observed_report_fields - IDENTIFICATION_REPORT_FIELDS)}"
        )

    if (
        report.get(
            "policy_version"
        )
        != expected_policy[
            "policy_version"
        ]
        or report.get(
            "policy_effective_date"
        )
        != expected_policy[
            "effective_date"
        ]
    ):
        raise RuntimeError("Identification report has inconsistent policy metadata.")

    if (
        report.get(
            "independent_unit"
        )
        != expected_policy[
            "independent_unit"
        ]
    ):
        raise RuntimeError("Identification report has an inconsistent independent unit.")

    if (
        report.get(
            "identification_decision_basis"
        )
        != "registered_policy_and_validated_design_evidence_only"
    ):
        raise RuntimeError("Identification report has an invalid decision basis.")

    if (
        report.get(
            "outcome_values_used_for_identification_decisions"
        )
        is not False
        or report.get(
            "predictive_performance_used_for_identification_decisions"
        )
        is not False
    ):
        raise RuntimeError("Identification report declares outcome-informed decisions.")

    expected_causal_status = expected_policy[
        "model_authorization"
    ][
        "causal_modeling"
    ][
        "authorization_status"
    ]

    if (
        report.get(
            "causal_modeling_status"
        )
        != expected_causal_status
    ):
        raise RuntimeError("Identification report has an inconsistent causal-modeling status.")

    expected_sensitivity_status = expected_policy[
        "model_authorization"
    ][
        "causal_sensitivity_analysis"
    ][
        "authorization_status"
    ]

    if (
        report.get(
            "causal_sensitivity_analysis_status"
        )
        != expected_sensitivity_status
    ):
        raise RuntimeError("Identification report has an inconsistent causal-sensitivity status.")

    expected_activation_requirement = expected_policy[
        "model_authorization"
    ][
        "causal_sensitivity_analysis"
    ][
        "activation_requirement"
    ]

    if (
        report.get(
            "causal_sensitivity_activation_requirement"
        )
        != expected_activation_requirement
    ):
        raise RuntimeError("Identification report has an inconsistent sensitivity activation requirement.")

    if set(
        report.get(
            "tables",
            {}
        )
    ) != {
        "assumption_inventory",
        "estimand_identification"
    }:
        raise RuntimeError("Identification report has an unexpected table inventory.")

    if report.get(
        "assessment_status"
    ) != "completed":
        raise RuntimeError("Identification report is not completed.")

    if report.get(
        "policy"
    ) != expected_policy:
        raise RuntimeError("Identification report does not preserve the registered policy.")

    if report.get(
        "policy_sha256"
    ) != expected_policy_sha256:
        raise RuntimeError("Identification report has an unexpected policy digest.")

    if report.get(
        "design_report_sha256"
    ) != expected_design_report_sha256:
        raise RuntimeError("Identification report has an unexpected design-report digest.")

    if report.get(
        "causal_modeling_status"
    ) != "not_authorized":
        raise RuntimeError("Identification report unexpectedly authorizes causal modeling.")

    _, estimand_rows = verify_report_table(
        report_path = report_path,
        report = report,
        table_name = "estimand_identification"
    )

    _, assumption_rows = verify_report_table(
        report_path = report_path,
        report = report,
        table_name = "assumption_inventory"
    )

    if len(
        estimand_rows
    ) != report.get(
        "estimand_count"
    ):
        raise RuntimeError("Identification estimand count does not reconcile.")

    if len(
        assumption_rows
    ) != report.get(
        "assumption_count"
    ):
        raise RuntimeError("Identification assumption count does not reconcile.")

    expected_serialized_estimands = [
        {
            column_name: str(
                serialize_csv_value(
                    row[
                        column_name
                    ]
                )
            )
            for column_name in ESTIMAND_OUTPUT_COLUMNS
        }
        for row in expected_estimand_rows
    ]

    if estimand_rows != expected_serialized_estimands:
        raise RuntimeError("Identification estimand table differs from the expected assessment.")

    expected_serialized_assumptions = [
        {
            column_name: str(
                serialize_csv_value(
                    row[
                        column_name
                    ]
                )
            )
            for column_name in ASSUMPTION_OUTPUT_COLUMNS
        }
        for row in expected_assumption_rows
    ]

    if assumption_rows != expected_serialized_assumptions:
        raise RuntimeError("Identification assumption table differs from the registered inventory.")

    point_status_counts = {}

    associational_status_counts = {}

    observed_identities = set()

    for estimand_row in estimand_rows:
        estimand_identity = estimand_row[
            "estimand_decision_identity"
        ]

        if estimand_identity in observed_identities:
            raise RuntimeError("Identification report contains a duplicate estimand.")

        observed_identities.add(
            estimand_identity
        )

        reconstructed_identity = build_estimand_decision_identity(
            estimand_row
        )

        if estimand_identity != reconstructed_identity:
            raise RuntimeError("Identification estimand identity is invalid.")

        point_status = estimand_row[
            "point_identification_status"
        ]

        associational_status = estimand_row[
            "associational_model_status"
        ]

        point_status_counts[
            point_status
        ] = point_status_counts.get(
            point_status,
            0
        ) + 1

        associational_status_counts[
            associational_status
        ] = associational_status_counts.get(
            associational_status,
            0
        ) + 1

    if dict(
        sorted(
            point_status_counts.items()
        )
    ) != report.get(
        "point_identification_status_counts"
    ):
        raise RuntimeError("Point-identification summaries do not reconcile.")

    if dict(
        sorted(
            associational_status_counts.items()
        )
    ) != report.get(
        "associational_model_status_counts"
    ):
        raise RuntimeError("Associational-model summaries do not reconcile.")

    authorized_associational_count = (
        associational_status_counts.get(
            "authorized_with_recorded_restrictions",
            0
        )
    )

    if authorized_associational_count == 0:
        expected_associational_status = (
            "not_authorized"
        )

    elif authorized_associational_count == len(
        estimand_rows
    ):
        expected_associational_status = (
            "fully_authorized_with_recorded_restrictions"
        )

    else:
        expected_associational_status = (
            "partially_authorized"
        )

    if (
        report.get(
            "associational_hierarchical_modeling_status"
        )
        != expected_associational_status
    ):
        raise RuntimeError("Associational-modeling authorization summary is inconsistent.")


# Running the causal-identification assessment
def main() -> None:
    args = parse_args()

    policy = load_json_object(
        args.policy_path
    )

    validate_policy(
        policy
    )

    design_report = load_json_object(
        args.design_report_path
    )

    validate_design_report(
        design_report = design_report,
        policy = policy
    )

    _, estimand_rows = verify_report_table(
        report_path = args.design_report_path,
        report = design_report,
        table_name = "estimand_support"
    )

    estimand_identification_rows = sorted(
        (
            assess_estimand_record(
                estimand_record = estimand_record,
                policy = policy
            )
            for estimand_record in estimand_rows
        ),
        key = lambda record: record[
            "estimand_decision_identity"
        ]
    )

    observed_identities = [
        record[
            "estimand_decision_identity"
        ]
        for record in estimand_identification_rows
    ]

    if len(
        observed_identities
    ) != len(
        set(
            observed_identities
        )
    ):
        raise RuntimeError("Identification assessment contains duplicate estimand identities.")

    assumption_rows = build_assumption_inventory(
        policy
    )

    args.output_directory.parent.mkdir(
        parents = True,
        exist_ok = True
    )

    staging_directory = Path(
        tempfile.mkdtemp(
            prefix = (
                f".{args.output_directory.name}."
            ),
            dir = args.output_directory.parent
        )
    )

    estimand_path = (
        staging_directory
        / policy[
            "output_contract"
        ][
            "estimand_table"
        ]
    )

    assumption_path = (
        staging_directory
        / policy[
            "output_contract"
        ][
            "assumption_table"
        ]
    )

    report_path = (
        staging_directory
        / policy[
            "output_contract"
        ][
            "report_file"
        ]
    )

    write_csv_table(
        path = estimand_path,
        columns = ESTIMAND_OUTPUT_COLUMNS,
        rows = estimand_identification_rows
    )

    write_csv_table(
        path = assumption_path,
        columns = ASSUMPTION_OUTPUT_COLUMNS,
        rows = assumption_rows
    )

    point_identification_status_counts = {}

    associational_status_counts = {}

    for record in estimand_identification_rows:
        point_identification_status = record[
            "point_identification_status"
        ]

        associational_status = record[
            "associational_model_status"
        ]

        point_identification_status_counts[
            point_identification_status
        ] = point_identification_status_counts.get(
            point_identification_status,
            0
        ) + 1

        associational_status_counts[
            associational_status
        ] = associational_status_counts.get(
            associational_status,
            0
        ) + 1

    if (
        sum(
            point_identification_status_counts.values()
        )
        != len(
            estimand_identification_rows
        )
    ):
        raise RuntimeError("Point-identification status counts do not reconcile to the estimand inventory.")

    if (
        sum(
            associational_status_counts.values()
        )
        != len(
            estimand_identification_rows
        )
    ):
        raise RuntimeError("Associational-status counts do not reconcile to the estimand inventory.")

    if (
        len(
            estimand_identification_rows
        )
        != len(
            estimand_rows
        )
    ):
        raise RuntimeError("Identification records do not reconcile to the design estimand inventory.")

    authorized_associational_count = (
        associational_status_counts.get(
            "authorized_with_recorded_restrictions",
            0
        )
    )

    if authorized_associational_count == 0:
        associational_authorization_status = (
            "not_authorized"
        )

    elif authorized_associational_count == len(
        estimand_identification_rows
    ):
        associational_authorization_status = (
            "fully_authorized_with_recorded_restrictions"
        )

    else:
        associational_authorization_status = (
            "partially_authorized"
        )

    registration = policy[
        "registration"
    ]

    model_authorization = policy[
        "model_authorization"
    ]

    causal_modeling_status = (
        model_authorization[
            "causal_modeling"
        ][
            "authorization_status"
        ]
    )

    causal_sensitivity_status = (
        model_authorization[
            "causal_sensitivity_analysis"
        ][
            "authorization_status"
        ]
    )

    report = {
        "assessment_status": "completed",
        "policy_version": policy[
            "policy_version"
        ],
        "policy_effective_date": policy[
            "effective_date"
        ],
        "policy_sha256": calculate_file_sha256(
            args.policy_path
        ),
        "policy": policy,
        "design_report_sha256": calculate_file_sha256(
            args.design_report_path
        ),
        "independent_unit": policy[
            "independent_unit"
        ],
        "identification_decision_basis": (
            "registered_policy_and_validated_design_evidence_only"
        ),
        "outcome_values_used_for_identification_decisions": (
            registration[
                "outcome_values_used_for_identification_decisions"
            ]
        ),
        "predictive_performance_used_for_identification_decisions": (
            registration[
                "predictive_performance_used_for_identification_decisions"
            ]
        ),
        "causal_modeling_status": causal_modeling_status,
        "associational_hierarchical_modeling_status": (
            associational_authorization_status
        ),
        "causal_sensitivity_analysis_status": (
            causal_sensitivity_status
        ),
        "causal_sensitivity_activation_requirement": (
            policy[
                "model_authorization"
            ][
                "causal_sensitivity_analysis"
            ][
                "activation_requirement"
            ]
        ),
        "estimand_count": len(
            estimand_identification_rows
        ),
        "assumption_count": len(
            assumption_rows
        ),
        "point_identification_status_counts": dict(
            sorted(
                point_identification_status_counts.items()
            )
        ),
        "associational_model_status_counts": dict(
            sorted(
                associational_status_counts.items()
            )
        ),
        "tables": {
            "assumption_inventory": build_table_manifest(
                path = assumption_path,
                columns = ASSUMPTION_OUTPUT_COLUMNS,
                row_count = len(
                    assumption_rows
                )
            ),
            "estimand_identification": build_table_manifest(
                path = estimand_path,
                columns = ESTIMAND_OUTPUT_COLUMNS,
                row_count = len(
                    estimand_identification_rows
                )
            )
        }
    }

    report_path.write_text(
        json.dumps(
            report,
            indent = 2,
            sort_keys = True,
            ensure_ascii = False
        )
        + "\n",
        encoding = "utf-8"
    )

    expected_policy_sha256 = calculate_file_sha256(
        args.policy_path
    )

    expected_design_report_sha256 = calculate_file_sha256(
        args.design_report_path
    )

    validate_identification_report(
        report_path = report_path,
        expected_policy = policy,
        expected_policy_sha256 = expected_policy_sha256,
        expected_design_report_sha256 = (
            expected_design_report_sha256
        ),
        expected_estimand_rows = (
            estimand_identification_rows
        ),
        expected_assumption_rows = assumption_rows
    )

    def validate_published_directory(published_directory: Path) -> None:
        validate_identification_report(
            report_path = (
                published_directory
                / policy[
                    "output_contract"
                ][
                    "report_file"
                ]
            ),
            expected_policy = policy,
            expected_policy_sha256 = (
                expected_policy_sha256
            ),
            expected_design_report_sha256 = (
                expected_design_report_sha256
            ),
            expected_estimand_rows = (
                estimand_identification_rows
            ),
            expected_assumption_rows = (
                assumption_rows
            )
        )

    publish_outputs(
        output_directory = args.output_directory,
        files = {
            "staging_directory": staging_directory
        },
        final_validator = validate_published_directory
    )

    print(
        "Causal-identification assessment finished."
    )

    print(
        "Estimand count :",
        len(
            estimand_identification_rows
        )
    )

    print(
        "Causal modeling authorized :",
        causal_modeling_status == "authorized"
    )

    print(
        "Associational hierarchical modeling status :",
        associational_authorization_status
    )

    print(
        "Identification report :",
        args.output_directory
        / policy[
            "output_contract"
        ][
            "report_file"
        ]
    )


if __name__ == "__main__":
    main()
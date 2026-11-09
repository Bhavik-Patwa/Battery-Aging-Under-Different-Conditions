import csv
import hashlib
import json

from pathlib import Path

import pandas as pd


REQUIRED_NAVIGATION = [
    "Evidence boundaries",
    "Experimental support",
    "Trajectories",
    "Predictive evidence",
    "Robustness",
    "Provenance"
]

EXPECTED_SCIENTIFIC_SCOPE = {
    "causal_interpretation_authorized": False,
    "policy_recommendation_authorized": False,
    "treatment_optimization_authorized": False,
    "novel_condition_prediction_authorized": False,
    "discrete_point_contrast_reporting_authorized": False,
    "continuous_supported_trajectory_reporting_authorized": True,
    "predictive_reporting_authorized": True,
    "restricted_robustness_reporting_authorized": True,
    "model_fitting_in_application_authorized": False,
    "artifact_mutation_authorized": False
}

EXPECTED_PRESENTATION_VALUES = {
    "application_role": "read_only_evidence_reporting",
    "independent_experimental_unit": "cell",
    "required_estimate_labels": [
        "descriptive",
        "associational",
        "predictive"
    ],
    "required_navigation": REQUIRED_NAVIGATION,
    "unsupported_results_remain_visible": True,
    "diagnostic_only_results_remain_labelled": True,
    "downloads_are_limited_to_displayed_summary_records": True
}

EXPECTED_RELEASE_CONTRACT = {
    "artifact_root_environment_variable": "EVIDENCE_ARTIFACT_ROOT",
    "policy_path_environment_variable": "EVIDENCE_DASHBOARD_POLICY",
    "default_artifact_root": "data/processed",
    "default_policy_path": "configs/evidence_dashboard_policy.json",
    "fail_closed_on_hash_mismatch": True,
    "fail_closed_on_schema_mismatch": True,
    "fail_closed_on_row_count_mismatch": True,
    "network_access_required": False
}


class ArtifactValidationError(RuntimeError):
    pass


# Calculating the SHA-256 digest of one artifact
def calculate_file_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


# Resolving one policy-declared path without permitting traversal
def resolve_relative_path(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = (
        resolved_root
        / relative_path
    ).resolve()

    if (
        resolved_path != resolved_root
        and resolved_root not in resolved_path.parents
    ):
        raise ArtifactValidationError(f"Artifact path escapes its declared root : {relative_path}")

    return resolved_path


# Recognizing a canonical lowercase SHA-256 digest
def is_sha256(value: object) -> bool:
    return (
        isinstance(
            value,
            str
        )
        and len(
            value
        ) == 64
        and all(
            character in "0123456789abcdef"
            for character in value
        )
    )


# Converting selected columns without changing unavailable values
def convert_numeric_columns(frame: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    converted = frame.copy()

    for column_name in column_names:
        if column_name in converted.columns:
            converted[
                column_name
            ] = pd.to_numeric(
                converted[
                    column_name
                ],
                errors = "coerce"
            )

    return converted


# Applying exact categorical filters
def filter_records(frame: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    filtered = frame.copy()

    for column_name, accepted_values in filters.items():
        if column_name not in filtered.columns:
            raise ArtifactValidationError(f"Requested filter column is unavailable : {column_name}")

        filtered = filtered[
            filtered[
                column_name
            ].isin(
                accepted_values
            )
        ]

    return filtered.reset_index(
        drop = True
    )


# Validating the immutable presentation policy
def validate_policy(policy: dict) -> None:
    if not isinstance(
        policy,
        dict
    ):
        raise ArtifactValidationError("Evidence-dashboard policy is not a JSON object.")

    if policy.get("policy_name") != "evidence_dashboard_policy":
        raise ArtifactValidationError("Unexpected evidence-dashboard policy name.")

    if (
        not isinstance(
            policy.get("policy_version"),
            str
        )
        or not policy[
            "policy_version"
        ]
    ):
        raise ArtifactValidationError("Evidence-dashboard policy version is missing.")

    if (
        not isinstance(
            policy.get("effective_date"),
            str
        )
        or not policy[
            "effective_date"
        ]
    ):
        raise ArtifactValidationError("Evidence-dashboard policy effective date is missing.")

    if policy.get(
        "registration_status"
    ) != "prospective_presentation_contract":
        raise ArtifactValidationError("Evidence-dashboard registration status is invalid.")

    if policy.get(
        "scientific_scope"
    ) != EXPECTED_SCIENTIFIC_SCOPE:
        raise ArtifactValidationError("Evidence-dashboard scientific scope does not match the registered restrictions.")

    presentation_contract = policy.get(
        "presentation_contract"
    )

    if not isinstance(
        presentation_contract,
        dict
    ):
        raise ArtifactValidationError("Evidence-dashboard presentation contract is missing.")

    if (
        not isinstance(
            presentation_contract.get(
                "application_title"
            ),
            str
        )
        or not presentation_contract[
            "application_title"
        ]
    ):
        raise ArtifactValidationError("Evidence-dashboard application title is missing.")

    for field_name, expected_value in EXPECTED_PRESENTATION_VALUES.items():
        if presentation_contract.get(
            field_name
        ) != expected_value:
            raise ArtifactValidationError(f"Evidence-dashboard presentation declaration is invalid : {field_name}")

    inputs = policy.get(
        "inputs"
    )

    if not isinstance(
        inputs,
        dict
    ) or not inputs:
        raise ArtifactValidationError("Evidence-dashboard input inventory is missing.")

    for source_name, source_contract in inputs.items():
        if not isinstance(
            source_contract,
            dict
        ):
            raise ArtifactValidationError(f"Evidence source contract is invalid : {source_name}")

        report_path = source_contract.get(
            "report_path"
        )

        if (
            not isinstance(
                report_path,
                str
            )
            or not report_path
            or Path(
                report_path
            ).is_absolute()
        ):
            raise ArtifactValidationError(f"Evidence source report path is invalid : {source_name}")

        if not is_sha256(
            source_contract.get(
                "report_sha256"
            )
        ):
            raise ArtifactValidationError(f"Evidence source report digest is invalid : {source_name}")

        required_values = source_contract.get(
            "required_values"
        )

        if (
            not isinstance(
                required_values,
                dict
            )
            or not required_values
        ):
            raise ArtifactValidationError(f"Evidence source required-value contract is invalid : {source_name}")

        table_names = source_contract.get(
            "tables"
        )

        if (
            not isinstance(
                table_names,
                list
            )
            or not all(
                isinstance(
                    table_name,
                    str
                )
                and table_name
                for table_name in table_names
            )
            or len(
                table_names
            ) != len(
                set(
                    table_names
                )
            )
        ):
            raise ArtifactValidationError(f"Evidence source table inventory is invalid : {source_name}")

    if policy.get(
        "release_contract"
    ) != EXPECTED_RELEASE_CONTRACT:
        raise ArtifactValidationError("Evidence-dashboard release contract does not match the registered fail-closed behavior.")


# Validating one report-declared table manifest before using it
def validate_table_manifest(source_name: str, table_name: str, table_manifest: object) -> dict:
    if not isinstance(
        table_manifest,
        dict
    ):
        raise ArtifactValidationError(f"Registered table manifest is invalid : {source_name} : {table_name}")

    file_name = table_manifest.get(
        "file_name"
    )

    if (
        not isinstance(
            file_name,
            str
        )
        or not file_name
        or Path(
            file_name
        ).is_absolute()
    ):
        raise ArtifactValidationError(f"Registered table file name is invalid : {source_name} : {table_name}")

    if not is_sha256(
        table_manifest.get(
            "sha256"
        )
    ):
        raise ArtifactValidationError(f"Registered table digest is invalid : {source_name} : {table_name}")

    row_count = table_manifest.get(
        "row_count"
    )

    if (
        isinstance(
            row_count,
            bool
        )
        or not isinstance(
            row_count,
            int
        )
        or row_count < 0
    ):
        raise ArtifactValidationError(f"Registered table row count is invalid : {source_name} : {table_name}")

    schema = table_manifest.get(
        "schema"
    )

    if (
        not isinstance(
            schema,
            list
        )
        or not schema
    ):
        raise ArtifactValidationError(f"Registered table schema is invalid : {source_name} : {table_name}")

    field_names = []

    for field in schema:
        if (
            not isinstance(
                field,
                dict
            )
            or not isinstance(
                field.get(
                    "name"
                ),
                str
            )
            or not field[
                "name"
            ]
        ):
            raise ArtifactValidationError(f"Registered table schema field is invalid : {source_name} : {table_name}")

        field_names.append(
            field[
                "name"
            ]
        )

    if len(
        field_names
    ) != len(
        set(
            field_names
        )
    ):
        raise ArtifactValidationError(f"Registered table schema contains duplicate fields : {source_name} : {table_name}")

    return table_manifest


class ArtifactStore:
    # Loading the presentation contract
    def __init__(
        self,
        artifact_root: Path,
        policy_path: Path
    ):
        self.artifact_root = artifact_root.resolve()
        self.policy_path = policy_path.resolve()

        if not self.policy_path.is_file():
            raise ArtifactValidationError(f"Evidence-dashboard policy does not exist : {self.policy_path}")

        self.policy = json.loads(
            self.policy_path.read_text(
                encoding = "utf-8"
            )
        )

        validate_policy(
            self.policy
        )

        self._reports = {}

    # Loading one validated evidence report
    def report(self, source_name: str) -> dict:
        if source_name in self._reports:
            return self._reports[
                source_name
            ]

        source_contract = self.policy[
            "inputs"
        ].get(
            source_name
        )

        if source_contract is None:
            raise ArtifactValidationError(f"Unknown evidence source : {source_name}")

        report_path = resolve_relative_path(
            self.artifact_root,
            source_contract[
                "report_path"
            ]
        )

        if not report_path.is_file():
            raise ArtifactValidationError(f"Required evidence report does not exist : {report_path}")

        observed_sha256 = calculate_file_sha256(
            report_path
        )

        if observed_sha256 != source_contract[
            "report_sha256"
        ]:
            raise ArtifactValidationError(f"Evidence report digest mismatch : {source_name}")

        report = json.loads(
            report_path.read_text(
                encoding = "utf-8"
            )
        )

        if not isinstance(
            report,
            dict
        ):
            raise ArtifactValidationError(f"Evidence report is not a JSON object : {source_name}")

        for field_name, expected_value in source_contract[
            "required_values"
        ].items():
            if report.get(
                field_name
            ) != expected_value:
                raise ArtifactValidationError(f"Evidence report declaration mismatch : {source_name} : {field_name}")

        self._reports[
            source_name
        ] = report

        return report

    # Validating one registered table
    def validate_table(self, source_name: str, table_name: str) -> Path:
        source_contract = self.policy[
            "inputs"
        ][
            source_name
        ]

        if table_name not in source_contract[
            "tables"
        ]:
            raise ArtifactValidationError(f"Table is not registered for presentation : {source_name} : {table_name}")

        report = self.report(
            source_name
        )
        report_tables = report.get(
            "tables"
        )

        if not isinstance(
            report_tables,
            dict
        ):
            raise ArtifactValidationError(f"Evidence report table inventory is invalid : {source_name}")

        if table_name not in report_tables:
            raise ArtifactValidationError(f"Registered table is absent from its source report : {source_name} : {table_name}")

        table_manifest = validate_table_manifest(
            source_name = source_name,
            table_name = table_name,
            table_manifest = report_tables[
                table_name
            ]
        )

        report_path = resolve_relative_path(
            self.artifact_root,
            source_contract[
                "report_path"
            ]
        )

        table_path = resolve_relative_path(
            report_path.parent,
            table_manifest[
                "file_name"
            ]
        )

        if not table_path.is_file():
            raise ArtifactValidationError(f"Registered evidence table does not exist : {table_path}")

        if calculate_file_sha256(
            table_path
        ) != table_manifest[
            "sha256"
        ]:
            raise ArtifactValidationError(f"Evidence table digest mismatch : {source_name} : {table_name}")

        expected_columns = [
            field[
                "name"
            ]
            for field in table_manifest[
                "schema"
            ]
        ]

        with table_path.open(
            mode = "r",
            encoding = "utf-8",
            newline = ""
        ) as table_stream:
            table_reader = csv.reader(
                table_stream
            )

            try:
                observed_columns = next(
                    table_reader
                )

            except StopIteration as exc:
                raise ArtifactValidationError(f"Evidence table is empty : {source_name} : {table_name}") from exc

            observed_row_count = sum(
                1
                for _ in table_reader
            )

        if observed_columns != expected_columns:
            raise ArtifactValidationError(f"Evidence table schema mismatch : {source_name} : {table_name}")

        if observed_row_count != table_manifest[
            "row_count"
        ]:
            raise ArtifactValidationError(f"Evidence table row-count mismatch : {source_name} : {table_name}")

        return table_path

    # Loading one validated evidence table
    def table(self, source_name: str, table_name: str) -> pd.DataFrame:
        table_path = self.validate_table(
            source_name,
            table_name
        )

        return pd.read_csv(
            table_path,
            dtype = str,
            keep_default_na = False
        )

    # Validating every report and table used by the application
    def validate_all(self) -> dict:
        source_count = 0
        table_count = 0

        for source_name, source_contract in self.policy[
            "inputs"
        ].items():
            self.report(
                source_name
            )

            source_count += 1

            for table_name in source_contract[
                "tables"
            ]:
                self.validate_table(
                    source_name,
                    table_name
                )

                table_count += 1

        return {
            "source_count": source_count,
            "table_count": table_count,
            "policy_sha256": calculate_file_sha256(
                self.policy_path
            )
        }

    # Producing a presentation-safe provenance inventory
    def provenance_records(self) -> list[dict]:
        records = []

        for source_name, source_contract in self.policy[
            "inputs"
        ].items():
            report_path = resolve_relative_path(
                self.artifact_root,
                source_contract[
                    "report_path"
                ]
            )

            records.append(
                {
                    "source": source_name,
                    "report_path": source_contract[
                        "report_path"
                    ],
                    "report_sha256": calculate_file_sha256(
                        report_path
                    ),
                    "registered_table_count": len(
                        source_contract[
                            "tables"
                        ]
                    )
                }
            )

        return records
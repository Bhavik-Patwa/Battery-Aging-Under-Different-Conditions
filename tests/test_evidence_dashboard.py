import csv
import json

from pathlib import Path

import pandas as pd
import pytest

from dashboard.artifacts import (
    REQUIRED_NAVIGATION,
    ArtifactStore,
    ArtifactValidationError,
    calculate_file_sha256,
    convert_numeric_columns,
    filter_records,
    validate_policy,
    validate_table_manifest
)

from tests.evidence_dashboard_fixture import (
    build_evidence_dashboard_fixture
)


# Writing one deterministic CSV table
def write_csv(path: Path, field_names: list[str], rows: list[dict]) -> None:
    with path.open(
        mode = "w",
        encoding = "utf-8",
        newline = ""
    ) as table_stream:
        writer = csv.DictWriter(
            table_stream,
            fieldnames = field_names,
            lineterminator = "\n"
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# Building a complete minimal evidence inventory
def build_fixture(temporary_directory: Path) -> tuple[Path, Path]:
    artifact_root = (
        temporary_directory
        / "artifacts"
    )
    report_directory = (
        artifact_root
        / "source"
    )

    report_directory.mkdir(
        parents = True
    )

    table_path = (
        report_directory
        / "records.csv"
    )

    write_csv(
        table_path,
        [
            "identity",
            "status",
            "value"
        ],
        [
            {
                "identity": "a",
                "status": "available",
                "value": "1.0"
            },
            {
                "identity": "b",
                "status": "unavailable",
                "value": ""
            }
        ]
    )

    report = {
        "execution_status": "completed",
        "tables": {
            "records": {
                "file_name": "records.csv",
                "row_count": 2,
                "sha256": calculate_file_sha256(
                    table_path
                ),
                "schema": [
                    {
                        "name": "identity"
                    },
                    {
                        "name": "status"
                    },
                    {
                        "name": "value"
                    }
                ]
            }
        }
    }

    report_path = (
        report_directory
        / "report.json"
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

    policy = {
        "policy_name": "evidence_dashboard_policy",
        "policy_version": "1.0",
        "effective_date": "2026-11-08",
        "registration_status": "prospective_presentation_contract",
        "scientific_scope": {
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
        },
        "presentation_contract": {
            "application_title": "Evidence Explorer",
            "application_role": "read_only_evidence_reporting",
            "independent_experimental_unit": "cell",
            "required_estimate_labels": [
                "descriptive",
                "associational",
                "predictive"
            ],
            "required_navigation": [
                "Evidence boundaries",
                "Experimental support",
                "Trajectories",
                "Predictive evidence",
                "Robustness",
                "Provenance"
            ],
            "unsupported_results_remain_visible": True,
            "diagnostic_only_results_remain_labelled": True,
            "downloads_are_limited_to_displayed_summary_records": True
        },
        "inputs": {
            "fixture": {
                "report_path": "source/report.json",
                "report_sha256": calculate_file_sha256(
                    report_path
                ),
                "required_values": {
                    "execution_status": "completed"
                },
                "tables": [
                    "records"
                ]
            }
        },
        "release_contract": {
            "artifact_root_environment_variable": "EVIDENCE_ARTIFACT_ROOT",
            "policy_path_environment_variable": "EVIDENCE_DASHBOARD_POLICY",
            "default_artifact_root": "data/processed",
            "default_policy_path": "configs/evidence_dashboard_policy.json",
            "fail_closed_on_hash_mismatch": True,
            "fail_closed_on_schema_mismatch": True,
            "fail_closed_on_row_count_mismatch": True,
            "network_access_required": False
        }
    }

    policy_path = (
        temporary_directory
        / "policy.json"
    )

    policy_path.write_text(
        json.dumps(
            policy,
            indent = 2,
            sort_keys = True
        )
        + "\n",
        encoding = "utf-8"
    )

    return (
        artifact_root,
        policy_path
    )


# Accepting a complete registered evidence inventory
def test_artifact_store_validates_complete_inventory(tmp_path: Path):
    artifact_root, policy_path = build_fixture(
        tmp_path
    )

    store = ArtifactStore(
        artifact_root = artifact_root,
        policy_path = policy_path
    )

    summary = store.validate_all()

    assert summary[
        "source_count"
    ] == 1
    assert summary[
        "table_count"
    ] == 1

    records = store.table(
        "fixture",
        "records"
    )

    assert len(
        records
    ) == 2
    assert records[
        "value"
    ].tolist() == [
        "1.0",
        ""
    ]


# Rejecting authorization of causal interpretation
def test_policy_rejects_causal_authorization(tmp_path: Path):
    _, policy_path = build_fixture(
        tmp_path
    )

    policy = json.loads(
        policy_path.read_text(
            encoding = "utf-8"
        )
    )

    policy[
        "scientific_scope"
    ][
        "causal_interpretation_authorized"
    ] = True

    with pytest.raises(
        ArtifactValidationError,
        match = "scientific scope"
    ):
        validate_policy(
            policy
        )


# Rejecting a modified report
def test_artifact_store_rejects_report_hash_mismatch(tmp_path: Path):
    artifact_root, policy_path = build_fixture(
        tmp_path
    )

    report_path = (
        artifact_root
        / "source"
        / "report.json"
    )

    report_path.write_text(
        "{}\n",
        encoding = "utf-8"
    )

    store = ArtifactStore(
        artifact_root = artifact_root,
        policy_path = policy_path
    )

    with pytest.raises(
        ArtifactValidationError,
        match = "report digest mismatch"
    ):
        store.validate_all()


# Rejecting a modified table
def test_artifact_store_rejects_table_hash_mismatch(tmp_path: Path):
    artifact_root, policy_path = build_fixture(
        tmp_path
    )

    table_path = (
        artifact_root
        / "source"
        / "records.csv"
    )

    table_path.write_text(
        "identity,status,value\nchanged,available,2.0\n",
        encoding = "utf-8"
    )

    store = ArtifactStore(
        artifact_root = artifact_root,
        policy_path = policy_path
    )

    with pytest.raises(
        ArtifactValidationError,
        match = "table digest mismatch"
    ):
        store.validate_all()


# Rejecting path traversal from a policy
def test_artifact_store_rejects_path_traversal(
    tmp_path: Path
):
    artifact_root, policy_path = build_fixture(
        tmp_path
    )

    policy = json.loads(
        policy_path.read_text(
            encoding = "utf-8"
        )
    )

    policy[
        "inputs"
    ][
        "fixture"
    ][
        "report_path"
    ] = "../outside.json"

    policy_path.write_text(
        json.dumps(
            policy
        ),
        encoding = "utf-8"
    )

    store = ArtifactStore(
        artifact_root = artifact_root,
        policy_path = policy_path
    )

    with pytest.raises(
        ArtifactValidationError,
        match = "escapes"
    ):
        store.validate_all()


# Preserving exact categorical filtering
def test_filter_records_uses_exact_values():
    frame = pd.DataFrame(
        [
            {
                "status": "available",
                "value": "1"
            },
            {
                "status": "unavailable",
                "value": "2"
            }
        ]
    )

    filtered = filter_records(
        frame,
        {
            "status": [
                "available"
            ]
        }
    )

    assert filtered.to_dict(
        orient = "records"
    ) == [
        {
            "status": "available",
            "value": "1"
        }
    ]


# Converting finite numeric strings while preserving unavailable values
def test_convert_numeric_columns_preserves_unavailable_values():
    frame = pd.DataFrame(
        {
            "value": [
                "1.5",
                ""
            ]
        }
    )

    converted = convert_numeric_columns(
        frame,
        [
            "value"
        ]
    )

    assert converted.loc[
        0,
        "value"
    ] == 1.5
    assert pd.isna(
        converted.loc[
            1,
            "value"
        ]
    )


# Validating the production inventory when local artifacts are present
def test_registered_policy_validates_current_artifacts():
    project_root = Path(__file__).resolve().parents[1]
    artifact_root = (
        project_root
        / "data"
        / "processed"
    )
    policy_path = (
        project_root
        / "configs"
        / "evidence_dashboard_policy.json"
    )

    if not artifact_root.is_dir():
        pytest.skip(
            "Production analytical artifacts are not available."
        )

    store = ArtifactStore(
        artifact_root = artifact_root,
        policy_path = policy_path
    )

    summary = store.validate_all()

    assert summary[
        "source_count"
    ] == 8
    assert summary[
        "table_count"
    ] == 15


# Returning no records for an explicit empty selection
def test_filter_records_empty_selection_returns_no_records():
    frame = pd.DataFrame(
        [
            {
                "status": "available"
            },
            {
                "status": "unavailable"
            }
        ]
    )

    filtered = filter_records(
        frame,
        {
            "status": []
        }
    )

    assert filtered.empty


# Rejecting a missing requested filter column
def test_filter_records_rejects_missing_column():
    frame = pd.DataFrame(
        {
            "status": [
                "available"
            ]
        }
    )

    with pytest.raises(
        ArtifactValidationError,
        match = "filter column is unavailable"
    ):
        filter_records(
            frame,
            {
                "missing": [
                    "value"
                ]
            }
        )


# Rejecting a weakened release contract
def test_policy_rejects_weakened_release_contract(tmp_path: Path):
    _, policy_path = build_fixture(
        tmp_path
    )

    policy = json.loads(
        policy_path.read_text(
            encoding = "utf-8"
        )
    )

    policy[
        "release_contract"
    ][
        "fail_closed_on_hash_mismatch"
    ] = False

    with pytest.raises(
        ArtifactValidationError,
        match = "release contract"
    ):
        validate_policy(
            policy
        )


# Rejecting a noncanonical report digest
def test_policy_rejects_noncanonical_report_digest(tmp_path: Path):
    _, policy_path = build_fixture(
        tmp_path
    )

    policy = json.loads(
        policy_path.read_text(
            encoding = "utf-8"
        )
    )

    policy[
        "inputs"
    ][
        "fixture"
    ][
        "report_sha256"
    ] = "z" * 64

    with pytest.raises(
        ArtifactValidationError,
        match = "report digest"
    ):
        validate_policy(
            policy
        )


# Rendering every registered view against current artifacts when available
def test_registered_application_renders_current_artifacts():
    project_root = Path(__file__).resolve().parents[1]
    artifact_root = (
        project_root
        / "data"
        / "processed"
    )

    required_report = (
        artifact_root
        / "robustness_assessment"
        / "robustness_assessment_report.json"
    )

    if not required_report.is_file():
        pytest.skip(
            "Production analytical artifacts are not available."
        )

    from streamlit.testing.v1 import AppTest

    application = AppTest.from_file(
        str(
            project_root
            / "dashboard"
            / "app.py"
        )
    )

    application.run(
        timeout = 120
    )

    assert not application.exception

    for page_name in REQUIRED_NAVIGATION:
        application.sidebar.radio[
            0
        ].set_value(
            page_name
        )

        application.run(
            timeout = 120
        )

        assert not application.exception


# Rejecting a manifest without a digest
def test_table_manifest_requires_digest():
    with pytest.raises(
        ArtifactValidationError,
        match = "table digest is invalid"
    ):
        validate_table_manifest(
            source_name = "fixture",
            table_name = "records",
            table_manifest = {
                "file_name": "records.csv",
                "row_count": 1,
                "schema": [{"name": "identity"}]
            }
        )


# Rejecting a negative table row count
def test_table_manifest_rejects_negative_row_count():
    with pytest.raises(
        ArtifactValidationError,
        match = "row count is invalid"
    ):
        validate_table_manifest(
            source_name = "fixture",
            table_name = "records",
            table_manifest = {
                "file_name": "records.csv",
                "row_count": -1,
                "sha256": "0" * 64,
                "schema": [{"name": "identity"}]
            }
        )


# Rejecting duplicate schema fields
def test_table_manifest_rejects_duplicate_schema_fields():
    with pytest.raises(
        ArtifactValidationError,
        match = "duplicate fields"
    ):
        validate_table_manifest(
            source_name = "fixture",
            table_name = "records",
            table_manifest = {
                "file_name": "records.csv",
                "row_count": 1,
                "sha256": "0" * 64,
                "schema": [
                    {"name": "identity"},
                    {"name": "identity"}
                ]
            }
        )


# Building the minimal fixture store used by the controlled-failure tests
def _build_fixture_store(tmp_path: Path) -> tuple[ArtifactStore, Path, Path]:
    artifact_root, policy_path = build_fixture(tmp_path)
    store = ArtifactStore(artifact_root = artifact_root, policy_path = policy_path)

    return store, artifact_root, policy_path


# Rejecting a schema mismatch in the synthetic fixture
def test_fixture_schema_mismatch_is_rejected(tmp_path: Path):
    store, artifact_root, policy_path = _build_fixture_store(tmp_path)
    store.validate_all()

    report_path = artifact_root / "source" / "report.json"
    table_path = artifact_root / "source" / "records.csv"
    table_path.write_text("identity,status\n_a,available\n", encoding = "utf-8")

    report = json.loads(report_path.read_text(encoding = "utf-8"))
    report["tables"]["records"]["sha256"] = calculate_file_sha256(table_path)
    report["tables"]["records"]["row_count"] = 1
    report_path.write_text(json.dumps(report), encoding = "utf-8")

    policy = json.loads(policy_path.read_text(encoding = "utf-8"))
    policy["inputs"]["fixture"]["report_sha256"] = calculate_file_sha256(report_path)
    policy_path.write_text(json.dumps(policy), encoding = "utf-8")

    store = ArtifactStore(artifact_root = artifact_root, policy_path = policy_path)
    with pytest.raises(ArtifactValidationError, match = "schema mismatch"):
        store.validate_all()


# Rejecting a row-count mismatch in the synthetic fixture
def test_fixture_row_count_mismatch_is_rejected(tmp_path: Path):
    store, artifact_root, policy_path = _build_fixture_store(tmp_path)
    store.validate_all()

    report_path = artifact_root / "source" / "report.json"
    report = json.loads(report_path.read_text(encoding = "utf-8"))
    report["tables"]["records"]["row_count"] = 99
    report_path.write_text(json.dumps(report), encoding = "utf-8")

    policy = json.loads(policy_path.read_text(encoding = "utf-8"))
    policy["inputs"]["fixture"]["report_sha256"] = calculate_file_sha256(report_path)
    policy_path.write_text(json.dumps(policy), encoding = "utf-8")

    store = ArtifactStore(artifact_root = artifact_root, policy_path = policy_path)
    with pytest.raises(ArtifactValidationError, match = "row-count mismatch"):
        store.validate_all()


# Rejecting path traversal from a fixture manifest
def test_fixture_table_path_traversal_is_rejected(tmp_path: Path):
    store, artifact_root, policy_path = _build_fixture_store(tmp_path)
    store.validate_all()

    report_path = artifact_root / "source" / "report.json"
    report = json.loads(report_path.read_text(encoding = "utf-8"))
    report["tables"]["records"]["file_name"] = "../../etc/passwd"
    report_path.write_text(json.dumps(report), encoding = "utf-8")

    policy = json.loads(policy_path.read_text(encoding = "utf-8"))
    policy["inputs"]["fixture"]["report_sha256"] = calculate_file_sha256(report_path)
    policy_path.write_text(json.dumps(policy), encoding = "utf-8")

    store = ArtifactStore(artifact_root = artifact_root, policy_path = policy_path)
    with pytest.raises(ArtifactValidationError, match = "escapes"):
        store.validate_all()


# Rejecting a missing report declaration
def test_missing_report_declaration_is_rejected(tmp_path: Path):
    store, artifact_root, policy_path = _build_fixture_store(tmp_path)

    report_path = artifact_root / "source" / "report.json"
    report = json.loads(report_path.read_text(encoding = "utf-8"))
    del report["execution_status"]
    report_path.write_text(json.dumps(report), encoding = "utf-8")

    policy = json.loads(policy_path.read_text(encoding = "utf-8"))
    policy["inputs"]["fixture"]["report_sha256"] = calculate_file_sha256(report_path)
    policy_path.write_text(json.dumps(policy), encoding = "utf-8")

    store = ArtifactStore(artifact_root = artifact_root, policy_path = policy_path)
    with pytest.raises(ArtifactValidationError, match = "declaration mismatch"):
        store.validate_all()


# Confirming every registered table loads from the synthetic fixture
def test_every_registered_table_loads_from_synthetic_inventory(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    artifact_root, policy_path = build_evidence_dashboard_fixture(
        output_directory = tmp_path / "dashboard_fixture",
        source_policy_path = project_root / "configs" / "evidence_dashboard_policy.json"
    )
    store = ArtifactStore(artifact_root = artifact_root, policy_path = policy_path)

    for source_name, source_contract in store.policy["inputs"].items():
        for table_name in source_contract["tables"]:
            frame = store.table(source_name, table_name)
            assert isinstance(frame, pd.DataFrame)


# Rendering every application view against a synthetic inventory
def test_application_renders_complete_synthetic_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project_root = Path(__file__).resolve().parents[1]

    artifact_root, policy_path = build_evidence_dashboard_fixture(
        output_directory = tmp_path / "dashboard_fixture",
        source_policy_path = project_root / "configs" / "evidence_dashboard_policy.json"
    )

    monkeypatch.setenv("EVIDENCE_ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("EVIDENCE_DASHBOARD_POLICY", str(policy_path))

    from streamlit.testing.v1 import AppTest

    application = AppTest.from_file(
        str(project_root / "dashboard" / "app.py")
    )

    application.run(timeout = 120)
    assert not application.exception

    for page_name in REQUIRED_NAVIGATION:
        application.sidebar.radio[0].set_value(page_name)
        application.run(timeout = 120)
        assert not application.exception

        if page_name == "Trajectories":
            application.radio[0].set_value("Bayesian associational summaries")
            application.run(timeout = 120)
            assert not application.exception


# Displaying a generic error when the artifact inventory is unavailable
def test_fail_closed_ui_shows_generic_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project_root = Path(__file__).resolve().parents[1]

    monkeypatch.setenv("EVIDENCE_ARTIFACT_ROOT", str(tmp_path / "nonexistent"))
    monkeypatch.setenv(
        "EVIDENCE_DASHBOARD_POLICY",
        str(project_root / "configs" / "evidence_dashboard_policy.json")
    )

    from streamlit.testing.v1 import AppTest

    application = AppTest.from_file(
        str(project_root / "dashboard" / "app.py")
    )

    application.run(timeout = 120)

    assert not application.exception
    assert application.error[0].value == (
        "Validated evidence could not be loaded. "
        "The application has stopped without presenting unverified results."
    )
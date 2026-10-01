import zipfile
from pathlib import Path

import pytest

from scripts.validate_ingested_dataset import (
    build_experiment_metadata_summary,
    inspect_archive_file,
    validate_archive_member_names,
    validate_measurement_metadata_pairs
)


# Confirming the experimental-unit and protocol counts
def test_build_experiment_metadata_summary() -> None:
    experiment_metadata_content = (
        "serial_internal,serial,lab,type,tp,cell,"
        "amb_temp_tp,soc_max_tp,dod_tp,c_ch_tp,"
        "c_dch_tp,sampling,stage\n"
        "S001,TP_k01_01,SIE,k,1,1,"
        "10,0.05,0,0,0,FF,1\n"
        "S002,TP_k01_04,SIE,k,1,4,"
        "10,0.05,0,0,0,FF,1\n"
    )

    metadata_summary = build_experiment_metadata_summary(
        experiment_metadata_content = (
            experiment_metadata_content
        )
    )

    assert metadata_summary == {
        "cell_count": 2,
        "test_point_count": 1,
        "calendar_protocol_variant_count": 1,
        "condition_protocol_count": 2,
        "laboratory_counts": {
            "SIE": 2
        },
        "aging_type_counts": {
            "k": 2
        }
    }


# Confirming that internal serials can be reused between experimental stages
def test_build_experiment_metadata_summary_allows_stage_specific_serials() -> None:
    experiment_metadata_content = (
        "serial_internal,serial,lab,type,tp,cell,"
        "amb_temp_tp,soc_max_tp,dod_tp,c_ch_tp,"
        "c_dch_tp,sampling,stage\n"
        "INT003,TP_k04_03,INT,k,4,3,"
        "23,0.05,0,0,0,FF,1\n"
        "INT003,TP_k07_01,INT,k,7,1,"
        "23,0.8,0,0,0,pi,2\n"
    )

    metadata_summary = build_experiment_metadata_summary(
        experiment_metadata_content = (
            experiment_metadata_content
        )
    )

    assert metadata_summary["cell_count"] == 2
    assert metadata_summary["test_point_count"] == 2


# Rejecting duplicate cell identifiers within one experimental stage
def test_build_experiment_metadata_summary_rejects_duplicate_cell_identifier() -> None:
    experiment_metadata_content = (
        "serial_internal,serial,lab,type,tp,cell,"
        "amb_temp_tp,soc_max_tp,dod_tp,c_ch_tp,"
        "c_dch_tp,sampling,stage\n"
        "INT003,TP_k04_03,INT,k,4,3,"
        "23,0.05,0,0,0,FF,1\n"
        "INT003,TP_k05_01,INT,k,5,1,"
        "23,0.5,0,0,0,FF,1\n"
    )

    with pytest.raises(
        RuntimeError,
        match = "duplicate stage-specific cell identifiers"
    ):
        build_experiment_metadata_summary(
            experiment_metadata_content = (
                experiment_metadata_content
            )
        )


# Rejecting archive members that could escape extraction
def test_validate_archive_member_names_rejects_parent_path() -> None:
    with pytest.raises(
        RuntimeError,
        match = "unsafe path"
    ):
        validate_archive_member_names(
            member_names = [
                "../measurement.csv"
            ]
        )


# Rejecting duplicate archive member names
def test_validate_archive_member_names_rejects_duplicates() -> None:
    with pytest.raises(
        RuntimeError,
        match = "duplicate member names"
    ):
        validate_archive_member_names(
            member_names = [
                "measurement.csv",
                "measurement.csv"
            ]
        )


# Rejecting measurement files without matching metadata
def test_validate_measurement_metadata_pairs_rejects_missing_metadata() -> None:
    with pytest.raises(
        RuntimeError,
        match = "missing metadata companions"
    ):
        validate_measurement_metadata_pairs(
            member_names = [
                "measurement.csv"
            ]
        )


# Rejecting metadata files without matching measurements
def test_validate_measurement_metadata_pairs_rejects_unmatched_metadata() -> None:
    with pytest.raises(
        RuntimeError,
        match = "unmatched metadata files"
    ):
        validate_measurement_metadata_pairs(
            member_names = [
                "measurement.csv",
                "measurement_meta.txt",
                "unmatched_meta.txt"
            ]
        )


# Confirming archive integrity and CSV schema discovery
def test_inspect_archive_file(tmp_path: Path) -> None:
    archive_path = (
        tmp_path
        / "battery_archive.zip"
    )

    with zipfile.ZipFile(
        archive_path,
        mode = "w"
    ) as archive:
        archive.writestr(
            "measurement.csv",
            "run_time,c_vol,c_cur,c_surf_temp,amb_temp,step_type\n"
            "0,4.1,0.5,23.1,23.0,21\n"
        )

        archive.writestr(
            "measurement_meta.txt",
            "laboratory = SIE\n"
        )

    archive_summary = inspect_archive_file(
        archive_path = archive_path
    )

    assert archive_summary["member_count"] == 2
    assert archive_summary["empty_member_count"] == 0
    assert archive_summary["measurement_file_count"] == 1
    assert archive_summary["measurement_file_names"] == [
        "measurement.csv"
    ]
    assert archive_summary["metadata_file_count"] == 1

    assert archive_summary["extension_counts"] == {
        ".csv": 1,
        ".txt": 1
    }

    assert archive_summary["csv_schemas"] == [
        {
            "file_count": 1,
            "columns": [
                "run_time",
                "c_vol",
                "c_cur",
                "c_surf_temp",
                "amb_temp",
                "step_type"
            ]
        }
    ]


# Confirming that archives may retain a leading source row-index column
def test_inspect_archive_file_accepts_source_row_index(tmp_path: Path) -> None:
    archive_path = (
        tmp_path
        / "indexed_battery_archive.zip"
    )

    with zipfile.ZipFile(
        archive_path,
        mode = "w"
    ) as archive:
        archive.writestr(
            "measurement.csv",
            (
                ",run_time,c_vol,c_cur,c_surf_temp,"
                "amb_temp,step_type\n"
                "0,00:00:00.000,4.1,0.5,23.1,23.0,21\n"
            )
        )

        archive.writestr(
            "measurement_meta.txt",
            "laboratory = SIE\n"
        )

    archive_summary = inspect_archive_file(
        archive_path = archive_path
    )

    assert archive_summary["csv_schemas"] == [
        {
            "file_count": 1,
            "columns": [
                "",
                "run_time",
                "c_vol",
                "c_cur",
                "c_surf_temp",
                "amb_temp",
                "step_type"
            ]
        }
    ]


# Confirming that archives may retain derived elapsed seconds
def test_inspect_archive_file_accepts_derived_elapsed_seconds(tmp_path: Path) -> None:
    archive_path = (
        tmp_path
        / "elapsed_seconds_battery_archive.zip"
    )

    with zipfile.ZipFile(
        archive_path,
        mode = "w"
    ) as archive:
        archive.writestr(
            "measurement.csv",
            (
                "run_time,c_vol,c_cur,c_surf_temp,"
                "amb_temp,step_type,time_to_sec\n"
                "00:00:00.000,4.1,0.5,23.1,"
                "23.0,21,0.0\n"
            )
        )

        archive.writestr(
            "measurement_meta.txt",
            "laboratory = SIE\n"
        )

    archive_summary = inspect_archive_file(
        archive_path = archive_path
    )

    assert archive_summary["csv_schemas"] == [
        {
            "file_count": 1,
            "columns": [
                "run_time",
                "c_vol",
                "c_cur",
                "c_surf_temp",
                "amb_temp",
                "step_type",
                "time_to_sec"
            ]
        }
    ]


# Reporting missing companion metadata without aborting inspection
def test_inspect_archive_file_reports_missing_metadata(tmp_path: Path) -> None:
    archive_path = (
        tmp_path
        / "missing_metadata_archive.zip"
    )

    with zipfile.ZipFile(
        archive_path,
        mode = "w"
    ) as archive:
        archive.writestr(
            "measurement.csv",
            (
                "run_time,c_vol,c_cur,c_surf_temp,"
                "amb_temp,step_type\n"
                "00:00:00.000,4.1,0.5,23.1,23.0,21\n"
            )
        )

    archive_summary = inspect_archive_file(
        archive_path = archive_path,
        require_complete_metadata_pairs = False
    )

    assert archive_summary[
        "missing_metadata_names"
    ] == [
        "measurement_meta.txt"
    ]


# Confirming that validated callers can defer repeated member checks
def test_inspect_archive_file_can_defer_member_checksum_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_path = (
        tmp_path
        / "validated_archive.zip"
    )

    with zipfile.ZipFile(
        archive_path,
        mode = "w"
    ) as archive:
        archive.writestr(
            "measurement.csv",
            (
                "run_time,c_vol,c_cur,c_surf_temp,"
                "amb_temp,step_type\n"
                "00:00:00.000,4.1,0.5,23.1,23.0,21\n"
            )
        )

        archive.writestr(
            "measurement_meta.txt",
            "laboratory = SIE\n"
        )

    def reject_checksum_repetition(
        archive: zipfile.ZipFile
    ) -> None:
        raise AssertionError("ZIP member checksums were repeated.")

    monkeypatch.setattr(
        zipfile.ZipFile,
        "testzip",
        reject_checksum_repetition
    )

    archive_summary = inspect_archive_file(
        archive_path = archive_path,
        validate_member_checksums = False
    )

    assert archive_summary[
        "measurement_file_count"
    ] == 1
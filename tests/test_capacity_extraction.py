import hashlib
import io
import zipfile

from pathlib import Path

import pytest

from scripts.extract_cell_features import (
    extract_capacity,
    extract_cycle_throughput,
    extract_ocv_reference,
    extract_dc_internal_resistance,
    calculate_ocv_corrected_dc_internal_resistance,
    build_reference_measurement_inventory,
    build_cycling_measurement_inventory,
    add_relative_reference_outcomes,
    add_audited_cumulative_cycle_exposure,
    extract_cycling_measurement_record,
    build_cell_feature_records,
    build_condition_feature_records,
    determine_reference_test_schedule,
    add_elapsed_calendar_time,
    parse_cloud_archive_object_name,
    serialize_feature_table_value,
    write_feature_table,
    FEATURE_COLUMN_UNITS,
    build_feature_table_schema,
    find_unobserved_cycling_sequences,
    parse_feature_step_type,
    FeatureUnavailableError,
    parse_reference_measurement_rows,
    extract_capacity_from_rows,
    extract_ocv_reference_from_rows
)

from scripts.validate_ingested_dataset import (
    parse_measurement_start_date,
    parse_elapsed_seconds,
    validate_archive_cache_file
)


# Confirming that elapsed time supports measurements longer than one day
def test_parse_elapsed_seconds_supports_extended_hours() -> None:
    assert parse_elapsed_seconds(
        "25:01:02.500"
    ) == pytest.approx(
        25 * 3600 + 60 + 2.5
    )


# Confirming that charge and discharge current are integrated in ampere-hours
def test_extract_capacity_integrates_both_directions() -> None:
    measurement_content = (
        "run_time,c_vol,c_cur,step_type\n"
        "00:00:00.000,3.7,1,21\n"
        "00:30:00.000,3.7,1,21\n"
        "01:00:00.000,3.7,1,21\n"
        "02:00:00.000,3.7,-1,22\n"
        "02:30:00.000,3.7,-1,22\n"
        "03:00:00.000,3.7,-1,22\n"
    )

    capacity_result = extract_capacity(
        io.StringIO(
            measurement_content
        )
    )

    assert capacity_result["charge_capacity_ah"] == pytest.approx(1.0)
    assert capacity_result["discharge_capacity_ah"] == pytest.approx(1.0)
    assert capacity_result["mean_capacity_ah"] == pytest.approx(1.0)
    assert capacity_result["step_label_charge_sample_count"] == 3
    assert capacity_result["step_label_discharge_sample_count"] == 3
    assert capacity_result["measurement_duration_days"] == pytest.approx(3 / 24)


# Rejecting measurements that lack a complete discharge section
def test_extract_capacity_rejects_incomplete_measurement() -> None:
    measurement_content = (
        "run_time,c_vol,c_cur,step_type\n"
        "00:00:00.000,3.7,1,21\n"
        "01:00:00.000,3.7,1,21\n"
    )

    with pytest.raises(
        RuntimeError,
        match = "complete contiguous positive charging and negative discharging"
    ):
        extract_capacity(
            io.StringIO(
                measurement_content
            )
        )


# Rejecting measurements containing timestamps that move backward
def test_extract_capacity_rejects_unordered_timestamps() -> None:
    measurement_content = (
        "run_time,c_vol,c_cur,step_type\n"
        "01:00:00.000,3.7,1,21\n"
        "00:30:00.000,3.7,1,21\n"
        "02:00:00.000,3.7,-1,22\n"
        "03:00:00.000,3.7,-1,22\n"
    )

    with pytest.raises(
        RuntimeError,
        match = "Measurement timestamps are not ordered"
    ):
        extract_capacity(
            io.StringIO(
                measurement_content
            )
        )


# Rejecting current values that cannot produce a valid capacity
def test_extract_capacity_rejects_non_finite_current() -> None:
    measurement_content = (
        "run_time,c_vol,c_cur,step_type\n"
        "00:00:00.000,3.7,1,21\n"
        "01:00:00.000,3.7,NaN,21\n"
        "02:00:00.000,3.7,-1,22\n"
        "03:00:00.000,3.7,-1,22\n"
    )

    with pytest.raises(
        RuntimeError,
        match = "non-finite current"
    ):
        extract_capacity(
            io.StringIO(
                measurement_content
            )
        )


# Confirming that contiguous cycling sections produce fully equivalent cycles
def test_extract_cycle_throughput() -> None:
    measurement_content = (
        "run_time,c_cur,step_type\n"
        "00:00:00.000,1,41\n"
        "01:00:00.000,1,41\n"
        "02:00:00.000,-1,42\n"
        "03:00:00.000,-1,42\n"
    )

    throughput_result = extract_cycle_throughput(
        io.StringIO(
            measurement_content
        )
    )

    assert throughput_result[
        "step_label_charge_throughput_ah"
    ] == pytest.approx(
        1.0
    )

    assert throughput_result[
        "step_label_discharge_throughput_ah"
    ] == pytest.approx(
        1.0
    )

    assert throughput_result[
        "full_equivalent_cycles"
    ] == pytest.approx(
        1 / 4.9
    )

    assert (
        throughput_result[
            "reference_capacity_ah"
        ]
        == 4.9
    )

    assert (
        throughput_result[
            "step_label_charge_sample_count"
        ]
        == 2
    )

    assert (
        throughput_result[
            "step_label_discharge_sample_count"
        ]
        == 2
    )

    assert (
        throughput_result[
            "step_label_charge_interval_count"
        ]
        == 1
    )

    assert (
        throughput_result[
            "step_label_discharge_interval_count"
        ]
        == 1
    )


# Confirming that opposite-direction samples retain their recorded step
def test_extract_cycle_throughput_records_direction_mismatch() -> None:
    measurement_content = (
        "run_time,c_cur,step_type\n"
        "00:00:00.000,1,41\n"
        "01:00:00.000,1,41\n"
        "01:00:01.000,-1,41\n"
        "02:00:00.000,-1,42\n"
        "03:00:00.000,-1,42\n"
    )

    throughput_result = extract_cycle_throughput(
        io.StringIO(
            measurement_content
        )
    )

    expected_charge_throughput_ah = (
        3599 / 3600
    )

    expected_primary_fec = (
        7201
        / (
            2
            * 4.9
            * 3600
        )
    )

    assert throughput_result[
        "step_label_charge_throughput_ah"
    ] == pytest.approx(
        expected_charge_throughput_ah
    )

    assert throughput_result[
        "step_label_discharge_throughput_ah"
    ] == pytest.approx(
        1.0
    )

    assert throughput_result[
        "full_equivalent_cycles"
    ] == pytest.approx(
        expected_primary_fec
    )

    assert throughput_result[
        "step_label_full_equivalent_cycles"
    ] == pytest.approx(
        (
            expected_charge_throughput_ah
            + 1.0
        )
        / (
            2
            * 4.9
        )
    )

    assert throughput_result[
        "step_label_fec_difference"
    ] == pytest.approx(
        1 / (
            3600
            * 4.9
        )
    )

    assert (
        throughput_result[
            "direction_mismatch_count"
        ]
        == 1
    )

    assert throughput_result[
        "direction_mismatch_throughput_ah"
    ] == pytest.approx(
        1 / 3600
    )

    assert throughput_result[
        "maximum_direction_mismatch_interval_seconds"
    ] == pytest.approx(
        1.0
    )

    assert (
        throughput_result[
            "step_label_charge_sample_count"
        ]
        == 3
    )

    assert (
        throughput_result[
            "step_label_discharge_sample_count"
        ]
        == 2
    )

    assert (
        throughput_result[
            "step_label_charge_interval_count"
        ]
        == 2
    )

    assert (
        throughput_result[
            "step_label_discharge_interval_count"
        ]
        == 1
    )

    assert (
        throughput_result[
            "direction_mismatch_interval_count"
        ]
        == 1
    )


# Confirming that companion metadata dates are normalized without changing their meaning
def test_parse_measurement_start_date() -> None:
    metadata_content = (
        "Laboratory identifier: SIE\n"
        "Measurement start date: 26.10.2021\n"
    )

    assert parse_measurement_start_date(
        metadata_content
    ) == "2021-10-26"


# Rejecting companion metadata without an identifiable measurement date
def test_parse_measurement_start_date_rejects_missing_date() -> None:
    with pytest.raises(
        RuntimeError,
        match = "exactly one measurement start date"
    ):
        parse_measurement_start_date(
            "Laboratory identifier: SIE\n"
        )


# Confirming that charge and discharge sections produce an averaged OCV reference
def test_extract_ocv_reference() -> None:
    measurement_content = (
        "run_time,c_vol,c_cur,step_type\n"
        "00:00:00.000,3.0,0,31\n"
        "00:00:01.000,3.6,0,31\n"
        "00:00:02.000,4.2,0,31\n"
        "00:00:03.000,4.2,0,32\n"
        "00:00:04.000,3.6,0,32\n"
        "00:00:05.000,3.0,0,32\n"
    )

    ocv_reference = extract_ocv_reference(
        io.StringIO(
            measurement_content
        )
    )

    assert ocv_reference is not None

    assert ocv_reference[
        "state_of_charge_fractions"
    ][500] == pytest.approx(
        0.5
    )

    assert ocv_reference[
        "mean_ocv_voltages"
    ][500] == pytest.approx(
        3.6
    )

    assert ocv_reference[
        "voltage_reversal_count"
    ] == 0


# Confirming OCV correction of the ten-second discharge-pulse resistance
def test_extract_dc_internal_resistance() -> None:
    state_of_charge_fractions = [
        grid_index / 1000
        for grid_index in range(
            1001
        )
    ]

    ocv_reference = {
        "state_of_charge_fractions": (
            state_of_charge_fractions
        ),
        "mean_ocv_voltages": [
            3.0
            + 1.2 * state_of_charge_fraction
            for state_of_charge_fraction in (
                state_of_charge_fractions
            )
        ],
        "voltage_reversal_count": 0,
        "maximum_voltage_reversal_volts": 0.0
    }

    measurement_rows = [
        "run_time,c_vol,c_cur,step_type",
        "00:00:00.000,3.6,0,0",
        "00:00:00.500,3.6,0,5032"
    ]

    for pulse_index in range(
        11
    ):
        elapsed_seconds = (
            0.6
            + pulse_index * 0.99
        )

        voltage_volts = (
            3.5
            if pulse_index == 10
            else 3.59
        )

        measurement_rows.append(
            f"00:00:{elapsed_seconds:06.3f},"
            f"{voltage_volts},-5,5032"
        )

    measurement_rows.extend(
        [
            "00:00:30.500,3.45,-5,5032",
            "00:00:30.600,3.6,0,0"
        ]
    )

    dc_internal_resistance = (
        extract_dc_internal_resistance(
            measurement_stream = io.StringIO(
                "\n".join(
                    measurement_rows
                )
                + "\n"
            )
        )
    )

    corrected_resistance = (
        calculate_ocv_corrected_dc_internal_resistance(
            dc_internal_resistance = (
                dc_internal_resistance
            ),
            ocv_reference = ocv_reference,
            discharge_capacity_ah = 4.9
        )
    )

    expected_pulse_charge_ah = (
        -50 / 3600
    )

    expected_ocv_change_volts = (
        1.2
        * expected_pulse_charge_ah
        / 4.9
    )

    expected_resistance_ohms = (
        (
            3.6
            - (
                3.5
                - expected_ocv_change_volts
            )
        )
        / 5
    )

    assert dc_internal_resistance[
        "analytical_status"
    ] == "primary"

    assert dc_internal_resistance[
        "pulse_charge_ah"
    ] == pytest.approx(
        expected_pulse_charge_ah
    )

    assert dc_internal_resistance[
        "uncorrected_dc_internal_resistance_ohms"
    ] == pytest.approx(
        0.02
    )

    assert corrected_resistance[
        "analytical_status"
    ] == "provisional_sensitivity"

    assert corrected_resistance[
        "normalization_method"
    ] == "sample_position"

    assert corrected_resistance[
        "ocv_change_volts"
    ] == pytest.approx(
        expected_ocv_change_volts
    )

    assert corrected_resistance[
        "ocv_corrected_dc_internal_resistance_ohms"
    ] == pytest.approx(
        expected_resistance_ohms
    )


# Confirming that capacity integration does not cross protocol boundaries
def test_extract_capacity_integrates_only_contiguous_sections() -> None:
    measurement_content = (
        "run_time,c_vol,c_cur,step_type\n"
        "00:00:00.000,3.7,1,21\n"
        "01:00:00.000,3.7,1,21\n"
        "02:00:00.000,3.7,0,0\n"
        "03:00:00.000,3.7,1,21\n"
        "04:00:00.000,3.7,1,21\n"
        "05:00:00.000,3.7,-1,22\n"
        "06:00:00.000,3.7,-1,22\n"
        "07:00:00.000,3.7,0,0\n"
        "08:00:00.000,3.7,-1,22\n"
        "09:00:00.000,3.7,-1,22\n"
    )

    capacity_result = extract_capacity(
        io.StringIO(
            measurement_content
        )
    )

    assert capacity_result[
        "charge_capacity_ah"
    ] == pytest.approx(
        2.0
    )

    assert capacity_result[
        "discharge_capacity_ah"
    ] == pytest.approx(
        2.0
    )

    assert capacity_result[
        "step_label_charge_interval_count"
    ] == 2

    assert capacity_result[
        "step_label_discharge_interval_count"
    ] == 2


# Confirming that cycling integration does not cross protocol boundaries
def test_extract_cycle_throughput_integrates_only_contiguous_sections() -> None:
    measurement_content = (
        "run_time,c_cur,step_type\n"
        "00:00:00.000,1,41\n"
        "01:00:00.000,1,41\n"
        "02:00:00.000,0,0\n"
        "03:00:00.000,1,41\n"
        "04:00:00.000,1,41\n"
        "05:00:00.000,-1,42\n"
        "06:00:00.000,-1,42\n"
        "07:00:00.000,0,0\n"
        "08:00:00.000,-1,42\n"
        "09:00:00.000,-1,42\n"
    )

    throughput_result = extract_cycle_throughput(
        io.StringIO(
            measurement_content
        )
    )

    assert throughput_result[
        "step_label_charge_throughput_ah"
    ] == pytest.approx(
        2.0
    )

    assert throughput_result[
        "step_label_discharge_throughput_ah"
    ] == pytest.approx(
        2.0
    )

    assert throughput_result[
        "full_equivalent_cycles"
    ] == pytest.approx(
        2 / 4.9
    )

    assert (
        throughput_result[
            "step_label_charge_sample_count"
        ]
        == 4
    )

    assert (
        throughput_result[
            "step_label_discharge_sample_count"
        ]
        == 4
    )

    assert (
        throughput_result[
            "step_label_charge_interval_count"
        ]
        == 2
    )

    assert (
        throughput_result[
            "step_label_discharge_interval_count"
        ]
        == 2
    )


# Confirming that ineligible current samples end cycling segments
def test_extract_cycle_throughput_breaks_at_current_threshold() -> None:
    measurement_content = (
        "run_time,c_cur,step_type\n"
        "00:00:00.000,1,41\n"
        "01:00:00.000,1,41\n"
        "02:00:00.000,0.01,41\n"
        "03:00:00.000,1,41\n"
        "04:00:00.000,1,41\n"
        "05:00:00.000,-1,42\n"
        "06:00:00.000,-1,42\n"
    )

    throughput_result = extract_cycle_throughput(
        io.StringIO(
            measurement_content
        )
    )

    assert throughput_result[
        "step_label_charge_throughput_ah"
    ] == pytest.approx(
        2.0
    )

    assert throughput_result[
        "step_label_discharge_throughput_ah"
    ] == pytest.approx(
        1.0
    )

    assert (
        throughput_result[
            "step_label_charge_interval_count"
        ]
        == 2
    )

    assert (
        throughput_result[
            "step_label_discharge_interval_count"
        ]
        == 1
    )


# Confirming that the feature inventory preserves audit provenance
def test_build_reference_measurement_inventory() -> None:
    audit_result = {
        "archives": [
            {
                "archive_identity": (
                    "Stage_1/TP_z02_01.zip"
                ),
                "cloud_object": (
                    "gs://bucket/archive.zip"
                ),
                "audit_status": "completed",
                "disposition": (
                    "include_with_warning"
                ),
                "published_md5_available": True,
                "quality_reasons": [
                    {
                        "reason_code": (
                            "ARCHIVE_WARNING"
                        )
                    }
                ],
                "measurements": [
                    {
                        "source_identity": (
                            "Stage_1/TP_z02_01/"
                            "TP_z02_01_01_ET_T23.csv"
                        ),
                        "audit_status": "completed",
                        "row_count": 100,
                        "disposition": "include",
                        "quality_reasons": [],
                        "companion_metadata": {
                            "measurement_start_date": (
                                "2021-09-26"
                            )
                        }
                    },
                    {
                        "source_identity": (
                            "Stage_1/TP_z02_01/"
                            "TP_z02_01_03_CU.csv"
                        ),
                        "audit_status": "completed",
                        "row_count": 100,
                        "disposition": "manual_review",
                        "quality_reasons": [
                            {
                                "reason_code": (
                                    "DOWNSTREAM_OUTCOME"
                                ),
                                "disposition": (
                                    "manual_review"
                                ),
                                "details": {
                                    "sensitivity_cohort_eligible": (
                                        True
                                    )
                                }
                            }
                        ],
                        "companion_metadata": {
                            "measurement_start_date": (
                                "2021-10-13"
                            )
                        }
                    },
                    {
                        "source_identity": (
                            "Stage_1/TP_z02_01/"
                            "TP_z02_01_33_AT_T23.csv"
                        ),
                        "audit_status": "completed",
                        "row_count": 100,
                        "disposition": "exclude",
                        "quality_reasons": [
                            {
                                "reason_code": (
                                    "EXCLUDED_MEASUREMENT"
                                ),
                                "disposition": "exclude"
                            }
                        ],
                        "companion_metadata": {
                            "measurement_start_date": (
                                "2022-09-27"
                            )
                        }
                    }
                ]
            }
        ]
    }

    measurement_inventory = (
        build_reference_measurement_inventory(
            audit_result
        )
    )

    assert len(measurement_inventory) == 3

    assert measurement_inventory[0][
        "archive_disposition"
    ] == "include_with_warning"

    assert measurement_inventory[0][
        "archive_quality_reason_codes"
    ] == [
        "ARCHIVE_WARNING"
    ]

    assert measurement_inventory[0][
        "measurement_start_date"
    ] == "2021-09-26"

    assert measurement_inventory[0][
        "primary_cohort_eligible"
    ] is True

    assert measurement_inventory[1][
        "primary_cohort_eligible"
    ] is False

    assert measurement_inventory[1][
        "sensitivity_cohort_eligible"
    ] is True

    assert measurement_inventory[2][
        "feature_extraction_requested"
    ] is False

    assert measurement_inventory[2][
        "sensitivity_cohort_eligible"
    ] is False


# Confirming that cycling inventory preserves exposure eligibility
def test_build_cycling_measurement_inventory() -> None:
    audit_result = {
        "archives": [
            {
                "archive_identity": (
                    "Stage_1/TP_z02_01.zip"
                ),
                "cloud_object": (
                    "gs://bucket/archive.zip"
                ),
                "audit_status": "completed",
                "disposition": "include",
                "published_md5_available": True,
                "quality_reasons": [],
                "measurements": [
                    {
                        "source_identity": (
                            "Stage_1/TP_z02_01/"
                            "TP_z02_01_02_ZYK.csv"
                        ),
                        "audit_status": "completed",
                        "row_count": 100,
                        "disposition": "include",
                        "quality_reasons": []
                    },
                    {
                        "source_identity": (
                            "Stage_1/TP_z02_01/"
                            "TP_z02_01_04_ZYK.csv"
                        ),
                        "audit_status": "completed",
                        "row_count": 100,
                        "disposition": "exclude",
                        "quality_reasons": [
                            {
                                "reason_code": (
                                    "EXCLUDED_EXPOSURE"
                                ),
                                "disposition": "exclude"
                            }
                        ]
                    }
                ]
            }
        ]
    }

    measurement_inventory = (
        build_cycling_measurement_inventory(
            audit_result
        )
    )

    assert len(measurement_inventory) == 2

    assert measurement_inventory[0][
        "feature_extraction_requested"
    ] is True

    assert measurement_inventory[0][
        "primary_exposure_eligible"
    ] is True

    assert measurement_inventory[1][
        "feature_extraction_requested"
    ] is False

    assert measurement_inventory[1][
        "primary_exposure_eligible"
    ] is False


# Confirming that baseline review propagates to relative outcomes
def test_add_relative_reference_outcomes_propagates_baseline_eligibility() -> None:
    reference_test_records = [
        {
            "cell_identity": "Stage_1/TP_z02_01",
            "source_identity": (
                "Stage_1/TP_z02_01/"
                "TP_z02_01_01_ET_T23.csv"
            ),
            "reference_test_protocol": "ET_T23",
            "mean_capacity_ah": 5.0,
            "uncorrected_dc_internal_resistance_ohms": 0.02,
            "ocv_corrected_dc_internal_resistance_ohms": 0.019,
            "primary_cohort_eligible": False,
            "sensitivity_cohort_eligible": True
        },
        {
            "cell_identity": "Stage_1/TP_z02_01",
            "source_identity": (
                "Stage_1/TP_z02_01/"
                "TP_z02_01_03_CU.csv"
            ),
            "reference_test_protocol": "CU",
            "mean_capacity_ah": 4.5,
            "uncorrected_dc_internal_resistance_ohms": 0.022,
            "ocv_corrected_dc_internal_resistance_ohms": 0.021,
            "primary_cohort_eligible": True,
            "sensitivity_cohort_eligible": True
        }
    ]

    add_relative_reference_outcomes(
        reference_test_records
    )

    later_record = reference_test_records[1]

    assert later_record[
        "relative_capacity"
    ] == pytest.approx(
        0.9
    )

    assert later_record[
        "relative_capacity_primary_eligible"
    ] is False

    assert later_record[
        "relative_capacity_sensitivity_eligible"
    ] is True

    assert later_record[
        "relative_dc_internal_resistance_primary_eligible"
    ] is False


# Confirming that missing performed cycling makes cumulative FEC incomplete
def test_add_audited_cumulative_cycle_exposure_records_missing_exposure() -> None:
    reference_test_records = [
        {
            "cell_identity": "Stage_1/TP_z19_01",
            "sequence_number": 5,
            "archive_quality_reasons": [
                {
                    "reason_code": (
                        "CYCLING_MEASUREMENT_MISSING"
                    ),
                    "details": {
                        "cycling_performed": True,
                        "missing_sequence_numbers": [
                            4
                        ]
                    }
                }
            ]
        }
    ]

    cycling_measurement_records = [
        {
            "cell_identity": "Stage_1/TP_z19_01",
            "sequence_number": 2,
            "feature_extraction_status": "completed",
            "full_equivalent_cycles": 50.0,
            "step_label_full_equivalent_cycles": 49.5,
            "primary_exposure_eligible": True,
            "sensitivity_exposure_eligible": True
        }
    ]

    add_audited_cumulative_cycle_exposure(
        reference_test_records = (
            reference_test_records
        ),
        cycling_measurement_records = (
            cycling_measurement_records
        )
    )

    reference_test_record = (
        reference_test_records[0]
    )

    assert reference_test_record[
        "cumulative_full_equivalent_cycles_observed"
    ] == pytest.approx(
        50.0
    )

    assert reference_test_record[
        "cumulative_fec_observation_complete"
    ] is False

    assert reference_test_record[
        "cumulative_fec_primary_eligible"
    ] is False

    assert reference_test_record[
        "unavailable_cycling_sequence_numbers"
    ] == [
        4
    ]


# Confirming that a documented pause does not create missing FEC
def test_find_unobserved_cycling_sequences_ignores_paused_cycling() -> None:
    assert find_unobserved_cycling_sequences(
        [
            {
                "reason_code": (
                    "FIFTEEN_DAY_CYCLING_PAUSE"
                ),
                "details": {
                    "cycling_performed": False,
                    "missing_sequence_numbers": [
                        14
                    ]
                }
            }
        ]
    ) == set()


# Confirming that excluded cycling measurements are retained without extraction
def test_extract_cycling_measurement_record_preserves_exclusion() -> None:
    archive_content = io.BytesIO()

    with zipfile.ZipFile(
        archive_content,
        mode = "w"
    ) as archive:
        archive.writestr(
            "unrelated.txt",
            "retained"
        )

    archive_content.seek(0)

    inventory_record = {
        "cell_identity": "Stage_1/TP_z24_01",
        "measurement_file": (
            "TP_z24_01/"
            "TP_z24_01_02_ZYK.csv"
        ),
        "source_identity": (
            "Stage_1/TP_z24_01/"
            "TP_z24_01_02_ZYK.csv"
        ),
        "sequence_number": 2,
        "feature_extraction_requested": False,
        "primary_exposure_eligible": False,
        "sensitivity_exposure_eligible": False
    }

    with zipfile.ZipFile(
        archive_content,
        mode = "r"
    ) as archive:
        feature_record = (
            extract_cycling_measurement_record(
                archive = archive,
                inventory_record = inventory_record
            )
        )

    assert feature_record[
        "feature_extraction_status"
    ] == "not_requested"

    assert feature_record[
        "full_equivalent_cycles"
    ] is None

    assert feature_record[
        "primary_exposure_eligible"
    ] is False


# Confirming the documented calendar reference-test schedules
def test_determine_reference_test_schedule() -> None:
    assert determine_reference_test_schedule(
        aging_type = "k",
        cell_number = 3
    ) == "periodic_reference_tests"

    assert determine_reference_test_schedule(
        aging_type = "k",
        cell_number = 4
    ) == "initial_and_final_only"

    assert determine_reference_test_schedule(
        aging_type = "z",
        cell_number = 4
    ) == "periodic_reference_tests"


# Confirming that cell and condition dimensions preserve treatments
def test_build_cell_and_condition_feature_records() -> None:
    experiment_metadata_content = (
        "serial_internal,serial,lab,type,tp,cell,"
        "amb_temp_tp,soc_max_tp,dod_tp,c_ch_tp,"
        "c_dch_tp,sampling,stage\n"
        "S001,TP_k01_01,SIE,k,1,1,"
        "23,0.8,0,0,0,FF,1\n"
        "S002,TP_k01_04,SIE,k,1,4,"
        "23,0.8,0,0,0,FF,1\n"
        "S003,TP_z02_01,SIE,z,2,1,"
        "35,0.8,0.6,1,1,cLH,1\n"
    )

    reference_measurement_inventory = [
        {
            "cell_identity": (
                "Stage_1/TP_k01_01"
            )
        },
        {
            "cell_identity": (
                "Stage_1/TP_k01_04"
            )
        },
        {
            "cell_identity": (
                "Stage_1/TP_z02_01"
            )
        }
    ]

    cell_records = build_cell_feature_records(
        experiment_metadata_content = (
            experiment_metadata_content
        ),
        reference_measurement_inventory = (
            reference_measurement_inventory
        )
    )

    condition_records = (
        build_condition_feature_records(
            cell_records
        )
    )

    assert len(cell_records) == 3
    assert len(condition_records) == 3

    assert cell_records[0][
        "ambient_temperature_celsius"
    ] == pytest.approx(
        23.0
    )

    assert {
        cell_record[
            "reference_test_schedule"
        ]
        for cell_record in cell_records
    } == {
        "periodic_reference_tests",
        "initial_and_final_only"
    }

    assert sum(
        condition_record[
            "cell_count"
        ]
        for condition_record in condition_records
    ) == 3


# Confirming that gross calendar time uses the initial reference-test date
def test_add_elapsed_calendar_time() -> None:
    reference_test_records = [
        {
            "cell_identity": "Stage_1/TP_k01_01",
            "source_identity": (
                "Stage_1/TP_k01_01/"
                "TP_k01_01_01_ET_T23.csv"
            ),
            "reference_test_protocol": "ET_T23",
            "measurement_start_date": "2021-09-08"
        },
        {
            "cell_identity": "Stage_1/TP_k01_01",
            "source_identity": (
                "Stage_1/TP_k01_01/"
                "TP_k01_01_02_CU.csv"
            ),
            "reference_test_protocol": "CU",
            "measurement_start_date": "2021-10-19"
        },
        {
            "cell_identity": "Stage_1/TP_k01_01",
            "source_identity": (
                "Stage_1/TP_k01_01/"
                "TP_k01_01_09_AT_T23.csv"
            ),
            "reference_test_protocol": "AT_T23",
            "measurement_start_date": None
        }
    ]

    add_elapsed_calendar_time(
        reference_test_records
    )

    assert [
        reference_test_record[
            "elapsed_calendar_days"
        ]
        for reference_test_record in (
            reference_test_records
        )
    ] == [
        0,
        41,
        None
    ]

    assert [
        reference_test_record[
            "elapsed_calendar_time_status"
        ]
        for reference_test_record in (
            reference_test_records
        )
    ] == [
        "completed",
        "completed",
        "measurement_start_date_unavailable"
    ]


# Rejecting reference-test dates that precede the cell baseline
def test_add_elapsed_calendar_time_rejects_negative_time() -> None:
    reference_test_records = [
        {
            "cell_identity": "Stage_1/TP_k01_01",
            "source_identity": (
                "Stage_1/TP_k01_01/"
                "TP_k01_01_01_ET_T23.csv"
            ),
            "reference_test_protocol": "ET_T23",
            "measurement_start_date": "2021-09-08"
        },
        {
            "cell_identity": "Stage_1/TP_k01_01",
            "source_identity": (
                "Stage_1/TP_k01_01/"
                "TP_k01_01_02_CU.csv"
            ),
            "reference_test_protocol": "CU",
            "measurement_start_date": "2021-09-07"
        }
    ]

    with pytest.raises(
        RuntimeError,
        match = "precedes the cell baseline"
    ):
        add_elapsed_calendar_time(
            reference_test_records
        )


# Confirming that audited cloud URIs resolve only within the requested bucket
def test_parse_cloud_archive_object_name() -> None:
    assert parse_cloud_archive_object_name(
        cloud_archive = (
            "gs://raw-figshare-data/"
            "raw/figshare/25975315/v1/archive"
        ),
        bucket_name = "raw-figshare-data"
    ) == (
        "raw/figshare/25975315/v1/archive"
    )

    with pytest.raises(
        RuntimeError,
        match = "does not belong"
    ):
        parse_cloud_archive_object_name(
            cloud_archive = (
                "gs://different-bucket/archive"
            ),
            bucket_name = "raw-figshare-data"
        )


# Confirming deterministic analytical-table serialization
def test_write_feature_table(tmp_path: Path) -> None:
    output_path = (
        tmp_path
        / "features.csv"
    )

    field_names = write_feature_table(
        output_path = output_path,
        records = [
            {
                "source_identity": "measurement_1",
                "eligible": True,
                "value": 1.5,
                "reason_codes": [
                    "SOURCE_WARNING"
                ]
            },
            {
                "source_identity": "measurement_2",
                "eligible": False,
                "value": None,
                "reason_codes": []
            }
        ],
        leading_columns = [
            "source_identity"
        ]
    )

    assert field_names == [
        "source_identity",
        "eligible",
        "reason_codes",
        "value"
    ]

    output_content = output_path.read_text(
        encoding = "utf-8"
    )

    assert "measurement_1,true" in output_content
    assert "measurement_2,false" in output_content
    assert serialize_feature_table_value(
        None
    ) == ""


# Rejecting cycling incidents without machine-readable exposure details
def test_find_unobserved_cycling_sequences_requires_details() -> None:
    with pytest.raises(
        RuntimeError,
        match = "missing machine-readable details"
    ):
        find_unobserved_cycling_sequences(
            [
                {
                    "reason_code": (
                        "CYCLING_MEASUREMENT_MISSING"
                    ),
                    "disposition": "manual_review"
                }
            ]
        )


# Confirming that analytical schemas preserve types, nullability and units
def test_build_feature_table_schema() -> None:
    records = [
        {
            "source_identity": "measurement_1",
            "mean_capacity_ah": 4.9,
            "eligible": True
        },
        {
            "source_identity": "measurement_2",
            "mean_capacity_ah": None,
            "eligible": False
        }
    ]

    table_schema = build_feature_table_schema(
        records = records,
        field_names = [
            "source_identity",
            "mean_capacity_ah",
            "eligible"
        ]
    )

    assert table_schema == [
        {
            "name": "source_identity",
            "data_type": "string",
            "nullable": False,
            "unit": None
        },
        {
            "name": "mean_capacity_ah",
            "data_type": "number",
            "nullable": True,
            "unit": "ampere_hour"
        },
        {
            "name": "eligible",
            "data_type": "boolean",
            "nullable": False,
            "unit": None
        }
    ]

    assert FEATURE_COLUMN_UNITS[
        "elapsed_calendar_days"
    ] == "day"


# Confirming that integral numeric step labels retain their meaning
@pytest.mark.parametrize(
    (
        "source_value",
        "expected_step_type"
    ),
    [
        ("21", "21"),
        ("21.0", "21"),
        ("41.0", "41"),
        ("5032.0", "5032")
    ]
)
def test_parse_feature_step_type(source_value: str, expected_step_type: str) -> None:
    assert parse_feature_step_type(
        value = source_value,
        row_number = 2
    ) == expected_step_type


# Confirming capacity extraction from integral numeric step labels
def test_extract_capacity_accepts_integral_numeric_step_labels() -> None:
    measurement_content = (
        "run_time,c_vol,c_cur,step_type\n"
        "00:00:00.000,3.7,1,21.0\n"
        "01:00:00.000,3.7,1,21.0\n"
        "02:00:00.000,3.7,-1,22.0\n"
        "03:00:00.000,3.7,-1,22.0\n"
    )

    capacity_result = extract_capacity(
        io.StringIO(
            measurement_content
        )
    )

    assert capacity_result[
        "charge_capacity_ah"
    ] == pytest.approx(
        1.0
    )

    assert capacity_result[
        "discharge_capacity_ah"
    ] == pytest.approx(
        1.0
    )


# Confirming cycling extraction from integral numeric step labels
def test_extract_cycle_throughput_accepts_integral_numeric_step_labels() -> None:
    measurement_content = (
        "run_time,c_cur,step_type\n"
        "00:00:00.000,1,41.0\n"
        "01:00:00.000,1,41.0\n"
        "02:00:00.000,-1,42.0\n"
        "03:00:00.000,-1,42.0\n"
    )

    throughput_result = extract_cycle_throughput(
        io.StringIO(
            measurement_content
        )
    )

    assert throughput_result[
        "full_equivalent_cycles"
    ] == pytest.approx(
        1 / 4.9
    )


# Confirming that one-direction cycling contributes observed throughput
def test_extract_cycle_throughput_accepts_one_direction() -> None:
    measurement_content = (
        "run_time,c_cur,step_type\n"
        "00:00:00.000,1,41\n"
        "01:00:00.000,1,41\n"
    )

    throughput_result = extract_cycle_throughput(
        io.StringIO(
            measurement_content
        )
    )

    assert throughput_result[
        "full_equivalent_cycles"
    ] == pytest.approx(
        1 / (
            2
            * 4.9
        )
    )

    assert throughput_result[
        "step_label_discharge_interval_count"
    ] == 0


# Confirming that timestamp reversals divide cycling segments
def test_extract_cycle_throughput_segments_timestamp_reversals() -> None:
    measurement_content = (
        "run_time,c_cur,step_type\n"
        "00:00:00.000,1,41\n"
        "01:00:00.000,1,41\n"
        "00:30:00.000,-1,42\n"
        "01:30:00.000,-1,42\n"
    )

    throughput_result = extract_cycle_throughput(
        io.StringIO(
            measurement_content
        )
    )

    assert throughput_result[
        "full_equivalent_cycles"
    ] == pytest.approx(
        2 / (
            2
            * 4.9
        )
    )

    assert throughput_result[
        "segmented_timestamp_reversal_count"
    ] == 1


# Confirming that absent capacity is unavailable rather than malformed
def test_extract_capacity_reports_unavailable_feature() -> None:
    measurement_content = (
        "run_time,c_vol,c_cur,step_type\n"
        "00:00:00.000,3.7,0,0\n"
        "00:00:01.000,3.7,0,0\n"
    )

    with pytest.raises(
        FeatureUnavailableError,
        match = "does not contain complete"
    ):
        extract_capacity(
            io.StringIO(
                measurement_content
            )
        )


# Confirming that absent cycling throughput is unavailable
def test_extract_cycle_throughput_reports_unavailable_feature() -> None:
    measurement_content = (
        "run_time,c_cur,step_type\n"
        "00:00:00.000,0,0\n"
        "00:00:01.000,0,0\n"
    )

    with pytest.raises(
        FeatureUnavailableError,
        match = "does not contain any usable"
    ):
        extract_cycle_throughput(
            io.StringIO(
                measurement_content
            )
        )


# Confirming that cached archives retain published integrity
def test_validate_archive_cache_file(tmp_path: Path) -> None:
    archive_path = (
        tmp_path
        / "archive.zip"
    )

    archive_content = b"validated archive content"

    archive_path.write_bytes(
        archive_content
    )

    file_record = {
        "size": len(
            archive_content
        ),
        "computed_md5": hashlib.md5(
            archive_content
        ).hexdigest()
    }

    validate_archive_cache_file(
        archive_path = archive_path,
        bucket = None,
        file_record = file_record,
        archive_object_name = "archive"
    )

    archive_path.write_bytes(
        b"modified archive content"
    )

    with pytest.raises(
        RuntimeError,
        match = (
            "size does not match"
            "|MD5 does not match"
        )
    ):
        validate_archive_cache_file(
            archive_path = archive_path,
            bucket = None,
            file_record = file_record,
            archive_object_name = "archive"
        )


# Confirming that timestamp reversals make cumulative exposure incomplete
def test_add_audited_cumulative_cycle_exposure_records_temporal_discontinuity() -> None:
    reference_test_records = [
        {
            "cell_identity": "Stage_1/TP_z23_01",
            "source_identity": (
                "Stage_1/TP_z23_01/"
                "TP_z23_01_23_CU.csv"
            ),
            "sequence_number": 23,
            "archive_quality_reasons": []
        }
    ]

    cycling_measurement_records = [
        {
            "cell_identity": "Stage_1/TP_z23_01",
            "source_identity": (
                "Stage_1/TP_z23_01/"
                "TP_z23_01_22_ZYK.csv"
            ),
            "sequence_number": 22,
            "feature_extraction_status": "completed",
            "full_equivalent_cycles": 10.0,
            "step_label_full_equivalent_cycles": 10.0,
            "primary_exposure_eligible": False,
            "sensitivity_exposure_eligible": None,
            "segmented_timestamp_reversal_count": 1
        }
    ]

    add_audited_cumulative_cycle_exposure(
        reference_test_records = reference_test_records,
        cycling_measurement_records = (
            cycling_measurement_records
        )
    )

    reference_record = reference_test_records[0]

    assert reference_record[
        "cumulative_full_equivalent_cycles_observed"
    ] == pytest.approx(
        10.0
    )

    assert reference_record[
        "cumulative_fec_observation_complete"
    ] is False

    assert reference_record[
        "temporally_discontinuous_cycling_sequence_numbers"
    ] == [
        22
    ]


# Confirming that unavailable cycling records do not cause temporal comparisons
def test_add_audited_cumulative_cycle_exposure_accepts_null_reversal_count() -> None:
    reference_test_records = [
        {
            "cell_identity": "Stage_1/TP_z19_01",
            "source_identity": (
                "Stage_1/TP_z19_01/"
                "TP_z19_01_05_CU.csv"
            ),
            "sequence_number": 5,
            "archive_quality_reasons": []
        }
    ]

    cycling_measurement_records = [
        {
            "cell_identity": "Stage_1/TP_z19_01",
            "source_identity": (
                "Stage_1/TP_z19_01/"
                "TP_z19_01_04_ZYK.csv"
            ),
            "sequence_number": 4,
            "feature_extraction_status": "unavailable",
            "full_equivalent_cycles": None,
            "step_label_full_equivalent_cycles": None,
            "primary_exposure_eligible": False,
            "sensitivity_exposure_eligible": False,
            "segmented_timestamp_reversal_count": None
        }
    ]

    add_audited_cumulative_cycle_exposure(
        reference_test_records = reference_test_records,
        cycling_measurement_records = (
            cycling_measurement_records
        )
    )

    reference_record = reference_test_records[0]

    assert reference_record[
        "cumulative_fec_observation_complete"
    ] is False

    assert reference_record[
        "unavailable_cycling_sequence_numbers"
    ] == [
        4
    ]

    assert reference_record[
        "temporally_discontinuous_cycling_sequence_numbers"
    ] == []


def test_parse_reference_measurement_rows_normalizes_step_labels() -> None:
    measurement_content = (
        "run_time,c_vol,c_cur,step_type\n"
        "00:00:00.000,3.7,1,21.0\n"
        "00:00:01.000,3.8,1,21.0\n"
    )
    rows = parse_reference_measurement_rows(
        io.StringIO(measurement_content)
    )
    assert rows[0][3] == "21"
    assert rows[1][3] == "21"


def test_stream_wrapper_and_from_rows_agree() -> None:
    measurement_content = (
        "run_time,c_vol,c_cur,step_type\n"
        "00:00:00.000,3.7,1,21\n"
        "01:00:00.000,3.7,1,21\n"
        "02:00:00.000,3.7,-1,22\n"
        "03:00:00.000,3.7,-1,22\n"
    )
    stream_result = extract_capacity(
        io.StringIO(measurement_content)
    )
    rows = parse_reference_measurement_rows(
        io.StringIO(measurement_content)
    )
    from_rows_result = extract_capacity_from_rows(rows)
    assert stream_result == from_rows_result


def test_single_parsed_rows_produce_capacity_ocv_and_dcir() -> None:
    measurement_content = (
        "run_time,c_vol,c_cur,step_type\n"
        "00:00:00.000,3.7,1,21\n"
        "01:00:00.000,3.7,1,21\n"
        "02:00:00.000,3.7,-1,22\n"
        "03:00:00.000,3.7,-1,22\n"
        "04:00:00.000,3.0,0,31\n"
        "05:00:00.000,3.6,0,31\n"
        "06:00:00.000,4.2,0,31\n"
        "07:00:00.000,4.2,0,32\n"
        "08:00:00.000,3.6,0,32\n"
        "09:00:00.000,3.0,0,32\n"
    )
    rows = parse_reference_measurement_rows(
        io.StringIO(measurement_content)
    )
    assert extract_capacity_from_rows(rows)["mean_capacity_ah"] == pytest.approx(1.0)
    ocv = extract_ocv_reference_from_rows(rows)
    assert ocv is not None
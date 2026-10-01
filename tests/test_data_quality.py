import io
import pytest

from pathlib import Path

from scripts.audit_data_quality import (
    audit_measurement_stream,
    load_quality_contract,
    match_incident_rules,
    audit_companion_metadata,
    resolve_disposition,
    validate_expected_absent_measurements,
    internal_cell_serials_match,
    parse_companion_test_point_label
)


CONTRACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "data_quality_contract.json"
)


# Loading the production quality contract for deterministic tests
def load_test_contract() -> dict:
    return load_quality_contract(
        contract_path = CONTRACT_PATH
    )


# Rejecting duplicate fields in the machine-readable quality contract
def test_load_quality_contract_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    duplicate_contract_path = (
        tmp_path
        / "duplicate_contract.json"
    )

    duplicate_contract_path.write_text(
        (
            "{\n"
            '  "contract_version": 12,\n'
            '  "contract_version": 13\n'
            "}\n"
        ),
        encoding = "utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match = "duplicate JSON field"
    ):
        load_quality_contract(
            contract_path = (
                duplicate_contract_path
            )
        )


# Confirming that stricter dispositions take precedence
def test_resolve_disposition_uses_strictest_reason() -> None:
    quality_contract = load_test_contract()

    resolved_disposition = resolve_disposition(
        quality_reasons = [
            {
                "disposition": "include_with_warning"
            },
            {
                "disposition": "manual_review"
            }
        ],
        quality_contract = quality_contract
    )

    assert resolved_disposition == "manual_review"


# Confirming that a documented final-test warning is matched
def test_match_incident_rules_for_final_reference_test() -> None:
    quality_contract = load_test_contract()

    matching_reasons = match_incident_rules(
        target_scope = "measurement",
        target_identity = (
            "Stage_1/TP_z02_01/"
            "TP_z02_01_33_AT_T23.csv"
        ),
        quality_contract = quality_contract
    )

    assert [
        quality_reason["reason_code"]
        for quality_reason in matching_reasons
    ] == [
        "FINAL_REFERENCE_TEST_PRECEDING_STRESS"
    ]


# Preventing one documented incident from matching undocumented files
def test_undefined_load_rule_matches_only_documented_measurements() -> None:
    quality_contract = load_test_contract()

    documented_identities = [
        (
            "Stage_1/TP_z24_01/"
            "TP_z24_01_02_ZYK.csv"
        ),
        (
            "Stage_1/TP_z25_01/"
            "TP_z25_01_04_ZYK.csv"
        )
    ]

    undocumented_identities = [
        (
            "Stage_1/TP_z24_01/"
            "TP_z24_01_04_ZYK.csv"
        ),
        (
            "Stage_1/TP_z25_01/"
            "TP_z25_01_02_ZYK.csv"
        )
    ]

    for source_identity in documented_identities:
        reason_codes = {
            quality_reason["reason_code"]
            for quality_reason in match_incident_rules(
                target_scope = "measurement",
                target_identity = source_identity,
                quality_contract = quality_contract
            )
        }

        assert (
            "UNDEFINED_LOAD_DURING_CYCLING"
            in reason_codes
        )

    for source_identity in undocumented_identities:
        reason_codes = {
            quality_reason["reason_code"]
            for quality_reason in match_incident_rules(
                target_scope = "measurement",
                target_identity = source_identity,
                quality_contract = quality_contract
            )
        }

        assert (
            "UNDEFINED_LOAD_DURING_CYCLING"
            not in reason_codes
        )

@pytest.mark.parametrize(
    (
        "target_scope",
        "reason_code",
        "target_identity"
    ),
    [
        (
            "archive",
            "CYCLING_MEASUREMENT_MISSING",
            "Stage_1/TP_z19_01.zip"
        ),
        (
            "archive",
            "EXTENDED_MISSING_CYCLING_MEASUREMENT",
            "Stage_2/TP_z01_01.zip"
        ),
        (
            "measurement",
            "PARTIALLY_MISSING_CYCLING_MEASUREMENT",
            (
                "Stage_1/TP_z21_01/"
                "TP_z21_01_22_ZYK.csv"
            )
        ),
        (
            "measurement",
            "TWO_DAYS_OF_CYCLING_DATA_MISSING",
            (
                "Stage_2/TP_z04_01/"
                "TP_z04_01_24_ZYK.csv"
            )
        ),
        (
            "measurement",
            "LOWER_VOLTAGE_BOUND_CORRECTED_DURING_CYCLING",
            (
                "Stage_1/TP_z17_03/"
                "TP_z17_03_18_ZYK.csv"
            )
        )
    ]
)

# Confirming that documented incident rules match their exact targets
def test_incident_rules_match_documented_targets(target_scope: str, reason_code: str, target_identity: str) -> None:
    quality_contract = load_test_contract()

    matching_reason_codes = {
        quality_reason["reason_code"]
        for quality_reason in match_incident_rules(
            target_scope = target_scope,
            target_identity = target_identity,
            quality_contract = quality_contract
        )
    }

    assert reason_code in matching_reason_codes


# Matching the documented Stage 2 fifteen-day cycling pause
def test_match_incident_rules_for_fifteen_day_pause() -> None:
    quality_contract = load_test_contract()

    matching_reasons = match_incident_rules(
        target_scope = "archive",
        target_identity = (
            "Stage_2/TP_z13_01.zip"
        ),
        quality_contract = quality_contract
    )

    assert [
        quality_reason["reason_code"]
        for quality_reason in matching_reasons
    ] == [
        "FIFTEEN_DAY_CYCLING_PAUSE"
    ]


# Matching the interrupted final capacity test to its pinned filename
def test_match_incident_rules_for_interrupted_final_capacity() -> None:
    quality_contract = load_test_contract()

    matching_reasons = match_incident_rules(
        target_scope = "measurement",
        target_identity = (
            "Stage_2/TP_z12_01/"
            "TP_z12_01_33_AT_T45.csv"
        ),
        quality_contract = quality_contract
    )

    assert [
        quality_reason["reason_code"]
        for quality_reason in matching_reasons
    ] == [
        "FINAL_CAPACITY_TEST_INTERRUPTED"
    ]


@pytest.mark.parametrize(
    (
        "target_identity",
        "expected_reason_code",
        "expected_incident_sequence"
    ),
    [
        (
            (
                "Stage_1/TP_z25_01/"
                "TP_z25_01_03_CU.csv"
            ),
            (
                "INITIAL_UNDEFINED_LOAD_"
                "DOWNSTREAM_OUTCOME"
            ),
            1
        ),
        (
            (
                "Stage_1/TP_z24_01/"
                "TP_z24_01_03_CU.csv"
            ),
            (
                "UNDEFINED_CYCLING_LOAD_"
                "DOWNSTREAM_OUTCOME"
            ),
            2
        ),
        (
            (
                "Stage_1/TP_z17_03/"
                "TP_z17_03_15_CU.csv"
            ),
            (
                "INCORRECT_MAXIMUM_SOC_"
                "DOWNSTREAM_OUTCOME"
            ),
            13
        ),
        (
            (
                "Stage_1/TP_z20_03/"
                "TP_z20_03_17_CU.csv"
            ),
            (
                "FAILED_SOC_RESET_"
                "DOWNSTREAM_OUTCOME"
            ),
            16
        ),
        (
            (
                "Stage_2/TP_z07_03/"
                "TP_z07_03_17_CU.csv"
            ),
            (
                "CELL_HOLDER_FAILURE_"
                "DOWNSTREAM_OUTCOME"
            ),
            16
        )
    ]
)

# Confirming that downstream exposure incidents reach later outcomes
def test_downstream_incident_rules_reach_later_outcomes(target_identity: str, expected_reason_code: str,
                                                        expected_incident_sequence: int
) -> None:
    quality_contract = load_test_contract()

    matching_reasons = match_incident_rules(
        target_scope = "measurement",
        target_identity = target_identity,
        quality_contract = quality_contract
    )

    matching_reason = next(
        quality_reason
        for quality_reason in matching_reasons
        if quality_reason[
            "reason_code"
        ] == expected_reason_code
    )

    assert (
        matching_reason["disposition"]
        == "manual_review"
    )

    assert (
        matching_reason["details"][
            "incident_sequence"
        ]
        == expected_incident_sequence
    )

    assert (
        matching_reason["details"][
            "primary_cohort_eligible"
        ]
        is False
    )

    assert (
        matching_reason["details"][
            "sensitivity_cohort_eligible"
        ]
        is True
    )


@pytest.mark.parametrize(
    "target_identity",
    [
        (
            "Stage_1/TP_z24_01/"
            "TP_z24_01_01_ET_T23.csv"
        ),
        (
            "Stage_1/TP_z17_03/"
            "TP_z17_03_11_CU.csv"
        ),
        (
            "Stage_1/TP_z20_03/"
            "TP_z20_03_15_CU.csv"
        ),
        (
            "Stage_2/TP_z07_03/"
            "TP_z07_03_15_CU.csv"
        )
    ]
)

# Confirming that downstream rules do not reach pre-incident outcomes
def test_downstream_incident_rules_exclude_preincident_outcomes(target_identity: str) -> None:
    quality_contract = load_test_contract()

    downstream_reason_codes = {
        incident_rule["reason_code"]
        for incident_rule in quality_contract[
            "downstream_incident_rules"
        ]
    }

    matched_reason_codes = {
        quality_reason["reason_code"]
        for quality_reason in match_incident_rules(
            target_scope = "measurement",
            target_identity = target_identity,
            quality_contract = quality_contract
        )
    }

    assert downstream_reason_codes.isdisjoint(
        matched_reason_codes
    )


@pytest.mark.parametrize(
    (
        "test_point_label",
        "expected_fields"
    ),
    [
        (
            "k05",
            {
                "type": "k",
                "tp": 5,
                "cell": None
            }
        ),
        (
            "k01noCU",
            {
                "type": "k",
                "tp": 1,
                "cell": None
            }
        ),
        (
            "TP_z05_03_MKII",
            {
                "type": "z",
                "tp": 5,
                "cell": 3
            }
        )
    ]
)

# Confirming supported companion test-point label representations
def test_parse_companion_test_point_label(test_point_label: str, expected_fields: dict) -> None:
    assert (
        parse_companion_test_point_label(
            test_point_label
        )
        == expected_fields
    )


# Rejecting an unrecognized companion test-point label
def test_parse_companion_test_point_label_rejects_unknown_value() -> None:
    assert (
        parse_companion_test_point_label(
            "unknown"
        )
        is None
    )


# Rejecting an unsupported extended test-point suffix
def test_parse_companion_test_point_label_rejects_unknown_suffix() -> None:
    assert (
        parse_companion_test_point_label(
            "TP_z05_03_UNKNOWN"
        )
        is None
    )


@pytest.mark.parametrize(
    (
        "expected_serial",
        "observed_serial"
    ),
    [
        (
            "INT005",
            "INT005"
        ),
        (
            "057",
            "57"
        ),
        (
            "010",
            "10"
        )
    ]
)

# Confirming exact and zero-padded internal-cell serial equivalence
def test_internal_cell_serials_match(expected_serial: str, observed_serial: str) -> None:
    assert internal_cell_serials_match(
        expected_serial = expected_serial,
        observed_serial = observed_serial
    )


@pytest.mark.parametrize(
    (
        "expected_serial",
        "observed_serial"
    ),
    [
        (
            "INT010",
            "INT046"
        ),
        (
            "INT023",
            "INT027"
        ),
        (
            "057",
            "58"
        )
    ]
)

# Preserving substantive internal-cell serial conflicts
def test_internal_cell_serials_match_rejects_conflict(expected_serial: str, observed_serial: str) -> None:
    assert not internal_cell_serials_match(
        expected_serial = expected_serial,
        observed_serial = observed_serial
    )


# Confirming that companion-metadata conflicts require manual review
def test_audit_companion_metadata_records_identity_mismatch() -> None:
    metadata_content = (
        "Laboratory identifier: SIE\n"
        "Internal cell serial: S004\n"
        "Testpoint: k02noCU\n"
        "Measurement start date: 01.01.2022\n"
        "Measurement device used: Battery tester\n"
        "Test channel: CH01\n"
        "Climate chamber used: Climate chamber\n"
        "Climate chamber temperature setpoint: 10°C\n"
        "Position within climate chamber: Position 1\n"
    )

    metadata_audit = audit_companion_metadata(
        metadata_content = metadata_content,
        measurement_name = (
            "TP_k03_04/TP_k03_04_01_ET_T10.csv"
        ),
        source_identity = (
            "Stage_1/TP_k03_04/"
            "TP_k03_04_01_ET_T10.csv"
        ),
        experiment_record = {
            "type": "k",
            "tp": "3",
            "cell": "4",
            "lab": "SIE",
            "serial_internal": "S004",
            "amb_temp_tp": "23"
        }
    )

    assert metadata_audit[
        "quality_reasons"
    ][0][
        "reason_code"
    ] == "COMPANION_METADATA_IDENTITY_MISMATCH"

    assert metadata_audit[
        "quality_reasons"
    ][0][
        "disposition"
    ] == "manual_review"


# Recording an unrecognized chamber setpoint for manual review
def test_audit_companion_metadata_records_unparseable_setpoint() -> None:
    metadata_content = (
        "Laboratory identifier: SIE\n"
        "Internal cell serial: S005\n"
        "Testpoint: k05\n"
        "Measurement start date: 01.01.2022\n"
        "Measurement device used: Device\n"
        "Test channel: Channel\n"
        "Climate chamber used: Chamber\n"
        "Climate chamber temperature setpoint: unavailable\n"
        "Position within climate chamber: Position\n"
    )

    experiment_record = {
        "lab": "SIE",
        "serial_internal": "S005",
        "type": "k",
        "tp": "5",
        "cell": "5",
        "amb_temp_tp": "23"
    }

    metadata_audit = audit_companion_metadata(
        metadata_content = metadata_content,
        measurement_name = (
            "TP_k05_05/TP_k05_05_01_ET_T10.csv"
        ),
        source_identity = (
            "Stage_2/TP_k05_05/"
            "TP_k05_05_01_ET_T10.csv"
        ),
        experiment_record = experiment_record
    )

    assert (
        metadata_audit[
            "climate_chamber_temperature_setpoint_celsius"
        ]
        is None
    )

    assert (
        metadata_audit[
            "raw_climate_chamber_temperature_setpoint"
        ]
        == "unavailable"
    )

    assert metadata_audit[
        "quality_reasons"
    ][0][
        "reason_code"
    ] == (
        "COMPANION_METADATA_TEMPERATURE_UNPARSABLE"
    )


# Accepting equivalent companion labels and numeric serial formatting
def test_audit_companion_metadata_accepts_equivalent_identity_formats() -> None:
    metadata_content = (
        "Laboratory identifier: SIE\n"
        "Internal cell serial: 57\n"
        "Testpoint: TP_k01_01_MKII\n"
        "Measurement start date: 01.01.2022\n"
        "Measurement device used: Battery tester\n"
        "Test channel: CH01\n"
        "Climate chamber used: Climate chamber\n"
        "Climate chamber temperature setpoint: 10°C\n"
        "Position within climate chamber: Position 1\n"
    )

    metadata_audit = audit_companion_metadata(
        metadata_content = metadata_content,
        measurement_name = (
            "TP_k01_01/TP_k01_01_01_ET_T10.csv"
        ),
        source_identity = (
            "Stage_2/TP_k01_01/"
            "TP_k01_01_01_ET_T10.csv"
        ),
        experiment_record = {
            "type": "k",
            "tp": "1",
            "cell": "1",
            "lab": "SIE",
            "serial_internal": "057",
            "amb_temp_tp": "23"
        }
    )

    reason_codes = {
        quality_reason[
            "reason_code"
        ]
        for quality_reason in metadata_audit[
            "quality_reasons"
        ]
    }

    assert (
        "COMPANION_METADATA_IDENTITY_MISMATCH"
        not in reason_codes
    )

    assert metadata_audit[
        "parsed_test_point"
    ] == {
        "type": "k",
        "tp": 1,
        "cell": 1
    }


# Detecting missing ambient temperature and current-direction conflicts
def test_audit_measurement_stream_records_quality_warnings() -> None:
    quality_contract = load_test_contract()

    measurement_content = (
        "run_time,c_vol,c_cur,c_surf_temp,"
        "amb_temp,step_type\n"
        "00:00:00.000,3.7,0,23,NaN,41\n"
        "00:00:01.000,3.8,1,23,NaN,41\n"
        "00:00:02.000,3.6,-1,23,NaN,41\n"
        "00:00:03.000,3.5,-1,23,NaN,42\n"
    )

    measurement_audit = audit_measurement_stream(
        measurement_stream = io.StringIO(
            measurement_content
        ),
        source_identity = (
            "Stage_1/TP_z02_01/"
            "TP_z02_01_02_ZYK.csv"
        ),
        quality_contract = quality_contract
    )

    reason_codes = {
        quality_reason["reason_code"]
        for quality_reason in measurement_audit[
            "quality_reasons"
        ]
    }

    assert measurement_audit["row_count"] == 4
    assert (
        measurement_audit[
            "ambient_temperature_missing_count"
        ]
        == 4
    )
    assert (
        measurement_audit[
            "cycling_direction_mismatch_count"
        ]
        == 1
    )
    assert (
        "AMBIENT_TEMPERATURE_NOT_RECORDED"
        in reason_codes
    )
    assert (
        "CYCLING_STEP_CURRENT_DIRECTION_MISMATCH"
        in reason_codes
    )
    assert (
        measurement_audit["disposition"]
        == "include_with_warning"
    )


# Confirming that diagnostic pulse current remains inside the measurement guardrail
def test_audit_measurement_stream_accepts_diagnostic_pulse_current() -> None:
    quality_contract = load_test_contract()

    measurement_content = (
        "run_time,c_vol,c_cur,c_surf_temp,"
        "amb_temp,step_type\n"
        "00:00:00.000,3.7,0,23,23,0\n"
        "00:00:01.000,3.4,-23.1,24,23,5032\n"
    )

    measurement_audit = audit_measurement_stream(
        measurement_stream = io.StringIO(
            measurement_content
        ),
        source_identity = (
            "Stage_1/TP_z02_01/"
            "TP_z02_01_03_CU.csv"
        ),
        quality_contract = quality_contract
    )

    assert (
        measurement_audit[
            "out_of_range_value_count"
        ]
        == 0
    )

    assert (
        measurement_audit[
            "disposition"
        ]
        == "include"
    )


# Confirming that absent sampled ambient temperature does not exclude usable data
def test_audit_measurement_stream_records_ambient_temperature_availability() -> None:
    quality_contract = load_test_contract()

    measurement_content = (
        "run_time,c_vol,c_cur,c_surf_temp,"
        "amb_temp,step_type\n"
        "00:00:00.000,3.7,0,23,NaN,0\n"
        "00:00:01.000,3.7,0,23,NaN,0\n"
    )

    measurement_audit = audit_measurement_stream(
        measurement_stream = io.StringIO(
            measurement_content
        ),
        source_identity = (
            "Stage_1/TP_k01_01/"
            "TP_k01_01_02_CU.csv"
        ),
        quality_contract = quality_contract
    )

    reason_codes = {
        quality_reason[
            "reason_code"
        ]
        for quality_reason in measurement_audit[
            "quality_reasons"
        ]
    }

    assert (
        "AMBIENT_TEMPERATURE_NOT_RECORDED"
        in reason_codes
    )

    assert (
        measurement_audit[
            "disposition"
        ]
        == "include"
    )


# Confirming that cycling-direction checks apply only to cycling measurements
def test_audit_measurement_stream_limits_direction_checks_to_cycling() -> None:
    quality_contract = load_test_contract()

    measurement_content = (
        "run_time,c_vol,c_cur,c_surf_temp,"
        "amb_temp,step_type\n"
        "00:00:00.000,3.7,0,23,23,41\n"
        "00:00:01.000,3.6,-1,23,23,41\n"
    )

    measurement_audit = audit_measurement_stream(
        measurement_stream = io.StringIO(
            measurement_content
        ),
        source_identity = (
            "Stage_1/TP_z02_01/"
            "TP_z02_01_03_CU.csv"
        ),
        quality_contract = quality_contract
    )

    assert (
        measurement_audit[
            "cycling_direction_mismatch_count"
        ]
        == 0
    )

    assert (
        measurement_audit[
            "disposition"
        ]
        == "include"
    )


# Confirming that a validated source row-index column remains visible
def test_audit_measurement_stream_accepts_source_row_index() -> None:
    quality_contract = load_test_contract()

    measurement_content = (
        ",run_time,c_vol,c_cur,c_surf_temp,"
        "amb_temp,step_type\n"
        "0,00:00:00.000,3.7,0,23,NaN,0\n"
        "1,00:00:01.000,3.8,1,23,NaN,41\n"
    )

    measurement_audit = audit_measurement_stream(
        measurement_stream = io.StringIO(
            measurement_content
        ),
        source_identity = (
            "Stage_1/TP_k05_05/"
            "TP_k05_05_01_ET_T10.csv"
        ),
        quality_contract = quality_contract
    )

    reason_codes = {
        quality_reason["reason_code"]
        for quality_reason in measurement_audit[
            "quality_reasons"
        ]
    }

    assert (
        measurement_audit[
            "source_row_index_present"
        ]
        is True
    )

    assert (
        measurement_audit[
            "first_source_row_index"
        ]
        == 0
    )

    assert (
        measurement_audit[
            "last_source_row_index"
        ]
        == 1
    )

    assert (
        "SOURCE_ROW_INDEX_COLUMN_PRESENT"
        in reason_codes
    )


# Recording source-index discontinuities without using them for ordering
def test_audit_measurement_stream_records_source_index_discontinuities() -> None:
    quality_contract = load_test_contract()

    measurement_content = (
        ",run_time,c_vol,c_cur,c_surf_temp,"
        "amb_temp,step_type\n"
        "10,00:00:00.000,3.7,0,23,NaN,0\n"
        "12,00:00:01.000,3.8,1,23,NaN,41\n"
        "5,00:00:02.000,3.6,-1,23,NaN,42\n"
    )

    measurement_audit = audit_measurement_stream(
        measurement_stream = io.StringIO(
            measurement_content
        ),
        source_identity = (
            "Stage_2/TP_k05_05/"
            "TP_k05_05_01_ET_T10.csv"
        ),
        quality_contract = quality_contract
    )

    assert (
        measurement_audit[
            "source_row_index_nonincreasing_count"
        ]
        == 1
    )

    assert measurement_audit[
        "source_row_index_discontinuity_examples"
    ] == [
        {
            "row_number": 4,
            "previous_source_row_index": 12,
            "current_source_row_index": 5
        }
    ]

    assert measurement_audit["row_count"] == 3


# Confirming that derived elapsed seconds agree with run_time
def test_audit_measurement_stream_accepts_derived_elapsed_seconds() -> None:
    quality_contract = load_test_contract()

    measurement_content = (
        "run_time,c_vol,c_cur,c_surf_temp,"
        "amb_temp,step_type,time_to_sec\n"
        "00:00:00.000,3.7,0,23,NaN,0,0.0\n"
        "25:01:02.500,3.8,1,23,NaN,41,90062.5\n"
    )

    measurement_audit = audit_measurement_stream(
        measurement_stream = io.StringIO(
            measurement_content
        ),
        source_identity = (
            "Stage_1/TP_k05_05/"
            "TP_k05_05_01_ET_T23.csv"
        ),
        quality_contract = quality_contract
    )

    reason_codes = {
        quality_reason["reason_code"]
        for quality_reason in measurement_audit[
            "quality_reasons"
        ]
    }

    assert (
        measurement_audit[
            "elapsed_seconds_column_present"
        ]
        is True
    )

    assert (
        measurement_audit[
            "maximum_elapsed_seconds_difference"
        ]
        == pytest.approx(
            0.0
        )
    )

    assert (
        "DERIVED_ELAPSED_SECONDS_COLUMN_PRESENT"
        in reason_codes
    )


# Rejecting derived elapsed seconds that conflict with run_time
def test_audit_measurement_stream_rejects_conflicting_elapsed_seconds() -> None:
    quality_contract = load_test_contract()

    measurement_content = (
        "run_time,c_vol,c_cur,c_surf_temp,"
        "amb_temp,step_type,time_to_sec\n"
        "00:00:00.000,3.7,0,23,NaN,0,1.0\n"
    )

    with pytest.raises(
        RuntimeError,
        match = (
            "derived elapsed time conflicts with run_time"
        )
    ):
        audit_measurement_stream(
            measurement_stream = io.StringIO(
                measurement_content
            ),
            source_identity = (
                "Stage_1/TP_k05_05/"
                "TP_k05_05_01_ET_T23.csv"
            ),
            quality_contract = quality_contract
        )


# Registering a numeric-export measurement for manual review
def test_audit_measurement_stream_records_numeric_export() -> None:
    quality_contract = load_test_contract()

    measurement_content = (
        ",run_time,c_vol,c_cur,c_surf_temp,"
        "amb_temp,step_type\n"
        "28560,0.0,3.7,0,23,0,0.0\n"
        "28638,750.141,3.8,1,23,22,41.0\n"
    )

    measurement_audit = audit_measurement_stream(
        measurement_stream = io.StringIO(
            measurement_content
        ),
        source_identity = (
            "Stage_2/TP_k05_05/"
            "TP_k05_05_01_ET_T10.csv"
        ),
        quality_contract = quality_contract
    )

    assert measurement_audit[
        "run_time_representations"
    ] == [
        "numeric_elapsed_value"
    ]

    assert measurement_audit[
        "step_type_representations"
    ] == [
        "integral_numeric_text"
    ]

    assert measurement_audit[
        "first_source_row_index"
    ] == 28560

    assert measurement_audit[
        "last_source_row_index"
    ] == 28638

    assert (
        measurement_audit["disposition"]
        == "manual_review"
    )


# Rejecting a fractional measurement step type
def test_audit_measurement_stream_rejects_fractional_step_type() -> None:
    quality_contract = load_test_contract()

    measurement_content = (
        "run_time,c_vol,c_cur,c_surf_temp,"
        "amb_temp,step_type\n"
        "00:00:00.000,3.7,0,23,NaN,31.5\n"
    )

    with pytest.raises(
        RuntimeError,
        match = "non-integral step type"
    ):
        audit_measurement_stream(
            measurement_stream = io.StringIO(
                measurement_content
            ),
            source_identity = (
                "Stage_2/TP_k05_05/"
                "TP_k05_05_01_ET_T10.csv"
            ),
            quality_contract = quality_contract
        )


@pytest.mark.parametrize(
    (
        "measurement_content",
        "expected_error"
    ),
    [
        (
            (
                "run_time,c_vol,c_cur,c_surf_temp,"
                "amb_temp,step_type\n"
                "00:00:00.000,3.7,0,23,23,0,unexpected\n"
            ),
            "row with extra columns"
        ),
        (
            (
                "run_time,c_vol,c_cur,c_surf_temp,"
                "amb_temp,step_type\n"
                "00:00:00.000,3.7,0,23,23\n"
            ),
            "row with missing columns"
        )
    ]
)
# Rejecting measurement rows that do not match the validated schema
def test_audit_measurement_stream_rejects_inconsistent_row_width(measurement_content: str, expected_error: str) -> None:
    with pytest.raises(
        RuntimeError,
        match = expected_error
    ):
        audit_measurement_stream(
            measurement_stream = io.StringIO(
                measurement_content
            ),
            source_identity = (
                "Stage_1/TP_k01_01/"
                "TP_k01_01_01_ET_T23.csv"
            ),
            quality_contract = load_test_contract()
        )


# Confirming the documented sequence-14 measurement remains absent
def test_validate_expected_absent_measurements() -> None:
    quality_contract = load_test_contract()

    validate_expected_absent_measurements(
        archive_identity = (
            "Stage_2/TP_z13_01.zip"
        ),
        measurement_names = [
            (
                "TP_z13_01/"
                "TP_z13_01_12_ZYK.csv"
            ),
            (
                "TP_z13_01/"
                "TP_z13_01_16_ZYK.csv"
            )
        ],
        quality_contract = quality_contract
    )


# Rejecting a measurement documented as absent
def test_validate_expected_absent_measurements_rejects_present_file() -> None:
    quality_contract = load_test_contract()

    with pytest.raises(
        RuntimeError,
        match = (
            "documented as absent are present"
        )
    ):
        validate_expected_absent_measurements(
            archive_identity = (
                "Stage_2/TP_z13_01.zip"
            ),
            measurement_names = [
                (
                    "TP_z13_01/"
                    "TP_z13_01_14_ZYK.csv"
                )
            ],
            quality_contract = quality_contract
        )


@pytest.mark.parametrize(
    (
        "archive_identity",
        "reason_code",
        "cycling_performed",
        "missing_sequence_number"
    ),
    [
        (
            "Stage_1/TP_z19_01.zip",
            "CYCLING_MEASUREMENT_MISSING",
            True,
            4
        ),
        (
            "Stage_2/TP_z01_01.zip",
            (
                "EXTENDED_MISSING_"
                "CYCLING_MEASUREMENT"
            ),
            True,
            24
        ),
        (
            "Stage_2/TP_z13_01.zip",
            "FIFTEEN_DAY_CYCLING_PAUSE",
            False,
            14
        )
    ]
)


# Confirming that absent cycling sequences preserve exposure status
def test_archive_incidents_record_missing_cycling_exposure(archive_identity: str, reason_code: str,
                                                           cycling_performed: bool, missing_sequence_number: int
) -> None:
    quality_contract = load_test_contract()

    matching_reason = next(
        quality_reason
        for quality_reason in match_incident_rules(
            target_scope = "archive",
            target_identity = archive_identity,
            quality_contract = quality_contract
        )
        if quality_reason[
            "reason_code"
        ] == reason_code
    )

    assert matching_reason["details"][
        "cycling_performed"
    ] is cycling_performed

    assert matching_reason["details"][
        "missing_sequence_numbers"
    ] == [
        missing_sequence_number
    ]
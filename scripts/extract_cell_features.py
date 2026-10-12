import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
import zipfile

from bisect import bisect_left
from collections import Counter
from datetime import date
from pathlib import Path, PurePosixPath
from statistics import median
from concurrent.futures import ProcessPoolExecutor

from google.cloud import storage

from scripts.validate_ingested_dataset import (
    ARTICLE_METADATA_OBJECT,
    EXPECTED_CELL_COUNT,
    EXPECTED_CONDITION_PROTOCOL_COUNT,
    REQUIRED_EXPERIMENT_COLUMNS,
    build_landed_object_name,
    calculate_file_digest,
    download_text_object,
    find_dataset_root_file,
    parse_elapsed_seconds,
    resolve_cached_archive_path
)

REQUIRED_COLUMNS = {
    "run_time",
    "c_cur",
    "step_type"
}

REQUIRED_CELL_FEATURE_EXPERIMENT_COLUMNS = {
    *REQUIRED_EXPERIMENT_COLUMNS,
    "m_0",
    "scale",
    "R_1khz_0",
    "resistance_meter",
    "U_0",
    "volt_meter"
}

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR
OCV_GRID_INTERVAL_COUNT = 1000
DCIR_PULSE_STEP_TYPE = "5032"
DCIR_EVALUATION_SECONDS = 10.0
DCIR_CURRENT_CHANGE_AMPS = 1.0
DCIR_BASELINE_VOLTAGE_CHANGE_VOLTS = 0.005
REFERENCE_CAPACITY_AH = 4.9
CYCLING_CURRENT_THRESHOLD_AMPERES = 0.05

ReferenceMeasurementRow = tuple[
    float,
    float,
    float,
    str
]

REFERENCE_MEASUREMENT_PATTERN = re.compile(
    r"^(?P<dataset_stage>Stage_[12])/"
    r"(?P<cell_folder>TP_[kz]\d{2}_\d{2})/"
    r"(?P=cell_folder)_(?P<sequence_number>\d{2})_"
    r"(?P<protocol>ET_T23|CU|exCU|AT_T23)\.csv$"
)

NON_ANALYTICAL_REFERENCE_MEASUREMENT_PATTERN = re.compile(
    r"^(?P<dataset_stage>Stage_[12])/"
    r"(?P<cell_folder>TP_[kz]\d{2}_\d{2})/"
    r"(?P=cell_folder)_(?P<sequence_number>\d{2})_"
    r"(?P<protocol>ET_T10|ET_T45|AT_T10|AT_T45)\.csv$"
)

CYCLING_MEASUREMENT_PATTERN = re.compile(
    r"^(?P<dataset_stage>Stage_[12])/"
    r"(?P<cell_folder>TP_z\d{2}_\d{2})/"
    r"(?P=cell_folder)_(?P<sequence_number>\d{2})_ZYK\.csv$"
)

PRIMARY_COHORT_DISPOSITIONS = {
    "include",
    "include_with_warning"
}

CYCLING_EXPOSURE_INCIDENT_REASON_CODES = {
    "CYCLING_MEASUREMENT_MISSING",
    "EXTENDED_MISSING_CYCLING_MEASUREMENT",
    "FIFTEEN_DAY_CYCLING_PAUSE"
}

PARTIALLY_OBSERVED_CYCLING_REASON_CODES = {
    "PARTIALLY_MISSING_CYCLING_MEASUREMENT",
    "TWO_DAYS_OF_CYCLING_DATA_MISSING"
}

FEATURE_COLUMN_UNITS = {
    "ambient_temperature_celsius": "degree_Celsius",
    "initial_cell_mass_grams": "gram",
    "initial_voltage_volts": "volt",
    "maximum_state_of_charge_fraction": "fraction",
    "depth_of_discharge_fraction": "fraction",
    "charge_rate_c": "C_rate",
    "discharge_rate_c": "C_rate",
    "elapsed_calendar_days": "day",
    "charge_capacity_ah": "ampere_hour",
    "discharge_capacity_ah": "ampere_hour",
    "mean_capacity_ah": "ampere_hour",
    "measurement_duration_days": "day",
    "relative_capacity": "fraction",
    "capacity_loss_percent": "percent",
    "baseline_voltage_volts": "volt",
    "evaluated_pulse_voltage_volts": "volt",
    "corrected_pulse_voltage_volts": "volt",
    "ocv_change_volts": "volt",
    "pulse_current_amperes": "ampere",
    "pulse_charge_ah": "ampere_hour",
    "evaluation_time_seconds": "second",
    "uncorrected_dc_internal_resistance_ohms": "ohm",
    "uncorrected_dc_internal_resistance_milliohms": "milliohm",
    "ocv_corrected_dc_internal_resistance_ohms": "ohm",
    "ocv_corrected_dc_internal_resistance_milliohms": "milliohm",
    "relative_dc_internal_resistance": "fraction",
    "dc_internal_resistance_growth_percent": "percent",
    "relative_ocv_corrected_dc_internal_resistance": "fraction",
    "ocv_corrected_dc_internal_resistance_growth_percent": "percent",
    "maximum_ocv_voltage_reversal_volts": "volt",
    "full_equivalent_cycles": "full_equivalent_cycle",
    "step_label_full_equivalent_cycles": "full_equivalent_cycle",
    "step_label_fec_difference": "full_equivalent_cycle",
    "step_label_charge_throughput_ah": "ampere_hour",
    "step_label_discharge_throughput_ah": "ampere_hour",
    "direction_mismatch_throughput_ah": "ampere_hour",
    "maximum_direction_mismatch_interval_seconds": "second",
    "reference_capacity_ah": "ampere_hour",
    "cumulative_full_equivalent_cycles_observed": (
        "full_equivalent_cycle"
    ),
    "cumulative_step_label_full_equivalent_cycles_observed": (
        "full_equivalent_cycle"
    ),
    "cumulative_step_label_fec_difference_observed": (
        "full_equivalent_cycle"
    )
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_AUDIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "dataset_quality_audit.json"
)

DEFAULT_CONTRACT_PATH = (
    PROJECT_ROOT
    / "configs"
    / "data_quality_contract.json"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

DEFAULT_ARCHIVE_CACHE_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "archives"
)


# Representing a scientifically unavailable feature without a parser failure
class FeatureUnavailableError(RuntimeError):
    pass


# Parsing the audited dataset and analytical-table destination
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = (
            "Extracting audited battery-aging features incrementally from every landed cell archive."
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
        "--audit-path",
        type = Path,
        default = DEFAULT_AUDIT_PATH,
        help = (
            "Completed data-quality audit controlling feature extraction and cohort eligibility."
        )
    )

    parser.add_argument(
        "--output-directory",
        type = Path,
        default = DEFAULT_OUTPUT_DIRECTORY,
        help = (
            "Directory receiving deterministic analytical tables and their manifest."
        )
    )

    parser.add_argument(
        "--archive-cache-directory",
        type = Path,
        default = DEFAULT_ARCHIVE_CACHE_DIRECTORY,
        help = (
            "Directory retaining integrity-validated compressed cell archives for repeatable local processing."
        )
    )

    parser.add_argument(
        "--worker-count",
        type = int,
        default = 1,
        help = (
            "Number of local archive-extraction processes. Use one for sequential processing."
        )
    )

    parser.add_argument(
        "--contract-path",
        type = Path,
        default = DEFAULT_CONTRACT_PATH,
        help = (
            "Exact data-quality contract used to produce the controlling audit."
        )
    )

    return parser.parse_args()


# Normalizing integral measurement step labels
def parse_feature_step_type(value: str, row_number: int) -> str:
    normalized_value = value.strip()

    try:
        numerical_step_type = float(
            normalized_value
        )

    except ValueError as exc:
        raise RuntimeError(f"Measurement contains a non-numerical step type : row {row_number} : {value!r}") from exc

    if (
        not math.isfinite(
            numerical_step_type
        )
        or numerical_step_type < 0
        or not numerical_step_type.is_integer()
    ):
        raise RuntimeError(f"Measurement contains a non-integral step type : row {row_number} : {value!r}")

    return str(
        int(
            numerical_step_type
        )
    )


# Parsing one complete reference measurement once
def parse_reference_measurement_rows(measurement_stream: io.TextIOBase) -> list[ReferenceMeasurementRow]:
    measurement_reader = csv.reader(
        measurement_stream
    )

    try:
        field_names = next(
            measurement_reader
        )

    except StopIteration as exc:
        raise RuntimeError(
            "Reference measurement does not contain a header."
        ) from exc

    required_columns = {
        "run_time",
        "c_vol",
        "c_cur",
        "step_type"
    }

    missing_columns = required_columns.difference(
        field_names
    )

    if missing_columns:
        raise RuntimeError(f"Reference measurement is missing required columns : {sorted(missing_columns)}")

    column_indices = {
        field_name: field_names.index(
            field_name
        )
        for field_name in required_columns
    }

    parsed_rows = []

    for row_number, measurement_row in enumerate(
        measurement_reader,
        start = 2
    ):
        if len(measurement_row) != len(
            field_names
        ):
            raise RuntimeError(f"CSV row {row_number} has an invalid column count.")

        elapsed_seconds = parse_elapsed_seconds(
            measurement_row[
                column_indices["run_time"]
            ]
        )

        voltage_volts = float(
            measurement_row[
                column_indices["c_vol"]
            ]
        )

        current_amperes = float(
            measurement_row[
                column_indices["c_cur"]
            ]
        )

        if not math.isfinite(voltage_volts):
            raise RuntimeError(f"CSV row {row_number} contains a non-finite voltage.")

        if not math.isfinite(current_amperes):
            raise RuntimeError(f"CSV row {row_number} contains a non-finite current.")

        step_type = parse_feature_step_type(
            value = measurement_row[
                column_indices["step_type"]
            ],
            row_number = row_number
        )

        parsed_rows.append(
            (
                elapsed_seconds,
                voltage_volts,
                current_amperes,
                step_type
            )
        )

    if not parsed_rows:
        raise FeatureUnavailableError("The reference measurement does not contain data rows.")

    return parsed_rows


# Integrating capacity from parsed reference-measurement rows
def extract_capacity_from_rows(measurement_rows: list[ReferenceMeasurementRow]) -> dict:
    step_label_charge_ampere_seconds = 0.0
    step_label_discharge_ampere_seconds = 0.0

    previous_elapsed_seconds = None
    previous_step_type = None
    previous_current_amperes = None

    first_measurement_seconds = None
    last_measurement_seconds = None

    step_label_charge_sample_count = 0
    step_label_discharge_sample_count = 0
    step_label_charge_interval_count = 0
    step_label_discharge_interval_count = 0

    for (
        elapsed_seconds,
        _voltage_volts,
        current_amperes,
        step_type
    ) in measurement_rows:
        if (
            previous_elapsed_seconds is not None
            and elapsed_seconds < previous_elapsed_seconds
        ):
            raise RuntimeError("Measurement timestamps are not ordered.")

        if first_measurement_seconds is None:
            first_measurement_seconds = elapsed_seconds

        last_measurement_seconds = elapsed_seconds

        current_amperes_eligible = None

        if step_type in {"21", "22"}:
            current_amperes_eligible = current_amperes

        if (
            step_type == "21"
            and current_amperes_eligible is not None
            and current_amperes_eligible > 0
        ):
            step_label_charge_sample_count += 1

            if (
                previous_step_type == "21"
                and previous_current_amperes is not None
                and previous_current_amperes > 0
                and previous_elapsed_seconds is not None
            ):
                step_label_charge_ampere_seconds += (
                    elapsed_seconds - previous_elapsed_seconds
                ) * current_amperes_eligible

                step_label_charge_interval_count += 1

        elif (
            step_type == "22"
            and current_amperes_eligible is not None
            and current_amperes_eligible < 0
        ):
            step_label_discharge_sample_count += 1

            if (
                previous_step_type == "22"
                and previous_current_amperes is not None
                and previous_current_amperes < 0
                and previous_elapsed_seconds is not None
            ):
                step_label_discharge_ampere_seconds += (
                    elapsed_seconds - previous_elapsed_seconds
                ) * abs(current_amperes_eligible)

                step_label_discharge_interval_count += 1

        previous_elapsed_seconds = elapsed_seconds
        previous_step_type = step_type
        previous_current_amperes = current_amperes_eligible

    if (
        first_measurement_seconds is None
        or last_measurement_seconds is None
    ):
        raise RuntimeError("The measurement does not contain data rows.")

    if (
        step_label_charge_interval_count == 0
        or step_label_discharge_interval_count == 0
        or step_label_charge_ampere_seconds <= 0
        or step_label_discharge_ampere_seconds <= 0
    ):
        raise FeatureUnavailableError(
            "The measurement does not contain complete contiguous positive charging and negative "
            "discharging capacity sections."
        )

    charge_capacity_ah = (
        step_label_charge_ampere_seconds
        / SECONDS_PER_HOUR
    )
    discharge_capacity_ah = (
        step_label_discharge_ampere_seconds
        / SECONDS_PER_HOUR
    )
    mean_capacity_ah = (
        charge_capacity_ah
        + discharge_capacity_ah
    ) / 2

    measurement_duration_days = (
        last_measurement_seconds
        - first_measurement_seconds
    ) / SECONDS_PER_DAY

    return {
        "charge_capacity_ah": charge_capacity_ah,
        "discharge_capacity_ah": discharge_capacity_ah,
        "mean_capacity_ah": mean_capacity_ah,
        "measurement_duration_days": (
            measurement_duration_days
        ),
        "step_label_charge_sample_count": (
            step_label_charge_sample_count
        ),
        "step_label_discharge_sample_count": (
            step_label_discharge_sample_count
        ),
        "step_label_charge_interval_count": (
            step_label_charge_interval_count
        ),
        "step_label_discharge_interval_count": (
            step_label_discharge_interval_count
        )
    }


# Integrating capacity from one reference-measurement stream
def extract_capacity(measurement_stream: io.TextIOBase) -> dict:
    return extract_capacity_from_rows(
        parse_reference_measurement_rows(
            measurement_stream
        )
    )


# Integrating contiguous cycling-current intervals
def extract_cycle_throughput(measurement_stream: io.TextIOBase, reference_capacity_ah: float = REFERENCE_CAPACITY_AH) -> dict:
    step_label_charge_sample_count = 0
    step_label_discharge_sample_count = 0
    step_label_charge_interval_count = 0
    step_label_discharge_interval_count = 0
    segmented_timestamp_reversal_count = 0
    direction_mismatch_count = 0
    direction_mismatch_interval_count = 0

    if (
        not math.isfinite(
            reference_capacity_ah
        )
        or reference_capacity_ah <= 0
    ):
        raise ValueError("Reference capacity must be positive and finite.")

    measurement_reader = csv.DictReader(
        measurement_stream
    )

    missing_columns = REQUIRED_COLUMNS.difference(
        measurement_reader.fieldnames
        or []
    )

    if missing_columns:
        raise RuntimeError(f"Cycling measurement is missing required columns : {sorted(missing_columns)}")

    step_label_charge_ampere_seconds = 0.0
    step_label_discharge_ampere_seconds = 0.0
    absolute_current_ampere_seconds = 0.0

    previous_elapsed_seconds = None
    previous_step_type = None
    previous_current_amperes = None

    direction_mismatch_ampere_seconds = 0.0
    maximum_direction_mismatch_interval_seconds = 0.0

    for row_number, measurement_row in enumerate(
        measurement_reader,
        start = 2
    ):
        if None in measurement_row:
            raise RuntimeError(f"CSV row {row_number} contains extra columns.")

        elapsed_seconds = parse_elapsed_seconds(
            measurement_row["run_time"]
        )

        current_amperes = float(
            measurement_row["c_cur"]
        )

        if not math.isfinite(
            current_amperes
        ):
            raise RuntimeError(f"CSV row {row_number} has a non-finite current.")

        step_type = parse_feature_step_type(
            value = measurement_row[
                "step_type"
            ],
            row_number = row_number
        )

        current_sample_is_eligible = (
            step_type in {"41", "42"}
            and abs(
                current_amperes
            ) > CYCLING_CURRENT_THRESHOLD_AMPERES
        )

        if current_sample_is_eligible:
            if step_type == "41":
                step_label_charge_sample_count += 1

                if current_amperes < 0:
                    direction_mismatch_count += 1

            else:
                step_label_discharge_sample_count += 1

                if current_amperes > 0:
                    direction_mismatch_count += 1

        if previous_elapsed_seconds is not None:
            time_difference_seconds = (
                elapsed_seconds
                - previous_elapsed_seconds
            )

            if time_difference_seconds < 0:
                segmented_timestamp_reversal_count += 1

                previous_elapsed_seconds = (
                    elapsed_seconds
                )
                previous_step_type = step_type
                previous_current_amperes = (
                    current_amperes
                )

                continue

            previous_sample_is_eligible = (
                previous_step_type in {"41", "42"}
                and previous_current_amperes is not None
                and abs(
                    previous_current_amperes
                ) > CYCLING_CURRENT_THRESHOLD_AMPERES
            )

            contiguous_cycling_interval = (
                current_sample_is_eligible
                and previous_sample_is_eligible
                and step_type == previous_step_type
            )

            if contiguous_cycling_interval:
                signed_ampere_seconds = (
                    time_difference_seconds
                    * current_amperes
                )

                # Tracking physical throughput within contiguous cycling sections
                absolute_current_ampere_seconds += abs(
                    signed_ampere_seconds
                )

                if step_type == "41":
                    step_label_charge_ampere_seconds += (
                        signed_ampere_seconds
                    )

                    step_label_charge_interval_count += 1

                    if current_amperes < 0:
                        direction_mismatch_interval_count += 1

                        direction_mismatch_ampere_seconds += abs(
                            signed_ampere_seconds
                        )

                        maximum_direction_mismatch_interval_seconds = max(
                            maximum_direction_mismatch_interval_seconds,
                            time_difference_seconds
                        )

                else:
                    step_label_discharge_ampere_seconds += (
                        signed_ampere_seconds
                    )

                    step_label_discharge_interval_count += 1

                    if current_amperes > 0:
                        direction_mismatch_interval_count += 1

                        direction_mismatch_ampere_seconds += abs(
                            signed_ampere_seconds
                        )

                        maximum_direction_mismatch_interval_seconds = max(
                            maximum_direction_mismatch_interval_seconds,
                            time_difference_seconds
                        )

        previous_elapsed_seconds = elapsed_seconds
        previous_step_type = step_type
        previous_current_amperes = current_amperes

    if absolute_current_ampere_seconds <= 0:
        raise FeatureUnavailableError(
            "Cycling measurement does not contain any usable contiguous current-throughput interval."
        )

    step_label_charge_throughput_ah = (
        step_label_charge_ampere_seconds
        / SECONDS_PER_HOUR
    )

    step_label_discharge_throughput_ah = (
        abs(
            step_label_discharge_ampere_seconds
        )
        / SECONDS_PER_HOUR
    )

    step_label_full_equivalent_cycles = (
        step_label_charge_throughput_ah
        + step_label_discharge_throughput_ah
    ) / (
        2
        * reference_capacity_ah
    )

    full_equivalent_cycles = (
        absolute_current_ampere_seconds
        / (
            2
            * reference_capacity_ah
            * SECONDS_PER_HOUR
        )
    )

    return {
        "full_equivalent_cycles": (
            full_equivalent_cycles
        ),
        "step_label_full_equivalent_cycles": (
            step_label_full_equivalent_cycles
        ),
        "step_label_fec_difference": (
            full_equivalent_cycles
            - step_label_full_equivalent_cycles
        ),
        "step_label_charge_throughput_ah": (
            step_label_charge_throughput_ah
        ),
        "step_label_discharge_throughput_ah": (
            step_label_discharge_throughput_ah
        ),
        "step_label_charge_interval_count": (
            step_label_charge_interval_count
        ),
        "step_label_discharge_interval_count": (
            step_label_discharge_interval_count
        ),
        "segmented_timestamp_reversal_count": (
            segmented_timestamp_reversal_count
        ),
        "direction_mismatch_interval_count": (
            direction_mismatch_interval_count
        ),
        "reference_capacity_ah": (
            reference_capacity_ah
        ),
        "step_label_charge_sample_count": (
            step_label_charge_sample_count
        ),
        "step_label_discharge_sample_count": (
            step_label_discharge_sample_count
        ),
        "direction_mismatch_count": (
            direction_mismatch_count
        ),
        "direction_mismatch_throughput_ah": (
            direction_mismatch_ampere_seconds
            / SECONDS_PER_HOUR
        ),
        "maximum_direction_mismatch_interval_seconds": (
            maximum_direction_mismatch_interval_seconds
        )
    }


# Interpolating one value within an ordered linear reference
def interpolate_linearly(independent_values: list[float], dependent_values: list[float],
                         target_value: float
) -> float:
    if (
        len(independent_values) != len(dependent_values)
        or len(independent_values) < 2
    ):
        raise ValueError("Linear interpolation requires matching reference arrays with at least two values.")

    if independent_values != sorted(independent_values):
        raise ValueError("Linear interpolation requires ordered independent values.")

    if len(set(independent_values)) != len(independent_values):
        raise ValueError("Linear interpolation requires unique independent values.")

    if not (
        independent_values[0]
        <= target_value
        <= independent_values[-1]
    ):
        raise ValueError(f"Interpolation target is outside the supported range : {target_value}")

    upper_index = bisect_left(
        independent_values,
        target_value
    )

    if upper_index == 0:
        return dependent_values[0]

    if upper_index == len(independent_values):
        return dependent_values[-1]

    if independent_values[upper_index] == target_value:
        return dependent_values[upper_index]

    lower_index = upper_index - 1

    lower_independent_value = independent_values[
        lower_index
    ]
    upper_independent_value = independent_values[
        upper_index
    ]

    interpolation_fraction = (
        target_value
        - lower_independent_value
    ) / (
        upper_independent_value
        - lower_independent_value
    )

    return (
        dependent_values[lower_index]
        + interpolation_fraction
        * (
            dependent_values[upper_index]
            - dependent_values[lower_index]
        )
    )


# Building an inverse OCV reference while resolving duplicate voltages
def build_inverse_ocv_reference(capacity_values_ah: list[float], voltage_values_volts: list[float]
) -> tuple[list[float], list[float]]:
    ordered_pairs = sorted(
        zip(
            voltage_values_volts,
            capacity_values_ah,
            strict = True
        )
    )

    grouped_capacities = {}

    for voltage_volts, capacity_ah in ordered_pairs:
        grouped_capacities.setdefault(
            voltage_volts,
            []
        ).append(
            capacity_ah
        )

    unique_voltages = sorted(
        grouped_capacities
    )

    mean_capacities = [
        sum(grouped_capacities[voltage_volts])
        / len(grouped_capacities[voltage_volts])
        for voltage_volts in unique_voltages
    ]

    if len(unique_voltages) < 2:
        raise RuntimeError("The OCV reference does not contain enough unique voltages.")

    return unique_voltages, mean_capacities


# Extracting the OCV reference from parsed measurement rows
def extract_ocv_reference_from_rows(measurement_rows: list[ReferenceMeasurementRow]) -> dict | None:
    section_voltages = {
        "31": [],
        "32": []
    }

    for (
        _elapsed_seconds,
        voltage_volts,
        _current_amperes,
        step_type
    ) in measurement_rows:
        if step_type not in section_voltages:
            continue

        section_voltages[step_type].append(
            voltage_volts
        )

    charge_voltages = section_voltages["31"]
    discharge_voltages = section_voltages["32"]

    if not charge_voltages and not discharge_voltages:
        return None

    if not charge_voltages or not discharge_voltages:
        raise RuntimeError("The measurement contains only one required OCV direction.")

    state_of_charge_fractions = [
        grid_index / OCV_GRID_INTERVAL_COUNT
        for grid_index in range(
            OCV_GRID_INTERVAL_COUNT + 1
        )
    ]

    mean_ocv_voltages = []

    for grid_index in range(
        OCV_GRID_INTERVAL_COUNT + 1
    ):
        charge_position = (
            grid_index
            * (len(charge_voltages) - 1)
            / OCV_GRID_INTERVAL_COUNT
        )
        discharge_position = (
            (OCV_GRID_INTERVAL_COUNT - grid_index)
            * (len(discharge_voltages) - 1)
            / OCV_GRID_INTERVAL_COUNT
        )
        charge_index = int(
            math.floor(charge_position + 0.5)
        )
        discharge_index = int(
            math.floor(discharge_position + 0.5)
        )
        mean_ocv_voltages.append(
            (
                charge_voltages[charge_index]
                + discharge_voltages[discharge_index]
            ) / 2
        )

    voltage_reversals = [
        mean_ocv_voltages[value_index - 1]
        - mean_ocv_voltages[value_index]
        for value_index in range(
            1,
            len(mean_ocv_voltages)
        )
        if (
            mean_ocv_voltages[value_index]
            < mean_ocv_voltages[value_index - 1]
        )
    ]

    return {
        "state_of_charge_fractions": (
            state_of_charge_fractions
        ),
        "mean_ocv_voltages": mean_ocv_voltages,
        "voltage_reversal_count": len(
            voltage_reversals
        ),
        "maximum_voltage_reversal_volts": (
            max(voltage_reversals, default = 0.0)
        )
    }


# Extracting the OCV reference from one measurement stream
def extract_ocv_reference(measurement_stream: io.TextIOBase) -> dict | None:
    return extract_ocv_reference_from_rows(
        parse_reference_measurement_rows(
            measurement_stream
        )
    )


# Extracting primary resistance from parsed measurement rows
def extract_dc_internal_resistance_from_rows(measurement_rows: list[ReferenceMeasurementRow]) -> dict:
    prior_elapsed_seconds = None

    for (
        elapsed_seconds,
        _voltage_volts,
        _current_amperes,
        _step_type
    ) in measurement_rows:
        if (
            prior_elapsed_seconds is not None
            and elapsed_seconds < prior_elapsed_seconds
        ):
            raise RuntimeError("Resistance-measurement timestamps are not ordered.")
        prior_elapsed_seconds = elapsed_seconds

    pulse_indices = [
        row_index
        for row_index, measurement_row in enumerate(
            measurement_rows
        )
        if measurement_row[3] == DCIR_PULSE_STEP_TYPE
    ]

    if not pulse_indices:
        raise FeatureUnavailableError("The measurement does not contain a step-5032 pulse.")

    expected_pulse_indices = list(
        range(
            pulse_indices[0],
            pulse_indices[-1] + 1
        )
    )

    if pulse_indices != expected_pulse_indices:
        raise RuntimeError("The measurement contains multiple disjoint step-5032 pulse sections.")

    pulse_start_seconds = measurement_rows[
        pulse_indices[0]
    ][0]

    pulse_end_seconds = measurement_rows[
        pulse_indices[-1]
    ][0]

    pulse_context_rows = [
        measurement_row
        for measurement_row in measurement_rows
        if (
            pulse_start_seconds - 1
            <= measurement_row[0]
            <= pulse_end_seconds + 1
        )
    ]

    if len(pulse_context_rows) < 3:
        raise RuntimeError("The resistance pulse does not contain sufficient context.")

    initial_voltage_volts = pulse_context_rows[0][1]

    voltage_trigger_index = next(
        (
            row_index
            for row_index, measurement_row in enumerate(
                pulse_context_rows
            )
            if abs(
                measurement_row[1]
                - initial_voltage_volts
            ) >= DCIR_BASELINE_VOLTAGE_CHANGE_VOLTS
        ),
        None
    )

    if (
        voltage_trigger_index is None
        or voltage_trigger_index == 0
    ):
        raise RuntimeError("The resistance pulse does not contain a valid pre-pulse voltage baseline.")

    baseline_voltage_volts = median(
        measurement_row[1]
        for measurement_row in pulse_context_rows[
            :voltage_trigger_index
        ]
    )

    current_change_indices = [
        row_index
        for row_index in range(
            1,
            len(pulse_context_rows)
        )
        if abs(
            pulse_context_rows[row_index][2]
            - pulse_context_rows[row_index - 1][2]
        ) > DCIR_CURRENT_CHANGE_AMPS
    ]

    if len(current_change_indices) < 2:
        raise RuntimeError("The resistance pulse does not contain both current transitions.")

    pulse_onset_index = current_change_indices[0]
    pulse_stop_index = current_change_indices[-1]

    pulse_rows = pulse_context_rows[
        pulse_onset_index - 1:
        pulse_stop_index
    ]

    pulse_time_origin_seconds = pulse_rows[0][0]

    relative_pulse_rows = [
        (
            measurement_row[0]
            - pulse_time_origin_seconds,
            measurement_row[1],
            measurement_row[2]
        )
        for measurement_row in pulse_rows
    ]

    evaluation_indices = [
        row_index
        for row_index, measurement_row in enumerate(
            relative_pulse_rows
        )
        if (
            measurement_row[0]
            <= DCIR_EVALUATION_SECONDS
        )
    ]

    if not evaluation_indices:
        raise RuntimeError("The resistance pulse does not reach the evaluation time.")

    evaluation_index = evaluation_indices[-1]

    if (
        DCIR_EVALUATION_SECONDS
        - relative_pulse_rows[evaluation_index][0]
        > 0.1
    ):
        raise RuntimeError("The resistance pulse has no sample sufficiently close to ten seconds.")

    if evaluation_index < 10:
        raise RuntimeError("The resistance pulse has insufficient samples for the current-stability check.")

    pulse_current_amperes = median(
        measurement_row[2]
        for measurement_row in relative_pulse_rows[
            :evaluation_index
        ]
    )

    if pulse_current_amperes >= -1:
        raise RuntimeError("The resistance pulse does not show the expected discharge current.")

    stability_current_amperes = relative_pulse_rows[
        evaluation_index - 10
    ][2]

    if abs(stability_current_amperes) < abs(
        0.9 * pulse_current_amperes
    ):
        raise RuntimeError("The resistance pulse fails the current-stability check.")

    pulse_charge_ampere_seconds = 0.0

    for row_index in range(
        1,
        evaluation_index + 1
    ):
        time_difference = (
            relative_pulse_rows[row_index][0]
            - relative_pulse_rows[row_index - 1][0]
        )

        pulse_charge_ampere_seconds += (
            time_difference
            * relative_pulse_rows[row_index][2]
        )

    pulse_charge_ah = (
        pulse_charge_ampere_seconds
        / SECONDS_PER_HOUR
    )

    measured_pulse_voltage_volts = (
        relative_pulse_rows[
            evaluation_index
        ][1]
    )

    uncorrected_dc_internal_resistance_ohms = abs(
        (
            measured_pulse_voltage_volts
            - baseline_voltage_volts
        )
        / pulse_current_amperes
    )

    return {
        "analytical_status": "primary",
        "baseline_voltage_volts": (
            baseline_voltage_volts
        ),
        "evaluated_pulse_voltage_volts": (
            measured_pulse_voltage_volts
        ),
        "pulse_current_amperes": (
            pulse_current_amperes
        ),
        "pulse_charge_ah": pulse_charge_ah,
        "evaluation_time_seconds": (
            relative_pulse_rows[
                evaluation_index
            ][0]
        ),
        "uncorrected_dc_internal_resistance_ohms": (
            uncorrected_dc_internal_resistance_ohms
        ),
        "uncorrected_dc_internal_resistance_milliohms": (
            uncorrected_dc_internal_resistance_ohms
            * 1000
        )
    }


# Extracting primary resistance from one measurement stream
def extract_dc_internal_resistance(measurement_stream: io.TextIOBase) -> dict:
    return extract_dc_internal_resistance_from_rows(
        parse_reference_measurement_rows(
            measurement_stream
        )
    )


# Calculating provisional OCV-corrected resistance
def calculate_ocv_corrected_dc_internal_resistance(dc_internal_resistance: dict, ocv_reference: dict,
                                                   discharge_capacity_ah: float
) -> dict:
    if (
        not math.isfinite(
            discharge_capacity_ah
        )
        or discharge_capacity_ah <= 0
    ):
        raise ValueError("Discharge capacity must be positive and finite.")

    state_of_charge_fractions = ocv_reference[
        "state_of_charge_fractions"
    ]

    mean_ocv_voltages = ocv_reference[
        "mean_ocv_voltages"
    ]

    capacity_values_ah = [
        state_of_charge_fraction
        * discharge_capacity_ah
        for state_of_charge_fraction in (
            state_of_charge_fractions
        )
    ]

    baseline_voltage_volts = (
        dc_internal_resistance[
            "baseline_voltage_volts"
        ]
    )

    measured_pulse_voltage_volts = (
        dc_internal_resistance[
            "evaluated_pulse_voltage_volts"
        ]
    )

    pulse_current_amperes = (
        dc_internal_resistance[
            "pulse_current_amperes"
        ]
    )

    pulse_charge_ah = dc_internal_resistance[
        "pulse_charge_ah"
    ]

    inverse_voltages, inverse_capacities_ah = (
        build_inverse_ocv_reference(
            capacity_values_ah = capacity_values_ah,
            voltage_values_volts = mean_ocv_voltages
        )
    )

    baseline_capacity_ah = interpolate_linearly(
        independent_values = inverse_voltages,
        dependent_values = inverse_capacities_ah,
        target_value = baseline_voltage_volts
    )

    evaluated_capacity_ah = (
        baseline_capacity_ah
        + pulse_charge_ah
    )

    evaluated_ocv_volts = interpolate_linearly(
        independent_values = capacity_values_ah,
        dependent_values = mean_ocv_voltages,
        target_value = evaluated_capacity_ah
    )

    ocv_change_volts = (
        evaluated_ocv_volts
        - baseline_voltage_volts
    )

    corrected_pulse_voltage_volts = (
        measured_pulse_voltage_volts
        - ocv_change_volts
    )

    corrected_resistance_ohms = abs(
        (
            corrected_pulse_voltage_volts
            - baseline_voltage_volts
        )
        / pulse_current_amperes
    )

    return {
        "analytical_status": (
            "provisional_sensitivity"
        ),
        "normalization_method": (
            "sample_position"
        ),
        "corrected_pulse_voltage_volts": (
            corrected_pulse_voltage_volts
        ),
        "ocv_change_volts": ocv_change_volts,
        "ocv_corrected_dc_internal_resistance_ohms": (
            corrected_resistance_ohms
        ),
        "ocv_corrected_dc_internal_resistance_milliohms": (
            corrected_resistance_ohms
            * 1000
        ),
        "ocv_voltage_reversal_count": (
            ocv_reference[
                "voltage_reversal_count"
            ]
        ),
        "maximum_ocv_voltage_reversal_volts": (
            ocv_reference[
                "maximum_voltage_reversal_volts"
            ]
        )
    }


# Loading the completed audit used for feature eligibility
def load_feature_audit(audit_path: Path) -> dict:
    with audit_path.open(
        mode = "r",
        encoding = "utf-8"
    ) as audit_stream:
        audit_result = json.load(
            audit_stream
        )

    if (
        audit_result[
            "archive_audit_failure_count"
        ] != 0
        or audit_result[
            "measurement_audit_failure_count"
        ] != 0
        or audit_result.get(
            "incident_rule_coverage_error"
        )
        is not None
    ):
        raise RuntimeError("Feature extraction requires a completed audit without execution failures.")

    return audit_result


# Validating the exact quality contract against the completed audit
def load_validated_quality_contract_binding(contract_path: Path, audit_result: dict) -> dict:
    contract_bytes = contract_path.read_bytes()

    quality_contract = json.loads(
        contract_bytes.decode(
            "utf-8"
        )
    )

    contract_version = quality_contract.get(
        "contract_version"
    )

    if (
        contract_version
        != audit_result["contract_version"]
    ):
        raise RuntimeError(
            "Data-quality contract version does not match the completed audit : "
            f"{contract_version} : {audit_result['contract_version']}"
        )

    observed_sha256 = hashlib.sha256(
        contract_bytes
    ).hexdigest()

    audit_binding = audit_result.get(
        "quality_contract"
    )

    if audit_binding is None:
        raise RuntimeError("Completed audit does not contain an exact quality-contract binding.")

    if (
        audit_binding.get(
            "contract_version"
        )
        != contract_version
    ):
        raise RuntimeError("Audit quality-contract version does not match the supplied contract.")

    if (
        audit_binding.get(
            "sha256"
        )
        != observed_sha256
    ):
        raise RuntimeError(
            "Data-quality contract SHA-256 does not match the completed audit : "
            f"{observed_sha256} : {audit_binding.get('sha256')}"
        )

    return {
        "file_name": contract_path.name,
        "contract_version": contract_version,
        "sha256": observed_sha256
    }


# Resolving sensitivity eligibility without overriding unresolved reviews
def resolve_sensitivity_eligibility(measurement_audit: dict) -> bool | None:
    disposition = measurement_audit[
        "disposition"
    ]

    if disposition in PRIMARY_COHORT_DISPOSITIONS:
        return True

    if disposition == "exclude":
        return False

    manual_review_reasons = [
        quality_reason
        for quality_reason in measurement_audit[
            "quality_reasons"
        ]
        if quality_reason[
            "disposition"
        ] == "manual_review"
    ]

    explicit_eligibility_values = [
        quality_reason.get(
            "details",
            {}
        ).get(
            "sensitivity_cohort_eligible"
        )
        for quality_reason in manual_review_reasons
    ]

    if (
        explicit_eligibility_values
        and all(
            eligibility_value is not None
            for eligibility_value in (
                explicit_eligibility_values
            )
        )
    ):
        return all(
            explicit_eligibility_values
        )

    return None


# Resolving eligibility across every required source record
def resolve_combined_eligibility(eligibility_values: list[bool | None]) -> bool | None:
    if any(
        eligibility_value is False
        for eligibility_value in eligibility_values
    ):
        return False

    if any(
        eligibility_value is None
        for eligibility_value in eligibility_values
    ):
        return None

    return True


# Confirming that one extracted value is finite
def finite_feature_available(feature_value: object) -> bool:
    return (
        isinstance(
            feature_value,
            (int, float)
        )
        and not isinstance(
            feature_value,
            bool
        )
        and math.isfinite(
            float(
                feature_value
            )
        )
    )


# Parsing one finite experiment-metadata value
def parse_experiment_number(value: str, field_name: str, cell_identity: str) -> float:
    try:
        numerical_value = float(
            value
        )

    except ValueError as exc:
        raise RuntimeError(
            f"Experiment metadata contains an invalid numerical value : {cell_identity} : {field_name} : {value!r}"
        ) from exc

    if not math.isfinite(
        numerical_value
    ):
        raise RuntimeError(
            f"Experiment metadata contains a non-finite value : {cell_identity} : {field_name} : {value!r}"
        )

    return numerical_value


# Parsing one required non-empty experiment-metadata text value
def parse_required_experiment_text(value: str, field_name: str, cell_identity: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"Experiment metadata contains an unavailable required text value : {cell_identity} : {field_name}"
        )

    return value


# Building the audited reference-measurement extraction inventory
def build_reference_measurement_inventory(audit_result: dict) -> list[dict]:
    measurement_inventory = []
    observed_source_identities = set()

    for archive_audit in audit_result[
        "archives"
    ]:
        if archive_audit[
            "audit_status"
        ] != "completed":
            raise RuntimeError(
                f"Feature inventory contains an incomplete archive audit : {archive_audit['archive_identity']}"
            )

        archive_disposition = archive_audit[
            "disposition"
        ]

        archive_quality_reason_codes = [
            quality_reason["reason_code"]
            for quality_reason in archive_audit[
                "quality_reasons"
            ]
        ]

        for measurement_audit in archive_audit[
            "measurements"
        ]:
            source_identity = measurement_audit[
                "source_identity"
            ]

            identity_match = REFERENCE_MEASUREMENT_PATTERN.fullmatch(
                source_identity
            )

            if identity_match is None:
                continue

            if source_identity in observed_source_identities:
                raise RuntimeError(
                    f"Feature inventory contains a duplicate measurement identity : {source_identity}"
                )

            observed_source_identities.add(
                source_identity
            )

            audit_status = measurement_audit[
                "audit_status"
            ]

            row_count = measurement_audit.get(
                "row_count",
                0
            )

            disposition = measurement_audit[
                "disposition"
            ]

            extraction_requested = (
                audit_status == "completed"
                and row_count > 0
                and disposition != "exclude"
            )

            companion_metadata = measurement_audit.get(
                "companion_metadata"
            )

            measurement_start_date = (
                None
                if companion_metadata is None
                else companion_metadata.get(
                    "measurement_start_date"
                )
            )

            resolved_sensitivity_eligibility = (
                resolve_sensitivity_eligibility(
                    measurement_audit
                )
            )

            measurement_inventory.append(
                {
                    "dataset_stage": identity_match.group(
                        "dataset_stage"
                    ),
                    "cell_identity": (
                        f"{identity_match.group('dataset_stage')}/"
                        f"{identity_match.group('cell_folder')}"
                    ),
                    "archive_identity": archive_audit[
                        "archive_identity"
                    ],
                    "cloud_archive": archive_audit[
                        "cloud_object"
                    ],
                    "measurement_file": (
                        f"{identity_match.group('cell_folder')}/"
                        f"{PurePosixPath(source_identity).name}"
                    ),
                    "source_identity": source_identity,
                    "sequence_number": int(
                        identity_match.group(
                            "sequence_number"
                        )
                    ),
                    "reference_test_protocol": (
                        identity_match.group(
                            "protocol"
                        )
                    ),
                    "audit_status": audit_status,
                    "disposition": disposition,
                    "archive_disposition": (
                        archive_disposition
                    ),
                    "archive_quality_reason_codes": (
                        archive_quality_reason_codes
                    ),
                    "archive_quality_reasons": (
                        archive_audit[
                            "quality_reasons"
                        ]
                    ),
                    "published_md5_available": (
                        archive_audit[
                            "published_md5_available"
                        ]
                    ),
                    "row_count": row_count,
                    "measurement_start_date": (
                        measurement_start_date
                    ),
                    "quality_reason_codes": [
                        quality_reason[
                            "reason_code"
                        ]
                        for quality_reason in measurement_audit[
                            "quality_reasons"
                        ]
                    ],
                    "audit_companion_metadata_available": (
                        companion_metadata is not None
                    ),
                    "audit_timestamp_reversal_count": (
                        measurement_audit.get(
                            "timestamp_reversal_count"
                        )
                    ),
                    "audit_run_time_representations": (
                        measurement_audit.get(
                            "run_time_representations",
                            []
                        )
                    ),
                    "audit_step_type_representations": (
                        measurement_audit.get(
                            "step_type_representations",
                            []
                        )
                    ),
                    "audit_source_row_index_present": (
                        measurement_audit.get(
                            "source_row_index_present"
                        )
                    ),
                    "audit_elapsed_seconds_column_present": (
                        measurement_audit.get(
                            "elapsed_seconds_column_present"
                        )
                    ),
                    "feature_extraction_requested": (
                        extraction_requested
                    ),
                    "primary_cohort_eligible": (
                        extraction_requested
                        and disposition
                        in PRIMARY_COHORT_DISPOSITIONS
                    ),
                    "sensitivity_cohort_eligible": (
                        resolved_sensitivity_eligibility
                        if extraction_requested
                        else False
                    )
                }
            )

    protocol_order = {
        "ET_T23": 0,
        "CU": 1,
        "exCU": 2,
        "AT_T23": 3
    }

    return sorted(
        measurement_inventory,
        key = lambda measurement_record: (
            measurement_record[
                "dataset_stage"
            ],
            measurement_record[
                "cell_identity"
            ],
            measurement_record[
                "sequence_number"
            ],
            protocol_order[
                measurement_record[
                    "reference_test_protocol"
                ]
            ]
        )
    )


# Building the audited cycling-measurement extraction inventory
def build_cycling_measurement_inventory(audit_result: dict) -> list[dict]:
    measurement_inventory = []
    observed_source_identities = set()

    for archive_audit in audit_result[
        "archives"
    ]:
        if archive_audit[
            "audit_status"
        ] != "completed":
            raise RuntimeError(
                f"Cycling inventory contains an incomplete archive audit : {archive_audit['archive_identity']}"
            )

        for measurement_audit in archive_audit[
            "measurements"
        ]:
            source_identity = measurement_audit[
                "source_identity"
            ]

            identity_match = (
                CYCLING_MEASUREMENT_PATTERN.fullmatch(
                    source_identity
                )
            )

            if identity_match is None:
                continue

            if source_identity in observed_source_identities:
                raise RuntimeError(f"Cycling inventory contains a duplicate measurement identity : {source_identity}")

            observed_source_identities.add(
                source_identity
            )

            audit_status = measurement_audit[
                "audit_status"
            ]

            row_count = measurement_audit.get(
                "row_count",
                0
            )

            disposition = measurement_audit[
                "disposition"
            ]

            extraction_requested = (
                audit_status == "completed"
                and row_count > 0
                and disposition != "exclude"
            )

            sensitivity_cohort_eligible = (
                resolve_sensitivity_eligibility(
                    measurement_audit
                )
            )

            companion_metadata = measurement_audit.get(
                "companion_metadata"
            )

            measurement_inventory.append(
                {
                    "dataset_stage": identity_match.group(
                        "dataset_stage"
                    ),
                    "cell_identity": (
                        f"{identity_match.group('dataset_stage')}/"
                        f"{identity_match.group('cell_folder')}"
                    ),
                    "archive_identity": archive_audit[
                        "archive_identity"
                    ],
                    "cloud_archive": archive_audit[
                        "cloud_object"
                    ],
                    "measurement_file": (
                        f"{identity_match.group('cell_folder')}/"
                        f"{PurePosixPath(source_identity).name}"
                    ),
                    "source_identity": source_identity,
                    "sequence_number": int(
                        identity_match.group(
                            "sequence_number"
                        )
                    ),
                    "published_md5_available": (
                        archive_audit[
                            "published_md5_available"
                        ]
                    ),
                    "archive_disposition": archive_audit[
                        "disposition"
                    ],
                    "archive_quality_reason_codes": [
                        quality_reason["reason_code"]
                        for quality_reason in archive_audit[
                            "quality_reasons"
                        ]
                    ],
                    "archive_quality_reasons": (
                        archive_audit[
                            "quality_reasons"
                        ]
                    ),
                    "audit_status": audit_status,
                    "disposition": disposition,
                    "quality_reason_codes": [
                        quality_reason["reason_code"]
                        for quality_reason in measurement_audit[
                            "quality_reasons"
                        ]
                    ],
                    "row_count": row_count,
                    "audit_companion_metadata_available": (
                        companion_metadata is not None
                    ),
                    "audit_timestamp_reversal_count": (
                        measurement_audit.get(
                            "timestamp_reversal_count"
                        )
                    ),
                    "audit_run_time_representations": (
                        measurement_audit.get(
                            "run_time_representations",
                            []
                        )
                    ),
                    "audit_step_type_representations": (
                        measurement_audit.get(
                            "step_type_representations",
                            []
                        )
                    ),
                    "audit_source_row_index_present": (
                        measurement_audit.get(
                            "source_row_index_present"
                        )
                    ),
                    "audit_elapsed_seconds_column_present": (
                        measurement_audit.get(
                            "elapsed_seconds_column_present"
                        )
                    ),
                    "feature_extraction_requested": (
                        extraction_requested
                    ),
                    "primary_exposure_eligible": (
                        extraction_requested
                        and disposition
                        in PRIMARY_COHORT_DISPOSITIONS
                    ),
                    "sensitivity_exposure_eligible": (
                        sensitivity_cohort_eligible
                        if extraction_requested
                        else False
                    )
                }
            )

    return sorted(
        measurement_inventory,
        key = lambda measurement_record: (
            measurement_record[
                "dataset_stage"
            ],
            measurement_record[
                "cell_identity"
            ],
            measurement_record[
                "sequence_number"
            ]
        )
    )


# Hashing a deterministic inventory of source identities
def calculate_identity_inventory_sha256(source_identities: set[str]) -> str:
    serialized_inventory = (
        json.dumps(
            sorted(
                source_identities
            ),
            ensure_ascii = False,
            separators = (
                ",",
                ":"
            )
        )
        + "\n"
    )

    return hashlib.sha256(
        serialized_inventory.encode(
            "utf-8"
        )
    ).hexdigest()


# Reconciling every audited measurement with extraction scope
def build_source_measurement_reconciliation(audit_result: dict, reference_measurement_inventory: list[dict],
                                            cycling_measurement_inventory: list[dict]
) -> dict:
    audited_source_identities = [
        measurement_record[
            "source_identity"
        ]
        for archive_record in audit_result[
            "archives"
        ]
        for measurement_record in archive_record[
            "measurements"
        ]
    ]

    if len(
        audited_source_identities
    ) != len(
        set(
            audited_source_identities
        )
    ):
        raise RuntimeError("Completed audit contains duplicate measurement source identities.")

    audited_source_identity_set = set(
        audited_source_identities
    )

    analytical_reference_identities = {
        record["source_identity"]
        for record in reference_measurement_inventory
    }

    analytical_cycling_identities = {
        record["source_identity"]
        for record in cycling_measurement_inventory
    }

    overlapping_analytical_identities = (
        analytical_reference_identities.intersection(
            analytical_cycling_identities
        )
    )

    if overlapping_analytical_identities:
        raise RuntimeError(
            "Measurements appear in both analytical reference and cycling inventories : "
            f"{sorted(overlapping_analytical_identities)[:20]}"
        )

    analytical_identities = (
        analytical_reference_identities
        | analytical_cycling_identities
    )

    unknown_analytical_identities = (
        analytical_identities.difference(
            audited_source_identity_set
        )
    )

    if unknown_analytical_identities:
        raise RuntimeError(
            "Analytical inventories contain measurements absent from the controlling audit : "
            f"{sorted(unknown_analytical_identities)[:20]}"
        )

    intentionally_not_extracted_identities = (
        audited_source_identity_set.difference(
            analytical_identities
        )
    )

    intentionally_not_extracted_protocol_counts = Counter()

    unexpected_unaccounted_identities = []

    for source_identity in sorted(
        intentionally_not_extracted_identities
    ):
        identity_match = (
            NON_ANALYTICAL_REFERENCE_MEASUREMENT_PATTERN.fullmatch(
                source_identity
            )
        )

        if identity_match is None:
            unexpected_unaccounted_identities.append(
                source_identity
            )

            continue

        intentionally_not_extracted_protocol_counts[
            identity_match.group(
                "protocol"
            )
        ] += 1

    if unexpected_unaccounted_identities:
        raise RuntimeError(
            "Audited measurements are neither analytically extracted nor covered by the approved non-analytical "
            f"reference scope : {unexpected_unaccounted_identities[:20]}"
        )

    reconciled_measurement_count = (
        len(
            analytical_reference_identities
        )
        + len(
            analytical_cycling_identities
        )
        + len(
            intentionally_not_extracted_identities
        )
    )

    audited_measurement_count = len(
        audited_source_identity_set
    )

    if (
        reconciled_measurement_count
        != audited_measurement_count
    ):
        raise RuntimeError(
            f"Audited measurement reconciliation failed : {reconciled_measurement_count} : {audited_measurement_count}"
        )

    return {
        "audited_measurement_count": (
            audited_measurement_count
        ),
        "analytical_reference_measurement_count": len(
            analytical_reference_identities
        ),
        "analytical_cycling_measurement_count": len(
            analytical_cycling_identities
        ),
        "analytical_measurement_count": len(
            analytical_identities
        ),
        "intentionally_not_extracted_measurement_count": len(
            intentionally_not_extracted_identities
        ),
        "intentionally_not_extracted_protocol_counts": dict(
            sorted(
                intentionally_not_extracted_protocol_counts.items()
            )
        ),
        "intentionally_not_extracted_reason": (
            "reference_temperature_measurement_outside_"
            "approved_23_celsius_longitudinal_outcome_definition"
        ),
        "intentionally_not_extracted_source_identity_sha256": (
            calculate_identity_inventory_sha256(
                intentionally_not_extracted_identities
            )
        ),
        "reconciled_measurement_count": (
            reconciled_measurement_count
        ),
        "reconciliation_complete": True
    }


# Adding gross calendar time from audited measurement dates
def add_elapsed_calendar_time(reference_test_records: list[dict]) -> None:
    records_by_cell = {}

    for reference_test_record in (
        reference_test_records
    ):
        records_by_cell.setdefault(
            reference_test_record[
                "cell_identity"
            ],
            []
        ).append(
            reference_test_record
        )

    for cell_identity, cell_records in (
        records_by_cell.items()
    ):
        baseline_records = [
            reference_test_record
            for reference_test_record in cell_records
            if (
                reference_test_record[
                    "reference_test_protocol"
                ]
                == "ET_T23"
            )
        ]

        if len(baseline_records) != 1:
            raise RuntimeError(
                f"Cell does not have exactly one initial 23°C reference test : {cell_identity} : {len(baseline_records)}"
            )

        baseline_start_date = baseline_records[0].get(
            "measurement_start_date"
        )

        if baseline_start_date is None:
            for reference_test_record in cell_records:
                reference_test_record[
                    "elapsed_calendar_days"
                ] = None

                reference_test_record[
                    "elapsed_calendar_time_status"
                ] = "baseline_start_date_unavailable"

            continue

        try:
            parsed_baseline_date = date.fromisoformat(
                baseline_start_date
            )

        except ValueError as exc:
            raise RuntimeError(
                f"Baseline measurement start date is not ISO formatted : {cell_identity} : {baseline_start_date}"
            ) from exc

        for reference_test_record in cell_records:
            measurement_start_date = (
                reference_test_record.get(
                    "measurement_start_date"
                )
            )

            if measurement_start_date is None:
                reference_test_record[
                    "elapsed_calendar_days"
                ] = None

                reference_test_record[
                    "elapsed_calendar_time_status"
                ] = "measurement_start_date_unavailable"

                continue

            try:
                parsed_measurement_date = date.fromisoformat(
                    measurement_start_date
                )

            except ValueError as exc:
                raise RuntimeError(
                    "Measurement start date is not ISO formatted : "
                    f"{reference_test_record['source_identity']} : {measurement_start_date}"
                ) from exc

            elapsed_calendar_days = (
                parsed_measurement_date
                - parsed_baseline_date
            ).days

            if elapsed_calendar_days < 0:
                raise RuntimeError(
                    "Reference-test date precedes the cell baseline : "
                    f"{reference_test_record['source_identity']} : {elapsed_calendar_days}"
                )

            reference_test_record[
                "elapsed_calendar_days"
            ] = elapsed_calendar_days

            reference_test_record[
                "elapsed_calendar_time_status"
            ] = "completed"


# Adding relative outcomes with baseline-dependent eligibility
def add_relative_reference_outcomes(reference_test_records: list[dict]) -> None:
    records_by_cell = {}

    for reference_test_record in (
        reference_test_records
    ):
        records_by_cell.setdefault(
            reference_test_record[
                "cell_identity"
            ],
            []
        ).append(
            reference_test_record
        )

    for cell_identity, cell_records in (
        records_by_cell.items()
    ):
        baseline_records = [
            reference_test_record
            for reference_test_record in cell_records
            if (
                reference_test_record[
                    "reference_test_protocol"
                ]
                == "ET_T23"
            )
        ]

        if len(baseline_records) != 1:
            raise RuntimeError(
                f"Cell does not have exactly one initial 23°C reference test : {cell_identity} : {len(baseline_records)}"
            )

        baseline_record = baseline_records[0]

        baseline_capacity_ah = (
            baseline_record.get(
                "mean_capacity_ah"
            )
        )

        baseline_resistance_ohms = (
            baseline_record.get(
                "uncorrected_dc_internal_resistance_ohms"
            )
        )

        baseline_corrected_resistance_ohms = (
            baseline_record.get(
                "ocv_corrected_dc_internal_resistance_ohms"
            )
        )

        baseline_capacity_available = (
            finite_feature_available(
                baseline_capacity_ah
            )
            and baseline_capacity_ah > 0
        )

        baseline_resistance_available = (
            finite_feature_available(
                baseline_resistance_ohms
            )
            and baseline_resistance_ohms > 0
        )

        baseline_corrected_resistance_available = (
            finite_feature_available(
                baseline_corrected_resistance_ohms
            )
            and baseline_corrected_resistance_ohms > 0
        )

        for reference_test_record in cell_records:
            reference_test_record[
                "baseline_source_identity"
            ] = baseline_record[
                "source_identity"
            ]

            current_capacity_ah = (
                reference_test_record.get(
                    "mean_capacity_ah"
                )
            )

            current_resistance_ohms = (
                reference_test_record.get(
                    "uncorrected_dc_internal_resistance_ohms"
                )
            )

            current_corrected_resistance_ohms = (
                reference_test_record.get(
                    "ocv_corrected_dc_internal_resistance_ohms"
                )
            )

            capacity_available = (
                baseline_capacity_available
                and finite_feature_available(
                    current_capacity_ah
                )
                and current_capacity_ah > 0
            )

            resistance_available = (
                baseline_resistance_available
                and finite_feature_available(
                    current_resistance_ohms
                )
                and current_resistance_ohms > 0
            )

            corrected_resistance_available = (
                baseline_corrected_resistance_available
                and finite_feature_available(
                    current_corrected_resistance_ohms
                )
                and current_corrected_resistance_ohms > 0
            )

            if capacity_available:
                relative_capacity = (
                    current_capacity_ah
                    / baseline_capacity_ah
                )

                reference_test_record[
                    "relative_capacity"
                ] = relative_capacity

                reference_test_record[
                    "capacity_loss_percent"
                ] = (
                    1
                    - relative_capacity
                ) * 100

            else:
                reference_test_record[
                    "relative_capacity"
                ] = None

                reference_test_record[
                    "capacity_loss_percent"
                ] = None

            if resistance_available:
                relative_resistance = (
                    current_resistance_ohms
                    / baseline_resistance_ohms
                )

                reference_test_record[
                    "relative_dc_internal_resistance"
                ] = relative_resistance

                reference_test_record[
                    "dc_internal_resistance_growth_percent"
                ] = (
                    relative_resistance
                    - 1
                ) * 100

            else:
                reference_test_record[
                    "relative_dc_internal_resistance"
                ] = None

                reference_test_record[
                    "dc_internal_resistance_growth_percent"
                ] = None

            if corrected_resistance_available:
                relative_corrected_resistance = (
                    current_corrected_resistance_ohms
                    / baseline_corrected_resistance_ohms
                )

                reference_test_record[
                    "relative_ocv_corrected_dc_internal_resistance"
                ] = relative_corrected_resistance

                reference_test_record[
                    "ocv_corrected_dc_internal_resistance_growth_percent"
                ] = (
                    relative_corrected_resistance
                    - 1
                ) * 100

            else:
                reference_test_record[
                    "relative_ocv_corrected_dc_internal_resistance"
                ] = None

                reference_test_record[
                    "ocv_corrected_dc_internal_resistance_growth_percent"
                ] = None

            reference_test_record[
                "relative_capacity_primary_eligible"
            ] = (
                capacity_available
                and resolve_combined_eligibility(
                    [
                        baseline_record[
                            "primary_cohort_eligible"
                        ],
                        reference_test_record[
                            "primary_cohort_eligible"
                        ]
                    ]
                )
                is True
            )

            reference_test_record[
                "relative_dc_internal_resistance_primary_eligible"
            ] = (
                resistance_available
                and resolve_combined_eligibility(
                    [
                        baseline_record[
                            "primary_cohort_eligible"
                        ],
                        reference_test_record[
                            "primary_cohort_eligible"
                        ]
                    ]
                )
                is True
            )

            reference_test_record[
                "relative_capacity_sensitivity_eligible"
            ] = (
                False
                if not capacity_available
                else resolve_combined_eligibility(
                    [
                        baseline_record[
                            "sensitivity_cohort_eligible"
                        ],
                        reference_test_record[
                            "sensitivity_cohort_eligible"
                        ]
                    ]
                )
            )

            reference_test_record[
                "relative_dc_internal_resistance_sensitivity_eligible"
            ] = (
                False
                if not resistance_available
                else resolve_combined_eligibility(
                    [
                        baseline_record[
                            "sensitivity_cohort_eligible"
                        ],
                        reference_test_record[
                            "sensitivity_cohort_eligible"
                        ]
                    ]
                )
            )

            reference_test_record[
                "ocv_corrected_resistance_sensitivity_eligible"
            ] = (
                False
                if not corrected_resistance_available
                else resolve_combined_eligibility(
                    [
                        baseline_record[
                            "sensitivity_cohort_eligible"
                        ],
                        reference_test_record[
                            "sensitivity_cohort_eligible"
                        ]
                    ]
                )
            )


# Reading unobserved cycling exposure from archive incidents
def find_unobserved_cycling_sequences(archive_quality_reasons: list[dict]) -> set[int]:
    missing_sequence_numbers = set()

    for quality_reason in archive_quality_reasons:
        reason_code = quality_reason[
            "reason_code"
        ]

        if (
            reason_code
            not in CYCLING_EXPOSURE_INCIDENT_REASON_CODES
        ):
            continue

        reason_details = quality_reason.get(
            "details"
        )

        if reason_details is None:
            raise RuntimeError(f"Cycling-exposure incident is missing machine-readable details : {reason_code}")

        required_detail_fields = {
            "cycling_performed",
            "missing_sequence_numbers"
        }

        missing_detail_fields = (
            required_detail_fields.difference(
                reason_details
            )
        )

        if missing_detail_fields:
            raise RuntimeError("Cycling-exposure incident is missing required detail fields : "
                               f"{reason_code} : {sorted(missing_detail_fields)}"
            )

        cycling_performed = reason_details[
            "cycling_performed"
        ]

        incident_sequence_numbers = reason_details[
            "missing_sequence_numbers"
        ]

        if not isinstance(
            cycling_performed,
            bool
        ):
            raise RuntimeError(f"Cycling-exposure incident contains an invalid cycling-performed value : {reason_code}")

        if (
            not isinstance(
                incident_sequence_numbers,
                list
            )
            or not incident_sequence_numbers
            or any(
                not isinstance(
                    sequence_number,
                    int
                )
                or isinstance(
                    sequence_number,
                    bool
                )
                or sequence_number <= 0
                for sequence_number in (
                    incident_sequence_numbers
                )
            )
        ):
            raise RuntimeError(f"Cycling-exposure incident contains invalid missing sequence numbers : {reason_code}")

        if cycling_performed:
            missing_sequence_numbers.update(
                incident_sequence_numbers
            )

    return missing_sequence_numbers


# Adding observed cumulative FEC and exposure eligibility
def add_audited_cumulative_cycle_exposure(reference_test_records: list[dict], cycling_measurement_records: list[dict]
) -> None:
    cycling_records_by_cell = {}

    for cycling_measurement_record in (
        cycling_measurement_records
    ):
        cycling_records_by_cell.setdefault(
            cycling_measurement_record[
                "cell_identity"
            ],
            []
        ).append(
            cycling_measurement_record
        )

    for reference_test_record in (
        reference_test_records
    ):
        cell_identity = reference_test_record[
            "cell_identity"
        ]

        cell_folder = PurePosixPath(
            cell_identity
        ).name

        if cell_folder.startswith(
            "TP_k"
        ):
            reference_test_record[
                "cumulative_full_equivalent_cycles_observed"
            ] = None

            reference_test_record[
                "cumulative_step_label_full_equivalent_cycles_observed"
            ] = None

            reference_test_record[
                "cumulative_step_label_fec_difference_observed"
            ] = None

            reference_test_record[
                "cumulative_fec_observation_complete"
            ] = None

            reference_test_record[
                "cumulative_fec_primary_eligible"
            ] = False

            reference_test_record[
                "cumulative_fec_sensitivity_eligible"
            ] = False

            reference_test_record[
                "cumulative_fec_status"
            ] = "not_applicable"

            reference_test_record[
                "unavailable_cycling_sequence_numbers"
            ] = []

            reference_test_record[
                "partially_observed_cycling_sequence_numbers"
            ] = []

            reference_test_record[
                "temporally_discontinuous_cycling_sequence_numbers"
            ] = []

            continue

        reference_sequence_number = (
            reference_test_record[
                "sequence_number"
            ]
        )

        preceding_cycling_records = [
            cycling_measurement_record
            for cycling_measurement_record in (
                cycling_records_by_cell.get(
                    cell_identity,
                    []
                )
            )
            if (
                cycling_measurement_record[
                    "sequence_number"
                ]
                < reference_sequence_number
            )
        ]

        missing_sequence_numbers = (
            find_unobserved_cycling_sequences(
                reference_test_record[
                    "archive_quality_reasons"
                ]
            )
        )

        preceding_missing_sequence_numbers = sorted(
            sequence_number
            for sequence_number in (
                missing_sequence_numbers
            )
            if sequence_number < reference_sequence_number
        )

        extracted_cycling_records = [
            cycling_measurement_record
            for cycling_measurement_record in (
                preceding_cycling_records
            )
            if (
                cycling_measurement_record.get(
                    "feature_extraction_status"
                )
                == "completed"
                and finite_feature_available(
                    cycling_measurement_record.get(
                        "full_equivalent_cycles"
                    )
                )
            )
        ]

        unavailable_sequence_numbers = sorted(
            {
                *preceding_missing_sequence_numbers,
                *(
                    cycling_measurement_record[
                        "sequence_number"
                    ]
                    for cycling_measurement_record in (
                        preceding_cycling_records
                    )
                    if cycling_measurement_record not in (
                        extracted_cycling_records
                    )
                )
            }
        )

        partially_observed_sequence_numbers = sorted(
            {
                cycling_measurement_record[
                    "sequence_number"
                ]
                for cycling_measurement_record in (
                    preceding_cycling_records
                )
                if PARTIALLY_OBSERVED_CYCLING_REASON_CODES.intersection(
                    cycling_measurement_record[
                        "quality_reason_codes"
                    ]
                )
            }
        )

        temporally_discontinuous_sequence_numbers = sorted(
            {
                cycling_measurement_record[
                    "sequence_number"
                ]
                for cycling_measurement_record in (
                    preceding_cycling_records
                )
                if (
                    cycling_measurement_record.get(
                        "segmented_timestamp_reversal_count"
                    )
                    or 0
                ) > 0
            }
        )

        observed_full_equivalent_cycles = sum(
            cycling_measurement_record[
                "full_equivalent_cycles"
            ]
            for cycling_measurement_record in (
                extracted_cycling_records
            )
        )

        observed_step_label_fec = sum(
            cycling_measurement_record[
                "step_label_full_equivalent_cycles"
            ]
            for cycling_measurement_record in (
                extracted_cycling_records
            )
        )

        observation_complete = (
            not unavailable_sequence_numbers
            and not partially_observed_sequence_numbers
            and not temporally_discontinuous_sequence_numbers
        )

        primary_eligibility = resolve_combined_eligibility(
            [
                observation_complete,
                *(
                    cycling_measurement_record[
                        "primary_exposure_eligible"
                    ]
                    for cycling_measurement_record in (
                        preceding_cycling_records
                    )
                )
            ]
        )

        sensitivity_eligibility = (
            resolve_combined_eligibility(
                [
                    observation_complete,
                    *(
                        cycling_measurement_record[
                            "sensitivity_exposure_eligible"
                        ]
                        for cycling_measurement_record in (
                            preceding_cycling_records
                        )
                    )
                ]
            )
        )

        reference_test_record[
            "cumulative_full_equivalent_cycles_observed"
        ] = observed_full_equivalent_cycles

        reference_test_record[
            "cumulative_step_label_full_equivalent_cycles_observed"
        ] = observed_step_label_fec

        reference_test_record[
            "cumulative_step_label_fec_difference_observed"
        ] = (
            observed_full_equivalent_cycles
            - observed_step_label_fec
        )

        reference_test_record[
            "cumulative_fec_observation_complete"
        ] = observation_complete

        reference_test_record[
            "cumulative_fec_primary_eligible"
        ] = primary_eligibility

        reference_test_record[
            "cumulative_fec_sensitivity_eligible"
        ] = sensitivity_eligibility

        reference_test_record[
            "cumulative_fec_status"
        ] = (
            "complete"
            if observation_complete
            else "incomplete"
        )

        reference_test_record[
            "unavailable_cycling_sequence_numbers"
        ] = unavailable_sequence_numbers

        reference_test_record[
            "partially_observed_cycling_sequence_numbers"
        ] = (
            partially_observed_sequence_numbers
        )

        reference_test_record[
            "temporally_discontinuous_cycling_sequence_numbers"
        ] = (
            temporally_discontinuous_sequence_numbers
        )


# Extracting one audited cycling-measurement record
def extract_cycling_measurement_record(archive: zipfile.ZipFile, inventory_record: dict) -> dict:
    feature_record = {
        **inventory_record,
        "feature_extraction_status": (
            "not_requested"
        ),
        "feature_extraction_error": None,
        "full_equivalent_cycles": None,
        "step_label_full_equivalent_cycles": None,
        "step_label_fec_difference": None,
        "step_label_charge_throughput_ah": None,
        "step_label_discharge_throughput_ah": None,
        "step_label_charge_sample_count": None,
        "step_label_discharge_sample_count": None,
        "step_label_charge_interval_count": None,
        "step_label_discharge_interval_count": None,
        "segmented_timestamp_reversal_count": None,
        "direction_mismatch_count": None,
        "direction_mismatch_interval_count": None,
        "direction_mismatch_throughput_ah": None,
        "maximum_direction_mismatch_interval_seconds": None,
        "reference_capacity_ah": None
    }

    if not inventory_record[
        "feature_extraction_requested"
    ]:
        return feature_record

    try:
        with archive.open(
            inventory_record[
                "measurement_file"
            ],
            mode = "r"
        ) as binary_stream:
            with io.TextIOWrapper(
                binary_stream,
                encoding = "utf-8-sig",
                newline = ""
            ) as text_stream:
                throughput_result = (
                    extract_cycle_throughput(
                        text_stream
                    )
                )

    except FeatureUnavailableError as exc:
        feature_record[
            "feature_extraction_status"
        ] = "unavailable"

        feature_record[
            "feature_extraction_error"
        ] = str(
            exc
        )

        return feature_record

    except (
        RuntimeError,
        ValueError,
        UnicodeDecodeError
    ) as exc:
        feature_record[
            "feature_extraction_status"
        ] = "failed"

        feature_record[
            "feature_extraction_error"
        ] = str(
            exc
        )

        return feature_record

    feature_record.update(
        throughput_result
    )

    feature_record[
        "feature_extraction_status"
    ] = "completed"

    return feature_record


# Extracting one audited reference-test record
def extract_reference_test_record(archive: zipfile.ZipFile, inventory_record: dict,
                                  initial_ocv_reference: dict | None, initial_ocv_source_identity: str | None
) -> tuple[dict, dict | None]:
    feature_record = {
        **inventory_record,
        "feature_extraction_status": (
            "not_requested"
        ),
        "capacity_extraction_status": (
            "not_requested"
        ),
        "capacity_extraction_error": None,
        "dc_internal_resistance_extraction_status": (
            "not_requested"
        ),
        "dc_internal_resistance_extraction_error": (
            None
        ),
        "ocv_corrected_resistance_extraction_status": (
            "not_requested"
        ),
        "ocv_corrected_resistance_extraction_error": (
            None
        ),
        "ocv_reference_source_identity": None,
        "ocv_normalization_method": None,
        "charge_capacity_ah": None,
        "discharge_capacity_ah": None,
        "mean_capacity_ah": None,
        "measurement_duration_days": None,
        "step_label_charge_sample_count": None,
        "step_label_discharge_sample_count": None,
        "step_label_charge_interval_count": None,
        "step_label_discharge_interval_count": None,
        "baseline_voltage_volts": None,
        "evaluated_pulse_voltage_volts": None,
        "pulse_current_amperes": None,
        "pulse_charge_ah": None,
        "evaluation_time_seconds": None,
        "uncorrected_dc_internal_resistance_ohms": (
            None
        ),
        "uncorrected_dc_internal_resistance_milliohms": (
            None
        ),
        "corrected_pulse_voltage_volts": None,
        "ocv_change_volts": None,
        "ocv_corrected_dc_internal_resistance_ohms": (
            None
        ),
        "ocv_corrected_dc_internal_resistance_milliohms": (
            None
        ),
        "ocv_voltage_reversal_count": None,
        "maximum_ocv_voltage_reversal_volts": None
    }

    if not inventory_record[
        "feature_extraction_requested"
    ]:
        return feature_record, None

    measurement_file = inventory_record[
        "measurement_file"
    ]

    try:
        with archive.open(
            measurement_file,
            mode = "r"
        ) as binary_stream:
            with io.TextIOWrapper(
                binary_stream,
                encoding = "utf-8-sig",
                newline = ""
            ) as text_stream:
                measurement_rows = (
                    parse_reference_measurement_rows(
                        text_stream
                    )
                )

    except (
        RuntimeError,
        ValueError,
        UnicodeDecodeError
    ) as exc:
        feature_record["capacity_extraction_status"] = "failed"
        feature_record["capacity_extraction_error"] = str(exc)
        feature_record[
            "dc_internal_resistance_extraction_status"
        ] = "failed"
        feature_record[
            "dc_internal_resistance_extraction_error"
        ] = str(exc)
        feature_record["feature_extraction_status"] = "failed"
        return feature_record, None

    try:
        capacity_result = extract_capacity_from_rows(
            measurement_rows
        )
        feature_record.update(capacity_result)
        feature_record["capacity_extraction_status"] = "completed"
    except FeatureUnavailableError as exc:
        feature_record["capacity_extraction_status"] = "unavailable"
        feature_record["capacity_extraction_error"] = str(exc)
    except (RuntimeError, ValueError) as exc:
        feature_record["capacity_extraction_status"] = "failed"
        feature_record["capacity_extraction_error"] = str(exc)

    try:
        resistance_result = (
            extract_dc_internal_resistance_from_rows(
                measurement_rows
            )
        )
        for field_name, field_value in (
            resistance_result.items()
        ):
            if field_name == "analytical_status":
                continue
            feature_record[field_name] = field_value
        feature_record[
            "dc_internal_resistance_extraction_status"
        ] = "completed"
    except FeatureUnavailableError as exc:
        feature_record[
            "dc_internal_resistance_extraction_status"
        ] = "unavailable"
        feature_record[
            "dc_internal_resistance_extraction_error"
        ] = str(exc)
    except (RuntimeError, ValueError) as exc:
        feature_record[
            "dc_internal_resistance_extraction_status"
        ] = "failed"
        feature_record[
            "dc_internal_resistance_extraction_error"
        ] = str(exc)

    measurement_ocv_reference = None
    measurement_ocv_error = None

    if inventory_record[
        "reference_test_protocol"
    ] in {
        "ET_T23",
        "AT_T23"
    }:
        try:
            measurement_ocv_reference = (
                extract_ocv_reference_from_rows(
                    measurement_rows
                )
            )
            measurement_ocv_error = None
        except (
            RuntimeError,
            ValueError,
            UnicodeDecodeError
        ) as exc:
            measurement_ocv_reference = None
            measurement_ocv_error = str(exc)

        selected_ocv_reference = (
            measurement_ocv_reference
        )
        selected_ocv_source_identity = (
            inventory_record["source_identity"]
            if measurement_ocv_reference
            is not None
            else None
        )
    else:
        selected_ocv_reference = (
            initial_ocv_reference
        )
        selected_ocv_source_identity = (
            initial_ocv_source_identity
        )

    feature_record[
        "ocv_reference_source_identity"
    ] = selected_ocv_source_identity

    correction_prerequisites_available = (
        selected_ocv_reference is not None
        and feature_record["capacity_extraction_status"] == "completed"
        and feature_record["dc_internal_resistance_extraction_status"] == "completed"
    )

    if not correction_prerequisites_available:
        feature_record[
            "ocv_corrected_resistance_extraction_status"
        ] = "unavailable"

        feature_record[
            "ocv_corrected_resistance_extraction_error"
        ] = (
            measurement_ocv_error
            or (
                "The provisional OCV correction does not have all required inputs."
            )
        )

    else:
        try:
            corrected_resistance_result = (
                calculate_ocv_corrected_dc_internal_resistance(
                    dc_internal_resistance = {
                        "baseline_voltage_volts": (
                            feature_record["baseline_voltage_volts"]
                        ),
                        "evaluated_pulse_voltage_volts": (
                            feature_record["evaluated_pulse_voltage_volts"]
                        ),
                        "pulse_current_amperes": (
                            feature_record["pulse_current_amperes"]
                        ),
                        "pulse_charge_ah": (
                            feature_record["pulse_charge_ah"]
                        )
                    },
                    ocv_reference = selected_ocv_reference,
                    discharge_capacity_ah = (
                        feature_record["discharge_capacity_ah"]
                    )
                )
            )

            for field_name, field_value in (
                corrected_resistance_result.items()
            ):
                if field_name == "analytical_status":
                    continue

                if field_name == "normalization_method":
                    feature_record[
                        "ocv_normalization_method"
                    ] = field_value
                    continue

                feature_record[field_name] = field_value

            feature_record[
                "ocv_corrected_resistance_extraction_status"
            ] = "completed"

        except (RuntimeError, ValueError) as exc:
            feature_record[
                "ocv_corrected_resistance_extraction_status"
            ] = "failed"

            feature_record[
                "ocv_corrected_resistance_extraction_error"
            ] = str(exc)

    primary_feature_statuses = {
        feature_record[
            "capacity_extraction_status"
        ],
        feature_record[
            "dc_internal_resistance_extraction_status"
        ]
    }

    if "failed" in primary_feature_statuses:
        feature_extraction_status = "failed"

    elif primary_feature_statuses == {
        "completed"
    }:
        feature_extraction_status = "completed"

    elif "completed" in primary_feature_statuses:
        feature_extraction_status = (
            "completed_with_unavailable_features"
        )

    else:
        feature_extraction_status = "unavailable"

    feature_record[
        "feature_extraction_status"
    ] = feature_extraction_status

    return feature_record, measurement_ocv_reference


# Extracting audited features from one cell archive
def extract_cell_archive_records(archive: zipfile.ZipFile, reference_inventory_records: list[dict],
                                 cycling_inventory_records: list[dict]
) -> tuple[list[dict], list[dict]]:
    baseline_inventory_records = [
        inventory_record
        for inventory_record in (
            reference_inventory_records
        )
        if (
            inventory_record[
                "reference_test_protocol"
            ]
            == "ET_T23"
        )
    ]

    if len(baseline_inventory_records) != 1:
        raise RuntimeError("Cell archive does not contain exactly one audited initial 23°C reference test.")

    baseline_inventory_record = (
        baseline_inventory_records[0]
    )

    (
        baseline_reference_record,
        initial_ocv_reference
    ) = extract_reference_test_record(
        archive = archive,
        inventory_record = baseline_inventory_record,
        initial_ocv_reference = None,
        initial_ocv_source_identity = None
    )

    initial_ocv_source_identity = (
        baseline_inventory_record["source_identity"]
        if initial_ocv_reference is not None
        else None
    )

    reference_test_records = [baseline_reference_record]

    for inventory_record in sorted(
        reference_inventory_records,
        key = lambda record: (
            record["sequence_number"],
            record["reference_test_protocol"]
        )
    ):
        if inventory_record is baseline_inventory_record:
            continue

        (
            reference_record,
            _
        ) = extract_reference_test_record(
            archive = archive,
            inventory_record = inventory_record,
            initial_ocv_reference = initial_ocv_reference,
            initial_ocv_source_identity = initial_ocv_source_identity
        )

        reference_test_records.append(reference_record)

    cycling_measurement_records = [
        extract_cycling_measurement_record(
            archive = archive,
            inventory_record = inventory_record
        )
        for inventory_record in sorted(
            cycling_inventory_records,
            key = lambda record: record[
                "sequence_number"
            ]
        )
    ]

    add_elapsed_calendar_time(
        reference_test_records
    )

    add_relative_reference_outcomes(
        reference_test_records
    )

    add_audited_cumulative_cycle_exposure(
        reference_test_records = (
            reference_test_records
        ),
        cycling_measurement_records = (
            cycling_measurement_records
        )
    )

    return (
        reference_test_records,
        cycling_measurement_records
    )


# Determining the planned reference-test schedule for one cell
def determine_reference_test_schedule(aging_type: str, cell_number: int) -> str:
    if (
        aging_type == "k"
        and cell_number >= 4
    ):
        return "initial_and_final_only"

    return "periodic_reference_tests"


# Building the validated cell dimension
def build_cell_feature_records(experiment_metadata_content: str, reference_measurement_inventory: list[dict]) -> list[dict]:
    metadata_reader = csv.DictReader(
        io.StringIO(
            experiment_metadata_content
        )
    )

    # Normalizing only the documented source header artifact. Any other surrounding whitespace on a
    #  header name is treated as an undocumented source malformation and rejected rather than silently
    #  normalized. Duplicate column names created by the documented alias are rejected as well.
    experiment_metadata_header_aliases = {
        " volt_meter": "volt_meter"
    }

    if metadata_reader.fieldnames is not None:
        source_field_names = metadata_reader.fieldnames

        undocumented_whitespace_field_names = sorted(
            field_name
            for field_name in source_field_names
            if (
                field_name != field_name.strip()
                and field_name
                not in experiment_metadata_header_aliases
            )
        )

        if undocumented_whitespace_field_names:
            raise RuntimeError(
                "Experiment metadata contains undocumented surrounding header whitespace : "
                f"{undocumented_whitespace_field_names}"
            )

        normalized_field_names = [
            experiment_metadata_header_aliases.get(
                field_name,
                field_name
            )
            for field_name in source_field_names
        ]

        duplicate_field_names = sorted(
            field_name
            for field_name, field_count in Counter(
                normalized_field_names
            ).items()
            if field_count > 1
        )

        if duplicate_field_names:
            raise RuntimeError(
                "Experiment metadata contains duplicate column names after documented header normalization : "
                f"{duplicate_field_names}"
            )

        metadata_reader.fieldnames = normalized_field_names

    available_columns = set(
        metadata_reader.fieldnames
        or []
    )

    missing_columns = (
        REQUIRED_CELL_FEATURE_EXPERIMENT_COLUMNS.difference(
            available_columns
        )
    )

    if missing_columns:
        raise RuntimeError(f"Experiment metadata is missing required columns : {sorted(missing_columns)}")

    audited_cell_identities = {
        measurement_record[
            "cell_identity"
        ]
        for measurement_record in (
            reference_measurement_inventory
        )
    }

    cell_records = []
    observed_cell_identities = set()

    for experiment_record in metadata_reader:
        stage_number = int(
            experiment_record[
                "stage"
            ]
        )

        aging_type = experiment_record[
            "type"
        ]

        test_point_number = int(
            experiment_record[
                "tp"
            ]
        )

        cell_number = int(
            experiment_record[
                "cell"
            ]
        )

        dataset_stage = (
            f"Stage_{stage_number}"
        )

        cell_folder = (
            f"TP_{aging_type}"
            f"{test_point_number:02d}_"
            f"{cell_number:02d}"
        )

        cell_identity = (
            f"{dataset_stage}/{cell_folder}"
        )

        if cell_identity in observed_cell_identities:
            raise RuntimeError(f"Experiment metadata contains a duplicate cell identity : {cell_identity}")

        observed_cell_identities.add(
            cell_identity
        )

        reference_test_schedule = (
            determine_reference_test_schedule(
                aging_type = aging_type,
                cell_number = cell_number
            )
        )

        condition_protocol_identity = (
            f"{dataset_stage}/"
            f"TP_{aging_type}"
            f"{test_point_number:02d}/"
            f"{reference_test_schedule}"
        )

        cell_records.append(
            {
                "cell_identity": cell_identity,
                "dataset_stage": dataset_stage,
                "cell_folder": cell_folder,
                "internal_cell_serial": (
                    experiment_record[
                        "serial_internal"
                    ]
                ),
                "published_cell_serial": (
                    experiment_record[
                        "serial"
                    ]
                ),
                "laboratory": experiment_record[
                    "lab"
                ],
                "aging_type": aging_type,
                "aging_mode": (
                    "calendar_aging"
                    if aging_type == "k"
                    else "cycle_aging"
                ),
                "test_point_number": (
                    test_point_number
                ),
                "cell_number": cell_number,
                "ambient_temperature_celsius": (
                    parse_experiment_number(
                        value = experiment_record[
                            "amb_temp_tp"
                        ],
                        field_name = "amb_temp_tp",
                        cell_identity = cell_identity
                    )
                ),
                "maximum_state_of_charge_fraction": (
                    parse_experiment_number(
                        value = experiment_record[
                            "soc_max_tp"
                        ],
                        field_name = "soc_max_tp",
                        cell_identity = cell_identity
                    )
                ),
                "depth_of_discharge_fraction": (
                    parse_experiment_number(
                        value = experiment_record[
                            "dod_tp"
                        ],
                        field_name = "dod_tp",
                        cell_identity = cell_identity
                    )
                ),
                "charge_rate_c": (
                    parse_experiment_number(
                        value = experiment_record[
                            "c_ch_tp"
                        ],
                        field_name = "c_ch_tp",
                        cell_identity = cell_identity
                    )
                ),
                "discharge_rate_c": (
                    parse_experiment_number(
                        value = experiment_record[
                            "c_dch_tp"
                        ],
                        field_name = "c_dch_tp",
                        cell_identity = cell_identity
                    )
                ),
                "sampling_method": (
                    experiment_record[
                        "sampling"
                    ]
                ),
                "reference_test_schedule": (
                    reference_test_schedule
                ),
                "condition_protocol_identity": (
                    condition_protocol_identity
                ),
                "initial_cell_mass_grams": (
                    parse_experiment_number(
                        value = experiment_record["m_0"],
                        field_name = "m_0",
                        cell_identity = cell_identity
                    )
                ),
                "initial_cell_mass_scale": (
                    parse_required_experiment_text(
                        value = experiment_record["scale"],
                        field_name = "scale",
                        cell_identity = cell_identity
                    )
                ),
                "initial_1khz_resistance_source_value": (
                    parse_experiment_number(
                        value = experiment_record["R_1khz_0"],
                        field_name = "R_1khz_0",
                        cell_identity = cell_identity
                    )
                ),
                "initial_1khz_resistance_source_unit_label": "ohm",
                "initial_1khz_resistance_unit_status": (
                    "unresolved_source_label_conflicts_with_observed_scale_and_cell_specification"
                ),
                "initial_1khz_resistance_meter": (
                    parse_required_experiment_text(
                        value = experiment_record["resistance_meter"],
                        field_name = "resistance_meter",
                        cell_identity = cell_identity
                    )
                ),
                "initial_voltage_volts": (
                    parse_experiment_number(
                        value = experiment_record["U_0"],
                        field_name = "U_0",
                        cell_identity = cell_identity
                    )
                ),
                "initial_voltage_meter": (
                    parse_required_experiment_text(
                        value = experiment_record["volt_meter"],
                        field_name = "volt_meter",
                        cell_identity = cell_identity
                    )
                )
            }
        )

    if observed_cell_identities != audited_cell_identities:
        missing_audit_cells = sorted(
            observed_cell_identities.difference(
                audited_cell_identities
            )
        )

        unexpected_audit_cells = sorted(
            audited_cell_identities.difference(
                observed_cell_identities
            )
        )

        raise RuntimeError(
            "Experiment metadata and reference inventory contain different cells : "
            f"missing audit cells : {missing_audit_cells[:20]} : unexpected audit cells : {unexpected_audit_cells[:20]}"
        )

    return sorted(
        cell_records,
        key = lambda cell_record: (
            cell_record[
                "dataset_stage"
            ],
            cell_record[
                "aging_type"
            ],
            cell_record[
                "test_point_number"
            ],
            cell_record[
                "cell_number"
            ]
        )
    )


# Building the condition-protocol dimension
def build_condition_feature_records(cell_records: list[dict]) -> list[dict]:
    cells_by_condition = {}

    for cell_record in cell_records:
        cells_by_condition.setdefault(
            cell_record[
                "condition_protocol_identity"
            ],
            []
        ).append(
            cell_record
        )

    condition_records = []

    condition_fields = [
        "dataset_stage",
        "laboratory",
        "aging_type",
        "aging_mode",
        "test_point_number",
        "ambient_temperature_celsius",
        "maximum_state_of_charge_fraction",
        "depth_of_discharge_fraction",
        "charge_rate_c",
        "discharge_rate_c",
        "sampling_method",
        "reference_test_schedule"
    ]

    for (
        condition_protocol_identity,
        condition_cells
    ) in cells_by_condition.items():
        first_cell = condition_cells[0]

        for condition_field in condition_fields:
            observed_values = {
                cell_record[
                    condition_field
                ]
                for cell_record in condition_cells
            }

            if len(observed_values) != 1:
                raise RuntimeError(
                    "Condition-protocol cells contain conflicting metadata : "
                    f"{condition_protocol_identity} : {condition_field} : {sorted(observed_values)}"
                )

        condition_records.append(
            {
                "condition_protocol_identity": (
                    condition_protocol_identity
                ),
                **{
                    condition_field: first_cell[
                        condition_field
                    ]
                    for condition_field in (
                        condition_fields
                    )
                },
                "cell_count": len(
                    condition_cells
                )
            }
        )

    return sorted(
        condition_records,
        key = lambda condition_record: (
            condition_record[
                "dataset_stage"
            ],
            condition_record[
                "aging_type"
            ],
            condition_record[
                "test_point_number"
            ],
            condition_record[
                "reference_test_schedule"
            ]
        )
    )


# Validating pinned cell and condition dimensions
def validate_feature_dimensions(cell_records: list[dict], condition_records: list[dict]) -> None:
    if len(cell_records) != EXPECTED_CELL_COUNT:
        raise RuntimeError(f"Cell feature table has an unexpected row count : {len(cell_records)}")

    if (
        len(condition_records)
        != EXPECTED_CONDITION_PROTOCOL_COUNT
    ):
        raise RuntimeError(f"Condition feature table has an unexpected row count : {len(condition_records)}")

    cell_identities = [
        cell_record[
            "cell_identity"
        ]
        for cell_record in cell_records
    ]

    if len(cell_identities) != len(
        set(
            cell_identities
        )
    ):
        raise RuntimeError("Cell feature table contains duplicate identities.")

    condition_identities = [
        condition_record[
            "condition_protocol_identity"
        ]
        for condition_record in condition_records
    ]

    if len(condition_identities) != len(
        set(
            condition_identities
        )
    ):
        raise RuntimeError("Condition feature table contains duplicate identities.")

    condition_identity_set = set(
        condition_identities
    )

    unknown_condition_identities = sorted(
        {
            cell_record[
                "condition_protocol_identity"
            ]
            for cell_record in cell_records
        }.difference(
            condition_identity_set
        )
    )

    if unknown_condition_identities:
        raise RuntimeError(f"Cells reference unknown condition protocols : {unknown_condition_identities}")

    represented_cell_count = sum(
        condition_record[
            "cell_count"
        ]
        for condition_record in (
            condition_records
        )
    )

    if represented_cell_count != len(
        cell_records
    ):
        raise RuntimeError(
            f"Condition feature table does not reconcile to the cell table : {represented_cell_count} : {len(cell_records)}"
        )


# Reading one Cloud Storage object name from its audited URI
def parse_cloud_archive_object_name(cloud_archive: str, bucket_name: str) -> str:
    expected_prefix = (
        f"gs://{bucket_name}/"
    )

    if not cloud_archive.startswith(
        expected_prefix
    ):
        raise RuntimeError(f"Audited cloud archive does not belong to the requested bucket : {cloud_archive}")

    object_name = cloud_archive[
        len(expected_prefix):
    ]

    if not object_name:
        raise RuntimeError(f"Audited cloud archive has an empty object name : {cloud_archive}")

    return object_name


# Serializing one analytical-table value without losing missingness
def serialize_feature_table_value(value: object) -> object:
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

    if (
        isinstance(
            value,
            float
        )
        and not math.isfinite(
            value
        )
    ):
        raise RuntimeError("Analytical tables cannot contain non-finite numerical values.")

    return value


# Writing one deterministic analytical CSV table
def write_feature_table(output_path: Path, records: list[dict],
                        leading_columns: list[str]
) -> list[str]:
    if not records:
        raise RuntimeError(f"Analytical table does not contain records : {output_path.name}")

    record_columns = set(
        records[0]
    )

    for row_number, record in enumerate(
        records,
        start = 1
    ):
        if set(record) != record_columns:
            missing_columns = sorted(
                record_columns.difference(
                    record
                )
            )

            unexpected_columns = sorted(
                set(record).difference(
                    record_columns
                )
            )

            raise RuntimeError(
                "Analytical table contains an inconsistent record schema : "
                f"{output_path.name} : row {row_number} : missing {missing_columns} : unexpected {unexpected_columns}"
            )

    missing_leading_columns = sorted(
        set(
            leading_columns
        ).difference(
            record_columns
        )
    )

    if missing_leading_columns:
        raise RuntimeError(
            f"Analytical table is missing required leading columns : {output_path.name} : {missing_leading_columns}"
        )

    remaining_columns = sorted(
        record_columns.difference(
            leading_columns
        )
    )

    field_names = [
        *leading_columns,
        *remaining_columns
    ]

    output_path.parent.mkdir(
        parents = True,
        exist_ok = True
    )

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
                    field_name: (
                        serialize_feature_table_value(
                            record[
                                field_name
                            ]
                        )
                    )
                    for field_name in field_names
                }
            )

    return field_names


# Classifying one analytical-table value for its schema
def classify_feature_table_value(value: object) -> str:
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

    raise RuntimeError(f"Analytical table contains an unsupported value type : {type(value).__name__}")


# Building one machine-readable analytical-table schema
def build_feature_table_schema(records: list[dict], field_names: list[str]) -> list[dict]:
    table_schema = []

    for field_name in field_names:
        observed_types = {
            classify_feature_table_value(
                record[
                    field_name
                ]
            )
            for record in records
            if record[
                field_name
            ] is not None
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
                iter(
                    observed_types
                )
            )

        elif not observed_types:
            data_type = "null"

        else:
            raise RuntimeError(
                f"Analytical-table column contains incompatible value types : {field_name} : {sorted(observed_types)}"
            )

        table_schema.append(
            {
                "name": field_name,
                "data_type": data_type,
                "nullable": any(
                    record[
                        field_name
                    ] is None
                    for record in records
                ),
                "unit": FEATURE_COLUMN_UNITS.get(
                    field_name
                )
            }
        )

    return table_schema


# Validating analytical-table keys and relationships
def validate_feature_tables(cell_records: list[dict], condition_records: list[dict],
                            reference_test_records: list[dict], cycling_measurement_records: list[dict],
                            reference_measurement_inventory: list[dict], cycling_measurement_inventory: list[dict]
) -> dict:
    validate_feature_dimensions(
        cell_records = cell_records,
        condition_records = condition_records
    )

    if (
        len(reference_test_records)
        != len(reference_measurement_inventory)
    ):
        raise RuntimeError(
            "Reference-test features do not reconcile to their audited inventory : "
            f"{len(reference_test_records)} : {len(reference_measurement_inventory)}"
        )

    if (
        len(cycling_measurement_records)
        != len(cycling_measurement_inventory)
    ):
        raise RuntimeError(
            "Cycling features do not reconcile to their audited inventory : "
            f"{len(cycling_measurement_records)} : {len(cycling_measurement_inventory)}"
        )

    cell_identities = {
        cell_record[
            "cell_identity"
        ]
        for cell_record in cell_records
    }

    reference_source_identities = [
        reference_test_record[
            "source_identity"
        ]
        for reference_test_record in (
            reference_test_records
        )
    ]

    cycling_source_identities = [
        cycling_measurement_record[
            "source_identity"
        ]
        for cycling_measurement_record in (
            cycling_measurement_records
        )
    ]

    if (
        len(reference_source_identities)
        != len(
            set(
                reference_source_identities
            )
        )
    ):
        raise RuntimeError("Reference-test feature table contains duplicate source identities.")

    if (
        len(cycling_source_identities)
        != len(
            set(
                cycling_source_identities
            )
        )
    ):
        raise RuntimeError("Cycling feature table contains duplicate source identities.")

    unknown_reference_cells = sorted(
        {
            reference_test_record[
                "cell_identity"
            ]
            for reference_test_record in (
                reference_test_records
            )
        }.difference(
            cell_identities
        )
    )

    unknown_cycling_cells = sorted(
        {
            cycling_measurement_record[
                "cell_identity"
            ]
            for cycling_measurement_record in (
                cycling_measurement_records
            )
        }.difference(
            cell_identities
        )
    )

    if unknown_reference_cells:
        raise RuntimeError(f"Reference tests contain unknown cells : {unknown_reference_cells}")

    if unknown_cycling_cells:
        raise RuntimeError(f"Cycling measurements contain unknown cells : {unknown_cycling_cells}")

    baseline_counts = Counter(
        reference_test_record[
            "cell_identity"
        ]
        for reference_test_record in (
            reference_test_records
        )
        if (
            reference_test_record[
                "reference_test_protocol"
            ]
            == "ET_T23"
        )
    )

    invalid_baseline_cells = sorted(
        cell_identity
        for cell_identity in cell_identities
        if baseline_counts[
            cell_identity
        ] != 1
    )

    if invalid_baseline_cells:
        raise RuntimeError(f"Cells do not contain exactly one initial 23°C reference test : {invalid_baseline_cells}")

    reference_extraction_failure_count = sum(
        reference_test_record[
            "feature_extraction_requested"
        ]
        and reference_test_record[
            "feature_extraction_status"
        ] == "failed"
        for reference_test_record in (
            reference_test_records
        )
    )

    cycling_extraction_failure_count = sum(
        cycling_measurement_record[
            "feature_extraction_requested"
        ]
        and cycling_measurement_record[
            "feature_extraction_status"
        ] == "failed"
        for cycling_measurement_record in (
            cycling_measurement_records
        )
    )

    provisional_ocv_unavailable_count = sum(
        reference_test_record[
            "feature_extraction_requested"
        ]
        and reference_test_record[
            "ocv_corrected_resistance_extraction_status"
        ] != "completed"
        for reference_test_record in (
            reference_test_records
        )
    )

    reference_unavailable_count = sum(
        reference_test_record[
            "feature_extraction_requested"
        ]
        and reference_test_record[
            "feature_extraction_status"
        ] in {
            "unavailable",
            "completed_with_unavailable_features"
        }
        for reference_test_record in (
            reference_test_records
        )
    )

    primary_reference_eligible_count = sum(
        reference_test_record[
            "primary_cohort_eligible"
        ]
        is True
        for reference_test_record in (
            reference_test_records
        )
    )

    primary_relative_capacity_eligible_count = sum(
        reference_test_record[
            "relative_capacity_primary_eligible"
        ]
        is True
        for reference_test_record in (
            reference_test_records
        )
    )

    primary_relative_resistance_eligible_count = sum(
        reference_test_record[
            "relative_dc_internal_resistance_primary_eligible"
        ]
        is True
        for reference_test_record in (
            reference_test_records
        )
    )

    primary_cycling_exposure_eligible_count = sum(
        cycling_measurement_record[
            "primary_exposure_eligible"
        ]
        is True
        for cycling_measurement_record in (
            cycling_measurement_records
        )
    )

    cumulative_fec_incomplete_count = sum(
        reference_test_record[
            "cumulative_fec_status"
        ] == "incomplete"
        for reference_test_record in (
            reference_test_records
        )
    )

    cumulative_fec_not_applicable_count = sum(
        reference_test_record[
            "cumulative_fec_status"
        ] == "not_applicable"
        for reference_test_record in (
            reference_test_records
        )
    )

    cycling_unavailable_count = sum(
        cycling_measurement_record[
            "feature_extraction_requested"
        ]
        and cycling_measurement_record[
            "feature_extraction_status"
        ] == "unavailable"
        for cycling_measurement_record in (
            cycling_measurement_records
        )
    )

    return {
        "cell_count": len(
            cell_records
        ),
        "condition_protocol_count": len(
            condition_records
        ),
        "reference_test_count": len(
            reference_test_records
        ),
        "cycling_measurement_count": len(
            cycling_measurement_records
        ),
        "primary_reference_eligible_count": (
            primary_reference_eligible_count
        ),
        "primary_relative_capacity_eligible_count": (
            primary_relative_capacity_eligible_count
        ),
        "primary_relative_resistance_eligible_count": (
            primary_relative_resistance_eligible_count
        ),
        "primary_cycling_exposure_eligible_count": (
            primary_cycling_exposure_eligible_count
        ),
        "cumulative_fec_incomplete_count": (
            cumulative_fec_incomplete_count
        ),
        "cumulative_fec_not_applicable_count": (
            cumulative_fec_not_applicable_count
        ),
        "reference_extraction_failure_count": (
            reference_extraction_failure_count
        ),
        "cycling_extraction_failure_count": (
            cycling_extraction_failure_count
        ),
        "provisional_ocv_unavailable_count": (
            provisional_ocv_unavailable_count
        ),
        "reference_unavailable_count": (
            reference_unavailable_count
        ),
        "cycling_unavailable_count": (
            cycling_unavailable_count
        )
    }


# Extracting one cached archive in an independent process
def extract_cached_archive_work_item(
    work_item: tuple[
        str,
        Path,
        list[dict],
        list[dict]
    ]
) -> tuple[
    str,
    list[dict],
    list[dict]
]:
    (
        archive_identity,
        archive_path,
        reference_inventory_records,
        cycling_inventory_records
    ) = work_item

    try:
        with zipfile.ZipFile(
            archive_path,
            mode = "r"
        ) as archive:
            (
                reference_test_records,
                cycling_measurement_records
            ) = extract_cell_archive_records(
                archive = archive,
                reference_inventory_records = (
                    reference_inventory_records
                ),
                cycling_inventory_records = (
                    cycling_inventory_records
                )
            )

    except Exception as exc:
        raise RuntimeError(
            f"Cached archive feature extraction failed : {archive_identity} : {exc}"
        ) from exc

    return (
        archive_identity,
        reference_test_records,
        cycling_measurement_records
    )


# Processing cached archives with deterministic result ordering
def extract_cached_archive_work_items(
    work_items: list[
        tuple[
            str,
            Path,
            list[dict],
            list[dict]
        ]
    ],
    worker_count: int
) -> list[
    tuple[
        str,
        list[dict],
        list[dict]
    ]
]:
    if worker_count <= 0:
        raise ValueError("Worker count must be a positive integer.")

    if worker_count == 1:
        return [
            extract_cached_archive_work_item(
                work_item
            )
            for work_item in work_items
        ]

    with ProcessPoolExecutor(
        max_workers = worker_count
    ) as executor:
        extraction_results = list(
            executor.map(
                extract_cached_archive_work_item,
                work_items,
                chunksize = 1
            )
        )

    return extraction_results


# Extracting complete audited feature tables incrementally
def main() -> None:
    args = parse_args()

    audit_result = load_feature_audit(
        audit_path = args.audit_path
    )

    quality_contract_binding = (
        load_validated_quality_contract_binding(
            contract_path = args.contract_path,
            audit_result = audit_result
        )
    )

    reference_measurement_inventory = (
        build_reference_measurement_inventory(
            audit_result
        )
    )

    cycling_measurement_inventory = (
        build_cycling_measurement_inventory(
            audit_result
        )
    )

    source_measurement_reconciliation = (
        build_source_measurement_reconciliation(
            audit_result = audit_result,
            reference_measurement_inventory = (
                reference_measurement_inventory
            ),
            cycling_measurement_inventory = (
                cycling_measurement_inventory
            )
        )
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

    reference_inventory_by_archive = {}

    for inventory_record in (
        reference_measurement_inventory
    ):
        reference_inventory_by_archive.setdefault(
            inventory_record[
                "archive_identity"
            ],
            []
        ).append(
            inventory_record
        )

    cycling_inventory_by_archive = {}

    for inventory_record in (
        cycling_measurement_inventory
    ):
        cycling_inventory_by_archive.setdefault(
            inventory_record[
                "archive_identity"
            ],
            []
        ).append(
            inventory_record
        )

    archive_work_items = []

    for archive_identity in sorted(
        reference_inventory_by_archive
    ):
        archive_reference_inventory = (
            reference_inventory_by_archive[
                archive_identity
            ]
        )

        archive_cycling_inventory = (
            cycling_inventory_by_archive.get(
                archive_identity,
                []
            )
        )

        cloud_archives = {
            inventory_record[
                "cloud_archive"
            ]
            for inventory_record in [
                *archive_reference_inventory,
                *archive_cycling_inventory
            ]
        }

        if len(cloud_archives) != 1:
            raise RuntimeError(
                f"One archive identity resolves to multiple cloud objects : {archive_identity} : {sorted(cloud_archives)}"
            )

        cloud_archive = next(
            iter(
                cloud_archives
            )
        )

        archive_object_name = (
            parse_cloud_archive_object_name(
                cloud_archive = cloud_archive,
                bucket_name = args.bucket_name
            )
        )

        archive_path = resolve_cached_archive_path(
            bucket = bucket,
            article_metadata = article_metadata,
            archive_object_name = archive_object_name,
            archive_cache_directory = (
                args.archive_cache_directory
            )
        )

        archive_work_items.append(
            (
                archive_identity,
                archive_path,
                archive_reference_inventory,
                archive_cycling_inventory
            )
        )

    print(
        "Processing cached archives : "
        f"{len(archive_work_items)} : workers {args.worker_count}"
    )

    archive_extraction_results = (
        extract_cached_archive_work_items(
            work_items = archive_work_items,
            worker_count = args.worker_count
        )
    )

    reference_test_records = []
    cycling_measurement_records = []

    for (
        archive_identity,
        archive_reference_records,
        archive_cycling_records
    ) in archive_extraction_results:
        reference_test_records.extend(
            archive_reference_records
        )

        cycling_measurement_records.extend(
            archive_cycling_records
        )

    reference_test_records.sort(
        key = lambda record: (
            record[
                "dataset_stage"
            ],
            record[
                "cell_identity"
            ],
            record[
                "sequence_number"
            ],
            record[
                "reference_test_protocol"
            ]
        )
    )

    cycling_measurement_records.sort(
        key = lambda record: (
            record[
                "dataset_stage"
            ],
            record[
                "cell_identity"
            ],
            record[
                "sequence_number"
            ]
        )
    )

    extraction_summary = validate_feature_tables(
        cell_records = cell_records,
        condition_records = condition_records,
        reference_test_records = (
            reference_test_records
        ),
        cycling_measurement_records = (
            cycling_measurement_records
        ),
        reference_measurement_inventory = (
            reference_measurement_inventory
        ),
        cycling_measurement_inventory = (
            cycling_measurement_inventory
        )
    )

    output_directory = args.output_directory

    table_records = {
        "cell_features": cell_records,
        "condition_features": condition_records,
        "reference_test_features": (
            reference_test_records
        ),
        "cycling_exposure_features": (
            cycling_measurement_records
        )
    }

    table_columns = {
        "cell_features": write_feature_table(
            output_path = (
                output_directory
                / "cell_features.csv"
            ),
            records = cell_records,
            leading_columns = [
                "cell_identity",
                "condition_protocol_identity",
                "dataset_stage",
                "cell_folder"
            ]
        ),
        "condition_features": write_feature_table(
            output_path = (
                output_directory
                / "condition_features.csv"
            ),
            records = condition_records,
            leading_columns = [
                "condition_protocol_identity",
                "dataset_stage",
                "aging_type",
                "test_point_number",
                "reference_test_schedule"
            ]
        ),
        "reference_test_features": write_feature_table(
            output_path = (
                output_directory
                / "reference_test_features.csv"
            ),
            records = reference_test_records,
            leading_columns = [
                "source_identity",
                "cell_identity",
                "archive_identity",
                "sequence_number",
                "reference_test_protocol",
                "measurement_start_date",
                "elapsed_calendar_days"
            ]
        ),
        "cycling_exposure_features": write_feature_table(
            output_path = (
                output_directory
                / "cycling_exposure_features.csv"
            ),
            records = cycling_measurement_records,
            leading_columns = [
                "source_identity",
                "cell_identity",
                "archive_identity",
                "sequence_number"
            ]
        )
    }

    audit_sha256 = hashlib.sha256(
        args.audit_path.read_bytes()
    ).hexdigest()

    extraction_has_failures = (
        extraction_summary[
            "reference_extraction_failure_count"
        ] > 0
        or extraction_summary[
            "cycling_extraction_failure_count"
        ] > 0
    )

    feature_manifest = {
        "data_quality_contract_version": (
            audit_result[
                "contract_version"
            ]
        ),
        "quality_contract": (
            quality_contract_binding
        ),
        "dataset": audit_result[
            "dataset"
        ],
        "audit_sha256": audit_sha256,
        "source_measurement_reconciliation": (
            source_measurement_reconciliation
        ),
        "feature_contract": audit_result[
            "feature_contract"
        ],
        "extraction_status": (
            "completed_with_unresolved_failures"
            if extraction_has_failures
            else "completed"
        ),
        "summary": extraction_summary,
        "tables": {
            table_name: {
                "file_name": (
                    f"{table_name}.csv"
                ),
                "sha256": calculate_file_digest(
                    file_path = (
                        output_directory
                        / f"{table_name}.csv"
                    ),
                    algorithm_name = "sha256"
                ),
                "row_count": len(
                    table_records[
                        table_name
                    ]
                ),
                "schema": build_feature_table_schema(
                    records = table_records[
                        table_name
                    ],
                    field_names = columns
                )
            }
            for table_name, columns in (
                table_columns.items()
            )
        }
    }

    manifest_path = (
        output_directory
        / "feature_table_manifest.json"
    )

    manifest_path.write_text(
        (
            json.dumps(
                feature_manifest,
                indent = 2,
                sort_keys = True
            )
            + "\n"
        ),
        encoding = "utf-8"
    )

    print(
        "Complete feature extraction finished."
    )
    print(
        f"Cell count : {extraction_summary['cell_count']}"
    )
    print(
        "Condition-protocol count : "
        f"{extraction_summary['condition_protocol_count']}"
    )
    print(
        "Reference-test count : "
        f"{extraction_summary['reference_test_count']}"
    )
    print(
        "Cycling-measurement count : "
        f"{extraction_summary['cycling_measurement_count']}"
    )
    print(
        f"Feature manifest : {manifest_path}"
    )

    if extraction_has_failures:
        print(
            "Feature extraction completed with unresolved failures.",
            file = sys.stderr
        )

        sys.exit(2)

    print(
        "Feature extraction completed successfully."
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            f"Cell-feature extraction failed : {exc}",
            file = sys.stderr
        )
        sys.exit(1)
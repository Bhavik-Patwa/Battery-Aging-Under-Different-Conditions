import argparse
import csv
import io
import json
import sys
import math
import tempfile
import zipfile
import hashlib
from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from datetime import datetime
from google.cloud import storage

from scripts.ingest_dataset import (
    calculate_blob_sha256,
    encode_md5_checksum,
    validate_size_only_integrity_baseline
)

SECONDS_PER_HOUR = 3600
FIGSHARE_ARTICLE_ID = 25975315
EXPECTED_DATASET_VERSION = 1

DATASET_FOLDER_NAME = "Multi-Stage_Aging_Study"
METADATA_OBJECT_PREFIX = (
    f"metadata/figshare/{FIGSHARE_ARTICLE_ID}/"
    f"v{EXPECTED_DATASET_VERSION}/"
)
RAW_OBJECT_PREFIX = (
    f"raw/figshare/{FIGSHARE_ARTICLE_ID}/"
    f"v{EXPECTED_DATASET_VERSION}/"
)

ARTICLE_METADATA_OBJECT = (
    f"{METADATA_OBJECT_PREFIX}article_metadata.json"
)

REQUIRED_EXPERIMENT_COLUMNS = {
    "serial_internal",
    "serial",
    "lab",
    "type",
    "tp",
    "cell",
    "amb_temp_tp",
    "soc_max_tp",
    "dod_tp",
    "c_ch_tp",
    "c_dch_tp",
    "sampling",
    "stage"
}

EXPECTED_MEASUREMENT_COLUMNS = (
    "run_time",
    "c_vol",
    "c_cur",
    "c_surf_temp",
    "amb_temp",
    "step_type"
)

OPTIONAL_ELAPSED_SECONDS_COLUMN = (
    "time_to_sec"
)

ALLOWED_MEASUREMENT_COLUMNS = {
    EXPECTED_MEASUREMENT_COLUMNS,
    (
        "",
        *EXPECTED_MEASUREMENT_COLUMNS
    ),
    (
        *EXPECTED_MEASUREMENT_COLUMNS,
        OPTIONAL_ELAPSED_SECONDS_COLUMN
    )
}

EXPECTED_ARCHIVE_EXTENSIONS = {
    ".csv",
    ".txt"
}

EXPECTED_CELL_COUNT = 279
EXPECTED_TEST_POINT_COUNT = 71
EXPECTED_CALENDAR_PROTOCOL_VARIANT_COUNT = 22
EXPECTED_CONDITION_PROTOCOL_COUNT = 93

EXPECTED_LABORATORY_COUNTS = {
    "HM": 123,
    "INT": 102,
    "SIE": 54
}

EXPECTED_AGING_TYPE_COUNTS = {
    "k": 132,
    "z": 147
}


# Parsing the cloud source and archive objects being validated
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = (
            "Validating landed Figshare metadata and battery archives."
        )
    )

    parser.add_argument(
        "--project-id",
        required = True,
        help = "Google Cloud project ID containing the dataset."
    )

    parser.add_argument(
        "--bucket-name",
        required = True,
        help = "Cloud Storage bucket containing the landed files."
    )

    parser.add_argument(
        "--archive-object",
        action = "append",
        required = True,
        help = (
            "Cloud Storage object containing one battery archive. Repeat this option for every archive being validated."
        )
    )

    return parser.parse_args()


# Reading one required text object from Cloud Storage
def download_text_object(bucket: storage.Bucket, object_name: str) -> str:
    blob = bucket.blob(object_name)

    if not blob.exists():
        raise FileNotFoundError(f"Required Cloud Storage object was not found : gs://{bucket.name}/{object_name}")

    return blob.download_as_text(
        encoding = "utf-8"
    )


# Constructing the landed object name from one Figshare file record
def build_landed_object_name(file_record: dict) -> str:
    parsed_url = urlparse(
        file_record["download_url"]
    )

    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.netloc
        or not parsed_url.path
    ):
        raise ValueError(f"Invalid public download URL : {file_record['download_url']}")

    return (
        f"{RAW_OBJECT_PREFIX}"
        f"{parsed_url.netloc}/"
        f"{parsed_url.path.lstrip('/')}"
    )


# Locating one uniquely named file in the published dataset root
def find_dataset_root_file(article_metadata: dict, file_name: str) -> dict:
    matching_files = [
        file_record
        for file_record in article_metadata["files"]
        if (
            file_record["name"] == file_name
            and article_metadata["folder_structure"].get(
                str(file_record["id"])
            ) == DATASET_FOLDER_NAME
        )
    ]

    if len(matching_files) != 1:
        raise RuntimeError(f"Expected exactly one published dataset-root file named {file_name}, found : {len(matching_files)}")

    return matching_files[0]


# Summarizing the experiment metadata without treating rows as independent outcomes
def build_experiment_metadata_summary(experiment_metadata_content: str) -> dict:
    metadata_reader = csv.DictReader(
        io.StringIO(
            experiment_metadata_content
        )
    )

    available_columns = set(
        metadata_reader.fieldnames
        or []
    )

    missing_columns = REQUIRED_EXPERIMENT_COLUMNS.difference(
        available_columns
    )

    if missing_columns:
        raise RuntimeError(f"Experiment metadata is missing required columns : {sorted(missing_columns)}")

    experiment_rows = list(metadata_reader)

    if not experiment_rows:
        raise RuntimeError("Experiment metadata does not contain any cell records.")

    cell_identifiers = [
        (
            experiment_row["stage"],
            experiment_row["serial_internal"]
        )
        for experiment_row in experiment_rows
    ]

    duplicate_cell_identifiers = sorted(
        cell_identifier
        for cell_identifier, record_count in Counter(
            cell_identifiers
        ).items()
        if record_count > 1
    )

    if duplicate_cell_identifiers:
        raise RuntimeError(
            f"Experiment metadata contains duplicate stage-specific cell identifiers : {duplicate_cell_identifiers}"
        )

    test_points = {
        (
            experiment_row["stage"],
            experiment_row["type"],
            experiment_row["tp"]
        )
        for experiment_row in experiment_rows
    }

    calendar_protocol_variants = {
        (
            experiment_row["stage"],
            experiment_row["tp"]
        )
        for experiment_row in experiment_rows
        if (
            experiment_row["type"] == "k"
            and int(experiment_row["cell"]) >= 4
        )
    }

    laboratory_counts = Counter(
        experiment_row["lab"]
        for experiment_row in experiment_rows
    )

    aging_type_counts = Counter(
        experiment_row["type"]
        for experiment_row in experiment_rows
    )

    return {
        "cell_count": len(experiment_rows),
        "test_point_count": len(test_points),
        "calendar_protocol_variant_count": (
            len(calendar_protocol_variants)
        ),
        "condition_protocol_count": (
            len(test_points)
            + len(calendar_protocol_variants)
        ),
        "laboratory_counts": dict(
            sorted(laboratory_counts.items())
        ),
        "aging_type_counts": dict(
            sorted(aging_type_counts.items())
        )
    }


# Confirming that the landed metadata matches the pinned dataset version
def validate_experiment_metadata_summary(metadata_summary: dict) -> None:
    expected_values = {
        "cell_count": EXPECTED_CELL_COUNT,
        "test_point_count": EXPECTED_TEST_POINT_COUNT,
        "calendar_protocol_variant_count": (
            EXPECTED_CALENDAR_PROTOCOL_VARIANT_COUNT
        ),
        "condition_protocol_count": (
            EXPECTED_CONDITION_PROTOCOL_COUNT
        ),
        "laboratory_counts": EXPECTED_LABORATORY_COUNTS,
        "aging_type_counts": EXPECTED_AGING_TYPE_COUNTS
    }

    mismatched_values = {
        value_name: {
            "expected": expected_value,
            "observed": metadata_summary.get(value_name)
        }
        for value_name, expected_value in expected_values.items()
        if metadata_summary.get(value_name) != expected_value
    }

    if mismatched_values:
        raise RuntimeError(f"Experiment metadata does not match the pinned dataset expectations : {mismatched_values}")


# Rejecting archive members that could escape an extraction directory
def validate_archive_member_names(member_names: list[str]) -> None:
    duplicate_names = sorted(
        member_name
        for member_name, member_count in Counter(
            member_names
        ).items()
        if member_count > 1
    )

    if duplicate_names:
        raise RuntimeError(f"Archive contains duplicate member names : {duplicate_names}")

    for member_name in member_names:
        if "\\" in member_name:
            raise RuntimeError(f"Archive member uses an unsupported path separator : {member_name}")

        member_path = PurePosixPath(
            member_name
        )

        if (
            member_path.is_absolute()
            or ".." in member_path.parts
        ):
            raise RuntimeError(f"Archive member contains an unsafe path : {member_name}")


# Confirming that every measurement has one metadata companion
def validate_measurement_metadata_pairs(member_names: list[str]) -> None:
    metadata_pair_summary = (
        build_measurement_metadata_pair_summary(
            member_names = member_names
        )
    )

    missing_metadata_names = metadata_pair_summary[
        "missing_metadata_names"
    ]

    unexpected_metadata_names = metadata_pair_summary[
        "unexpected_metadata_names"
    ]

    if missing_metadata_names:
        raise RuntimeError(f"Measurement files are missing metadata companions : {missing_metadata_names}")

    if unexpected_metadata_names:
        raise RuntimeError(f"Archive contains unmatched metadata files : {unexpected_metadata_names}")


# Summarizing measurement and companion-metadata pairing
def build_measurement_metadata_pair_summary(member_names: list[str]) -> dict:
    measurement_names = {
        member_name
        for member_name in member_names
        if PurePosixPath(
            member_name
        ).suffix.lower() == ".csv"
    }

    metadata_names = {
        member_name
        for member_name in member_names
        if PurePosixPath(
            member_name
        ).suffix.lower() == ".txt"
    }

    expected_metadata_names = {
        str(
            PurePosixPath(
                measurement_name
            ).with_name(
                f"{PurePosixPath(measurement_name).stem}_meta.txt"
            )
        )
        for measurement_name in measurement_names
    }

    return {
        "missing_metadata_names": sorted(
            expected_metadata_names.difference(
                metadata_names
            )
        ),
        "unexpected_metadata_names": sorted(
            metadata_names.difference(
                expected_metadata_names
            )
        )
    }


# Inspecting one archive without extracting its contents
def inspect_archive_file(archive_path: Path, require_complete_metadata_pairs: bool = True,
                         validate_member_checksums: bool = True) -> dict:
    with zipfile.ZipFile(
        archive_path,
        mode = "r"
    ) as archive:
        if validate_member_checksums:
            corrupted_member = archive.testzip()

            if corrupted_member is not None:
                raise RuntimeError(f"Archive checksum validation failed for member : {corrupted_member}")

        file_members = [
            member_record
            for member_record in archive.infolist()
            if not member_record.is_dir()
        ]

        if not file_members:
            raise RuntimeError("Archive does not contain any files.")

        member_names = [
            member_record.filename
            for member_record in file_members
        ]

        validate_archive_member_names(
            member_names = member_names
        )

        metadata_pair_summary = (
            build_measurement_metadata_pair_summary(
                member_names = member_names
            )
        )

        if require_complete_metadata_pairs:
            validate_measurement_metadata_pairs(
                member_names = member_names
            )

        encrypted_members = sorted(
            member_record.filename
            for member_record in file_members
            if member_record.flag_bits & 0x1
        )

        if encrypted_members:
            raise RuntimeError(f"Archive contains encrypted members : {encrypted_members}")

        extension_counts = Counter(
            PurePosixPath(
                member_record.filename
            ).suffix.lower()
            for member_record in file_members
        )

        unexpected_extensions = sorted(
            set(extension_counts).difference(
                EXPECTED_ARCHIVE_EXTENSIONS
            )
        )

        if unexpected_extensions:
            raise RuntimeError(f"Archive contains unsupported file extensions : {unexpected_extensions}")

        csv_schema_counts = Counter()

        for member_record in file_members:
            if (
                PurePosixPath(
                    member_record.filename
                ).suffix.lower()
                != ".csv"
            ):
                continue

            with archive.open(
                member_record,
                mode = "r"
            ) as binary_stream:
                with io.TextIOWrapper(
                    binary_stream,
                    encoding = "utf-8-sig",
                    newline = ""
                ) as text_stream:
                    csv_reader = csv.reader(
                        text_stream
                    )

                    header = next(
                        csv_reader,
                        None
                    )

            if not header:
                raise RuntimeError(f"CSV archive member does not contain a header : {member_record.filename}")

            if tuple(header) not in ALLOWED_MEASUREMENT_COLUMNS:
                raise RuntimeError(
                    f"CSV archive member has unexpected columns : {member_record.filename} : {header}"
                )

            csv_schema_counts[
                tuple(header)
            ] += 1

        if not csv_schema_counts:
            raise RuntimeError("Archive does not contain any CSV files.")

        return {
            "measurement_file_names": sorted(
                member_name
                for member_name in member_names
                if PurePosixPath(member_name).suffix.lower() == ".csv"
            ),
            "member_count": len(file_members),
            "compressed_size_bytes": archive_path.stat().st_size,
            "uncompressed_size_bytes": sum(
                member_record.file_size
                for member_record in file_members
            ),
            "empty_member_count": sum(
                member_record.file_size == 0
                for member_record in file_members
            ),
            "measurement_file_count": extension_counts[".csv"],
            "metadata_file_count": extension_counts[".txt"],
            "extension_counts": dict(
                sorted(extension_counts.items())
            ),
            "missing_metadata_names": (
                metadata_pair_summary[
                    "missing_metadata_names"
                ]
            ),
            "unexpected_metadata_names": (
                metadata_pair_summary[
                    "unexpected_metadata_names"
                ]
            ),
            "csv_schemas": [
                {
                    "file_count": file_count,
                    "columns": list(columns)
                }
                for columns, file_count in sorted(
                    csv_schema_counts.items(),
                    key = lambda schema_record: (
                        schema_record[0]
                    )
                )
            ]
        }


# Resolving one landed archive to its published Figshare record
def find_archive_file_record(article_metadata: dict, archive_object_name: str) -> dict:
    file_identifier = PurePosixPath(
        archive_object_name
    ).name

    matching_files = [
        file_record
        for file_record in article_metadata["files"]
        if str(file_record["id"]) == file_identifier
    ]

    if len(matching_files) != 1:
        raise RuntimeError(
            "Expected exactly one Figshare file for landed object "
            f"{archive_object_name}, found : {len(matching_files)}"
        )

    file_record = matching_files[0]

    expected_object_name = build_landed_object_name(
        file_record = file_record
    )

    if archive_object_name != expected_object_name:
        raise RuntimeError(f"Landed archive object does not match its published source identity : {archive_object_name}")

    if not file_record["name"].endswith(".zip"):
        raise RuntimeError(f"Requested archive object is not a published ZIP file : {archive_object_name}")

    return file_record


# Downloading and inspecting one landed archive in temporary storage
def inspect_cloud_archive(bucket: storage.Bucket, article_metadata: dict, archive_object_name: str) -> dict:
    file_record = find_archive_file_record(
        article_metadata = article_metadata,
        archive_object_name = archive_object_name
    )

    blob = bucket.blob(
        archive_object_name
    )

    if not blob.exists():
        raise FileNotFoundError(f"Landed archive was not found : gs://{bucket.name}/{archive_object_name}")

    blob.reload()

    if int(blob.size) != int(file_record["size"]):
        raise RuntimeError(f"Landed archive size does not match the published size : {archive_object_name}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = (
            Path(temporary_directory)
            / file_record["name"]
        )

        blob.download_to_filename(
            archive_path
        )

        archive_summary = inspect_archive_file(
            archive_path = archive_path
        )

    source_folder = article_metadata[
        "folder_structure"
    ][str(file_record["id"])]

    source_path = (
        f"{source_folder}/{file_record['name']}"
    )

    return {
        "source_path": source_path,
        "cloud_object": (
            f"gs://{bucket.name}/{archive_object_name}"
        ),
        **archive_summary
    }


# Converting a measurement's elapsed-time value into seconds
def parse_elapsed_seconds(value: str) -> float:
    time_parts = value.split(":")

    if len(time_parts) != 3:
        raise ValueError(f"Invalid run_time value : {value}")

    hours = int(time_parts[0])
    minutes = int(time_parts[1])
    seconds = float(time_parts[2])

    if (
        hours < 0
        or not 0 <= minutes < 60
        or not math.isfinite(seconds)
        or not 0 <= seconds < 60
    ):
        raise ValueError(f"Invalid run_time value : {value}")

    return (
        hours * SECONDS_PER_HOUR
        + minutes * 60
        + seconds
    )


# Extracting the recorded start date from one measurement's companion metadata
def parse_measurement_start_date(metadata_content: str) -> str:
    date_values = [
        line.partition(":")[2].strip()
        for line in metadata_content.splitlines()
        if line.startswith("Measurement start date:")
    ]

    if len(date_values) != 1:
        raise RuntimeError("Companion metadata must contain exactly one measurement start date.")

    for date_format in ("%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(
                date_values[0],
                date_format
            ).date().isoformat()

        except ValueError:
            continue

    raise ValueError(f"Invalid measurement start date : {date_values[0]}")


# Calculating one local file digest without loading it into memory
def calculate_file_digest(file_path: Path, algorithm_name: str) -> str:
    digest = hashlib.new(
        algorithm_name
    )

    with file_path.open(
        mode = "rb"
    ) as file_stream:
        while True:
            file_chunk = file_stream.read(
                1024
                * 1024
            )

            if not file_chunk:
                break

            digest.update(
                file_chunk
            )

    return digest.hexdigest()


# Validating one cached archive against published integrity evidence
def validate_archive_cache_file(archive_path: Path, bucket: storage.Bucket,
                                file_record: dict, archive_object_name: str
) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Cached archive was not found : {archive_path}")

    observed_size_bytes = archive_path.stat().st_size
    expected_size_bytes = int(
        file_record[
            "size"
        ]
    )

    if observed_size_bytes != expected_size_bytes:
        raise RuntimeError(
            "Cached archive size does not match Figshare metadata : "
            f"{archive_path.name} : {observed_size_bytes} : {expected_size_bytes}"
        )

    published_md5 = file_record.get(
        "computed_md5"
    )

    if published_md5:
        observed_md5 = calculate_file_digest(
            file_path = archive_path,
            algorithm_name = "md5"
        )

        if observed_md5.lower() != published_md5.lower():
            raise RuntimeError(f"Cached archive MD5 does not match Figshare metadata : {archive_path.name}")

        return

    observed_sha256 = calculate_file_digest(
        file_path = archive_path,
        algorithm_name = "sha256"
    )

    validate_size_only_integrity_baseline(
        bucket = bucket,
        file_record = file_record,
        destination_object_name = archive_object_name,
        destination_sha256 = observed_sha256,
        allow_creation = False
    )


# Resolving one validated archive from persistent local storage
def resolve_cached_archive_path(bucket: storage.Bucket, article_metadata: dict,
                                archive_object_name: str, archive_cache_directory: Path
) -> Path:
    file_record = find_archive_file_record(
        article_metadata = article_metadata,
        archive_object_name = archive_object_name
    )

    archive_blob = bucket.blob(
        archive_object_name
    )

    if not archive_blob.exists():
        raise FileNotFoundError(f"Audited cloud archive was not found : gs://{bucket.name}/{archive_object_name}")

    archive_blob.reload()

    expected_size_bytes = int(
        file_record[
            "size"
        ]
    )

    if int(
        archive_blob.size
    ) != expected_size_bytes:
        raise RuntimeError(f"Cloud archive size does not match Figshare metadata : {archive_object_name}")

    published_md5 = file_record.get(
        "computed_md5"
    )

    if published_md5:
        expected_cloud_md5 = encode_md5_checksum(
            published_md5
        )

        if archive_blob.md5_hash != expected_cloud_md5:
            raise RuntimeError(f"Cloud archive MD5 does not match Figshare metadata : {archive_object_name}")

    if (
        not published_md5
        and archive_blob.md5_hash is None
    ):
        destination_sha256 = calculate_blob_sha256(
            blob = archive_blob
        )

        validate_size_only_integrity_baseline(
            bucket = bucket,
            file_record = file_record,
            destination_object_name = archive_object_name,
            destination_sha256 = destination_sha256,
            allow_creation = False
        )

    archive_cache_directory.mkdir(
        parents = True,
        exist_ok = True
    )

    archive_path = (
        archive_cache_directory
        / (
            f"{file_record['id']}_"
            f"{file_record['name']}"
        )
    )

    if archive_path.is_file():
        try:
            validate_archive_cache_file(
                archive_path = archive_path,
                bucket = bucket,
                file_record = file_record,
                archive_object_name = (
                    archive_object_name
                )
            )

            return archive_path

        except (
            RuntimeError,
            FileNotFoundError
        ):
            pass

    download_path = archive_path.with_suffix(
        f"{archive_path.suffix}.download"
    )

    archive_blob.download_to_filename(
        download_path
    )

    validate_archive_cache_file(
        archive_path = download_path,
        bucket = bucket,
        file_record = file_record,
        archive_object_name = archive_object_name
    )

    download_path.replace(
        archive_path
    )

    return archive_path


# Validating the landed metadata and selected archives
def main() -> None:
    args = parse_args()

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

    experiment_metadata_file = find_dataset_root_file(
        article_metadata = article_metadata,
        file_name = "experiments_meta.csv"
    )

    experiment_metadata_object = build_landed_object_name(
        file_record = experiment_metadata_file
    )

    experiment_metadata_content = download_text_object(
        bucket = bucket,
        object_name = experiment_metadata_object
    )

    experiment_metadata_summary = (
        build_experiment_metadata_summary(
            experiment_metadata_content = (
                experiment_metadata_content
            )
        )
    )

    validate_experiment_metadata_summary(
        metadata_summary = experiment_metadata_summary
    )

    requested_archive_objects = list(
        dict.fromkeys(
            args.archive_object
        )
    )

    archive_summaries = [
        inspect_cloud_archive(
            bucket = bucket,
            article_metadata = article_metadata,
            archive_object_name = archive_object_name
        )
        for archive_object_name in requested_archive_objects
    ]

    validation_summary = {
        "experiment_metadata": (
            experiment_metadata_summary
        ),
        "archives": archive_summaries
    }

    print(
        json.dumps(
            validation_summary,
            indent = 2,
            sort_keys = True
        )
    )

    print("Dataset validation completed successfully.")


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(f"Dataset validation failed : {exc}")
        sys.exit(1)
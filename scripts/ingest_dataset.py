import argparse
import base64
import json
import sys
import urllib.request
import hashlib
from pathlib import PurePosixPath
from urllib.parse import urlparse

from collections import Counter
from google.cloud import storage
from google.cloud import storage_transfer

FIGSHARE_ARTICLE_ID = 25975315
EXPECTED_DATASET_VERSION = 1
EXPECTED_DATASET_DOI = (
    "10.6084/m9.figshare.25975315.v1"
)

FIGSHARE_API_URL = (
    "https://api.figshare.com/v2/articles/"
    f"{FIGSHARE_ARTICLE_ID}/versions/{EXPECTED_DATASET_VERSION}"
)

TRANSFER_CHUNK_SIZE_BYTES = 8 * 1024 * 1024

DATASET_FOLDER_NAME = "Multi-Stage_Aging_Study"
METADATA_OBJECT_PREFIX = (
    f"metadata/figshare/{FIGSHARE_ARTICLE_ID}/v{EXPECTED_DATASET_VERSION}/"
)
RAW_OBJECT_PREFIX = (
    f"raw/figshare/{FIGSHARE_ARTICLE_ID}/v{EXPECTED_DATASET_VERSION}/"
)

REQUEST_TIMEOUT_SECONDS = 60
DEFAULT_TRANSFER_TIMEOUT_SECONDS = None

EXPECTED_COMPLETE_FILE_COUNT = 280
EXPECTED_COMPLETE_PUBLISHER_MD5_COUNT = 279
EXPECTED_COMPLETE_SIZE_ONLY_COUNT = 1
EXPECTED_SIZE_ONLY_FILE_NAMES = {
    "TP_z05_01.zip"
}


# Parsing the cloud destination and requested source files
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = ("Transferring selected Figshare dataset files into Google Cloud Storage.")
    )

    parser.add_argument(
        "--project-id",
        required = True,
        help = "Google Cloud project ID owning the transfer job."
    )

    parser.add_argument(
        "--bucket-name",
        required = True,
        help = "Existing destination Cloud Storage bucket name."
    )

    parser.add_argument(
        "--selection-mode",
        choices = [
            "specified",
            "complete"
        ],
        default = "specified",
        help = (
            "Selecting explicitly listed files or every file published in the pinned dataset version."
        )
    )

    parser.add_argument(
        "--source-file",
        action = "append",
        default = [],
        help = (
            "Dataset-relative source path. Repeat this option for every required file when selection mode is specified."
        )
    )

    parser.add_argument(
        "--transfer-timeout-seconds",
        type = int,
        default = DEFAULT_TRANSFER_TIMEOUT_SECONDS,
        help = (
            "Optional client-side waiting limit. Omitting this option waits until the cloud transfer operation finishes."
        )
    )

    return parser.parse_args()


# Retrieving the published Figshare article metadata
def retrieve_article_metadata() -> dict:
    request = urllib.request.Request(
        url = FIGSHARE_API_URL,
        headers = {
            "User-Agent": "battery-policy-ingestion"
        }
    )

    with urllib.request.urlopen(
        request,
        timeout = REQUEST_TIMEOUT_SECONDS
    ) as response:
        article_metadata = json.loads(
            response.read().decode("utf-8")
        )

    if article_metadata.get("id") != FIGSHARE_ARTICLE_ID:
        raise RuntimeError(f"Unexpected Figshare article ID : {article_metadata.get('id')}")

    if (
        article_metadata.get("version")
        != EXPECTED_DATASET_VERSION
    ):
        raise RuntimeError(f"Unexpected Figshare dataset version : {article_metadata.get('version')}")

    if article_metadata.get("doi") != EXPECTED_DATASET_DOI:
        raise RuntimeError(f"Unexpected Figshare dataset DOI : {article_metadata.get('doi')}")

    return article_metadata


# Building the integrity-relevant article metadata identity
def build_article_metadata_integrity_view(article_metadata: dict) -> dict:
    file_records = [
        {
            "file_id": int(
                file_record["id"]
            ),
            "file_name": file_record["name"],
            "size_bytes": int(
                file_record["size"]
            ),
            "download_url": file_record[
                "download_url"
            ],
            "published_md5": file_record.get(
                "computed_md5"
            ),
            "source_folder": article_metadata[
                "folder_structure"
            ][str(file_record["id"])]
        }
        for file_record in article_metadata[
            "files"
        ]
    ]

    return {
        "figshare_article_id": article_metadata[
            "id"
        ],
        "dataset_version": article_metadata[
            "version"
        ],
        "dataset_doi": article_metadata[
            "doi"
        ],
        "files": sorted(
            file_records,
            key = lambda file_record: (
                file_record["file_id"]
            )
        )
    }


# Confirming that two metadata responses identify the same pinned dataset
def validate_article_metadata_compatibility(existing_article_metadata: dict, retrieved_article_metadata: dict) -> None:
    existing_integrity_view = (
        build_article_metadata_integrity_view(
            existing_article_metadata
        )
    )

    retrieved_integrity_view = (
        build_article_metadata_integrity_view(
            retrieved_article_metadata
        )
    )

    if (
        existing_integrity_view
        != retrieved_integrity_view
    ):
        raise RuntimeError("Existing article metadata conflicts with the retrieved pinned dataset inventory.")


# Recording or reusing the compatible pinned article metadata
def validate_or_upload_article_metadata(bucket: storage.Bucket, object_name: str, article_metadata: dict) -> None:
    article_metadata_content = (
        json.dumps(
            article_metadata,
            indent = 2,
            sort_keys = True
        )
        + "\n"
    )

    metadata_blob = bucket.blob(
        object_name
    )

    if metadata_blob.exists():
        existing_article_metadata = json.loads(
            metadata_blob.download_as_text(
                encoding = "utf-8"
            )
        )

        validate_article_metadata_compatibility(
            existing_article_metadata = (
                existing_article_metadata
            ),
            retrieved_article_metadata = (
                article_metadata
            )
        )

        print(
            "Reusing compatible article metadata : "
            f"gs://{bucket.name}/{object_name}"
        )
        return

    metadata_blob.upload_from_string(
        data = article_metadata_content,
        content_type = "application/json",
        if_generation_match = 0
    )

    print(
        "Uploaded article metadata : "
        f"gs://{bucket.name}/{object_name}"
    )


# Validating one dataset-relative source path
def validate_source_path(source_path: str) -> PurePosixPath:
    normalized_path = PurePosixPath(source_path)

    if normalized_path.is_absolute():
        raise ValueError(f"Source path must be relative : {source_path}")

    if ".." in normalized_path.parts:
        raise ValueError(f"Source path cannot contain parent traversal : {source_path}")

    if not normalized_path.name:
        raise ValueError(f"Source path does not identify a file : {source_path}")

    return normalized_path


# Locating one published file within the Figshare folder structure
def find_published_file(article_metadata: dict, source_path: str) -> dict:
    normalized_path = validate_source_path(source_path)
    parent_path = normalized_path.parent.as_posix()

    if parent_path == ".":
        expected_folder = DATASET_FOLDER_NAME

    else:
        expected_folder = (
            f"{DATASET_FOLDER_NAME}/{parent_path}"
        )

    matching_files = [
        file_record
        for file_record in article_metadata["files"]
        if (
            file_record["name"] == normalized_path.name
            and article_metadata["folder_structure"].get(
                str(file_record["id"])
            ) == expected_folder
        )
    ]

    if len(matching_files) != 1:
        raise RuntimeError(f"Expected exactly one published file for {source_path}, found : {len(matching_files)}")

    file_record = matching_files[0]

    return file_record


# Selecting published files for the requested ingestion scope
def select_published_files(article_metadata: dict, selection_mode: str, source_paths: list[str]) -> list[dict]:
    if selection_mode == "complete":
        if source_paths:
            raise ValueError("Source files cannot be supplied when selection mode is complete.")

        file_records = list(
            article_metadata["files"]
        )

    else:
        if not source_paths:
            raise ValueError("At least one source file is required when selection mode is specified.")

        file_records = [
            find_published_file(
                article_metadata = article_metadata,
                source_path = source_path
            )
            for source_path in source_paths
        ]

    if not file_records:
        raise RuntimeError("The requested ingestion selection does not contain files.")

    for file_record in file_records:
        if int(file_record["size"]) <= 0:
            raise RuntimeError(f"Published file has a non-positive size : {file_record['name']}")

        if not file_record.get("download_url"):
            raise RuntimeError(f"Published file does not have a download URL : {file_record['name']}")

        if (
            str(file_record["id"])
            not in article_metadata["folder_structure"]
        ):
            raise RuntimeError(f"Published file does not have a folder assignment : {file_record['name']}")

    return sorted(
        file_records,
        key = lambda file_record: (
            file_record["download_url"]
        )
    )


# Validating the expected contents of the complete pinned dataset
def validate_complete_selection(file_records: list[dict]) -> dict:
    published_md5_count = sum(
        bool(
            file_record.get(
                "computed_md5"
            )
        )
        for file_record in file_records
    )

    size_only_records = [
        file_record
        for file_record in file_records
        if not file_record.get(
            "computed_md5"
        )
    ]

    size_only_file_names = {
        file_record["name"]
        for file_record in size_only_records
    }

    if len(file_records) != EXPECTED_COMPLETE_FILE_COUNT:
        raise RuntimeError(f"Complete dataset selection has an unexpected file count : {len(file_records)}")

    if (
        published_md5_count
        != EXPECTED_COMPLETE_PUBLISHER_MD5_COUNT
    ):
        raise RuntimeError(
            "Complete dataset selection has an unexpected number of published MD5 "
            f"checksums : {published_md5_count}"
        )

    if (
        len(size_only_records)
        != EXPECTED_COMPLETE_SIZE_ONLY_COUNT
    ):
        raise RuntimeError(
            "Complete dataset selection has an unexpected number of files without published MD5 "
            f"checksums : {len(size_only_records)}"
        )

    if size_only_file_names != EXPECTED_SIZE_ONLY_FILE_NAMES:
        raise RuntimeError(
            "Complete dataset selection contains unexpected files without published MD5 "
            f"checksums : {sorted(size_only_file_names)}"
        )

    return {
        "expected_object_count": len(
            file_records
        ),
        "expected_total_size_bytes": sum(
            int(
                file_record["size"]
            )
            for file_record in file_records
        ),
        "expected_publisher_md5_count": (
            published_md5_count
        ),
        "expected_size_only_count": len(
            size_only_records
        )
    }


# Validating one existing destination object before transfer
def validate_existing_object(file_record: dict, destination_size_bytes: int, destination_md5: str | None) -> str:
    expected_size_bytes = int(
        file_record["size"]
    )

    if destination_size_bytes != expected_size_bytes:
        raise RuntimeError(f"Existing destination object conflicts with the published size : {file_record['name']}")

    published_md5 = file_record.get(
        "computed_md5"
    )

    if published_md5:
        expected_md5 = encode_md5_checksum(
            published_md5
        )

        if destination_md5 != expected_md5:
            raise RuntimeError(
                "Existing destination object conflicts with the published MD5 checksum : "
                f"{file_record['name']}"
            )

        return "published_size_and_md5_verified"

    return "published_size_verified_destination_checksum_unavailable"


# Separating reusable destination objects from objects requiring transfer
def identify_transfer_requirements(bucket: storage.Bucket, file_records: list[dict]) -> tuple[list[dict], list[dict]]:
    transfer_file_records = []
    reusable_file_records = []

    for file_record in file_records:
        object_name = build_destination_object_name(
            file_record["download_url"]
        )

        blob = bucket.blob(
            object_name
        )

        if not blob.exists():
            transfer_file_records.append(
                file_record
            )
            continue

        blob.reload()

        verification_status = validate_existing_object(
            file_record = file_record,
            destination_size_bytes = int(
                blob.size
            ),
            destination_md5 = blob.md5_hash
        )

        if not file_record.get(
            "computed_md5"
        ):
            destination_sha256 = calculate_blob_sha256(
                blob = blob
            )

            validate_size_only_integrity_baseline(
                bucket = bucket,
                file_record = file_record,
                destination_object_name = object_name,
                destination_sha256 = destination_sha256,
                allow_creation = False
            )

        reusable_file_records.append(
            file_record
        )

        print(
            "Reusing existing validated object : "
            f"gs://{bucket.name}/{object_name} : "
            f"{verification_status}"
        )

    return (
        transfer_file_records,
        reusable_file_records
    )


# Building the immutable published-source manifest
def build_source_manifest_content(article_metadata: dict, file_records: list[dict]) -> str:
    source_records = [
        {
            "source_path": build_published_source_path(
                article_metadata = article_metadata,
                file_record = file_record
            ),
            "file_id": file_record["id"],
            "file_name": file_record["name"],
            "download_url": file_record[
                "download_url"
            ],
            "size_bytes": int(
                file_record["size"]
            ),
            "published_md5": file_record.get(
                "computed_md5"
            ),
            "destination_object": (
                build_destination_object_name(
                    file_record[
                        "download_url"
                    ]
                )
            )
        }
        for file_record in file_records
    ]

    return (
        json.dumps(
            {
                "figshare_article_id": (
                    FIGSHARE_ARTICLE_ID
                ),
                "dataset_version": (
                    EXPECTED_DATASET_VERSION
                ),
                "dataset_doi": (
                    EXPECTED_DATASET_DOI
                ),
                "file_count": len(
                    source_records
                ),
                "total_size_bytes": sum(
                    source_record[
                        "size_bytes"
                    ]
                    for source_record in source_records
                ),
                "files": source_records
            },
            indent = 2,
            sort_keys = True
        )
        + "\n"
    )


# Building the immutable integrity baseline for a size-only source
def build_size_only_integrity_baseline_content(file_record: dict, destination_object_name: str,
                                               destination_sha256: str
) -> str:
    return (
        json.dumps(
            {
                "figshare_article_id": (
                    FIGSHARE_ARTICLE_ID
                ),
                "dataset_version": (
                    EXPECTED_DATASET_VERSION
                ),
                "file_id": file_record["id"],
                "file_name": file_record["name"],
                "download_url": file_record[
                    "download_url"
                ],
                "size_bytes": int(
                    file_record["size"]
                ),
                "destination_object": (
                    destination_object_name
                ),
                "destination_sha256": (
                    destination_sha256
                )
            },
            indent = 2,
            sort_keys = True
        )
        + "\n"
    )


# Recording or validating the destination integrity baseline
def validate_size_only_integrity_baseline(bucket: storage.Bucket, file_record: dict,
                                          destination_object_name: str, destination_sha256: str,
                                          allow_creation: bool
) -> str:
    baseline_object_name = (
        f"{METADATA_OBJECT_PREFIX}"
        f"size_only_integrity_{file_record['id']}.json"
    )

    baseline_blob = bucket.blob(
        baseline_object_name
    )

    if (
        not baseline_blob.exists()
        and not allow_creation
    ):
        raise RuntimeError(
            "Existing size-only destination object does not have a recorded integrity "
            f"baseline : {file_record['name']}"
        )

    baseline_content = (
        build_size_only_integrity_baseline_content(
            file_record = file_record,
            destination_object_name = (
                destination_object_name
            ),
            destination_sha256 = destination_sha256
        )
    )

    upload_text_artifact(
        bucket = bucket,
        object_name = baseline_object_name,
        content = baseline_content,
        content_type = "application/json"
    )

    return baseline_object_name


# Validating the final complete raw-object inventory
def validate_complete_inventory(file_records: list[dict], validation_records: list[dict],
                                observed_object_names: set[str]
) -> dict:
    selection_summary = (
        validate_complete_selection(
            file_records = file_records
        )
    )

    expected_object_names = {
        build_destination_object_name(
            file_record["download_url"]
        )
        for file_record in file_records
    }

    missing_object_names = sorted(
        expected_object_names.difference(
            observed_object_names
        )
    )

    unexpected_object_names = sorted(
        observed_object_names.difference(
            expected_object_names
        )
    )

    if missing_object_names:
        raise RuntimeError(f"Complete cloud inventory is missing expected objects : {missing_object_names[:20]}")

    if unexpected_object_names:
        raise RuntimeError(f"Complete cloud inventory contains unexpected objects : {unexpected_object_names[:20]}")

    if (
        len(validation_records)
        != selection_summary[
            "expected_object_count"
        ]
    ):
        raise RuntimeError(f"Complete cloud inventory validation has an unexpected record count : {len(validation_records)}")

    validated_total_size_bytes = sum(
        validation_record[
            "size_bytes"
        ]
        for validation_record in validation_records
    )

    if (
        validated_total_size_bytes
        != selection_summary[
            "expected_total_size_bytes"
        ]
    ):
        raise RuntimeError(f"Complete cloud inventory has an unexpected total size : {validated_total_size_bytes}")

    verification_status_counts = Counter(
        validation_record[
            "verification_status"
        ]
        for validation_record in validation_records
    )

    if (
        verification_status_counts[
            "published_size_and_md5_verified"
        ]
        != EXPECTED_COMPLETE_PUBLISHER_MD5_COUNT
    ):
        raise RuntimeError("Complete cloud inventory has an unexpected publisher-MD5 validation count.")

    if (
        verification_status_counts[
            "published_size_verified_destination_sha256_recorded"
        ]
        != EXPECTED_COMPLETE_SIZE_ONLY_COUNT
    ):
        raise RuntimeError("Complete cloud inventory has an unexpected size-only validation count.")

    return {
        **selection_summary,
        "validated_object_count": len(
            validation_records
        ),
        "validated_total_size_bytes": (
            validated_total_size_bytes
        ),
        "verification_status_counts": dict(
            sorted(
                verification_status_counts.items()
            )
        ),
        "missing_object_count": 0,
        "unexpected_object_count": 0
    }


# Converting the published hexadecimal MD5 into the transfer format

def encode_md5_checksum(hexadecimal_md5: str) -> str:
    try:
        checksum_bytes = bytes.fromhex(hexadecimal_md5)

    except ValueError as exc:
        raise ValueError(f"Invalid hexadecimal MD5 checksum : {hexadecimal_md5}") from exc

    if len(checksum_bytes) != 16:
        raise ValueError(f"MD5 checksum must contain 16 bytes : {hexadecimal_md5}")

    return base64.b64encode(
        checksum_bytes
    ).decode("ascii")


# Building the integrity-protected URL list required by Storage Transfer
def build_transfer_list(file_records: list[dict]) -> str:
    transfer_lines = [
        "TsvHttpData-1.0"
    ]

    for file_record in sorted(
        file_records,
        key = lambda record: record["download_url"]
    ):
        transfer_fields = [
            file_record["download_url"],
            str(file_record["size"])
        ]

        published_md5 = file_record.get(
            "computed_md5"
        )

        if published_md5:
            transfer_fields.append(
                encode_md5_checksum(
                    published_md5
                )
            )

        transfer_lines.append(
            "\t".join(
                transfer_fields
            )
        )

    return "\n".join(
        transfer_lines
    ) + "\n"


# Creating an immutable object name from the transfer-list content
def build_transfer_list_object_name(transfer_list_content: str) -> str:
    content_digest = hashlib.sha256(
        transfer_list_content.encode("utf-8")
    ).hexdigest()

    return (
        f"{METADATA_OBJECT_PREFIX}"
        f"source_urls_{content_digest}.tsv"
    )


# Uploading one immutable metadata artifact or reusing an identical copy
def upload_text_artifact(bucket: storage.Bucket, object_name: str,
                         content: str, content_type: str
) -> None:
    blob = bucket.blob(object_name)

    if blob.exists():
        existing_content = blob.download_as_text()

        if existing_content != content:
            raise RuntimeError(
                f"Existing Cloud Storage metadata differs from the requested content : gs://{bucket.name}/"
                f"{object_name}"
            )

        print(
            "Reusing existing metadata artifact : "
            f"gs://{bucket.name}/{object_name}"
        )
        return

    blob.upload_from_string(
        data = content,
        content_type = content_type,
        if_generation_match = 0
    )

    print(
        "Uploaded metadata artifact : "
        f"gs://{bucket.name}/{object_name}"
    )


# Constructing the destination object name used by URL-list transfers
def build_destination_object_name(download_url: str) -> str:
    parsed_url = urlparse(download_url)

    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.netloc
        or not parsed_url.path
    ):
        raise ValueError(f"Invalid public download URL : {download_url}")

    return (
        f"{RAW_OBJECT_PREFIX}"
        f"{parsed_url.netloc}/"
        f"{parsed_url.path.lstrip('/')}"
    )


# Creating and running one non-overwriting transfer job
def run_transfer_job(project_id: str, bucket_name: str,
                     transfer_list_url: str, transfer_timeout_seconds: int | None
) -> str:
    transfer_client = (
        storage_transfer.StorageTransferServiceClient()
    )

    create_request = (
        storage_transfer.CreateTransferJobRequest(
            {
                "transfer_job": {
                    "project_id": project_id,
                    "description": (
                        "Figshare battery aging dataset ingestion"
                    ),
                    "status": (
                        storage_transfer.TransferJob.Status.ENABLED
                    ),
                    "transfer_spec": {
                        "http_data_source": {
                            "list_url": transfer_list_url
                        },
                        "gcs_data_sink": {
                            "bucket_name": bucket_name,
                            "path": RAW_OBJECT_PREFIX
                        },
                        "transfer_options": {
                            "overwrite_when": (
                                storage_transfer
                                .TransferOptions
                                .OverwriteWhen
                                .NEVER
                            )
                        }
                    }
                }
            }
        )
    )

    transfer_job = transfer_client.create_transfer_job(
        request = create_request
    )

    operation = transfer_client.run_transfer_job(
        request = {
            "job_name": transfer_job.name,
            "project_id": project_id
        }
    )

    print(f"Transfer job created : {transfer_job.name}")
    print("Waiting for the transfer operation to complete.")

    operation.result(
        timeout = transfer_timeout_seconds
    )

    return transfer_job.name


# Calculating a destination SHA-256 baseline without loading the object into memory
def calculate_blob_sha256(blob: storage.Blob) -> str:
    content_digest = hashlib.sha256()

    with blob.open(
        mode = "rb"
    ) as binary_stream:
        while True:
            content_chunk = binary_stream.read(
                TRANSFER_CHUNK_SIZE_BYTES
            )

            if not content_chunk:
                break

            content_digest.update(
                content_chunk
            )

    return content_digest.hexdigest()


# Building the published source identity for one file
def build_published_source_path(article_metadata: dict, file_record: dict) -> str:
    source_folder = article_metadata[
        "folder_structure"
    ][str(file_record["id"])]

    return (
        f"{source_folder}/{file_record['name']}"
    )


# Validating every transferred object against available source metadata
def validate_transferred_objects(bucket: storage.Bucket, article_metadata: dict, file_records: list[dict]) -> list[dict]:
    validation_records = []

    for file_record in file_records:
        integrity_baseline_object = None
        object_name = build_destination_object_name(
            file_record["download_url"]
        )

        blob = bucket.blob(
            object_name
        )

        if not blob.exists():
            raise RuntimeError(f"Transferred object was not found : gs://{bucket.name}/{object_name}")

        blob.reload()

        if int(blob.size) != int(file_record["size"]):
            raise RuntimeError(f"Transferred object size does not match the published size : {object_name}")

        published_md5 = file_record.get(
            "computed_md5"
        )

        destination_sha256 = None

        if published_md5:
            expected_md5 = encode_md5_checksum(
                published_md5
            )

            if blob.md5_hash != expected_md5:
                raise RuntimeError(f"Transferred object MD5 does not match the published checksum : {object_name}"
                )

            verification_status = (
                "published_size_and_md5_verified"
            )

        else:
            destination_sha256 = calculate_blob_sha256(
                blob = blob
            )

            verification_status = (
                "published_size_verified_destination_sha256_recorded"
            )

            integrity_baseline_object_name = (
                validate_size_only_integrity_baseline(
                    bucket = bucket,
                    file_record = file_record,
                    destination_object_name = object_name,
                    destination_sha256 = destination_sha256,
                    allow_creation = True
                )
            )

            integrity_baseline_object = (
                f"gs://{bucket.name}/"
                f"{integrity_baseline_object_name}"
            )

        validation_records.append(
            {
                "source_path": build_published_source_path(
                    article_metadata = article_metadata,
                    file_record = file_record
                ),
                "file_id": file_record["id"],
                "download_url": file_record[
                    "download_url"
                ],
                "destination_object": object_name,
                "cloud_object": (
                    f"gs://{bucket.name}/{object_name}"
                ),
                "size_bytes": int(blob.size),
                "published_md5": published_md5,
                "destination_md5": blob.md5_hash,
                "destination_sha256": destination_sha256,
                "verification_status": verification_status,
                "integrity_baseline_object": (
                    integrity_baseline_object
                )
            }
        )

        print(
            "Validated transferred object : "
            f"gs://{bucket.name}/{object_name}"
        )

    return validation_records


# Transferring selected immutable source files into Cloud Storage
def main() -> None:
    args = parse_args()

    if (
        args.transfer_timeout_seconds is not None
        and args.transfer_timeout_seconds <= 0
    ):
        raise ValueError("Transfer timeout must be greater than zero.")

    article_metadata = retrieve_article_metadata()

    requested_source_paths = list(
        dict.fromkeys(
            args.source_file
        )
    )

    file_records = select_published_files(
        article_metadata = article_metadata,
        selection_mode = args.selection_mode,
        source_paths = requested_source_paths
    )

    storage_client = storage.Client(
        project = args.project_id
    )

    bucket = storage_client.bucket(
        args.bucket_name
    )

    if not bucket.exists():
        raise RuntimeError(f"Cloud Storage bucket was not found : {args.bucket_name}")

    if args.selection_mode == "complete":
        validate_complete_selection(
            file_records = file_records
        )

    article_metadata_object = (
        f"{METADATA_OBJECT_PREFIX}"
        "article_metadata.json"
    )

    # Recording or reusing compatible pinned article metadata
    validate_or_upload_article_metadata(
        bucket = bucket,
        object_name = article_metadata_object,
        article_metadata = article_metadata
    )

    # Recording the complete selected source inventory
    source_manifest_content = (
        build_source_manifest_content(
            article_metadata = article_metadata,
            file_records = file_records
        )
    )

    source_manifest_digest = hashlib.sha256(
        source_manifest_content.encode(
            "utf-8"
        )
    ).hexdigest()

    source_manifest_object = (
        f"{METADATA_OBJECT_PREFIX}"
        f"source_manifest_{source_manifest_digest}.json"
    )

    upload_text_artifact(
        bucket = bucket,
        object_name = source_manifest_object,
        content = source_manifest_content,
        content_type = "application/json"
    )

    # Identifying only the objects that still require transfer
    (
        transfer_file_records,
        reusable_file_records
    ) = identify_transfer_requirements(
        bucket = bucket,
        file_records = file_records
    )

    transfer_job_name = None
    transfer_list_object = None

    if transfer_file_records:
        transfer_list_content = build_transfer_list(
            file_records = transfer_file_records
        )

        transfer_list_object = (
            build_transfer_list_object_name(
                transfer_list_content = (
                    transfer_list_content
                )
            )
        )

        upload_text_artifact(
            bucket = bucket,
            object_name = transfer_list_object,
            content = transfer_list_content,
            content_type = "text/tab-separated-values"
        )

        transfer_list_url = (
            f"gs://{args.bucket_name}/"
            f"{transfer_list_object}"
        )

        transfer_job_name = run_transfer_job(
            project_id = args.project_id,
            bucket_name = args.bucket_name,
            transfer_list_url = transfer_list_url,
            transfer_timeout_seconds = (
                args.transfer_timeout_seconds
            )
        )

    else:
        print(
            "No transfer job is required; every selected object already passes its available source-integrity checks."
        )

    # Revalidating every selected object after transfer or reuse
    validation_records = validate_transferred_objects(
        bucket = bucket,
        article_metadata = article_metadata,
        file_records = file_records
    )

    if args.selection_mode == "complete":
        observed_object_names = {
            blob.name
            for blob in bucket.list_blobs(
                prefix = RAW_OBJECT_PREFIX
            )
        }

        inventory_summary = validate_complete_inventory(
            file_records = file_records,
            validation_records = validation_records,
            observed_object_names = (
                observed_object_names
            )
        )

    else:
        inventory_summary = {
            "expected_object_count": len(
                file_records
            ),
            "expected_total_size_bytes": sum(
                int(
                    file_record["size"]
                )
                for file_record in file_records
            ),
            "validated_object_count": len(
                validation_records
            ),
            "validated_total_size_bytes": sum(
                validation_record[
                    "size_bytes"
                ]
                for validation_record in (
                    validation_records
                )
            )
        }

    landing_validation_content = (
        json.dumps(
            {
                "figshare_article_id": (
                    FIGSHARE_ARTICLE_ID
                ),
                "dataset_version": (
                    EXPECTED_DATASET_VERSION
                ),
                "dataset_doi": (
                    EXPECTED_DATASET_DOI
                ),
                "selection_mode": (
                    args.selection_mode
                ),
                "source_manifest_object": (
                    f"gs://{args.bucket_name}/"
                    f"{source_manifest_object}"
                ),
                "transfer_list_object": (
                    None
                    if transfer_list_object is None
                    else (
                        f"gs://{args.bucket_name}/"
                        f"{transfer_list_object}"
                    )
                ),
                "transfer_job_name": (
                    transfer_job_name
                ),
                "transferred_object_count": len(
                    transfer_file_records
                ),
                "reused_object_count": len(
                    reusable_file_records
                ),
                "inventory": inventory_summary,
                "objects": validation_records
            },
            indent = 2,
            sort_keys = True
        )
        + "\n"
    )

    landing_validation_digest = hashlib.sha256(
        landing_validation_content.encode(
            "utf-8"
        )
    ).hexdigest()

    landing_validation_object = (
        f"{METADATA_OBJECT_PREFIX}"
        f"landing_validation_"
        f"{landing_validation_digest}.json"
    )

    # Recording the immutable landing-validation result
    upload_text_artifact(
        bucket = bucket,
        object_name = landing_validation_object,
        content = landing_validation_content,
        content_type = "application/json"
    )

    print("Dataset ingestion completed successfully.")

    if transfer_job_name is not None:
        print(
            f"Transfer job : {transfer_job_name}"
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(f"Dataset ingestion failed : {exc}")
        sys.exit(1)
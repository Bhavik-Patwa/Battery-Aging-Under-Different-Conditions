import base64

import pytest

from scripts.ingest_dataset import (
    build_destination_object_name,
    build_source_manifest_content,
    build_transfer_list,
    build_transfer_list_object_name,
    encode_md5_checksum,
    find_published_file,
    select_published_files,
    validate_complete_inventory,
    validate_complete_selection,
    validate_existing_object,
    validate_article_metadata_compatibility
)

ARTICLE_METADATA = {
    "files": [
        {
            "id": 101,
            "name": "TP_k01_01.zip",
            "size": 200,
            "download_url": (
                "https://ndownloader.figshare.com/files/101"
            ),
            "computed_md5": (
                "d41d8cd98f00b204e9800998ecf8427e"
            )
        },
        {
            "id": 102,
            "name": "TP_k01_01.zip",
            "size": 300,
            "download_url": (
                "https://ndownloader.figshare.com/files/102"
            ),
            "computed_md5": (
                "0cc175b9c0f1b6a831c399e269772661"
            )
        }
    ],
    "folder_structure": {
        "101": "Multi-Stage_Aging_Study/Stage_1",
        "102": "Multi-Stage_Aging_Study/Stage_2"
    }
}


# Building the expected complete pinned-dataset records
def build_complete_file_records() -> list[dict]:
    file_records = []

    for file_number in range(
        1,
        281
    ):
        if file_number == 280:
            file_name = "TP_z05_01.zip"
            published_md5 = None

        else:
            file_name = (
                f"published_file_{file_number:03d}.zip"
            )
            published_md5 = (
                "d41d8cd98f00b204e9800998ecf8427e"
            )

        file_records.append(
            {
                "id": file_number,
                "name": file_name,
                "size": file_number * 100,
                "download_url": (
                    "https://ndownloader.figshare.com/"
                    f"files/{file_number}"
                ),
                "computed_md5": published_md5
            }
        )

    return file_records


# Confirming that stage-specific paths resolve duplicate filenames
def test_find_published_file_uses_source_folder() -> None:
    file_record = find_published_file(
        article_metadata = ARTICLE_METADATA,
        source_path = "Stage_2/TP_k01_01.zip"
    )

    assert file_record["id"] == 102


# Confirming that hexadecimal checksums use the required base64 format
def test_encode_md5_checksum() -> None:
    hexadecimal_md5 = (
        "d41d8cd98f00b204e9800998ecf8427e"
    )

    expected_checksum = base64.b64encode(
        bytes.fromhex(hexadecimal_md5)
    ).decode("ascii")

    assert (
        encode_md5_checksum(hexadecimal_md5)
        == expected_checksum
    )


# Rejecting malformed checksum values before cloud transfer
def test_encode_md5_checksum_rejects_invalid_value() -> None:
    with pytest.raises(
        ValueError,
        match = "Invalid hexadecimal MD5 checksum"
    ):
        encode_md5_checksum("not-a-checksum")


# Confirming that transfer records are sorted and checksum protected
def test_build_transfer_list() -> None:
    file_records = [
        ARTICLE_METADATA["files"][1],
        ARTICLE_METADATA["files"][0]
    ]

    transfer_list = build_transfer_list(
        file_records = file_records
    )

    transfer_lines = transfer_list.splitlines()

    expected_first_checksum = encode_md5_checksum(
        ARTICLE_METADATA["files"][0]["computed_md5"]
    )

    expected_second_checksum = encode_md5_checksum(
        ARTICLE_METADATA["files"][1]["computed_md5"]
    )

    assert transfer_lines == [
        "TsvHttpData-1.0",
        (
            "https://ndownloader.figshare.com/files/101"
            f"\t200\t{expected_first_checksum}"
        ),
        (
            "https://ndownloader.figshare.com/files/102"
            f"\t300\t{expected_second_checksum}"
        )
    ]


# Confirming that files without published checksums retain size validation
def test_build_transfer_list_allows_missing_md5() -> None:
    file_records = [
        {
            "id": 103,
            "name": "TP_z05_01.zip",
            "size": 400,
            "download_url": (
                "https://ndownloader.figshare.com/files/103"
            ),
            "computed_md5": None
        }
    ]

    transfer_list = build_transfer_list(
        file_records = file_records
    )

    assert transfer_list.splitlines() == [
        "TsvHttpData-1.0",
        (
            "https://ndownloader.figshare.com/files/103"
            "\t400"
        )
    ]


# Confirming that transfer-list names change only when their content changes
def test_build_transfer_list_object_name() -> None:
    transfer_list_content = (
        "TsvHttpData-1.0\n"
        "https://example.com/file.zip\t100\tchecksum\n"
    )

    first_object_name = build_transfer_list_object_name(
        transfer_list_content = transfer_list_content
    )

    repeated_object_name = build_transfer_list_object_name(
        transfer_list_content = transfer_list_content
    )

    changed_object_name = build_transfer_list_object_name(
        transfer_list_content = (
            transfer_list_content
            + "https://example.com/other.zip\t200\tchecksum\n"
        )
    )

    assert first_object_name == repeated_object_name
    assert first_object_name != changed_object_name
    assert first_object_name.endswith(".tsv")


# Confirming the complete destination naming used by Storage Transfer
def test_build_destination_object_name() -> None:
    object_name = build_destination_object_name(
        "https://ndownloader.figshare.com/files/101"
    )

    assert object_name == (
        "raw/figshare/25975315/v1/"
        "ndownloader.figshare.com/files/101"
    )


# Confirming that complete selection discovers every published file
def test_select_published_files_complete() -> None:
    selected_file_records = select_published_files(
        article_metadata = ARTICLE_METADATA,
        selection_mode = "complete",
        source_paths = []
    )

    assert {
        file_record["id"]
        for file_record in selected_file_records
    } == {
        101,
        102
    }


# Confirming the expected complete-dataset integrity split
def test_validate_complete_selection() -> None:
    selection_summary = validate_complete_selection(
        file_records = build_complete_file_records()
    )

    assert (
        selection_summary[
            "expected_object_count"
        ]
        == 280
    )

    assert (
        selection_summary[
            "expected_publisher_md5_count"
        ]
        == 279
    )

    assert (
        selection_summary[
            "expected_size_only_count"
        ]
        == 1
    )


# Confirming that a matching existing object is reusable
def test_validate_existing_object_accepts_matching_object() -> None:
    file_record = ARTICLE_METADATA["files"][0]

    verification_status = validate_existing_object(
        file_record = file_record,
        destination_size_bytes = 200,
        destination_md5 = encode_md5_checksum(
            file_record["computed_md5"]
        )
    )

    assert (
        verification_status
        == "published_size_and_md5_verified"
    )


# Rejecting an existing object with a conflicting size
def test_validate_existing_object_rejects_size_conflict() -> None:
    with pytest.raises(
        RuntimeError,
        match = "conflicts with the published size"
    ):
        validate_existing_object(
            file_record = ARTICLE_METADATA["files"][0],
            destination_size_bytes = 201,
            destination_md5 = encode_md5_checksum(
                ARTICLE_METADATA[
                    "files"
                ][0][
                    "computed_md5"
                ]
            )
        )


# Rejecting an existing object with a conflicting checksum
def test_validate_existing_object_rejects_md5_conflict() -> None:
    with pytest.raises(
        RuntimeError,
        match = "conflicts with the published MD5 checksum"
    ):
        validate_existing_object(
            file_record = ARTICLE_METADATA["files"][0],
            destination_size_bytes = 200,
            destination_md5 = encode_md5_checksum(
                ARTICLE_METADATA[
                    "files"
                ][1][
                    "computed_md5"
                ]
            )
        )


# Confirming that the source manifest preserves provenance
def test_build_source_manifest_content() -> None:
    source_manifest = build_source_manifest_content(
        article_metadata = ARTICLE_METADATA,
        file_records = [
            ARTICLE_METADATA["files"][0]
        ]
    )

    assert (
        "https://ndownloader.figshare.com/files/101"
        in source_manifest
    )

    assert (
        "Multi-Stage_Aging_Study/Stage_1/"
        "TP_k01_01.zip"
        in source_manifest
    )


# Confirming that the complete inventory rejects unexpected objects
def test_validate_complete_inventory_rejects_unexpected_object() -> None:
    file_records = build_complete_file_records()

    validation_records = []

    for file_record in file_records:
        if file_record["computed_md5"]:
            verification_status = (
                "published_size_and_md5_verified"
            )

        else:
            verification_status = (
                "published_size_verified_"
                "destination_sha256_recorded"
            )

        validation_records.append(
            {
                "size_bytes": int(
                    file_record["size"]
                ),
                "verification_status": (
                    verification_status
                )
            }
        )

    observed_object_names = {
        build_destination_object_name(
            file_record["download_url"]
        )
        for file_record in file_records
    }

    observed_object_names.add(
        (
            "raw/figshare/25975315/v1/"
            "unexpected/object"
        )
    )

    with pytest.raises(
        RuntimeError,
        match = "unexpected objects"
    ):
        validate_complete_inventory(
            file_records = file_records,
            validation_records = validation_records,
            observed_object_names = (
                observed_object_names
            )
        )


# Confirming that non-integrity metadata changes do not invalidate the pinned dataset
def test_validate_article_metadata_compatibility() -> None:
    existing_article_metadata = {
        "id": 25975315,
        "version": 1,
        "doi": "10.6084/m9.figshare.25975315.v1",
        "title": "Existing title",
        **ARTICLE_METADATA
    }

    retrieved_article_metadata = {
        "id": 25975315,
        "version": 1,
        "doi": "10.6084/m9.figshare.25975315.v1",
        "title": "Updated descriptive title",
        **ARTICLE_METADATA
    }

    validate_article_metadata_compatibility(
        existing_article_metadata = (
            existing_article_metadata
        ),
        retrieved_article_metadata = (
            retrieved_article_metadata
        )
    )


# Rejecting article metadata that changes the pinned file inventory
def test_validate_article_metadata_compatibility_rejects_inventory_change() -> None:
    existing_article_metadata = {
        "id": 25975315,
        "version": 1,
        "doi": "10.6084/m9.figshare.25975315.v1",
        **ARTICLE_METADATA
    }

    conflicting_article_metadata = {
        **existing_article_metadata,
        "files": [
            {
                **ARTICLE_METADATA["files"][0],
                "size": 201
            },
            ARTICLE_METADATA["files"][1]
        ]
    }

    with pytest.raises(
        RuntimeError,
        match = "conflicts with the retrieved pinned dataset inventory"
    ):
        validate_article_metadata_compatibility(
            existing_article_metadata = (
                existing_article_metadata
            ),
            retrieved_article_metadata = (
                conflicting_article_metadata
            )
        )
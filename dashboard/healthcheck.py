import argparse
import os

from pathlib import Path

from dashboard.artifacts import ArtifactStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Parsing immutable artifact locations
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = (
            "Validate every report and table required by the evidence dashboard."
        )
    )

    parser.add_argument(
        "--artifact-root",
        type = Path,
        default = Path(
            os.environ.get(
                "EVIDENCE_ARTIFACT_ROOT",
                PROJECT_ROOT
                / "data"
                / "processed"
            )
        )
    )

    parser.add_argument(
        "--policy-path",
        type = Path,
        default = Path(
            os.environ.get(
                "EVIDENCE_DASHBOARD_POLICY",
                PROJECT_ROOT
                / "configs"
                / "evidence_dashboard_policy.json"
            )
        )
    )

    return parser.parse_args()


# Failing closed when any presentation artifact is invalid
def main() -> None:
    arguments = parse_args()

    store = ArtifactStore(
        artifact_root = arguments.artifact_root,
        policy_path = arguments.policy_path
    )

    summary = store.validate_all()

    print(
        "Evidence artifacts validated."
    )
    print(
        "Report count :",
        summary[
            "source_count"
        ]
    )
    print(
        "Table count :",
        summary[
            "table_count"
        ]
    )
    print(
        "Policy SHA-256 :",
        summary[
            "policy_sha256"
        ]
    )


if __name__ == "__main__":
    main()
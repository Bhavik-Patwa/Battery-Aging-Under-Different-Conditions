import json
import os
import logging

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.artifacts import (
    ArtifactStore,
    ArtifactValidationError,
    convert_numeric_columns,
    filter_records
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ARTIFACT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

DEFAULT_POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "evidence_dashboard_policy.json"
)

LOGGER = logging.getLogger(
    __name__
)


# Rendering one deterministic categorical filter
def categorical_filter(frame: pd.DataFrame, column_name: str,
                       label: str, key: str,
                       default_values: list[str] | None = None
) -> list[str]:
    values = sorted(
        value
        for value in frame[
            column_name
        ].unique()
        if value
    )

    if default_values is None:
        default_values = values

    registered_defaults = [
        value
        for value in default_values
        if value in values
    ]

    return st.multiselect(
        label,
        options = values,
        default = registered_defaults,
        key = key
    )


# Selecting one categorical context for a comparable display
def categorical_selection(frame: pd.DataFrame, column_name: str,
                          label: str, key: str
) -> str:
    if column_name not in frame.columns:
        raise ArtifactValidationError(f"Requested filter column is unavailable : {column_name}")

    values = sorted(
        value
        for value in frame[
            column_name
        ].unique()
        if value
    )

    if not values:
        raise ArtifactValidationError(f"Categorical context has no available values : {column_name}")

    return st.selectbox(
        label,
        options = values,
        key = key
    )


# Loading the complete validated evidence inventory once
@st.cache_resource
def load_store(artifact_root: str, policy_path: str) -> tuple[ArtifactStore, dict]:
    store = ArtifactStore(
        artifact_root = Path(
            artifact_root
        ),
        policy_path = Path(
            policy_path
        )
    )

    validation_summary = store.validate_all()

    return (
        store,
        validation_summary
    )


# Rendering the scientific claim boundary
def render_boundaries(store: ArtifactStore) -> None:
    st.header(
        "Evidence boundaries"
    )

    st.warning(
        "Results are descriptive, associational, or predictive. Causal treatment effects, operating-policy recommendations, "
        "treatment optimization, novel-condition prediction, and discrete point contrasts are not authorized."
    )

    feature_report = store.report(
        "feature_extraction"
    )
    design_report = store.report(
        "experimental_design"
    )
    identification_report = store.report(
        "causal_identification"
    )
    robustness_report = store.report(
        "robustness_assessment"
    )

    columns = st.columns(
        4
    )

    columns[0].metric(
        "Cells",
        feature_report[
            "summary"
        ][
            "cell_count"
        ]
    )
    columns[1].metric(
        "Condition protocols",
        feature_report[
            "summary"
        ][
            "condition_protocol_count"
        ]
    )
    columns[2].metric(
        "Estimands assessed",
        identification_report[
            "estimand_count"
        ]
    )
    columns[3].metric(
        "Assessment status",
        robustness_report[
            "scientific_assessment_status"
        ].replace(
            "_",
            " "
        )
    )

    st.subheader(
        "Authorization status"
    )

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "claim": "Continuous supported trajectories",
                    "status": design_report[
                        "trajectory_estimand_status"
                    ]
                },
                {
                    "claim": "Causal modeling",
                    "status": identification_report[
                        "causal_modeling_status"
                    ]
                },
                {
                    "claim": "Associational hierarchical modeling",
                    "status": identification_report[
                        "associational_hierarchical_modeling_status"
                    ]
                },
                {
                    "claim": "Discrete point comparisons",
                    "status": design_report[
                        "discrete_comparison_horizon_status"
                    ]
                },
                {
                    "claim": "Policy recommendations",
                    "status": "not_authorized"
                }
            ]
        ),
        hide_index = True,
        width = "stretch"
    )

    st.subheader(
        "Identification assumptions"
    )

    assumptions = store.table(
        "causal_identification",
        "assumption_inventory"
    )

    evidence_statuses = categorical_filter(
        assumptions,
        "evidence_status",
        "Evidence status",
        "assumption_evidence_status"
    )

    assumptions = filter_records(
        assumptions,
        {
            "evidence_status": evidence_statuses
        }
    )

    st.dataframe(
        assumptions[
            [
                "assumption_name",
                "assumption_scope",
                "evidence_status",
                "required_for_point_identification",
                "evidence_source",
                "observed_evidence",
                "analytical_implication",
                "reason_code"
            ]
        ],
        hide_index = True,
        width = "stretch"
    )


# Rendering treatment support without interpreting it causally
def render_experimental_support(store: ArtifactStore) -> None:
    st.header(
        "Experimental support"
    )

    support = store.table(
        "experimental_design",
        "treatment_combination_support"
    )

    aging_modes = categorical_filter(
        support,
        "aging_mode",
        "Aging mode",
        "support_aging_mode"
    )

    support = filter_records(
        support,
        {
            "aging_mode": aging_modes
        }
    )

    visible_columns = [
        "treatment_combination_identity",
        "aging_mode",
        "ambient_temperature_celsius",
        "maximum_state_of_charge_fraction",
        "depth_of_discharge_fraction",
        "charge_rate_c",
        "discharge_rate_c",
        "reference_test_schedule",
        "condition_protocol_count",
        "independent_cell_count",
        "stage_counts",
        "laboratory_counts"
    ]

    st.dataframe(
        support[
            visible_columns
        ],
        hide_index = True,
        width = "stretch"
    )

    st.subheader(
        "Identification decisions"
    )

    identification = store.table(
        "causal_identification",
        "estimand_identification"
    )

    outcomes = categorical_filter(
        identification,
        "outcome",
        "Outcome",
        "identification_outcome"
    )
    cohorts = categorical_filter(
        identification,
        "cohort",
        "Cohort",
        "identification_cohort"
    )
    statuses = categorical_filter(
        identification,
        "point_identification_status",
        "Identification status",
        "identification_status"
    )

    identification = filter_records(
        identification,
        {
            "outcome": outcomes,
            "cohort": cohorts,
            "point_identification_status": statuses
        }
    )

    st.dataframe(
        identification[
            [
                "estimand_decision_identity",
                "aging_mode",
                "first_condition_protocol_identity",
                "second_condition_protocol_identity",
                "outcome",
                "cohort",
                "horizon_axis",
                "final_support_classification",
                "point_identification_status",
                "associational_model_status",
                "causal_identification_reason_codes"
            ]
        ],
        hide_index = True,
        width = "stretch"
    )


# Rendering only supported continuous trajectories
def render_trajectories(store: ArtifactStore) -> None:
    st.header(
        "Continuous trajectories"
    )

    source = st.radio(
        "Evidence source",
        options = [
            "Descriptive summaries",
            "Bayesian associational summaries"
        ],
        horizontal = True
    )

    if source == "Descriptive summaries":
        frame = store.table(
            "descriptive_modeling",
            "trajectory_summaries"
        )

        frame = frame[
            frame[
                "record_type"
            ] == "condition"
        ].copy()

        for column_name, label in [
            ("aging_mode", "Aging mode"),
            ("dataset_stage", "Dataset stage"),
            ("outcome", "Outcome"),
            ("cohort", "Cohort"),
            ("horizon_axis", "Horizon axis")
        ]:
            selected_value = categorical_selection(
                frame,
                column_name,
                label,
                f"descriptive_{column_name}"
            )

            frame = filter_records(
                frame,
                {
                    column_name: [
                        selected_value
                    ]
                }
            )

        condition_values = sorted(
            frame[
                "condition_protocol_identity"
            ].unique()
        )

        selected_conditions = st.multiselect(
            "Condition protocols",
            options = condition_values,
            default = condition_values[
                :min(
                    4,
                    len(
                        condition_values
                    )
                )
            ],
            key = "descriptive_conditions"
        )

        frame = filter_records(
            frame,
            {
                "condition_protocol_identity": selected_conditions
            }
        )

        frame = convert_numeric_columns(
            frame,
            [
                "horizon",
                "mean_outcome",
                "standard_deviation_outcome"
            ]
        )

        st.line_chart(
            frame,
            x = "horizon",
            y = "mean_outcome",
            color = "condition_protocol_identity"
        )

        st.dataframe(
            frame[
                [
                    "condition_protocol_identity",
                    "horizon",
                    "independent_cell_count",
                    "observation_count",
                    "mean_outcome",
                    "standard_deviation_outcome",
                    "estimate_label"
                ]
            ],
            hide_index = True,
            width = "stretch"
        )

        return

    trajectories = store.table(
        "bayesian_modeling",
        "continuous_trajectory_summaries"
    )
    candidates = store.table(
        "bayesian_modeling",
        "candidate_registry"
    )

    trajectories = trajectories.merge(
        candidates[
            [
                "candidate_identity",
                "design_scope",
                "aging_mode",
                "sampling_design_scope"
            ]
        ],
        on = "candidate_identity",
        how = "left",
        validate = "many_to_one"
    )

    for column_name, label in [
        ("design_scope", "Dataset stage"),
        ("aging_mode", "Aging mode"),
        ("outcome", "Outcome"),
        ("cohort", "Cohort"),
        ("horizon_axis", "Horizon axis"),
        ("analytical_status", "Analytical status"),
        ("trajectory_target", "Trajectory target")
    ]:
        selected_value = categorical_selection(
            trajectories,
            column_name,
            label,
            f"bayesian_{column_name}"
        )

        trajectories = filter_records(
            trajectories,
            {
                column_name: [
                    selected_value
                ]
            }
        )

    condition_values = sorted(
        trajectories[
            "condition_protocol_identity"
        ].unique()
    )

    selected_conditions = st.multiselect(
        "Condition protocols",
        options = condition_values,
        default = condition_values[
            :min(
                4,
                len(
                    condition_values
                )
            )
        ],
        key = "bayesian_conditions"
    )

    trajectories = filter_records(
        trajectories,
        {
            "condition_protocol_identity": selected_conditions
        }
    )

    trajectories = convert_numeric_columns(
        trajectories,
        [
            "horizon",
            "posterior_median",
            "credible_interval_lower",
            "credible_interval_upper",
            "predictive_interval_lower",
            "predictive_interval_upper"
        ]
    )

    st.line_chart(
        trajectories,
        x = "horizon",
        y = "posterior_median",
        color = "condition_protocol_identity"
    )

    st.dataframe(
        trajectories[
            [
                "condition_protocol_identity",
                "horizon",
                "posterior_median",
                "credible_interval_lower",
                "credible_interval_upper",
                "predictive_interval_lower",
                "predictive_interval_upper",
                "analytical_status",
                "estimate_label",
                "causal_interpretation"
            ]
        ],
        hide_index = True,
        width = "stretch"
    )

    st.subheader(
        "Bayesian held-out-cell validation"
    )

    candidate_identities = sorted(
        set(
            trajectories[
                "candidate_identity"
            ]
        )
    )

    if not candidate_identities:
        st.info(
            "No Bayesian validation records match the selected trajectory context."
        )

        return

    validation = store.table(
        "bayesian_modeling",
        "grouped_validation_metrics"
    )

    validation = filter_records(
        validation,
        {
            "candidate_identity": candidate_identities
        }
    )

    validation = convert_numeric_columns(
        validation,
        [
            "equal_cell_weighted_mean_absolute_error",
            "equal_cell_weighted_root_mean_squared_error",
            "equal_cell_weighted_median_error",
            "posterior_predictive_interval_coverage",
            "mean_posterior_predictive_interval_width",
            "equal_cell_weighted_predictive_r_squared"
        ]
    )

    st.dataframe(
        validation[
            [
                "candidate_identity",
                "outcome",
                "cohort",
                "fold_count",
                "test_cell_count",
                "test_observation_count",
                "equal_cell_weighted_mean_absolute_error",
                "equal_cell_weighted_root_mean_squared_error",
                "equal_cell_weighted_predictive_r_squared",
                "posterior_predictive_interval_coverage",
                "validation_status",
                "validation_reason_code",
                "estimate_label"
            ]
        ],
        hide_index = True,
        width = "stretch"
    )


# Rendering grouped predictive evidence without selecting a universal winner
def render_predictive_evidence(store: ArtifactStore) -> None:
    st.header(
        "Predictive evidence"
    )

    metrics = store.table(
        "predictive_modeling",
        "validation_metrics"
    )
    candidates = store.table(
        "predictive_modeling",
        "candidate_registry"
    )

    metrics = metrics.merge(
        candidates[
            [
                "benchmark_identity",
                "design_scope",
                "aging_mode",
                "outcome",
                "cohort",
                "horizon_axis"
            ]
        ],
        on = "benchmark_identity",
        how = "left",
        validate = "one_to_one"
    )

    filters = {}

    for column_name, label in [
        ("design_scope", "Dataset stage"),
        ("aging_mode", "Aging mode"),
        ("outcome", "Outcome"),
        ("cohort", "Cohort"),
        ("horizon_axis", "Horizon axis"),
        ("model_family", "Model family"),
        ("analytical_status", "Analytical status")
    ]:
        default = None

        if column_name == "analytical_status":
            default = [
                "available"
            ]

        filters[
            column_name
        ] = categorical_filter(
            metrics,
            column_name,
            label,
            f"predictive_{column_name}",
            default
        )

    metrics = filter_records(
        metrics,
        filters
    )

    metrics = convert_numeric_columns(
        metrics,
        [
            "equal_cell_weighted_mean_absolute_error",
            "equal_cell_weighted_root_mean_squared_error",
            "equal_cell_weighted_predictive_r_squared",
            "prediction_interval_coverage",
            "mean_prediction_interval_width"
        ]
    )

    st.dataframe(
        metrics[
            [
                "benchmark_identity",
                "design_scope",
                "aging_mode",
                "outcome",
                "cohort",
                "horizon_axis",
                "model_family",
                "test_cell_count",
                "test_observation_count",
                "equal_cell_weighted_mean_absolute_error",
                "equal_cell_weighted_root_mean_squared_error",
                "equal_cell_weighted_predictive_r_squared",
                "prediction_interval_coverage",
                "analytical_status",
                "analytical_reason_code",
                "estimate_label"
            ]
        ],
        hide_index = True,
        width = "stretch"
    )

    st.caption(
        "Metrics describe held-out-cell prediction."
        " They do not establish causal importance or select a universally preferred model."
    )

    st.subheader(
        "Predictive model inspection"
    )

    benchmark_identities = sorted(
        set(
            metrics[
                "benchmark_identity"
            ]
        )
    )

    if not benchmark_identities:
        st.info(
            "No model-inspection records match the selected predictive context."
        )

        return

    importance = store.table(
        "predictive_modeling",
        "feature_importance"
    )

    importance = filter_records(
        importance,
        {
            "benchmark_identity": benchmark_identities
        }
    )

    importance_statuses = categorical_filter(
        importance,
        "importance_status",
        "Importance status",
        "predictive_importance_status"
    )

    importance = filter_records(
        importance,
        {
            "importance_status": importance_statuses
        }
    )

    importance = convert_numeric_columns(
        importance,
        [
            "importance_value"
        ]
    )

    st.warning(
        "Feature-importance values describe fitted predictive behavior. They do not identify causal effects or treatment importance."
    )

    st.dataframe(
        importance[
            [
                "benchmark_identity",
                "model_family",
                "configuration_identity",
                "encoded_feature",
                "feature_group",
                "importance_value",
                "importance_method",
                "importance_status",
                "importance_reason_code",
                "estimate_label"
            ]
        ],
        hide_index = True,
        width = "stretch"
    )


# Rendering restricted robustness evidence and unresolved limitations
def render_robustness(store: ArtifactStore) -> None:
    st.header(
        "Robustness and limitations"
    )

    coverage = store.table(
        "robustness_assessment",
        "assessment_coverage"
    )

    statuses = categorical_filter(
        coverage,
        "assessment_status",
        "Assessment status",
        "robustness_status"
    )

    coverage = filter_records(
        coverage,
        {
            "assessment_status": statuses
        }
    )

    st.subheader(
        "Assessment coverage"
    )
    st.dataframe(
        coverage,
        hide_index = True,
        width = "stretch"
    )

    st.subheader(
        "Predictive sensitivity comparisons"
    )

    sensitivity = store.table(
        "predictive_sensitivity",
        "sensitivity_comparisons"
    )

    sensitivity_filters = {}

    for column_name, label in [
        ("variant_type", "Sensitivity variant"),
        ("model_family", "Model family"),
        ("aging_mode", "Aging mode"),
        ("outcome", "Outcome"),
        ("cohort", "Cohort"),
        ("comparison_status", "Comparison status")
    ]:
        sensitivity_filters[
            column_name
        ] = categorical_filter(
            sensitivity,
            column_name,
            label,
            f"sensitivity_{column_name}"
        )

    sensitivity = filter_records(
        sensitivity,
        sensitivity_filters
    )

    sensitivity = convert_numeric_columns(
        sensitivity,
        [
            "shared_observation_count",
            "shared_independent_cell_count",
            "root_mean_squared_error_difference",
            "mean_absolute_error_difference",
            "predictive_r_squared_difference"
        ]
    )

    st.dataframe(
        sensitivity[
            [
                "variant_identity",
                "variant_type",
                "model_family",
                "design_scope",
                "aging_mode",
                "outcome",
                "cohort",
                "horizon_axis",
                "shared_observation_count",
                "shared_independent_cell_count",
                "root_mean_squared_error_difference",
                "mean_absolute_error_difference",
                "predictive_r_squared_difference",
                "exact_inventory_match",
                "variant_analytical_status",
                "comparison_status",
                "comparison_reason_code",
                "estimate_label"
            ]
        ],
        hide_index = True,
        width = "stretch"
    )

    st.subheader(
        "Structural limitations"
    )
    st.dataframe(
        store.table(
            "robustness_assessment",
            "structural_limitations"
        ),
        hide_index = True,
        width = "stretch"
    )

    st.subheader(
        "External-validity boundaries"
    )
    st.dataframe(
        store.table(
            "robustness_assessment",
            "external_validity"
        ),
        hide_index = True,
        width = "stretch"
    )

    fragility = store.table(
        "robustness_assessment",
        "fragility_registry"
    )

    fragility[
        "fragility_indicator_count"
    ] = pd.to_numeric(
        fragility[
            "fragility_indicator_count"
        ],
        errors = "coerce"
    )

    fragility_summary = (
        fragility.groupby(
            [
                "source_analysis",
                "analytical_status"
            ],
            dropna = False
        )
        .agg(
            record_count = (
                "fragility_scope_identity",
                "count"
            ),
            median_fragility_indicator_count = (
                "fragility_indicator_count",
                "median"
            ),
            maximum_fragility_indicator_count = (
                "fragility_indicator_count",
                "max"
            )
        )
        .reset_index()
    )

    st.subheader(
        "Fragility inventory"
    )
    st.dataframe(
        fragility_summary,
        hide_index = True,
        width = "stretch"
    )


# Rendering immutable source provenance
def render_provenance(store: ArtifactStore, validation_summary: dict) -> None:
    st.header(
        "Provenance"
    )

    columns = st.columns(
        3
    )

    columns[0].metric(
        "Validated reports",
        validation_summary[
            "source_count"
        ]
    )
    columns[1].metric(
        "Validated tables",
        validation_summary[
            "table_count"
        ]
    )
    columns[2].metric(
        "Policy version",
        store.policy[
            "policy_version"
        ]
    )

    st.code(
        validation_summary[
            "policy_sha256"
        ],
        language = "text"
    )

    st.dataframe(
        pd.DataFrame(
            store.provenance_records()
        ),
        hide_index = True,
        width = "stretch"
    )


# Running the read-only evidence application
def main() -> None:
    st.set_page_config(
        page_title = "Battery Aging Evidence Explorer",
        page_icon = "🔋",
        layout = "wide"
    )

    artifact_root = os.environ.get(
        "EVIDENCE_ARTIFACT_ROOT",
        str(
            DEFAULT_ARTIFACT_ROOT
        )
    )
    policy_path = os.environ.get(
        "EVIDENCE_DASHBOARD_POLICY",
        str(
            DEFAULT_POLICY_PATH
        )
    )

    try:
        store, validation_summary = load_store(
            artifact_root,
            policy_path
        )

    except (
        ArtifactValidationError,
        json.JSONDecodeError,
        OSError,
        ValueError
    ) as exc:
        LOGGER.exception(
            "Validated evidence could not be loaded.",
            exc_info = exc
        )

        st.error(
            "Validated evidence could not be loaded. The application has stopped without presenting unverified results."
        )

        st.stop()

    st.title(
        store.policy[
            "presentation_contract"
        ][
            "application_title"
        ]
    )

    page = st.sidebar.radio(
        "View",
        options = store.policy[
            "presentation_contract"
        ][
            "required_navigation"
        ]
    )

    if page == "Evidence boundaries":
        render_boundaries(
            store
        )

    elif page == "Experimental support":
        render_experimental_support(
            store
        )

    elif page == "Trajectories":
        render_trajectories(
            store
        )

    elif page == "Predictive evidence":
        render_predictive_evidence(
            store
        )

    elif page == "Robustness":
        render_robustness(
            store
        )

    elif page == "Provenance":
        render_provenance(
            store,
            validation_summary
        )


if __name__ == "__main__":
    main()
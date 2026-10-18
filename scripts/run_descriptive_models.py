import argparse
import csv
import hashlib
import json
import math
import os
import warnings
import shutil
import tempfile
import platform

from importlib.metadata import version as distribution_version
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path


REQUIRED_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1"
}

for (
    environment_name,
    environment_value
) in REQUIRED_THREAD_ENVIRONMENT.items():
    os.environ[
        environment_name
    ] = environment_value


import numpy as np

from scipy.stats import t as student_t
from statsmodels.regression.mixed_linear_model import MixedLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FEATURE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_table_manifest.json"
)

DEFAULT_DESIGN_REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "experimental_design"
    / "experimental_design_report.json"
)

DEFAULT_MODELING_POLICY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "descriptive_modeling_policy.json"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "descriptive_models"
)

REQUIRED_POLICY_SECTIONS = {
    "policy_version",
    "effective_date",
    "previous_policy_sha256",
    "amendment",
    "analysis_scope",
    "outcomes",
    "horizon_axes",
    "candidate_templates",
    "candidate_assembly",
    "candidate_generation",
    "observation_assembly",
    "observation_accounting",
    "baseline_adjustment",
    "encoding",
    "restricted_cubic_spline",
    "horizon_interactions",
    "admissibility",
    "estimation",
    "uncertainty",
    "validation",
    "validation_metrics",
    "prediction_intervals",
    "contrast_evaluation",
    "trajectory_summaries",
    "diagnostics",
    "influence",
    "condition_calibration",
    "model_comparison",
    "missing_data",
    "multiplicity",
    "candidate_dispositions",
    "candidate_reconciliation",
    "reporting",
    "artifact_contract",
    "reproducibility"
}

REQUIRED_DESIGN_TABLES = {
    "analytical_scope",
    "bracketing_support_membership",
    "bracketing_support_regions",
    "estimand_support"
}

REQUIRED_FEATURE_TABLES = {
    "cell_features",
    "condition_features",
    "reference_test_features"
}

REQUIRED_OUTPUT_TABLES = {
    "candidate_registry",
    "candidate_terms",
    "observation_accounting",
    "trajectory_summaries",
    "model_coefficients",
    "model_diagnostics",
    "validation_folds",
    "validation_predictions",
    "validation_metrics",
    "model_comparisons",
    "condition_calibration",
    "influence_diagnostics",
    "model_contrasts"
}

OUTPUT_TABLE_CONTRACTS = {
    "candidate_registry": {
        "field_names": [
            "candidate_identity",
            "support_region_identity",
            "design_scope",
            "sampling_design_scope",
            "aging_mode",
            "outcome",
            "cohort",
            "horizon_axis",
            "specification",
            "context_family",
            "template_name",
            "support_region_kind",
            "support_interval_minimum",
            "support_interval_maximum",
            "support_interval_lower_inclusive",
            "support_interval_upper_inclusive",
            "analysis_role",
            "analysis_population",
            "support_condition_count",
            "support_member_cell_count",
            "assembled_model_observation_count",
            "assembled_model_condition_count",
            "assembled_model_cell_count",
            "inside_evaluation_interval_observation_count",
            "outside_evaluation_interval_observation_count",
            "repeated_cell_count",
            "single_observation_cell_count",
            "minimum_condition_unique_horizon_count",
            "minimum_condition_repeated_cell_count",
            "longitudinal_structure_status",
            "analytical_observation_count",
            "permitted_adjustment_variables",
            "target_estimand_available",
            "target_estimand_status",
            "exposure_time_interpretation",
            "required_validation_grouping",
            "stage_pooling_status",
            "causal_identification_status",
            "question_classification",
            "authorization_status",
            "estimator",
            "horizon_basis",
            "include_horizon_by_treatment_interactions",
            "cell_random_intercept",
            "authorized_term_count",
            "interaction_hierarchy_status",
            "interaction_hierarchy_reason_code",
            "admissibility_status",
            "admissibility_reason_code",
            "observation_assembly_status",
            "candidate_source_record_count",
            "candidate_included_record_count",
            "candidate_excluded_record_count",
            "coverage_anchor_support_region_identity",
            "duration_anchor_support_region_identity",
            "model_matrix_status",
            "model_matrix_error",
            "model_matrix_column_count",
            "model_matrix_rank",
            "effective_condition_number",
            "residual_independent_cell_degrees_of_freedom",
            "fit_identity",
            "model_matrix_column_names",
            "model_encoding",
            "fit_status",
            "fit_reason_code",
            "fit_error",
            "canonical_fit_candidate_identity",
            "fitted_coefficient_count",
            "uncertainty_status",
            "uncertainty_reason_code",
            "validation_status",
            "validation_reason_code",
            "validation_interval_status",
            "influence_status",
            "influence_reason_code",
            "selection_status",
            "selection_reason_code",
            "final_disposition",
            "final_reason_code",
            "numerical_fit_status",
            "validation_execution_status",
            "predictive_benchmark_status",
            "prediction_domain_status",
            "prediction_interval_assessment_status",
            "condition_calibration_assessment_status",
            "model_adequacy_status",
            "model_adequacy_reason_code",
            "analytical_result_status",
            "analytical_result_reason_code"
        ],
        "primary_key": [
            "candidate_identity"
        ]
    },
    "candidate_terms": {
        "field_names": [
            "candidate_identity",
            "support_region_identity",
            "template_name",
            "term_name",
            "term_type",
            "term_source",
            "source_question_classification",
            "source_authorization_status",
            "term_disposition",
            "term_reason_code"
        ],
        "primary_key": [
            "candidate_identity",
            "term_type",
            "term_name"
        ]
    },
    "observation_accounting": {
        "field_names": [
            "record_type",
            "candidate_identity",
            "support_region_identity",
            "source_record_count",
            "included_record_count",
            "excluded_record_count",
            "missing_horizon_count",
            "nonpositive_horizon_count",
            "missing_outcome_count",
            "cohort_ineligible_count",
            "incomplete_cumulative_fec_count",
            "inside_evaluation_interval_count",
            "outside_evaluation_interval_count",
            "accounting_scope",
            "aging_mode",
            "outcome",
            "cohort",
            "horizon_axis",
            "member_cell_count",
            "eligible_positive_record_count",
            "outside_support_record_count"
        ],
        "primary_key": [
            "record_type",
            "candidate_identity",
            "aging_mode",
            "outcome",
            "cohort",
            "horizon_axis"
        ]
    },
    "trajectory_summaries": {
        "field_names": [
            "record_type",
            "aging_mode",
            "dataset_stage",
            "condition_protocol_identity",
            "cell_identity",
            "outcome",
            "cohort",
            "horizon_axis",
            "horizon",
            "independent_cell_count",
            "observation_count",
            "minimum_horizon",
            "maximum_horizon",
            "first_observed_outcome",
            "last_observed_outcome",
            "mean_outcome",
            "standard_deviation_outcome",
            "minimum_outcome",
            "maximum_outcome",
            "estimate_label"
        ],
        "primary_key": [
            "record_type",
            "aging_mode",
            "dataset_stage",
            "condition_protocol_identity",
            "outcome",
            "cohort",
            "horizon_axis",
            "cell_identity",
            "horizon"
        ]
    },
    "model_coefficients": {
        "field_names": [
            "candidate_identity",
            "fit_identity",
            "support_region_identity",
            "template_name",
            "estimator",
            "column_index",
            "column_name",
            "term_type",
            "source_term_name",
            "horizon_basis_name",
            "estimate",
            "uncertainty_status",
            "uncertainty_reason_code",
            "confidence_level",
            "degrees_of_freedom",
            "standard_error",
            "confidence_interval_lower",
            "confidence_interval_upper",
            "estimate_label"
        ],
        "primary_key": [
            "candidate_identity",
            "column_index"
        ]
    },
    "model_diagnostics": {
        "field_names": [
            "record_type",
            "candidate_identity",
            "support_region_identity",
            "template_name",
            "fit_identity",
            "observation_count",
            "independent_cell_count",
            "condition_protocol_count",
            "matrix_column_count",
            "matrix_rank",
            "rank_tolerance",
            "largest_singular_value",
            "smallest_singular_value",
            "smallest_retained_singular_value",
            "effective_condition_number",
            "residual_independent_cell_degrees_of_freedom",
            "unique_horizon_count",
            "non_boundary_unique_horizon_count",
            "admissibility_status",
            "admissibility_reason_code",
            "estimator",
            "fit_status",
            "fit_reason_code",
            "fit_error",
            "convergence_status",
            "optimizer",
            "optimizer_attempts",
            "random_effect_singularity_status",
            "residual_mean",
            "residual_standard_deviation",
            "maximum_absolute_standardized_residual",
            "marginal_residual_mean",
            "marginal_residual_standard_deviation",
            "conditional_residual_mean",
            "conditional_residual_standard_deviation",
            "observation_variance",
            "random_intercept_variance",
            "residual_variance",
            "weighted_matrix_rank",
            "weighted_rank_tolerance",
            "weighted_condition_number"
        ],
        "primary_key": [
            "record_type",
            "candidate_identity"
        ]
    },
    "validation_folds": {
        "field_names": [
            "candidate_identity",
            "fit_identity",
            "support_region_identity",
            "template_name",
            "fold_identity",
            "fold_index",
            "training_cell_count",
            "test_cell_count",
            "training_cell_identities",
            "test_cell_identities",
            "training_observation_count",
            "raw_test_observation_count",
            "test_observation_count",
            "outside_support_test_observation_count",
            "fold_status",
            "fold_reason_code",
            "prediction_interval_status",
            "prediction_interval_reason_code"
        ],
        "primary_key": [
            "candidate_identity",
            "fold_index"
        ]
    },
    "validation_predictions": {
        "field_names": [
            "candidate_identity",
            "fit_identity",
            "support_region_identity",
            "template_name",
            "fold_identity",
            "fold_index",
            "source_identity",
            "cell_identity",
            "condition_protocol_identity",
            "horizon",
            "observed_outcome",
            "predicted_outcome",
            "prediction_error",
            "prediction_interval_status",
            "prediction_interval_reason_code",
            "prediction_interval_lower",
            "prediction_interval_upper",
            "estimate_label"
        ],
        "primary_key": [
            "candidate_identity",
            "source_identity"
        ]
    },
    "validation_metrics": {
        "field_names": [
            "candidate_identity",
            "fit_identity",
            "support_region_identity",
            "template_name",
            "validation_status",
            "validation_reason_code",
            "independent_cell_count",
            "observation_count",
            "mean_absolute_error",
            "root_mean_squared_error",
            "mean_error",
            "predictive_r_squared",
            "predictive_r_squared_status",
            "prediction_interval_coverage",
            "mean_prediction_interval_width",
            "interval_metric_status",
            "estimate_label"
        ],
        "primary_key": [
            "candidate_identity"
        ]
    },
    "model_comparisons": {
        "field_names": [
            "support_region_identity",
            "design_scope",
            "sampling_design_scope",
            "aging_mode",
            "outcome",
            "cohort",
            "horizon_axis",
            "specification",
            "context_family",
            "first_candidate_identity",
            "second_candidate_identity",
            "first_template_name",
            "second_template_name",
            "comparison_status",
            "comparison_reason_code",
            "shared_prediction_count",
            "first_mean_absolute_error",
            "second_mean_absolute_error",
            "mean_absolute_error_difference",
            "first_root_mean_squared_error",
            "second_root_mean_squared_error",
            "root_mean_squared_error_difference",
            "estimate_label"
        ],
        "primary_key": [
            "first_candidate_identity",
            "second_candidate_identity"
        ]
    },
    "condition_calibration": {
        "field_names": [
            "candidate_identity",
            "fit_identity",
            "support_region_identity",
            "template_name",
            "condition_protocol_identity",
            "calibration_status",
            "calibration_reason_code",
            "independent_cell_count",
            "observation_count",
            "mean_observed",
            "mean_predicted",
            "mean_error",
            "mean_absolute_error",
            "root_mean_squared_error",
            "estimate_label"
        ],
        "primary_key": [
            "candidate_identity",
            "condition_protocol_identity"
        ]
    },
    "influence_diagnostics": {
        "field_names": [
            "candidate_identity",
            "fit_identity",
            "support_region_identity",
            "template_name",
            "deleted_cell_identity",
            "column_index",
            "column_name",
            "fit_status",
            "fit_reason_code",
            "fit_error",
            "full_estimate",
            "deleted_estimate",
            "coefficient_change",
            "standardized_coefficient_change",
            "maximum_absolute_standardized_coefficient_change",
            "uncertainty_status",
            "estimate_label"
        ],
        "primary_key": [
            "candidate_identity",
            "deleted_cell_identity",
            "column_index"
        ]
    },
    "model_contrasts": {
        "field_names": [
            "candidate_identity",
            "fit_identity",
            "support_region_identity",
            "template_name",
            "estimator",
            "estimand_decision_identity",
            "decision_scope",
            "first_condition_protocol_identity",
            "second_condition_protocol_identity",
            "outcome",
            "cohort",
            "horizon_axis",
            "horizon",
            "evaluation_laboratory",
            "model_based_interpolation",
            "first_predicted_outcome",
            "second_predicted_outcome",
            "contrast_estimate",
            "contrast_direction",
            "contrast_status",
            "contrast_reason_code",
            "candidate_validation_status",
            "candidate_validation_reason_code",
            "candidate_analytical_result_status",
            "uncertainty_status",
            "uncertainty_reason_code",
            "confidence_level",
            "degrees_of_freedom",
            "standard_error",
            "confidence_interval_lower",
            "confidence_interval_upper",
            "final_support_classification",
            "restriction_reasons",
            "causal_interpretation",
            "estimate_label"
        ],
        "primary_key": [
            "candidate_identity",
            "estimand_decision_identity",
            "horizon"
        ]
    }
}

CANDIDATE_KEY_FIELDS = [
    "support_region_identity",
    "design_scope",
    "sampling_design_scope",
    "aging_mode",
    "outcome",
    "cohort",
    "horizon_axis",
    "specification",
    "context_family"
]

CANDIDATE_SHARED_FIELDS = [
    "support_region_kind",
    "support_interval_minimum",
    "support_interval_maximum",
    "support_interval_lower_inclusive",
    "support_interval_upper_inclusive",
    "analysis_role",
    "analysis_population",
    "condition_protocol_count",
    "independent_cell_count",
    "analytical_observation_count",
    "permitted_adjustment_variables",
    "target_estimand_available",
    "target_estimand_status",
    "exposure_time_interpretation",
    "required_validation_grouping",
    "stage_pooling_status",
    "causal_identification_status"
]

REQUIRED_SCOPE_COLUMNS = {
    *CANDIDATE_KEY_FIELDS,
    *CANDIDATE_SHARED_FIELDS,
    "question_classification",
    "authorization_status",
    "term_name",
    "term_type",
    "structurally_estimable"
}

OBSERVATION_ACCOUNTING_STATUSES = [
    "missing_horizon",
    "nonpositive_horizon",
    "missing_outcome",
    "cohort_ineligible",
    "incomplete_cumulative_fec",
    "included"
]

CELL_MODEL_FIELDS = [
    "ambient_temperature_celsius",
    "maximum_state_of_charge_fraction",
    "depth_of_discharge_fraction",
    "charge_rate_c",
    "discharge_rate_c",
    "reference_test_schedule",
    "laboratory",
    "sampling_method",
    "dataset_stage",
    "aging_mode",
    "condition_protocol_identity"
]

NUMERICAL_MODEL_FIELDS = {
    "ambient_temperature_celsius",
    "maximum_state_of_charge_fraction",
    "depth_of_discharge_fraction",
    "charge_rate_c",
    "discharge_rate_c"
}

IMPLEMENTED_CANDIDATE_TEMPLATES = [
    {
        "template_name": "linear_common_trajectory",
        "estimator": "weighted_least_squares",
        "horizon_basis": "linear",
        "include_authorized_terms": True,
        "include_horizon_by_treatment_interactions": False,
        "cell_random_intercept": False
    },
    {
        "template_name": "linear_differential_trajectory",
        "estimator": "weighted_least_squares",
        "horizon_basis": "linear",
        "include_authorized_terms": True,
        "include_horizon_by_treatment_interactions": True,
        "cell_random_intercept": False
    },
    {
        "template_name": (
            "restricted_cubic_spline_common_trajectory"
        ),
        "estimator": "weighted_least_squares",
        "horizon_basis": "restricted_cubic_spline",
        "include_authorized_terms": True,
        "include_horizon_by_treatment_interactions": False,
        "cell_random_intercept": False
    },
    {
        "template_name": (
            "restricted_cubic_spline_differential_trajectory"
        ),
        "estimator": "weighted_least_squares",
        "horizon_basis": "restricted_cubic_spline",
        "include_authorized_terms": True,
        "include_horizon_by_treatment_interactions": True,
        "cell_random_intercept": False
    },
    {
        "template_name": "mixed_linear_common_trajectory",
        "estimator": "linear_mixed_effects",
        "horizon_basis": "linear",
        "include_authorized_terms": True,
        "include_horizon_by_treatment_interactions": False,
        "cell_random_intercept": True
    },
    {
        "template_name": (
            "mixed_linear_differential_trajectory"
        ),
        "estimator": "linear_mixed_effects",
        "horizon_basis": "linear",
        "include_authorized_terms": True,
        "include_horizon_by_treatment_interactions": True,
        "cell_random_intercept": True
    }
]

IMPLEMENTED_POLICY_VALUES = [
    (
        (
            "policy_version",
        ),
        "1.16"
    ),
    (
        (
            "analysis_scope",
            "independent_unit"
        ),
        "cell_identity"
    ),
    (
        (
            "analysis_scope",
            "aging_modes_fitted_separately"
        ),
        True
    ),
    (
        (
            "analysis_scope",
            "dataset_stages_pooled"
        ),
        False
    ),
    (
        (
            "analysis_scope",
            "cohorts_fitted_separately"
        ),
        True
    ),
    (
        (
            "analysis_scope",
            "horizon_axes_fitted_separately"
        ),
        True
    ),
    (
        (
            "analysis_scope",
            "observed_condition_combinations_only"
        ),
        True
    ),
    (
        (
            "analysis_scope",
            "unobserved_treatment_combinations_permitted"
        ),
        False
    ),
    (
        (
            "candidate_assembly",
            "required_terms_may_be_removed"
        ),
        False
    ),
    (
        (
            "candidate_assembly",
            "unauthorized_terms_may_be_added"
        ),
        False
    ),
    (
        (
            "candidate_assembly",
            "duplicate_observations_within_candidate_permitted"
        ),
        False
    ),
    (
        (
            "candidate_assembly",
            "source_term_sort_order"
        ),
        [
            "main_effect",
            "target_treatment",
            "pairwise_interaction"
        ]
    ),
    (
        (
            "candidate_assembly",
            "candidate_key_fields"
        ),
        CANDIDATE_KEY_FIELDS
    ),
    (
        (
            "candidate_generation",
            "candidate_identity_fields"
        ),
        [
            *CANDIDATE_KEY_FIELDS,
            "template_name"
        ]
    ),
    (
        (
            "trajectory_summaries",
            "cell_summary_fields"
        ),
        [
            "observation_count",
            "minimum_horizon",
            "maximum_horizon",
            "first_observed_outcome",
            "last_observed_outcome",
            "minimum_outcome",
            "maximum_outcome"
        ]
    ),
    (
        (
            "candidate_assembly",
            "source_term_authorization",
            "classification_fields"
        ),
        [
            "question_classification",
            "authorization_status"
        ]
    ),
    (
        (
            "candidate_assembly",
            "source_term_authorization",
            "question_classification_precedence"
        ),
        [
            "supported_with_restrictions",
            "sensitivity_only"
        ]
    ),
    (
        (
            "candidate_assembly",
            "source_term_authorization",
            "authorization_by_question_classification"
        ),
        {
            "supported_with_restrictions": (
                "permitted_for_model_evaluation_with_restrictions"
            ),
            "sensitivity_only": (
                "permitted_for_model_evaluation_sensitivity_only"
            )
        }
    ),
    (
        (
            "candidate_assembly",
            "source_term_authorization",
            "candidate_aggregation"
        ),
        (
            "most_restrictive_selected_source_term_"
            "classification"
        )
    ),
    (
        (
            "candidate_assembly",
            "source_term_authorization",
            "source_term_classification_preserved"
        ),
        True
    ),
    (
        (
            "candidate_assembly",
            "source_term_authorization",
            "derived_terms_inherit_source_classification"
        ),
        False
    ),
    (
        (
            "candidate_generation",
            "support_anchor_selection",
            "duplicate_anchor_resolution"
        ),
        "retain_once"
    ),
    (
        (
            "candidate_generation",
            "support_anchor_selection",
            "outcome_values_used"
        ),
        False
    ),
    (
        (
            "observation_assembly",
            "membership_join_key"
        ),
        [
            "support_region_identity",
            "cell_identity"
        ]
    ),
    (
        (
            "observation_assembly",
            "observation_identity_field"
        ),
        "source_identity"
    ),
    (
        (
            "observation_assembly",
            "baseline_horizon_included"
        ),
        False
    ),
    (
        (
            "observation_assembly",
            "missing_outcomes_included"
        ),
        False
    ),
    (
        (
            "observation_assembly",
            "ineligible_observations_included"
        ),
        False
    ),
    (
        (
            "observation_assembly",
            "fec_incomplete_observations_included"
        ),
        False
    ),
    (
        (
            "observation_assembly",
            "predictions_outside_evaluation_interval"
        ),
        False
    ),
    (
        (
            "observation_assembly",
            "extrapolation_permitted"
        ),
        False
    ),
    (
        (
            "observation_accounting",
            "statuses_are_mutually_exclusive"
        ),
        True
    ),
    (
        (
            "observation_accounting",
            "status_evaluation_order"
        ),
        OBSERVATION_ACCOUNTING_STATUSES
    ),
    (
        (
            "baseline_adjustment",
            "baseline_characteristics_used_as_model_covariates"
        ),
        False
    ),
    (
        (
            "baseline_adjustment",
            "exception_permitted"
        ),
        False
    ),
    (
        (
            "encoding",
            "intercept_included"
        ),
        True
    ),
    (
        (
            "encoding",
            "numerical_treatments"
        ),
        "centered_and_scaled"
    ),
    (
        (
            "encoding",
            "horizon"
        ),
        "centered_and_scaled"
    ),
    (
        (
            "encoding",
            "centering_population"
        ),
        "unique_cells_with_equal_cell_weight"
    ),
    (
        (
            "encoding",
            "scaling_standard_deviation"
        ),
        "population_standard_deviation"
    ),
    (
        (
            "encoding",
            "categorical_encoding"
        ),
        "reference_coded"
    ),
    (
        (
            "encoding",
            "categorical_reference_level"
        ),
        "lexicographically_first_observed_level"
    ),
    (
        (
            "encoding",
            "hierarchy_required"
        ),
        True
    ),
    (
        (
            "encoding",
            "novel_category_prediction_permitted"
        ),
        False
    ),
    (
        (
            "restricted_cubic_spline",
            "knot_count"
        ),
        3
    ),
    (
        (
            "restricted_cubic_spline",
            "knot_probabilities"
        ),
        [
            0.0,
            0.5,
            1.0
        ]
    ),
    (
        (
            "restricted_cubic_spline",
            "quantile_weighting"
        ),
        "equal_total_weight_per_cell"
    ),
    (
        (
            "restricted_cubic_spline",
            "basis"
        ),
        "three_knot_natural_restricted_cubic"
    ),
    (
        (
            "restricted_cubic_spline",
            "basis_columns"
        ),
        [
            "standardized_horizon",
            "restricted_cubic_nonlinear_component"
        ]
    ),
    (
        (
            "restricted_cubic_spline",
            "strictly_increasing_knots_required"
        ),
        True
    ),
    (
        (
            "restricted_cubic_spline",
            "minimum_unique_horizon_count"
        ),
        4
    ),
    (
        (
            "restricted_cubic_spline",
            "minimum_non_boundary_unique_horizon_count"
        ),
        2
    ),
    (
        (
            "restricted_cubic_spline",
            "prediction_outside_training_boundary_knots"
        ),
        False
    ),
    (
        (
            "horizon_interactions",
            "context_terms_interacted_with_horizon"
        ),
        False
    ),
    (
        (
            "horizon_interactions",
            "all_eligible_authorized_terms_required"
        ),
        True
    ),
    (
        (
            "horizon_interactions",
            "partial_interaction_removal_permitted"
        ),
        False
    ),
    (
        (
            "admissibility",
            "minimum_independent_cells"
        ),
        6
    ),
    (
        (
            "admissibility",
            "minimum_residual_independent_cell_degrees_of_freedom"
        ),
        4
    ),
    (
        (
            "admissibility",
            "full_column_rank_required"
        ),
        True
    ),
    (
        (
            "admissibility",
            "rank_method"
        ),
        "singular_value_decomposition"
    ),
    (
        (
            "admissibility",
            "admissibility_order"
        ),
        [
            "assembly_complete",
            "complete_interaction_hierarchy",
            "spline_requirements",
            "longitudinal_structure_requirements",
            "minimum_condition_protocol_count",
            "minimum_independent_cells",
            "outcome_variation",
            "finite_values",
            "positive_weights",
            "full_column_rank",
            "residual_independent_cell_degrees_of_freedom",
            "effective_condition_number"
        ]
    ),
    (
        (
            "admissibility",
            "maximum_effective_condition_number"
        ),
        67108864.0
    ),
    (
        (
            "admissibility",
            "required_condition_protocol_count"
        ),
        2
    ),
    (
        (
            "admissibility",
            "required_outcome_variation"
        ),
        True
    ),
    (
        (
            "admissibility",
            "term_removal_to_obtain_admissibility_permitted"
        ),
        False
    ),
    (
        (
            "estimation",
            "weighted_least_squares",
            "observation_weight"
        ),
        "inverse_cell_observation_count"
    ),
    (
        (
            "estimation",
            "weighted_least_squares",
            "weight_normalization"
        ),
        "mean_observation_weight_equals_one"
    ),
    (
        (
            "estimation",
            "weighted_least_squares",
            "coefficient_solver"
        ),
        "singular_value_decomposition"
    ),
    (
        (
            "estimation",
            "weighted_least_squares",
            "regularization"
        ),
        "none"
    ),
    (
        (
            "estimation",
            "linear_mixed_effects",
            "grouping_field"
        ),
        "cell_identity"
    ),
    (
        (
            "estimation",
            "linear_mixed_effects",
            "random_effects"
        ),
        [
            "intercept"
        ]
    ),
    (
        (
            "estimation",
            "linear_mixed_effects",
            "random_slope"
        ),
        False
    ),
    (
        (
            "estimation",
            "linear_mixed_effects",
            "observation_weights_used"
        ),
        False
    ),
    (
        (
            "estimation",
            "linear_mixed_effects",
            "estimation_method"
        ),
        "restricted_maximum_likelihood"
    ),
    (
        (
            "estimation",
            "linear_mixed_effects",
            "optimizer_sequence"
        ),
        [
            "lbfgs",
            "bfgs"
        ]
    ),
    (
        (
            "estimation",
            "linear_mixed_effects",
            "maximum_iterations"
        ),
        1000
    ),
    (
        (
            "estimation",
            "linear_mixed_effects",
            "convergence_required"
        ),
        True
    ),
    (
        (
            "estimation",
            "linear_mixed_effects",
            "optimizer_convergence_source"
        ),
        "candidate_result_converged"
    ),
    (
        (
            "estimation",
            "linear_mixed_effects",
            "warning_only_failure_permitted"
        ),
        False
    ),
    (
        (
            "estimation",
            "linear_mixed_effects",
            "singular_random_effect_icc_tolerance"
        ),
        1.4901161193847656e-08
    ),
    (
        (
            "uncertainty",
            "confidence_level"
        ),
        0.95
    ),
    (
        (
            "uncertainty",
            "coefficient_covariance"
        ),
        "delete_one_cell_jackknife"
    ),
    (
        (
            "uncertainty",
            "contrast_covariance"
        ),
        "delete_one_cell_jackknife"
    ),
    (
        (
            "uncertainty",
            "jackknife_unit"
        ),
        "cell_identity"
    ),
    (
        (
            "uncertainty",
            "all_deletion_refits_required"
        ),
        True
    ),
    (
        (
            "uncertainty",
            "reference_distribution"
        ),
        "student_t"
    ),
    (
        (
            "validation",
            "grouping_field"
        ),
        "cell_identity"
    ),
    (
        (
            "validation",
            "cell_leakage_permitted"
        ),
        False
    ),
    (
        (
            "validation",
            "maximum_fold_count"
        ),
        3
    ),
    (
        (
            "validation",
            "minimum_fold_count"
        ),
        2
    ),
    (
        (
            "validation",
            "condition_protocol_representation_required_in_every_training_fold"
        ),
        True
    ),
    (
        (
            "validation",
            "condition_protocol_representation_required_in_every_test_fold"
        ),
        True
    ),
    (
        (
            "validation",
            "randomized_assignment"
        ),
        False
    ),
    (
        (
            "validation",
            "fold_assignment"
        ),
        "condition_stratified_round_robin"
    ),
    (
        (
            "validation",
            "training_fold_admissibility_required"
        ),
        True
    ),
    (
        (
            "validation",
            "held_out_prediction_random_effect"
        ),
        "zero"
    ),
    (
        (
            "validation",
            "partially_valid_fold_permitted"
        ),
        False
    ),
    (
        (
            "validation",
            "test_fold_information_used_during_training"
        ),
        False
    ),
    (
        (
            "validation",
            "outcome_dependent_model_selection"
        ),
        False
    ),
    (
        (
            "validation",
            "performance_selected_winner_permitted"
        ),
        False
    ),
    (
        (
            "validation",
            "training_fold_uncertainty"
        ),
        "estimator_specific_fixed_effect_covariance"
    ),
    (
        (
            "validation",
            "training_fold_fixed_effect_covariance",
            "weighted_least_squares",
            "method"
        ),
        "delete_one_training_cell_jackknife"
    ),
    (
        (
            "validation",
            "training_fold_fixed_effect_covariance",
            "weighted_least_squares",
            "deletion_unit"
        ),
        "cell_identity"
    ),
    (
        (
            "validation",
            "training_fold_fixed_effect_covariance",
            "weighted_least_squares",
            "all_deletion_refits_required"
        ),
        True
    ),
    (
        (
            "validation",
            "training_fold_fixed_effect_covariance",
            "linear_mixed_effects",
            "method"
        ),
        "model_based_fixed_effect_covariance"
    ),
    (
        (
            "validation",
            "validation_prediction_interval_covariance"
        ),
        "training_fold_fixed_effect_covariance"
    ),
    (
        (
            "prediction_intervals",
            "validation_prediction_interval_covariance_source"
        ),
        (
            "weighted_least_squares_delete_one_training_cell_jackknife_or_linear_mixed_effects_model_based_"
            "fixed_effect_covariance"
        )
    ),
    (
        (
            "validation_metrics",
            "cell_weighting"
        ),
        "equal"
    ),
    (
        (
            "validation_metrics",
            "primary_metrics"
        ),
        [
            "mean_absolute_error",
            "root_mean_squared_error"
        ]
    ),
    (
        (
            "validation_metrics",
            "point_metrics_require_complete_out_of_fold_predictions"
        ),
        True
    ),
    (
        (
            "validation_metrics",
            "interval_metrics_require_complete_out_of_fold_intervals"
        ),
        True
    ),
    (
        (
            "prediction_intervals",
            "confidence_level"
        ),
        0.95
    ),
    (
        (
            "prediction_intervals",
            "target"
        ),
        "new_observation_from_a_new_cell"
    ),
    (
        (
            "prediction_intervals",
            "reference_distribution"
        ),
        "student_t"
    ),
    (
        (
            "prediction_intervals",
            "held_out_cell_random_effect_used"
        ),
        False
    ),
    (
        (
            "prediction_intervals",
            "distinguished_from_confidence_interval"
        ),
        True
    ),
    (
        (
            "contrast_evaluation",
            "authorization_required"
        ),
        True
    ),
    (
        (
            "contrast_evaluation",
            "point_selection_using_outcome_values_permitted"
        ),
        False
    ),
    (
        (
            "contrast_evaluation",
            "prediction_for_unobserved_condition_combination_permitted"
        ),
        False
    ),
    (
        (
            "contrast_evaluation",
            "contrast_direction"
        ),
        "second_condition_minus_first_condition"
    ),
    (
        (
            "contrast_evaluation",
            "causal_label_permitted"
        ),
        False
    ),
    (
        (
            "model_comparison",
            "outcome_dependent_candidate_selection"
        ),
        False
    ),
    (
        (
            "model_comparison",
            "performance_ranking_reported"
        ),
        False
    ),
    (
        (
            "model_comparison",
            "performance_selected_winner_reported"
        ),
        False
    ),
    (
        (
            "model_comparison",
            "all_admissible_candidates_retained"
        ),
        True
    ),
    (
        (
            "missing_data",
            "imputation_permitted"
        ),
        False
    ),
    (
        (
            "multiplicity",
            "null_hypothesis_tests_reported"
        ),
        False
    ),
    (
        (
            "multiplicity",
            "p_values_reported"
        ),
        False
    ),
    (
        (
            "multiplicity",
            "statistical_significance_labels_reported"
        ),
        False
    ),
    (
        (
            "candidate_dispositions",
        ),
        [
            "fitted",
            "inadmissible",
            "fit_failed",
            "not_assembled",
            "not_selected"
        ]
    ),
    (
        (
            "artifact_contract",
            "empty_table_permitted"
        ),
        True
    ),
    (
        (
            "artifact_contract",
            "empty_table_requires_schema"
        ),
        True
    ),
    (
        (
            "artifact_contract",
            "row_order"
        ),
        "lexicographical_by_declared_primary_key"
    ),
    (
        (
            "artifact_contract",
            "field_order"
        ),
        "declared_schema_order"
    ),
    (
        (
            "artifact_contract",
            "report_replaced_last"
        ),
        True
    ),
    (
        (
            "artifact_contract",
            "table_schemas_declared_in_implementation"
        ),
        True
    ),
    (
        (
            "artifact_contract",
            "table_primary_keys_declared_in_implementation"
        ),
        True
    ),
    (
        (
            "artifact_contract",
            "duplicate_primary_key_action"
        ),
        "fail_run"
    ),
    (
        (
            "artifact_contract",
            "inconsistent_record_schema_action"
        ),
        "fail_run"
    ),
    (
        (
            "artifact_contract",
            "non_finite_output_action"
        ),
        "fail_run"
    ),
    (
        (
            "reproducibility",
            "stochastic_procedures_permitted"
        ),
        False
    ),
    (
        (
            "reproducibility",
            "random_seed"
        ),
        None
    ),
    (
        (
            "reproducibility",
            "single_threaded_numerical_execution"
        ),
        True
    ),
    (
        (
            "reproducibility",
            "required_thread_environment"
        ),
        REQUIRED_THREAD_ENVIRONMENT
    ),
    (
        (
            "reproducibility",
            "byte_identical_repeat_execution_required"
        ),
        True
    ),
    (
        (
            "reporting",
            "analysis_status_rule"
        ),
        (
            "completed_with_model_family_failures_when_any_attempted_estimator_"
            "has_zero_successful_fits_otherwise_completed"
        )
    ),
    (
        (
            "reporting",
            "estimator_outcome_summary_required"
        ),
        True
    ),
    (
        (
            "reporting",
            "template_outcome_summary_required"
        ),
        True
    ),
    (
        (
            "reporting",
            "fitted_result_availability_summary_required"
        ),
        True
    ),
    (
        (
            "reporting",
            "candidate_analytical_result_statuses"
        ),
        [
            "available",
            "diagnostic_only",
            "not_available"
        ]
    ),
    (
        (
            "reporting",
            "candidate_analytical_result_status_rule"
        ),
        (
            "not_available_if_fit_not_completed_else_diagnostic_only_if_validation_uncertainty_influence_or_"
            "analytical_qualification_fails_else_available"
        )
    ),
    (
        (
            "reporting",
            "analytical_handoff_statuses"
        ),
        [
            "ready_for_continuous_trajectory_analysis",
            "ready_for_continuous_trajectory_analysis_with_model_family_and_candidate_restrictions",
            "not_ready_no_analytically_qualified_trajectory_models"
        ]
    ),
    (
        (
            "reporting",
            "analytical_handoff_status_rule"
        ),
        (
            "not_ready_if_no_analytically_qualified_trajectory_models_else_ready_with_model_family_and_candidate_"
            "restrictions_if_an_attempted_estimator_has_no_successful_fit_or_any_fitted_candidate_is_"
            "diagnostic_only_else_ready"
        )
    ),
    (
        (
            "reporting",
            "analysis_registration_status"
        ),
        "result_informed_exploratory"
    ),
    (
        (
            "reporting",
            "analysis_registration_status_required"
        ),
        True
    ),
    (
        (
            "reporting",
            "analysis_registration_rationale"
        ),
        "This version preserves outcome-blind candidate construction and specification selection while applying result-informed analytical qualification based on grouped predictive benchmarking and complete diagnostic availability. It also separates optimizer convergence from warning diagnostics, uses scale-aware numerical random-effect checks and requires upstream discrete-horizon authorization for point contrasts. Analyses governed by this version are exploratory rather than confirmatory."
    ),
    (
        (
            "observation_assembly",
            "observations_outside_evaluation_interval"
        ),
        (
            "included_in_model_assembly_for_support_region_member_cells_"
            "and_flagged_outside_evaluation_domain"
        )
    ),
    (
        (
            "validation",
            "training_observations_outside_support_region"
        ),
        (
            "included_for_support_region_member_cells_"
            "when_all_other_observation_requirements_are_satisfied"
        )
    ),
    (
        (
            "observation_accounting",
            "evaluation_interval_accounting",
            "applies_to_included_observations"
        ),
        True
    ),
    (
        (
            "observation_accounting",
            "evaluation_interval_accounting",
            "inside_evaluation_interval_flag_required"
        ),
        True
    ),
    (
        (
            "observation_accounting",
            "evaluation_interval_accounting",
            "inside_evaluation_interval_count_required"
        ),
        True
    ),
    (
        (
            "observation_accounting",
            "evaluation_interval_accounting",
            "outside_evaluation_interval_count_required"
        ),
        True
    ),
    (
        (
            "observation_accounting",
            "evaluation_interval_accounting",
            "inside_and_outside_counts_must_equal_included_record_count"
        ),
        True
    ),
    (
        (
            "admissibility",
            "longitudinal_structure",
            "minimum_repeated_cells_for_random_intercept"
        ),
        2
    ),
    (
        (
            "admissibility",
            "longitudinal_structure",
            "differential_trajectory_minimum_unique_horizons_per_condition"
        ),
        2
    ),
    (
        (
            "admissibility",
            "longitudinal_structure",
            "differential_trajectory_minimum_repeated_cells_per_condition"
        ),
        2
    ),
    (
        (
            "admissibility",
            "longitudinal_structure",
            "common_trajectory_minimum_repeated_cells"
        ),
        2
    ),
    (
        (
            "admissibility",
            "longitudinal_structure",
            "purely_cross_sectional_model_fitting_permitted"
        ),
        False
    ),
    (
        (
            "admissibility",
            "longitudinal_structure",
            "cross_sectional_trajectory_interpretation_permitted"
        ),
        False
    ),
    (
        (
            "admissibility",
            "longitudinal_structure",
            "cross_sectional_contrast_interpolation_permitted"
        ),
        False
    ),
    (
        (
            "contrast_evaluation",
            "completed_contrast_requires_completed_validation"
        ),
        True
    ),
    (
        (
            "contrast_evaluation",
            "completed_contrast_requires_completed_uncertainty"
        ),
        True
    ),
    (
        (
            "contrast_evaluation",
            "positive_ratio_prediction_required"
        ),
        True
    ),
    (
        (
            "contrast_evaluation",
            "nonpositive_ratio_prediction_status"
        ),
        "unavailable"
    ),
    (
        (
            "contrast_evaluation",
            "diagnostic_point_calculation_preserved"
        ),
        True
    )
]


# Parsing validated analytical inputs, output location and requested operation
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = (
            "Validate or fit deterministic descriptive battery-aging models within authorized analytical support."
        )
    )

    parser.add_argument(
        "--feature-manifest-path",
        type = Path,
        default = DEFAULT_FEATURE_MANIFEST_PATH
    )

    parser.add_argument(
        "--design-report-path",
        type = Path,
        default = DEFAULT_DESIGN_REPORT_PATH
    )

    parser.add_argument(
        "--modeling-policy-path",
        type = Path,
        default = DEFAULT_MODELING_POLICY_PATH
    )

    parser.add_argument(
        "--output-directory",
        type = Path,
        default = DEFAULT_OUTPUT_DIRECTORY
    )

    parser.add_argument(
        "--operation",
        choices = [
            "validate",
            "fit"
        ],
        default = "validate",
        help = (
            "Validate inputs and prepare admissible model matrices, or explicitly fit models and publish"
            " the complete analytical artifact set."
        )
    )

    return parser.parse_args()


# Rejecting duplicate JSON fields during deserialization
def reject_duplicate_json_fields(object_pairs: list[tuple[str, object]]) -> dict:
    parsed_object = {}

    for field_name, field_value in object_pairs:
        if field_name in parsed_object:
            raise RuntimeError(f"JSON document contains a duplicate field : {field_name}")

        parsed_object[field_name] = field_value

    return parsed_object


# Loading one strict JSON object
def load_strict_json_object(path: Path) -> dict:
    try:
        with path.open(
            mode = "r",
            encoding = "utf-8"
        ) as input_stream:
            parsed_value = json.load(
                input_stream,
                object_pairs_hook = reject_duplicate_json_fields
            )

    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON document is invalid : {path} : {exc}") from exc

    if not isinstance(
        parsed_value,
        dict
    ):
        raise RuntimeError(f"JSON document must contain an object : {path}")

    return parsed_value


# Calculating one file SHA-256 digest
def calculate_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with file_path.open(
        mode = "rb"
    ) as input_stream:
        while content_chunk := input_stream.read(
            chunk_size
        ):
            digest.update(
                content_chunk
            )

    return digest.hexdigest()


# Producing deterministic canonical JSON
def serialize_canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys = True,
        separators = (
            ",",
            ":"
        ),
        allow_nan = False
    )


# Building one deterministic identity
def build_deterministic_identity(value: object) -> str:
    canonical_value = serialize_canonical_json(
        value
    )

    return hashlib.sha256(
        canonical_value.encode(
            "utf-8"
        )
    ).hexdigest()


# Parsing one serialized JSON field
def parse_json_field(value: str, field_name: str) -> object:
    try:
        return json.loads(
            value
        )

    except json.JSONDecodeError as exc:
        raise RuntimeError(f"CSV field contains invalid JSON : {field_name} : {value!r}") from exc


# Parsing one CSV boolean
def parse_boolean(value: str, field_name: str) -> bool:
    if value == "true":
        return True

    if value == "false":
        return False

    raise RuntimeError(f"CSV field contains an invalid boolean : {field_name} : {value!r}")


# Parsing one finite numerical value
def parse_finite_number(value: str, field_name: str) -> float:
    try:
        numerical_value = float(
            value
        )

    except ValueError as exc:
        raise RuntimeError(f"CSV field contains an invalid number : {field_name} : {value!r}") from exc

    if not math.isfinite(
        numerical_value
    ):
        raise RuntimeError(f"CSV field contains a non-finite number : {field_name} : {value!r}")

    return numerical_value


# Loading one CSV table
def load_csv_table(table_path: Path) -> tuple[list[dict], list[str]]:
    with table_path.open(
        mode = "r",
        encoding = "utf-8",
        newline = ""
    ) as input_stream:
        table_reader = csv.DictReader(
            input_stream
        )

        field_names = table_reader.fieldnames

        if field_names is None:
            raise RuntimeError(f"CSV table does not contain a header : {table_path}")

        if len(field_names) != len(
            set(
                field_names
            )
        ):
            raise RuntimeError(f"CSV table contains duplicate columns : {table_path}")

        records = list(
            table_reader
        )

    return (
        records,
        field_names
    )


# Resolving a declared table path
def resolve_table_path(manifest_path: Path, table_record: dict) -> Path:
    file_name = table_record.get(
        "file_name"
    )

    if (
        not isinstance(
            file_name,
            str
        )
        or not file_name
    ):
        raise RuntimeError("Table manifest contains an invalid file name.")

    table_path = (
        manifest_path.parent
        / file_name
    ).resolve()

    manifest_directory = (
        manifest_path.parent.resolve()
    )

    if (
        table_path.parent
        != manifest_directory
    ):
        raise RuntimeError(f"Table manifest references a file outside its artifact directory : {file_name}")

    return table_path


# Loading and verifying one declared table
def load_verified_table(manifest_path: Path, table_name: str, table_record: dict) -> tuple[list[dict], list[str], Path]:
    table_path = resolve_table_path(
        manifest_path = manifest_path,
        table_record = table_record
    )

    if not table_path.is_file():
        raise RuntimeError(f"Declared analytical table was not found : {table_path}")

    expected_sha256 = table_record.get(
        "sha256"
    )

    observed_sha256 = calculate_sha256(
        table_path
    )

    if observed_sha256 != expected_sha256:
        raise RuntimeError(f"Analytical-table SHA-256 does not match its manifest : {table_name}")

    (
        records,
        field_names
    ) = load_csv_table(
        table_path
    )

    expected_row_count = table_record.get(
        "row_count"
    )

    if len(records) != expected_row_count:
        raise RuntimeError(
            f"Analytical-table row count does not match its manifest : {table_name} : {len(records)} : {expected_row_count}"
        )

    declared_schema = table_record.get(
        "schema"
    )

    if (
        not isinstance(
            declared_schema,
            list
        )
        or not declared_schema
    ):
        raise RuntimeError(f"Analytical table does not have a declared schema : {table_name}")

    declared_field_names = [
        schema_field.get(
            "name"
        )
        for schema_field in declared_schema
    ]

    if declared_field_names != field_names:
        raise RuntimeError(f"Analytical-table columns do not match their declared schema : {table_name}")

    return (
        records,
        field_names,
        table_path
    )


# Validating the declared analytical-artifact inventory
def validate_policy_artifact_inventory(modeling_policy: dict) -> None:
    artifact_contract = modeling_policy.get(
        "artifact_contract"
    )

    if not isinstance(
        artifact_contract,
        dict
    ):
        raise RuntimeError("Modeling policy does not define an artifact contract.")

    output_tables = artifact_contract.get(
        "output_tables"
    )

    if (
        not isinstance(
            output_tables,
            list
        )
        or not output_tables
        or any(
            not isinstance(
                table_name,
                str
            )
            or not table_name
            for table_name in output_tables
        )
    ):
        raise RuntimeError("Modeling policy contains an invalid output-table inventory.")

    if len(output_tables) != len(
        set(
            output_tables
        )
    ):
        raise RuntimeError("Modeling policy contains duplicate output-table names.")

    declared_output_tables = set(
        output_tables
    )

    if declared_output_tables != REQUIRED_OUTPUT_TABLES:
        missing_output_tables = sorted(
            REQUIRED_OUTPUT_TABLES.difference(
                declared_output_tables
            )
        )

        unexpected_output_tables = sorted(
            declared_output_tables.difference(
                REQUIRED_OUTPUT_TABLES
            )
        )

        raise RuntimeError(
            "Modeling policy output-table inventory does not match the implemented artifact inventory : "
            f"missing {missing_output_tables} : unexpected {unexpected_output_tables}"
        )

    table_file_extension = artifact_contract.get(
        "table_file_extension"
    )

    if table_file_extension != ".csv":
        raise RuntimeError("Analytical tables must use the declared CSV extension.")

    report_file_name = artifact_contract.get(
        "report_file_name"
    )

    if (
        not isinstance(
            report_file_name,
            str
        )
        or not report_file_name
        or Path(
            report_file_name
        ).name != report_file_name
        or Path(
            report_file_name
        ).suffix != ".json"
    ):
        raise RuntimeError("Modeling policy contains an invalid report file name.")

    expected_file_names = [
        *(
            f"{table_name}{table_file_extension}"
            for table_name in output_tables
        ),
        report_file_name
    ]

    if len(expected_file_names) != len(
        set(
            expected_file_names
        )
    ):
        raise RuntimeError("Modeling policy artifact file names are not unique.")


# Reading one required nested policy value
def read_required_policy_value(modeling_policy: dict, policy_path: tuple[str, ...]) -> object:
    current_value: object = modeling_policy

    for field_name in policy_path:
        if (
            not isinstance(
                current_value,
                dict
            )
            or field_name not in current_value
        ):
            raise RuntimeError(f"Modeling policy is missing an implementation-required value : {'.'.join(policy_path)}")

        current_value = current_value[
            field_name
        ]

    return current_value


# Confirming that the policy matches implemented analytical behavior
def validate_implemented_policy_values(modeling_policy: dict) -> None:
    for (
        policy_path,
        expected_value
    ) in IMPLEMENTED_POLICY_VALUES:
        observed_value = read_required_policy_value(
            modeling_policy = modeling_policy,
            policy_path = policy_path
        )

        if (
            type(
                observed_value
            )
            is not type(
                expected_value
            )
            or observed_value != expected_value
        ):
            raise RuntimeError(
                f"Modeling policy value is not supported by the current implementation : {'.'.join(policy_path)} : "
                f"expected {serialize_canonical_json(expected_value)} : "
                f"observed {serialize_canonical_json(observed_value)}"
            )

    observed_candidate_templates = read_required_policy_value(
        modeling_policy = modeling_policy,
        policy_path = (
            "candidate_templates",
        )
    )

    if observed_candidate_templates != IMPLEMENTED_CANDIDATE_TEMPLATES:
        raise RuntimeError("Modeling policy candidate templates do not match the implemented model templates.")


# Validating the implementation-declared analytical-table contracts
def validate_output_table_contracts() -> None:
    declared_table_names = set(
        OUTPUT_TABLE_CONTRACTS
    )

    if declared_table_names != REQUIRED_OUTPUT_TABLES:
        missing_table_names = sorted(
            REQUIRED_OUTPUT_TABLES.difference(
                declared_table_names
            )
        )

        unexpected_table_names = sorted(
            declared_table_names.difference(
                REQUIRED_OUTPUT_TABLES
            )
        )

        raise RuntimeError(
            "Implemented analytical-table contracts do not match the required inventory : "
            f"missing {missing_table_names} : unexpected {unexpected_table_names}"
        )

    for (
        table_name,
        table_contract
    ) in OUTPUT_TABLE_CONTRACTS.items():
        if set(
            table_contract
        ) != {
            "field_names",
            "primary_key"
        }:
            raise RuntimeError(f"Analytical-table contract contains unsupported fields : {table_name}")

        field_names = table_contract[
            "field_names"
        ]

        primary_key = table_contract[
            "primary_key"
        ]

        if (
            not isinstance(
                field_names,
                list
            )
            or not field_names
            or any(
                not isinstance(
                    field_name,
                    str
                )
                or not field_name
                for field_name in field_names
            )
            or len(
                field_names
            )
            != len(
                set(
                    field_names
                )
            )
        ):
            raise RuntimeError(f"Analytical-table contract has an invalid field inventory : {table_name}")

        if (
            not isinstance(
                primary_key,
                list
            )
            or not primary_key
            or len(
                primary_key
            )
            != len(
                set(
                    primary_key
                )
            )
        ):
            raise RuntimeError(f"Analytical-table contract has an invalid primary key : {table_name}")

        missing_primary_key_fields = sorted(
            set(
                primary_key
            ).difference(
                field_names
            )
        )

        if missing_primary_key_fields:
            raise RuntimeError(
                f"Analytical-table primary key references undeclared fields : {table_name} : {missing_primary_key_fields}"
            )


# Validating the frozen modeling policy
def validate_modeling_policy(modeling_policy: dict) -> None:
    observed_sections = set(
        modeling_policy
    )

    missing_sections = sorted(
        REQUIRED_POLICY_SECTIONS.difference(
            observed_sections
        )
    )

    unexpected_sections = sorted(
        observed_sections.difference(
            REQUIRED_POLICY_SECTIONS
        )
    )

    if (
        missing_sections
        or unexpected_sections
    ):
        raise RuntimeError(
            "Modeling policy sections do not match the implemented contract : "
            f"missing {missing_sections} : unexpected {unexpected_sections}"
        )

    policy_version = modeling_policy[
        "policy_version"
    ]

    if (
        not isinstance(
            policy_version,
            str
        )
        or not policy_version
    ):
        raise RuntimeError("Modeling policy contains an invalid version.")

    effective_date = modeling_policy[
        "effective_date"
    ]

    if (
        not isinstance(
            effective_date,
            str
        )
        or not effective_date
    ):
        raise RuntimeError("Modeling policy contains an invalid effective date.")

    previous_policy_sha256 = modeling_policy[
        "previous_policy_sha256"
    ]

    if (
        not isinstance(
            previous_policy_sha256,
            str
        )
        or len(
            previous_policy_sha256
        )
        != 64
        or any(
            character
            not in "0123456789abcdef"
            for character in previous_policy_sha256
        )
    ):
        raise RuntimeError("Modeling policy contains an invalid previous-policy SHA-256 digest.")

    amendment = modeling_policy[
        "amendment"
    ]

    required_amendment_fields = {
        "reason",
        "result_informed",
        "affected_analyses",
        "complete_rerun_required"
    }

    if (
        not isinstance(
            amendment,
            dict
        )
        or set(
            amendment
        )
        != required_amendment_fields
    ):
        raise RuntimeError("Modeling policy contains an invalid amendment record.")

    if (
        not isinstance(
            amendment[
                "reason"
            ],
            str
        )
        or not amendment[
            "reason"
        ].strip()
    ):
        raise RuntimeError("Modeling policy amendment does not contain a reason.")

    if not isinstance(
        amendment[
            "result_informed"
        ],
        bool
    ):
        raise RuntimeError("Modeling policy amendment contains an invalid result-informed flag.")

    affected_analyses = amendment[
        "affected_analyses"
    ]

    if (
        not isinstance(
            affected_analyses,
            list
        )
        or not affected_analyses
        or any(
            not isinstance(
                affected_analysis,
                str
            )
            or not affected_analysis
            for affected_analysis in affected_analyses
        )
        or len(
            affected_analyses
        )
        != len(
            set(
                affected_analyses
            )
        )
    ):
        raise RuntimeError("Modeling policy amendment contains invalid affected analyses.")

    if amendment[
        "complete_rerun_required"
    ] is not True:
        raise RuntimeError("Modeling policy amendments must require a complete rerun.")

    validate_implemented_policy_values(
        modeling_policy
    )

    validate_policy_artifact_inventory(
        modeling_policy
    )

    validate_output_table_contracts()


# Loading and validating the feature manifest
def load_feature_inputs(feature_manifest_path: Path) -> tuple[dict, dict[str, list[dict]], dict[str, Path]]:
    feature_manifest = load_strict_json_object(
        feature_manifest_path
    )

    if feature_manifest.get(
        "extraction_status"
    ) != "completed":
        raise RuntimeError("Feature manifest does not describe a completed extraction.")

    table_manifest = feature_manifest.get(
        "tables"
    )

    if not isinstance(
        table_manifest,
        dict
    ):
        raise RuntimeError("Feature manifest does not contain a table manifest.")

    missing_tables = REQUIRED_FEATURE_TABLES.difference(
        table_manifest
    )

    if missing_tables:
        raise RuntimeError(f"Feature manifest is missing required tables : {sorted(missing_tables)}")

    loaded_tables = {}
    table_paths = {}

    for table_name in sorted(
        REQUIRED_FEATURE_TABLES
    ):
        (
            table_records,
            _,
            table_path
        ) = load_verified_table(
            manifest_path = feature_manifest_path,
            table_name = table_name,
            table_record = table_manifest[
                table_name
            ]
        )

        loaded_tables[
            table_name
        ] = table_records

        table_paths[
            table_name
        ] = table_path

    return (
        feature_manifest,
        loaded_tables,
        table_paths
    )


# Loading and validating the experimental-design report
def load_design_inputs(design_report_path: Path) -> tuple[dict, dict[str, list[dict]], dict[str, Path]]:
    design_report = load_strict_json_object(
        design_report_path
    )

    if design_report.get(
        "audit_status"
    ) != "completed":
        raise RuntimeError("Experimental-design report does not describe a completed audit.")

    if design_report.get(
        "independent_experimental_unit"
    ) != "cell":
        raise RuntimeError("Experimental-design report does not identify the cell as the independent experimental unit.")

    if design_report.get(
        "calendar_and_cycle_designs_separated"
    ) is not True:
        raise RuntimeError("Experimental-design report does not separate calendar and cycle designs.")

    if design_report.get(
        "outcome_magnitude_or_direction_used_for_support_selection"
    ) is not False:
        raise RuntimeError("Experimental-design support was not outcome-blind.")

    if design_report.get(
        "discrete_comparison_horizon_status"
    ) != (
        "not_authorized_without_outcome_blind_pre_specification"
    ):
        raise RuntimeError("Experimental-design report contains an unexpected discrete-horizon status.")

    if design_report.get(
        "point_contrast_requirement"
    ) != (
        "outcome_blind_scientific_or_protocol_pre_specification_before_comparative_results_are_examined"
    ):
        raise RuntimeError("Experimental-design report contains an unexpected point-contrast requirement.")

    horizon_selection_policy = design_report.get(
        "horizon_selection_policy"
    )

    if not isinstance(
        horizon_selection_policy,
        dict
    ):
        raise RuntimeError("Experimental-design report does not contain a horizon-selection policy.")

    approved_discrete_horizons = (
        horizon_selection_policy.get(
            "approved_discrete_horizons"
        )
    )

    if approved_discrete_horizons != []:
        raise RuntimeError("The current modeling policy does not implement approved discrete horizons.")

    table_manifest = design_report.get(
        "tables"
    )

    if not isinstance(
        table_manifest,
        dict
    ):
        raise RuntimeError("Experimental-design report does not contain a table manifest.")

    missing_tables = REQUIRED_DESIGN_TABLES.difference(
        table_manifest
    )

    if missing_tables:
        raise RuntimeError(f"Experimental-design report is missing required tables : {sorted(missing_tables)}")

    loaded_tables = {}
    table_paths = {}

    for table_name in sorted(
        REQUIRED_DESIGN_TABLES
    ):
        (
            table_records,
            _,
            table_path
        ) = load_verified_table(
            manifest_path = design_report_path,
            table_name = table_name,
            table_record = table_manifest[
                table_name
            ]
        )

        loaded_tables[
            table_name
        ] = table_records

        table_paths[
            table_name
        ] = table_path

    return (
        design_report,
        loaded_tables,
        table_paths
    )


# Confirming that feature and design artifacts share one source
def validate_input_bindings(feature_manifest_path: Path, feature_manifest: dict,
                            design_report: dict
) -> None:
    observed_feature_manifest_sha256 = calculate_sha256(
        feature_manifest_path
    )

    if design_report.get(
        "feature_manifest_sha256"
    ) != observed_feature_manifest_sha256:
        raise RuntimeError("Experimental-design report is not bound to the provided feature manifest.")

    if (
        feature_manifest.get(
            "data_quality_contract_version"
        )
        != design_report.get(
            "data_quality_contract_version"
        )
    ):
        raise RuntimeError("Feature and experimental-design artifacts use different data-quality contract versions.")


# Validating one analytical-scope table
def validate_analytical_scope(scope_records: list[dict]) -> None:
    if not scope_records:
        raise RuntimeError("Analytical scope does not contain records.")

    available_columns = set(
        scope_records[0]
    )

    missing_columns = REQUIRED_SCOPE_COLUMNS.difference(
        available_columns
    )

    if missing_columns:
        raise RuntimeError(f"Analytical scope is missing required columns : {sorted(missing_columns)}")

    for row_number, scope_record in enumerate(
        scope_records,
        start = 2
    ):
        if set(scope_record) != available_columns:
            raise RuntimeError(f"Analytical scope contains an inconsistent row schema : row {row_number}")

        if scope_record[
            "authorization_status"
        ] not in {
            "permitted_for_model_evaluation_with_restrictions",
            "permitted_for_model_evaluation_sensitivity_only"
        }:
            raise RuntimeError(f"Analytical scope contains an unauthorized record : row {row_number}")

        if parse_boolean(
            scope_record[
                "structurally_estimable"
            ],
            "structurally_estimable"
        ) is not True:
            raise RuntimeError(f"Analytical scope contains a non-estimable term : row {row_number}")

        if scope_record[
            "stage_pooling_status"
        ] != "stage_specific":
            raise RuntimeError(f"Analytical scope contains a non-stage-specific record : row {row_number}")

        if scope_record[
            "causal_identification_status"
        ] != "not_established":
            raise RuntimeError(f"Analytical scope contains an invalid causal status : row {row_number}")

        parse_finite_number(
            scope_record[
                "support_interval_minimum"
            ],
            "support_interval_minimum"
        )

        parse_finite_number(
            scope_record[
                "support_interval_maximum"
            ],
            "support_interval_maximum"
        )

        parse_json_field(
            scope_record[
                "permitted_adjustment_variables"
            ],
            "permitted_adjustment_variables"
        )

        parse_json_field(
            scope_record[
                "required_validation_grouping"
            ],
            "required_validation_grouping"
        )


# Confirming that candidate fields are constant within one group
def validate_candidate_shared_fields(candidate_records: list[dict], candidate_key: tuple[str, ...]) -> None:
    for field_name in CANDIDATE_SHARED_FIELDS:
        observed_values = {
            candidate_record[
                field_name
            ]
            for candidate_record in candidate_records
        }

        if len(observed_values) != 1:
            raise RuntimeError(
                "Analytical-scope candidate contains conflicting values : "
                f"{candidate_key} : {field_name} : {sorted(observed_values)}"
            )


# Resolving candidate authorization from selected source terms
def resolve_candidate_authorization(scope_records: list[dict], modeling_policy: dict) -> tuple[str, str]:
    authorization_policy = modeling_policy[
        "candidate_assembly"
    ][
        "source_term_authorization"
    ]

    classification_precedence = (
        authorization_policy[
            "question_classification_precedence"
        ]
    )

    authorization_by_classification = (
        authorization_policy[
            "authorization_by_question_classification"
        ]
    )

    if (
        not isinstance(
            classification_precedence,
            list
        )
        or not classification_precedence
        or len(
            classification_precedence
        )
        != len(
            set(
                classification_precedence
            )
        )
    ):
        raise RuntimeError("Modeling policy contains an invalid source-term classification precedence.")

    if set(
        classification_precedence
    ) != set(
        authorization_by_classification
    ):
        raise RuntimeError("Source-term classifications and authorization mappings do not reconcile.")

    precedence_by_classification = {
        classification_name: precedence_index
        for precedence_index, classification_name in enumerate(
            classification_precedence
        )
    }

    observed_classifications = []

    for scope_record in scope_records:
        question_classification = scope_record[
            "question_classification"
        ]

        authorization_status = scope_record[
            "authorization_status"
        ]

        if question_classification not in (
            precedence_by_classification
        ):
            raise RuntimeError(
                f"Analytical-scope term contains an unsupported question classification : {question_classification}"
            )

        expected_authorization_status = (
            authorization_by_classification[
                question_classification
            ]
        )

        if (
            authorization_status
            != expected_authorization_status
        ):
            raise RuntimeError(
                "Analytical-scope term classification conflicts with its authorization status : "
                f"{scope_record['term_name']} : {question_classification} : {authorization_status}"
            )

        observed_classifications.append(
            question_classification
        )

    candidate_question_classification = max(
        observed_classifications,
        key = lambda classification_name: (
            precedence_by_classification[
                classification_name
            ]
        )
    )

    candidate_authorization_status = (
        authorization_by_classification[
            candidate_question_classification
        ]
    )

    return (
        candidate_question_classification,
        candidate_authorization_status
    )


# Building complete context-term accounting for one candidate
def build_context_candidate_term_records(candidate_identity: str, support_region_identity: str,
                                         template_name: str, scope_record: dict
) -> list[dict]:
    context_family = scope_record[
        "context_family"
    ]

    permitted_adjustment_variables = (
        parse_json_field(
            scope_record[
                "permitted_adjustment_variables"
            ],
            "permitted_adjustment_variables"
        )
    )

    if (
        not isinstance(
            permitted_adjustment_variables,
            list
        )
        or any(
            not isinstance(
                field_name,
                str
            )
            or not field_name
            for field_name in (
                permitted_adjustment_variables
            )
        )
        or len(
            permitted_adjustment_variables
        )
        != len(
            set(
                permitted_adjustment_variables
            )
        )
    ):
        raise RuntimeError("Candidate contains invalid permitted adjustment variables.")

    if context_family == "no_additional_context":
        if permitted_adjustment_variables:
            raise RuntimeError("Candidate without additional context declares adjustment variables.")

        return []

    if context_family == "laboratory":
        expected_adjustment_variables = [
            "laboratory"
        ]

    else:
        raise RuntimeError(f"Candidate contains an unsupported context family : {context_family}")

    if (
        permitted_adjustment_variables
        != expected_adjustment_variables
    ):
        raise RuntimeError("Candidate context family conflicts with its permitted adjustment variables.")

    return [
        {
            "candidate_identity": candidate_identity,
            "support_region_identity": (
                support_region_identity
            ),
            "template_name": template_name,
            "term_name": field_name,
            "term_type": "design_context",
            "term_source": "analytical_scope.csv",
            "source_question_classification": None,
            "source_authorization_status": None,
            "term_disposition": "selected",
            "term_reason_code": None
        }
        for field_name in (
            permitted_adjustment_variables
        )
    ]


# Building complete policy-derived term accounting for one candidate
def build_derived_candidate_term_records(candidate_identity: str, support_region_identity: str,
                                         candidate_template: dict, source_terms: list[tuple[str, str]],
                                         modeling_policy: dict
) -> list[dict]:
    template_name = candidate_template[
        "template_name"
    ]

    horizon_basis = candidate_template[
        "horizon_basis"
    ]

    if horizon_basis == "linear":
        horizon_term_names = [
            "horizon_linear"
        ]

    elif horizon_basis == "restricted_cubic_spline":
        horizon_term_names = [
            "horizon_linear",
            "horizon_restricted_cubic"
        ]

    else:
        raise RuntimeError(f"Candidate template contains an unsupported horizon basis : {template_name} : {horizon_basis}")

    derived_term_records = [
        {
            "candidate_identity": candidate_identity,
            "support_region_identity": (
                support_region_identity
            ),
            "template_name": template_name,
            "term_name": "intercept",
            "term_type": "intercept",
            "term_source": (
                "descriptive_modeling_policy"
            ),
            "source_question_classification": None,
            "source_authorization_status": None,
            "term_disposition": "selected",
            "term_reason_code": None
        }
    ]

    for horizon_term_name in horizon_term_names:
        derived_term_records.append(
            {
                "candidate_identity": (
                    candidate_identity
                ),
                "support_region_identity": (
                    support_region_identity
                ),
                "template_name": template_name,
                "term_name": horizon_term_name,
                "term_type": "horizon_basis",
                "term_source": (
                    "descriptive_modeling_policy"
                ),
                "source_question_classification": None,
                "source_authorization_status": None,
                "term_disposition": "selected",
                "term_reason_code": None
            }
        )

    include_horizon_interactions = (
        candidate_template[
            "include_horizon_by_treatment_interactions"
        ]
    )

    interaction_disposition = (
        "selected"
        if include_horizon_interactions
        else "not_selected"
    )

    interaction_reason_code = (
        None
        if include_horizon_interactions
        else modeling_policy[
            "candidate_generation"
        ][
            "horizon_interactions_not_selected_reason"
        ]
    )

    for horizon_term_name in horizon_term_names:
        for source_term_name, source_term_type in (
            source_terms
        ):
            if source_term_type not in {
                "main_effect",
                "target_treatment",
                "pairwise_interaction"
            }:
                raise RuntimeError(
                    "Derived horizon interaction references an unsupported source-term type : "
                    f"{candidate_identity} : {source_term_name} : {source_term_type}"
                )

            derived_term_records.append(
                {
                    "candidate_identity": (
                        candidate_identity
                    ),
                    "support_region_identity": (
                        support_region_identity
                    ),
                    "template_name": template_name,
                    "term_name": (
                        f"{horizon_term_name}:"
                        f"{source_term_name}"
                    ),
                    "term_type": (
                        "horizon_by_treatment_interaction"
                    ),
                    "term_source": (
                        "descriptive_modeling_policy"
                    ),
                    "term_disposition": (
                        interaction_disposition
                    ),
                    "term_reason_code": (
                        interaction_reason_code
                    ),
                    "source_question_classification": None,
                    "source_authorization_status": None
                }
            )

    return derived_term_records


# Building the complete model-candidate registry
def build_candidate_registry(scope_records: list[dict], modeling_policy: dict) -> tuple[list[dict], list[dict]]:
    records_by_candidate = defaultdict(
        list
    )

    for scope_record in scope_records:
        candidate_key = tuple(
            scope_record[
                field_name
            ]
            for field_name in CANDIDATE_KEY_FIELDS
        )

        records_by_candidate[
            candidate_key
        ].append(
            scope_record
        )

    candidate_templates = modeling_policy[
        "candidate_templates"
    ]

    candidate_records = []
    candidate_term_records = []

    for candidate_key in sorted(
        records_by_candidate
    ):
        scope_group = records_by_candidate[
            candidate_key
        ]

        validate_candidate_shared_fields(
            candidate_records = scope_group,
            candidate_key = candidate_key
        )

        first_scope_record = scope_group[0]

        (
            candidate_question_classification,
            candidate_authorization_status
        ) = resolve_candidate_authorization(
            scope_records = scope_group,
            modeling_policy = modeling_policy
        )

        scope_terms = sorted(
            (
                scope_record[
                    "term_name"
                ],
                scope_record[
                    "term_type"
                ]
            )
            for scope_record in scope_group
        )

        if len(scope_terms) != len(
            set(
                scope_terms
            )
        ):
            raise RuntimeError(f"Analytical-scope candidate contains duplicate terms : {candidate_key}")

        scope_record_by_term = {
            (
                scope_record[
                    "term_name"
                ],
                scope_record[
                    "term_type"
                ]
            ): scope_record
            for scope_record in scope_group
        }

        if len(
            scope_record_by_term
        ) != len(
            scope_group
        ):
            raise RuntimeError(f"Analytical-scope candidate contains duplicate source-term records : {candidate_key}")

        source_term_names = [
            term_name
            for term_name, term_type in scope_terms
        ]

        if len(source_term_names) != len(
            set(
                source_term_names
            )
        ):
            raise RuntimeError(f"Analytical-scope candidate contains duplicate encoded term names : {candidate_key}")

        authorized_term_names = {
            term_name
            for term_name, term_type in scope_terms
        }

        for candidate_template in candidate_templates:
            template_name = candidate_template[
                "template_name"
            ]

            identity_payload = {
                **{
                    field_name: first_scope_record[
                        field_name
                    ]
                    for field_name in CANDIDATE_KEY_FIELDS
                },
                "template_name": template_name
            }

            candidate_identity = (
                build_deterministic_identity(
                    identity_payload
                )
            )

            template_selected_interaction_terms = [
                term_name
                for term_name, term_type in scope_terms
                if term_type == "pairwise_interaction"
            ]

            missing_lower_order_components = []

            for interaction_term_name in (
                template_selected_interaction_terms
            ):
                for component_name in (
                    interaction_term_name.split(":")
                ):
                    if component_name not in (
                        authorized_term_names
                    ):
                        missing_lower_order_components.append(
                            component_name
                        )

            if missing_lower_order_components:
                hierarchy_status = "inadmissible"
                hierarchy_reason_code = (
                    "REQUIRED_LOWER_ORDER_TERM_NOT_AUTHORIZED"
                )
                admissibility_status = "inadmissible"
            else:
                hierarchy_status = "complete"
                hierarchy_reason_code = None
                admissibility_status = "pending"

            candidate_record = {
                "candidate_identity": candidate_identity,
                **identity_payload,
                **{
                    field_name: first_scope_record[
                        field_name
                    ]
                    for field_name in (
                        CANDIDATE_SHARED_FIELDS
                    )
                    if field_name not in {
                        "condition_protocol_count",
                        "independent_cell_count"
                    }
                },
                "support_condition_count": (
                    parse_nonnegative_integer(
                        first_scope_record[
                            "condition_protocol_count"
                        ],
                        "condition_protocol_count"
                    )
                ),
                "support_member_cell_count": (
                    parse_nonnegative_integer(
                        first_scope_record[
                            "independent_cell_count"
                        ],
                        "independent_cell_count"
                    )
                ),
                "assembled_model_observation_count": None,
                "assembled_model_condition_count": None,
                "assembled_model_cell_count": None,
                "inside_evaluation_interval_observation_count": None,
                "outside_evaluation_interval_observation_count": None,
                "repeated_cell_count": None,
                "single_observation_cell_count": None,
                "minimum_condition_unique_horizon_count": None,
                "minimum_condition_repeated_cell_count": None,
                "longitudinal_structure_status": None,
                "numerical_fit_status": "not_attempted",
                "analytical_result_status": "not_available",
                "analytical_result_reason_code": "MODEL_NOT_FITTED",
                "question_classification": (
                    candidate_question_classification
                ),
                "authorization_status": (
                    candidate_authorization_status
                ),
                "estimator": candidate_template[
                    "estimator"
                ],
                "horizon_basis": candidate_template[
                    "horizon_basis"
                ],
                "include_horizon_by_treatment_interactions": (
                    candidate_template[
                        "include_horizon_by_treatment_interactions"
                    ]
                ),
                "cell_random_intercept": (
                    candidate_template[
                        "cell_random_intercept"
                    ]
                ),
                "authorized_term_count": len(
                    scope_terms
                ),
                "interaction_hierarchy_status": (
                    hierarchy_status
                ),
                "interaction_hierarchy_reason_code": (
                    hierarchy_reason_code
                ),
                "admissibility_status": (
                    admissibility_status
                ),
                "admissibility_reason_code": None,
                "observation_assembly_status": None,
                "candidate_source_record_count": None,
                "candidate_included_record_count": None,
                "candidate_excluded_record_count": None,
                "coverage_anchor_support_region_identity": None,
                "duration_anchor_support_region_identity": None,
                "model_matrix_status": None,
                "model_matrix_error": None,
                "model_matrix_column_count": None,
                "model_matrix_rank": None,
                "effective_condition_number": None,
                "residual_independent_cell_degrees_of_freedom": None,
                "fit_identity": None,
                "model_matrix_column_names": None,
                "model_encoding": None,
                "fit_status": "not_attempted",
                "fit_reason_code": None,
                "fit_error": None,
                "canonical_fit_candidate_identity": None,
                "fitted_coefficient_count": None,
                "uncertainty_status": "not_attempted",
                "uncertainty_reason_code": None,
                "validation_status": "not_attempted",
                "validation_reason_code": None,
                "validation_interval_status": None,
                "influence_status": "not_attempted",
                "influence_reason_code": None,
                "selection_status": "pending",
                "selection_reason_code": None,
                "final_disposition": None,
                "final_reason_code": None
            }

            candidate_records.append(
                candidate_record
            )

            for term_name, term_type in scope_terms:
                source_scope_record = (
                    scope_record_by_term[
                        (
                            term_name,
                            term_type
                        )
                    ]
                )

                candidate_term_records.append(
                    {
                        "candidate_identity": (
                            candidate_identity
                        ),
                        "support_region_identity": (
                            first_scope_record[
                                "support_region_identity"
                            ]
                        ),
                        "template_name": template_name,
                        "term_name": term_name,
                        "term_type": term_type,
                        "term_source": (
                            "analytical_scope.csv"
                        ),
                        "source_question_classification": (
                            source_scope_record[
                                "question_classification"
                            ]
                        ),
                        "source_authorization_status": (
                            source_scope_record[
                                "authorization_status"
                            ]
                        ),
                        "term_disposition": "selected",
                        "term_reason_code": None
                    }
                )

            candidate_term_records.extend(
                build_context_candidate_term_records(
                    candidate_identity = (
                        candidate_identity
                    ),
                    support_region_identity = (
                        first_scope_record[
                            "support_region_identity"
                        ]
                    ),
                    template_name = template_name,
                    scope_record = first_scope_record
                )
            )

            candidate_term_records.extend(
                build_derived_candidate_term_records(
                    candidate_identity = (
                        candidate_identity
                    ),
                    support_region_identity = (
                        first_scope_record[
                            "support_region_identity"
                        ]
                    ),
                    candidate_template = (
                        candidate_template
                    ),
                    source_terms = scope_terms,
                    modeling_policy = (
                        modeling_policy
                    )
                )
            )

            for missing_component_name in sorted(
                set(
                    missing_lower_order_components
                )
            ):
                candidate_term_records.append(
                    {
                        "candidate_identity": (
                            candidate_identity
                        ),
                        "support_region_identity": (
                            first_scope_record[
                                "support_region_identity"
                            ]
                        ),
                        "template_name": template_name,
                        "term_name": (
                            missing_component_name
                        ),
                        "term_type": (
                            "required_lower_order_component"
                        ),
                        "source_question_classification": None,
                        "source_authorization_status": None,
                        "term_source": (
                            "interaction_hierarchy"
                        ),
                        "term_disposition": (
                            "required_but_not_authorized"
                        ),
                        "term_reason_code": (
                            "REQUIRED_LOWER_ORDER_TERM_NOT_AUTHORIZED"
                        )
                    }
                )

    candidate_identities = [
        candidate_record[
            "candidate_identity"
        ]
        for candidate_record in candidate_records
    ]

    if len(candidate_identities) != len(
        set(
            candidate_identities
        )
    ):
        raise RuntimeError("Candidate registry contains duplicate identities.")

    return (
        candidate_records,
        candidate_term_records
    )


# Parsing one non-negative integral field
def parse_nonnegative_integer(value: str, field_name: str) -> int:
    numerical_value = parse_finite_number(
        value = value,
        field_name = field_name
    )

    if (
        numerical_value < 0
        or not numerical_value.is_integer()
    ):
        raise RuntimeError(f"CSV field does not contain a non-negative integer : {field_name} : {value!r}")

    return int(
        numerical_value
    )


# Calculating the declared width of one support interval
def calculate_support_interval_width(candidate_record: dict) -> float:
    lower_horizon = parse_finite_number(
        candidate_record[
            "support_interval_minimum"
        ],
        "support_interval_minimum"
    )

    upper_horizon = parse_finite_number(
        candidate_record[
            "support_interval_maximum"
        ],
        "support_interval_maximum"
    )

    if upper_horizon < lower_horizon:
        raise RuntimeError(
            f"Candidate support interval has reversed boundaries : {candidate_record['candidate_identity']}"
        )

    return (
        upper_horizon
        - lower_horizon
    )


# Selecting outcome-blind support anchors and candidate templates
def select_model_candidates(candidate_records: list[dict], candidate_term_records: list[dict],
                            modeling_policy: dict
) -> None:
    candidate_generation_policy = modeling_policy[
        "candidate_generation"
    ]

    fit_selection_fields = candidate_generation_policy[
        "fit_selection_unit_fields"
    ]

    template_application = candidate_generation_policy[
        "template_application"
    ]

    coverage_templates = set(
        template_application[
            "coverage_anchor"
        ]
    )

    duration_templates = set(
        template_application[
            "duration_anchor_when_distinct"
        ]
    )

    records_by_selection_unit = defaultdict(
        list
    )

    for candidate_record in candidate_records:
        selection_key = tuple(
            candidate_record[
                field_name
            ]
            for field_name in fit_selection_fields
        )

        records_by_selection_unit[
            selection_key
        ].append(
            candidate_record
        )

    selection_by_candidate_identity = {}

    for selection_key in sorted(
        records_by_selection_unit
    ):
        selection_records = records_by_selection_unit[
            selection_key
        ]

        region_records = {}

        for candidate_record in selection_records:
            region_identity = candidate_record[
                "support_region_identity"
            ]

            existing_region_record = (
                region_records.get(
                    region_identity
                )
            )

            if existing_region_record is None:
                region_records[
                    region_identity
                ] = candidate_record

                continue

            fields_requiring_agreement = [
                "support_condition_count",
                "support_member_cell_count",
                "analytical_observation_count",
                "support_interval_minimum",
                "support_interval_maximum",
                "support_interval_lower_inclusive",
                "support_interval_upper_inclusive"
            ]

            for field_name in (
                fields_requiring_agreement
            ):
                if (
                    existing_region_record[
                        field_name
                    ]
                    != candidate_record[
                        field_name
                    ]
                ):
                    raise RuntimeError(
                        "Candidate templates contain conflicting support-region metadata : "
                        f"{selection_key} : {region_identity} : {field_name}"
                    )

        distinct_region_records = list(
            region_records.values()
        )

        coverage_anchor_record = min(
            distinct_region_records,
            key = lambda record: (
                -parse_nonnegative_integer(
                    record["support_condition_count"],
                    "support_condition_count"
                ),
                -parse_nonnegative_integer(
                    record["support_member_cell_count"],
                    "support_member_cell_count"
                ),
                -calculate_support_interval_width(record),
                -parse_nonnegative_integer(
                    record["analytical_observation_count"],
                    "analytical_observation_count"
                ),
                record["support_region_identity"]
            )
        )

        duration_anchor_record = min(
            distinct_region_records,
            key = lambda record: (
                -calculate_support_interval_width(record),
                -parse_nonnegative_integer(
                    record["support_condition_count"],
                    "support_condition_count"
                ),
                -parse_nonnegative_integer(
                    record["support_member_cell_count"],
                    "support_member_cell_count"
                ),
                -parse_nonnegative_integer(
                    record["analytical_observation_count"],
                    "analytical_observation_count"
                ),
                record["support_region_identity"]
            )
        )

        coverage_anchor_identity = (
            coverage_anchor_record[
                "support_region_identity"
            ]
        )

        duration_anchor_identity = (
            duration_anchor_record[
                "support_region_identity"
            ]
        )

        for candidate_record in selection_records:
            candidate_identity = candidate_record[
                "candidate_identity"
            ]

            region_identity = candidate_record[
                "support_region_identity"
            ]

            template_name = candidate_record[
                "template_name"
            ]

            selected_as_coverage_anchor = (
                region_identity
                == coverage_anchor_identity
                and template_name
                in coverage_templates
            )

            selected_as_duration_anchor = (
                duration_anchor_identity
                != coverage_anchor_identity
                and region_identity
                == duration_anchor_identity
                and template_name
                in duration_templates
            )

            if selected_as_coverage_anchor:
                selection_status = "selected"
                selection_reason_code = (
                    "COVERAGE_ANCHOR"
                )

            elif selected_as_duration_anchor:
                selection_status = "selected"
                selection_reason_code = (
                    "DURATION_ANCHOR"
                )

            elif region_identity not in {
                coverage_anchor_identity,
                duration_anchor_identity
            }:
                selection_status = "not_selected"
                selection_reason_code = (
                    candidate_generation_policy[
                        "unselected_source_group_reason_code"
                    ]
                )

            else:
                selection_status = "not_selected"
                selection_reason_code = (
                    candidate_generation_policy[
                        "unselected_template_reason_code"
                    ]
                )

            candidate_record[
                "selection_status"
            ] = selection_status

            candidate_record[
                "selection_reason_code"
            ] = selection_reason_code

            candidate_record[
                "coverage_anchor_support_region_identity"
            ] = coverage_anchor_identity

            candidate_record[
                "duration_anchor_support_region_identity"
            ] = duration_anchor_identity

            if selection_status == "not_selected":
                candidate_record[
                    "admissibility_status"
                ] = "not_selected"

            selection_by_candidate_identity[
                candidate_identity
            ] = {
                "selection_status": selection_status,
                "selection_reason_code": (
                    selection_reason_code
                )
            }

    for candidate_term_record in (
        candidate_term_records
    ):
        candidate_identity = (
            candidate_term_record[
                "candidate_identity"
            ]
        )

        candidate_selection = (
            selection_by_candidate_identity.get(
                candidate_identity
            )
        )

        if candidate_selection is None:
            raise RuntimeError(f"Candidate term references an unknown candidate : {candidate_identity}")

        if (
            candidate_selection[
                "selection_status"
            ]
            == "not_selected"
        ):
            candidate_term_record[
                "term_disposition"
            ] = "not_selected"

            candidate_term_record[
                "term_reason_code"
            ] = candidate_selection[
                "selection_reason_code"
            ]


# Building a unique record index
def build_unique_record_index(records: list[dict], identity_field: str,
                              record_description: str
) -> dict[str, dict]:
    records_by_identity = {}

    for record in records:
        record_identity = record.get(
            identity_field
        )

        if (
            not isinstance(
                record_identity,
                str
            )
            or not record_identity
        ):
            raise RuntimeError(f"{record_description} contains an invalid {identity_field}.")

        if record_identity in records_by_identity:
            raise RuntimeError(f"{record_description} contains a duplicate {identity_field} : {record_identity}")

        records_by_identity[
            record_identity
        ] = record

    return records_by_identity


# Confirming that one cell belongs to the declared design scope
def cell_matches_design_scope(cell_record: dict, candidate_record: dict) -> bool:
    design_scope = candidate_record[
        "design_scope"
    ]

    if "/" in design_scope:
        (
            dataset_stage,
            scope_component
        ) = design_scope.split(
            "/",
            maxsplit = 1
        )

        expected_prefix = (
            "sampling_method="
        )

        if not scope_component.startswith(
            expected_prefix
        ):
            raise RuntimeError(f"Candidate contains an unsupported design scope : {design_scope}")

        required_sampling_method = (
            scope_component[
                len(
                    expected_prefix
                ):
            ]
        )

        return (
            cell_record[
                "dataset_stage"
            ]
            == dataset_stage
            and cell_record[
                "sampling_method"
            ]
            == required_sampling_method
        )

    if design_scope not in {
        "Stage_1",
        "Stage_2"
    }:
        raise RuntimeError(f"Candidate contains an unsupported design scope : {design_scope}")

    return (
        cell_record[
            "dataset_stage"
        ]
        == design_scope
    )


# Validating one serialized boolean or missing value
def validate_optional_boolean(value: str, field_name: str) -> None:
    if value not in {
        "",
        "true",
        "false"
    }:
        raise RuntimeError(f"CSV field contains an invalid optional boolean : {field_name} : {value!r}")


# Assembling candidate observations with explicit exclusion accounting
def assemble_candidate_observations(candidate_records: list[dict], modeling_policy: dict,
                                    cell_records: list[dict], reference_records: list[dict],
                                    support_membership_records: list[dict]
) -> tuple[list[dict], list[dict]]:
    cell_records_by_identity = (
        build_unique_record_index(
            records = cell_records,
            identity_field = "cell_identity",
            record_description = "Cell feature table"
        )
    )

    reference_records_by_identity = (
        build_unique_record_index(
            records = reference_records,
            identity_field = "source_identity",
            record_description = (
                "Reference-test feature table"
            )
        )
    )

    references_by_cell = defaultdict(
        list
    )

    for reference_record in (
        reference_records_by_identity.values()
    ):
        cell_identity = reference_record[
            "cell_identity"
        ]

        if cell_identity not in (
            cell_records_by_identity
        ):
            raise RuntimeError(
                "Reference-test record identifies an unknown cell : "
                f"{reference_record['source_identity']} : {cell_identity}"
            )

        references_by_cell[
            cell_identity
        ].append(
            reference_record
        )

    memberships_by_region = defaultdict(
        list
    )

    observed_membership_keys = set()

    for membership_record in (
        support_membership_records
    ):
        membership_key = (
            membership_record[
                "support_region_identity"
            ],
            membership_record[
                "cell_identity"
            ]
        )

        if membership_key in observed_membership_keys:
            raise RuntimeError(f"Support membership contains a duplicate key : {membership_key}")

        observed_membership_keys.add(
            membership_key
        )

        memberships_by_region[
            membership_record[
                "support_region_identity"
            ]
        ].append(
            membership_record
        )

    outcome_definitions = modeling_policy[
        "outcomes"
    ]

    horizon_definitions = modeling_policy[
        "horizon_axes"
    ]

    accounting_order = modeling_policy[
        "observation_accounting"
    ][
        "status_evaluation_order"
    ]

    if accounting_order != (
        OBSERVATION_ACCOUNTING_STATUSES
    ):
        raise RuntimeError("Observation-accounting order does not match the implemented precedence.")

    observation_records = []
    accounting_records = []

    for candidate_record in candidate_records:
        if candidate_record[
            "selection_status"
        ] != "selected":
            continue

        candidate_identity = candidate_record[
            "candidate_identity"
        ]

        region_identity = candidate_record[
            "support_region_identity"
        ]

        outcome_name = candidate_record[
            "outcome"
        ]

        cohort_name = candidate_record[
            "cohort"
        ]

        horizon_axis = candidate_record[
            "horizon_axis"
        ]

        if outcome_name not in outcome_definitions:
            raise RuntimeError(f"Candidate references an unknown outcome : {candidate_identity} : {outcome_name}")

        if horizon_axis not in horizon_definitions:
            raise RuntimeError(f"Candidate references an unknown horizon axis : {candidate_identity} : {horizon_axis}")

        if cohort_name not in {
            "primary",
            "sensitivity"
        }:
            raise RuntimeError(f"Candidate references an unknown cohort : {candidate_identity} : {cohort_name}")

        outcome_definition = outcome_definitions[
            outcome_name
        ]

        outcome_field = outcome_definition[
            "value_field"
        ]

        eligibility_field = outcome_definition[
            (
                "primary_eligibility_field"
                if cohort_name == "primary"
                else "sensitivity_eligibility_field"
            )
        ]

        member_records = []

        for membership_record in (
            memberships_by_region.get(
                region_identity,
                []
            )
        ):
            matching_design_fields = [
                "aging_mode",
                "outcome",
                "cohort",
                "horizon_axis"
            ]

            for field_name in matching_design_fields:
                if (
                    membership_record[
                        field_name
                    ]
                    != candidate_record[
                        field_name
                    ]
                ):
                    raise RuntimeError(
                        f"Support membership conflicts with its candidate definition : {candidate_identity} : {field_name}"
                    )

            cell_identity = membership_record[
                "cell_identity"
            ]

            cell_record = (
                cell_records_by_identity.get(
                    cell_identity
                )
            )

            if cell_record is None:
                raise RuntimeError(
                    f"Support membership identifies an unknown cell : {region_identity} : {cell_identity}"
                )

            matching_cell_fields = [
                "condition_protocol_identity",
                "dataset_stage",
                "laboratory",
                "sampling_method",
                "aging_mode"
            ]

            for field_name in matching_cell_fields:
                if (
                    membership_record[
                        field_name
                    ]
                    != cell_record[
                        field_name
                    ]
                ):
                    raise RuntimeError(
                        "Support membership conflicts with cell metadata : "
                        f"{region_identity} : {cell_identity} : {field_name}"
                    )

            if cell_matches_design_scope(
                cell_record = cell_record,
                candidate_record = candidate_record
            ):
                member_records.append(
                    membership_record
                )

        member_cell_identities = sorted(
            {
                membership_record[
                    "cell_identity"
                ]
                for membership_record in member_records
            }
        )

        expected_independent_cell_count = (
            parse_nonnegative_integer(
                candidate_record[
                    "support_member_cell_count"
                ],
                "support_member_cell_count"
            )
        )

        if (
            len(
                member_cell_identities
            )
            != expected_independent_cell_count
        ):
            raise RuntimeError(
                "Candidate member-cell count does not match the analytical scope : "
                f"{candidate_identity} : {len(member_cell_identities)} : {expected_independent_cell_count}"
            )

        observed_condition_count = len(
            {
                cell_records_by_identity[
                    cell_identity
                ][
                    "condition_protocol_identity"
                ]
                for cell_identity in (
                    member_cell_identities
                )
            }
        )

        expected_condition_count = (
            parse_nonnegative_integer(
                candidate_record[
                    "support_condition_count"
                ],
                "support_condition_count"
            )
        )

        if (
            observed_condition_count
            != expected_condition_count
        ):
            raise RuntimeError(
                "Candidate condition count does not match the analytical scope : "
                f"{candidate_identity} : {observed_condition_count} : {expected_condition_count}"
            )

        status_counts = Counter()

        candidate_observations = []

        observed_source_identities = set()

        evaluation_interval_counts = Counter()

        for cell_identity in member_cell_identities:
            cell_record = cell_records_by_identity[
                cell_identity
            ]

            for reference_record in sorted(
                references_by_cell[
                    cell_identity
                ],
                key = lambda record: record[
                    "source_identity"
                ]
            ):
                source_identity = reference_record[
                    "source_identity"
                ]

                if source_identity in observed_source_identities:
                    raise RuntimeError(
                        "Candidate source universe contains a duplicate source identity : "
                        f"{candidate_identity} : {source_identity}"
                    )

                observed_source_identities.add(
                    source_identity
                )

                horizon_text = reference_record[
                    horizon_axis
                ]

                outcome_text = reference_record[
                    outcome_field
                ]

                eligibility_text = reference_record[
                    eligibility_field
                ]

                validate_optional_boolean(
                    eligibility_text,
                    eligibility_field
                )

                if horizon_text == "":
                    status_name = "missing_horizon"
                    horizon_value = None
                    outcome_value = None

                else:
                    horizon_value = parse_finite_number(
                        horizon_text,
                        horizon_axis
                    )

                    if (
                        horizon_value
                        <= horizon_definitions[
                            horizon_axis
                        ][
                            "minimum_value_exclusive"
                        ]
                    ):
                        status_name = "nonpositive_horizon"
                        outcome_value = None

                    elif outcome_text == "":
                        status_name = "missing_outcome"
                        outcome_value = None

                    else:
                        outcome_value = parse_finite_number(
                            outcome_text,
                            outcome_field
                        )

                        if eligibility_text != "true":
                            status_name = "cohort_ineligible"

                        elif (
                            horizon_axis
                            == (
                                "cumulative_full_equivalent_cycles_observed"
                            )
                        ):
                            completeness_text = (
                                reference_record[
                                    "cumulative_fec_observation_complete"
                                ]
                            )

                            validate_optional_boolean(
                                completeness_text,
                                (
                                    "cumulative_fec_"
                                    "observation_complete"
                                )
                            )

                            if completeness_text != "true":
                                status_name = (
                                    "incomplete_cumulative_fec"
                                )

                            else:
                                status_name = "included"

                        else:
                            status_name = "included"

                inside_evaluation_interval = (
                    status_name == "included"
                    and horizon_inside_candidate_support(
                        horizon_value = horizon_value,
                        candidate_record = candidate_record
                    )
                )

                status_counts[
                    status_name
                ] += 1

                if status_name == "included":
                    evaluation_interval_counts[
                        (
                            "inside"
                            if inside_evaluation_interval
                            else "outside"
                        )
                    ] += 1

                if status_name != "included":
                    continue

                observation_record = {
                    "candidate_identity": candidate_identity,
                    "support_region_identity": region_identity,
                    "source_identity": source_identity,
                    "cell_identity": cell_identity,
                    "condition_protocol_identity": (
                        cell_record[
                            "condition_protocol_identity"
                        ]
                    ),
                    "dataset_stage": (
                        cell_record[
                            "dataset_stage"
                        ]
                    ),
                    "aging_mode": (
                        candidate_record[
                            "aging_mode"
                        ]
                    ),
                    "outcome": outcome_name,
                    "cohort": cohort_name,
                    "horizon_axis": horizon_axis,
                    "horizon": horizon_value,
                    "outcome_value": outcome_value,
                    **{
                        field_name: (
                            cell_record[
                                field_name
                            ]
                        )
                        for field_name in (
                            CELL_MODEL_FIELDS
                        )
                    },
                    "inside_evaluation_interval": (
                        inside_evaluation_interval
                    )
                }

                candidate_observations.append(
                    observation_record
                )

        source_record_count = sum(
            status_counts.values()
        )

        included_record_count = status_counts[
            "included"
        ]

        excluded_record_count = (
            source_record_count
            - included_record_count
        )

        inside_evaluation_interval_count = (
            evaluation_interval_counts[
                "inside"
            ]
        )

        outside_evaluation_interval_count = (
            evaluation_interval_counts[
                "outside"
            ]
        )

        if (
            inside_evaluation_interval_count
            + outside_evaluation_interval_count
            != included_record_count
        ):
            raise RuntimeError(f"Evaluation-interval accounting does not reconcile : {candidate_identity}")

        if source_record_count != len(
            observed_source_identities
        ):
            raise RuntimeError(f"Observation-accounting counts do not reconcile : {candidate_identity}")

        cell_observation_counts = Counter(
            observation_record[
                "cell_identity"
            ]
            for observation_record in (
                candidate_observations
            )
        )

        for observation_record in (
            candidate_observations
        ):
            observation_record[
                "cell_candidate_observation_count"
            ] = cell_observation_counts[
                observation_record[
                    "cell_identity"
                ]
            ]

        candidate_observations.sort(
            key = lambda record: (
                record["cell_identity"],
                record["horizon"],
                record["source_identity"]
            )
        )

        observation_records.extend(
            candidate_observations
        )

        accounting_record = {
            "candidate_identity": candidate_identity,
            "support_region_identity": region_identity,
            "source_record_count": source_record_count,
            "included_record_count": (
                included_record_count
            ),
            "excluded_record_count": (
                excluded_record_count
            ),
            **{
                f"{status_name}_count": (
                    status_counts[
                        status_name
                    ]
                )
                for status_name in (
                    OBSERVATION_ACCOUNTING_STATUSES
                )
                if status_name != "included"
            },
            "inside_evaluation_interval_count": (
                inside_evaluation_interval_count
            ),
            "outside_evaluation_interval_count": (
                outside_evaluation_interval_count
            )
        }

        accounting_records.append(
            accounting_record
        )

        candidate_record[
            "candidate_source_record_count"
        ] = source_record_count

        candidate_record[
            "candidate_included_record_count"
        ] = included_record_count

        candidate_record[
            "candidate_excluded_record_count"
        ] = excluded_record_count

        if included_record_count == 0:
            candidate_record[
                "observation_assembly_status"
            ] = "not_assembled"

            candidate_record[
                "admissibility_status"
            ] = "not_assembled"

            candidate_record[
                "admissibility_reason_code"
            ] = "NO_ELIGIBLE_OBSERVATIONS"

        else:
            candidate_record[
                "observation_assembly_status"
            ] = "completed"

            candidate_record[
                "admissibility_reason_code"
            ] = (
                candidate_record[
                    "interaction_hierarchy_reason_code"
                ]
            )

    observation_keys = [
        (
            observation_record[
                "candidate_identity"
            ],
            observation_record[
                "source_identity"
            ]
        )
        for observation_record in observation_records
    ]

    if len(observation_keys) != len(
        set(
            observation_keys
        )
    ):
        raise RuntimeError("Assembled observations contain duplicate candidate/source identities.")

    return (
        observation_records,
        sorted(
            accounting_records,
            key = lambda record: record[
                "candidate_identity"
            ]
        )
    )


# Representing an expected candidate-specific matrix limitation
class CandidateMatrixUnavailableError(RuntimeError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(
            message
        )

        self.reason_code = reason_code


# Building normalized equal-cell observation weights
def build_equal_cell_weights(observation_records: list[dict]) -> np.ndarray:
    if not observation_records:
        raise CandidateMatrixUnavailableError(
            reason_code = "NO_ELIGIBLE_OBSERVATIONS",
            message = (
                "Cannot build weights without observations."
            )
        )

    observation_counts_by_cell = Counter(
        observation_record[
            "cell_identity"
        ]
        for observation_record in observation_records
    )

    weights = np.asarray(
        [
            1.0
            / observation_counts_by_cell[
                observation_record[
                    "cell_identity"
                ]
            ]
            for observation_record in (
                observation_records
            )
        ],
        dtype = float
    )

    if (
        not np.all(
            np.isfinite(
                weights
            )
        )
        or np.any(
            weights <= 0
        )
    ):
        raise RuntimeError("Equal-cell weighting produced invalid weights.")

    mean_weight = float(
        np.mean(
            weights
        )
    )

    if (
        not math.isfinite(
            mean_weight
        )
        or mean_weight <= 0
    ):
        raise RuntimeError("Equal-cell weighting produced an invalid mean weight.")

    return (
        weights
        / mean_weight
    )


# Calculating an equal-cell weighted location and scale
def calculate_weighted_location_scale(values: np.ndarray, weights: np.ndarray,
                                      field_name: str
) -> tuple[float, float]:
    if (
        values.ndim != 1
        or weights.ndim != 1
        or len(values) != len(weights)
        or not len(values)
    ):
        raise RuntimeError(f"Weighted location and scale require matching one-dimensional arrays : {field_name}")

    if (
        not np.all(
            np.isfinite(
                values
            )
        )
        or not np.all(
            np.isfinite(
                weights
            )
        )
        or np.any(
            weights <= 0
        )
    ):
        raise RuntimeError(f"Weighted location and scale received invalid values : {field_name}")

    total_weight = float(
        np.sum(
            weights
        )
    )

    mean_value = float(
        np.sum(
            weights
            * values
        )
        / total_weight
    )

    variance = float(
        np.sum(
            weights
            * (
                values
                - mean_value
            ) ** 2
        )
        / total_weight
    )

    if variance < 0:
        raise RuntimeError(f"Weighted variance is negative : {field_name} : {variance}")

    standard_deviation = math.sqrt(
        variance
    )

    if (
        not math.isfinite(
            standard_deviation
        )
        or standard_deviation <= 0
    ):
        raise CandidateMatrixUnavailableError(
            reason_code = "ZERO_VARIANCE_MODEL_FIELD",
            message = (
                f"Model field does not vary within the candidate : {field_name}")
        )

    return (
        mean_value,
        standard_deviation
    )


# Calculating one deterministic weighted quantile
def calculate_weighted_quantile(values: np.ndarray, weights: np.ndarray,
                                observation_records: list[dict], probability: float
) -> float:
    if (
        values.ndim != 1
        or weights.ndim != 1
        or len(values) != len(weights)
        or len(values) != len(
            observation_records
        )
        or not len(values)
    ):
        raise RuntimeError("Weighted quantile requires matching non-empty inputs.")

    if (
        not math.isfinite(
            probability
        )
        or not (
            0.0
            <= probability
            <= 1.0
        )
    ):
        raise ValueError("Weighted-quantile probability must be between zero and one.")

    ordered_indices = sorted(
        range(
            len(
                values
            )
        ),
        key = lambda row_index: (
            float(
                values[
                    row_index
                ]
            ),
            observation_records[
                row_index
            ][
                "cell_identity"
            ],
            observation_records[
                row_index
            ][
                "source_identity"
            ]
        )
    )

    total_weight = float(
        np.sum(
            weights
        )
    )

    if (
        not math.isfinite(
            total_weight
        )
        or total_weight <= 0
    ):
        raise RuntimeError("Weighted quantile received an invalid total weight.")

    if probability == 0.0:
        return float(
            values[
                ordered_indices[0]
            ]
        )

    cumulative_weight = 0.0

    for row_index in ordered_indices:
        cumulative_weight += float(
            weights[
                row_index
            ]
        )

        if (
            cumulative_weight
            / total_weight
            >= probability
        ):
            return float(
                values[
                    row_index
                ]
            )

    return float(
        values[
            ordered_indices[-1]
        ]
    )


# Ordering authorized source terms
def sort_candidate_terms(candidate_term_records: list[dict], modeling_policy: dict) -> list[dict]:
    source_term_order = modeling_policy[
        "candidate_assembly"
    ][
        "source_term_sort_order"
    ]

    order_by_term_type = {
        term_type: order_index
        for order_index, term_type in enumerate(
            source_term_order
        )
    }

    unknown_term_types = sorted(
        {
            candidate_term_record[
                "term_type"
            ]
            for candidate_term_record in (
                candidate_term_records
            )
        }.difference(
            order_by_term_type
        )
    )

    if unknown_term_types:
        raise RuntimeError(f"Candidate contains unsupported source-term types : {unknown_term_types}")

    return sorted(
        candidate_term_records,
        key = lambda record: (
            order_by_term_type[
                record[
                    "term_type"
                ]
            ],
            record[
                "term_name"
            ]
        )
    )


# Reading one finite numerical model field
def extract_numerical_model_values(observation_records: list[dict], field_name: str) -> np.ndarray:
    if field_name not in NUMERICAL_MODEL_FIELDS:
        raise RuntimeError(f"Source term does not identify a supported numerical field : {field_name}")

    values = []

    for observation_record in (
        observation_records
    ):
        raw_value = observation_record.get(
            field_name
        )

        if (
            not isinstance(
                raw_value,
                str
            )
            or raw_value == ""
        ):
            raise CandidateMatrixUnavailableError(
                reason_code = "MISSING_MODEL_FIELD",
                message = (
                    f"Observation is missing a required numerical model field : {field_name}"
                )
            )

        values.append(
            parse_finite_number(
                raw_value,
                field_name
            )
        )

    return np.asarray(
        values,
        dtype = float
    )


# Parsing one categorical indicator term
def parse_indicator_term(term_name: str) -> tuple[str, str]:
    if "=" not in term_name:
        raise RuntimeError(f"Categorical indicator term does not contain a level : {term_name}")

    (
        field_name,
        level_name
    ) = term_name.split(
        "=",
        maxsplit = 1
    )

    if (
        not field_name
        or not level_name
    ):
        raise RuntimeError(f"Categorical indicator term is invalid : {term_name}")

    return (
        field_name,
        level_name
    )


# Building the fitted encoding definition for one candidate
def build_model_encoding(candidate_record: dict, candidate_term_records: list[dict],
                         observation_records: list[dict], modeling_policy: dict
) -> dict:
    if not observation_records:
        raise CandidateMatrixUnavailableError(
            reason_code = "NO_ELIGIBLE_OBSERVATIONS",
            message = (
                "Candidate does not contain eligible observations."
            )
        )

    equal_cell_weights = build_equal_cell_weights(
        observation_records
    )

    horizon_values = np.asarray(
        [
            observation_record[
                "horizon"
            ]
            for observation_record in (
                observation_records
            )
        ],
        dtype = float
    )

    (
        horizon_mean,
        horizon_standard_deviation
    ) = calculate_weighted_location_scale(
        values = horizon_values,
        weights = equal_cell_weights,
        field_name = "horizon"
    )

    ordered_candidate_terms = sort_candidate_terms(
        candidate_term_records = (
            candidate_term_records
        ),
        modeling_policy = modeling_policy
    )

    source_term_encodings = {}

    for candidate_term_record in (
        ordered_candidate_terms
    ):
        term_name = candidate_term_record[
            "term_name"
        ]

        term_type = candidate_term_record[
            "term_type"
        ]

        if term_type == "pairwise_interaction":
            interaction_components = (
                term_name.split(
                    ":"
                )
            )

            if len(interaction_components) != 2:
                raise RuntimeError(
                    f"Pairwise interaction does not contain exactly two components : {term_name}"
                )

            source_term_encodings[
                term_name
            ] = {
                "encoding_type": "pairwise_interaction",
                "components": interaction_components
            }

            continue

        if "=" in term_name:
            (
                field_name,
                level_name
            ) = parse_indicator_term(
                term_name
            )

            observed_levels = sorted(
                {
                    observation_record.get(
                        field_name,
                        ""
                    )
                    for observation_record in (
                        observation_records
                    )
                }
            )

            if "" in observed_levels:
                raise CandidateMatrixUnavailableError(
                    reason_code = "MISSING_MODEL_CATEGORY",
                    message = (
                        f"Observation is missing a required categorical model field : {field_name}"
                    )
                )

            if level_name not in observed_levels:
                raise CandidateMatrixUnavailableError(
                    reason_code = "AUTHORIZED_CATEGORY_LEVEL_UNOBSERVED",
                    message = (
                        f"Authorized categorical level is not observed within the candidate : {term_name}"
                    )
                )

            source_term_encodings[
                term_name
            ] = {
                "encoding_type": "categorical_indicator",
                "field_name": field_name,
                "level_name": level_name,
                "observed_levels": observed_levels
            }

            continue

        numerical_values = (
            extract_numerical_model_values(
                observation_records = (
                    observation_records
                ),
                field_name = term_name
            )
        )

        (
            source_mean,
            source_standard_deviation
        ) = calculate_weighted_location_scale(
            values = numerical_values,
            weights = equal_cell_weights,
            field_name = term_name
        )

        source_term_encodings[
            term_name
        ] = {
            "encoding_type": "standardized_numerical",
            "field_name": term_name,
            "mean": source_mean,
            "standard_deviation": (
                source_standard_deviation
            )
        }

    permitted_adjustment_variables = (
        parse_json_field(
            candidate_record[
                "permitted_adjustment_variables"
            ],
            "permitted_adjustment_variables"
        )
    )

    if (
        not isinstance(
            permitted_adjustment_variables,
            list
        )
        or any(
            not isinstance(
                field_name,
                str
            )
            or not field_name
            for field_name in (
                permitted_adjustment_variables
            )
        )
    ):
        raise RuntimeError("Candidate contains invalid permitted adjustment variables.")

    if (
        candidate_record[
            "context_family"
        ]
        == "no_additional_context"
        and permitted_adjustment_variables
    ):
        raise RuntimeError("Candidate without additional context declares adjustment variables.")

    if (
        candidate_record[
            "context_family"
        ]
        == "laboratory"
        and permitted_adjustment_variables
        != [
            "laboratory"
        ]
    ):
        raise RuntimeError("Laboratory-context candidate does not declare exactly the laboratory adjustment.")

    context_encodings = []

    for field_name in sorted(
        permitted_adjustment_variables
    ):
        observed_levels = sorted(
            {
                observation_record.get(
                    field_name,
                    ""
                )
                for observation_record in (
                    observation_records
                )
            }
        )

        if "" in observed_levels:
            raise CandidateMatrixUnavailableError(
                reason_code = "MISSING_CONTEXT_CATEGORY",
                message = (
                    f"Observation is missing a required context category : {field_name}"
                )
            )

        if len(observed_levels) < 2:
            raise CandidateMatrixUnavailableError(
                reason_code = "CONTEXT_TERM_NOT_IDENTIFIABLE",
                message = (
                    f"Context field does not vary within the candidate : {field_name}"
                )
            )

        context_encodings.append(
            {
                "field_name": field_name,
                "reference_level": (
                    observed_levels[0]
                ),
                "levels": observed_levels
            }
        )

    horizon_basis = candidate_record[
        "horizon_basis"
    ]

    spline_knots = None
    unique_horizon_count = len(
        set(
            horizon_values.tolist()
        )
    )
    non_boundary_unique_horizon_count = None

    if horizon_basis == "restricted_cubic_spline":
        spline_policy = modeling_policy[
            "restricted_cubic_spline"
        ]

        if (
            unique_horizon_count
            < spline_policy[
                "minimum_unique_horizon_count"
            ]
        ):
            raise CandidateMatrixUnavailableError(
                reason_code = "INSUFFICIENT_UNIQUE_SPLINE_HORIZONS",
                message = (
                    "Candidate does not contain enough unique horizons for its spline basis."
                )
            )

        spline_knots = [
            calculate_weighted_quantile(
                values = horizon_values,
                weights = equal_cell_weights,
                observation_records = (
                    observation_records
                ),
                probability = probability
            )
            for probability in spline_policy[
                "knot_probabilities"
            ]
        ]

        if not (
            spline_knots[0]
            < spline_knots[1]
            < spline_knots[2]
        ):
            raise CandidateMatrixUnavailableError(
                reason_code = "NON_DISTINCT_SPLINE_KNOTS",
                message = (
                    "Candidate spline knots are not strictly increasing."
                )
            )

        non_boundary_unique_horizon_count = len(
            {
                horizon_value
                for horizon_value in (
                    horizon_values.tolist()
                )
                if (
                    spline_knots[0]
                    < horizon_value
                    < spline_knots[2]
                )
            }
        )

        if (
            non_boundary_unique_horizon_count
            < spline_policy[
                "minimum_non_boundary_unique_horizon_count"
            ]
        ):
            raise CandidateMatrixUnavailableError(
                reason_code = (
                    "INSUFFICIENT_INTERIOR_SPLINE_HORIZONS"
                ),
                message = (
                    "Candidate does not contain enough unique interior spline horizons."
                )
            )

    elif horizon_basis != "linear":
        raise RuntimeError(f"Candidate contains an unsupported horizon basis : {horizon_basis}")

    return {
        "horizon_basis": horizon_basis,
        "horizon_mean": horizon_mean,
        "horizon_standard_deviation": (
            horizon_standard_deviation
        ),
        "spline_knots": spline_knots,
        "unique_horizon_count": unique_horizon_count,
        "non_boundary_unique_horizon_count": (
            non_boundary_unique_horizon_count
        ),
        "ordered_source_terms": [
            {
                "term_name": term_record[
                    "term_name"
                ],
                "term_type": term_record[
                    "term_type"
                ]
            }
            for term_record in (
                ordered_candidate_terms
            )
        ],
        "source_term_encodings": (
            source_term_encodings
        ),
        "context_encodings": context_encodings
    }


# Transforming horizons into the configured basis
def transform_horizon_basis(horizon_values: np.ndarray, model_encoding: dict) -> tuple[list[np.ndarray], list[str]]:
    standardized_horizon = (
        horizon_values
        - model_encoding[
            "horizon_mean"
        ]
    ) / model_encoding[
        "horizon_standard_deviation"
    ]

    if model_encoding[
        "horizon_basis"
    ] == "linear":
        return (
            [
                standardized_horizon
            ],
            [
                "horizon_linear"
            ]
        )

    (
        lower_knot,
        middle_knot,
        upper_knot
    ) = model_encoding[
        "spline_knots"
    ]

    knot_range = (
        upper_knot
        - lower_knot
    )

    nonlinear_component = (
        (
            np.maximum(
                horizon_values
                - lower_knot,
                0.0
            ) ** 3
            - np.maximum(
                horizon_values
                - middle_knot,
                0.0
            ) ** 3
            * (
                upper_knot
                - lower_knot
            )
            / (
                upper_knot
                - middle_knot
            )
            + np.maximum(
                horizon_values
                - upper_knot,
                0.0
            ) ** 3
            * (
                middle_knot
                - lower_knot
            )
            / (
                upper_knot
                - middle_knot
            )
        )
        / knot_range ** 3
    )

    return (
        [
            standardized_horizon,
            nonlinear_component
        ],
        [
            "horizon_linear",
            "horizon_restricted_cubic"
        ]
    )


# Transforming one encoded source term
def transform_source_term(term_name: str, observation_records: list[dict],
                          model_encoding: dict, transformed_source_terms: dict[str, np.ndarray]
) -> np.ndarray:
    existing_column = transformed_source_terms.get(
        term_name
    )

    if existing_column is not None:
        return existing_column

    term_encoding = model_encoding[
        "source_term_encodings"
    ][
        term_name
    ]

    encoding_type = term_encoding[
        "encoding_type"
    ]

    if encoding_type == "standardized_numerical":
        field_name = term_encoding[
            "field_name"
        ]

        raw_values = extract_numerical_model_values(
            observation_records = observation_records,
            field_name = field_name
        )

        transformed_column = (
            raw_values
            - term_encoding[
                "mean"
            ]
        ) / term_encoding[
            "standard_deviation"
        ]

    elif encoding_type == "categorical_indicator":
        field_name = term_encoding[
            "field_name"
        ]

        level_name = term_encoding[
            "level_name"
        ]

        observed_levels = set(
            term_encoding[
                "observed_levels"
            ]
        )

        source_values = []

        for observation_record in (
            observation_records
        ):
            source_value = observation_record.get(
                field_name,
                ""
            )

            if source_value == "":
                raise CandidateMatrixUnavailableError(
                    reason_code = "MISSING_MODEL_CATEGORY",
                    message = (
                        f"Prediction record is missing a required category : {field_name}"
                    )
                )

            if source_value not in observed_levels:
                raise CandidateMatrixUnavailableError(
                    reason_code = "UNKNOWN_MODEL_CATEGORY",
                    message = (
                        f"Prediction record contains an unrecognized category : {field_name} : {source_value}"
                    )
                )

            source_values.append(
                1.0
                if source_value == level_name
                else 0.0
            )

        transformed_column = np.asarray(
            source_values,
            dtype = float
        )

    elif encoding_type == "pairwise_interaction":
        (
            first_component,
            second_component
        ) = term_encoding[
            "components"
        ]

        first_column = transform_source_term(
            term_name = first_component,
            observation_records = observation_records,
            model_encoding = model_encoding,
            transformed_source_terms = (
                transformed_source_terms
            )
        )

        second_column = transform_source_term(
            term_name = second_component,
            observation_records = observation_records,
            model_encoding = model_encoding,
            transformed_source_terms = (
                transformed_source_terms
            )
        )

        transformed_column = (
            first_column
            * second_column
        )

    else:
        raise RuntimeError(f"Unknown source-term encoding : {encoding_type}")

    transformed_source_terms[
        term_name
    ] = transformed_column

    return transformed_column


# Transforming observations into one complete model matrix
def transform_model_matrix(candidate_record: dict, observation_records: list[dict],
                           model_encoding: dict
) -> tuple[np.ndarray, list[dict]]:
    horizon_values = np.asarray(
        [
            observation_record[
                "horizon"
            ]
            for observation_record in (
                observation_records
            )
        ],
        dtype = float
    )

    (
        horizon_columns,
        horizon_column_names
    ) = transform_horizon_basis(
        horizon_values = horizon_values,
        model_encoding = model_encoding
    )

    matrix_columns = [
        np.ones(
            len(
                observation_records
            ),
            dtype = float
        ),
        *horizon_columns
    ]

    matrix_terms = [
        {
            "column_name": "intercept",
            "term_type": "intercept",
            "source_term_name": None,
            "horizon_basis_name": None
        },
        *[
            {
                "column_name": column_name,
                "term_type": "horizon_basis",
                "source_term_name": None,
                "horizon_basis_name": column_name
            }
            for column_name in (
                horizon_column_names
            )
        ]
    ]

    transformed_source_terms = {}

    for source_term_record in model_encoding[
        "ordered_source_terms"
    ]:
        term_name = source_term_record[
            "term_name"
        ]

        transformed_column = transform_source_term(
            term_name = term_name,
            observation_records = observation_records,
            model_encoding = model_encoding,
            transformed_source_terms = (
                transformed_source_terms
            )
        )

        matrix_columns.append(
            transformed_column
        )

        matrix_terms.append(
            {
                "column_name": term_name,
                "term_type": source_term_record[
                    "term_type"
                ],
                "source_term_name": term_name,
                "horizon_basis_name": None
            }
        )

    for context_encoding in model_encoding[
        "context_encodings"
    ]:
        field_name = context_encoding[
            "field_name"
        ]

        observed_levels = set(
            context_encoding[
                "levels"
            ]
        )

        for level_name in context_encoding[
            "levels"
        ][1:]:
            context_values = []

            for observation_record in (
                observation_records
            ):
                source_value = observation_record.get(
                    field_name,
                    ""
                )

                if source_value == "":
                    raise CandidateMatrixUnavailableError(
                        reason_code = "MISSING_CONTEXT_CATEGORY",
                        message = (
                            f"Prediction record is missing a required context category : {field_name}"
                        )
                    )

                if source_value not in observed_levels:
                    raise CandidateMatrixUnavailableError(
                        reason_code = "UNKNOWN_CONTEXT_CATEGORY",
                        message = (
                            f"Prediction record contains an unrecognized context category : {field_name} : {source_value}"
                        )
                    )

                context_values.append(
                    1.0
                    if source_value == level_name
                    else 0.0
                )

            column_name = (
                f"{field_name} = {level_name}"
            )

            matrix_columns.append(
                np.asarray(
                    context_values,
                    dtype = float
                )
            )

            matrix_terms.append(
                {
                    "column_name": column_name,
                    "term_type": "design_context",
                    "source_term_name": field_name,
                    "horizon_basis_name": None
                }
            )

    if candidate_record[
        "include_horizon_by_treatment_interactions"
    ]:
        source_terms_by_name = {
            source_term_record[
                "term_name"
            ]: source_term_record
            for source_term_record in (
                model_encoding[
                    "ordered_source_terms"
                ]
            )
        }

        for (
            horizon_column,
            horizon_column_name
        ) in zip(
            horizon_columns,
            horizon_column_names,
            strict = True
        ):
            for term_name in sorted(
                source_terms_by_name,
                key = lambda source_name: (
                    [
                        "main_effect",
                        "target_treatment",
                        "pairwise_interaction"
                    ].index(
                        source_terms_by_name[
                            source_name
                        ][
                            "term_type"
                        ]
                    ),
                    source_name
                )
            ):
                source_column = (
                    transformed_source_terms[
                        term_name
                    ]
                )

                interaction_column_name = (
                    f"{horizon_column_name}:"
                    f"{term_name}"
                )

                matrix_columns.append(
                    horizon_column
                    * source_column
                )

                matrix_terms.append(
                    {
                        "column_name": (
                            interaction_column_name
                        ),
                        "term_type": (
                            "horizon_by_treatment_interaction"
                        ),
                        "source_term_name": term_name,
                        "horizon_basis_name": (
                            horizon_column_name
                        )
                    }
                )

    column_names = [
        matrix_term[
            "column_name"
        ]
        for matrix_term in matrix_terms
    ]

    if len(column_names) != len(
        set(
            column_names
        )
    ):
        raise RuntimeError("Model matrix contains duplicate column names.")

    model_matrix = np.column_stack(
        matrix_columns
    )

    if model_matrix.shape != (
        len(
            observation_records
        ),
        len(
            matrix_terms
        )
    ):
        raise RuntimeError("Model matrix has an inconsistent shape.")

    return (
        model_matrix,
        matrix_terms
    )


# Describing within-cell and condition-specific horizon replication
def build_longitudinal_structure_diagnostics(observation_records: list[dict]) -> dict:
    horizons_by_cell = defaultdict(set)
    cells_by_condition = defaultdict(set)
    horizons_by_condition = defaultdict(set)
    repeated_cells_by_condition = defaultdict(set)

    for observation_record in observation_records:
        cell_identity = observation_record["cell_identity"]
        condition_identity = observation_record["condition_protocol_identity"]
        horizon_value = observation_record["horizon"]

        horizons_by_cell[cell_identity].add(horizon_value)
        cells_by_condition[condition_identity].add(cell_identity)
        horizons_by_condition[condition_identity].add(horizon_value)

    repeated_cell_identities = {
        cell_identity
        for cell_identity, horizon_values in horizons_by_cell.items()
        if len(horizon_values) >= 2
    }

    for condition_identity, cell_identities in cells_by_condition.items():
        repeated_cells_by_condition[condition_identity] = {
            cell_identity
            for cell_identity in cell_identities
            if cell_identity in repeated_cell_identities
        }

    independent_cell_count = len(horizons_by_cell)
    repeated_cell_count = len(repeated_cell_identities)

    if repeated_cell_count == 0:
        longitudinal_structure_status = "cross_sectional"
    elif repeated_cell_count == independent_cell_count:
        longitudinal_structure_status = "fully_longitudinal"
    else:
        longitudinal_structure_status = "partially_longitudinal"

    return {
        "repeated_cell_count": repeated_cell_count,
        "single_observation_cell_count": (
            independent_cell_count - repeated_cell_count
        ),
        "minimum_condition_unique_horizon_count": min(
            (
                len(horizon_values)
                for horizon_values in horizons_by_condition.values()
            ),
            default = 0
        ),
        "minimum_condition_repeated_cell_count": min(
            (
                len(cell_identities)
                for cell_identities in repeated_cells_by_condition.values()
            ),
            default = 0
        ),
        "longitudinal_structure_status": longitudinal_structure_status
    }


# Evaluating the complete numerical admissibility contract
def evaluate_candidate_admissibility(candidate_record: dict, observation_records: list[dict],
                                     model_matrix: np.ndarray, outcome_values: np.ndarray,
                                     regression_weights: np.ndarray, model_encoding: dict,
                                     modeling_policy: dict
) -> dict:
    admissibility_policy = modeling_policy[
        "admissibility"
    ]

    independent_cell_count = len(
        {
            observation_record[
                "cell_identity"
            ]
            for observation_record in (
                observation_records
            )
        }
    )

    condition_protocol_count = len(
        {
            observation_record[
                "condition_protocol_identity"
            ]
            for observation_record in (
                observation_records
            )
        }
    )

    longitudinal_diagnostics = build_longitudinal_structure_diagnostics(
        observation_records
    )

    diagnostic = {
        "observation_count": len(
            observation_records
        ),
        "independent_cell_count": (
            independent_cell_count
        ),
        "condition_protocol_count": (
            condition_protocol_count
        ),
        "matrix_column_count": int(
            model_matrix.shape[1]
        ),
        "matrix_rank": None,
        "rank_tolerance": None,
        "largest_singular_value": None,
        "smallest_singular_value": None,
        "smallest_retained_singular_value": None,
        "effective_condition_number": None,
        "residual_independent_cell_degrees_of_freedom": None,
        "unique_horizon_count": (
            model_encoding[
                "unique_horizon_count"
            ]
        ),
        "non_boundary_unique_horizon_count": (
            model_encoding[
                "non_boundary_unique_horizon_count"
            ]
        ),
        "admissibility_status": "inadmissible",
        "admissibility_reason_code": None,
        "repeated_cell_count": (
            longitudinal_diagnostics["repeated_cell_count"]
        ),
        "single_observation_cell_count": (
            longitudinal_diagnostics["single_observation_cell_count"]
        ),
        "minimum_condition_unique_horizon_count": (
            longitudinal_diagnostics["minimum_condition_unique_horizon_count"]
        ),
        "minimum_condition_repeated_cell_count": (
            longitudinal_diagnostics["minimum_condition_repeated_cell_count"]
        ),
        "longitudinal_structure_status": (
            longitudinal_diagnostics["longitudinal_structure_status"]
        )
    }

    longitudinal_policy = admissibility_policy["longitudinal_structure"]
    repeated_cell_count = longitudinal_diagnostics["repeated_cell_count"]

    if (
        candidate_record["estimator"] == "linear_mixed_effects"
        and repeated_cell_count
        < longitudinal_policy["minimum_repeated_cells_for_random_intercept"]
    ):
        diagnostic["admissibility_reason_code"] = (
            "RANDOM_INTERCEPT_NOT_STRUCTURALLY_IDENTIFIABLE"
        )
        return diagnostic

    if candidate_record["include_horizon_by_treatment_interactions"]:
        if (
            longitudinal_diagnostics["minimum_condition_unique_horizon_count"]
            < longitudinal_policy[
                "differential_trajectory_minimum_unique_horizons_per_condition"
            ]
        ):
            diagnostic["admissibility_reason_code"] = (
                "INSUFFICIENT_CONDITION_HORIZON_VARIATION"
            )
            return diagnostic

        if (
            longitudinal_diagnostics["minimum_condition_repeated_cell_count"]
            < longitudinal_policy[
                "differential_trajectory_minimum_repeated_cells_per_condition"
            ]
        ):
            diagnostic["admissibility_reason_code"] = (
                "INSUFFICIENT_CONDITION_REPEATED_CELLS"
            )
            return diagnostic

    elif (
        repeated_cell_count
        < longitudinal_policy["common_trajectory_minimum_repeated_cells"]
    ):
        diagnostic["admissibility_reason_code"] = "INSUFFICIENT_REPEATED_CELLS"
        return diagnostic

    if (
        condition_protocol_count
        < admissibility_policy[
            "required_condition_protocol_count"
        ]
    ):
        diagnostic[
            "admissibility_reason_code"
        ] = "INSUFFICIENT_CONDITION_PROTOCOLS"

        return diagnostic

    if (
        independent_cell_count
        < admissibility_policy[
            "minimum_independent_cells"
        ]
    ):
        diagnostic[
            "admissibility_reason_code"
        ] = "INSUFFICIENT_INDEPENDENT_CELLS"

        return diagnostic

    if (
        admissibility_policy[
            "required_outcome_variation"
        ]
        and not (
            float(
                np.max(
                    outcome_values
                )
            )
            > float(
                np.min(
                    outcome_values
                )
            )
        )
    ):
        diagnostic[
            "admissibility_reason_code"
        ] = "OUTCOME_DOES_NOT_VARY"

        return diagnostic

    if (
        not np.all(
            np.isfinite(
                model_matrix
            )
        )
        or not np.all(
            np.isfinite(
                outcome_values
            )
        )
        or not np.all(
            np.isfinite(
                regression_weights
            )
        )
    ):
        diagnostic[
            "admissibility_reason_code"
        ] = "NONFINITE_MODEL_INPUT"

        return diagnostic

    if np.any(
        regression_weights <= 0
    ):
        diagnostic[
            "admissibility_reason_code"
        ] = "NONPOSITIVE_REGRESSION_WEIGHT"

        return diagnostic

    singular_values = np.linalg.svd(
        model_matrix,
        compute_uv = False
    )

    if not len(
        singular_values
    ):
        diagnostic[
            "admissibility_reason_code"
        ] = "EMPTY_MODEL_MATRIX"

        return diagnostic

    largest_singular_value = float(
        singular_values[0]
    )

    rank_tolerance = (
        max(
            model_matrix.shape
        )
        * np.finfo(
            float
        ).eps
        * largest_singular_value
    )

    matrix_rank = int(
        np.sum(
            singular_values
            > rank_tolerance
        )
    )

    retained_singular_values = (
        singular_values[
            singular_values
            > rank_tolerance
        ]
    )

    smallest_singular_value = float(
        singular_values[-1]
    )

    smallest_retained_singular_value = (
        None
        if not len(
            retained_singular_values
        )
        else float(
            retained_singular_values[-1]
        )
    )

    effective_condition_number = (
        None
        if (
            smallest_retained_singular_value
            is None
            or smallest_retained_singular_value
            <= 0
        )
        else (
            largest_singular_value
            / smallest_retained_singular_value
        )
    )

    residual_independent_cell_degrees_of_freedom = (
        independent_cell_count
        - matrix_rank
    )

    diagnostic.update(
        {
            "matrix_rank": matrix_rank,
            "rank_tolerance": float(
                rank_tolerance
            ),
            "largest_singular_value": (
                largest_singular_value
            ),
            "smallest_singular_value": (
                smallest_singular_value
            ),
            "smallest_retained_singular_value": (
                smallest_retained_singular_value
            ),
            "effective_condition_number": (
                effective_condition_number
            ),
            "residual_independent_cell_degrees_of_freedom": (
                residual_independent_cell_degrees_of_freedom
            )
        }
    )

    if (
        admissibility_policy[
            "full_column_rank_required"
        ]
        and matrix_rank
        != model_matrix.shape[1]
    ):
        diagnostic[
            "admissibility_reason_code"
        ] = "MODEL_MATRIX_RANK_DEFICIENT"

        return diagnostic

    if (
        residual_independent_cell_degrees_of_freedom
        < admissibility_policy[
            "minimum_residual_independent_cell_degrees_of_freedom"
        ]
    ):
        diagnostic[
            "admissibility_reason_code"
        ] = (
            "INSUFFICIENT_RESIDUAL_INDEPENDENT_CELL_DEGREES_OF_FREEDOM"
        )

        return diagnostic

    if (
        effective_condition_number is None
        or not math.isfinite(
            effective_condition_number
        )
        or effective_condition_number
        > admissibility_policy[
            "maximum_effective_condition_number"
        ]
    ):
        diagnostic[
            "admissibility_reason_code"
        ] = "MODEL_MATRIX_ILL_CONDITIONED"

        return diagnostic

    diagnostic[
        "admissibility_status"
    ] = "admissible"

    diagnostic[
        "admissibility_reason_code"
    ] = None

    return diagnostic


# Reconciling selected candidate terms with the constructed matrix
def validate_selected_matrix_term_accounting(candidate_identity: str, candidate_term_records: list[dict],
                                             matrix_terms: list[dict]
) -> None:
    selected_accounted_terms = [
        (
            candidate_term_record[
                "term_type"
            ],
            candidate_term_record[
                "term_name"
            ]
        )
        for candidate_term_record in (
            candidate_term_records
        )
        if (
            candidate_term_record[
                "term_disposition"
            ]
            == "selected"
        )
    ]

    if len(
        selected_accounted_terms
    ) != len(
        set(
            selected_accounted_terms
        )
    ):
        raise RuntimeError(f"Candidate contains duplicate selected term-accounting records : {candidate_identity}")

    constructed_terms = [
        (
            matrix_term[
                "term_type"
            ],
            matrix_term[
                "column_name"
            ]
        )
        for matrix_term in matrix_terms
    ]

    if len(
        constructed_terms
    ) != len(
        set(
            constructed_terms
        )
    ):
        raise RuntimeError(f"Candidate model matrix contains duplicate term definitions : {candidate_identity}")

    selected_noncontext_terms = {
        selected_term
        for selected_term in (
            selected_accounted_terms
        )
        if selected_term[0] != "design_context"
    }

    constructed_noncontext_terms = {
        constructed_term
        for constructed_term in (
            constructed_terms
        )
        if constructed_term[0] != "design_context"
    }

    if (
        selected_noncontext_terms
        != constructed_noncontext_terms
    ):
        missing_matrix_terms = sorted(
            selected_noncontext_terms.difference(
                constructed_noncontext_terms
            )
        )

        unaccounted_matrix_terms = sorted(
            constructed_noncontext_terms.difference(
                selected_noncontext_terms
            )
        )

        raise RuntimeError(
            "Selected candidate terms do not reconcile with the constructed model matrix : "
            f"{candidate_identity} : missing {missing_matrix_terms} : unaccounted {unaccounted_matrix_terms}"
        )

    selected_context_fields = {
        term_name
        for term_type, term_name in (
            selected_accounted_terms
        )
        if term_type == "design_context"
    }

    constructed_context_fields = {
        matrix_term[
            "source_term_name"
        ]
        for matrix_term in matrix_terms
        if matrix_term[
            "term_type"
        ] == "design_context"
    }

    if (
        None in constructed_context_fields
        or selected_context_fields
        != constructed_context_fields
    ):
        missing_context_fields = sorted(
            selected_context_fields.difference(
                constructed_context_fields
            )
        )

        unaccounted_context_fields = sorted(
            constructed_context_fields.difference(
                selected_context_fields
            ),
            key = str
        )

        raise RuntimeError(
            "Selected context terms do not reconcile with the constructed model matrix : "
            f"{candidate_identity} : missing {missing_context_fields} : unaccounted {unaccounted_context_fields}"
        )


# Building deterministic matrices for every selected candidate
def prepare_candidate_model_matrices(candidate_records: list[dict], candidate_term_records: list[dict],
                                     candidate_observation_records: list[dict], modeling_policy: dict
) -> tuple[dict[str, dict], list[dict]]:
    all_terms_by_candidate = defaultdict(
        list
    )

    source_terms_by_candidate = defaultdict(
        list
    )

    authorized_source_term_types = set(
        modeling_policy[
            "candidate_assembly"
        ][
            "authorized_term_types"
        ]
    )

    for candidate_term_record in (
        candidate_term_records
    ):
        candidate_identity = candidate_term_record[
            "candidate_identity"
        ]

        all_terms_by_candidate[
            candidate_identity
        ].append(
            candidate_term_record
        )

        if (
            candidate_term_record[
                "term_disposition"
            ]
            == "selected"
            and candidate_term_record[
                "term_source"
            ]
            == "analytical_scope.csv"
            and candidate_term_record[
                "term_type"
            ]
            in authorized_source_term_types
        ):
            source_terms_by_candidate[
                candidate_identity
            ].append(
                candidate_term_record
            )

    observations_by_candidate = defaultdict(
        list
    )

    for observation_record in (
        candidate_observation_records
    ):
        observations_by_candidate[
            observation_record[
                "candidate_identity"
            ]
        ].append(
            observation_record
        )

    prepared_models = {}
    matrix_diagnostic_records = []

    for candidate_record in sorted(
        candidate_records,
        key = lambda record: record[
            "candidate_identity"
        ]
    ):
        if (
            candidate_record[
                "selection_status"
            ]
            != "selected"
        ):
            candidate_record[
                "model_matrix_status"
            ] = "not_selected"

            continue

        candidate_identity = candidate_record[
            "candidate_identity"
        ]

        if (
            candidate_record[
                "interaction_hierarchy_status"
            ]
            != "complete"
        ):
            candidate_record[
                "model_matrix_status"
            ] = "not_constructed"

            candidate_record[
                "admissibility_status"
            ] = "inadmissible"

            candidate_record[
                "admissibility_reason_code"
            ] = (
                candidate_record[
                    "interaction_hierarchy_reason_code"
                ]
            )

            continue

        if (
            candidate_record.get(
                "observation_assembly_status"
            )
            != "completed"
        ):
            candidate_record[
                "model_matrix_status"
            ] = "not_constructed"

            continue

        observation_records = sorted(
            observations_by_candidate[
                candidate_identity
            ],
            key = lambda record: (
                record[
                    "cell_identity"
                ],
                record[
                    "horizon"
                ],
                record[
                    "source_identity"
                ]
            )
        )

        candidate_terms = (
            source_terms_by_candidate[
                candidate_identity
            ]
        )

        all_candidate_terms = (
            all_terms_by_candidate[
                candidate_identity
            ]
        )

        try:
            model_encoding = (
                build_model_encoding(
                    candidate_record = (
                        candidate_record
                    ),
                    candidate_term_records = (
                        candidate_terms
                    ),
                    observation_records = (
                        observation_records
                    ),
                    modeling_policy = (
                        modeling_policy
                    )
                )
            )

            (
                model_matrix,
                matrix_terms
            ) = transform_model_matrix(
                candidate_record = candidate_record,
                observation_records = (
                    observation_records
                ),
                model_encoding = model_encoding
            )

            validate_selected_matrix_term_accounting(
                candidate_identity = (
                    candidate_identity
                ),
                candidate_term_records = (
                    all_candidate_terms
                ),
                matrix_terms = matrix_terms
            )

        except CandidateMatrixUnavailableError as exc:
            candidate_record[
                "model_matrix_status"
            ] = "unavailable"

            candidate_record[
                "admissibility_status"
            ] = "inadmissible"

            candidate_record[
                "admissibility_reason_code"
            ] = exc.reason_code

            candidate_record[
                "model_matrix_error"
            ] = str(
                exc
            )

            continue

        outcome_values = np.asarray(
            [
                observation_record[
                    "outcome_value"
                ]
                for observation_record in (
                    observation_records
                )
            ],
            dtype = float
        )

        regression_weights = (
            build_equal_cell_weights(
                observation_records
            )
        )

        admissibility_result = (
            evaluate_candidate_admissibility(
                candidate_record = (
                    candidate_record
                ),
                observation_records = (
                    observation_records
                ),
                model_matrix = model_matrix,
                outcome_values = outcome_values,
                regression_weights = (
                    regression_weights
                ),
                model_encoding = model_encoding,
                modeling_policy = modeling_policy
            )
        )

        candidate_record[
            "model_matrix_status"
        ] = "completed"

        candidate_record[
            "model_matrix_error"
        ] = None

        candidate_record[
            "admissibility_status"
        ] = admissibility_result[
            "admissibility_status"
        ]

        candidate_record[
            "admissibility_reason_code"
        ] = admissibility_result[
            "admissibility_reason_code"
        ]

        candidate_record[
            "model_matrix_column_count"
        ] = admissibility_result[
            "matrix_column_count"
        ]

        candidate_record[
            "model_matrix_rank"
        ] = admissibility_result[
            "matrix_rank"
        ]

        candidate_record[
            "effective_condition_number"
        ] = admissibility_result[
            "effective_condition_number"
        ]

        candidate_record[
            "residual_independent_cell_degrees_of_freedom"
        ] = admissibility_result[
            "residual_independent_cell_degrees_of_freedom"
        ]

        candidate_record[
            "assembled_model_observation_count"
        ] = admissibility_result["observation_count"]

        candidate_record[
            "assembled_model_condition_count"
        ] = admissibility_result["condition_protocol_count"]

        candidate_record[
            "assembled_model_cell_count"
        ] = admissibility_result["independent_cell_count"]

        candidate_record[
            "inside_evaluation_interval_observation_count"
        ] = sum(
            observation_record["inside_evaluation_interval"]
            for observation_record in observation_records
        )

        candidate_record[
            "outside_evaluation_interval_observation_count"
        ] = sum(
            not observation_record["inside_evaluation_interval"]
            for observation_record in observation_records
        )

        for field_name in [
            "repeated_cell_count",
            "single_observation_cell_count",
            "minimum_condition_unique_horizon_count",
            "minimum_condition_repeated_cell_count",
            "longitudinal_structure_status"
        ]:
            candidate_record[field_name] = admissibility_result[field_name]

        matrix_diagnostic_records.append(
            {
                "candidate_identity": (
                    candidate_identity
                ),
                "support_region_identity": (
                    candidate_record[
                        "support_region_identity"
                    ]
                ),
                "template_name": (
                    candidate_record[
                        "template_name"
                    ]
                ),
                **admissibility_result
            }
        )

        if (
            admissibility_result[
                "admissibility_status"
            ]
            != "admissible"
        ):
            continue

        estimator_name = candidate_record[
            "estimator"
        ]

        estimator_configuration = (
            modeling_policy[
                "estimation"
            ][
                estimator_name
            ]
        )

        fit_weights = (
            regression_weights.tolist()
            if estimator_name
            == "weighted_least_squares"
            else None
        )

        fit_identity = build_deterministic_identity(
            {
                "ordered_source_identities": [
                    observation_record[
                        "source_identity"
                    ]
                    for observation_record in (
                        observation_records
                    )
                ],
                "outcome": candidate_record[
                    "outcome"
                ],
                "cohort": candidate_record[
                    "cohort"
                ],
                "horizon_axis": (
                    candidate_record[
                        "horizon_axis"
                    ]
                ),
                "ordered_matrix_terms": (
                    matrix_terms
                ),
                "model_matrix": (
                    model_matrix.tolist()
                ),
                "outcome_values": (
                    outcome_values.tolist()
                ),
                "weights": fit_weights,
                "estimator": estimator_name,
                "estimator_configuration": (
                    estimator_configuration
                )
            }
        )

        candidate_record[
            "fit_identity"
        ] = fit_identity

        candidate_record[
            "model_matrix_column_names"
        ] = [
            matrix_term[
                "column_name"
            ]
            for matrix_term in (
                matrix_terms
            )
        ]

        candidate_record[
            "model_encoding"
        ] = model_encoding

        prepared_models[
            candidate_identity
        ] = {
            "candidate_identity": (
                candidate_identity
            ),
            "fit_identity": fit_identity,
            "candidate_record": candidate_record,
            "candidate_term_records": (
                candidate_terms
            ),
            "observation_records": (
                observation_records
            ),
            "model_encoding": model_encoding,
            "model_matrix": model_matrix,
            "matrix_terms": matrix_terms,
            "outcome_values": outcome_values,
            "regression_weights": (
                regression_weights
            ),
            "cell_identities": [
                observation_record[
                    "cell_identity"
                ]
                for observation_record in (
                    observation_records
                )
            ],
            "condition_protocol_identities": [
                observation_record[
                    "condition_protocol_identity"
                ]
                for observation_record in (
                    observation_records
                )
            ]
        }

    return (
        prepared_models,
        sorted(
            matrix_diagnostic_records,
            key = lambda record: record[
                "candidate_identity"
            ]
        )
    )


# Representing an expected candidate-specific fitting failure
class CandidateFitUnavailableError(RuntimeError):
    def __init__(self, reason_code: str, message: str,
                 failure_details: list[dict] | None = None
    ) -> None:
        super().__init__(
            message
        )

        self.reason_code = reason_code
        self.failure_details = (
            []
            if failure_details is None
            else failure_details
        )


# Calculating equal-cell residual diagnostics
def calculate_equal_cell_residual_diagnostics(residual_values: np.ndarray, cell_identities: list[str]) -> dict:
    if (
        residual_values.ndim != 1
        or len(residual_values)
        != len(cell_identities)
        or not len(residual_values)
    ):
        raise RuntimeError("Residual diagnostics require matching non-empty inputs.")

    if not np.all(
        np.isfinite(
            residual_values
        )
    ):
        raise RuntimeError("Residual diagnostics received non-finite residuals.")

    residuals_by_cell = defaultdict(
        list
    )

    for (
        residual_value,
        cell_identity
    ) in zip(
        residual_values,
        cell_identities,
        strict = True
    ):
        residuals_by_cell[
            cell_identity
        ].append(
            float(
                residual_value
            )
        )

    ordered_cell_identities = sorted(
        residuals_by_cell
    )

    cell_mean_residuals = [
        sum(
            residuals_by_cell[
                cell_identity
            ]
        )
        / len(
            residuals_by_cell[
                cell_identity
            ]
        )
        for cell_identity in (
            ordered_cell_identities
        )
    ]

    residual_mean = (
        sum(
            cell_mean_residuals
        )
        / len(
            cell_mean_residuals
        )
    )

    cell_mean_squared_deviations = []

    for cell_identity in (
        ordered_cell_identities
    ):
        cell_residuals = residuals_by_cell[
            cell_identity
        ]

        cell_mean_squared_deviations.append(
            sum(
                (
                    residual_value
                    - residual_mean
                ) ** 2
                for residual_value in (
                    cell_residuals
                )
            )
            / len(
                cell_residuals
            )
        )

    residual_variance = (
        sum(
            cell_mean_squared_deviations
        )
        / len(
            cell_mean_squared_deviations
        )
    )

    if residual_variance < 0:
        raise RuntimeError("Residual diagnostics produced a negative variance.")

    residual_standard_deviation = math.sqrt(
        residual_variance
    )

    if residual_standard_deviation == 0:
        maximum_absolute_standardized_residual = None

    else:
        maximum_absolute_standardized_residual = max(
            abs(
                float(
                    residual_value
                )
                / residual_standard_deviation
            )
            for residual_value in (
                residual_values
            )
        )

    return {
        "residual_mean": residual_mean,
        "residual_standard_deviation": (
            residual_standard_deviation
        ),
        "maximum_absolute_standardized_residual": (
            maximum_absolute_standardized_residual
        )
    }


# Calculating the equal-cell residual mean square
def calculate_equal_cell_residual_mean_square(residual_values: np.ndarray, cell_identities: list[str]) -> float:
    if (
        residual_values.ndim != 1
        or len(residual_values)
        != len(cell_identities)
        or not len(residual_values)
    ):
        raise RuntimeError("Residual mean square requires matching non-empty inputs.")

    residuals_by_cell = defaultdict(
        list
    )

    for (
        residual_value,
        cell_identity
    ) in zip(
        residual_values,
        cell_identities,
        strict = True
    ):
        if not math.isfinite(
            float(
                residual_value
            )
        ):
            raise RuntimeError("Residual mean square received a non-finite residual.")

        residuals_by_cell[
            cell_identity
        ].append(
            float(
                residual_value
            )
        )

    cell_mean_squared_residuals = [
        sum(
            residual_value ** 2
            for residual_value in (
                residuals_by_cell[
                    cell_identity
                ]
            )
        )
        / len(
            residuals_by_cell[
                cell_identity
            ]
        )
        for cell_identity in sorted(
            residuals_by_cell
        )
    ]

    return (
        sum(
            cell_mean_squared_residuals
        )
        / len(
            cell_mean_squared_residuals
        )
    )


# Calculating a stable inverse cross-product from one weighted matrix
def calculate_weighted_cross_product_inverse(weighted_matrix: np.ndarray) -> tuple[np.ndarray, dict]:
    if (
        weighted_matrix.ndim != 2
        or not weighted_matrix.shape[0]
        or not weighted_matrix.shape[1]
    ):
        raise RuntimeError("Weighted model matrix must be two-dimensional and non-empty.")

    try:
        (
            left_singular_vectors,
            singular_values,
            right_singular_vectors_transposed
        ) = np.linalg.svd(
            weighted_matrix,
            full_matrices = False
        )

    except np.linalg.LinAlgError as exc:
        raise CandidateFitUnavailableError(
            reason_code = "WEIGHTED_SINGULAR_VALUE_DECOMPOSITION_FAILED",
            message = (
                "Weighted model-matrix singular-value decomposition failed."
            )
        ) from exc

    if not len(
        singular_values
    ):
        raise CandidateFitUnavailableError(
            reason_code = "EMPTY_WEIGHTED_MODEL_MATRIX",
            message = (
                "Weighted model matrix does not contain singular values."
            )
        )

    largest_singular_value = float(
        singular_values[0]
    )

    rank_tolerance = (
        max(
            weighted_matrix.shape
        )
        * np.finfo(
            float
        ).eps
        * largest_singular_value
    )

    retained = (
        singular_values
        > rank_tolerance
    )

    weighted_rank = int(
        np.sum(
            retained
        )
    )

    if weighted_rank != weighted_matrix.shape[1]:
        raise CandidateFitUnavailableError(
            reason_code = "WEIGHTED_MODEL_MATRIX_RANK_DEFICIENT",
            message = (
                "Weighted model matrix is not full column rank."
            )
        )

    smallest_singular_value = float(
        singular_values[-1]
    )

    if smallest_singular_value <= 0:
        raise CandidateFitUnavailableError(
            reason_code = "WEIGHTED_MODEL_MATRIX_SINGULAR",
            message = (
                "Weighted model matrix contains a non-positive singular value."
            )
        )

    weighted_condition_number = (
        largest_singular_value
        / smallest_singular_value
    )

    inverse_cross_product = (
        right_singular_vectors_transposed.T
        @ np.diag(
            1.0
            / singular_values ** 2
        )
        @ right_singular_vectors_transposed
    )

    if not np.all(
        np.isfinite(
            inverse_cross_product
        )
    ):
        raise CandidateFitUnavailableError(
            reason_code = "NONFINITE_WEIGHTED_CROSS_PRODUCT_INVERSE",
            message = (
                "Weighted cross-product inverse contains non-finite values."
            )
        )

    return (
        inverse_cross_product,
        {
            "weighted_matrix_rank": (
                weighted_rank
            ),
            "weighted_rank_tolerance": float(
                rank_tolerance
            ),
            "weighted_largest_singular_value": (
                largest_singular_value
            ),
            "weighted_smallest_singular_value": (
                smallest_singular_value
            ),
            "weighted_condition_number": (
                weighted_condition_number
            ),
            "left_singular_vectors": (
                left_singular_vectors
            ),
            "singular_values": singular_values,
            "right_singular_vectors_transposed": (
                right_singular_vectors_transposed
            )
        }
    )


# Fitting one weighted least-squares model
def fit_weighted_least_squares(prepared_model: dict, modeling_policy: dict) -> dict:
    model_matrix = prepared_model[
        "model_matrix"
    ]

    outcome_values = prepared_model[
        "outcome_values"
    ]

    regression_weights = prepared_model[
        "regression_weights"
    ]

    cell_identities = prepared_model[
        "cell_identities"
    ]

    if (
        model_matrix.ndim != 2
        or outcome_values.ndim != 1
        or regression_weights.ndim != 1
        or model_matrix.shape[0]
        != len(
            outcome_values
        )
        or len(
            outcome_values
        )
        != len(
            regression_weights
        )
        or len(
            outcome_values
        )
        != len(
            cell_identities
        )
    ):
        raise RuntimeError("Weighted least-squares inputs have inconsistent dimensions.")

    if (
        not np.all(
            np.isfinite(
                model_matrix
            )
        )
        or not np.all(
            np.isfinite(
                outcome_values
            )
        )
        or not np.all(
            np.isfinite(
                regression_weights
            )
        )
        or np.any(
            regression_weights <= 0
        )
    ):
        raise CandidateFitUnavailableError(
            reason_code = "INVALID_WEIGHTED_LEAST_SQUARES_INPUT",
            message = (
                "Weighted least-squares inputs must be finite with strictly positive weights."
            )
        )

    square_root_weights = np.sqrt(
        regression_weights
    )

    weighted_matrix = (
        model_matrix
        * square_root_weights[
            :,
            np.newaxis
        ]
    )

    weighted_outcome = (
        outcome_values
        * square_root_weights
    )

    (
        inverse_cross_product,
        decomposition
    ) = calculate_weighted_cross_product_inverse(
        weighted_matrix
    )

    weighted_condition_number = float(
        decomposition[
            "weighted_condition_number"
        ]
    )

    maximum_condition_number = (
        modeling_policy[
            "admissibility"
        ][
            "maximum_effective_condition_number"
        ]
    )

    if (
        not math.isfinite(
            weighted_condition_number
        )
        or weighted_condition_number
        > maximum_condition_number
    ):
        raise CandidateFitUnavailableError(
            reason_code = (
                "WEIGHTED_MODEL_MATRIX_ILL_CONDITIONED"
            ),
            message = (
                "Weighted model matrix exceeds the configured condition-number limit."
            )
        )

    left_singular_vectors = decomposition[
        "left_singular_vectors"
    ]

    singular_values = decomposition[
        "singular_values"
    ]

    right_singular_vectors_transposed = (
        decomposition[
            "right_singular_vectors_transposed"
        ]
    )

    coefficients = (
        right_singular_vectors_transposed.T
        @ (
            (
                left_singular_vectors.T
                @ weighted_outcome
            )
            / singular_values
        )
    )

    if not np.all(
        np.isfinite(
            coefficients
        )
    ):
        raise CandidateFitUnavailableError(
            reason_code = "NONFINITE_WEIGHTED_LEAST_SQUARES_COEFFICIENT",
            message = (
                "Weighted least-squares fitting produced non-finite coefficients."
            )
        )

    marginal_predictions = (
        model_matrix
        @ coefficients
    )

    residual_values = (
        outcome_values
        - marginal_predictions
    )

    independent_cell_count = len(
        set(
            cell_identities
        )
    )

    matrix_rank = int(
        decomposition[
            "weighted_matrix_rank"
        ]
    )

    independent_cell_residual_degrees_of_freedom = (
        independent_cell_count
        - matrix_rank
    )

    if independent_cell_residual_degrees_of_freedom <= 0:
        raise CandidateFitUnavailableError(
            reason_code = "NONPOSITIVE_INDEPENDENT_CELL_RESIDUAL_DEGREES_OF_FREEDOM",
            message = (
                "Weighted least-squares fitting does not have positive independent-cell residual degrees of freedom."
            )
        )

    uncorrected_residual_mean_square = (
        calculate_equal_cell_residual_mean_square(
            residual_values = residual_values,
            cell_identities = cell_identities
        )
    )

    observation_variance = (
        independent_cell_count
        / independent_cell_residual_degrees_of_freedom
        * uncorrected_residual_mean_square
    )

    if (
        not math.isfinite(
            observation_variance
        )
        or observation_variance < 0
    ):
        raise CandidateFitUnavailableError(
            reason_code = "INVALID_WEIGHTED_LEAST_SQUARES_OBSERVATION_VARIANCE",
            message = (
                "Weighted least-squares fitting produced an invalid observation variance."
            )
        )

    model_based_fixed_effect_covariance = (
        observation_variance
        * inverse_cross_product
    )

    if not np.all(
        np.isfinite(
            model_based_fixed_effect_covariance
        )
    ):
        raise CandidateFitUnavailableError(
            reason_code = "NONFINITE_WEIGHTED_LEAST_SQUARES_COVARIANCE",
            message = (
                "Weighted least-squares covariance contains non-finite values."
            )
        )

    residual_diagnostics = (
        calculate_equal_cell_residual_diagnostics(
            residual_values = residual_values,
            cell_identities = cell_identities
        )
    )

    return {
        "estimator": "weighted_least_squares",
        "coefficients": coefficients,
        "model_based_fixed_effect_covariance": (
            model_based_fixed_effect_covariance
        ),
        "marginal_predictions": (
            marginal_predictions
        ),
        "conditional_predictions": (
            marginal_predictions.copy()
        ),
        "primary_residuals": residual_values,
        "marginal_residuals": residual_values.copy(),
        "conditional_residuals": residual_values.copy(),
        "observation_variance": (
            observation_variance
        ),
        "random_intercept_variance": None,
        "residual_variance": (
            observation_variance
        ),
        "random_effects_by_cell": {},
        "convergence_status": "completed",
        "optimizer": "closed_form_singular_value_decomposition",
        "optimizer_attempts": [],
        "random_effect_singularity_status": (
            "not_applicable"
        ),
        "residual_diagnostics": (
            residual_diagnostics
        ),
        "marginal_residual_diagnostics": (
            residual_diagnostics
        ),
        "conditional_residual_diagnostics": (
            residual_diagnostics
        ),
        "weighted_matrix_rank": (
            decomposition[
                "weighted_matrix_rank"
            ]
        ),
        "weighted_rank_tolerance": (
            decomposition[
                "weighted_rank_tolerance"
            ]
        ),
        "weighted_condition_number": (
            weighted_condition_number
        ),
        "fit_object": None
    }


# Reading one scalar random intercept
def extract_random_intercept(random_effect_value: object, cell_identity: str) -> float:
    random_effect_array = np.asarray(
        random_effect_value,
        dtype = float
    ).reshape(
        -1
    )

    if len(random_effect_array) != 1:
        raise CandidateFitUnavailableError(
            reason_code = "INVALID_RANDOM_INTERCEPT_DIMENSION",
            message = (
                f"Fitted random effect does not contain exactly one random intercept : {cell_identity}"
            )
        )

    random_intercept = float(
        random_effect_array[0]
    )

    if not math.isfinite(
        random_intercept
    ):
        raise CandidateFitUnavailableError(
            reason_code = "NONFINITE_RANDOM_INTERCEPT",
            message = (
                f"Fitted random intercept is non-finite : {cell_identity}"
            )
        )

    return random_intercept


# Classifying mixed-effects warnings without equating every warning with non-convergence
def classify_mixed_effects_warning(warning_message: str) -> str:
    normalized_message = (
        warning_message
        .strip()
        .lower()
    )

    if (
        "maximum likelihood optimization failed"
        in normalized_message
        or "mixedlm optimization failed"
        in normalized_message
        or "gradient optimization failed"
        in normalized_message
    ):
        return "optimizer_nonconvergence"

    if (
        "boundary of the parameter space"
        in normalized_message
    ):
        return "parameter_boundary"

    if (
        "random effects covariance is singular"
        in normalized_message
    ):
        return (
            "random_effect_covariance_singular"
        )

    if (
        "hessian matrix"
        in normalized_message
        and "not positive definite"
        in normalized_message
    ):
        return "hessian_not_positive_definite"

    return "unclassified_warning"


# Validating one fitted fixed-effect covariance matrix
def validate_fixed_effect_covariance(covariance_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    if (
        covariance_matrix.ndim != 2
        or covariance_matrix.shape[0]
        != covariance_matrix.shape[1]
        or not np.all(
            np.isfinite(
                covariance_matrix
            )
        )
    ):
        raise ValueError("Fixed-effect covariance is not finite and square.")

    symmetric_covariance = (
        covariance_matrix
        + covariance_matrix.T
    ) / 2

    covariance_eigenvalues = np.linalg.eigvalsh(
        symmetric_covariance
    )

    eigenvalue_tolerance = (
        covariance_matrix.shape[0]
        * np.finfo(
            float
        ).eps
        * max(
            1.0,
            float(
                np.max(
                    np.abs(
                        covariance_eigenvalues
                    )
                )
            )
        )
    )

    if (
        float(
            np.min(
                covariance_eigenvalues
            )
        )
        < -eigenvalue_tolerance
    ):
        raise ValueError("Fixed-effect covariance is not positive semidefinite.")

    return (
        symmetric_covariance,
        eigenvalue_tolerance
    )


# Fitting one random-intercept linear mixed-effects model
def fit_linear_mixed_effects(prepared_model: dict, modeling_policy: dict) -> dict:
    model_matrix = prepared_model[
        "model_matrix"
    ]

    outcome_values = prepared_model[
        "outcome_values"
    ]

    cell_identities = prepared_model[
        "cell_identities"
    ]

    if (
        model_matrix.ndim != 2
        or outcome_values.ndim != 1
        or model_matrix.shape[0]
        != len(
            outcome_values
        )
        or len(
            outcome_values
        )
        != len(
            cell_identities
        )
    ):
        raise RuntimeError("Mixed-effects inputs have inconsistent dimensions.")

    if (
        not np.all(
            np.isfinite(
                model_matrix
            )
        )
        or not np.all(
            np.isfinite(
                outcome_values
            )
        )
    ):
        raise CandidateFitUnavailableError(
            reason_code = "INVALID_MIXED_EFFECTS_INPUT",
            message = (
                "Mixed-effects inputs must contain only finite values."
            )
        )

    estimation_policy = modeling_policy[
        "estimation"
    ][
        "linear_mixed_effects"
    ]

    optimizer_attempts = []

    fitted_result = None

    successful_optimizer = None

    successful_fixed_effect_covariance = None

    for optimizer_name in estimation_policy[
        "optimizer_sequence"
    ]:
        attempt_record = {
            "optimizer": optimizer_name,
            "status": "failed",
            "optimizer_converged": None,
            "warning_messages": [],
            "warning_codes": [],
            "random_intercept_variance": None,
            "residual_variance": None,
            "intraclass_correlation": None,
            "fixed_effect_covariance_status": None,
            "fixed_effect_covariance_eigenvalue_tolerance": None,
            "error": None
        }

        try:
            with warnings.catch_warnings(
                record = True
            ) as captured_warnings:
                warnings.simplefilter(
                    "always"
                )

                model = MixedLM(
                    endog = outcome_values,
                    exog = model_matrix,
                    groups = np.asarray(
                        cell_identities,
                        dtype = object
                    ),
                    exog_re = np.ones(
                        (
                            len(
                                outcome_values
                            ),
                            1
                        ),
                        dtype = float
                    )
                )

                candidate_result = model.fit(
                    reml = True,
                    method = optimizer_name,
                    maxiter = estimation_policy[
                        "maximum_iterations"
                    ],
                    full_output = True,
                    disp = False
                )

            warning_messages = [
                str(
                    captured_warning.message
                )
                for captured_warning in (
                    captured_warnings
                )
            ]

            attempt_record[
                "warning_messages"
            ] = warning_messages

            warning_codes = [
                classify_mixed_effects_warning(
                    str(
                        captured_warning.message
                    )
                )
                for captured_warning in (
                    captured_warnings
                )
            ]

            attempt_record[
                "warning_codes"
            ] = warning_codes

            optimizer_converged = bool(
                candidate_result.converged
            )

            attempt_record[
                "optimizer_converged"
            ] = optimizer_converged

            if not optimizer_converged:
                attempt_record[
                    "error"
                ] = "non_convergence"

                optimizer_attempts.append(
                    attempt_record
                )

                continue

            fixed_effects = np.asarray(
                candidate_result.fe_params,
                dtype = float
            )

            if (
                fixed_effects.ndim != 1
                or len(
                    fixed_effects
                )
                != model_matrix.shape[1]
                or not np.all(
                    np.isfinite(
                        fixed_effects
                    )
                )
            ):
                attempt_record[
                    "error"
                ] = "invalid_fixed_effects"

                optimizer_attempts.append(
                    attempt_record
                )

                continue

            covariance_parameters = np.asarray(
                candidate_result.cov_params(),
                dtype = float
            )

            fixed_effect_covariance = (
                covariance_parameters[
                    :model_matrix.shape[1],
                    :model_matrix.shape[1]
                ]
            )

            try:
                (
                    fixed_effect_covariance,
                    covariance_eigenvalue_tolerance
                ) = validate_fixed_effect_covariance(
                    fixed_effect_covariance
                )

            except ValueError as exc:
                attempt_record[
                    "fixed_effect_covariance_status"
                ] = "invalid"

                attempt_record[
                    "error"
                ] = "invalid_fixed_effect_covariance"

                attempt_record[
                    "warning_messages"
                ].append(
                    str(
                        exc
                    )
                )

                optimizer_attempts.append(
                    attempt_record
                )

                continue

            attempt_record[
                "fixed_effect_covariance_status"
            ] = "completed"

            attempt_record[
                "fixed_effect_covariance_eigenvalue_tolerance"
            ] = covariance_eigenvalue_tolerance

            random_effect_covariance = np.asarray(
                candidate_result.cov_re,
                dtype = float
            )

            if random_effect_covariance.shape != (
                1,
                1
            ):
                attempt_record[
                    "error"
                ] = "invalid_random_effect_covariance"

                optimizer_attempts.append(
                    attempt_record
                )

                continue

            random_intercept_variance = float(
                random_effect_covariance[
                    0,
                    0
                ]
            )

            residual_variance = float(
                candidate_result.scale
            )

            if (
                not math.isfinite(
                    residual_variance
                )
                or residual_variance <= 0
            ):
                attempt_record[
                    "error"
                ] = "invalid_residual_variance"

                optimizer_attempts.append(
                    attempt_record
                )

                continue

            total_variance = (
                random_intercept_variance
                + residual_variance
            )

            if (
                not math.isfinite(
                    total_variance
                )
                or total_variance <= 0
            ):
                attempt_record[
                    "error"
                ] = "invalid_total_variance"

                optimizer_attempts.append(
                    attempt_record
                )

                continue

            intraclass_correlation = (
                random_intercept_variance
                / total_variance
            )

            attempt_record[
                "random_intercept_variance"
            ] = random_intercept_variance

            attempt_record[
                "residual_variance"
            ] = residual_variance

            attempt_record[
                "intraclass_correlation"
            ] = intraclass_correlation

            if (
                not math.isfinite(
                    intraclass_correlation
                )
                or intraclass_correlation
                <= estimation_policy[
                    "singular_random_effect_icc_tolerance"
                ]
            ):
                attempt_record[
                    "error"
                ] = (
                    "singular_random_effect_intraclass_correlation"
                )

                optimizer_attempts.append(
                    attempt_record
                )

                continue

            attempt_record[
                "status"
            ] = "completed"

            optimizer_attempts.append(
                attempt_record
            )

            successful_fixed_effect_covariance = (
                fixed_effect_covariance.copy()
            )

            fitted_result = candidate_result
            successful_optimizer = optimizer_name

            break

        except (
            np.linalg.LinAlgError,
            RuntimeError,
            ValueError
        ) as exc:
            attempt_record[
                "error"
            ] = (
                f"{type(exc).__name__} : {exc}"
            )

            optimizer_attempts.append(
                attempt_record
            )

    if fitted_result is None:
        raise CandidateFitUnavailableError(
            reason_code = "MIXED_EFFECTS_FIT_FAILED",
            message = (
                "Every configured mixed-effects optimizer failed."
            ),
            failure_details = optimizer_attempts
        )

    coefficients = np.asarray(
        fitted_result.fe_params,
        dtype = float
    )

    if successful_fixed_effect_covariance is None:
        raise RuntimeError("Completed mixed-effects fit is missing its validated fixed-effect covariance.")

    model_based_fixed_effect_covariance = (
        successful_fixed_effect_covariance
    )

    random_intercept_variance = float(
        np.asarray(
            fitted_result.cov_re,
            dtype = float
        )[
            0,
            0
        ]
    )

    residual_variance = float(
        fitted_result.scale
    )

    marginal_predictions = (
        model_matrix
        @ coefficients
    )

    fitted_random_effects = (
        fitted_result.random_effects
    )

    random_effects_by_cell = {}

    for cell_identity in sorted(
        set(
            cell_identities
        )
    ):
        if cell_identity not in (
            fitted_random_effects
        ):
            raise CandidateFitUnavailableError(
                reason_code = "MISSING_FITTED_RANDOM_INTERCEPT",
                message = (
                    f"Mixed-effects result does not contain a random intercept for cell : {cell_identity}"
                )
            )

        random_effects_by_cell[
            cell_identity
        ] = extract_random_intercept(
            random_effect_value = (
                fitted_random_effects[
                    cell_identity
                ]
            ),
            cell_identity = cell_identity
        )

    conditional_predictions = np.asarray(
        [
            marginal_prediction
            + random_effects_by_cell[
                cell_identity
            ]
            for (
                marginal_prediction,
                cell_identity
            ) in zip(
                marginal_predictions,
                cell_identities,
                strict = True
            )
        ],
        dtype = float
    )

    marginal_residuals = (
        outcome_values
        - marginal_predictions
    )

    conditional_residuals = (
        outcome_values
        - conditional_predictions
    )

    marginal_residual_diagnostics = (
        calculate_equal_cell_residual_diagnostics(
            residual_values = (
                marginal_residuals
            ),
            cell_identities = cell_identities
        )
    )

    conditional_residual_diagnostics = (
        calculate_equal_cell_residual_diagnostics(
            residual_values = (
                conditional_residuals
            ),
            cell_identities = cell_identities
        )
    )

    return {
        "estimator": "linear_mixed_effects",
        "coefficients": coefficients,
        "model_based_fixed_effect_covariance": (
            model_based_fixed_effect_covariance
        ),
        "marginal_predictions": (
            marginal_predictions
        ),
        "conditional_predictions": (
            conditional_predictions
        ),
        "primary_residuals": (
            conditional_residuals
        ),
        "marginal_residuals": (
            marginal_residuals
        ),
        "conditional_residuals": (
            conditional_residuals
        ),
        "observation_variance": (
            random_intercept_variance
            + residual_variance
        ),
        "random_intercept_variance": (
            random_intercept_variance
        ),
        "residual_variance": (
            residual_variance
        ),
        "random_effects_by_cell": (
            random_effects_by_cell
        ),
        "convergence_status": "completed",
        "optimizer": successful_optimizer,
        "optimizer_attempts": optimizer_attempts,
        "random_effect_singularity_status": (
            "non_singular"
        ),
        "residual_diagnostics": (
            conditional_residual_diagnostics
        ),
        "marginal_residual_diagnostics": (
            marginal_residual_diagnostics
        ),
        "conditional_residual_diagnostics": (
            conditional_residual_diagnostics
        ),
        "weighted_matrix_rank": None,
        "weighted_rank_tolerance": None,
        "weighted_condition_number": None,
        "fit_object": fitted_result
    }


# Confirming that two prepared models are exactly reusable
def validate_fit_reuse(first_prepared_model: dict, second_prepared_model: dict) -> None:
    scalar_fields = [
        "fit_identity"
    ]

    for field_name in scalar_fields:
        if (
            first_prepared_model[
                field_name
            ]
            != second_prepared_model[
                field_name
            ]
        ):
            raise RuntimeError("Prepared models do not have the same fit identity.")

    sequence_fields = [
        "cell_identities",
        "condition_protocol_identities"
    ]

    for field_name in sequence_fields:
        if (
            first_prepared_model[
                field_name
            ]
            != second_prepared_model[
                field_name
            ]
        ):
            raise RuntimeError(f"Fit-identity reuse contains conflicting {field_name}.")

    array_fields = [
        "model_matrix",
        "outcome_values",
        "regression_weights"
    ]

    for field_name in array_fields:
        if not np.array_equal(
            first_prepared_model[
                field_name
            ],
            second_prepared_model[
                field_name
            ]
        ):
            raise RuntimeError(f"Fit-identity reuse contains conflicting {field_name}.")

    if (
        first_prepared_model[
            "matrix_terms"
        ]
        != second_prepared_model[
            "matrix_terms"
        ]
    ):
        raise RuntimeError("Fit-identity reuse contains conflicting matrix terms.")

    if (
        serialize_canonical_json(
            first_prepared_model[
                "model_encoding"
            ]
        )
        != serialize_canonical_json(
            second_prepared_model[
                "model_encoding"
            ]
        )
    ):
        raise RuntimeError("Fit-identity reuse contains conflicting model encodings.")

    first_candidate = first_prepared_model[
        "candidate_record"
    ]

    second_candidate = second_prepared_model[
        "candidate_record"
    ]

    reusable_candidate_fields = [
        "estimator",
        "horizon_basis",
        "include_horizon_by_treatment_interactions"
    ]

    for field_name in reusable_candidate_fields:
        if (
            first_candidate[
                field_name
            ]
            != second_candidate[
                field_name
            ]
        ):
            raise RuntimeError(f"Fit-identity reuse contains conflicting candidate field : {field_name}")


# Fitting one prepared model with its declared estimator
def fit_prepared_model(prepared_model: dict, modeling_policy: dict) -> dict:
    estimator_name = prepared_model[
        "candidate_record"
    ][
        "estimator"
    ]

    if estimator_name == "weighted_least_squares":
        return fit_weighted_least_squares(
            prepared_model = prepared_model,
            modeling_policy = modeling_policy
        )

    if estimator_name == "linear_mixed_effects":
        return fit_linear_mixed_effects(
            prepared_model = prepared_model,
            modeling_policy = modeling_policy
        )

    raise RuntimeError(f"Unsupported estimator : {estimator_name}")


# Fitting each distinct admissible model once
def fit_candidate_models(prepared_models: dict[str, dict], modeling_policy: dict
) -> tuple[dict[str, dict], list[dict], list[dict]]:
    models_by_fit_identity = defaultdict(
        list
    )

    for prepared_model in (
        prepared_models.values()
    ):
        models_by_fit_identity[
            prepared_model[
                "fit_identity"
            ]
        ].append(
            prepared_model
        )

    fitted_models_by_candidate = {}

    coefficient_records = []

    fit_diagnostic_records = []

    for fit_identity in sorted(
        models_by_fit_identity
    ):
        reusable_models = sorted(
            models_by_fit_identity[
                fit_identity
            ],
            key = lambda prepared_model: (
                prepared_model[
                    "candidate_identity"
                ]
            )
        )

        canonical_model = reusable_models[0]

        for reusable_model in (
            reusable_models[1:]
        ):
            validate_fit_reuse(
                first_prepared_model = canonical_model,
                second_prepared_model = reusable_model
            )

        canonical_candidate_identity = (
            canonical_model[
                "candidate_identity"
            ]
        )

        try:
            fitted_result = fit_prepared_model(
                prepared_model = canonical_model,
                modeling_policy = modeling_policy
            )

            fit_status = "completed"
            fit_reason_code = None
            fit_error = None
            failure_details = []

        except CandidateFitUnavailableError as exc:
            fitted_result = None
            fit_status = "failed"
            fit_reason_code = exc.reason_code
            fit_error = str(
                exc
            )
            failure_details = (
                exc.failure_details
            )

        for prepared_model in (
            reusable_models
        ):
            candidate_identity = prepared_model[
                "candidate_identity"
            ]

            candidate_record = prepared_model[
                "candidate_record"
            ]

            candidate_record[
                "fit_status"
            ] = fit_status

            candidate_record[
                "fit_reason_code"
            ] = fit_reason_code

            candidate_record[
                "fit_error"
            ] = fit_error

            candidate_record[
                "canonical_fit_candidate_identity"
            ] = canonical_candidate_identity

            if fitted_result is None:
                candidate_record[
                    "uncertainty_status"
                ] = "not_attempted"

                candidate_record[
                    "validation_status"
                ] = "not_attempted"

                candidate_record[
                    "influence_status"
                ] = "not_attempted"

                fit_diagnostic_records.append(
                    {
                        "candidate_identity": (
                            candidate_identity
                        ),
                        "fit_identity": fit_identity,
                        "support_region_identity": (
                            candidate_record[
                                "support_region_identity"
                            ]
                        ),
                        "template_name": (
                            candidate_record[
                                "template_name"
                            ]
                        ),
                        "estimator": (
                            candidate_record[
                                "estimator"
                            ]
                        ),
                        "fit_status": fit_status,
                        "fit_reason_code": (
                            fit_reason_code
                        ),
                        "fit_error": fit_error,
                        "convergence_status": "failed",
                        "optimizer": None,
                        "optimizer_attempts": (
                            failure_details
                        ),
                        "random_effect_singularity_status": (
                            None
                        ),
                        "residual_mean": None,
                        "residual_standard_deviation": None,
                        "maximum_absolute_standardized_residual": (
                            None
                        ),
                        "marginal_residual_mean": None,
                        "marginal_residual_standard_deviation": (
                            None
                        ),
                        "conditional_residual_mean": None,
                        "conditional_residual_standard_deviation": (
                            None
                        ),
                        "observation_variance": None,
                        "random_intercept_variance": None,
                        "residual_variance": None,
                        "weighted_matrix_rank": None,
                        "weighted_rank_tolerance": None,
                        "weighted_condition_number": None
                    }
                )

                continue

            fitted_models_by_candidate[
                candidate_identity
            ] = fitted_result

            candidate_record[
                "fitted_coefficient_count"
            ] = len(
                fitted_result[
                    "coefficients"
                ]
            )

            for (
                column_index,
                matrix_term
            ) in enumerate(
                prepared_model[
                    "matrix_terms"
                ]
            ):
                coefficient_records.append(
                    {
                        "candidate_identity": (
                            candidate_identity
                        ),
                        "fit_identity": fit_identity,
                        "support_region_identity": (
                            candidate_record[
                                "support_region_identity"
                            ]
                        ),
                        "template_name": (
                            candidate_record[
                                "template_name"
                            ]
                        ),
                        "estimator": (
                            candidate_record[
                                "estimator"
                            ]
                        ),
                        "column_index": column_index,
                        "column_name": (
                            matrix_term[
                                "column_name"
                            ]
                        ),
                        "term_type": (
                            matrix_term[
                                "term_type"
                            ]
                        ),
                        "source_term_name": (
                            matrix_term[
                                "source_term_name"
                            ]
                        ),
                        "horizon_basis_name": (
                            matrix_term[
                                "horizon_basis_name"
                            ]
                        ),
                        "estimate": float(
                            fitted_result[
                                "coefficients"
                            ][
                                column_index
                            ]
                        ),
                        "uncertainty_status": (
                            "not_attempted"
                        ),
                        "standard_error": None,
                        "confidence_interval_lower": None,
                        "confidence_interval_upper": None,
                        "estimate_label": "associational"
                    }
                )

            primary_residual_diagnostics = (
                fitted_result[
                    "residual_diagnostics"
                ]
            )

            marginal_residual_diagnostics = (
                fitted_result[
                    "marginal_residual_diagnostics"
                ]
            )

            conditional_residual_diagnostics = (
                fitted_result[
                    "conditional_residual_diagnostics"
                ]
            )

            fit_diagnostic_records.append(
                {
                    "candidate_identity": (
                        candidate_identity
                    ),
                    "fit_identity": fit_identity,
                    "support_region_identity": (
                        candidate_record[
                            "support_region_identity"
                        ]
                    ),
                    "template_name": (
                        candidate_record[
                            "template_name"
                        ]
                    ),
                    "estimator": (
                        candidate_record[
                            "estimator"
                        ]
                    ),
                    "fit_status": "completed",
                    "fit_reason_code": None,
                    "fit_error": None,
                    "convergence_status": (
                        fitted_result[
                            "convergence_status"
                        ]
                    ),
                    "optimizer": (
                        fitted_result[
                            "optimizer"
                        ]
                    ),
                    "optimizer_attempts": (
                        fitted_result[
                            "optimizer_attempts"
                        ]
                    ),
                    "random_effect_singularity_status": (
                        fitted_result[
                            "random_effect_singularity_status"
                        ]
                    ),
                    "residual_mean": (
                        primary_residual_diagnostics[
                            "residual_mean"
                        ]
                    ),
                    "residual_standard_deviation": (
                        primary_residual_diagnostics[
                            "residual_standard_deviation"
                        ]
                    ),
                    "maximum_absolute_standardized_residual": (
                        primary_residual_diagnostics[
                            "maximum_absolute_standardized_residual"
                        ]
                    ),
                    "marginal_residual_mean": (
                        marginal_residual_diagnostics[
                            "residual_mean"
                        ]
                    ),
                    "marginal_residual_standard_deviation": (
                        marginal_residual_diagnostics[
                            "residual_standard_deviation"
                        ]
                    ),
                    "conditional_residual_mean": (
                        conditional_residual_diagnostics[
                            "residual_mean"
                        ]
                    ),
                    "conditional_residual_standard_deviation": (
                        conditional_residual_diagnostics[
                            "residual_standard_deviation"
                        ]
                    ),
                    "observation_variance": (
                        fitted_result[
                            "observation_variance"
                        ]
                    ),
                    "random_intercept_variance": (
                        fitted_result[
                            "random_intercept_variance"
                        ]
                    ),
                    "residual_variance": (
                        fitted_result[
                            "residual_variance"
                        ]
                    ),
                    "weighted_matrix_rank": (
                        fitted_result[
                            "weighted_matrix_rank"
                        ]
                    ),
                    "weighted_rank_tolerance": (
                        fitted_result[
                            "weighted_rank_tolerance"
                        ]
                    ),
                    "weighted_condition_number": (
                        fitted_result[
                            "weighted_condition_number"
                        ]
                    )
                }
            )

    return (
        fitted_models_by_candidate,
        sorted(
            coefficient_records,
            key = lambda record: (
                record[
                    "candidate_identity"
                ],
                record[
                    "column_index"
                ]
            )
        ),
        sorted(
            fit_diagnostic_records,
            key = lambda record: record[
                "candidate_identity"
            ]
        )
    )


# Checking a delete-one-cell model matrix
def validate_deleted_refit_matrix(model_matrix: np.ndarray, regression_weights: np.ndarray,
                                  modeling_policy: dict
) -> dict:
    if (
        model_matrix.ndim != 2
        or not model_matrix.shape[0]
        or not model_matrix.shape[1]
    ):
        raise CandidateFitUnavailableError(
            reason_code = "EMPTY_DELETED_REFIT_MATRIX",
            message = (
                "Delete-one-cell model matrix is empty."
            )
        )

    if (
        regression_weights.ndim != 1
        or len(
            regression_weights
        )
        != model_matrix.shape[0]
    ):
        raise CandidateFitUnavailableError(
            reason_code = "INVALID_DELETED_REFIT_WEIGHTS",
            message = (
                "Delete-one-cell weights do not match the model matrix."
            )
        )

    if (
        not np.all(
            np.isfinite(
                model_matrix
            )
        )
        or not np.all(
            np.isfinite(
                regression_weights
            )
        )
        or np.any(
            regression_weights <= 0
        )
    ):
        raise CandidateFitUnavailableError(
            reason_code = "NONFINITE_DELETED_REFIT_INPUT",
            message = (
                "Delete-one-cell model inputs must be finite with positive weights."
            )
        )

    try:
        singular_values = np.linalg.svd(
            model_matrix,
            compute_uv = False
        )

    except np.linalg.LinAlgError as exc:
        raise CandidateFitUnavailableError(
            reason_code = (
                "DELETED_REFIT_SINGULAR_VALUE_DECOMPOSITION_FAILED"
            ),
            message = (
                "Delete-one-cell matrix decomposition failed."
            )
        ) from exc

    if not len(
        singular_values
    ):
        raise CandidateFitUnavailableError(
            reason_code = "EMPTY_DELETED_REFIT_SINGULAR_VALUES",
            message = (
                "Delete-one-cell matrix does not contain singular values."
            )
        )

    largest_singular_value = float(
        singular_values[0]
    )

    rank_tolerance = (
        max(
            model_matrix.shape
        )
        * np.finfo(
            float
        ).eps
        * largest_singular_value
    )

    matrix_rank = int(
        np.sum(
            singular_values
            > rank_tolerance
        )
    )

    if matrix_rank != model_matrix.shape[1]:
        raise CandidateFitUnavailableError(
            reason_code = (
                "DELETED_REFIT_MODEL_MATRIX_RANK_DEFICIENT"
            ),
            message = (
                "Delete-one-cell model matrix is not full column rank."
            )
        )

    smallest_retained_singular_value = float(
        singular_values[
            matrix_rank - 1
        ]
    )

    if smallest_retained_singular_value <= 0:
        raise CandidateFitUnavailableError(
            reason_code = (
                "DELETED_REFIT_MODEL_MATRIX_SINGULAR"
            ),
            message = (
                "Delete-one-cell model matrix contains a non-positive retained singular value."
            )
        )

    effective_condition_number = (
        largest_singular_value
        / smallest_retained_singular_value
    )

    maximum_condition_number = (
        modeling_policy[
            "admissibility"
        ][
            "maximum_effective_condition_number"
        ]
    )

    if (
        not math.isfinite(
            effective_condition_number
        )
        or effective_condition_number
        > maximum_condition_number
    ):
        raise CandidateFitUnavailableError(
            reason_code = (
                "DELETED_REFIT_MODEL_MATRIX_ILL_CONDITIONED"
            ),
            message = (
                "Delete-one-cell model matrix exceeds the configured condition-number limit."
            )
        )

    return {
        "matrix_rank": matrix_rank,
        "rank_tolerance": float(
            rank_tolerance
        ),
        "effective_condition_number": float(
            effective_condition_number
        )
    }


# Building one delete-one-cell prepared model
def build_deleted_cell_prepared_model(prepared_model: dict, deleted_cell_identity: str,
                                      modeling_policy: dict
) -> dict:
    cell_identities = prepared_model[
        "cell_identities"
    ]

    retained_row_indices = [
        row_index
        for row_index, cell_identity in enumerate(
            cell_identities
        )
        if cell_identity != deleted_cell_identity
    ]

    if len(
        retained_row_indices
    ) == len(
        cell_identities
    ):
        raise RuntimeError(
            "Delete-one-cell refit identifies an unknown cell : "
            f"{deleted_cell_identity}"
        )

    if not retained_row_indices:
        raise CandidateFitUnavailableError(
            reason_code = "NO_REMAINING_DELETED_REFIT_OBSERVATIONS",
            message = (
                "Delete-one-cell refit does not retain observations."
            )
        )

    retained_observation_records = [
        prepared_model[
            "observation_records"
        ][
            row_index
        ]
        for row_index in retained_row_indices
    ]

    retained_model_matrix = prepared_model[
        "model_matrix"
    ][
        retained_row_indices,
        :
    ]

    retained_outcome_values = prepared_model[
        "outcome_values"
    ][
        retained_row_indices
    ]

    retained_cell_identities = [
        prepared_model[
            "cell_identities"
        ][
            row_index
        ]
        for row_index in retained_row_indices
    ]

    retained_condition_identities = [
        prepared_model[
            "condition_protocol_identities"
        ][
            row_index
        ]
        for row_index in retained_row_indices
    ]

    if not np.all(
        np.isfinite(
            retained_outcome_values
        )
    ):
        raise CandidateFitUnavailableError(
            reason_code = "NONFINITE_DELETED_REFIT_OUTCOME",
            message = (
                "Delete-one-cell outcomes contain non-finite values."
            )
        )

    estimator_name = prepared_model[
        "candidate_record"
    ][
        "estimator"
    ]

    if estimator_name == "weighted_least_squares":
        retained_regression_weights = (
            build_equal_cell_weights(
                retained_observation_records
            )
        )

    elif estimator_name == "linear_mixed_effects":
        retained_regression_weights = np.ones(
            len(
                retained_observation_records
            ),
            dtype = float
        )

    else:
        raise RuntimeError(f"Unsupported estimator : {estimator_name}")

    deleted_refit_diagnostics = (
        validate_deleted_refit_matrix(
            model_matrix = retained_model_matrix,
            regression_weights = (
                retained_regression_weights
            ),
            modeling_policy = modeling_policy
        )
    )

    return {
        "candidate_identity": (
            prepared_model[
                "candidate_identity"
            ]
        ),
        "fit_identity": (
            prepared_model[
                "fit_identity"
            ]
        ),
        "candidate_record": (
            prepared_model[
                "candidate_record"
            ]
        ),
        "observation_records": (
            retained_observation_records
        ),
        "model_encoding": (
            prepared_model[
                "model_encoding"
            ]
        ),
        "model_matrix": retained_model_matrix,
        "matrix_terms": (
            prepared_model[
                "matrix_terms"
            ]
        ),
        "outcome_values": retained_outcome_values,
        "regression_weights": (
            retained_regression_weights
        ),
        "cell_identities": retained_cell_identities,
        "condition_protocol_identities": (
            retained_condition_identities
        ),
        "deleted_cell_identity": (
            deleted_cell_identity
        ),
        "deleted_refit_diagnostics": (
            deleted_refit_diagnostics
        )
    }


# Calculating one delete-one-cell coefficient inventory
def fit_deleted_cell_coefficients(prepared_model: dict, modeling_policy: dict) -> list[dict]:
    distinct_cell_identities = sorted(
        set(
            prepared_model[
                "cell_identities"
            ]
        )
    )

    deletion_records = []

    for deleted_cell_identity in (
        distinct_cell_identities
    ):
        try:
            deleted_prepared_model = (
                build_deleted_cell_prepared_model(
                    prepared_model = prepared_model,
                    deleted_cell_identity = (
                        deleted_cell_identity
                    ),
                    modeling_policy = modeling_policy
                )
            )

            deleted_fit = fit_prepared_model(
                prepared_model = (
                    deleted_prepared_model
                ),
                modeling_policy = modeling_policy
            )

            deleted_coefficients = np.asarray(
                deleted_fit[
                    "coefficients"
                ],
                dtype = float
            )

            if (
                deleted_coefficients.ndim != 1
                or len(
                    deleted_coefficients
                )
                != prepared_model[
                    "model_matrix"
                ].shape[1]
                or not np.all(
                    np.isfinite(
                        deleted_coefficients
                    )
                )
            ):
                raise CandidateFitUnavailableError(
                    reason_code = (
                        "INVALID_DELETED_REFIT_COEFFICIENTS"
                    ),
                    message = (
                        "Delete-one-cell refit returned invalid coefficients."
                    )
                )

            deletion_records.append(
                {
                    "deleted_cell_identity": (
                        deleted_cell_identity
                    ),
                    "fit_status": "completed",
                    "fit_reason_code": None,
                    "fit_error": None,
                    "coefficients": (
                        deleted_coefficients
                    ),
                    "optimizer": (
                        deleted_fit[
                            "optimizer"
                        ]
                    ),
                    "optimizer_attempts": (
                        deleted_fit[
                            "optimizer_attempts"
                        ]
                    )
                }
            )

        except CandidateFitUnavailableError as exc:
            deletion_records.append(
                {
                    "deleted_cell_identity": (
                        deleted_cell_identity
                    ),
                    "fit_status": "failed",
                    "fit_reason_code": (
                        exc.reason_code
                    ),
                    "fit_error": str(
                        exc
                    ),
                    "coefficients": None,
                    "optimizer": None,
                    "optimizer_attempts": (
                        exc.failure_details
                    )
                }
            )

    return deletion_records


# Calculating delete-one-cell covariance
def calculate_jackknife_covariance(deletion_records: list[dict], coefficient_count: int) -> np.ndarray:
    if not deletion_records:
        raise CandidateFitUnavailableError(
            reason_code = "NO_DELETED_REFITS",
            message = (
                "Jackknife covariance requires deletion refits."
            )
        )

    failed_deletions = [
        deletion_record
        for deletion_record in deletion_records
        if deletion_record[
            "fit_status"
        ] != "completed"
    ]

    if failed_deletions:
        raise CandidateFitUnavailableError(
            reason_code = "DELETED_REFIT_FAILED",
            message = (
                "At least one required delete-one-cell refit failed."
            ),
            failure_details = [
                {
                    "deleted_cell_identity": (
                        deletion_record[
                            "deleted_cell_identity"
                        ]
                    ),
                    "fit_reason_code": (
                        deletion_record[
                            "fit_reason_code"
                        ]
                    ),
                    "fit_error": (
                        deletion_record[
                            "fit_error"
                        ]
                    )
                }
                for deletion_record in failed_deletions
            ]
        )

    deleted_coefficient_matrix = np.vstack(
        [
            deletion_record[
                "coefficients"
            ]
            for deletion_record in deletion_records
        ]
    )

    if deleted_coefficient_matrix.shape != (
        len(
            deletion_records
        ),
        coefficient_count
    ):
        raise CandidateFitUnavailableError(
            reason_code = (
                "INVALID_DELETED_COEFFICIENT_MATRIX"
            ),
            message = (
                "Delete-one-cell coefficient matrix has an inconsistent shape."
            )
        )

    if not np.all(
        np.isfinite(
            deleted_coefficient_matrix
        )
    ):
        raise CandidateFitUnavailableError(
            reason_code = (
                "NONFINITE_DELETED_COEFFICIENT_MATRIX"
            ),
            message = (
                "Delete-one-cell coefficient matrix contains non-finite values."
            )
        )

    deleted_coefficient_mean = np.mean(
        deleted_coefficient_matrix,
        axis = 0
    )

    centered_coefficients = (
        deleted_coefficient_matrix
        - deleted_coefficient_mean
    )

    deletion_count = len(
        deletion_records
    )

    covariance_matrix = (
        (
            deletion_count
            - 1
        )
        / deletion_count
        * (
            centered_coefficients.T
            @ centered_coefficients
        )
    )

    if (
        covariance_matrix.shape
        != (
            coefficient_count,
            coefficient_count
        )
        or not np.all(
            np.isfinite(
                covariance_matrix
            )
        )
    ):
        raise CandidateFitUnavailableError(
            reason_code = "INVALID_JACKKNIFE_COVARIANCE",
            message = (
                "Jackknife covariance is invalid."
            )
        )

    covariance_matrix = (
        covariance_matrix
        + covariance_matrix.T
    ) / 2.0

    if not np.all(
        np.isfinite(
            covariance_matrix
        )
    ):
        raise CandidateFitUnavailableError(
            reason_code = (
                "NONFINITE_JACKKNIFE_COVARIANCE"
            ),
            message = (
                "Symmetrized jackknife covariance contains non-finite values."
            )
        )

    covariance_diagonal = np.diag(
        covariance_matrix
    )

    if np.any(
        covariance_diagonal < 0
    ):
        raise CandidateFitUnavailableError(
            reason_code = (
                "NEGATIVE_JACKKNIFE_COVARIANCE_DIAGONAL"
            ),
            message = (
                "Jackknife covariance contains a negative diagonal."
            )
        )

    return covariance_matrix


# Adding coefficient uncertainty and cell influence diagnostics
def calculate_uncertainty_and_influence(prepared_models: dict[str, dict], fitted_models_by_candidate: dict[str, dict],
                                        coefficient_records: list[dict], modeling_policy: dict
) -> list[dict]:
    prepared_models_by_fit_identity = defaultdict(
        list
    )

    for candidate_identity, prepared_model in (
        prepared_models.items()
    ):
        if candidate_identity not in (
            fitted_models_by_candidate
        ):
            continue

        prepared_models_by_fit_identity[
            prepared_model[
                "fit_identity"
            ]
        ].append(
            prepared_model
        )

    coefficient_records_by_candidate = defaultdict(
        list
    )

    for coefficient_record in coefficient_records:
        coefficient_records_by_candidate[
            coefficient_record[
                "candidate_identity"
            ]
        ].append(
            coefficient_record
        )

    influence_records = []

    confidence_level = modeling_policy[
        "uncertainty"
    ][
        "confidence_level"
    ]

    for fit_identity in sorted(
        prepared_models_by_fit_identity
    ):
        reusable_models = sorted(
            prepared_models_by_fit_identity[
                fit_identity
            ],
            key = lambda model: model[
                "candidate_identity"
            ]
        )

        canonical_model = reusable_models[0]

        for reusable_model in reusable_models[1:]:
            validate_fit_reuse(
                first_prepared_model = canonical_model,
                second_prepared_model = reusable_model
            )

        canonical_candidate_identity = (
            canonical_model[
                "candidate_identity"
            ]
        )

        canonical_fitted_result = (
            fitted_models_by_candidate[
                canonical_candidate_identity
            ]
        )

        full_coefficients = np.asarray(
            canonical_fitted_result[
                "coefficients"
            ],
            dtype = float
        )

        deletion_records = (
            fit_deleted_cell_coefficients(
                prepared_model = canonical_model,
                modeling_policy = modeling_policy
            )
        )

        independent_cell_count = len(
            deletion_records
        )

        uncertainty_status = "completed"
        uncertainty_reason_code = None
        jackknife_covariance = None
        standard_errors = None
        critical_value = None

        try:
            jackknife_covariance = (
                calculate_jackknife_covariance(
                    deletion_records = (
                        deletion_records
                    ),
                    coefficient_count = len(
                        full_coefficients
                    )
                )
            )

            standard_errors = np.sqrt(
                np.diag(
                    jackknife_covariance
                )
            )

            if not np.all(
                np.isfinite(
                    standard_errors
                )
            ):
                raise CandidateFitUnavailableError(
                    reason_code = (
                        "NONFINITE_JACKKNIFE_STANDARD_ERROR"
                    ),
                    message = (
                        "Jackknife standard errors contain non-finite values."
                    )
                )

            degrees_of_freedom = (
                independent_cell_count
                - 1
            )

            if degrees_of_freedom <= 0:
                raise CandidateFitUnavailableError(
                    reason_code = (
                        "NONPOSITIVE_JACKKNIFE_DEGREES_OF_FREEDOM"
                    ),
                    message = (
                        "Jackknife intervals require positive degrees of freedom."
                    )
                )

            critical_value = float(
                student_t.ppf(
                    (
                        1.0
                        + confidence_level
                    )
                    / 2.0,
                    degrees_of_freedom
                )
            )

            if not math.isfinite(
                critical_value
            ):
                raise CandidateFitUnavailableError(
                    reason_code = (
                        "NONFINITE_JACKKNIFE_CRITICAL_VALUE"
                    ),
                    message = (
                        "Jackknife interval critical value is non-finite."
                    )
                )

        except CandidateFitUnavailableError as exc:
            uncertainty_status = "unavailable"
            uncertainty_reason_code = (
                exc.reason_code
            )

        for prepared_model in reusable_models:
            candidate_identity = (
                prepared_model[
                    "candidate_identity"
                ]
            )

            candidate_record = prepared_model[
                "candidate_record"
            ]

            fitted_result = (
                fitted_models_by_candidate[
                    candidate_identity
                ]
            )

            candidate_record[
                "uncertainty_status"
            ] = uncertainty_status

            candidate_record[
                "uncertainty_reason_code"
            ] = (
                uncertainty_reason_code
            )

            candidate_record[
                "influence_status"
            ] = (
                "completed"
                if uncertainty_status
                == "completed"
                else "unavailable"
            )

            candidate_record[
                "influence_reason_code"
            ] = (
                uncertainty_reason_code
            )

            fitted_result[
                "jackknife_covariance"
            ] = jackknife_covariance

            fitted_result[
                "deleted_cell_refits"
            ] = deletion_records

            fitted_result[
                "jackknife_degrees_of_freedom"
            ] = (
                independent_cell_count
                - 1
            )

            candidate_coefficient_records = sorted(
                coefficient_records_by_candidate[
                    candidate_identity
                ],
                key = lambda record: record[
                    "column_index"
                ]
            )

            if len(
                candidate_coefficient_records
            ) != len(
                full_coefficients
            ):
                raise RuntimeError(f"Coefficient inventory does not match the fitted model : {candidate_identity}")

            for coefficient_record in (
                candidate_coefficient_records
            ):
                column_index = coefficient_record[
                    "column_index"
                ]

                coefficient_record[
                    "uncertainty_status"
                ] = uncertainty_status

                coefficient_record[
                    "uncertainty_reason_code"
                ] = uncertainty_reason_code

                coefficient_record[
                    "confidence_level"
                ] = confidence_level

                coefficient_record[
                    "degrees_of_freedom"
                ] = (
                    independent_cell_count
                    - 1
                )

                if uncertainty_status == "completed":
                    standard_error = float(
                        standard_errors[
                            column_index
                        ]
                    )

                    estimate = coefficient_record[
                        "estimate"
                    ]

                    coefficient_record[
                        "standard_error"
                    ] = standard_error

                    coefficient_record[
                        "confidence_interval_lower"
                    ] = (
                        estimate
                        - critical_value
                        * standard_error
                    )

                    coefficient_record[
                        "confidence_interval_upper"
                    ] = (
                        estimate
                        + critical_value
                        * standard_error
                    )

                else:
                    coefficient_record[
                        "standard_error"
                    ] = None

                    coefficient_record[
                        "confidence_interval_lower"
                    ] = None

                    coefficient_record[
                        "confidence_interval_upper"
                    ] = None

            for deletion_record in deletion_records:
                deleted_coefficients = deletion_record[
                    "coefficients"
                ]

                coefficient_changes = None
                standardized_changes = None
                maximum_standardized_change = None

                if deleted_coefficients is not None:
                    coefficient_changes = (
                        deleted_coefficients
                        - full_coefficients
                    )

                if (
                    uncertainty_status
                    == "completed"
                    and coefficient_changes
                    is not None
                ):
                    standardized_changes = []

                    for (
                        coefficient_change,
                        standard_error
                    ) in zip(
                        coefficient_changes,
                        standard_errors,
                        strict = True
                    ):
                        if standard_error == 0:
                            standardized_changes.append(
                                None
                            )

                        else:
                            standardized_changes.append(
                                float(
                                    coefficient_change
                                    / standard_error
                                )
                            )

                    available_standardized_changes = [
                        abs(
                            standardized_change
                        )
                        for standardized_change in (
                            standardized_changes
                        )
                        if standardized_change is not None
                    ]

                    if available_standardized_changes:
                        maximum_standardized_change = max(
                            available_standardized_changes
                        )

                for (
                    column_index,
                    matrix_term
                ) in enumerate(
                    prepared_model[
                        "matrix_terms"
                    ]
                ):
                    influence_records.append(
                        {
                            "candidate_identity": (
                                candidate_identity
                            ),
                            "fit_identity": fit_identity,
                            "support_region_identity": (
                                candidate_record[
                                    "support_region_identity"
                                ]
                            ),
                            "template_name": (
                                candidate_record[
                                    "template_name"
                                ]
                            ),
                            "deleted_cell_identity": (
                                deletion_record[
                                    "deleted_cell_identity"
                                ]
                            ),
                            "column_index": column_index,
                            "column_name": (
                                matrix_term[
                                    "column_name"
                                ]
                            ),
                            "fit_status": (
                                deletion_record[
                                    "fit_status"
                                ]
                            ),
                            "fit_reason_code": (
                                deletion_record[
                                    "fit_reason_code"
                                ]
                            ),
                            "fit_error": (
                                deletion_record[
                                    "fit_error"
                                ]
                            ),
                            "full_estimate": float(
                                full_coefficients[
                                    column_index
                                ]
                            ),
                            "deleted_estimate": (
                                None
                                if deleted_coefficients is None
                                else float(
                                    deleted_coefficients[
                                        column_index
                                    ]
                                )
                            ),
                            "coefficient_change": (
                                None
                                if coefficient_changes is None
                                else float(
                                    coefficient_changes[
                                        column_index
                                    ]
                                )
                            ),
                            "standardized_coefficient_change": (
                                None
                                if standardized_changes is None
                                else standardized_changes[
                                    column_index
                                ]
                            ),
                            "maximum_absolute_standardized_coefficient_change": (
                                maximum_standardized_change
                            ),
                            "uncertainty_status": (
                                uncertainty_status
                            ),
                            "estimate_label": (
                                "associational"
                            )
                        }
                    )

    return sorted(
        influence_records,
        key = lambda record: (
            record[
                "candidate_identity"
            ],
            record[
                "deleted_cell_identity"
            ],
            record[
                "column_index"
            ]
        )
    )


# Representing an expected candidate-specific validation limitation
class CandidateValidationUnavailableError(RuntimeError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(
            message
        )

        self.reason_code = reason_code


# Determining whether one horizon lies inside a candidate support interval
def horizon_inside_candidate_support(horizon_value: float, candidate_record: dict) -> bool:
    lower_horizon = parse_finite_number(
        candidate_record[
            "support_interval_minimum"
        ],
        "support_interval_minimum"
    )

    upper_horizon = parse_finite_number(
        candidate_record[
            "support_interval_maximum"
        ],
        "support_interval_maximum"
    )

    lower_inclusive = parse_boolean(
        candidate_record[
            "support_interval_lower_inclusive"
        ],
        "support_interval_lower_inclusive"
    )

    upper_inclusive = parse_boolean(
        candidate_record[
            "support_interval_upper_inclusive"
        ],
        "support_interval_upper_inclusive"
    )

    lower_satisfied = (
        horizon_value >= lower_horizon
        if lower_inclusive
        else horizon_value > lower_horizon
    )

    upper_satisfied = (
        horizon_value <= upper_horizon
        if upper_inclusive
        else horizon_value < upper_horizon
    )

    return (
        lower_satisfied
        and upper_satisfied
    )


# Building one template-independent validation-fold identity
def build_validation_fold_identity(candidate_record: dict, fold_index: int,
                                   test_cell_identities: list[str], modeling_policy: dict
) -> str:
    validation_policy = modeling_policy[
        "validation"
    ]

    sharing_key_fields = validation_policy[
        "fold_sharing_key_fields"
    ]

    excluded_fields = set(
        validation_policy[
            "fold_sharing_excludes"
        ]
    )

    if (
        not sharing_key_fields
        or len(
            sharing_key_fields
        )
        != len(
            set(
                sharing_key_fields
            )
        )
    ):
        raise RuntimeError("Validation fold-sharing fields must be unique and nonempty.")

    conflicting_fields = sorted(
        set(
            sharing_key_fields
        ).intersection(
            excluded_fields
        )
    )

    if conflicting_fields:
        raise RuntimeError(f"Validation fold-sharing fields contain explicitly excluded fields : {conflicting_fields}")

    missing_fields = sorted(
        set(
            sharing_key_fields
        ).difference(
            candidate_record
        )
    )

    if missing_fields:
        raise RuntimeError(f"Candidate is missing validation fold-sharing fields : {missing_fields}")

    if (
        not isinstance(
            fold_index,
            int
        )
        or isinstance(
            fold_index,
            bool
        )
        or fold_index < 0
    ):
        raise ValueError("Validation fold index must be a nonnegative integer.")

    normalized_test_cell_identities = sorted(
        test_cell_identities
    )

    if (
        not normalized_test_cell_identities
        or len(
            normalized_test_cell_identities
        )
        != len(
            set(
                normalized_test_cell_identities
            )
        )
        or any(
            not isinstance(
                cell_identity,
                str
            )
            or not cell_identity
            for cell_identity in (
                normalized_test_cell_identities
            )
        )
    ):
        raise RuntimeError("Validation fold identity requires unique nonempty test-cell identities.")

    sharing_key = {
        field_name: candidate_record[
            field_name
        ]
        for field_name in sharing_key_fields
    }

    return build_deterministic_identity(
        {
            "fold_sharing_key": sharing_key,
            "fold_index": fold_index,
            "test_cell_identities": (
                normalized_test_cell_identities
            )
        }
    )


# Constructing deterministic condition-stratified cell folds
def build_validation_fold_assignments(prepared_model: dict, modeling_policy: dict) -> tuple[int, dict[str, int]]:
    validation_policy = modeling_policy[
        "validation"
    ]

    condition_by_cell = {}

    for observation_record in prepared_model[
        "observation_records"
    ]:
        cell_identity = observation_record[
            "cell_identity"
        ]

        condition_identity = observation_record[
            "condition_protocol_identity"
        ]

        existing_condition = condition_by_cell.get(
            cell_identity
        )

        if (
            existing_condition is not None
            and existing_condition
            != condition_identity
        ):
            raise RuntimeError(f"One validation cell belongs to multiple condition protocols : {cell_identity}")

        condition_by_cell[
            cell_identity
        ] = condition_identity

    cells_by_condition = defaultdict(
        list
    )

    for (
        cell_identity,
        condition_identity
    ) in condition_by_cell.items():
        cells_by_condition[
            condition_identity
        ].append(
            cell_identity
        )

    if not cells_by_condition:
        raise CandidateValidationUnavailableError(
            reason_code = "NO_VALIDATION_CELLS",
            message = (
                "Grouped validation does not contain cells."
            )
        )

    smallest_condition_cell_count = min(
        len(
            condition_cells
        )
        for condition_cells in (
            cells_by_condition.values()
        )
    )

    fold_count = min(
        validation_policy[
            "maximum_fold_count"
        ],
        smallest_condition_cell_count
    )

    if (
        fold_count
        < validation_policy[
            "minimum_fold_count"
        ]
    ):
        raise CandidateValidationUnavailableError(
            reason_code = (
                "INSUFFICIENT_CELLS_FOR_GROUPED_VALIDATION"
            ),
            message = (
                "At least one condition does not contain enough cells for grouped validation."
            )
        )

    fold_by_cell = {}

    for condition_identity in sorted(
        cells_by_condition
    ):
        condition_cells = sorted(
            cells_by_condition[
                condition_identity
            ],
            key = lambda cell_identity: (
                build_deterministic_identity(
                    {
                        "condition_protocol_identity": (
                            condition_identity
                        ),
                        "cell_identity": cell_identity
                    }
                ),
                cell_identity
            )
        )

        for cell_index, cell_identity in enumerate(
            condition_cells
        ):
            fold_by_cell[
                cell_identity
            ] = (
                cell_index
                % fold_count
            )

    all_conditions = set(
        cells_by_condition
    )

    for fold_index in range(
        fold_count
    ):
        test_conditions = {
            condition_by_cell[
                cell_identity
            ]
            for (
                cell_identity,
                assigned_fold
            ) in fold_by_cell.items()
            if assigned_fold == fold_index
        }

        training_conditions = {
            condition_by_cell[
                cell_identity
            ]
            for (
                cell_identity,
                assigned_fold
            ) in fold_by_cell.items()
            if assigned_fold != fold_index
        }

        if test_conditions != all_conditions:
            raise CandidateValidationUnavailableError(
                reason_code = (
                    "TEST_FOLD_MISSING_CONDITION_PROTOCOL"
                ),
                message = (
                    "A validation test fold does not represent every condition protocol."
                )
            )

        if training_conditions != all_conditions:
            raise CandidateValidationUnavailableError(
                reason_code = (
                    "TRAINING_FOLD_MISSING_CONDITION_PROTOCOL"
                ),
                message = (
                    "A validation training fold does not represent every condition protocol."
                )
            )

    return (
        fold_count,
        fold_by_cell
    )


# Building one training-fold model from training observations only
def build_validation_training_model(prepared_model: dict, training_observations: list[dict],
                                    modeling_policy: dict
) -> dict:
    candidate_record = prepared_model[
        "candidate_record"
    ]

    try:
        training_encoding = build_model_encoding(
            candidate_record = candidate_record,
            candidate_term_records = (
                prepared_model[
                    "candidate_term_records"
                ]
            ),
            observation_records = (
                training_observations
            ),
            modeling_policy = modeling_policy
        )

        (
            training_matrix,
            training_matrix_terms
        ) = transform_model_matrix(
            candidate_record = candidate_record,
            observation_records = (
                training_observations
            ),
            model_encoding = training_encoding
        )

    except CandidateMatrixUnavailableError as exc:
        raise CandidateValidationUnavailableError(
            reason_code = exc.reason_code,
            message = str(
                exc
            )
        ) from exc

    training_outcomes = np.asarray(
        [
            observation_record[
                "outcome_value"
            ]
            for observation_record in (
                training_observations
            )
        ],
        dtype = float
    )

    training_weights = build_equal_cell_weights(
        training_observations
    )

    admissibility_result = (
        evaluate_candidate_admissibility(
            candidate_record = candidate_record,
            observation_records = (
                training_observations
            ),
            model_matrix = training_matrix,
            outcome_values = training_outcomes,
            regression_weights = training_weights,
            model_encoding = training_encoding,
            modeling_policy = modeling_policy
        )
    )

    if (
        admissibility_result[
            "admissibility_status"
        ]
        != "admissible"
    ):
        raise CandidateValidationUnavailableError(
            reason_code = (
                f"TRAINING_FOLD_{admissibility_result['admissibility_reason_code']}"
            ),
            message = (
                f"Validation training fold is inadmissible : {admissibility_result['admissibility_reason_code']}"
            )
        )

    return {
        "candidate_identity": (
            prepared_model[
                "candidate_identity"
            ]
        ),
        "fit_identity": (
            prepared_model[
                "fit_identity"
            ]
        ),
        "candidate_record": candidate_record,
        "candidate_term_records": (
            prepared_model[
                "candidate_term_records"
            ]
        ),
        "observation_records": (
            training_observations
        ),
        "model_encoding": training_encoding,
        "model_matrix": training_matrix,
        "matrix_terms": training_matrix_terms,
        "outcome_values": training_outcomes,
        "regression_weights": training_weights,
        "cell_identities": [
            observation_record[
                "cell_identity"
            ]
            for observation_record in (
                training_observations
            )
        ],
        "condition_protocol_identities": [
            observation_record[
                "condition_protocol_identity"
            ]
            for observation_record in (
                training_observations
            )
        ],
        "admissibility_result": (
            admissibility_result
        )
    }


# Constructing fixed-effect covariance from one validation training fold
def build_validation_fixed_effect_covariance(training_prepared_model: dict, training_fit: dict,
                                             modeling_policy: dict
) -> np.ndarray:
    estimator_name = training_prepared_model[
        "candidate_record"
    ][
        "estimator"
    ]

    coefficient_count = (
        training_prepared_model[
            "model_matrix"
        ].shape[1]
    )

    if estimator_name == "weighted_least_squares":
        deletion_records = (
            fit_deleted_cell_coefficients(
                prepared_model = (
                    training_prepared_model
                ),
                modeling_policy = modeling_policy
            )
        )

        try:
            covariance_matrix = (
                calculate_jackknife_covariance(
                    deletion_records = deletion_records,
                    coefficient_count = (
                        coefficient_count
                    )
                )
            )

        except CandidateFitUnavailableError as exc:
            raise CandidateValidationUnavailableError(
                reason_code = (
                    f"TRAINING_FOLD_INTERVAL_{exc.reason_code}"
                ),
                message = str(
                    exc
                )
            ) from exc

    elif estimator_name == "linear_mixed_effects":
        covariance_matrix = training_fit[
            "model_based_fixed_effect_covariance"
        ]

    else:
        raise RuntimeError(f"Validation covariance received an unsupported estimator : {estimator_name}")

    return validate_validation_covariance(
        covariance_matrix = covariance_matrix,
        coefficient_count = coefficient_count,
        modeling_policy = modeling_policy
    )


# Validating one training-fold fixed-effect covariance
def validate_validation_covariance(covariance_matrix: np.ndarray, coefficient_count: int,
                                   modeling_policy: dict
) -> np.ndarray:
    covariance_policy = modeling_policy[
        "validation"
    ][
        "training_fold_fixed_effect_covariance"
    ]

    covariance_matrix = np.asarray(
        covariance_matrix,
        dtype = float
    )

    if covariance_matrix.shape != (
        coefficient_count,
        coefficient_count
    ):
        raise CandidateValidationUnavailableError(
            reason_code = (
                "INVALID_TRAINING_COVARIANCE_DIMENSION"
            ),
            message = (
                "Training-fold covariance has an invalid dimension."
            )
        )

    if not np.all(
        np.isfinite(
            covariance_matrix
        )
    ):
        raise CandidateValidationUnavailableError(
            reason_code = (
                "NONFINITE_TRAINING_COVARIANCE"
            ),
            message = (
                "Training-fold covariance contains non-finite values."
            )
        )

    symmetry_tolerance = covariance_policy[
        "symmetry_absolute_tolerance"
    ]

    if not np.allclose(
        covariance_matrix,
        covariance_matrix.T,
        rtol = 0.0,
        atol = symmetry_tolerance
    ):
        raise CandidateValidationUnavailableError(
            reason_code = (
                "NONSYMMETRIC_TRAINING_COVARIANCE"
            ),
            message = (
                "Training-fold covariance is not symmetric within the configured tolerance."
            )
        )

    covariance_matrix = (
        covariance_matrix
        + covariance_matrix.T
    ) / 2.0

    negative_diagonal_tolerance = covariance_policy[
        "negative_diagonal_tolerance"
    ]

    covariance_diagonal = np.diag(
        covariance_matrix
    )

    if np.any(
        covariance_diagonal
        < -negative_diagonal_tolerance
    ):
        raise CandidateValidationUnavailableError(
            reason_code = (
                "NEGATIVE_TRAINING_COVARIANCE_DIAGONAL"
            ),
            message = (
                "Training-fold covariance contains a materially negative diagonal."
            )
        )

    for diagonal_index in range(
        coefficient_count
    ):
        if (
            covariance_matrix[
                diagonal_index,
                diagonal_index
            ]
            < 0
        ):
            covariance_matrix[
                diagonal_index,
                diagonal_index
            ] = 0.0

    return covariance_matrix


# Constructing new-cell prediction intervals from training data only
def build_validation_prediction_intervals(prediction_matrix: np.ndarray, training_prepared_model: dict,
                                          training_fit: dict, training_cell_count: int,
                                          modeling_policy: dict
) -> tuple[
    list[float | None],
    list[float | None],
    str,
    str | None
]:
    prediction_count = prediction_matrix.shape[
        0
    ]

    unavailable_lower = [
        None
        for _ in range(
            prediction_count
        )
    ]

    unavailable_upper = [
        None
        for _ in range(
            prediction_count
        )
    ]

    try:
        covariance_matrix = (
            build_validation_fixed_effect_covariance(
                training_prepared_model = (
                    training_prepared_model
                ),
                training_fit = training_fit,
                modeling_policy = modeling_policy
            )
        )

        observation_variance = float(
            training_fit[
                "observation_variance"
            ]
        )

        if (
            not math.isfinite(
                observation_variance
            )
            or observation_variance < 0
        ):
            raise CandidateValidationUnavailableError(
                reason_code = (
                    "INVALID_TRAINING_OBSERVATION_VARIANCE"
                ),
                message = (
                    "Training-fold observation variance is invalid."
                )
            )

        degrees_of_freedom = (
            training_cell_count
            - 1
        )

        if degrees_of_freedom <= 0:
            raise CandidateValidationUnavailableError(
                reason_code = (
                    "NONPOSITIVE_VALIDATION_INTERVAL_DEGREES_OF_FREEDOM"
                ),
                message = (
                    "Validation prediction intervals require positive cell-level degrees of freedom."
                )
            )

        confidence_level = modeling_policy[
            "prediction_intervals"
        ][
            "confidence_level"
        ]

        critical_value = float(
            student_t.ppf(
                (
                    1.0
                    + confidence_level
                )
                / 2.0,
                degrees_of_freedom
            )
        )

        if not math.isfinite(
            critical_value
        ):
            raise CandidateValidationUnavailableError(
                reason_code = (
                    "NONFINITE_VALIDATION_INTERVAL_CRITICAL_VALUE"
                ),
                message = (
                    "Validation interval critical value is non-finite."
                )
            )

        fixed_effect_variances = np.einsum(
            "ij,jk,ik->i",
            prediction_matrix,
            covariance_matrix,
            prediction_matrix
        )

        negative_variance_tolerance = (
            modeling_policy[
                "validation"
            ][
                "training_fold_fixed_effect_covariance"
            ][
                "negative_diagonal_tolerance"
            ]
        )

        if np.any(
            fixed_effect_variances
            < -negative_variance_tolerance
        ):
            raise CandidateValidationUnavailableError(
                reason_code = (
                    "NEGATIVE_VALIDATION_FIXED_EFFECT_VARIANCE"
                ),
                message = (
                    "Validation fixed-effect prediction variance is materially negative."
                )
            )

        fixed_effect_variances = np.maximum(
            fixed_effect_variances,
            0.0
        )

        total_variances = (
            fixed_effect_variances
            + observation_variance
        )

        if (
            not np.all(
                np.isfinite(
                    total_variances
                )
            )
            or np.any(
                total_variances < 0
            )
        ):
            raise CandidateValidationUnavailableError(
                reason_code = (
                    "INVALID_VALIDATION_PREDICTION_VARIANCE"
                ),
                message = (
                    "Validation prediction variance is invalid."
                )
            )

        interval_half_widths = (
            critical_value
            * np.sqrt(
                total_variances
            )
        )

        return (
            interval_half_widths.tolist(),
            interval_half_widths.tolist(),
            "completed",
            None
        )

    except CandidateValidationUnavailableError as exc:
        return (
            unavailable_lower,
            unavailable_upper,
            "unavailable",
            exc.reason_code
        )


# Calculating equally cell-weighted validation metrics
def calculate_grouped_validation_metrics(prediction_records: list[dict]) -> dict:
    if not prediction_records:
        raise CandidateValidationUnavailableError(
            reason_code = "NO_VALIDATION_PREDICTIONS",
            message = (
                "Validation metrics require predictions."
            )
        )

    predictions_by_cell = defaultdict(
        list
    )

    observed_prediction_keys = set()

    for prediction_record in prediction_records:
        prediction_key = (
            prediction_record[
                "candidate_identity"
            ],
            prediction_record[
                "source_identity"
            ]
        )

        if prediction_key in observed_prediction_keys:
            raise RuntimeError(f"Validation predictions contain a duplicate candidate/source identity : {prediction_key}")

        observed_prediction_keys.add(
            prediction_key
        )

        predictions_by_cell[
            prediction_record[
                "cell_identity"
            ]
        ].append(
            prediction_record
        )

    cell_absolute_errors = []

    cell_squared_errors = []

    cell_signed_errors = []

    cell_observed_means = []

    for cell_identity in sorted(
        predictions_by_cell
    ):
        cell_predictions = predictions_by_cell[
            cell_identity
        ]

        observed_values = np.asarray(
            [
                prediction_record[
                    "observed_outcome"
                ]
                for prediction_record in (
                    cell_predictions
                )
            ],
            dtype = float
        )

        predicted_values = np.asarray(
            [
                prediction_record[
                    "predicted_outcome"
                ]
                for prediction_record in (
                    cell_predictions
                )
            ],
            dtype = float
        )

        errors = (
            predicted_values
            - observed_values
        )

        cell_absolute_errors.append(
            float(
                np.mean(
                    np.abs(
                        errors
                    )
                )
            )
        )

        cell_squared_errors.append(
            float(
                np.mean(
                    errors ** 2
                )
            )
        )

        cell_signed_errors.append(
            float(
                np.mean(
                    errors
                )
            )
        )

        cell_observed_means.append(
            float(
                np.mean(
                    observed_values
                )
            )
        )

    equal_cell_observed_mean = float(
        np.mean(
            cell_observed_means
        )
    )

    cell_total_squared_deviations = []

    for cell_identity in sorted(
        predictions_by_cell
    ):
        observed_values = np.asarray(
            [
                prediction_record[
                    "observed_outcome"
                ]
                for prediction_record in (
                    predictions_by_cell[
                        cell_identity
                    ]
                )
            ],
            dtype = float
        )

        cell_total_squared_deviations.append(
            float(
                np.mean(
                    (
                        observed_values
                        - equal_cell_observed_mean
                    ) ** 2
                )
            )
        )

    mean_squared_error = float(
        np.mean(
            cell_squared_errors
        )
    )

    total_squared_deviation = float(
        np.mean(
            cell_total_squared_deviations
        )
    )

    predictive_r_squared = (
        None
        if total_squared_deviation == 0
        else (
            1.0
            - mean_squared_error
            / total_squared_deviation
        )
    )

    intervals_complete = all(
        prediction_record[
            "prediction_interval_status"
        ]
        == "completed"
        and prediction_record[
            "prediction_interval_lower"
        ]
        is not None
        and prediction_record[
            "prediction_interval_upper"
        ]
        is not None
        for prediction_record in prediction_records
    )

    if intervals_complete:
        cell_coverages = []

        cell_interval_widths = []

        for cell_identity in sorted(
            predictions_by_cell
        ):
            cell_predictions = predictions_by_cell[
                cell_identity
            ]

            cell_coverages.append(
                float(
                    np.mean(
                        [
                            (
                                prediction_record[
                                    "prediction_interval_lower"
                                ]
                                <= prediction_record[
                                    "observed_outcome"
                                ]
                                <= prediction_record[
                                    "prediction_interval_upper"
                                ]
                            )
                            for prediction_record in (
                                cell_predictions
                            )
                        ]
                    )
                )
            )

            cell_interval_widths.append(
                float(
                    np.mean(
                        [
                            (
                                prediction_record[
                                    "prediction_interval_upper"
                                ]
                                - prediction_record[
                                    "prediction_interval_lower"
                                ]
                            )
                            for prediction_record in (
                                cell_predictions
                            )
                        ]
                    )
                )
            )

        prediction_interval_coverage = float(
            np.mean(
                cell_coverages
            )
        )

        mean_prediction_interval_width = float(
            np.mean(
                cell_interval_widths
            )
        )

        interval_metric_status = "completed"

    else:
        prediction_interval_coverage = None
        mean_prediction_interval_width = None
        interval_metric_status = "unavailable"

    metric_result = {
        "independent_cell_count": len(
            predictions_by_cell
        ),
        "observation_count": len(
            prediction_records
        ),
        "mean_absolute_error": float(
            np.mean(
                cell_absolute_errors
            )
        ),
        "root_mean_squared_error": math.sqrt(
            mean_squared_error
        ),
        "mean_error": float(
            np.mean(
                cell_signed_errors
            )
        ),
        "predictive_r_squared": (
            predictive_r_squared
        ),
        "predictive_r_squared_status": (
            "completed"
            if predictive_r_squared is not None
            else "unavailable_zero_total_variation"
        ),
        "prediction_interval_coverage": (
            prediction_interval_coverage
        ),
        "mean_prediction_interval_width": (
            mean_prediction_interval_width
        ),
        "interval_metric_status": (
            interval_metric_status
        )
    }

    numerical_metric_fields = [
        "mean_absolute_error",
        "root_mean_squared_error",
        "mean_error",
        "predictive_r_squared",
        "prediction_interval_coverage",
        "mean_prediction_interval_width"
    ]

    nonfinite_metric_fields = [
        field_name
        for field_name in numerical_metric_fields
        if (
            metric_result[
                field_name
            ]
            is not None
            and not math.isfinite(
                metric_result[
                    field_name
                ]
            )
        )
    ]

    if nonfinite_metric_fields:
        raise CandidateValidationUnavailableError(
            reason_code = (
                "NONFINITE_VALIDATION_METRIC"
            ),
            message = (
                f"Grouped validation produced non-finite metrics : {nonfinite_metric_fields}"
            )
        )

    return metric_result


# Building one unavailable validation-metric record
def build_unavailable_validation_metric_record(prepared_model: dict, reason_code: str) -> dict:
    candidate_record = prepared_model[
        "candidate_record"
    ]

    return {
        "candidate_identity": (
            prepared_model[
                "candidate_identity"
            ]
        ),
        "fit_identity": (
            prepared_model[
                "fit_identity"
            ]
        ),
        "support_region_identity": (
            candidate_record[
                "support_region_identity"
            ]
        ),
        "template_name": (
            candidate_record[
                "template_name"
            ]
        ),
        "validation_status": "unavailable",
        "validation_reason_code": reason_code,
        "independent_cell_count": None,
        "observation_count": None,
        "mean_absolute_error": None,
        "root_mean_squared_error": None,
        "mean_error": None,
        "predictive_r_squared": None,
        "predictive_r_squared_status": (
            "unavailable"
        ),
        "prediction_interval_coverage": None,
        "mean_prediction_interval_width": None,
        "interval_metric_status": (
            "unavailable"
        ),
        "estimate_label": "predictive"
    }


# Calculating the maximum declared fitting workload
def build_modeling_execution_plan(prepared_models: dict[str, dict], modeling_policy: dict) -> dict:
    canonical_models_by_fit_identity = {}

    for candidate_identity in sorted(
        prepared_models
    ):
        prepared_model = prepared_models[
            candidate_identity
        ]

        fit_identity = prepared_model[
            "fit_identity"
        ]

        canonical_model = (
            canonical_models_by_fit_identity.get(
                fit_identity
            )
        )

        if canonical_model is None:
            canonical_models_by_fit_identity[
                fit_identity
            ] = prepared_model

        else:
            validate_fit_reuse(
                first_prepared_model = canonical_model,
                second_prepared_model = prepared_model
            )

    unique_fit_counts_by_estimator = Counter()

    full_data_deletion_refit_upper_bound = 0

    for prepared_model in (
        canonical_models_by_fit_identity.values()
    ):
        estimator_name = prepared_model[
            "candidate_record"
        ][
            "estimator"
        ]

        unique_fit_counts_by_estimator[
            estimator_name
        ] += 1

        full_data_deletion_refit_upper_bound += len(
            set(
                prepared_model[
                    "cell_identities"
                ]
            )
        )

    validation_fold_fit_upper_bound = 0
    validation_interval_deletion_refit_upper_bound = 0
    validation_assignment_unavailable_count = 0

    for candidate_identity in sorted(
        prepared_models
    ):
        prepared_model = prepared_models[
            candidate_identity
        ]

        try:
            (
                fold_count,
                fold_by_cell
            ) = build_validation_fold_assignments(
                prepared_model = prepared_model,
                modeling_policy = modeling_policy
            )

        except CandidateValidationUnavailableError:
            validation_assignment_unavailable_count += 1
            continue

        validation_fold_fit_upper_bound += (
            fold_count
        )

        if (
            prepared_model[
                "candidate_record"
            ][
                "estimator"
            ]
            != "weighted_least_squares"
        ):
            continue

        for fold_index in range(
            fold_count
        ):
            training_cell_identities = {
                cell_identity
                for cell_identity in prepared_model[
                    "cell_identities"
                ]
                if (
                    fold_by_cell[
                        cell_identity
                    ]
                    != fold_index
                )
            }

            validation_interval_deletion_refit_upper_bound += len(
                training_cell_identities
            )

    unique_full_fit_count = len(
        canonical_models_by_fit_identity
    )

    total_fit_call_upper_bound = (
        unique_full_fit_count
        + full_data_deletion_refit_upper_bound
        + validation_fold_fit_upper_bound
        + validation_interval_deletion_refit_upper_bound
    )

    return {
        "prepared_candidate_count": len(
            prepared_models
        ),
        "unique_full_fit_count": (
            unique_full_fit_count
        ),
        "unique_weighted_least_squares_fit_count": (
            unique_fit_counts_by_estimator[
                "weighted_least_squares"
            ]
        ),
        "unique_linear_mixed_effects_fit_count": (
            unique_fit_counts_by_estimator[
                "linear_mixed_effects"
            ]
        ),
        "full_data_deletion_refit_upper_bound": (
            full_data_deletion_refit_upper_bound
        ),
        "validation_fold_fit_upper_bound": (
            validation_fold_fit_upper_bound
        ),
        "validation_interval_deletion_refit_upper_bound": (
            validation_interval_deletion_refit_upper_bound
        ),
        "validation_assignment_unavailable_count": (
            validation_assignment_unavailable_count
        ),
        "total_fit_call_upper_bound": (
            total_fit_call_upper_bound
        )
    }


# Performing deterministic grouped validation
def run_grouped_validation(prepared_models: dict[str, dict], fitted_models_by_candidate: dict[str, dict],
                           modeling_policy: dict
) -> tuple[list[dict], list[dict], list[dict]]:
    fold_records = []

    prediction_records = []

    metric_records = []

    for candidate_identity in sorted(
        prepared_models
    ):
        prepared_model = prepared_models[
            candidate_identity
        ]

        candidate_record = prepared_model[
            "candidate_record"
        ]

        if candidate_identity not in (
            fitted_models_by_candidate
        ):
            continue

        try:
            (
                fold_count,
                fold_by_cell
            ) = build_validation_fold_assignments(
                prepared_model = prepared_model,
                modeling_policy = modeling_policy
            )

        except CandidateValidationUnavailableError as exc:
            candidate_record[
                "validation_status"
            ] = "unavailable"

            candidate_record[
                "validation_reason_code"
            ] = exc.reason_code

            candidate_record[
                "validation_interval_status"
            ] = "unavailable"

            metric_records.append(
                build_unavailable_validation_metric_record(
                    prepared_model = prepared_model,
                    reason_code = exc.reason_code
                )
            )

            continue

        candidate_predictions = []

        candidate_fold_records = []

        fold_failure_reason_codes = []

        for fold_index in range(
            fold_count
        ):
            training_observations = [
                observation_record
                for observation_record in (
                    prepared_model[
                        "observation_records"
                    ]
                )
                if fold_by_cell[
                    observation_record[
                        "cell_identity"
                    ]
                ] != fold_index
            ]

            raw_test_observations = [
                observation_record
                for observation_record in (
                    prepared_model[
                        "observation_records"
                    ]
                )
                if fold_by_cell[
                    observation_record[
                        "cell_identity"
                    ]
                ] == fold_index
            ]

            test_observations = [
                observation_record
                for observation_record in (
                    raw_test_observations
                )
                if observation_record[
                    "inside_evaluation_interval"
                ]
            ]

            training_cell_identities = sorted(
                {
                    observation_record[
                        "cell_identity"
                    ]
                    for observation_record in (
                        training_observations
                    )
                }
            )

            test_cell_identities = sorted(
                {
                    observation_record[
                        "cell_identity"
                    ]
                    for observation_record in (
                        raw_test_observations
                    )
                }
            )

            fold_identity = (
                build_validation_fold_identity(
                    candidate_record = candidate_record,
                    fold_index = fold_index,
                    test_cell_identities = (
                        test_cell_identities
                    ),
                    modeling_policy = modeling_policy
                )
            )

            fold_record = {
                "candidate_identity": candidate_identity,
                "fit_identity": (
                    prepared_model[
                        "fit_identity"
                    ]
                ),
                "support_region_identity": (
                    candidate_record[
                        "support_region_identity"
                    ]
                ),
                "template_name": (
                    candidate_record[
                        "template_name"
                    ]
                ),
                "fold_identity": fold_identity,
                "fold_index": fold_index,
                "training_cell_count": len(
                    training_cell_identities
                ),
                "test_cell_count": len(
                    test_cell_identities
                ),
                "training_cell_identities": (
                    training_cell_identities
                ),
                "test_cell_identities": (
                    test_cell_identities
                ),
                "training_observation_count": len(
                    training_observations
                ),
                "raw_test_observation_count": len(
                    raw_test_observations
                ),
                "test_observation_count": len(
                    test_observations
                ),
                "outside_support_test_observation_count": (
                    len(
                        raw_test_observations
                    )
                    - len(
                        test_observations
                    )
                ),
                "fold_status": "pending",
                "fold_reason_code": None,
                "prediction_interval_status": (
                    "not_attempted"
                ),
                "prediction_interval_reason_code": None
            }

            try:
                if not test_observations:
                    raise CandidateValidationUnavailableError(
                        reason_code = (
                            "NO_TEST_OBSERVATIONS_IN_SUPPORT"
                        ),
                        message = (
                            "Validation fold does not contain a test observation inside the support region."
                        )
                    )

                training_horizons_by_condition = defaultdict(
                    list
                )

                for observation_record in (
                    training_observations
                ):
                    training_horizons_by_condition[
                        observation_record[
                            "condition_protocol_identity"
                        ]
                    ].append(
                        observation_record[
                            "horizon"
                        ]
                    )

                for observation_record in (
                    test_observations
                ):
                    condition_identity = (
                        observation_record[
                            "condition_protocol_identity"
                        ]
                    )

                    condition_training_horizons = (
                        training_horizons_by_condition.get(
                            condition_identity,
                            []
                        )
                    )

                    if not condition_training_horizons:
                        raise CandidateValidationUnavailableError(
                            reason_code = (
                                "TEST_CONDITION_ABSENT_FROM_TRAINING"
                            ),
                            message = (
                                "Validation condition is absent from training observations."
                            )
                        )

                    if not (
                        min(
                            condition_training_horizons
                        )
                        <= observation_record[
                            "horizon"
                        ]
                        <= max(
                            condition_training_horizons
                        )
                    ):
                        raise CandidateValidationUnavailableError(
                            reason_code = (
                                "TEST_HORIZON_OUTSIDE_CONDITION_TRAINING_RANGE"
                            ),
                            message = (
                                "A held-out observation lies outside the same-condition training horizon range."
                            )
                        )

                training_prepared_model = (
                    build_validation_training_model(
                        prepared_model = prepared_model,
                        training_observations = (
                            training_observations
                        ),
                        modeling_policy = modeling_policy
                    )
                )

                try:
                    training_fit = fit_prepared_model(
                        prepared_model = (
                            training_prepared_model
                        ),
                        modeling_policy = modeling_policy
                    )

                except CandidateFitUnavailableError as exc:
                    raise CandidateValidationUnavailableError(
                        reason_code = (
                            f"TRAINING_FOLD_{exc.reason_code}"
                        ),
                        message = str(
                            exc
                        )
                    ) from exc

                try:
                    (
                        test_matrix,
                        test_matrix_terms
                    ) = transform_model_matrix(
                        candidate_record = candidate_record,
                        observation_records = (
                            test_observations
                        ),
                        model_encoding = (
                            training_prepared_model[
                                "model_encoding"
                            ]
                        )
                    )

                except CandidateMatrixUnavailableError as exc:
                    raise CandidateValidationUnavailableError(
                        reason_code = exc.reason_code,
                        message = str(
                            exc
                        )
                    ) from exc

                if (
                    test_matrix_terms
                    != training_prepared_model[
                        "matrix_terms"
                    ]
                ):
                    raise RuntimeError("Validation training and prediction matrices contain different terms.")

                predicted_outcomes = (
                    test_matrix
                    @ training_fit[
                        "coefficients"
                    ]
                )

                if not np.all(
                    np.isfinite(
                        predicted_outcomes
                    )
                ):
                    raise CandidateValidationUnavailableError(
                        reason_code = (
                            "NONFINITE_VALIDATION_PREDICTION"
                        ),
                        message = (
                            "Validation produced a non-finite prediction."
                        )
                    )

                (
                    interval_half_widths_lower,
                    interval_half_widths_upper,
                    interval_status,
                    interval_reason_code
                ) = build_validation_prediction_intervals(
                    prediction_matrix = test_matrix,
                    training_prepared_model = (
                        training_prepared_model
                    ),
                    training_fit = training_fit,
                    training_cell_count = (
                        fold_record[
                            "training_cell_count"
                        ]
                    ),
                    modeling_policy = modeling_policy
                )

                fold_record[
                    "fold_status"
                ] = "completed"

                fold_record[
                    "prediction_interval_status"
                ] = interval_status

                fold_record[
                    "prediction_interval_reason_code"
                ] = interval_reason_code

                for (
                    observation_record,
                    predicted_outcome,
                    lower_half_width,
                    upper_half_width
                ) in zip(
                    test_observations,
                    predicted_outcomes,
                    interval_half_widths_lower,
                    interval_half_widths_upper,
                    strict = True
                ):
                    prediction_records_available = (
                        interval_status
                        == "completed"
                    )

                    candidate_predictions.append(
                        {
                            "candidate_identity": (
                                candidate_identity
                            ),
                            "fit_identity": (
                                prepared_model[
                                    "fit_identity"
                                ]
                            ),
                            "support_region_identity": (
                                candidate_record[
                                    "support_region_identity"
                                ]
                            ),
                            "template_name": (
                                candidate_record[
                                    "template_name"
                                ]
                            ),
                            "fold_identity": fold_identity,
                            "fold_index": fold_index,
                            "source_identity": (
                                observation_record[
                                    "source_identity"
                                ]
                            ),
                            "cell_identity": (
                                observation_record[
                                    "cell_identity"
                                ]
                            ),
                            "condition_protocol_identity": (
                                observation_record[
                                    "condition_protocol_identity"
                                ]
                            ),
                            "horizon": (
                                observation_record[
                                    "horizon"
                                ]
                            ),
                            "observed_outcome": (
                                observation_record[
                                    "outcome_value"
                                ]
                            ),
                            "predicted_outcome": float(
                                predicted_outcome
                            ),
                            "prediction_error": float(
                                predicted_outcome
                                - observation_record[
                                    "outcome_value"
                                ]
                            ),
                            "prediction_interval_status": (
                                interval_status
                            ),
                            "prediction_interval_reason_code": (
                                interval_reason_code
                            ),
                            "prediction_interval_lower": (
                                None
                                if not prediction_records_available
                                else float(
                                    predicted_outcome
                                    - lower_half_width
                                )
                            ),
                            "prediction_interval_upper": (
                                None
                                if not prediction_records_available
                                else float(
                                    predicted_outcome
                                    + upper_half_width
                                )
                            ),
                            "estimate_label": "predictive"
                        }
                    )

            except CandidateValidationUnavailableError as exc:
                fold_record[
                    "fold_status"
                ] = "unavailable"

                fold_record[
                    "fold_reason_code"
                ] = exc.reason_code

                fold_failure_reason_codes.append(
                    exc.reason_code
                )

            candidate_fold_records.append(
                fold_record
            )

        fold_records.extend(
            candidate_fold_records
        )

        if fold_failure_reason_codes:
            candidate_record[
                "validation_status"
            ] = "unavailable"

            candidate_record[
                "validation_reason_code"
            ] = fold_failure_reason_codes[0]

            candidate_record[
                "validation_interval_status"
            ] = "unavailable"

            metric_records.append(
                build_unavailable_validation_metric_record(
                    prepared_model = prepared_model,
                    reason_code = (
                        fold_failure_reason_codes[0]
                    )
                )
            )

            continue

        try:
            validation_metrics = (
                calculate_grouped_validation_metrics(
                    candidate_predictions
                )
            )

        except CandidateValidationUnavailableError as exc:
            candidate_record[
                "validation_status"
            ] = "unavailable"

            candidate_record[
                "validation_reason_code"
            ] = exc.reason_code

            candidate_record[
                "validation_interval_status"
            ] = "unavailable"

            metric_records.append(
                build_unavailable_validation_metric_record(
                    prepared_model = prepared_model,
                    reason_code = exc.reason_code
                )
            )

            continue

        candidate_record[
            "validation_status"
        ] = "completed"

        candidate_record[
            "validation_reason_code"
        ] = None

        candidate_record[
            "validation_interval_status"
        ] = validation_metrics[
            "interval_metric_status"
        ]

        prediction_records.extend(
            candidate_predictions
        )

        metric_records.append(
            {
                "candidate_identity": candidate_identity,
                "fit_identity": (
                    prepared_model[
                        "fit_identity"
                    ]
                ),
                "support_region_identity": (
                    candidate_record[
                        "support_region_identity"
                    ]
                ),
                "template_name": (
                    candidate_record[
                        "template_name"
                    ]
                ),
                "validation_status": "completed",
                "validation_reason_code": None,
                **validation_metrics,
                "estimate_label": "predictive"
            }
        )

    return (
        sorted(
            fold_records,
            key = lambda record: (
                record[
                    "candidate_identity"
                ],
                record[
                    "fold_index"
                ]
            )
        ),
        sorted(
            prediction_records,
            key = lambda record: (
                record[
                    "candidate_identity"
                ],
                record[
                    "cell_identity"
                ],
                record[
                    "horizon"
                ],
                record[
                    "source_identity"
                ]
            )
        ),
        sorted(
            metric_records,
            key = lambda record: record[
                "candidate_identity"
            ]
        )
    )


# Calculating one equal-cell mean
def calculate_equal_cell_mean(records: list[dict], value_field: str) -> float:
    values_by_cell = defaultdict(
        list
    )

    for record in records:
        numerical_value = record[
            value_field
        ]

        if (
            not isinstance(
                numerical_value,
                (int, float)
            )
            or isinstance(
                numerical_value,
                bool
            )
            or not math.isfinite(
                float(
                    numerical_value
                )
            )
        ):
            raise RuntimeError(f"Equal-cell aggregation received an invalid value : {value_field}")

        values_by_cell[
            record[
                "cell_identity"
            ]
        ].append(
            float(
                numerical_value
            )
        )

    if not values_by_cell:
        raise RuntimeError("Equal-cell aggregation requires at least one cell.")

    return float(
        np.mean(
            [
                np.mean(
                    values_by_cell[
                        cell_identity
                    ]
                )
                for cell_identity in sorted(
                    values_by_cell
                )
            ]
        )
    )


# Building condition-level calibration from out-of-fold predictions
def build_condition_calibration(prepared_models: dict[str, dict], prediction_records: list[dict]) -> list[dict]:
    predictions_by_candidate_and_condition = defaultdict(
        list
    )

    for prediction_record in prediction_records:
        predictions_by_candidate_and_condition[
            (
                prediction_record[
                    "candidate_identity"
                ],
                prediction_record[
                    "condition_protocol_identity"
                ]
            )
        ].append(
            prediction_record
        )

    calibration_records = []

    for candidate_identity in sorted(
        prepared_models
    ):
        prepared_model = prepared_models[
            candidate_identity
        ]

        candidate_record = prepared_model[
            "candidate_record"
        ]

        if candidate_record.get(
            "fit_status"
        ) != "completed":
            continue

        condition_identities = sorted(
            {
                observation_record[
                    "condition_protocol_identity"
                ]
                for observation_record in (
                    prepared_model[
                        "observation_records"
                    ]
                )
            }
        )

        for condition_identity in (
            condition_identities
        ):
            base_record = {
                "candidate_identity": (
                    candidate_identity
                ),
                "fit_identity": (
                    prepared_model[
                        "fit_identity"
                    ]
                ),
                "support_region_identity": (
                    candidate_record[
                        "support_region_identity"
                    ]
                ),
                "template_name": (
                    candidate_record[
                        "template_name"
                    ]
                ),
                "condition_protocol_identity": (
                    condition_identity
                ),
                "calibration_status": None,
                "calibration_reason_code": None,
                "independent_cell_count": None,
                "observation_count": None,
                "mean_observed": None,
                "mean_predicted": None,
                "mean_error": None,
                "mean_absolute_error": None,
                "root_mean_squared_error": None,
                "estimate_label": "predictive"
            }

            if candidate_record.get(
                "validation_status"
            ) != "completed":
                base_record[
                    "calibration_status"
                ] = "unavailable"

                base_record[
                    "calibration_reason_code"
                ] = (
                    candidate_record.get(
                        "validation_reason_code"
                    )
                    or "VALIDATION_UNAVAILABLE"
                )

                calibration_records.append(
                    base_record
                )

                continue

            condition_predictions = (
                predictions_by_candidate_and_condition.get(
                    (
                        candidate_identity,
                        condition_identity
                    ),
                    []
                )
            )

            if not condition_predictions:
                base_record[
                    "calibration_status"
                ] = "unavailable"

                base_record[
                    "calibration_reason_code"
                ] = (
                    "CONDITION_PREDICTIONS_UNAVAILABLE"
                )

                calibration_records.append(
                    base_record
                )

                continue

            observed_mean = (
                calculate_equal_cell_mean(
                    records = condition_predictions,
                    value_field = (
                        "observed_outcome"
                    )
                )
            )

            predicted_mean = (
                calculate_equal_cell_mean(
                    records = condition_predictions,
                    value_field = (
                        "predicted_outcome"
                    )
                )
            )

            errors_by_cell = defaultdict(
                list
            )

            for prediction_record in (
                condition_predictions
            ):
                prediction_error = (
                    prediction_record[
                        "predicted_outcome"
                    ]
                    - prediction_record[
                        "observed_outcome"
                    ]
                )

                errors_by_cell[
                    prediction_record[
                        "cell_identity"
                    ]
                ].append(
                    prediction_error
                )

            cell_mean_errors = []

            cell_mean_absolute_errors = []

            cell_mean_squared_errors = []

            for cell_identity in sorted(
                errors_by_cell
            ):
                cell_errors = np.asarray(
                    errors_by_cell[
                        cell_identity
                    ],
                    dtype = float
                )

                cell_mean_errors.append(
                    float(
                        np.mean(
                            cell_errors
                        )
                    )
                )

                cell_mean_absolute_errors.append(
                    float(
                        np.mean(
                            np.abs(
                                cell_errors
                            )
                        )
                    )
                )

                cell_mean_squared_errors.append(
                    float(
                        np.mean(
                            np.square(
                                cell_errors
                            )
                        )
                    )
                )

            base_record.update(
                {
                    "calibration_status": "completed",
                    "calibration_reason_code": None,
                    "independent_cell_count": len(
                        errors_by_cell
                    ),
                    "observation_count": len(
                        condition_predictions
                    ),
                    "mean_observed": observed_mean,
                    "mean_predicted": predicted_mean,
                    "mean_error": float(
                        np.mean(
                            cell_mean_errors
                        )
                    ),
                    "mean_absolute_error": float(
                        np.mean(
                            cell_mean_absolute_errors
                        )
                    ),
                    "root_mean_squared_error": math.sqrt(
                        float(
                            np.mean(
                                cell_mean_squared_errors
                            )
                        )
                    )
                }
            )

            calibration_records.append(
                base_record
            )

    return sorted(
        calibration_records,
        key = lambda record: (
            record[
                "candidate_identity"
            ],
            record[
                "condition_protocol_identity"
            ]
        )
    )


# Comparing validation performance without selecting a winner
def build_model_comparisons(candidate_records: list[dict], validation_prediction_records: list[dict],
                            validation_metric_records: list[dict], modeling_policy: dict
) -> list[dict]:
    comparison_policy = modeling_policy[
        "model_comparison"
    ]

    comparison_group_fields = comparison_policy[
        "comparison_group_fields"
    ]

    selected_candidates_by_group = defaultdict(
        list
    )

    for candidate_record in candidate_records:
        if candidate_record[
            "selection_status"
        ] != "selected":
            continue

        comparison_key = tuple(
            candidate_record[
                field_name
            ]
            for field_name in (
                comparison_group_fields
            )
        )

        selected_candidates_by_group[
            comparison_key
        ].append(
            candidate_record
        )

    metrics_by_candidate = {}

    for metric_record in (
        validation_metric_records
    ):
        candidate_identity = metric_record[
            "candidate_identity"
        ]

        if candidate_identity in (
            metrics_by_candidate
        ):
            raise RuntimeError(f"Validation metrics contain a duplicate candidate : {candidate_identity}")

        metrics_by_candidate[
            candidate_identity
        ] = metric_record

    prediction_identities_by_candidate = defaultdict(
        set
    )

    for prediction_record in (
        validation_prediction_records
    ):
        prediction_key = (
            prediction_record[
                "source_identity"
            ],
            prediction_record[
                "cell_identity"
            ]
        )

        candidate_identity = prediction_record[
            "candidate_identity"
        ]

        if prediction_key in (
            prediction_identities_by_candidate[
                candidate_identity
            ]
        ):
            raise RuntimeError(
                f"Validation predictions contain a duplicate candidate observation : {candidate_identity} : {prediction_key}"
            )

        prediction_identities_by_candidate[
            candidate_identity
        ].add(
            prediction_key
        )

    comparison_records = []

    for comparison_key in sorted(
        selected_candidates_by_group
    ):
        group_candidates = sorted(
            selected_candidates_by_group[
                comparison_key
            ],
            key = lambda record: record[
                "candidate_identity"
            ]
        )

        for first_index in range(
            len(
                group_candidates
            )
        ):
            for second_index in range(
                first_index + 1,
                len(
                    group_candidates
                )
            ):
                first_candidate = (
                    group_candidates[
                        first_index
                    ]
                )

                second_candidate = (
                    group_candidates[
                        second_index
                    ]
                )

                first_identity = (
                    first_candidate[
                        "candidate_identity"
                    ]
                )

                second_identity = (
                    second_candidate[
                        "candidate_identity"
                    ]
                )

                first_metrics = (
                    metrics_by_candidate.get(
                        first_identity
                    )
                )

                second_metrics = (
                    metrics_by_candidate.get(
                        second_identity
                    )
                )

                comparison_status = "completed"
                comparison_reason_code = None

                if (
                    first_metrics is None
                    or first_metrics[
                        "validation_status"
                    ]
                    != "completed"
                ):
                    comparison_status = "unavailable"
                    comparison_reason_code = (
                        "FIRST_CANDIDATE_VALIDATION_UNAVAILABLE"
                    )

                elif (
                    second_metrics is None
                    or second_metrics[
                        "validation_status"
                    ]
                    != "completed"
                ):
                    comparison_status = "unavailable"
                    comparison_reason_code = (
                        "SECOND_CANDIDATE_VALIDATION_UNAVAILABLE"
                    )

                elif (
                    prediction_identities_by_candidate[
                        first_identity
                    ]
                    != prediction_identities_by_candidate[
                        second_identity
                    ]
                ):
                    comparison_status = "unavailable"
                    comparison_reason_code = (
                        "OUT_OF_FOLD_POPULATIONS_DIFFER"
                    )

                else:
                    required_metrics = (
                        comparison_policy[
                            "comparison_metrics"
                        ]
                    )

                    for metric_name in (
                        required_metrics
                    ):
                        first_metric_value = (
                            first_metrics.get(
                                metric_name
                            )
                        )

                        second_metric_value = (
                            second_metrics.get(
                                metric_name
                            )
                        )

                        if (
                            first_metric_value is None
                            or second_metric_value is None
                        ):
                            comparison_status = (
                                "unavailable"
                            )

                            comparison_reason_code = (
                                "VALIDATION_METRIC_UNAVAILABLE"
                            )

                            break

                comparison_record = {
                    **{
                        field_name: field_value
                        for (
                            field_name,
                            field_value
                        ) in zip(
                            comparison_group_fields,
                            comparison_key,
                            strict = True
                        )
                    },
                    "first_candidate_identity": (
                        first_identity
                    ),
                    "second_candidate_identity": (
                        second_identity
                    ),
                    "first_template_name": (
                        first_candidate[
                            "template_name"
                        ]
                    ),
                    "second_template_name": (
                        second_candidate[
                            "template_name"
                        ]
                    ),
                    "comparison_status": (
                        comparison_status
                    ),
                    "comparison_reason_code": (
                        comparison_reason_code
                    ),
                    "shared_prediction_count": (
                        len(
                            prediction_identities_by_candidate[
                                first_identity
                            ]
                        )
                        if comparison_status
                        == "completed"
                        else None
                    ),
                    "first_mean_absolute_error": None,
                    "second_mean_absolute_error": None,
                    "mean_absolute_error_difference": None,
                    "first_root_mean_squared_error": None,
                    "second_root_mean_squared_error": None,
                    "root_mean_squared_error_difference": None,
                    "estimate_label": "predictive"
                }

                if comparison_status == "completed":
                    first_mae = float(
                        first_metrics[
                            "mean_absolute_error"
                        ]
                    )

                    second_mae = float(
                        second_metrics[
                            "mean_absolute_error"
                        ]
                    )

                    first_rmse = float(
                        first_metrics[
                            "root_mean_squared_error"
                        ]
                    )

                    second_rmse = float(
                        second_metrics[
                            "root_mean_squared_error"
                        ]
                    )

                    comparison_record.update(
                        {
                            "first_mean_absolute_error": (
                                first_mae
                            ),
                            "second_mean_absolute_error": (
                                second_mae
                            ),
                            "mean_absolute_error_difference": (
                                second_mae
                                - first_mae
                            ),
                            "first_root_mean_squared_error": (
                                first_rmse
                            ),
                            "second_root_mean_squared_error": (
                                second_rmse
                            ),
                            "root_mean_squared_error_difference": (
                                second_rmse
                                - first_rmse
                            )
                        }
                    )

                comparison_records.append(
                    comparison_record
                )

    return sorted(
        comparison_records,
        key = lambda record: (
            *(
                record[
                    field_name
                ]
                for field_name in (
                    comparison_group_fields
                )
            ),
            record[
                "first_candidate_identity"
            ],
            record[
                "second_candidate_identity"
            ]
        )
    )


# Determining whether a horizon lies inside one membership interval
def horizon_inside_membership_support(horizon_value: float, membership_record: dict) -> bool:
    lower_horizon = parse_finite_number(
        membership_record[
            "lower_horizon"
        ],
        "lower_horizon"
    )

    upper_horizon = parse_finite_number(
        membership_record[
            "upper_horizon"
        ],
        "upper_horizon"
    )

    lower_inclusive = parse_boolean(
        membership_record[
            "lower_inclusive"
        ],
        "lower_inclusive"
    )

    upper_inclusive = parse_boolean(
        membership_record[
            "upper_inclusive"
        ],
        "upper_inclusive"
    )

    if upper_horizon < lower_horizon:
        raise RuntimeError("Support membership contains reversed horizon boundaries.")

    lower_satisfied = (
        horizon_value >= lower_horizon
        if lower_inclusive
        else horizon_value > lower_horizon
    )

    upper_satisfied = (
        horizon_value <= upper_horizon
        if upper_inclusive
        else horizon_value < upper_horizon
    )

    return (
        lower_satisfied
        and upper_satisfied
    )


# Building supported cell and condition trajectory summaries
def build_trajectory_summaries(modeling_policy: dict, cell_records: list[dict],
                               reference_records: list[dict], support_membership_records: list[dict]
) -> tuple[list[dict], list[dict]]:
    cell_records_by_identity = (
        build_unique_record_index(
            records = cell_records,
            identity_field = "cell_identity",
            record_description = "Cell feature table"
        )
    )

    reference_records_by_identity = (
        build_unique_record_index(
            records = reference_records,
            identity_field = "source_identity",
            record_description = (
                "Reference-test feature table"
            )
        )
    )

    references_by_cell = defaultdict(
        list
    )

    for reference_record in (
        reference_records_by_identity.values()
    ):
        cell_identity = reference_record[
            "cell_identity"
        ]

        if cell_identity not in (
            cell_records_by_identity
        ):
            raise RuntimeError(
                f"Reference-test record identifies an unknown cell : {reference_record['source_identity']} : {cell_identity}"
            )

        references_by_cell[
            cell_identity
        ].append(
            reference_record
        )

    memberships_by_question_and_cell = defaultdict(
        list
    )

    observed_membership_keys = set()

    for membership_record in (
        support_membership_records
    ):
        membership_key = (
            membership_record[
                "support_region_identity"
            ],
            membership_record[
                "cell_identity"
            ]
        )

        if membership_key in observed_membership_keys:
            raise RuntimeError(f"Support membership contains a duplicate key : {membership_key}")

        observed_membership_keys.add(
            membership_key
        )

        question_key = (
            membership_record[
                "aging_mode"
            ],
            membership_record[
                "outcome"
            ],
            membership_record[
                "cohort"
            ],
            membership_record[
                "horizon_axis"
            ]
        )

        cell_identity = membership_record[
            "cell_identity"
        ]

        cell_record = (
            cell_records_by_identity.get(
                cell_identity
            )
        )

        if cell_record is None:
            raise RuntimeError(f"Support membership identifies an unknown cell : {cell_identity}")

        for field_name in [
            "aging_mode",
            "dataset_stage",
            "condition_protocol_identity",
            "laboratory",
            "sampling_method"
        ]:
            if (
                membership_record[
                    field_name
                ]
                != cell_record[
                    field_name
                ]
            ):
                raise RuntimeError(f"Support membership conflicts with cell metadata : {cell_identity} : {field_name}")

        memberships_by_question_and_cell[
            (
                question_key,
                cell_identity
            )
        ].append(
            membership_record
        )

    outcome_definitions = modeling_policy[
        "outcomes"
    ]

    horizon_definitions = modeling_policy[
        "horizon_axes"
    ]

    summary_observations = []

    accounting_records = []

    observed_summary_keys = set()

    question_keys = sorted(
        {
            question_key
            for (
                question_key,
                cell_identity
            ) in memberships_by_question_and_cell
        }
    )

    for question_key in question_keys:
        (
            aging_mode,
            outcome_name,
            cohort_name,
            horizon_axis
        ) = question_key

        if outcome_name not in outcome_definitions:
            raise RuntimeError(f"Support membership references an unknown outcome : {outcome_name}")

        if horizon_axis not in horizon_definitions:
            raise RuntimeError(f"Support membership references an unknown horizon axis : {horizon_axis}")

        if cohort_name not in {
            "primary",
            "sensitivity"
        }:
            raise RuntimeError(f"Support membership references an unknown cohort : {cohort_name}")

        outcome_definition = outcome_definitions[
            outcome_name
        ]

        outcome_field = outcome_definition[
            "value_field"
        ]

        eligibility_field = outcome_definition[
            (
                "primary_eligibility_field"
                if cohort_name == "primary"
                else "sensitivity_eligibility_field"
            )
        ]

        question_cell_identities = sorted(
            cell_identity
            for (
                observed_question_key,
                cell_identity
            ) in memberships_by_question_and_cell
            if observed_question_key == question_key
        )

        eligible_positive_record_count = 0
        included_record_count = 0
        outside_support_record_count = 0

        for cell_identity in question_cell_identities:
            cell_record = cell_records_by_identity[
                cell_identity
            ]

            membership_records = (
                memberships_by_question_and_cell[
                    (
                        question_key,
                        cell_identity
                    )
                ]
            )

            for reference_record in sorted(
                references_by_cell[
                    cell_identity
                ],
                key = lambda record: record[
                    "source_identity"
                ]
            ):
                horizon_text = reference_record[
                    horizon_axis
                ]

                outcome_text = reference_record[
                    outcome_field
                ]

                eligibility_text = reference_record[
                    eligibility_field
                ]

                validate_optional_boolean(
                    eligibility_text,
                    eligibility_field
                )

                if (
                    horizon_text == ""
                    or outcome_text == ""
                    or eligibility_text != "true"
                ):
                    continue

                horizon_value = parse_finite_number(
                    horizon_text,
                    horizon_axis
                )

                if (
                    horizon_value
                    <= horizon_definitions[
                        horizon_axis
                    ][
                        "minimum_value_exclusive"
                    ]
                ):
                    continue

                outcome_value = parse_finite_number(
                    outcome_text,
                    outcome_field
                )

                if (
                    horizon_axis
                    == (
                        "cumulative_full_equivalent_cycles_observed"
                    )
                ):
                    completeness_text = (
                        reference_record[
                            "cumulative_fec_observation_complete"
                        ]
                    )

                    validate_optional_boolean(
                        completeness_text,
                        (
                            "cumulative_fec_"
                            "observation_complete"
                        )
                    )

                    if completeness_text != "true":
                        continue

                eligible_positive_record_count += 1

                supported = any(
                    horizon_inside_membership_support(
                        horizon_value = horizon_value,
                        membership_record = (
                            membership_record
                        )
                    )
                    for membership_record in (
                        membership_records
                    )
                )

                if not supported:
                    outside_support_record_count += 1
                    continue

                included_record_count += 1

                summary_key = (
                    aging_mode,
                    reference_record[
                        "source_identity"
                    ],
                    outcome_name,
                    cohort_name,
                    horizon_axis
                )

                if summary_key in observed_summary_keys:
                    raise RuntimeError(f"Trajectory summary contains a duplicate source question : {summary_key}")

                observed_summary_keys.add(
                    summary_key
                )

                summary_observations.append(
                    {
                        "aging_mode": aging_mode,
                        "dataset_stage": (
                            cell_record[
                                "dataset_stage"
                            ]
                        ),
                        "condition_protocol_identity": (
                            cell_record[
                                "condition_protocol_identity"
                            ]
                        ),
                        "cell_identity": cell_identity,
                        "source_identity": (
                            reference_record[
                                "source_identity"
                            ]
                        ),
                        "outcome": outcome_name,
                        "cohort": cohort_name,
                        "horizon_axis": horizon_axis,
                        "horizon": horizon_value,
                        "outcome_value": outcome_value
                    }
                )

        if (
            included_record_count
            + outside_support_record_count
            != eligible_positive_record_count
        ):
            raise RuntimeError(f"Trajectory-summary observation accounting does not reconcile : {question_key}")

        accounting_records.append(
            {
                "accounting_scope": (
                    "trajectory_summary"
                ),
                "aging_mode": aging_mode,
                "outcome": outcome_name,
                "cohort": cohort_name,
                "horizon_axis": horizon_axis,
                "member_cell_count": len(
                    question_cell_identities
                ),
                "eligible_positive_record_count": (
                    eligible_positive_record_count
                ),
                "included_record_count": (
                    included_record_count
                ),
                "outside_support_record_count": (
                    outside_support_record_count
                )
            }
        )

    cell_groups = defaultdict(
        list
    )

    condition_groups = defaultdict(
        list
    )

    condition_horizon_groups = defaultdict(
        list
    )

    for observation_record in (
        summary_observations
    ):
        common_key = (
            observation_record[
                "aging_mode"
            ],
            observation_record[
                "dataset_stage"
            ],
            observation_record[
                "condition_protocol_identity"
            ],
            observation_record[
                "outcome"
            ],
            observation_record[
                "cohort"
            ],
            observation_record[
                "horizon_axis"
            ]
        )

        cell_groups[
            (
                *common_key,
                observation_record[
                    "cell_identity"
                ]
            )
        ].append(
            observation_record
        )

        condition_groups[
            common_key
        ].append(
            observation_record
        )

        condition_horizon_groups[
            (
                *common_key,
                observation_record[
                    "horizon"
                ]
            )
        ].append(
            observation_record
        )

    summary_records = []

    def build_empty_summary_record() -> dict:
        return {
            "record_type": None,
            "aging_mode": None,
            "dataset_stage": None,
            "condition_protocol_identity": None,
            "cell_identity": None,
            "outcome": None,
            "cohort": None,
            "horizon_axis": None,
            "horizon": None,
            "independent_cell_count": None,
            "observation_count": None,
            "minimum_horizon": None,
            "maximum_horizon": None,
            "first_observed_outcome": None,
            "last_observed_outcome": None,
            "mean_outcome": None,
            "standard_deviation_outcome": None,
            "minimum_outcome": None,
            "maximum_outcome": None,
            "estimate_label": "descriptive"
        }

    for cell_key in sorted(
        cell_groups
    ):
        cell_observations = sorted(
            cell_groups[
                cell_key
            ],
            key = lambda record: (
                record[
                    "horizon"
                ],
                record[
                    "source_identity"
                ]
            )
        )

        outcome_values = [
            observation_record[
                "outcome_value"
            ]
            for observation_record in (
                cell_observations
            )
        ]

        summary_record = (
            build_empty_summary_record()
        )

        summary_record.update(
            {
                "record_type": "cell",
                "aging_mode": cell_key[0],
                "dataset_stage": cell_key[1],
                "condition_protocol_identity": (
                    cell_key[2]
                ),
                "outcome": cell_key[3],
                "cohort": cell_key[4],
                "horizon_axis": cell_key[5],
                "cell_identity": cell_key[6],
                "independent_cell_count": 1,
                "observation_count": len(
                    cell_observations
                ),
                "minimum_horizon": (
                    cell_observations[0][
                        "horizon"
                    ]
                ),
                "maximum_horizon": (
                    cell_observations[-1][
                        "horizon"
                    ]
                ),
                "first_observed_outcome": (
                    cell_observations[0][
                        "outcome_value"
                    ]
                ),
                "last_observed_outcome": (
                    cell_observations[-1][
                        "outcome_value"
                    ]
                ),
                "minimum_outcome": min(
                    outcome_values
                ),
                "maximum_outcome": max(
                    outcome_values
                )
            }
        )

        summary_records.append(
            summary_record
        )

    for condition_key in sorted(
        condition_groups
    ):
        condition_observations = (
            condition_groups[
                condition_key
            ]
        )

        horizon_values = [
            observation_record[
                "horizon"
            ]
            for observation_record in (
                condition_observations
            )
        ]

        summary_record = (
            build_empty_summary_record()
        )

        summary_record.update(
            {
                "record_type": "condition",
                "aging_mode": condition_key[0],
                "dataset_stage": condition_key[1],
                "condition_protocol_identity": (
                    condition_key[2]
                ),
                "outcome": condition_key[3],
                "cohort": condition_key[4],
                "horizon_axis": condition_key[5],
                "independent_cell_count": len(
                    {
                        observation_record[
                            "cell_identity"
                        ]
                        for observation_record in (
                            condition_observations
                        )
                    }
                ),
                "observation_count": len(
                    condition_observations
                ),
                "minimum_horizon": min(
                    horizon_values
                ),
                "maximum_horizon": max(
                    horizon_values
                )
            }
        )

        summary_records.append(
            summary_record
        )

    for condition_horizon_key in sorted(
        condition_horizon_groups
    ):
        horizon_observations = (
            condition_horizon_groups[
                condition_horizon_key
            ]
        )

        observations_by_cell = defaultdict(
            list
        )

        for observation_record in (
            horizon_observations
        ):
            observations_by_cell[
                observation_record[
                    "cell_identity"
                ]
            ].append(
                observation_record[
                    "outcome_value"
                ]
            )

        cell_values = [
            float(
                np.mean(
                    observations_by_cell[
                        cell_identity
                    ]
                )
            )
            for cell_identity in sorted(
                observations_by_cell
            )
        ]

        standard_deviation = (
            None
            if len(cell_values) < 2
            else float(
                np.std(
                    np.asarray(
                        cell_values,
                        dtype = float
                    ),
                    ddof = 1
                )
            )
        )

        summary_record = (
            build_empty_summary_record()
        )

        summary_record.update(
            {
                "record_type": "condition_horizon",
                "aging_mode": (
                    condition_horizon_key[0]
                ),
                "dataset_stage": (
                    condition_horizon_key[1]
                ),
                "condition_protocol_identity": (
                    condition_horizon_key[2]
                ),
                "outcome": (
                    condition_horizon_key[3]
                ),
                "cohort": (
                    condition_horizon_key[4]
                ),
                "horizon_axis": (
                    condition_horizon_key[5]
                ),
                "horizon": (
                    condition_horizon_key[6]
                ),
                "independent_cell_count": len(
                    cell_values
                ),
                "observation_count": len(
                    horizon_observations
                ),
                "minimum_horizon": (
                    condition_horizon_key[6]
                ),
                "maximum_horizon": (
                    condition_horizon_key[6]
                ),
                "mean_outcome": float(
                    np.mean(
                        cell_values
                    )
                ),
                "standard_deviation_outcome": (
                    standard_deviation
                ),
                "minimum_outcome": min(
                    cell_values
                ),
                "maximum_outcome": max(
                    cell_values
                )
            }
        )

        summary_records.append(
            summary_record
        )

    return (
        sorted(
            summary_records,
            key = lambda record: (
                record[
                    "record_type"
                ],
                record[
                    "aging_mode"
                ],
                record[
                    "dataset_stage"
                ],
                record[
                    "condition_protocol_identity"
                ],
                record[
                    "outcome"
                ],
                record[
                    "cohort"
                ],
                record[
                    "horizon_axis"
                ],
                (
                    ""
                    if record[
                        "cell_identity"
                    ]
                    is None
                    else record[
                        "cell_identity"
                    ]
                ),
                (
                    float(
                        "-inf"
                    )
                    if record[
                        "horizon"
                    ]
                    is None
                    else record[
                        "horizon"
                    ]
                )
            )
        ),
        sorted(
            accounting_records,
            key = lambda record: (
                record[
                    "aging_mode"
                ],
                record[
                    "outcome"
                ],
                record[
                    "cohort"
                ],
                record[
                    "horizon_axis"
                ]
            )
        )
    )


# Parsing one finite numerical value from a JSON field
def parse_json_finite_number(value: object, field_name: str) -> float:
    if (
        isinstance(
            value,
            bool
        )
        or not isinstance(
            value,
            (int, float)
        )
    ):
        raise RuntimeError(f"JSON field does not contain a numerical value : {field_name} : {value!r}")

    numerical_value = float(
        value
    )

    if not math.isfinite(
        numerical_value
    ):
        raise RuntimeError(f"JSON field contains a non-finite numerical value : {field_name} : {value!r}")

    return numerical_value


# Parsing authorized pairwise comparison ranges
def parse_permitted_contrast_ranges(estimand_record: dict) -> list[dict]:
    parsed_ranges = parse_json_field(
        estimand_record[
            "permitted_bracketing_ranges"
        ],
        "permitted_bracketing_ranges"
    )

    if not isinstance(
        parsed_ranges,
        list
    ):
        raise RuntimeError("Estimand support contains invalid permitted bracketing ranges.")

    normalized_ranges = []

    for range_index, range_record in enumerate(
        parsed_ranges
    ):
        if not isinstance(
            range_record,
            dict
        ):
            raise RuntimeError("Estimand support contains an invalid bracketing-range record."            )

        missing_fields = {
            "minimum",
            "maximum"
        }.difference(
            range_record
        )

        if missing_fields:
            raise RuntimeError(f"Estimand bracketing range is missing fields : {sorted(missing_fields)}")

        minimum_horizon = parse_json_finite_number(
            range_record[
                "minimum"
            ],
            (
                f"permitted_bracketing_ranges[{range_index}].minimum"
            )
        )

        maximum_horizon = parse_json_finite_number(
            range_record[
                "maximum"
            ],
            (
                f"permitted_bracketing_ranges[{range_index}].maximum"
            )
        )

        if maximum_horizon < minimum_horizon:
            raise RuntimeError("Estimand bracketing range has reversed boundaries.")

        normalized_ranges.append(
            {
                "minimum": minimum_horizon,
                "maximum": maximum_horizon
            }
        )

    return sorted(
        normalized_ranges,
        key = lambda record: (
            record["minimum"],
            record["maximum"]
        )
    )


# Checking whether one horizon belongs to an authorized pairwise range
def horizon_inside_permitted_contrast_ranges(horizon_value: float, permitted_ranges: list[dict]) -> bool:
    return any(
        permitted_range[
            "minimum"
        ]
        <= horizon_value
        <= permitted_range[
            "maximum"
        ]
        for permitted_range in permitted_ranges
    )


# Building one condition-level prediction record
def build_condition_prediction_record(condition_record: dict, horizon_value: float) -> dict:
    missing_fields = sorted(
        set(
            CELL_MODEL_FIELDS
        ).difference(
            condition_record
        )
    )

    if missing_fields:
        raise RuntimeError(
            "Condition feature record is missing model fields : "
            f"{condition_record.get('condition_protocol_identity')} : {missing_fields}"
        )

    return {
        "source_identity": None,
        "cell_identity": None,
        "horizon": horizon_value,
        "outcome_value": None,
        **{
            field_name: condition_record[
                field_name
            ]
            for field_name in (
                CELL_MODEL_FIELDS
            )
        }
    }


# Building the deterministic estimand decision identity
def build_estimand_decision_identity(estimand_record: dict, modeling_policy: dict) -> str:
    identity_fields = modeling_policy[
        "contrast_evaluation"
    ][
        "decision_identity_fields"
    ]

    missing_fields = sorted(
        set(
            identity_fields
        ).difference(
            estimand_record
        )
    )

    if missing_fields:
        raise RuntimeError(f"Estimand record is missing identity fields : {missing_fields}")

    return build_deterministic_identity(
        {
            field_name: estimand_record[
                field_name
            ]
            for field_name in identity_fields
        }
    )


# Producing one unavailable contrast record
def build_unavailable_contrast_record(candidate_record: dict, estimand_record: dict,
                                      estimand_decision_identity: str, horizon_value: float | None,
                                      reason_code: str, evaluation_laboratory: str | None = None,
                                      model_based_interpolation: bool | None = None
) -> dict:
    return {
        "candidate_identity": candidate_record[
            "candidate_identity"
        ],
        "fit_identity": candidate_record.get(
            "fit_identity"
        ),
        "support_region_identity": candidate_record[
            "support_region_identity"
        ],
        "template_name": candidate_record[
            "template_name"
        ],
        "estimator": candidate_record[
            "estimator"
        ],
        "estimand_decision_identity": (
            estimand_decision_identity
        ),
        "decision_scope": estimand_record[
            "decision_scope"
        ],
        "first_condition_protocol_identity": (
            estimand_record[
                "first_condition_protocol_identity"
            ]
        ),
        "second_condition_protocol_identity": (
            estimand_record[
                "second_condition_protocol_identity"
            ]
        ),
        "outcome": estimand_record[
            "outcome"
        ],
        "cohort": estimand_record[
            "cohort"
        ],
        "horizon_axis": estimand_record[
            "horizon_axis"
        ],
        "horizon": horizon_value,
        "evaluation_laboratory": (
            evaluation_laboratory
        ),
        "model_based_interpolation": (
            model_based_interpolation
        ),
        "first_predicted_outcome": None,
        "second_predicted_outcome": None,
        "contrast_estimate": None,
        "contrast_direction": (
            "second_condition_minus_first_condition"
        ),
        "contrast_status": "unavailable",
        "contrast_reason_code": reason_code,
        "uncertainty_status": "unavailable",
        "uncertainty_reason_code": reason_code,
        "confidence_level": None,
        "degrees_of_freedom": None,
        "standard_error": None,
        "confidence_interval_lower": None,
        "confidence_interval_upper": None,
        "final_support_classification": (
            estimand_record[
                "final_support_classification"
            ]
        ),
        "restriction_reasons": (
            estimand_record[
                "restriction_reasons"
            ]
        ),
        "causal_interpretation": (
            "not_established"
        ),
        "estimate_label": "associational",
        "candidate_validation_status": (
            candidate_record.get(
                "validation_status"
            )
        ),
        "candidate_validation_reason_code": (
            candidate_record.get(
                "validation_reason_code"
            )
        ),
        "candidate_analytical_result_status": (
            candidate_record.get(
                "analytical_result_status"
            )
        )
    }


# Constructing condition contrasts authorized by the design report
def build_model_contrasts(prepared_models: dict[str, dict], fitted_models_by_candidate: dict[str, dict],
                          estimand_records: list[dict], condition_records: list[dict],
                          approved_discrete_horizons: list[dict], discrete_horizon_status: str,
                          modeling_policy: dict
) -> list[dict]:
    contrast_policy = modeling_policy[
        "contrast_evaluation"
    ]

    permitted_classifications = set(
        contrast_policy[
            "permitted_support_classifications"
        ]
    )

    condition_records_by_identity = (
        build_unique_record_index(
            records = condition_records,
            identity_field = (
                "condition_protocol_identity"
            ),
            record_description = (
                "Condition feature table"
            )
        )
    )

    estimands_by_question = defaultdict(
        list
    )

    observed_decision_identities = set()

    for estimand_record in estimand_records:
        estimand_decision_identity = (
            build_estimand_decision_identity(
                estimand_record = estimand_record,
                modeling_policy = modeling_policy
            )
        )

        if (
            estimand_decision_identity
            in observed_decision_identities
        ):
            raise RuntimeError(f"Estimand support contains a duplicate decision : {estimand_decision_identity}")

        observed_decision_identities.add(
            estimand_decision_identity
        )

        if (
            estimand_record[
                "final_support_classification"
            ]
            not in permitted_classifications
        ):
            continue

        if parse_boolean(
            estimand_record[
                "target_estimand_available"
            ],
            "target_estimand_available"
        ) is not True:
            raise RuntimeError("Permitted estimand does not identify an available target.")

        if (
            estimand_record[
                "decision_scope"
            ]
            != "condition_protocol_pair"
        ):
            raise RuntimeError(f"Permitted estimand has an unsupported decision scope : {estimand_record['decision_scope']}")

        first_condition_identity = (
            estimand_record[
                "first_condition_protocol_identity"
            ]
        )

        second_condition_identity = (
            estimand_record[
                "second_condition_protocol_identity"
            ]
        )

        if (
            first_condition_identity
            == second_condition_identity
        ):
            raise RuntimeError("Estimand comparison identifies the same condition twice.")

        permitted_ranges = (
            parse_permitted_contrast_ranges(
                estimand_record
            )
        )

        if not permitted_ranges:
            raise RuntimeError(f"Permitted estimand does not contain a comparison range : {estimand_decision_identity}")

        estimands_by_question[
            (
                estimand_record[
                    "aging_mode"
                ],
                estimand_record[
                    "outcome"
                ],
                estimand_record[
                    "cohort"
                ],
                estimand_record[
                    "horizon_axis"
                ]
            )
        ].append(
            {
                "record": estimand_record,
                "identity": (
                    estimand_decision_identity
                ),
                "permitted_ranges": (
                    permitted_ranges
                )
            }
        )

    contrast_records = []

    observed_contrast_keys = set()

    confidence_level = modeling_policy[
        "uncertainty"
    ][
        "confidence_level"
    ]

    for candidate_identity in sorted(
        prepared_models
    ):
        prepared_model = prepared_models[
            candidate_identity
        ]

        candidate_record = prepared_model[
            "candidate_record"
        ]

        question_key = (
            candidate_record[
                "aging_mode"
            ],
            candidate_record[
                "outcome"
            ],
            candidate_record[
                "cohort"
            ],
            candidate_record[
                "horizon_axis"
            ]
        )

        candidate_estimands = (
            estimands_by_question.get(
                question_key,
                []
            )
        )

        observations_by_condition = defaultdict(
            list
        )

        for observation_record in (
            prepared_model[
                "observation_records"
            ]
        ):
            observations_by_condition[
                observation_record[
                    "condition_protocol_identity"
                ]
            ].append(
                observation_record
            )

        observed_condition_identities = set(
            observations_by_condition
        )

        for estimand_definition in (
            candidate_estimands
        ):
            estimand_record = (
                estimand_definition[
                    "record"
                ]
            )

            estimand_decision_identity = (
                estimand_definition[
                    "identity"
                ]
            )

            permitted_ranges = (
                estimand_definition[
                    "permitted_ranges"
                ]
            )

            first_condition_identity = (
                estimand_record[
                    "first_condition_protocol_identity"
                ]
            )

            second_condition_identity = (
                estimand_record[
                    "second_condition_protocol_identity"
                ]
            )

            if not {
                first_condition_identity,
                second_condition_identity
            }.issubset(
                observed_condition_identities
            ):
                continue

            first_condition_record = (
                condition_records_by_identity.get(
                    first_condition_identity
                )
            )

            second_condition_record = (
                condition_records_by_identity.get(
                    second_condition_identity
                )
            )

            if (
                first_condition_record is None
                or second_condition_record is None
            ):
                raise RuntimeError("Estimand comparison identifies an unknown condition.")

            for condition_record in [
                first_condition_record,
                second_condition_record
            ]:
                if (
                    condition_record[
                        "aging_mode"
                    ]
                    != candidate_record[
                        "aging_mode"
                    ]
                ):
                    raise RuntimeError("Estimand condition and candidate have different aging modes.")

                if not cell_matches_design_scope(
                    cell_record = condition_record,
                    candidate_record = candidate_record
                ):
                    raise RuntimeError("Estimand condition lies outside the candidate design scope.")

            evaluation_laboratory = None

            if (
                candidate_record[
                    "context_family"
                ]
                == "laboratory"
            ):
                first_laboratory = (
                    first_condition_record[
                        "laboratory"
                    ]
                )

                second_laboratory = (
                    second_condition_record[
                        "laboratory"
                    ]
                )

                if first_laboratory != second_laboratory:
                    unavailable_record = (
                        build_unavailable_contrast_record(
                            candidate_record = (
                                candidate_record
                            ),
                            estimand_record = (
                                estimand_record
                            ),
                            estimand_decision_identity = (
                                estimand_decision_identity
                            ),
                            horizon_value = None,
                            reason_code = (
                                "NO_SHARED_OBSERVED_LABORATORY"
                            )
                        )
                    )

                    contrast_key = (
                        candidate_identity,
                        estimand_decision_identity,
                        None
                    )

                    if contrast_key in observed_contrast_keys:
                        raise RuntimeError("Model contrasts contain a duplicate key.")

                    observed_contrast_keys.add(
                        contrast_key
                    )

                    contrast_records.append(
                        unavailable_record
                    )

                    continue

                evaluation_laboratory = (
                    first_laboratory
                )

            elif (
                candidate_record[
                    "context_family"
                ]
                != "no_additional_context"
            ):
                raise RuntimeError(f"Candidate has an unsupported contrast context : {candidate_record['context_family']}")

            if (
                discrete_horizon_status
                != (
                    "authorized_with_outcome_blind_pre_specification"
                )
                or not approved_discrete_horizons
            ):
                unavailable_record = (
                    build_unavailable_contrast_record(
                        candidate_record = (
                            candidate_record
                        ),
                        estimand_record = (
                            estimand_record
                        ),
                        estimand_decision_identity = (
                            estimand_decision_identity
                        ),
                        horizon_value = None,
                        reason_code = (
                            "DISCRETE_HORIZON_NOT_AUTHORIZED"
                        ),
                        evaluation_laboratory = (
                            evaluation_laboratory
                        ),
                        model_based_interpolation = None
                    )
                )

                contrast_key = (
                    candidate_identity,
                    estimand_decision_identity,
                    None
                )

                if contrast_key in observed_contrast_keys:
                    raise RuntimeError("Model contrasts contain a duplicate key.")

                observed_contrast_keys.add(
                    contrast_key
                )

                contrast_records.append(
                    unavailable_record
                )

                continue

            first_observed_horizons = {
                observation_record[
                    "horizon"
                ]
                for observation_record in (
                    observations_by_condition[
                        first_condition_identity
                    ]
                )
            }

            second_observed_horizons = {
                observation_record[
                    "horizon"
                ]
                for observation_record in (
                    observations_by_condition[
                        second_condition_identity
                    ]
                )
            }

            evaluation_horizons = sorted(
                {
                    parse_finite_number(
                        horizon_definition[
                            "horizon"
                        ],
                        "approved discrete horizon"
                    )
                    for horizon_definition in (
                        approved_discrete_horizons
                    )
                    if (
                        horizon_definition.get(
                            "aging_mode"
                        )
                        == candidate_record[
                            "aging_mode"
                        ]
                        and horizon_definition.get(
                            "horizon_axis"
                        )
                        == candidate_record[
                            "horizon_axis"
                        ]
                        and horizon_inside_candidate_support(
                            horizon_value = float(
                                horizon_definition[
                                    "horizon"
                                ]
                            ),
                            candidate_record = (
                                candidate_record
                            )
                        )
                        and horizon_inside_permitted_contrast_ranges(
                            horizon_value = float(
                                horizon_definition[
                                    "horizon"
                                ]
                            ),
                            permitted_ranges = (
                                permitted_ranges
                            )
                        )
                    )
                }
            )

            if (
                prepared_model[
                    "model_encoding"
                ][
                    "horizon_basis"
                ]
                == "restricted_cubic_spline"
            ):
                (
                    lower_knot,
                    _,
                    upper_knot
                ) = prepared_model[
                    "model_encoding"
                ][
                    "spline_knots"
                ]

                evaluation_horizons = [
                    horizon_value
                    for horizon_value in (
                        evaluation_horizons
                    )
                    if (
                        lower_knot
                        <= horizon_value
                        <= upper_knot
                    )
                ]

            if not evaluation_horizons:
                unavailable_record = (
                    build_unavailable_contrast_record(
                        candidate_record = candidate_record,
                        estimand_record = estimand_record,
                        estimand_decision_identity = (
                            estimand_decision_identity
                        ),
                        horizon_value = None,
                        reason_code = (
                            "NO_AUTHORIZED_EVALUATION_HORIZON"
                        ),
                        evaluation_laboratory = (
                            evaluation_laboratory
                        )
                    )
                )

                contrast_key = (
                    candidate_identity,
                    estimand_decision_identity,
                    None
                )

                if contrast_key in observed_contrast_keys:
                    raise RuntimeError("Model contrasts contain a duplicate key.")

                observed_contrast_keys.add(
                    contrast_key
                )

                contrast_records.append(
                    unavailable_record
                )

                continue

            fitted_result = (
                fitted_models_by_candidate.get(
                    candidate_identity
                )
            )

            candidate_validation_status = candidate_record.get(
                "validation_status"
            )
            candidate_validation_reason_code = candidate_record.get(
                "validation_reason_code"
            )

            for horizon_value in evaluation_horizons:
                model_based_interpolation = not (
                    horizon_value
                    in first_observed_horizons
                    and horizon_value
                    in second_observed_horizons
                )

                contrast_key = (
                    candidate_identity,
                    estimand_decision_identity,
                    horizon_value
                )

                if contrast_key in observed_contrast_keys:
                    raise RuntimeError("Model contrasts contain a duplicate key.")

                observed_contrast_keys.add(
                    contrast_key
                )

                if fitted_result is None:
                    contrast_records.append(
                        build_unavailable_contrast_record(
                            candidate_record = (
                                candidate_record
                            ),
                            estimand_record = (
                                estimand_record
                            ),
                            estimand_decision_identity = (
                                estimand_decision_identity
                            ),
                            horizon_value = horizon_value,
                            reason_code = (
                                candidate_record.get(
                                    "fit_reason_code"
                                )
                                or "MODEL_FIT_UNAVAILABLE"
                            ),
                            evaluation_laboratory = (
                                evaluation_laboratory
                            ),
                            model_based_interpolation = (
                                model_based_interpolation
                            )
                        )
                    )

                    continue

                first_prediction_record = (
                    build_condition_prediction_record(
                        condition_record = (
                            first_condition_record
                        ),
                        horizon_value = horizon_value
                    )
                )

                second_prediction_record = (
                    build_condition_prediction_record(
                        condition_record = (
                            second_condition_record
                        ),
                        horizon_value = horizon_value
                    )
                )

                try:
                    (
                        prediction_matrix,
                        prediction_terms
                    ) = transform_model_matrix(
                        candidate_record = (
                            candidate_record
                        ),
                        observation_records = [
                            first_prediction_record,
                            second_prediction_record
                        ],
                        model_encoding = (
                            prepared_model[
                                "model_encoding"
                            ]
                        )
                    )

                except CandidateMatrixUnavailableError as exc:
                    contrast_records.append(
                        build_unavailable_contrast_record(
                            candidate_record = (
                                candidate_record
                            ),
                            estimand_record = (
                                estimand_record
                            ),
                            estimand_decision_identity = (
                                estimand_decision_identity
                            ),
                            horizon_value = horizon_value,
                            reason_code = exc.reason_code,
                            evaluation_laboratory = (
                                evaluation_laboratory
                            ),
                            model_based_interpolation = (
                                model_based_interpolation
                            )
                        )
                    )

                    continue

                if (
                    prediction_terms
                    != prepared_model[
                        "matrix_terms"
                    ]
                ):
                    raise RuntimeError("Contrast prediction matrix contains different terms from the fitted model.")

                predicted_outcomes = (
                    prediction_matrix
                    @ fitted_result[
                        "coefficients"
                    ]
                )

                if not np.all(
                    np.isfinite(
                        predicted_outcomes
                    )
                ):
                    contrast_records.append(
                        build_unavailable_contrast_record(
                            candidate_record = (
                                candidate_record
                            ),
                            estimand_record = (
                                estimand_record
                            ),
                            estimand_decision_identity = (
                                estimand_decision_identity
                            ),
                            horizon_value = horizon_value,
                            reason_code = (
                                "NONFINITE_CONTRAST_PREDICTION"
                            ),
                            evaluation_laboratory = (
                                evaluation_laboratory
                            ),
                            model_based_interpolation = (
                                model_based_interpolation
                            )
                        )
                    )

                    continue

                first_prediction = float(
                    predicted_outcomes[0]
                )

                second_prediction = float(
                    predicted_outcomes[1]
                )

                contrast_estimate = (
                    second_prediction
                    - first_prediction
                )

                uncertainty_status = "completed"
                uncertainty_reason_code = None
                standard_error = None
                confidence_interval_lower = None
                confidence_interval_upper = None
                degrees_of_freedom = None

                jackknife_covariance = (
                    fitted_result.get(
                        "jackknife_covariance"
                    )
                )

                if jackknife_covariance is None:
                    uncertainty_status = "unavailable"
                    uncertainty_reason_code = (
                        "JACKKNIFE_COVARIANCE_UNAVAILABLE"
                    )

                else:
                    contrast_vector = (
                        prediction_matrix[1]
                        - prediction_matrix[0]
                    )

                    contrast_variance = float(
                        contrast_vector
                        @ jackknife_covariance
                        @ contrast_vector
                    )

                    if (
                        not math.isfinite(
                            contrast_variance
                        )
                        or contrast_variance < 0
                    ):
                        uncertainty_status = "unavailable"
                        uncertainty_reason_code = (
                            "INVALID_CONTRAST_VARIANCE"
                        )

                    else:
                        standard_error = math.sqrt(
                            contrast_variance
                        )

                        degrees_of_freedom = (
                            fitted_result[
                                "jackknife_degrees_of_freedom"
                            ]
                        )

                        if degrees_of_freedom <= 0:
                            uncertainty_status = "unavailable"
                            uncertainty_reason_code = (
                                "NONPOSITIVE_CONTRAST_DEGREES_OF_FREEDOM"
                            )

                            standard_error = None
                            degrees_of_freedom = None

                        else:
                            critical_value = float(
                                student_t.ppf(
                                    (
                                        1.0
                                        + confidence_level
                                    )
                                    / 2.0,
                                    degrees_of_freedom
                                )
                            )

                            if not math.isfinite(
                                critical_value
                            ):
                                uncertainty_status = "unavailable"
                                uncertainty_reason_code = (
                                    "NONFINITE_CONTRAST_CRITICAL_VALUE"
                                )

                                standard_error = None
                                degrees_of_freedom = None

                            else:
                                confidence_interval_lower = (
                                    contrast_estimate
                                    - critical_value
                                    * standard_error
                                )

                                confidence_interval_upper = (
                                    contrast_estimate
                                    + critical_value
                                    * standard_error
                                )

                predictions_are_positive = (
                    first_prediction > 0
                    and second_prediction > 0
                )

                if not predictions_are_positive:
                    contrast_status = "unavailable"
                    contrast_reason_code = (
                        "NONPOSITIVE_RATIO_PREDICTION"
                    )

                elif candidate_validation_status != "completed":
                    contrast_status = "diagnostic_only"
                    contrast_reason_code = (
                        candidate_validation_reason_code
                        or "GROUPED_VALIDATION_UNAVAILABLE"
                    )

                elif uncertainty_status != "completed":
                    contrast_status = "diagnostic_only"
                    contrast_reason_code = (
                        uncertainty_reason_code
                        or "CONTRAST_UNCERTAINTY_UNAVAILABLE"
                    )

                else:
                    contrast_status = "completed"
                    contrast_reason_code = None

                contrast_records.append(
                    {
                        "candidate_identity": (
                            candidate_identity
                        ),
                        "fit_identity": (
                            prepared_model[
                                "fit_identity"
                            ]
                        ),
                        "support_region_identity": (
                            candidate_record[
                                "support_region_identity"
                            ]
                        ),
                        "template_name": (
                            candidate_record[
                                "template_name"
                            ]
                        ),
                        "estimator": (
                            candidate_record[
                                "estimator"
                            ]
                        ),
                        "estimand_decision_identity": (
                            estimand_decision_identity
                        ),
                        "decision_scope": (
                            estimand_record[
                                "decision_scope"
                            ]
                        ),
                        "first_condition_protocol_identity": (
                            first_condition_identity
                        ),
                        "second_condition_protocol_identity": (
                            second_condition_identity
                        ),
                        "outcome": (
                            estimand_record[
                                "outcome"
                            ]
                        ),
                        "cohort": (
                            estimand_record[
                                "cohort"
                            ]
                        ),
                        "horizon_axis": (
                            estimand_record[
                                "horizon_axis"
                            ]
                        ),
                        "horizon": horizon_value,
                        "evaluation_laboratory": (
                            evaluation_laboratory
                        ),
                        "model_based_interpolation": (
                            model_based_interpolation
                        ),
                        "first_predicted_outcome": (
                            first_prediction
                        ),
                        "second_predicted_outcome": (
                            second_prediction
                        ),
                        "contrast_estimate": (
                            contrast_estimate
                        ),
                        "contrast_direction": (
                            "second_condition_minus_first_condition"
                        ),
                        "contrast_status": (
                            contrast_status
                        ),
                        "contrast_reason_code": (
                            contrast_reason_code
                        ),
                        "candidate_validation_status": (
                            candidate_validation_status
                        ),
                        "candidate_validation_reason_code": (
                            candidate_validation_reason_code
                        ),
                        "candidate_analytical_result_status": (
                            candidate_record.get(
                                "analytical_result_status"
                            )
                        ),
                        "uncertainty_status": (
                            uncertainty_status
                        ),
                        "uncertainty_reason_code": (
                            uncertainty_reason_code
                        ),
                        "confidence_level": (
                            confidence_level
                            if uncertainty_status
                            == "completed"
                            else None
                        ),
                        "degrees_of_freedom": (
                            degrees_of_freedom
                        ),
                        "standard_error": standard_error,
                        "confidence_interval_lower": (
                            confidence_interval_lower
                        ),
                        "confidence_interval_upper": (
                            confidence_interval_upper
                        ),
                        "final_support_classification": (
                            estimand_record[
                                "final_support_classification"
                            ]
                        ),
                        "restriction_reasons": (
                            estimand_record[
                                "restriction_reasons"
                            ]
                        ),
                        "causal_interpretation": (
                            "not_established"
                        ),
                        "estimate_label": (
                            "associational"
                        )
                    }
                )

    return sorted(
        contrast_records,
        key = lambda record: (
            record[
                "candidate_identity"
            ],
            record[
                "estimand_decision_identity"
            ],
            (
                float(
                    "-inf"
                )
                if record[
                    "horizon"
                ]
                is None
                else record[
                    "horizon"
                ]
            )
        )
    )


# Assigning one final disposition to every candidate
def assign_final_candidate_dispositions(candidate_records: list[dict], modeling_policy: dict) -> dict:
    permitted_dispositions = set(
        modeling_policy[
            "candidate_dispositions"
        ]
    )

    disposition_counts = Counter()

    for candidate_record in candidate_records:
        selection_status = candidate_record[
            "selection_status"
        ]

        observation_assembly_status = (
            candidate_record.get(
                "observation_assembly_status"
            )
        )

        admissibility_status = candidate_record[
            "admissibility_status"
        ]

        fit_status = candidate_record[
            "fit_status"
        ]

        if selection_status == "not_selected":
            final_disposition = "not_selected"
            final_reason_code = candidate_record[
                "selection_reason_code"
            ]

        elif observation_assembly_status == "not_assembled":
            final_disposition = "not_assembled"
            final_reason_code = (
                candidate_record.get(
                    "admissibility_reason_code"
                )
                or "NO_ELIGIBLE_OBSERVATIONS"
            )

        elif admissibility_status == "inadmissible":
            final_disposition = "inadmissible"
            final_reason_code = (
                candidate_record.get(
                    "admissibility_reason_code"
                )
                or "CANDIDATE_INADMISSIBLE"
            )

        elif admissibility_status != "admissible":
            raise RuntimeError(
                "Selected candidate does not have a terminal admissibility status : "
                f"{candidate_record['candidate_identity']} : {admissibility_status}"
            )

        elif fit_status == "failed":
            final_disposition = "fit_failed"
            final_reason_code = (
                candidate_record.get(
                    "fit_reason_code"
                )
                or "MODEL_FIT_FAILED"
            )

        elif fit_status == "completed":
            final_disposition = "fitted"
            final_reason_code = None

        else:
            raise RuntimeError(
                "Admissible candidate does not have a terminal fit status : "
                f"{candidate_record['candidate_identity']} : {fit_status}"
            )

        if final_disposition not in (
            permitted_dispositions
        ):
            raise RuntimeError(
                "Candidate received an unsupported final disposition : "
                f"{candidate_record['candidate_identity']} : {final_disposition}"
            )

        candidate_record[
            "final_disposition"
        ] = final_disposition

        candidate_record[
            "final_reason_code"
        ] = final_reason_code

        disposition_counts[
            final_disposition
        ] += 1

    if sum(
        disposition_counts.values()
    ) != len(
        candidate_records
    ):
        raise RuntimeError("Final candidate dispositions do not reconcile to the candidate registry.")

    return {
        disposition_name: disposition_counts[
            disposition_name
        ]
        for disposition_name in sorted(
            permitted_dispositions
        )
    }


# Classifying fitted candidates for analytical trajectory reporting
def assign_candidate_analytical_result_status(candidate_records: list[dict], prepared_models: dict[str, dict],
                                              validation_metric_records: list[dict],
                                              validation_prediction_records: list[dict],
                                              condition_calibration_records: list[dict],
                                              modeling_policy: dict
) -> None:
    qualification_policy = modeling_policy[
        "validation_metrics"
    ][
        "analytical_qualification"
    ]

    r_squared_minimum = qualification_policy[
        "predictive_r_squared_minimum"
    ]

    validation_required = qualification_policy[
        "validation_execution_completed_required"
    ]

    uncertainty_required = qualification_policy[
        "uncertainty_completed_required"
    ]

    influence_required = qualification_policy[
        "influence_completed_required"
    ]

    predictive_completed_required = qualification_policy[
        "predictive_r_squared_completed_required"
    ]

    positive_required = qualification_policy[
        "strictly_positive_out_of_fold_predictions_required"
    ]

    interval_required = qualification_policy[
        "prediction_interval_metric_completed_required"
    ]

    calibration_required = qualification_policy[
        "condition_calibration_computation_required_for_every_condition_with_evaluation_domain_observations"
    ]

    metrics_by_candidate = {
        metric_record[
            "candidate_identity"
        ]: metric_record
        for metric_record in (
            validation_metric_records
        )
    }

    predictions_by_candidate = defaultdict(
        list
    )

    for prediction_record in (
        validation_prediction_records
    ):
        predictions_by_candidate[
            prediction_record[
                "candidate_identity"
            ]
        ].append(
            prediction_record
        )

    calibration_by_candidate = defaultdict(
        list
    )

    for calibration_record in (
        condition_calibration_records
    ):
        calibration_by_candidate[
            calibration_record[
                "candidate_identity"
            ]
        ].append(
            calibration_record
        )

    for candidate_record in candidate_records:
        candidate_identity = candidate_record[
            "candidate_identity"
        ]

        candidate_record[
            "numerical_fit_status"
        ] = candidate_record[
            "fit_status"
        ]

        candidate_record[
            "validation_execution_status"
        ] = candidate_record.get(
            "validation_status",
            "not_attempted"
        )

        candidate_record[
            "predictive_benchmark_status"
        ] = "not_assessed"

        candidate_record[
            "prediction_domain_status"
        ] = "not_assessed"

        candidate_record[
            "prediction_interval_assessment_status"
        ] = "not_assessed"

        candidate_record[
            "condition_calibration_assessment_status"
        ] = "not_assessed"

        if candidate_record[
            "fit_status"
        ] != "completed":
            candidate_record[
                "model_adequacy_status"
            ] = "not_available"

            candidate_record[
                "model_adequacy_reason_code"
            ] = (
                candidate_record.get(
                    "fit_reason_code"
                )
                or "MODEL_NOT_FITTED"
            )

            candidate_record[
                "analytical_result_status"
            ] = "not_available"

            candidate_record[
                "analytical_result_reason_code"
            ] = candidate_record[
                "model_adequacy_reason_code"
            ]

            continue

        if (
            validation_required
            and candidate_record[
                "validation_status"
            ] != "completed"
        ):
            candidate_record[
                "model_adequacy_status"
            ] = "diagnostic_only"

            candidate_record[
                "model_adequacy_reason_code"
            ] = (
                candidate_record[
                    "validation_reason_code"
                ]
                or "GROUPED_VALIDATION_UNAVAILABLE"
            )

        elif (
            uncertainty_required
            and candidate_record[
                "uncertainty_status"
            ] != "completed"
        ):
            candidate_record[
                "model_adequacy_status"
            ] = "diagnostic_only"

            candidate_record[
                "model_adequacy_reason_code"
            ] = (
                candidate_record[
                    "uncertainty_reason_code"
                ]
                or "UNCERTAINTY_UNAVAILABLE"
            )

        elif (
            influence_required
            and candidate_record[
                "influence_status"
            ] != "completed"
        ):
            candidate_record[
                "model_adequacy_status"
            ] = "diagnostic_only"

            candidate_record[
                "model_adequacy_reason_code"
            ] = (
                candidate_record[
                    "influence_reason_code"
                ]
                or "INFLUENCE_DIAGNOSTICS_UNAVAILABLE"
            )

        else:
            metric_record = metrics_by_candidate.get(
                candidate_identity
            )

            candidate_predictions = (
                predictions_by_candidate[
                    candidate_identity
                ]
            )

            candidate_calibration = (
                calibration_by_candidate[
                    candidate_identity
                ]
            )

            if metric_record is None:
                raise RuntimeError(f"Fitted candidate is missing validation metrics : {candidate_identity}")

            predictive_r_squared = (
                metric_record[
                    "predictive_r_squared"
                ]
            )

            if (
                predictive_completed_required
                and (
                    metric_record[
                        "predictive_r_squared_status"
                    ]
                    != "completed"
                    or predictive_r_squared is None
                )
            ):
                candidate_record[
                    "predictive_benchmark_status"
                ] = "unavailable"

                candidate_record[
                    "model_adequacy_status"
                ] = "diagnostic_only"

                candidate_record[
                    "model_adequacy_reason_code"
                ] = (
                    "PREDICTIVE_BENCHMARK_UNAVAILABLE"
                )

            elif (
                predictive_r_squared is not None
                and predictive_r_squared
                < r_squared_minimum
            ):
                candidate_record[
                    "predictive_benchmark_status"
                ] = "not_met"

                candidate_record[
                    "model_adequacy_status"
                ] = "diagnostic_only"

                candidate_record[
                    "model_adequacy_reason_code"
                ] = (
                    "PREDICTIVE_BENCHMARK_NOT_MET"
                )

            else:
                candidate_record[
                    "predictive_benchmark_status"
                ] = "met"

                predictions_are_positive = (
                    not positive_required
                    or (
                        bool(
                            candidate_predictions
                        )
                        and all(
                            prediction_record[
                                "predicted_outcome"
                            ] > 0
                            for prediction_record in (
                                candidate_predictions
                            )
                        )
                    )
                )

                candidate_record[
                    "prediction_domain_status"
                ] = (
                    "completed"
                    if predictions_are_positive
                    else "failed"
                )

                if not predictions_are_positive:
                    candidate_record[
                        "model_adequacy_status"
                    ] = "diagnostic_only"

                    candidate_record[
                        "model_adequacy_reason_code"
                    ] = (
                        "NONPOSITIVE_OUT_OF_FOLD_PREDICTION"
                    )

                elif (
                    interval_required
                    and metric_record[
                        "interval_metric_status"
                    ] != "completed"
                ):
                    candidate_record[
                        "prediction_interval_assessment_status"
                    ] = "unavailable"

                    candidate_record[
                        "model_adequacy_status"
                    ] = "diagnostic_only"

                    candidate_record[
                        "model_adequacy_reason_code"
                    ] = (
                        "PREDICTION_INTERVAL_ASSESSMENT_UNAVAILABLE"
                    )

                else:
                    candidate_record[
                        "prediction_interval_assessment_status"
                    ] = (
                        "completed"
                        if metric_record[
                            "interval_metric_status"
                        ] == "completed"
                        else "unavailable"
                    )

                    if calibration_required:
                        expected_conditions = {
                            observation_record[
                                "condition_protocol_identity"
                            ]
                            for observation_record in prepared_models[
                                candidate_identity
                            ][
                                "observation_records"
                            ]
                            if observation_record[
                                "inside_evaluation_interval"
                            ]
                        }

                        completed_calibration_conditions = {
                            calibration_record[
                                "condition_protocol_identity"
                            ]
                            for calibration_record in (
                                candidate_calibration
                            )
                            if calibration_record[
                                "calibration_status"
                            ] == "completed"
                        }

                        calibration_complete = (
                            expected_conditions
                            == completed_calibration_conditions
                        )

                    else:
                        calibration_complete = True

                    candidate_record[
                        "condition_calibration_assessment_status"
                    ] = (
                        "completed"
                        if calibration_complete
                        else "unavailable"
                    )

                    if not calibration_complete:
                        candidate_record[
                            "model_adequacy_status"
                        ] = "diagnostic_only"

                        candidate_record[
                            "model_adequacy_reason_code"
                        ] = (
                            "CONDITION_CALIBRATION_INCOMPLETE"
                        )

                    else:
                        candidate_record[
                            "model_adequacy_status"
                        ] = "qualified"

                        candidate_record[
                            "model_adequacy_reason_code"
                        ] = None

        if candidate_record[
            "model_adequacy_status"
        ] == "qualified":
            candidate_record[
                "analytical_result_status"
            ] = "available"

            candidate_record[
                "analytical_result_reason_code"
            ] = None

        else:
            candidate_record[
                "analytical_result_status"
            ] = candidate_record[
                "model_adequacy_status"
            ]

            candidate_record[
                "analytical_result_reason_code"
            ] = candidate_record[
                "model_adequacy_reason_code"
            ]


# Confirming complete candidate and result reconciliation
def validate_modeling_reconciliation(candidate_records: list[dict], candidate_term_records: list[dict],
                                     prepared_models: dict[str, dict], fitted_models_by_candidate: dict[str, dict],
                                     coefficient_records: list[dict], matrix_diagnostic_records: list[dict],
                                     fit_diagnostic_records: list[dict], validation_metric_records: list[dict],
                                     influence_records: list[dict], modeling_policy: dict
) -> dict:
    candidate_records_by_identity = (
        build_unique_record_index(
            records = candidate_records,
            identity_field = "candidate_identity",
            record_description = "Candidate registry"
        )
    )

    candidate_identities = set(
        candidate_records_by_identity
    )

    unknown_prepared_candidates = sorted(
        set(
            prepared_models
        ).difference(
            candidate_identities
        )
    )

    unknown_fitted_candidates = sorted(
        set(
            fitted_models_by_candidate
        ).difference(
            candidate_identities
        )
    )

    if unknown_prepared_candidates:
        raise RuntimeError(f"Prepared models reference unknown candidates : {unknown_prepared_candidates}")

    if unknown_fitted_candidates:
        raise RuntimeError(f"Fitted models reference unknown candidates : {unknown_fitted_candidates}")

    result_collections = {
        "candidate terms": candidate_term_records,
        "coefficients": coefficient_records,
        "matrix diagnostics": matrix_diagnostic_records,
        "fit diagnostics": fit_diagnostic_records,
        "validation metrics": validation_metric_records,
        "influence diagnostics": influence_records
    }

    for (
        collection_name,
        collection_records
    ) in result_collections.items():
        unknown_candidates = sorted(
            {
                record[
                    "candidate_identity"
                ]
                for record in collection_records
            }.difference(
                candidate_identities
            )
        )

        if unknown_candidates:
            raise RuntimeError(f"{collection_name} reference unknown candidates : {unknown_candidates}")

    permitted_term_dispositions = set(
        modeling_policy[
            "candidate_generation"
        ][
            "candidate_term_dispositions"
        ]
    )

    terms_by_candidate = defaultdict(
        list
    )

    for candidate_term_record in (
        candidate_term_records
    ):
        term_disposition = (
            candidate_term_record[
                "term_disposition"
            ]
        )

        if term_disposition not in (
            permitted_term_dispositions
        ):
            raise RuntimeError(
                "Candidate term contains an unsupported disposition : "
                f"{candidate_term_record['candidate_identity']} : {term_disposition}"
            )

        terms_by_candidate[
            candidate_term_record[
                "candidate_identity"
            ]
        ].append(
            candidate_term_record
        )

    candidates_without_terms = sorted(
        candidate_identity
        for candidate_identity in candidate_identities
        if candidate_identity not in terms_by_candidate
    )

    if candidates_without_terms:
        raise RuntimeError(f"Candidates do not have term-accounting records : {candidates_without_terms}")

    coefficients_by_candidate = defaultdict(
        list
    )

    for coefficient_record in coefficient_records:
        coefficients_by_candidate[
            coefficient_record[
                "candidate_identity"
            ]
        ].append(
            coefficient_record
        )

    matrix_diagnostics_by_candidate = defaultdict(
        list
    )

    for diagnostic_record in (
        matrix_diagnostic_records
    ):
        matrix_diagnostics_by_candidate[
            diagnostic_record[
                "candidate_identity"
            ]
        ].append(
            diagnostic_record
        )

    fit_diagnostics_by_candidate = defaultdict(
        list
    )

    for diagnostic_record in (
        fit_diagnostic_records
    ):
        fit_diagnostics_by_candidate[
            diagnostic_record[
                "candidate_identity"
            ]
        ].append(
            diagnostic_record
        )

    validation_metrics_by_candidate = defaultdict(
        list
    )

    for metric_record in (
        validation_metric_records
    ):
        validation_metrics_by_candidate[
            metric_record[
                "candidate_identity"
            ]
        ].append(
            metric_record
        )

    influence_by_candidate = defaultdict(
        list
    )

    for influence_record in influence_records:
        influence_by_candidate[
            influence_record[
                "candidate_identity"
            ]
        ].append(
            influence_record
        )

    fitted_candidate_count = 0
    failed_fit_candidate_count = 0
    inadmissible_candidate_count = 0
    not_assembled_candidate_count = 0
    not_selected_candidate_count = 0

    for candidate_identity in sorted(
        candidate_identities
    ):
        candidate_record = (
            candidate_records_by_identity[
                candidate_identity
            ]
        )

        final_disposition = (
            candidate_record[
                "final_disposition"
            ]
        )

        candidate_is_prepared = (
            candidate_identity
            in prepared_models
        )

        candidate_is_fitted = (
            candidate_identity
            in fitted_models_by_candidate
        )

        candidate_coefficients = (
            coefficients_by_candidate[
                candidate_identity
            ]
        )

        candidate_matrix_diagnostics = (
            matrix_diagnostics_by_candidate[
                candidate_identity
            ]
        )

        candidate_fit_diagnostics = (
            fit_diagnostics_by_candidate[
                candidate_identity
            ]
        )

        candidate_validation_metrics = (
            validation_metrics_by_candidate[
                candidate_identity
            ]
        )

        candidate_influence_records = (
            influence_by_candidate[
                candidate_identity
            ]
        )

        if final_disposition == "not_selected":
            not_selected_candidate_count += 1

            if (
                candidate_is_prepared
                or candidate_is_fitted
                or candidate_coefficients
                or candidate_matrix_diagnostics
                or candidate_fit_diagnostics
                or candidate_validation_metrics
                or candidate_influence_records
            ):
                raise RuntimeError(f"Non-selected candidate produced model results : {candidate_identity}")

            continue

        if final_disposition == "not_assembled":
            not_assembled_candidate_count += 1

            if (
                candidate_is_prepared
                or candidate_is_fitted
                or candidate_coefficients
                or candidate_matrix_diagnostics
                or candidate_fit_diagnostics
                or candidate_validation_metrics
                or candidate_influence_records
            ):
                raise RuntimeError(f"Unassembled candidate produced model results : {candidate_identity}")

            continue

        if final_disposition == "inadmissible":
            inadmissible_candidate_count += 1

            if (
                candidate_is_prepared
                or candidate_is_fitted
                or candidate_coefficients
                or candidate_fit_diagnostics
                or candidate_validation_metrics
                or candidate_influence_records
            ):
                raise RuntimeError(f"Inadmissible candidate produced fitted results : {candidate_identity}")

            if (
                candidate_record.get(
                    "model_matrix_status"
                )
                == "completed"
                and len(
                    candidate_matrix_diagnostics
                )
                != 1
            ):
                raise RuntimeError(
                    "Numerically evaluated inadmissible candidate does not have exactly one matrix diagnostic : "
                    f"{candidate_identity}"
                )

            continue

        if final_disposition == "fit_failed":
            failed_fit_candidate_count += 1

            if (
                not candidate_is_prepared
                or candidate_is_fitted
                or candidate_coefficients
                or len(
                    candidate_matrix_diagnostics
                )
                != 1
                or len(
                    candidate_fit_diagnostics
                )
                != 1
                or candidate_validation_metrics
                or candidate_influence_records
            ):
                raise RuntimeError(f"Failed-fit candidate has inconsistent result accounting : {candidate_identity}")

            continue

        if final_disposition != "fitted":
            raise RuntimeError(
                f"Candidate contains an unknown final disposition : {candidate_identity} : {final_disposition}"
            )

        fitted_candidate_count += 1

        if (
            not candidate_is_prepared
            or not candidate_is_fitted
            or len(
                candidate_matrix_diagnostics
            )
            != 1
            or len(
                candidate_fit_diagnostics
            )
            != 1
            or len(
                candidate_validation_metrics
            )
            != 1
        ):
            raise RuntimeError(
                f"Fitted candidate does not have complete model, diagnostic and validation accounting : {candidate_identity}"
            )

        expected_coefficient_count = len(
            prepared_models[
                candidate_identity
            ][
                "matrix_terms"
            ]
        )

        if len(
            candidate_coefficients
        ) != expected_coefficient_count:
            raise RuntimeError(f"Fitted candidate coefficient count does not match its model matrix : {candidate_identity}")

        uncertainty_status = candidate_record[
            "uncertainty_status"
        ]

        influence_status = candidate_record[
            "influence_status"
        ]

        if uncertainty_status not in {
            "completed",
            "unavailable"
        }:
            raise RuntimeError(
                f"Fitted candidate does not have a terminal uncertainty status : {candidate_identity} : {uncertainty_status}"
            )

        if influence_status not in {
            "completed",
            "unavailable"
        }:
            raise RuntimeError(
                f"Fitted candidate does not have a terminal influence status : {candidate_identity} : {influence_status}"
            )

        if not candidate_influence_records:
            raise RuntimeError(f"Fitted candidate does not have cell-influence records : {candidate_identity}")

        if candidate_record[
            "validation_status"
        ] not in {
            "completed",
            "unavailable"
        }:
            raise RuntimeError(
                "Fitted candidate does not have a terminal validation status : "
                f"{candidate_identity} : {candidate_record['validation_status']}"
            )

    reconciled_candidate_count = (
        fitted_candidate_count
        + failed_fit_candidate_count
        + inadmissible_candidate_count
        + not_assembled_candidate_count
        + not_selected_candidate_count
    )

    if reconciled_candidate_count != len(
        candidate_records
    ):
        raise RuntimeError("Candidate result populations do not reconcile to the candidate registry.")

    return {
        "candidate_count": len(
            candidate_records
        ),
        "fitted_candidate_count": (
            fitted_candidate_count
        ),
        "failed_fit_candidate_count": (
            failed_fit_candidate_count
        ),
        "inadmissible_candidate_count": (
            inadmissible_candidate_count
        ),
        "not_assembled_candidate_count": (
            not_assembled_candidate_count
        ),
        "not_selected_candidate_count": (
            not_selected_candidate_count
        ),
        "reconciliation_complete": True
    }


# Converting analytical output values to supported built-in types
def normalize_artifact_value(value: object) -> object:
    if isinstance(
        value,
        np.generic
    ):
        value = value.item()

    if isinstance(
        value,
        np.ndarray
    ):
        value = value.tolist()

    if value is None:
        return None

    if isinstance(
        value,
        bool
    ):
        return value

    if isinstance(
        value,
        int
    ):
        return value

    if isinstance(
        value,
        float
    ):
        if not math.isfinite(
            value
        ):
            raise RuntimeError("Analytical artifacts cannot contain non-finite numerical values.")

        return value

    if isinstance(
        value,
        str
    ):
        return value

    if isinstance(
        value,
        tuple
    ):
        value = list(
            value
        )

    if isinstance(
        value,
        list
    ):
        return [
            normalize_artifact_value(
                item
            )
            for item in value
        ]

    if isinstance(
        value,
        dict
    ):
        if any(
            not isinstance(
                field_name,
                str
            )
            for field_name in value
        ):
            raise RuntimeError("Analytical artifact JSON objects require string field names.")

        return {
            field_name: normalize_artifact_value(
                value[
                    field_name
                ]
            )
            for field_name in sorted(
                value
            )
        }

    raise RuntimeError(f"Analytical artifacts contain an unsupported value type : {type(value).__name__}")


# Serializing one analytical artifact value
def serialize_artifact_value(value: object) -> object:
    normalized_value = normalize_artifact_value(
        value
    )

    if normalized_value is None:
        return ""

    if isinstance(
        normalized_value,
        bool
    ):
        return (
            "true"
            if normalized_value
            else "false"
        )

    if isinstance(
        normalized_value,
        (list, dict)
    ):
        return json.dumps(
            normalized_value,
            sort_keys = True,
            separators = (
                ",",
                ":"
            ),
            allow_nan = False
        )

    return normalized_value


# Classifying one analytical artifact value for a table schema
def classify_artifact_value(value: object) -> str:
    normalized_value = normalize_artifact_value(
        value
    )

    if normalized_value is None:
        return "null"

    if isinstance(
        normalized_value,
        bool
    ):
        return "boolean"

    if isinstance(
        normalized_value,
        int
    ):
        return "integer"

    if isinstance(
        normalized_value,
        float
    ):
        return "number"

    if isinstance(
        normalized_value,
        str
    ):
        return "string"

    if isinstance(
        normalized_value,
        (list, dict)
    ):
        return "json"

    raise RuntimeError("Analytical artifact contains an unsupported classified value.")


# Building a canonical representation of one table-key value
def build_artifact_key_component(value: object) -> str:
    normalized_value = normalize_artifact_value(
        value
    )

    return json.dumps(
        normalized_value,
        sort_keys = True,
        separators = (
            ",",
            ":"
        ),
        allow_nan = False
    )


# Validating and sorting records against one declared table contract
def normalize_artifact_records(table_name: str, records: list[dict],
                               field_names: list[str], primary_key: list[str]
) -> list[dict]:
    if not table_name:
        raise ValueError("Analytical table name cannot be empty.")

    if (
        not field_names
        or len(field_names)
        != len(
            set(
                field_names
            )
        )
    ):
        raise RuntimeError(f"Analytical table has an invalid declared schema : {table_name}")

    if (
        not primary_key
        or len(primary_key)
        != len(
            set(
                primary_key
            )
        )
    ):
        raise RuntimeError(f"Analytical table has an invalid primary key : {table_name}")

    missing_primary_key_fields = sorted(
        set(
            primary_key
        ).difference(
            field_names
        )
    )

    if missing_primary_key_fields:
        raise RuntimeError(
            f"Analytical table primary key references undeclared fields : {table_name} : {missing_primary_key_fields}"
        )

    declared_fields = set(
        field_names
    )

    normalized_records = []
    observed_primary_keys = set()

    for row_number, record in enumerate(
        records,
        start = 1
    ):
        if not isinstance(
            record,
            dict
        ):
            raise RuntimeError(f"Analytical table contains a non-object record : {table_name} : row {row_number}")

        record_fields = set(
            record
        )

        missing_fields = sorted(
            declared_fields.difference(
                record_fields
            )
        )

        unexpected_fields = sorted(
            record_fields.difference(
                declared_fields
            )
        )

        if (
            missing_fields
            or unexpected_fields
        ):
            raise RuntimeError(
                "Analytical table record does not match its declared schema : "
                f"{table_name} : row {row_number} : missing {missing_fields} : unexpected {unexpected_fields}"
            )

        normalized_record = {
            field_name: normalize_artifact_value(
                record[
                    field_name
                ]
            )
            for field_name in field_names
        }

        primary_key_value = tuple(
            build_artifact_key_component(
                normalized_record[
                    field_name
                ]
            )
            for field_name in primary_key
        )

        if primary_key_value in observed_primary_keys:
            raise RuntimeError(f"Analytical table contains a duplicate primary key : {table_name} : {primary_key_value}")

        observed_primary_keys.add(
            primary_key_value
        )

        normalized_records.append(
            normalized_record
        )

    return sorted(
        normalized_records,
        key = lambda record: tuple(
            build_artifact_key_component(
                record[
                    field_name
                ]
            )
            for field_name in primary_key
        )
    )


# Building one deterministic analytical-table schema
def build_artifact_table_schema(records: list[dict], field_names: list[str]) -> list[dict]:
    table_schema = []

    for field_name in field_names:
        observed_types = {
            classify_artifact_value(
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

        elif len(
            observed_types
        ) == 1:
            data_type = next(
                iter(
                    observed_types
                )
            )

        elif not observed_types:
            data_type = "null"

        else:
            raise RuntimeError(
                f"Analytical table field contains incompatible value types : {field_name} : {sorted(observed_types)}"
            )

        table_schema.append(
            {
                "name": field_name,
                "data_type": data_type,
                "nullable": (
                    not records
                    or any(
                        record[
                            field_name
                        ]
                        is None
                        for record in records
                    )
                )
            }
        )

    return table_schema


# Writing one deterministic analytical table
def write_artifact_table(output_path: Path, table_name: str,
                         records: list[dict], field_names: list[str],
                         primary_key: list[str]
) -> dict:
    normalized_records = normalize_artifact_records(
        table_name = table_name,
        records = records,
        field_names = field_names,
        primary_key = primary_key
    )

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
            extrasaction = "raise",
            lineterminator = "\n"
        )

        table_writer.writeheader()

        for record in normalized_records:
            table_writer.writerow(
                {
                    field_name: serialize_artifact_value(
                        record[
                            field_name
                        ]
                    )
                    for field_name in field_names
                }
            )

    return {
        "file_name": output_path.name,
        "row_count": len(
            normalized_records
        ),
        "sha256": calculate_sha256(
            output_path
        ),
        "primary_key": list(
            primary_key
        ),
        "schema": build_artifact_table_schema(
            records = normalized_records,
            field_names = field_names
        )
    }


# Recording the declared software environment
def build_software_version_record() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": distribution_version(
            "numpy"
        ),
        "scipy": distribution_version(
            "scipy"
        ),
        "statsmodels": distribution_version(
            "statsmodels"
        )
    }


# Writing one canonical JSON artifact
def write_canonical_json(output_path: Path, content: dict) -> None:
    normalized_content = normalize_artifact_value(
        content
    )

    output_path.parent.mkdir(
        parents = True,
        exist_ok = True
    )

    with output_path.open(
        mode = "w",
        encoding = "utf-8",
        newline = "\n"
    ) as output_stream:
        json.dump(
            normalized_content,
            output_stream,
            sort_keys = True,
            separators = (
                ",",
                ":"
            ),
            allow_nan = False
        )

        output_stream.write(
            "\n"
        )


# Classifying model availability from completed fit attempts
def resolve_model_availability_status(fitted_count: int, failed_fit_count: int) -> str:
    attempted_fit_count = (
        fitted_count
        + failed_fit_count
    )

    if fitted_count > 0 and failed_fit_count > 0:
        return "partially_available"

    if fitted_count > 0:
        return "available"

    if attempted_fit_count > 0:
        return "unavailable_all_fit_attempts_failed"

    return "not_evaluated_no_fit_attempts"


# Summarizing candidate outcomes by estimator or template
def build_candidate_outcome_summary(candidate_records: list[dict], grouping_field: str,
                                    expected_values: list[str], modeling_policy: dict
) -> dict:
    expected_value_set = set(
        expected_values
    )

    observed_values = {
        candidate_record[
            grouping_field
        ]
        for candidate_record in candidate_records
    }

    unexpected_values = sorted(
        observed_values.difference(
            expected_value_set
        )
    )

    if unexpected_values:
        raise RuntimeError(f"Candidate outcome summary contains unexpected values : {grouping_field} : {unexpected_values}")

    candidate_dispositions = (
        modeling_policy[
            "candidate_dispositions"
        ]
    )

    summaries = {}

    for grouping_value in sorted(
        expected_value_set
    ):
        grouped_records = [
            candidate_record
            for candidate_record in candidate_records
            if candidate_record[
                grouping_field
            ] == grouping_value
        ]

        disposition_counts = Counter(
            candidate_record[
                "final_disposition"
            ]
            for candidate_record in grouped_records
        )

        unknown_dispositions = sorted(
            set(
                disposition_counts
            ).difference(
                candidate_dispositions
            )
        )

        if unknown_dispositions:
            raise RuntimeError(
                "Candidate outcome summary contains unsupported dispositions : "
                f"{grouping_field} : {grouping_value} : {unknown_dispositions}"
            )

        fitted_count = disposition_counts[
            "fitted"
        ]

        failed_fit_count = disposition_counts[
            "fit_failed"
        ]

        summaries[
            grouping_value
        ] = {
            "candidate_count": len(
                grouped_records
            ),
            "fit_attempted_count": (
                fitted_count
                + failed_fit_count
            ),
            "fitted_count": fitted_count,
            "failed_fit_count": (
                failed_fit_count
            ),
            "availability_status": (
                resolve_model_availability_status(
                    fitted_count = fitted_count,
                    failed_fit_count = (
                        failed_fit_count
                    )
                )
            ),
            "candidate_disposition_counts": {
                disposition: disposition_counts[
                    disposition
                ]
                for disposition in sorted(
                    candidate_dispositions
                )
            }
        }

    return summaries


# Summarizing downstream availability for successfully fitted candidates
def build_fitted_result_availability(candidate_records: list[dict]) -> dict:
    fitted_records = [
        candidate_record
        for candidate_record in candidate_records
        if candidate_record[
            "final_disposition"
        ] == "fitted"
    ]

    status_fields = {
        "uncertainty": "uncertainty_status",
        "validation": "validation_status",
        "influence": "influence_status"
    }

    availability_summary = {}

    for result_name, status_field in (
        status_fields.items()
    ):
        status_counts = Counter(
            candidate_record[
                status_field
            ]
            for candidate_record in fitted_records
        )

        unexpected_statuses = sorted(
            set(
                status_counts
            ).difference(
                {
                    "completed",
                    "unavailable"
                }
            )
        )

        if unexpected_statuses:
            raise RuntimeError(
                f"Fitted-result availability contains unsupported statuses : {result_name} : {unexpected_statuses}"
            )

        availability_summary[
            result_name
        ] = {
            "completed": status_counts[
                "completed"
            ],
            "unavailable": status_counts[
                "unavailable"
            ]
        }

    return availability_summary


# Resolving the completed report status
def resolve_completed_analysis_status(estimator_outcomes: dict) -> str:
    unavailable_estimators = [
        estimator_name
        for estimator_name, estimator_outcome in (
            estimator_outcomes.items()
        )
        if estimator_outcome[
            "availability_status"
        ] == "unavailable_all_fit_attempts_failed"
    ]

    if unavailable_estimators:
        return (
            "completed_with_model_family_failures"
        )

    return "completed"


# Resolving analytical handoff availability
def resolve_analytical_handoff_status(candidate_status_counts: Counter, unavailable_estimator_families: list[str]) -> str:
    if candidate_status_counts[
        "available"
    ] == 0:
        return (
            "not_ready_no_analytically_qualified_trajectory_models"
        )

    if (
        unavailable_estimator_families
        or candidate_status_counts[
            "diagnostic_only"
        ] > 0
    ):
        return (
            "ready_for_continuous_trajectory_analysis_with_model_family_and_candidate_restrictions"
        )

    return ("ready_for_continuous_trajectory_analysis")


# Summarizing downstream analytical availability
def build_analytical_handoff_summary(candidate_records: list[dict], contrast_records: list[dict],
                                     estimator_outcomes: dict
) -> dict:
    candidate_status_counts = Counter(
        candidate_record["analytical_result_status"]
        for candidate_record in candidate_records
    )

    contrast_status_counts = Counter(
        contrast_record["contrast_status"]
        for contrast_record in contrast_records
    )

    unavailable_estimator_families = sorted(
        estimator_name
        for estimator_name, estimator_outcome in estimator_outcomes.items()
        if estimator_outcome["availability_status"]
        == "unavailable_all_fit_attempts_failed"
    )

    handoff_status = (
        resolve_analytical_handoff_status(
            candidate_status_counts = (
                candidate_status_counts
            ),
            unavailable_estimator_families = (
                unavailable_estimator_families
            )
        )
    )

    return {
        "analytical_handoff_status": handoff_status,
        "candidate_analytical_status_counts": dict(
            sorted(candidate_status_counts.items())
        ),
        "contrast_status_counts": dict(
            sorted(contrast_status_counts.items())
        ),
        "unavailable_estimator_families": unavailable_estimator_families
    }


# Summarizing diagnostic-only analytical qualification reasons for reporting
def build_analytical_qualification_summary(candidate_records: list[dict]) -> dict:
    fitted_diagnostic_candidates = [
        candidate_record
        for candidate_record in candidate_records
        if (
            candidate_record.get(
                "final_disposition"
            )
            == "fitted"
            and candidate_record.get(
                "model_adequacy_status"
            )
            == "diagnostic_only"
        )
    ]

    adequacy_reason_counts = Counter(
        candidate_record[
            "model_adequacy_reason_code"
        ]
        for candidate_record in (
            fitted_diagnostic_candidates
        )
    )

    if None in adequacy_reason_counts:
        raise RuntimeError("Diagnostic-only fitted candidate is missing its adequacy reason code.")

    adequacy_reason_counts_by_estimator = {}

    for candidate_record in (
        fitted_diagnostic_candidates
    ):
        estimator_name = candidate_record[
            "estimator"
        ]

        reason_code = candidate_record[
            "model_adequacy_reason_code"
        ]

        adequacy_reason_counts_by_estimator.setdefault(
            estimator_name,
            Counter()
        )[
            reason_code
        ] += 1

    return {
        "diagnostic_only_candidate_count": len(
            fitted_diagnostic_candidates
        ),
        "model_adequacy_reason_counts": {
            reason_code: reason_count
            for reason_code, reason_count in sorted(
                adequacy_reason_counts.items()
            )
        },
        "model_adequacy_reason_counts_by_estimator": {
            estimator_name: {
                reason_code: reason_count
                for reason_code, reason_count in sorted(
                    estimator_reason_counts.items()
                )
            }
            for estimator_name, estimator_reason_counts in sorted(
                adequacy_reason_counts_by_estimator.items()
            )
        }
    }


# Building the completed descriptive-modeling report
def build_descriptive_modeling_report(validated_inputs: dict, workflow_results: dict, table_manifests: dict) -> dict:
    modeling_policy = validated_inputs[
        "modeling_policy"
    ]

    candidate_records = workflow_results[
        "candidate_registry"
    ]

    estimator_names = sorted(
        {
            candidate_template[
                "estimator"
            ]
            for candidate_template in modeling_policy[
                "candidate_templates"
            ]
        }
    )

    template_names = sorted(
        candidate_template[
            "template_name"
        ]
        for candidate_template in modeling_policy[
            "candidate_templates"
        ]
    )

    estimator_outcomes = (
        build_candidate_outcome_summary(
            candidate_records = candidate_records,
            grouping_field = "estimator",
            expected_values = estimator_names,
            modeling_policy = modeling_policy
        )
    )

    template_outcomes = (
        build_candidate_outcome_summary(
            candidate_records = candidate_records,
            grouping_field = "template_name",
            expected_values = template_names,
            modeling_policy = modeling_policy
        )
    )

    fitted_result_availability = (
        build_fitted_result_availability(
            candidate_records
        )
    )

    analytical_qualification_summary = (
        build_analytical_qualification_summary(
            candidate_records
        )
    )

    analysis_status = (
        resolve_completed_analysis_status(
            estimator_outcomes
        )
    )

    return {
        "analytical_handoff": (
            build_analytical_handoff_summary(
                candidate_records = candidate_records,
                contrast_records = workflow_results["model_contrasts"],
                estimator_outcomes = estimator_outcomes
            )
        ),
        "analysis_status": analysis_status,
        "analysis_registration_status": (
            modeling_policy[
                "reporting"
            ][
                "analysis_registration_status"
            ]
        ),
        "analysis_registration_rationale": (
            modeling_policy[
                "reporting"
            ][
                "analysis_registration_rationale"
            ]
        ),
        "independent_experimental_unit": "cell",
        "calendar_and_cycle_models_separated": True,
        "primary_and_sensitivity_cohorts_separated": True,
        "outcome_magnitude_or_direction_used_for_candidate_selection": False,
        "performance_selected_model_declared": False,
        "causal_interpretation_authorized": False,
        "policy_recommendation_authorized": False,
        "modeling_policy_version": modeling_policy[
            "policy_version"
        ],
        "modeling_policy_effective_date": modeling_policy[
            "effective_date"
        ],
        "modeling_policy_sha256": validated_inputs[
            "modeling_policy_sha256"
        ],
        "modeling_policy": modeling_policy,
        "inputs": {
            "feature_manifest": {
                "sha256": validated_inputs[
                    "feature_manifest_sha256"
                ],
                "tables": validated_inputs[
                    "feature_manifest"
                ][
                    "tables"
                ]
            },
            "experimental_design_report": {
                "sha256": validated_inputs[
                    "design_report_sha256"
                ],
                "tables": validated_inputs[
                    "design_report"
                ][
                    "tables"
                ]
            }
        },
        "candidate_disposition_counts": workflow_results[
            "candidate_disposition_counts"
        ],
        "estimator_outcomes": estimator_outcomes,
        "template_outcomes": template_outcomes,
        "fitted_result_availability": (
            fitted_result_availability
        ),
        "analytical_qualification_summary": (
            analytical_qualification_summary
        ),
        "reconciliation": workflow_results[
            "reconciliation"
        ],
        "tables": {
            table_name: table_manifests[
                table_name
            ]
            for table_name in sorted(
                table_manifests
            )
        },
        "software_versions": (
            build_software_version_record()
        ),
        "numerical_thread_environment": {
            environment_name: os.environ.get(
                environment_name
            )
            for environment_name in sorted(
                REQUIRED_THREAD_ENVIRONMENT
            )
        }
    }


# Validating one completed descriptive-modeling report
def validate_descriptive_modeling_report(report: dict, modeling_policy: dict) -> None:
    estimator_outcomes = report.get(
        "estimator_outcomes"
    )

    if not isinstance(
        estimator_outcomes,
        dict
    ):
        raise RuntimeError("Descriptive-modeling report does not contain estimator outcomes.")

    expected_analysis_status = (
        resolve_completed_analysis_status(
            estimator_outcomes
        )
    )

    if report.get(
        "analysis_status"
    ) != expected_analysis_status:
        raise RuntimeError("Descriptive-modeling report status does not agree with estimator outcomes.")

    if report.get(
        "modeling_policy"
    ) != modeling_policy:
        raise RuntimeError("Descriptive-modeling report does not preserve the exact modeling policy.")

    if report.get(
        "independent_experimental_unit"
    ) != "cell":
        raise RuntimeError("Descriptive-modeling report contains an invalid experimental unit.")

    expected_registration_status = (
        modeling_policy[
            "reporting"
        ][
            "analysis_registration_status"
        ]
    )

    if report.get(
        "analysis_registration_status"
    ) != expected_registration_status:
        raise RuntimeError("Descriptive-modeling report analysis_registration_status does not match the policy.")

    expected_registration_rationale = (
        modeling_policy[
            "reporting"
        ][
            "analysis_registration_rationale"
        ]
    )

    if report.get(
        "analysis_registration_rationale"
    ) != expected_registration_rationale:
        raise RuntimeError("Descriptive-modeling report analysis_registration_rationale does not match the policy.")

    if (
        modeling_policy[
            "reporting"
        ][
            "analysis_registration_status_required"
        ]
        and "analysis_registration_status"
        not in report
    ):
        raise RuntimeError("Descriptive-modeling report is missing a policy-required analysis_registration_status.")

    required_declaration_values = {
        "outcome_magnitude_or_direction_used_for_candidate_selection": False,
        "performance_selected_model_declared": False,
        "causal_interpretation_authorized": False,
        "policy_recommendation_authorized": False,
        "calendar_and_cycle_models_separated": True,
        "primary_and_sensitivity_cohorts_separated": True
    }

    invalid_declarations = sorted(
        field_name
        for field_name, required_value in required_declaration_values.items()
        if report.get(field_name) is not required_value
    )

    if invalid_declarations:
        raise RuntimeError(f"Descriptive-modeling report contains invalid analytical declarations : {invalid_declarations}")

    table_manifests = report.get(
        "tables"
    )

    if (
        not isinstance(
            table_manifests,
            dict
        )
        or set(
            table_manifests
        )
        != REQUIRED_OUTPUT_TABLES
    ):
        raise RuntimeError("Descriptive-modeling report does not contain the complete table inventory.")

    reconciliation = report.get(
        "reconciliation",
        {}
    )

    if reconciliation.get(
        "reconciliation_complete"
    ) is not True:
        raise RuntimeError("Descriptive-modeling report does not contain completed reconciliation.")

    candidate_disposition_counts = report.get(
        "candidate_disposition_counts"
    )

    if not isinstance(
        candidate_disposition_counts,
        dict
    ):
        raise RuntimeError("Descriptive-modeling report does not contain candidate disposition counts.")

    expected_estimator_names = {
        candidate_template[
            "estimator"
        ]
        for candidate_template in modeling_policy[
            "candidate_templates"
        ]
    }

    expected_template_names = {
        candidate_template[
            "template_name"
        ]
        for candidate_template in modeling_policy[
            "candidate_templates"
        ]
    }

    grouped_outcomes = {
        "estimator_outcomes": (
            expected_estimator_names
        ),
        "template_outcomes": (
            expected_template_names
        )
    }

    for summary_name, expected_names in (
        grouped_outcomes.items()
    ):
        summary = report.get(
            summary_name
        )

        if (
            not isinstance(
                summary,
                dict
            )
            or set(
                summary
            )
            != expected_names
        ):
            raise RuntimeError(f"Descriptive-modeling report contains an invalid outcome summary : {summary_name}")

        aggregated_disposition_counts = Counter()

        for group_name, group_summary in (
            summary.items()
        ):
            disposition_counts = group_summary.get(
                "candidate_disposition_counts"
            )

            if (
                not isinstance(
                    disposition_counts,
                    dict
                )
                or set(
                    disposition_counts
                )
                != set(
                    modeling_policy[
                        "candidate_dispositions"
                    ]
                )
            ):
                raise RuntimeError(
                    f"Model outcome summary contains invalid disposition counts : {summary_name} : {group_name}"
                )

            if any(
                not isinstance(
                    disposition_count,
                    int
                )
                or isinstance(
                    disposition_count,
                    bool
                )
                or disposition_count < 0
                for disposition_count in (
                    disposition_counts.values()
                )
            ):
                raise RuntimeError(f"Model outcome summary contains invalid count values : {summary_name} : {group_name}")

            fitted_count = disposition_counts[
                "fitted"
            ]

            failed_fit_count = disposition_counts[
                "fit_failed"
            ]

            if group_summary.get(
                "candidate_count"
            ) != sum(
                disposition_counts.values()
            ):
                raise RuntimeError(
                    f"Model outcome summary candidate count does not reconcile : {summary_name} : {group_name}"
                )

            if group_summary.get(
                "fit_attempted_count"
            ) != (
                fitted_count
                + failed_fit_count
            ):
                raise RuntimeError(
                    f"Model outcome summary fit-attempt count does not reconcile : {summary_name} : {group_name}"
                )

            if (
                group_summary.get(
                    "fitted_count"
                )
                != fitted_count
                or group_summary.get(
                    "failed_fit_count"
                )
                != failed_fit_count
            ):
                raise RuntimeError(f"Model outcome summary fit counts do not reconcile : {summary_name} : {group_name}")

            expected_availability_status = (
                resolve_model_availability_status(
                    fitted_count = fitted_count,
                    failed_fit_count = (
                        failed_fit_count
                    )
                )
            )

            if group_summary.get(
                "availability_status"
            ) != expected_availability_status:
                raise RuntimeError(f"Model outcome summary availability does not reconcile : {summary_name} : {group_name}")

            aggregated_disposition_counts.update(
                disposition_counts
            )

        if dict(
            sorted(
                aggregated_disposition_counts.items()
            )
        ) != candidate_disposition_counts:
            raise RuntimeError(f"Model outcome summary does not reconcile to candidate dispositions : {summary_name}")

    fitted_candidate_count = reconciliation[
        "fitted_candidate_count"
    ]

    fitted_result_availability = report.get(
        "fitted_result_availability"
    )

    if (
        not isinstance(
            fitted_result_availability,
            dict
        )
        or set(
            fitted_result_availability
        )
        != {
            "uncertainty",
            "validation",
            "influence"
        }
    ):
        raise RuntimeError("Descriptive-modeling report contains invalid fitted-result availability.")

    analytical_handoff = report.get("analytical_handoff")

    if not isinstance(analytical_handoff, dict):
        raise RuntimeError("Descriptive-modeling report does not contain analytical handoff.")

    required_handoff_fields = {
        "analytical_handoff_status",
        "candidate_analytical_status_counts",
        "contrast_status_counts",
        "unavailable_estimator_families"
    }

    if set(analytical_handoff) != required_handoff_fields:
        raise RuntimeError("Analytical handoff contains an invalid field set.")

    if analytical_handoff["analytical_handoff_status"] not in {
        "ready_for_continuous_trajectory_analysis",
        "ready_for_continuous_trajectory_analysis_with_model_family_and_candidate_restrictions",
        "not_ready_no_analytically_qualified_trajectory_models"
    }:
        raise RuntimeError("Analytical handoff status is invalid.")

    for result_name, result_counts in (
        fitted_result_availability.items()
    ):
        if (
            not isinstance(
                result_counts,
                dict
            )
            or set(
                result_counts
            )
            != {
                "completed",
                "unavailable"
            }
            or any(
                not isinstance(
                    result_count,
                    int
                )
                or isinstance(
                    result_count,
                    bool
                )
                or result_count < 0
                for result_count in (
                    result_counts.values()
                )
            )
            or sum(
                result_counts.values()
            )
            != fitted_candidate_count
        ):
            raise RuntimeError(f"Fitted-result availability does not reconcile : {result_name}")

    candidate_analytical_status_counts = (
        analytical_handoff[
            "candidate_analytical_status_counts"
        ]
    )

    contrast_status_counts = (
        analytical_handoff[
            "contrast_status_counts"
        ]
    )

    permitted_candidate_statuses = set(
        modeling_policy[
            "reporting"
        ][
            "candidate_analytical_result_statuses"
        ]
    )

    permitted_contrast_statuses = set(
        modeling_policy[
            "contrast_evaluation"
        ][
            "contrast_statuses"
        ]
    )

    if not set(
        candidate_analytical_status_counts
    ).issubset(
        permitted_candidate_statuses
    ):
        raise RuntimeError("Analytical handoff contains unsupported candidate statuses.")

    if not set(
        contrast_status_counts
    ).issubset(
        permitted_contrast_statuses
    ):
        raise RuntimeError("Analytical handoff contains unsupported contrast statuses.")

    if sum(
        candidate_analytical_status_counts.values()
    ) != reconciliation[
        "candidate_count"
    ]:
        raise RuntimeError("Analytical candidate-status counts do not reconcile.")

    if sum(
        contrast_status_counts.values()
    ) != report[
        "tables"
    ][
        "model_contrasts"
    ][
        "row_count"
    ]:
        raise RuntimeError("Analytical contrast-status counts do not reconcile.")

    expected_unavailable_estimators = sorted(
        estimator_name
        for estimator_name, estimator_outcome in (
            estimator_outcomes.items()
        )
        if estimator_outcome[
            "availability_status"
        ] == (
            "unavailable_all_fit_attempts_failed"
        )
    )

    if (
        analytical_handoff[
            "unavailable_estimator_families"
        ]
        != expected_unavailable_estimators
    ):
        raise RuntimeError("Analytical handoff contains inconsistent estimator availability.")

    expected_handoff_status = (
        resolve_analytical_handoff_status(
            candidate_status_counts = Counter(
                candidate_analytical_status_counts
            ),
            unavailable_estimator_families = (
                expected_unavailable_estimators
            )
        )
    )

    if (
        analytical_handoff[
            "analytical_handoff_status"
        ]
        != expected_handoff_status
    ):
        raise RuntimeError("Analytical handoff status does not reconcile.")

    analytical_qualification_summary = report.get(
        "analytical_qualification_summary"
    )

    if not isinstance(
        analytical_qualification_summary,
        dict
    ):
        raise RuntimeError("Descriptive-modeling report does not contain an analytical qualification summary.")

    if set(
        analytical_qualification_summary
    ) != {
        "diagnostic_only_candidate_count",
        "model_adequacy_reason_counts",
        "model_adequacy_reason_counts_by_estimator"
    }:
        raise RuntimeError("Analytical qualification summary contains an invalid field set.")

    diagnostic_only_candidate_count = (
        analytical_qualification_summary[
            "diagnostic_only_candidate_count"
        ]
    )

    if (
        not isinstance(
            diagnostic_only_candidate_count,
            int
        )
        or isinstance(
            diagnostic_only_candidate_count,
            bool
        )
        or diagnostic_only_candidate_count < 0
    ):
        raise RuntimeError("Analytical qualification summary contains an invalid diagnostic-only candidate count.")

    model_adequacy_reason_counts = (
        analytical_qualification_summary[
            "model_adequacy_reason_counts"
        ]
    )

    if not isinstance(
        model_adequacy_reason_counts,
        dict
    ):
        raise RuntimeError("Analytical qualification summary contains an invalid reason-count mapping.")

    if any(
        not isinstance(reason_code, str)
        or not reason_code
        or not isinstance(reason_count, int)
        or isinstance(reason_count, bool)
        or reason_count < 0
        for reason_code, reason_count in (
            model_adequacy_reason_counts.items()
        )
    ):
        raise RuntimeError("Analytical qualification summary contains an invalid reason-count entry.")

    if sum(
        model_adequacy_reason_counts.values()
    ) != diagnostic_only_candidate_count:
        raise RuntimeError("Analytical qualification reason counts do not reconcile to the diagnostic-only candidate count.")

    model_adequacy_reason_counts_by_estimator = (
        analytical_qualification_summary[
            "model_adequacy_reason_counts_by_estimator"
        ]
    )

    if not isinstance(
        model_adequacy_reason_counts_by_estimator,
        dict
    ):
        raise RuntimeError("Analytical qualification summary contains an invalid estimator breakdown.")

    if not set(
        model_adequacy_reason_counts_by_estimator
    ).issubset(
        expected_estimator_names
    ):
        raise RuntimeError("Analytical qualification summary references an unknown estimator.")

    aggregated_estimator_reason_counts = Counter()

    for (
        estimator_name,
        estimator_reason_counts
    ) in model_adequacy_reason_counts_by_estimator.items():
        if not isinstance(
            estimator_reason_counts,
            dict
        ):
            raise RuntimeError(
                f"Analytical qualification summary contains an invalid estimator reason mapping : {estimator_name}"
            )

        if any(
            not isinstance(reason_code, str)
            or not reason_code
            or not isinstance(reason_count, int)
            or isinstance(reason_count, bool)
            or reason_count < 0
            for reason_code, reason_count in (
                estimator_reason_counts.items()
            )
        ):
            raise RuntimeError(
                f"Analytical qualification summary contains an invalid estimator reason entry : {estimator_name}"
            )

        aggregated_estimator_reason_counts.update(
            estimator_reason_counts
        )

    if dict(
        sorted(
            aggregated_estimator_reason_counts.items()
        )
    ) != dict(
        sorted(
            model_adequacy_reason_counts.items()
        )
    ):
        raise RuntimeError("Analytical qualification summary estimator breakdown does not reconcile to the reason counts.")

    if (
        candidate_analytical_status_counts.get(
            "diagnostic_only",
            0
        )
        != diagnostic_only_candidate_count
    ):
        raise RuntimeError("Analytical qualification summary does not reconcile to the analytical handoff.")
    

# Writing and validating a complete staged analytical artifact set
def stage_descriptive_modeling_artifacts(validated_inputs: dict, workflow_results: dict,
                                         staging_directory: Path
)-> dict:
    modeling_policy = validated_inputs[
        "modeling_policy"
    ]

    artifact_contract = modeling_policy[
        "artifact_contract"
    ]

    table_file_extension = artifact_contract[
        "table_file_extension"
    ]

    table_manifests = {}

    for table_name in artifact_contract[
        "output_tables"
    ]:
        if table_name not in workflow_results:
            raise RuntimeError(f"Modeling workflow did not produce a required table : {table_name}")

        table_contract = OUTPUT_TABLE_CONTRACTS[
            table_name
        ]

        table_path = (
            staging_directory
            / f"{table_name}{table_file_extension}"
        )

        table_manifest = write_artifact_table(
            output_path = table_path,
            table_name = table_name,
            records = workflow_results[
                table_name
            ],
            field_names = table_contract[
                "field_names"
            ],
            primary_key = table_contract[
                "primary_key"
            ]
        )

        validate_written_artifact_table(
            table_path = table_path,
            table_manifest = table_manifest
        )

        table_manifests[
            table_name
        ] = table_manifest

    report = build_descriptive_modeling_report(
        validated_inputs = validated_inputs,
        workflow_results = workflow_results,
        table_manifests = table_manifests
    )

    validate_descriptive_modeling_report(
        report = report,
        modeling_policy = modeling_policy
    )

    report_file_name = artifact_contract[
        "report_file_name"
    ]

    report_path = (
        staging_directory
        / report_file_name
    )

    write_canonical_json(
        output_path = report_path,
        content = report
    )

    reloaded_report = load_strict_json_object(
        report_path
    )

    validate_descriptive_modeling_report(
        report = reloaded_report,
        modeling_policy = modeling_policy
    )

    for (
        table_name,
        table_manifest
    ) in table_manifests.items():
        validate_written_artifact_table(
            table_path = (
                staging_directory
                / table_manifest[
                    "file_name"
                ]
            ),
            table_manifest = table_manifest
        )

    return report


# Confirming that a written analytical table matches its manifest entry
def validate_written_artifact_table(table_path: Path, table_manifest: dict) -> None:
    if not table_path.is_file():
        raise RuntimeError(f"Declared analytical table does not exist : {table_path.name}")

    if calculate_sha256(
        table_path
    ) != table_manifest[
        "sha256"
    ]:
        raise RuntimeError(f"Analytical table digest does not match its manifest : {table_path.name}")

    with table_path.open(
        mode = "r",
        encoding = "utf-8",
        newline = ""
    ) as table_stream:
        table_reader = csv.DictReader(
            table_stream
        )

        expected_field_names = [
            schema_field[
                "name"
            ]
            for schema_field in table_manifest[
                "schema"
            ]
        ]

        if table_reader.fieldnames != expected_field_names:
            raise RuntimeError(f"Analytical table header does not match its declared schema : {table_path.name}")

        observed_row_count = sum(
            1
            for _ in table_reader
        )

    if observed_row_count != table_manifest[
        "row_count"
    ]:
        raise RuntimeError(
            "Analytical table row count does not match its manifest : "
            f"{table_path.name} : {observed_row_count} : {table_manifest['row_count']}"
        )


# Validating a complete published analytical artifact set
def validate_published_artifact_set(output_directory: Path, expected_report: dict,
                                    modeling_policy: dict
) -> None:
    artifact_contract = modeling_policy[
        "artifact_contract"
    ]

    report_file_name = artifact_contract[
        "report_file_name"
    ]

    table_file_extension = artifact_contract[
        "table_file_extension"
    ]

    expected_table_names = set(
        artifact_contract[
            "output_tables"
        ]
    )

    expected_file_names = {
        *(
            f"{table_name}{table_file_extension}"
            for table_name in expected_table_names
        ),
        report_file_name
    }

    observed_file_names = {
        output_path.name
        for output_path in output_directory.iterdir()
    }

    if observed_file_names != expected_file_names:
        missing_file_names = sorted(
            expected_file_names.difference(
                observed_file_names
            )
        )

        unexpected_file_names = sorted(
            observed_file_names.difference(
                expected_file_names
            )
        )

        raise RuntimeError(
            "Published artifact inventory does not match the declared inventory : "
            f"missing {missing_file_names} : unexpected {unexpected_file_names}"
        )

    report_path = (
        output_directory
        / report_file_name
    )

    published_report = load_strict_json_object(
        report_path
    )

    normalized_expected_report = (
        normalize_artifact_value(
            expected_report
        )
    )

    if published_report != normalized_expected_report:
        raise RuntimeError("Published modeling report does not match the validated staged report.")

    validate_descriptive_modeling_report(
        report = published_report,
        modeling_policy = modeling_policy
    )

    table_manifests = published_report[
        "tables"
    ]

    if set(
        table_manifests
    ) != expected_table_names:
        raise RuntimeError("Published modeling report does not contain the declared table inventory.")

    observed_manifest_file_names = []

    for table_name in sorted(
        expected_table_names
    ):
        table_manifest = table_manifests[
            table_name
        ]

        expected_table_file_name = (
            f"{table_name}{table_file_extension}"
        )

        if table_manifest.get(
            "file_name"
        ) != expected_table_file_name:
            raise RuntimeError(f"Published analytical-table filename does not match its declared identity : {table_name}")

        observed_manifest_file_names.append(
            table_manifest[
                "file_name"
            ]
        )

        validate_written_artifact_table(
            table_path = (
                output_directory
                / table_manifest[
                    "file_name"
                ]
            ),
            table_manifest = table_manifest
        )

    if len(
        observed_manifest_file_names
    ) != len(
        set(
            observed_manifest_file_names
        )
    ):
        raise RuntimeError("Published table manifests contain duplicate filenames.")


# Publishing a complete validated staging directory transactionally
def publish_staged_artifacts(staging_directory: Path, output_directory: Path,
                             expected_file_names: list[str], report_file_name: str,
                             final_validator: Callable[[Path], None] | None = None
) -> None:
    if (
        not expected_file_names
        or len(expected_file_names)
        != len(
            set(
                expected_file_names
            )
        )
    ):
        raise RuntimeError("Artifact publication requires unique expected file names.")

    if report_file_name not in expected_file_names:
        raise RuntimeError("Artifact publication requires the report in the expected file inventory.")

    expected_file_name_set = set(
        expected_file_names
    )

    observed_staging_entries = {
        staging_entry.name
        for staging_entry in staging_directory.iterdir()
    }

    if observed_staging_entries != expected_file_name_set:
        missing_staging_files = sorted(
            expected_file_name_set.difference(
                observed_staging_entries
            )
        )

        unexpected_staging_files = sorted(
            observed_staging_entries.difference(
                expected_file_name_set
            )
        )

        raise RuntimeError(
            "Staging directory does not match the declared artifact inventory : "
            f"missing {missing_staging_files} : unexpected {unexpected_staging_files}"
        )

    non_file_staging_entries = sorted(
        file_name
        for file_name in expected_file_names
        if not (
            staging_directory
            / file_name
        ).is_file()
    )

    if non_file_staging_entries:
        raise RuntimeError(f"Staging inventory contains non-file entries : {non_file_staging_entries}")

    output_directory.parent.mkdir(
        parents = True,
        exist_ok = True
    )

    output_directory.mkdir(
        parents = True,
        exist_ok = True
    )

    if (
        staging_directory.stat().st_dev
        != output_directory.stat().st_dev
    ):
        raise RuntimeError("Staging and destination directories must use the same filesystem.")

    existing_entries = {
        output_entry.name
        for output_entry in output_directory.iterdir()
    }

    unexpected_existing_entries = sorted(
        existing_entries.difference(
            expected_file_name_set
        )
    )

    if unexpected_existing_entries:
        raise RuntimeError(f"Artifact destination contains unexpected entries : {unexpected_existing_entries}")

    if (
        existing_entries
        and existing_entries
        != expected_file_name_set
    ):
        missing_existing_entries = sorted(
            expected_file_name_set.difference(
                existing_entries
            )
        )

        raise RuntimeError(f"Existing artifact destination is incomplete : {missing_existing_entries}")

    backup_directory = Path(
        tempfile.mkdtemp(
            prefix = (
                f".{output_directory.name}.publication-backup-"
            ),
            dir = output_directory.parent
        )
    )

    moved_existing_file_names = []
    published_file_names = []

    table_file_names = sorted(
        expected_file_name_set.difference(
            {
                report_file_name
            }
        )
    )

    publication_order = [
        *table_file_names,
        report_file_name
    ]

    try:
        for file_name in sorted(
            existing_entries
        ):
            os.replace(
                output_directory
                / file_name,
                backup_directory
                / file_name
            )

            moved_existing_file_names.append(
                file_name
            )

        for file_name in publication_order:
            os.replace(
                staging_directory
                / file_name,
                output_directory
                / file_name
            )

            published_file_names.append(
                file_name
            )

        final_entries = {
            output_entry.name
            for output_entry in output_directory.iterdir()
        }

        if final_entries != expected_file_name_set:
            missing_final_files = sorted(
                expected_file_name_set.difference(
                    final_entries
                )
            )

            unexpected_final_files = sorted(
                final_entries.difference(
                    expected_file_name_set
                )
            )

            raise RuntimeError(
                "Published artifact inventory does not match the declared inventory : "
                f"missing {missing_final_files} : unexpected {unexpected_final_files}"
            )

        if final_validator is not None:
            final_validator(
                output_directory
            )

    except Exception as publication_error:
        rollback_errors = []

        for file_name in reversed(
            published_file_names
        ):
            published_path = (
                output_directory
                / file_name
            )

            try:
                if published_path.exists():
                    published_path.unlink()

            except OSError as exc:
                rollback_errors.append(
                    (
                        file_name,
                        str(
                            exc
                        )
                    )
                )

        for file_name in moved_existing_file_names:
            backup_path = (
                backup_directory
                / file_name
            )

            try:
                if backup_path.exists():
                    os.replace(
                        backup_path,
                        output_directory
                        / file_name
                    )

            except OSError as exc:
                rollback_errors.append(
                    (
                        file_name,
                        str(
                            exc
                        )
                    )
                )

        if rollback_errors:
            raise RuntimeError(
                "Artifact publication failed and the preceding artifact set could not be restored completely : "
                f"{rollback_errors}"
            ) from publication_error

        raise

    finally:
        shutil.rmtree(
            backup_directory,
            ignore_errors = True
        )


# Executing the complete in-memory descriptive-modeling workflow
def execute_descriptive_modeling_workflow(validated_inputs: dict) -> dict:
    modeling_policy = validated_inputs[
        "modeling_policy"
    ]

    candidate_records = validated_inputs[
        "candidate_records"
    ]

    candidate_term_records = validated_inputs[
        "candidate_term_records"
    ]

    prepared_models = validated_inputs[
        "prepared_candidate_models"
    ]

    (
        fitted_models_by_candidate,
        coefficient_records,
        fit_diagnostic_records
    ) = fit_candidate_models(
        prepared_models = prepared_models,
        modeling_policy = modeling_policy
    )

    influence_records = (
        calculate_uncertainty_and_influence(
            prepared_models = prepared_models,
            fitted_models_by_candidate = (
                fitted_models_by_candidate
            ),
            coefficient_records = coefficient_records,
            modeling_policy = modeling_policy
        )
    )

    (
        validation_fold_records,
        validation_prediction_records,
        validation_metric_records
    ) = run_grouped_validation(
        prepared_models = prepared_models,
        fitted_models_by_candidate = (
            fitted_models_by_candidate
        ),
        modeling_policy = modeling_policy
    )

    condition_calibration_records = (
        build_condition_calibration(
            prepared_models = prepared_models,
            prediction_records = (
                validation_prediction_records
            )
        )
    )

    model_comparison_records = (
        build_model_comparisons(
            candidate_records = candidate_records,
            validation_prediction_records = (
                validation_prediction_records
            ),
            validation_metric_records = (
                validation_metric_records
            ),
            modeling_policy = modeling_policy
        )
    )

    (
        trajectory_summary_records,
        trajectory_accounting_records
    ) = build_trajectory_summaries(
        modeling_policy = modeling_policy,
        cell_records = validated_inputs[
            "feature_tables"
        ][
            "cell_features"
        ],
        reference_records = validated_inputs[
            "feature_tables"
        ][
            "reference_test_features"
        ],
        support_membership_records = validated_inputs[
            "design_tables"
        ][
            "bracketing_support_membership"
        ]
    )

    assign_candidate_analytical_result_status(
        candidate_records = candidate_records,
        prepared_models = prepared_models,
        validation_metric_records = (
            validation_metric_records
        ),
        validation_prediction_records = (
            validation_prediction_records
        ),
        condition_calibration_records = (
            condition_calibration_records
        ),
        modeling_policy = modeling_policy
    )

    contrast_records = build_model_contrasts(
        prepared_models = prepared_models,
        fitted_models_by_candidate = (
            fitted_models_by_candidate
        ),
        estimand_records = validated_inputs[
            "design_tables"
        ][
            "estimand_support"
        ],
        condition_records = validated_inputs[
            "feature_tables"
        ][
            "condition_features"
        ],
        approved_discrete_horizons = (
            validated_inputs[
                "design_report"
            ][
                "horizon_selection_policy"
            ][
                "approved_discrete_horizons"
            ]
        ),
        discrete_horizon_status = (
            validated_inputs[
                "design_report"
            ][
                "discrete_comparison_horizon_status"
            ]
        ),
        modeling_policy = modeling_policy
    )

    disposition_counts = (
        assign_final_candidate_dispositions(
            candidate_records = candidate_records,
            modeling_policy = modeling_policy
        )
    )

    reconciliation = (
        validate_modeling_reconciliation(
            candidate_records = candidate_records,
            candidate_term_records = (
                candidate_term_records
            ),
            prepared_models = prepared_models,
            fitted_models_by_candidate = (
                fitted_models_by_candidate
            ),
            coefficient_records = coefficient_records,
            matrix_diagnostic_records = (
                validated_inputs[
                    "matrix_diagnostic_records"
                ]
            ),
            fit_diagnostic_records = (
                fit_diagnostic_records
            ),
            validation_metric_records = (
                validation_metric_records
            ),
            influence_records = influence_records,
            modeling_policy = modeling_policy
        )
    )

    model_diagnostic_union_field_names = [
        "support_region_identity",
        "template_name",
        "fit_identity",
        "observation_count",
        "independent_cell_count",
        "condition_protocol_count",
        "matrix_column_count",
        "matrix_rank",
        "rank_tolerance",
        "largest_singular_value",
        "smallest_singular_value",
        "smallest_retained_singular_value",
        "effective_condition_number",
        "residual_independent_cell_degrees_of_freedom",
        "unique_horizon_count",
        "non_boundary_unique_horizon_count",
        "admissibility_status",
        "admissibility_reason_code",
        "estimator",
        "fit_status",
        "fit_reason_code",
        "fit_error",
        "convergence_status",
        "optimizer",
        "optimizer_attempts",
        "random_effect_singularity_status",
        "residual_mean",
        "residual_standard_deviation",
        "maximum_absolute_standardized_residual",
        "marginal_residual_mean",
        "marginal_residual_standard_deviation",
        "conditional_residual_mean",
        "conditional_residual_standard_deviation",
        "observation_variance",
        "random_intercept_variance",
        "residual_variance",
        "weighted_matrix_rank",
        "weighted_rank_tolerance",
        "weighted_condition_number"
    ]

    model_diagnostic_records = []

    for matrix_diagnostic_record in (
        validated_inputs[
            "matrix_diagnostic_records"
        ]
    ):
        model_diagnostic_records.append(
            {
                "record_type": "matrix",
                "candidate_identity": (
                    matrix_diagnostic_record[
                        "candidate_identity"
                    ]
                ),
                **{
                    field_name: (
                        matrix_diagnostic_record.get(
                            field_name
                        )
                    )
                    for field_name in (
                        model_diagnostic_union_field_names
                    )
                }
            }
        )

    for fit_diagnostic_record in (
        fit_diagnostic_records
    ):
        model_diagnostic_records.append(
            {
                "record_type": "fit",
                "candidate_identity": (
                    fit_diagnostic_record[
                        "candidate_identity"
                    ]
                ),
                **{
                    field_name: (
                        fit_diagnostic_record.get(
                            field_name
                        )
                    )
                    for field_name in (
                        model_diagnostic_union_field_names
                    )
                }
            }
        )

    observation_accounting_union_field_names = [
        "candidate_identity",
        "support_region_identity",
        "source_record_count",
        "included_record_count",
        "excluded_record_count",
        "missing_horizon_count",
        "nonpositive_horizon_count",
        "missing_outcome_count",
        "cohort_ineligible_count",
        "incomplete_cumulative_fec_count",
        "inside_evaluation_interval_count",
        "outside_evaluation_interval_count",
        "accounting_scope",
        "aging_mode",
        "outcome",
        "cohort",
        "horizon_axis",
        "member_cell_count",
        "eligible_positive_record_count",
        "outside_support_record_count"
    ]

    observation_accounting_records = []

    for candidate_accounting_record in (
        validated_inputs[
            "observation_accounting_records"
        ]
    ):
        observation_accounting_records.append(
            {
                "record_type": "candidate",
                **{
                    field_name: (
                        candidate_accounting_record.get(
                            field_name
                        )
                    )
                    for field_name in (
                        observation_accounting_union_field_names
                    )
                }
            }
        )

    for trajectory_accounting_record in (
        trajectory_accounting_records
    ):
        observation_accounting_records.append(
            {
                "record_type": "trajectory_summary",
                **{
                    field_name: (
                        trajectory_accounting_record.get(
                            field_name
                        )
                    )
                    for field_name in (
                        observation_accounting_union_field_names
                    )
                }
            }
        )

    return {
        "candidate_registry": sorted(
            candidate_records,
            key = lambda record: record[
                "candidate_identity"
            ]
        ),
        "candidate_terms": sorted(
            candidate_term_records,
            key = lambda record: (
                record[
                    "candidate_identity"
                ],
                record[
                    "term_type"
                ],
                record[
                    "term_name"
                ]
            )
        ),
        "observation_accounting": sorted(
            observation_accounting_records,
            key = lambda record: (
                record[
                    "record_type"
                ],
                record.get(
                    "candidate_identity",
                    ""
                ),
                record.get(
                    "aging_mode",
                    ""
                ),
                record.get(
                    "outcome",
                    ""
                ),
                record.get(
                    "cohort",
                    ""
                ),
                record.get(
                    "horizon_axis",
                    ""
                )
            )
        ),
        "trajectory_summaries": (
            trajectory_summary_records
        ),
        "model_coefficients": (
            coefficient_records
        ),
        "model_diagnostics": sorted(
            model_diagnostic_records,
            key = lambda record: (
                record[
                    "candidate_identity"
                ],
                record[
                    "record_type"
                ]
            )
        ),
        "validation_folds": (
            validation_fold_records
        ),
        "validation_predictions": (
            validation_prediction_records
        ),
        "validation_metrics": (
            validation_metric_records
        ),
        "model_comparisons": (
            model_comparison_records
        ),
        "condition_calibration": (
            condition_calibration_records
        ),
        "influence_diagnostics": (
            influence_records
        ),
        "model_contrasts": contrast_records,
        "candidate_disposition_counts": (
            disposition_counts
        ),
        "reconciliation": reconciliation,
        "fitted_models_by_candidate": (
            fitted_models_by_candidate
        )
    }


# Executing and publishing one complete descriptive-modeling analysis
def run_descriptive_modeling_analysis(validated_inputs: dict, output_directory: Path) -> dict:
    workflow_results = (
        execute_descriptive_modeling_workflow(
            validated_inputs
        )
    )

    artifact_contract = validated_inputs[
        "modeling_policy"
    ][
        "artifact_contract"
    ]

    output_directory.parent.mkdir(
        parents = True,
        exist_ok = True
    )

    staging_directory = Path(
        tempfile.mkdtemp(
            prefix = (
                f".{output_directory.name}.staging-"
            ),
            dir = output_directory.parent
        )
    )

    try:
        report = stage_descriptive_modeling_artifacts(
            validated_inputs = validated_inputs,
            workflow_results = workflow_results,
            staging_directory = staging_directory
        )

        expected_file_names = [
            *(
                f"{table_name}{artifact_contract['table_file_extension']}"
                for table_name in artifact_contract[
                    "output_tables"
                ]
            ),
            artifact_contract[
                "report_file_name"
            ]
        ]

        def validate_final_publication(published_directory: Path) -> None:
            validate_published_artifact_set(
                output_directory = published_directory,
                expected_report = report,
                modeling_policy = validated_inputs[
                    "modeling_policy"
                ]
            )

        publish_staged_artifacts(
            staging_directory = staging_directory,
            output_directory = output_directory,
            expected_file_names = expected_file_names,
            report_file_name = artifact_contract[
                "report_file_name"
            ],
            final_validator = (
                validate_final_publication
            )
        )

    finally:
        shutil.rmtree(
            staging_directory,
            ignore_errors = True
        )

    return report


# Loading and reconciling all required inputs
def load_validated_inputs(feature_manifest_path: Path, design_report_path: Path,
                          modeling_policy_path: Path
) -> dict:
    modeling_policy = load_strict_json_object(
        modeling_policy_path
    )

    validate_modeling_policy(
        modeling_policy
    )

    (
        feature_manifest,
        feature_tables,
        feature_table_paths
    ) = load_feature_inputs(
        feature_manifest_path
    )

    (
        design_report,
        design_tables,
        design_table_paths
    ) = load_design_inputs(
        design_report_path
    )

    validate_input_bindings(
        feature_manifest_path = feature_manifest_path,
        feature_manifest = feature_manifest,
        design_report = design_report
    )

    analytical_scope_records = design_tables[
        "analytical_scope"
    ]

    validate_analytical_scope(
        analytical_scope_records
    )

    (
        candidate_records,
        candidate_term_records
    ) = build_candidate_registry(
        scope_records = analytical_scope_records,
        modeling_policy = modeling_policy
    )

    select_model_candidates(
        candidate_records = candidate_records,
        candidate_term_records = (
            candidate_term_records
        ),
        modeling_policy = modeling_policy
    )

    (
        candidate_observation_records,
        observation_accounting_records
    ) = assemble_candidate_observations(
        candidate_records = candidate_records,
        modeling_policy = modeling_policy,
        cell_records = feature_tables[
            "cell_features"
        ],
        reference_records = feature_tables[
            "reference_test_features"
        ],
        support_membership_records = design_tables[
            "bracketing_support_membership"
        ]
    )

    (
        prepared_candidate_models,
        matrix_diagnostic_records
    ) = prepare_candidate_model_matrices(
        candidate_records = candidate_records,
        candidate_term_records = (
            candidate_term_records
        ),
        candidate_observation_records = (
            candidate_observation_records
        ),
        modeling_policy = modeling_policy
    )

    return {
        "modeling_policy": modeling_policy,
        "modeling_policy_sha256": calculate_sha256(
            modeling_policy_path
        ),
        "feature_manifest": feature_manifest,
        "feature_manifest_sha256": calculate_sha256(
            feature_manifest_path
        ),
        "design_report": design_report,
        "design_report_sha256": calculate_sha256(
            design_report_path
        ),
        "feature_tables": feature_tables,
        "feature_table_paths": feature_table_paths,
        "design_tables": design_tables,
        "design_table_paths": design_table_paths,
        "candidate_records": candidate_records,
        "candidate_term_records": (
            candidate_term_records
        ),
        "candidate_observation_records": (
            candidate_observation_records
        ),
        "observation_accounting_records": (
            observation_accounting_records
        ),
        "prepared_candidate_models": (
            prepared_candidate_models
        ),
        "matrix_diagnostic_records": (
            matrix_diagnostic_records
        )
    }


# Executing only the explicitly requested analytical operation
def execute_requested_operation(validated_inputs: dict, operation: str, output_directory: Path) -> dict | None:
    if operation == "validate":
        return None

    if operation == "fit":
        return run_descriptive_modeling_analysis(
            validated_inputs = validated_inputs,
            output_directory = output_directory
        )

    raise ValueError(f"Unsupported descriptive-modeling operation : {operation}")


# Validating inputs or explicitly executing the complete analysis
def main() -> None:
    args = parse_args()

    validated_inputs = load_validated_inputs(
        feature_manifest_path = (
            args.feature_manifest_path
        ),
        design_report_path = (
            args.design_report_path
        ),
        modeling_policy_path = (
            args.modeling_policy_path
        )
    )

    candidate_records = validated_inputs[
        "candidate_records"
    ]

    print(
        "Descriptive-modeling inputs validated."
    )

    print(
        f"Candidate count : {len(candidate_records)}"
    )

    print(
        f"Candidate-term count : {len(validated_inputs['candidate_term_records'])}"
    )

    selected_candidate_count = sum(
        candidate_record[
            "selection_status"
        ]
        == "selected"
        for candidate_record in candidate_records
    )

    print(
        f"Selected candidate count : {selected_candidate_count}"
    )

    print(
        f"Assembled candidate-observation count : {len(validated_inputs['candidate_observation_records'])}"
    )

    admissible_candidate_count = sum(
        candidate_record[
            "admissibility_status"
        ]
        == "admissible"
        for candidate_record in candidate_records
    )

    print(
        f"Admissible candidate count : {admissible_candidate_count}"
    )

    print(
        f"Prepared model-matrix count : {len(validated_inputs['prepared_candidate_models'])}"
    )

    execution_plan = build_modeling_execution_plan(
        prepared_models = validated_inputs[
            "prepared_candidate_models"
        ],
        modeling_policy = validated_inputs[
            "modeling_policy"
        ]
    )

    print(
        f"Unique full-fit count : {execution_plan['unique_full_fit_count']}"
    )

    print(
        f"Full-data deletion-refit upper bound : {execution_plan['full_data_deletion_refit_upper_bound']}"
    )

    print(
        f"Validation-fold fit upper bound : {execution_plan['validation_fold_fit_upper_bound']}"
    )

    print(
        f"Validation-interval deletion-refit upper bound : {execution_plan['validation_interval_deletion_refit_upper_bound']}"
    )

    print(
        f"Total estimator-call upper bound : {execution_plan['total_fit_call_upper_bound']}"
    )

    report = execute_requested_operation(
        validated_inputs = validated_inputs,
        operation = args.operation,
        output_directory = args.output_directory
    )

    if report is None:
        print(
            "Model fitting was not requested."
        )

        return

    reconciliation = report[
        "reconciliation"
    ]

    print(
        "Descriptive modeling completed."
    )

    print(
        f"Fitted candidate count : {reconciliation['fitted_candidate_count']}"
    )

    print(
        f"Failed-fit candidate count : {reconciliation['failed_fit_candidate_count']}"
    )

    print(
        f"Inadmissible candidate count : {reconciliation['inadmissible_candidate_count']}"
    )

    print(
        f"Not-assembled candidate count : {reconciliation['not_assembled_candidate_count']}"
    )

    print(
        f"Not-selected candidate count : {reconciliation['not_selected_candidate_count']}"
    )

    report_path = (
        args.output_directory
        / validated_inputs[
            "modeling_policy"
        ][
            "artifact_contract"
        ][
            "report_file_name"
        ]
    )

    print(
        f"Descriptive-modeling report : {report_path}"
    )


if __name__ == "__main__":
    main()
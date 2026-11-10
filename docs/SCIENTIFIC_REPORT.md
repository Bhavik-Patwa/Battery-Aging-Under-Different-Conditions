# Evidence-Constrained Analysis of Multi-Stage Lithium-Ion Battery Aging

## Abstract

This project provides a reproducible analysis of the Multi-Stage Lithium Ion Battery Aging Study for 279 Samsung INR21700-50E cells. The implementation integrates verified cloud acquisition, archive- and measurement-level quality assessment, longitudinal feature extraction, experimental-support analysis, causal identification assessment, descriptive modeling, hierarchical Bayesian modeling, predictive benchmarking, sensitivity analysis, robustness synthesis and read-only evidence reporting. The analysis treats the cell as the independent experimental unit and evaluates relative capacity and relative uncorrected DC internal resistance over observed longitudinal support. The completed predictive inventory contains 393 family-specific benchmarks across elastic-net, random-forest and gradient-boosting models; 273 satisfy the registered analytical requirements, with median equal-cell-weighted predictive R² ranging from 0.867 for elastic net to 0.928 for random forest among available benchmarks. A further 4,353 sensitivity variants evaluate grouped partitions, bounded configuration search, training windows, documented incidents and predictor ablations. The project demonstrates that useful within-condition degradation prediction is possible while also showing that predictive accuracy does not resolve limitations in treatment assignment, confounding control, experimental overlap or transportability. Results are therefore reported as observational, associational or predictive evidence within represented conditions, not as causal treatment effects or operating-policy recommendations.

## 1. Research scope

The source dataset contains calendar-aging and cycle-aging experiments conducted on Samsung INR21700-50E cells. The analytical objectives are to :

- construct reproducible longitudinal capacity, resistance, elapsed-time and cycling-exposure records;
- identify condition comparisons supported by the experimental inventory;
- characterize observed degradation trajectories;
- quantify associational uncertainty where hierarchical modeling is supportable;
- evaluate held-out predictive performance using cell-grouped validation; and
- determine which findings remain stable under registered analytical perturbations.

The intended domain is the represented cell type, equipment, laboratories, operating conditions and observation ranges. The analysis does not directly describe battery packs, other chemistries, other cell formats or unsupported operating regimes.

## 2. Data source and experimental unit

The analysis uses version 1 of the Multi-Stage Lithium Ion Battery Aging Study :

- 279 physical cells;
- 132 calendar-aging cells;
- 147 cycle-aging cells;
- 71 operating test points;
- 93 condition and testing-protocol combinations;
- experiment metadata; and
- 279 cell archives.

The difference between the 71 operating test points and 93 analytical combinations arises because calendar-aging groups with different reference-testing schedules represent distinct condition and protocol combinations.

The cell is the experimental unit. Repeated reference tests and cycling records are longitudinal observations nested within a cell.

## 3. Data integrity and quality assessment

### 3.1 Source integrity

The complete source inventory contains 280 published files. All objects matched their published sizes. Published MD5 checksums were available and verified for 279 objects.

The source record does not provide an MD5 checksum for `TP_z05_01.zip`. That archive is therefore identified as size-verified with a destination SHA-256 baseline, rather than publisher-checksum verified.

Archive processing is incremental because a complete local extraction would require approximately 100 GB. Each archive is validated and processed without retaining a full extracted copy.

### 3.2 Quality inventory

The completed quality assessment covers all 279 archives and 5,763 measurements. Significant observations include :

- 2,462 measurements with nonfinite ambient-temperature values;
- 1,149 companion-metadata identity mismatches;
- 205 measurements outside explicit physical guardrails;
- 24 empty measurements;
- 24 numeric runtime representations;
- 24 step/current-direction mismatches;
- 13 invalid companion dates; and
- 2 timestamp reversals.

Quality status is represented explicitly. Raw records are not silently repaired, deleted or reclassified. Every exclusion, warning and manual-review record retains its reason and source identity.

Ambient temperature sampled inside many measurement files is unavailable. Assigned experimental temperature is obtained from experiment metadata and is kept distinct from sampled temperature.

## 4. Feature definitions

### 4.1 Relative capacity

Charge capacity is obtained by integrating positive current in the registered charge segment. Discharge capacity is obtained by integrating negative current in the registered discharge segment. Their mean defines the primary capacity estimate.

Relative capacity is normalized to the corresponding cell’s initial reference test.

### 4.2 Cycling exposure

Primary fully equivalent cycles are calculated as :

\[
\mathrm{FEC} =
\frac{\int |I(t)|\,dt}{2C_\mathrm{ref}},
\]

where \(C_\mathrm{ref}=4.9\) Ah.

Absolute current is used because observed records include current-direction conflicts with nominal step labels. A signed-step calculation is retained as a sensitivity quantity.

### 4.3 DC internal resistance

Primary DC internal resistance is calculated from the observed voltage change during the approximately ten-second discharge pulse :

\[
R_\mathrm{DC} = \frac{\Delta V}{|I_\mathrm{pulse}|}.
\]

The primary outcome is uncorrected resistance because it depends directly on observed pulse behavior.

OCV-corrected resistance remains provisional. Its available OCV curves are normalized by sample position rather than integrated capacity, which does not provide an approved capacity-aligned correction.

### 4.4 Longitudinal axes

Calendar-aging models use gross elapsed calendar time between reference-test start dates. This quantity does not measure net storage exposure.

Cycle-aging models use either fully equivalent cycles or elapsed time. These represent different scientific quantities and are not combined.

## 5. Extracted analytical inventory

Feature extraction produced :

| Analytical table | Rows |
|---|---:|
| Cell features | 279 |
| Condition features | 93 |
| Cycling-exposure features | 1,893 |
| Reference-test features | 2,766 |

The longitudinal inventory includes :

- 1,965 primary relative-capacity observations;
- 1,962 primary relative-resistance observations; and
- 1,390 primary cycling-exposure measurements.

Gross elapsed calendar time is available for 2,763 reference-test measurements.

## 6. Experimental support and identification

### 6.1 Support assessment

The support analysis evaluates 13,032 candidate condition comparisons :

| Support classification | Count |
|---|---:|
| Supported with restrictions | 1,004 |
| Sensitivity-only | 230 |
| Descriptive-only | 3,984 |
| Unsupported | 7,814 |

Among supported or sensitivity comparisons, 210 differ in one assigned treatment and 1,024 differ in joint condition combinations.

### 6.2 Structural limitations

The following restrictions materially affect interpretation :

- treatment assignment is not documented as randomized;
- sufficient confounding control cannot be established;
- laboratory allocation is tied to test point;
- experimental stage and batch are aliased;
- exact treatment combinations do not overlap across experimental stages;
- reference-testing schedules form part of the experimental protocol;
- calendar-aging and cycle-aging populations differ in treatment structure; and
- common-support ranges vary by outcome, condition and observation axis.

These properties prevent identification of isolated causal treatment effects.

Associational hierarchical modeling is supportable for a restricted subset of the inventory. Causal modeling and treatment optimization are not supported.

## 7. Descriptive trajectory modeling

The descriptive inventory contains 21,168 registered candidates :

| Final disposition | Count |
|---|---:|
| Fitted | 724 |
| Structurally inadmissible | 386 |
| No usable fitted result | 30 |
| Outside selected inventory | 20,028 |

The fitted inventory includes :

- 502 weighted least-squares models;
- 222 linear mixed-effects models;
- 362 analytically available fitted candidates; and
- 362 diagnostic-only fitted candidates.

Equal-cell weighting prevents cells with more repeated observations from dominating the fit. Validation folds are grouped by cell. Influence and uncertainty calculations use cell-level deletion where available.

No discrete point contrast is reported as completed. The approved estimand is a continuous trajectory over observed common-support ranges.

## 8. Hierarchical Bayesian modeling

The Bayesian inventory contains 198 candidate models :

| Analytical status | Count |
|---|---:|
| Available | 7 |
| Diagnostic-only | 169 |
| Unavailable | 22 |

The models estimate associational longitudinal trajectories with hierarchical cell structure and registered prior-scale sensitivity.

Analytical restrictions arise from :

- incomplete condition-level calibration;
- incomplete grouped validation;
- predictive benchmark restrictions; and
- reference-sampler diagnostic restrictions.

Prior-predictive checks, posterior-predictive checks, grouped validation, sampler diagnostics and prior sensitivity are retained for every applicable candidate.

The posterior artifact inventory comprises 528 posterior NetCDF artifacts and 1,449 cache entries covering prior-predictive, full-posterior and grouped-validation operations. These models quantify associational uncertainty; they do not identify causal treatment effects.

## 9. Predictive benchmarking

### 9.1 Model families

The predictive benchmark evaluates :

- elastic net;
- random forest; and
- gradient boosting.

The inventory contains 131 selected analytical populations and 393 family-specific benchmarks.

### 9.2 Validation design

Validation is nested and grouped by cell. No cell appears in both training and test populations within a fold. Hyperparameter selection occurs using training data only.

Metrics use equal-cell weighting and include :

- root mean squared error;
- mean absolute error;
- predictive \(R^2\);
- calibration summaries; and
- empirical prediction-interval coverage and width.

A same-condition training-mean predictor provides the benchmark reference.

### 9.3 Results

Of 393 benchmarks :

- 273 are analytically available;
- 120 are diagnostic-only;
- 114 have incomplete condition-level calibration; and
- 6 do not exceed the registered predictive benchmark.

Among available benchmarks :

| Model family | Count | Median weighted RMSE | Median predictive R² |
|---|---:|---:|---:|
| Elastic net | 91 | 0.00631 | 0.867 |
| Gradient boosting | 91 | 0.00515 | 0.919 |
| Random forest | 91 | 0.00462 | 0.928 |

Observed held-out predictions range from approximately 0.839 to 1.330 on the relative-outcome scale. Published point predictions and lower interval bounds are positive and finite.

The model-family summaries do not define a universal ranking. Differences depend on the outcome, population, observation axis, condition support and held-out inventory.

Native feature-importance values are retained for model inspection only. They do not estimate causal effects or held-out predictor contributions.

## 10. Predictive sensitivity

The sensitivity inventory contains 4,353 one-component perturbations :

| Sensitivity type | Count |
|---|---:|
| Alternative grouped partitions | 786 |
| Expanded configuration search | 393 |
| Training-window restrictions | 786 |
| Documented incident exclusions | 1,209 |
| Predictor ablations | 1,179 |

Validation completed for 3,325 variants and was unavailable for 1,028.

Analytical classification was :

| Status | Count |
|---|---:|
| Available | 2,094 |
| Diagnostic-only | 1,231 |
| Unavailable | 1,028 |

Comparisons with their corresponding primary benchmark produced :

- 2,087 completed comparisons;
- 236 diagnostic-only comparisons; and
- 2,030 unavailable comparisons.

Expanded configuration search is bounded to the registered search inventory and does not establish global hyperparameter optimality. Predictor ablations measure held-out predictive dependence, not causality. Incident exclusions measure sensitivity to provenance-based cohort definitions and do not identify incident effects.

## 11. Robustness synthesis

The robustness assessment integrates descriptive, Bayesian, predictive and predictive-sensitivity evidence.

Its scientific status is a **restricted evidence summary** :

| Domain status | Count |
|---|---:|
| Completed with restrictions | 14 |
| Not authorized | 3 |
| Unavailable | 2 |
| Unsupported | 2 |

The assessment covers :

- grouped validation;
- condition calibration;
- model-family agreement;
- descriptive specification sensitivity;
- hierarchical prior sensitivity;
- training-window sensitivity;
- incident-exclusion sensitivity;
- predictor ablation;
- prediction-interval behavior;
- influence diagnostics;
- external validity;
- unsupported conditions; and
- fragile findings.

Sixty-three Bayesian candidate identities contain held-out predictions without a corresponding grouped-validation metric record. Their 744 predictions are excluded from robustness computations. This exclusion is explicit in the reconciliation inventory.

Exact shared-inventory cross-framework comparisons show strong agreement for a limited subset, but partial overlap and diagnostic-only source models prevent broad equivalence claims.

## 12. Interpretation

The analysis supports three conclusions.

First, the source data can be transformed into reproducible longitudinal capacity, resistance, elapsed-time and cycling-exposure inventories while retaining measurement and provenance limitations.

Second, predictive models can provide useful held-out accuracy for many supported analytical populations. Random forest and gradient boosting have lower median prediction error than elastic net across the available benchmark inventory, but no model family is uniformly superior across every population.

Third, predictive performance does not resolve experimental-identification limitations. Treatment assignment, laboratory allocation, stage and batch structure, incomplete overlap, protocol differences and calibration availability remain decisive constraints.

Accordingly, the evidence supports :

- characterization of observed degradation trajectories;
- associational uncertainty summaries;
- predictive benchmarking within represented conditions;
- identification of fragile or unsupported analytical regions; and
- guidance for future experimental design.

It does not support :

- isolated causal treatment effects;
- universal charging or storage recommendations;
- treatment optimization;
- extrapolation to unobserved operating conditions; or
- direct pack-level deployment.

## 13. External validity

Results are limited to :

- Samsung INR21700-50E cells;
- represented laboratories and equipment;
- represented storage and cycling conditions;
- observed reference-testing schedules;
- available experimental populations; and
- observed longitudinal support.

The analysis does not establish behavior for other chemistries, form factors, pack configurations, thermal systems, control systems, fast-charging regimes or field-use environments.

## 14. Reproducibility

The implementation provides :

- pinned source identity;
- recoverable cloud acquisition;
- source-size and checksum validation;
- explicit quality dispositions;
- deterministic feature definitions;
- machine-readable analytical contracts;
- input and output SHA-256 bindings;
- schema and row-count validation;
- cell-grouped validation;
- deterministic output ordering;
- transactional artifact publication;
- byte-identical reproduction checks;
- immutable release-image references;
- source and evidence manifests; and
- automated application and release verification.

The read-only evidence application validates eight reports and fifteen displayed tables before rendering. It refuses startup when required evidence is missing or inconsistent.

The verified repository state contains 836 passing tests. The self-contained test selection contains 822 passing tests, with 14 additional tests requiring generated production evidence.

## 15. Software release

The evidence dashboard is distributed as an immutable container :

```text
us-east1-docker.pkg.dev/battery-aging-policy/evidence-dashboard/battery-aging-evidence-dashboard@sha256 :49bb232f355678b2a27eb6e7f1eb0fc03d92c5831a55d1bdee03f5e8953a74d1
```

It is bound to source commit :

```text
d9fc2e25bf58a15fd354f31c2f9047045c454356
```

The image runs as a non-root user and requires analytical evidence to be mounted read-only. It contains no model-fitting workflow and does not embed analytical artifacts.

The preserved vulnerability assessment records zero critical findings and zero high-severity findings with an available package fix. Residual findings without an available upstream fix remain recorded for deployment-specific review.

## 16. Conclusion

The project delivers a complete, reproducible evidence system for analyzing battery-aging trajectories without exceeding the information supported by the experimental record.

The data support useful descriptive and predictive analysis over represented conditions. They do not support isolated causal treatment claims or an optimized operating policy. The appropriate final use of the system is therefore evidence exploration, predictive evaluation, limitation reporting and experimental-design guidance.

## References

1. *Multi-Stage Lithium Ion Battery Aging Study*. Figshare. <https://doi.org/10.6084/m9.figshare.25975315.v1>
2. Associated Scientific Data publication. <https://doi.org/10.1038/s41597-024-03859-z>
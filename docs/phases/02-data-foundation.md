# Phase 2: Reproducible Data Foundation

**Status:** Approved<br>
**Project:** AeroReliability Predictive Maintenance Platform<br>
**Dataset:** NASA C-MAPSS, initial subset FD001<br>
**Completion date:** 2026-09-05

## 1. Objective

Establish a reproducible and trustworthy data foundation before feature engineering or model training.

The phase answers four production questions:

1. Can the exact source data be acquired again?
2. Can corrupt, unexpected, or unsafe input be detected before use?
3. Can every trusted dataset be traced to its source and schema?
4. Can a failed data-quality check stop downstream publication?

## 2. Delivered outcome

The implemented pipeline:

1. Downloads the official NASA C-MAPSS archive.
2. Verifies the archive against its expected SHA-256 checksum.
3. Safely inspects and extracts the outer and nested ZIP archives.
4. Records source provenance and artifact hashes.
5. Parses the FD001 train, test, and RUL files into typed tables.
6. Applies structural and record-level data-quality checks.
7. Blocks and quarantines unacceptable publications.
8. Atomically publishes accepted datasets as compressed Parquet files.
9. Verifies and reuses existing artifacts on subsequent runs.

The implemented flow is documented in [the current data-flow diagram](../diagrams/implemented-data-flow.md).

## 3. Source contract

| Property | Contract |
| --- | --- |
| Dataset ID | `nasa-cmapss-classic` |
| Dataset version | `v1` |
| Initial subset | `FD001` |
| Source | Official NASA-hosted C-MAPSS archive |
| Archive integrity | Fixed SHA-256 checksum |
| Telemetry schema | `telemetry-v1` |
| Test-RUL schema | `test-rul-v1` |
| Expected FD001 train engines | 100 |
| Expected FD001 test engines | 100 |

The source archive and extracted raw files are treated as immutable. Corrections and transformations must create a new downstream version rather than silently changing source history.

## 4. Canonical telemetry schema

Each train or test record contains 29 canonical columns:

- Three lineage columns: `dataset_version`, `subset_id`, and `split`.
- Two identity columns: `engine_id` and `cycle`.
- Three operating-condition columns: `operating_setting_1` through `operating_setting_3`.
- Twenty-one measurement columns: `sensor_1` through `sensor_21`.

Identifiers use integer types. Operating settings and sensor measurements use floating-point types. Explicit missing sensor values remain null so that a later, versioned preprocessing pipeline can decide how to handle them.

The separate test-RUL table contains:

- `engine_id`
- `observed_final_cycle`
- `additional_rul`
- `true_failure_cycle`

`additional_rul` and `true_failure_cycle` are evaluation labels derived from future information. They must not enter model features or production inference inputs.

## 5. Data-quality policy

The publication gate distinguishes errors from warnings.

### Blocking errors

- Wrong telemetry column count.
- Invalid, non-positive, or non-integer engine and cycle identifiers.
- Missing required operating settings.
- Non-numeric or non-finite numeric values.
- Duplicate engine-cycle pairs.
- Unexpected FD001 engine identities or engine counts.
- Non-contiguous cycles for an engine.
- Invalid RUL labels or inconsistent test/RUL relationships.
- Empty files or files with no valid records.
- Source or published-artifact checksum mismatches.

Any error makes validation status `FAILED`, publishes diagnostic information to quarantine, and prevents creation of a trusted interim dataset.

### Warnings requiring an explicit decision

A missing sensor measurement is preserved as null and recorded as a data-quality warning for investigation. Warnings do not silently pass: publication is blocked unless the caller explicitly enables `allow_warnings=True`.

This implements the agreed policy of retaining observable missingness while ensuring that questionable data cannot be mistaken for healthy, complete input.

### Rejected records

Malformed records are excluded from canonical tables and recorded with:

- Source filename
- Original line number
- Original raw record
- Rejection reason

When rejected records exist, they are written as `rejected-records.jsonl` under the versioned quarantine directory.

## 6. Integrity and safety controls

### Download controls

- TLS certificate verification uses the `certifi` trust store.
- Downloads are written to a temporary `.part` file.
- The expected SHA-256 is checked before publication.
- A failed download removes the temporary file.
- A verified existing archive is reused unless a forced download is requested.

### Extraction controls

- Outer and nested ZIP CRC checks.
- Rejection of absolute paths, parent traversal, and symbolic links.
- Limits for archive members, individual size, total expanded size, and compression ratio.
- An exact allowlist of expected C-MAPSS members.
- Extraction into a temporary directory followed by an atomic rename.

### Publication controls

- Parquet artifacts use Zstandard compression and include statistics.
- Publication is staged in a temporary directory.
- The complete FD001 directory appears only through an atomic rename.
- Manifests store file paths, byte sizes, and SHA-256 hashes.
- Existing raw and interim datasets are verified before reuse.

## 7. Published artifacts

### Immutable raw layer

Location: `data/raw/cmapss/v1/`

- `extracted/`: all expected source files from the C-MAPSS inner archive.
- `source-manifest.json`: source URL, citation, observation time, source and inner-archive hashes, plus hashes and CRC values for extracted files.

### Trusted interim FD001 layer

Location: `data/interim/cmapss/v1/fd001/`

| Artifact | Purpose | Current result |
| --- | --- | --- |
| `train.parquet` | FD001 run-to-failure training telemetry | 20,631 rows, 100 engines |
| `test.parquet` | FD001 truncated test telemetry | 13,096 rows, 100 engines |
| `test-rul.parquet` | Held-out test outcomes and derived failure cycle | 100 rows |
| `validation-report.json` | Quality status, issues, and structural statistics | `PASSED`, no issues |
| `dataset-manifest.json` | Dataset identity, schema versions, lineage, and artifact hashes | Published |

### Quarantine layer

Location: `data/quarantine/cmapss/v1/fd001/`

This directory is populated only when validation prevents publication. It contains the failed validation report and rejected records when applicable.

## 8. Implementation map

| Module | Responsibility |
| --- | --- |
| `data/download.py` | Secure download, checksum calculation, verification, and atomic archive publication |
| `data/contracts.py` | Dataset identity, schema versions, canonical columns, types, and expected source members |
| `data/manifest.py` | Immutable typed source, dataset, validation, statistics, issue, and rejection models |
| `data/extract.py` | Safe nested-archive inspection, extraction, provenance, and idempotent raw reuse |
| `data/validate.py` | Strict parsing, rejected-record capture, structural validation, and quality reporting |
| `data/prepare.py` | End-to-end FD001 orchestration, publication gate, quarantine, and atomic interim publication |

## 9. Operating commands

Download and verify the source archive:

```bash
uv run python -m predictive_maintenance.data.download
```

Prepare or verify the trusted FD001 publication:

```bash
uv run python -m predictive_maintenance.data.prepare
```

Run the automated acceptance checks:

```bash
uv run pytest -q
uv run ruff check src tests
uv run mypy src
```

Expected preparation result:

```text
Published FD001 dataset: data/interim/cmapss/v1/fd001
Validation status: PASSED
Manifest digest: <sha256>
```

## 10. Verification evidence

At phase completion:

- `pytest`: 13 tests passed.
- `ruff`: all checks passed.
- Strict `mypy`: no issues found in 14 source files.
- Real FD001 validation: `PASSED` with no reported issues.
- Re-running preparation verifies and reuses the existing publication.

Tests cover checksum behavior, archive reuse, safe extraction, parsing and validation failures, warning policy, quarantine behavior, atomic publication, artifact contents, and idempotent reuse.

## 11. Phase boundaries and non-goals

This phase intentionally does not implement:

- RUL target construction for training rows.
- Train, validation, and test split policy for model development.
- Imputation, scaling, rolling windows, or other feature transformations.
- Baseline or candidate model training.
- Experiment tracking or model registration.
- Batch or online inference.
- APIs, event streaming, monitoring, containers, or cloud resources.

Those concerns require their own contracts and leakage controls. They must consume the trusted Phase 2 artifacts rather than re-reading unvalidated source files.

## 12. Completion decision

Phase 2 is accepted because source acquisition is reproducible, unsafe or corrupt input is rejected, data-quality outcomes are explicit, trusted publication is atomic and traceable, and the behavior is protected by automated tests.

The next phase is **Phase 3: Production Architecture Design**. It will define component responsibilities, offline and online data flows, storage ownership, deployment boundaries, and reliability behavior before more model or infrastructure code is added.

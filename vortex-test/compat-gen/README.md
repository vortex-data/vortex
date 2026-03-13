# vortex-compat: Backward-Compatibility Testing

This crate provides two binaries that together ensure Vortex can always read files
written by older versions:

- **`compat-gen`** — generates deterministic fixture files for a given Vortex version.
- **`compat-validate`** — reads fixtures from every historical version and validates
  they round-trip to the expected arrays.

Fixtures are stored in an S3 bucket. CI uploads new fixtures on every release tag
and runs weekly validation against all prior versions.

## Fixture Contract

Fixtures are the unit of backward-compatibility. Each fixture is a named file
(e.g. `primitives.vortex`) whose contents are defined by a deterministic `build()`
method. The following rules apply:

- **Immutable data.** Once a fixture's `build()` is defined, its output (columns,
  values, nulls, ordering) must never change. Every version that includes that
  fixture must produce byte-for-byte identical logical arrays. `compat-validate`
  validates this by rebuilding expected arrays from `build()` and comparing them
  against what was read from the stored file.

- **New capabilities get new files.** To test a new encoding, data type, or
  structural pattern, add a new fixture with a new filename. Never modify an
  existing fixture to cover new ground.

- **Additive-only fixture list.** The fixture list only ever grows; fixtures are
  never removed. The CLI (`scripts/compat.py`) enforces this by checking
  that every fixture in the previous version's manifest still exists in the
  generated output. Each fixture's `since` field in the manifest records the first
  version that introduced it.

- **`versions.json`** is the top-level index listing every version that has
  uploaded fixtures. `compat-validate` iterates over all listed versions.

- **Watch for dependency drift.** `compat-validate` compares stored files against
  `build()` output from the *current* code. If a dependency (e.g. `tpchgen`)
  silently changes its output across versions, old fixtures will fail validation
  even though the Vortex reader is fine. If you see unexpected failures across
  all old versions for a specific fixture, check whether its `build()` deps
  changed before blaming the reader.

## CLI Tool (`scripts/compat.py`)

All operations go through the unified `compat.py` CLI, which supports both local
directories and S3 as targets.

```bash
python3 vortex-test/compat-gen/scripts/compat.py --help
```

### `add-version` — generate and store fixtures

```bash
# Upload to S3
python3 vortex-test/compat-gen/scripts/compat.py add-version \
  --version 0.63.0 --target s3

# Dry run (generate + merge manifest, skip S3 upload)
python3 vortex-test/compat-gen/scripts/compat.py add-version \
  --version 0.63.0 --target s3 --dry-run

# Store locally (no S3 needed)
python3 vortex-test/compat-gen/scripts/compat.py add-version \
  --version 0.63.0 --target local:/tmp/compat

# Skip the cargo build (if you already have fixtures generated)
python3 vortex-test/compat-gen/scripts/compat.py add-version \
  --version 0.63.0 --target s3 --output /tmp/fixtures/ --skip-build
```

### `check` — validate fixtures against the current reader

```bash
# Validate all versions from S3
python3 vortex-test/compat-gen/scripts/compat.py check --target s3

# Validate all versions from a local directory
python3 vortex-test/compat-gen/scripts/compat.py check --target local:/tmp/compat

# Validate specific versions only
python3 vortex-test/compat-gen/scripts/compat.py check \
  --target s3 --versions 0.62.0,0.63.0
```

### Target types

- **`s3`** — use the shared S3 bucket (`vortex-compat-fixtures`). Requires AWS
  credentials for uploads; reads are public.
- **`local:<path>`** — use a local directory. The tool creates versioned
  subdirectories (`v0.62.0/`, `v0.63.0/`, ...) and a `versions.json` index
  automatically. No AWS credentials needed.

## First-Time Setup: Bootstrap the Bucket

After creating the S3 bucket (see [AWS Setup](#aws-setup-one-time) below), seed it
with the first fixture set:

```bash
# Generate + upload (first version, no previous manifest to merge)
python3 vortex-test/compat-gen/scripts/compat.py add-version \
  --version 0.62.0 --target s3

# Verify the round-trip
python3 vortex-test/compat-gen/scripts/compat.py check --target s3
```

## Re-uploading Fixtures for an Existing Version

The CLI will overwrite the existing prefix in S3:

```bash
python3 vortex-test/compat-gen/scripts/compat.py add-version \
  --version 0.62.0 --target s3
```

No need to update `versions.json` — the CLI handles it idempotently.

## Local-Only Workflow

You can skip S3 entirely and work against local directories:

```bash
# Generate fixtures locally
python3 vortex-test/compat-gen/scripts/compat.py add-version \
  --version 0.62.0 --target local:/tmp/compat

# Add another version
python3 vortex-test/compat-gen/scripts/compat.py add-version \
  --version 0.63.0 --target local:/tmp/compat

# Validate all local versions
python3 vortex-test/compat-gen/scripts/compat.py check \
  --target local:/tmp/compat
```

If the bucket requires authenticated access, set your AWS profile:

```bash
AWS_PROFILE=vortex-ci python3 vortex-test/compat-gen/scripts/compat.py check \
  --target s3
```

## AWS Setup (one-time)

All resources live in the **benchmark account (245040174862)**, region **us-east-1**.

### 1. Create the S3 bucket

```bash
aws s3api create-bucket \
  --bucket vortex-compat-fixtures \
  --region us-east-1
```

### 2. Enable public read access

Disable the "Block Public Access" settings that prevent a public bucket policy:

```bash
aws s3api put-public-access-block \
  --bucket vortex-compat-fixtures \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=false,RestrictPublicBuckets=false
```

Then attach a bucket policy that grants unauthenticated read:

```bash
aws s3api put-bucket-policy \
  --bucket vortex-compat-fixtures \
  --policy '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Sid": "PublicRead",
        "Effect": "Allow",
        "Principal": "*",
        "Action": ["s3:GetObject", "s3:ListBucket"],
        "Resource": [
          "arn:aws:s3:::vortex-compat-fixtures",
          "arn:aws:s3:::vortex-compat-fixtures/*"
        ]
      }
    ]
  }'
```

### 3. Grant the benchmark role access to the compat bucket

The CI workflow reuses the existing `GitHubBenchmarkRole`
(`arn:aws:iam::245040174862:role/GitHubBenchmarkRole`).
Add an inline policy granting it S3 access to the compat fixtures bucket:

```bash
aws iam put-role-policy \
  --role-name GitHubBenchmarkRole \
  --policy-name CompatFixturesS3Access \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket"
        ],
        "Resource": [
          "arn:aws:s3:::vortex-compat-fixtures",
          "arn:aws:s3:::vortex-compat-fixtures/*"
        ]
      }
    ]
  }'
```

## CI Workflows

### Fixture upload (`.github/workflows/compat-gen-upload.yml`)

Triggered via **manual dispatch** with a required `version` input (e.g. `0.62.0`).
Will be updated to also trigger on release tag pushes once the workflow is proven.

1. Checks out the current branch
2. Runs `scripts/compat.py add-version --version <input> --target s3` which:
   - Builds and runs `compat-gen` to generate fixtures
   - Fetches the previous version's manifest and merges `since` values
   - Enforces additive-only (no fixtures removed)
   - Uploads fixtures to `s3://vortex-compat-fixtures/v<version>/`
   - Updates `versions.json` with ETag-based optimistic locking

### Weekly compat test (`.github/workflows/compat-test-weekly.yml`)

Runs **every Monday at 06:00 UTC** and on **manual dispatch**.

1. Checks out `main` at HEAD
2. Runs `scripts/compat.py check --target s3`
3. Validates every version listed in `versions.json`

## Fixture Suite

| Fixture | File | Since | Description |
|---------|------|-------|-------------|
| Primitives | `primitives.vortex` | 0.62.0 | All numeric types (u8–u64, i32, i64, f32, f64) with min/mid/max values |
| Strings | `strings.vortex` | 0.62.0 | Variable-length strings including empty, ASCII, Unicode, and emoji |
| Booleans | `booleans.vortex` | 0.62.0 | Boolean array with mixed true/false values |
| Nullable | `nullable.vortex` | 0.62.0 | Nullable int and string columns with interleaved nulls |
| Nested Struct | `struct_nested.vortex` | 0.62.0 | Two-level nested struct (inner struct within outer struct) |
| Chunked | `chunked.vortex` | 0.62.0 | Multi-chunk file: 3 chunks of 1000 rows each |
| TPC-H Lineitem | `tpch_lineitem.vortex` | 0.62.0 | TPC-H lineitem table at scale factor 0.01 |
| TPC-H Orders | `tpch_orders.vortex` | 0.62.0 | TPC-H orders table at scale factor 0.01 |
| ClickBench Hits | `clickbench_hits_1k.vortex` | 0.62.0 | First 1000 rows of the ClickBench hits table |

### Adding a new fixture

New encodings, data types, or structural patterns always get a **new fixture file**.
Never modify an existing fixture's `build()` output (see [Fixture Contract](#fixture-contract)).

1. Create a struct implementing the `Fixture` trait in `src/fixtures/`:
   ```rust
   pub struct MyFixture;
   impl Fixture for MyFixture {
       fn name(&self) -> &str { "my_fixture.vortex" }
       fn build(&self) -> VortexResult<Vec<ArrayRef>> { /* deterministic array construction */ }
   }
   ```
2. Register it in `all_fixtures()` in `src/fixtures/mod.rs`.
3. Run `compat-gen` locally to verify it produces a valid file.
4. Upload fixtures for the current version — the upload script merges the manifest
   so the new fixture gets `since` set to the current version while existing
   fixtures keep their original `since` values.

## Adapter Epochs

The adapter module (`src/adapter.rs`) contains the read/write logic for the Vortex file
format. As the format API evolves across major versions, new "epochs" are introduced:

| Epoch | Vortex Versions | Key API Surface |
|-------|----------------|-----------------|
| A | v0.36.0 | Original `VortexFileWriter` / `VortexOpenOptions` |
| B | v0.45.0 – v0.52.0 | Intermediate session-based API |
| C | v0.58.0 – HEAD | `session.write_options()` / `session.open_options().open_buffer()` |

Only Epoch C is currently active. Earlier epochs were used during initial development
and can be resurrected by cherry-picking the adapter code onto an older release branch
if retroactive fixture generation is needed.

### Cherry-picking to older releases

To generate fixtures for a version in Epoch A or B:

1. Check out the target tag (e.g. `git checkout v0.45.0`)
2. Cherry-pick the compat-gen crate: `git cherry-pick --no-commit <commit-range>`
3. Swap `src/adapter.rs` to the appropriate epoch's implementation
4. Resolve any dependency mismatches in `Cargo.toml`
5. Run `compat-gen` and upload the resulting fixtures

# vortex-compat: Backward-Compatibility Testing

This crate provides two binaries that together ensure Vortex can always read files
written by older versions:

- **`compat-gen`** — generates deterministic fixture files for a given Vortex version.
- **`compat-test`** — reads fixtures from every historical version and validates
  they round-trip to the expected arrays.

Fixtures are stored in an S3 bucket. CI uploads new fixtures on every release tag
and runs weekly validation against all prior versions.

## First-Time Setup: Bootstrap the Bucket

After creating the S3 bucket (see [AWS Setup](#aws-setup-one-time) below), seed it
with the first fixture set:

```bash
# 1. Generate fixtures for the current version
cargo run -p vortex-compat --release --bin compat-gen -- \
  --version 0.62.0 --output /tmp/fixtures/

# 2. Upload to S3
AWS_PROFILE=vortex-ci aws s3 cp /tmp/fixtures/ \
  s3://vortex-compat-fixtures/v0.62.0/ --recursive

# 3. Create the initial versions.json
echo '["0.62.0"]' > /tmp/versions.json
AWS_PROFILE=vortex-ci aws s3 cp /tmp/versions.json \
  s3://vortex-compat-fixtures/versions.json

# 4. Verify the round-trip
AWS_PROFILE=vortex-ci cargo run -p vortex-compat --release --bin compat-test -- \
  --fixtures-url https://vortex-compat-fixtures.s3.amazonaws.com
```

## Uploading Fixtures for a New Version

When a new Vortex version is tagged and you want to upload its fixtures manually
(CI does this automatically on tag push):

```bash
VERSION=0.63.0

# 1. Generate fixtures
cargo run -p vortex-compat --release --bin compat-gen -- \
  --version "$VERSION" --output /tmp/fixtures/

# 2. Upload to S3 under the new version prefix
AWS_PROFILE=vortex-ci aws s3 cp /tmp/fixtures/ \
  "s3://vortex-compat-fixtures/v${VERSION}/" --recursive

# 3. Append the version to versions.json
AWS_PROFILE=vortex-ci aws s3 cp \
  s3://vortex-compat-fixtures/versions.json /tmp/versions.json
python3 -c "
import json, sys
with open('/tmp/versions.json') as f:
    versions = json.load(f)
v = sys.argv[1]
if v not in versions:
    versions.append(v)
    versions.sort(key=lambda x: list(map(int, x.split('.'))))
with open('/tmp/versions.json', 'w') as f:
    json.dump(versions, f, indent=2)
" "$VERSION"
AWS_PROFILE=vortex-ci aws s3 cp /tmp/versions.json \
  s3://vortex-compat-fixtures/versions.json

# 4. Verify all versions (including the new one)
AWS_PROFILE=vortex-ci cargo run -p vortex-compat --release --bin compat-test -- \
  --fixtures-url https://vortex-compat-fixtures.s3.amazonaws.com
```

## Re-uploading Fixtures for an Existing Version

If a fixture was added or changed and you need to regenerate for a version that
already exists in the bucket, the upload overwrites the existing prefix:

```bash
VERSION=0.62.0

# 1. Regenerate
cargo run -p vortex-compat --release --bin compat-gen -- \
  --version "$VERSION" --output /tmp/fixtures/

# 2. Overwrite in S3
AWS_PROFILE=vortex-ci aws s3 cp /tmp/fixtures/ \
  "s3://vortex-compat-fixtures/v${VERSION}/" --recursive

# 3. Verify
AWS_PROFILE=vortex-ci cargo run -p vortex-compat --release --bin compat-test -- \
  --fixtures-url https://vortex-compat-fixtures.s3.amazonaws.com
```

No need to update `versions.json` — the version is already listed.

## Local-Only Workflow

You can skip S3 entirely and work against local directories:

```bash
# Generate into a versioned subdirectory
cargo run -p vortex-compat --release --bin compat-gen -- \
  --version 0.62.0 --output /tmp/compat-root/v0.62.0/

# Validate all local versions
cargo run -p vortex-compat --release --bin compat-test -- \
  --fixtures-dir /tmp/compat-root/
```

If the bucket requires authenticated access, set your AWS profile:

```bash
AWS_PROFILE=vortex-ci cargo run -p vortex-compat --release --bin compat-test -- \
  --fixtures-url https://vortex-compat-fixtures.s3.amazonaws.com
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

### 3. Create an IAM OIDC provider for GitHub Actions

Skip this step if the account already has a GitHub OIDC provider configured.

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

### 4. Create the IAM role for CI

Create the trust policy file (`trust-policy.json`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::245040174862:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:spiraldb/vortex:ref:refs/tags/*"
        }
      }
    }
  ]
}
```

Create the role:

```bash
aws iam create-role \
  --role-name GitHubCompatFixturesRole \
  --assume-role-policy-document file://trust-policy.json
```

Attach an inline permission policy:

```bash
aws iam put-role-policy \
  --role-name GitHubCompatFixturesRole \
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

### 5. Store the role ARN as a GitHub secret

```bash
gh secret set COMPAT_FIXTURES_ROLE_ARN \
  --body "arn:aws:iam::245040174862:role/GitHubCompatFixturesRole"
```

## CI Workflows

### Fixture upload (`.github/workflows/compat-gen-upload.yml`)

Triggered via **manual dispatch** with a required `version` input (e.g. `0.62.0`).
Will be updated to also trigger on release tag pushes once the workflow is proven.

1. Checks out the current branch
2. Runs `compat-gen --version <input> --output /tmp/fixtures/`
3. Assumes the `GitHubCompatFixturesRole` via OIDC
4. Uploads fixtures to `s3://vortex-compat-fixtures/v<version>/`
5. Appends the version to `versions.json`

### Weekly compat test (`.github/workflows/compat-test-weekly.yml`)

Runs **every Monday at 06:00 UTC** and on **manual dispatch**.

1. Checks out `main` at HEAD
2. Runs `compat-test --fixtures-url https://vortex-compat-fixtures.s3.amazonaws.com`
3. Validates every version listed in `versions.json`

## Fixture Suite

| Fixture | File | Description |
|---------|------|-------------|
| Primitives | `primitives.vortex` | All numeric types (u8–u64, i32, i64, f32, f64) with min/mid/max values |
| Strings | `strings.vortex` | Variable-length strings including empty, ASCII, Unicode, and emoji |
| Booleans | `booleans.vortex` | Boolean array with mixed true/false values |
| Nullable | `nullable.vortex` | Nullable int and string columns with interleaved nulls |
| Nested Struct | `struct_nested.vortex` | Two-level nested struct (inner struct within outer struct) |
| Chunked | `chunked.vortex` | Multi-chunk file: 3 chunks of 1000 rows each |
| TPC-H Lineitem | `tpch_lineitem.vortex` | TPC-H lineitem table at scale factor 0.01 |
| TPC-H Orders | `tpch_orders.vortex` | TPC-H orders table at scale factor 0.01 |
| ClickBench Hits | `clickbench_hits_1k.vortex` | First 1000 rows of the ClickBench hits table |

Encoding-specific fixtures (Dict, RunEnd, Constant, Sparse, ALP, BitPacked, FSST) are
stubbed and will be enabled once the stable-encodings RFC lands.

### Adding a new fixture

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

The `build()` method **must be deterministic** — `compat-test` calls it to produce the
expected arrays and compares against what was read from disk.

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

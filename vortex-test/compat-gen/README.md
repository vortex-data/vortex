# vortex-compat

Backward-compatibility testing for the Vortex file format. Ensures the
current reader can always decode `.vortex` files written by any older
released version.

See [DESIGN.md](DESIGN.md) for full architecture and design rationale.

## Quick Start

```bash
# Generate fixtures locally
cargo run -p vortex-compat --release -- generate --output /tmp/fixtures

# Check fixtures against current reader
cargo run -p vortex-compat --release -- check --dir /tmp/fixtures --mode exact

# Publish to S3 (requires AWS credentials)
python scripts/compat.py publish

# Check all published versions
python scripts/compat.py check

# Publish from a historical tag via git worktree
python scripts/compat.py publish --git-ref v0.62.0

# Local-only workflow (no S3)
python scripts/compat.py publish --store /tmp/compat-store
python scripts/compat.py check --store /tmp/compat-store
```

## AWS Setup (one-time)

All resources live in the benchmark account (245040174862), region us-east-1.

### 1. Create the S3 bucket

```bash
aws s3api create-bucket --bucket vortex-compat-fixtures --region us-east-1
```

### 2. Enable public read access

```bash
aws s3api put-public-access-block \
  --bucket vortex-compat-fixtures \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=false,RestrictPublicBuckets=false

aws s3api put-bucket-policy \
  --bucket vortex-compat-fixtures \
  --policy '{
    "Version": "2012-10-17",
    "Statement": [{
      "Sid": "PublicRead",
      "Effect": "Allow",
      "Principal": "*",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::vortex-compat-fixtures",
        "arn:aws:s3:::vortex-compat-fixtures/*"
      ]
    }]
  }'
```

### 3. Grant CI role access

```bash
aws iam put-role-policy \
  --role-name GitHubBenchmarkRole \
  --policy-name CompatFixturesS3Access \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::vortex-compat-fixtures",
        "arn:aws:s3:::vortex-compat-fixtures/*"
      ]
    }]
  }'
```

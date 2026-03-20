#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# One-time migration script: split the monolithic data.json.gz into
# per-benchmark files at data/<id>.data.json.gz on S3.
#
# Uses cat-s3.sh to atomically merge historical data into any records
# that CI may have already written to the split files. If a split file
# doesn't exist yet on S3, it is created directly.
#
# Usage:
#   bash scripts/migrate-benchmark-data.sh
#
# Prerequisites:
#   - AWS CLI configured with write access to vortex-ci-benchmark-results
#   - python3 available

set -Eeu -o pipefail

BUCKET="vortex-ci-benchmark-results"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

echo "Downloading data.json.gz..."
aws s3 cp "s3://$BUCKET/data.json.gz" "$WORK_DIR/data.json.gz"
gzip -d -c "$WORK_DIR/data.json.gz" > "$WORK_DIR/data.json"

echo "Splitting records into per-benchmark files..."
python3 "$SCRIPT_DIR/split-benchmark-data.py" "$WORK_DIR/data.json" "$WORK_DIR/split"

echo ""
echo "Merging historical data into split S3 files..."
for f in "$WORK_DIR/split"/*.jsonl; do
    benchmark_id=$(basename "$f" .jsonl)
    key="data/${benchmark_id}.data.json.gz"

    # Check if the file already exists on S3
    if aws s3api head-object --bucket "$BUCKET" --key "$key" > /dev/null 2>&1; then
        echo "  Merging into existing $key..."
        bash "$SCRIPT_DIR/cat-s3.sh" "$BUCKET" "$key" "$f"
    else
        echo "  Creating new $key..."
        gzip -c "$f" > "$f.gz"
        aws s3api put-object --bucket "$BUCKET" --key "$key" --body "$f.gz"
    fi
done

echo ""
echo "Migration complete!"
echo "The original data.json.gz has NOT been deleted."
echo "Verify the split files look correct, then remove it manually if desired:"
echo "  aws s3 rm s3://$BUCKET/data.json.gz"

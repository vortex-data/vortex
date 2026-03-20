#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# One-time migration script: split the monolithic data.json.gz into
# per-benchmark files at data/<id>.data.json.gz on S3.
#
# Usage:
#   bash scripts/migrate-benchmark-data.sh
#
# Prerequisites:
#   - AWS CLI configured with write access to vortex-ci-benchmark-results
#   - python3 available
#   - gzip available

set -Eeu -o pipefail

BUCKET="vortex-ci-benchmark-results"
WORK_DIR=$(mktemp -d)
trap 'rm -rf "$WORK_DIR"' EXIT

echo "Downloading data.json.gz..."
aws s3 cp "s3://$BUCKET/data.json.gz" "$WORK_DIR/data.json.gz"
gzip -d -c "$WORK_DIR/data.json.gz" > "$WORK_DIR/data.json"

echo "Splitting records into per-benchmark files..."
python3 scripts/split-benchmark-data.py "$WORK_DIR/data.json" "$WORK_DIR/split"

echo "Uploading split files to S3..."
for f in "$WORK_DIR/split"/*.data.json.gz; do
    key="data/$(basename "$f")"
    echo "  Uploading $key..."
    aws s3 cp "$f" "s3://$BUCKET/$key"
done

echo ""
echo "Migration complete!"
echo "The original data.json.gz has NOT been deleted."
echo "Verify the split files look correct, then remove it manually if desired:"
echo "  aws s3 rm s3://$BUCKET/data.json.gz"

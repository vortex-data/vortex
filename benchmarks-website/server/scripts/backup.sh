#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
#
# Daily DuckDB backup for the vortex-bench-server v3 instance.
# Runs on the EC2 host via cron (see benchmarks-website/ec2-init.txt).
#
# Exports the running container's DuckDB to a local directory and uploads
# it to s3://vortex-ci-benchmark-results/v3-backups/<date>/. The instance
# IAM role already grants write access to that bucket (it is the same
# bucket cat-s3.sh uses for v2).
#
# At alpha this is a convenience backup: the data is also reproducible
# from CI dual-writes to the v3 ingest endpoint, so RPO is bounded by
# what CI has posted, not by this script's cadence.

set -euo pipefail

CONTAINER="${CONTAINER:-vortex-bench-server}"
DB_PATH="${DB_PATH:-/app/data/bench.duckdb}"
DATA_DIR="${DATA_DIR:-/opt/benchmarks-website/data}"
S3_PREFIX="${S3_PREFIX:-s3://vortex-ci-benchmark-results/v3-backups}"

date_stamp="$(date -u +%Y%m%d)"
export_dir="backup-${date_stamp}"
host_export_dir="${DATA_DIR}/${export_dir}"

# Run EXPORT DATABASE inside the container so we hit the same DuckDB
# build that wrote the file. The container path mirrors the host path
# under /app/data, so the export lands on the EBS volume.
docker exec "${CONTAINER}" \
    duckdb "${DB_PATH}" \
    -c "EXPORT DATABASE '/app/data/${export_dir}'"

aws s3 cp \
    --recursive \
    "${host_export_dir}" \
    "${S3_PREFIX}/${date_stamp}/"

# Keep the latest local export, drop older ones to bound disk use.
find "${DATA_DIR}" \
    -maxdepth 1 \
    -type d \
    -name "backup-*" \
    ! -path "${host_export_dir}" \
    -exec rm -rf {} +

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <corpus-directory> <object-key> <etag-state-file>" >&2
  exit 2
fi

corpus_dir=$1
object_key=$2
etag_file=$3
checkpoint_dir=$(mktemp -d "${TMPDIR:-/tmp}/fuzz-corpus.XXXXXX")
archive="$checkpoint_dir/corpus.tar.zst"
file_list="$checkpoint_dir/files"
latest_archive="$checkpoint_dir/latest.tar.zst"
latest_etag="$checkpoint_dir/latest.etag"

# A periodic checkpoint may still be finishing when the final checkpoint starts.
exec 9>"${RUNNER_TEMP:-${TMPDIR:-/tmp}}/vortex-fuzz-corpus-checkpoint.lock"
flock 9

cleanup() {
  rm -f "$archive" "$file_list" "$latest_archive" "$latest_etag"
  rmdir "$checkpoint_dir"
}
trap cleanup EXIT

if [ ! -f "$etag_file" ]; then
  echo "Missing corpus ETag state: $etag_file" >&2
  exit 1
fi

for attempt in 1 2 3; do
  # Corpus entries are immutable. Snapshotting the file list first makes an archive that is
  # consistent even while libFuzzer is adding new entries to the directory.
  find "$corpus_dir" -type f -print0 > "$file_list"
  tar --create --auto-compress --file "$archive" --null --files-from "$file_list"

  etag=$(tr -d '\r\n' < "$etag_file")
  if [ -z "$etag" ]; then
    echo "Corpus ETag state is empty: $etag_file" >&2
    exit 1
  fi
  condition=(--if-match "$etag")
  if [ "$etag" = "CREATE_ONLY" ]; then
    condition=(--if-none-match "*")
  fi

  set +e
  python3 scripts/s3-upload.py \
    --bucket vortex-fuzz-corpus \
    --key "$object_key" \
    --body "$archive" \
    --checksum-algorithm CRC32 \
    --etag-output "$etag_file" \
    "${condition[@]}"
  status=$?
  set -e

  if [ "$status" -eq 0 ]; then
    exit 0
  fi
  if [ "$status" -ne 3 ]; then
    exit "$status"
  fi
  if [ "$attempt" -eq 3 ]; then
    break
  fi

  echo "Corpus changed remotely; merging before CAS retry $attempt/3"
  python3 scripts/s3-download.py \
    "s3://vortex-fuzz-corpus/$object_key" "$latest_archive" \
    --etag-output "$latest_etag"
  tar -xf "$latest_archive"
  cp "$latest_etag" "$etag_file"
done

echo "Unable to checkpoint corpus after three concurrent updates" >&2
exit 1

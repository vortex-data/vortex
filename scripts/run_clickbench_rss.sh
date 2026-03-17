#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v gtime >/dev/null 2>&1; then
  echo "error: gtime not found in PATH" >&2
  exit 1
fi

BINARY="${BINARY:-$ROOT_DIR/target/release/datafusion-bench}"
FORMAT="${FORMAT:-vortex}"
START_QUERY="${START_QUERY:-0}"
END_QUERY="${END_QUERY:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/data/clickbench-rss}"

mkdir -p "$OUTPUT_DIR"

echo "building release benchmark binary..."
cargo build -p datafusion-bench --release

timestamp="$(date +%Y%m%d-%H%M%S)"
csv_path="$OUTPUT_DIR/clickbench-rss-$timestamp.csv"

cat >"$csv_path" <<'EOF'
query,max_rss_kb,elapsed,user_seconds,system_seconds,cpu_percent,exit_status
EOF

for ((q = START_QUERY; q <= END_QUERY; q++)); do
  log_path="$OUTPUT_DIR/q${q}.gtime.$timestamp.log"
  echo "running clickbench query $q ..."

  gtime --verbose \
    "$BINARY" clickbench --formats "$FORMAT" --queries "$q" \
    >"$OUTPUT_DIR/q${q}.stdout.$timestamp.log" \
    2>"$log_path"

  max_rss_kb="$(awk -F': ' '/Maximum resident set size/ {print $2}' "$log_path" | tr -d '[:space:]')"
  elapsed="$(awk -F': ' '/Elapsed \(wall clock\) time/ {print $2}' "$log_path" | sed 's/^ *//')"
  user_seconds="$(awk -F': ' '/User time \(seconds\)/ {print $2}' "$log_path" | tr -d '[:space:]')"
  system_seconds="$(awk -F': ' '/System time \(seconds\)/ {print $2}' "$log_path" | tr -d '[:space:]')"
  cpu_percent="$(awk -F': ' '/Percent of CPU this job got/ {print $2}' "$log_path" | tr -d '[:space:]')"
  exit_status="$(awk -F': ' '/Exit status/ {print $2}' "$log_path" | tr -d '[:space:]')"

  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "$q" \
    "${max_rss_kb:-}" \
    "${elapsed:-}" \
    "${user_seconds:-}" \
    "${system_seconds:-}" \
    "${cpu_percent:-}" \
    "${exit_status:-}" \
    >>"$csv_path"
done

echo "wrote summary to $csv_path"

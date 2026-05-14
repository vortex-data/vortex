#!/usr/bin/env bash
# Run the funnel_shift_fix bench best-of-3 at --min-time 1.0
# Produces measurements/funnel_fix.csv
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$1"
OUT="$2"

VARIANTS=(
    baseline_macro_bare
    baseline_macro_fused
    hand_legacy
    hand_funnel
)
WIDTHS=(51 63)

echo "variant,W,best_median_ns" > "$OUT"

for variant in "${VARIANTS[@]}"; do
    for W in "${WIDTHS[@]}"; do
        name="${variant}__u64__w${W}"
        best=""
        for i in 1 2 3; do
            line=$("$BIN" "${name}\$" --bench --min-time 1.0 2>&1 \
                | grep -E "[├╰]─\s+${name}\b" || true)
            # Extract the median column (column 6, 7 = val, unit).
            # Divan format: name fastest u | slowest u | median u | mean u | samples | iters
            if [ -n "$line" ]; then
                # Parse with awk: split on │ and grab the third group (median val unit)
                ns=$(echo "$line" | awk -F'│' '{print $3}' | awk '{
                    val=$1; unit=$2;
                    mult=1.0;
                    if (unit == "µs" || unit == "us") mult=1000.0;
                    else if (unit == "ms") mult=1000000.0;
                    else if (unit == "s") mult=1000000000.0;
                    printf "%.3f", val*mult;
                }')
                if [ -z "$best" ] || [ "$(echo "$ns < $best" | bc)" = "1" ]; then
                    best="$ns"
                fi
                echo "  iter $i: ${ns} ns" >&2
            fi
        done
        echo "${variant},${W},${best}" | tee -a "$OUT" >&2
    done
done

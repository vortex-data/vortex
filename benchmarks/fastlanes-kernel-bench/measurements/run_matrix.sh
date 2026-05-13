#!/usr/bin/env bash
# Collect best-of-3 medians for every (T, W, SIMD, variant) cell.
# Appends to /tmp/full_matrix.csv. Re-runnable; will overwrite the file
# at start by default. If RESUME=1 in the env, skips already-measured rows.

set -u

BIN_SSE2="target/release/deps/unpack_vs_fused-ea96ed7a581f4adc"
BIN_AVX2="target/release/deps/unpack_vs_fused-16e476fea4eedc3d"
BIN_AVX512="target/release/deps/unpack_vs_fused-afe3f8c643e9a77f"

OUT="/tmp/full_matrix.csv"

if [ "${RESUME:-0}" != "1" ]; then
    echo "T,W,simd,variant,best_median_ns" > "$OUT"
fi

# Build "already measured" set if resuming.
declare -A DONE
if [ "${RESUME:-0}" = "1" ] && [ -f "$OUT" ]; then
    while IFS=, read -r t w s v ns; do
        [ "$t" = "T" ] && continue
        DONE["$t,$w,$s,$v"]=1
    done < "$OUT"
fi

# Convert a value+unit like "63.93 ns" or "1.234 µs" to nanoseconds.
to_ns() {
    local val="$1"
    local unit="$2"
    case "$unit" in
        ns)  awk -v v="$val" 'BEGIN{ printf "%.6f", v }' ;;
        µs|us) awk -v v="$val" 'BEGIN{ printf "%.6f", v*1000 }' ;;
        ms)  awk -v v="$val" 'BEGIN{ printf "%.6f", v*1000000 }' ;;
        s)   awk -v v="$val" 'BEGIN{ printf "%.6f", v*1000000000 }' ;;
        *)   echo "ERROR_unit_$unit"; return 1 ;;
    esac
}

# Parse one bench run: read stdin, return the median ns for the line whose
# function name exactly equals $1. Outputs the median ns or empty string.
parse_median() {
    local target="$1"
    # divan output line format (relevant fields):
    #   ├─ <name>  <fastest> <unit> │ <slowest> <unit> │ <median> <unit> │ ...
    # Strip the leading ├─ or ╰─, then split on │.
    awk -v target="$target" '
        /^[├╰]─/ {
            # Drop the tree prefix.
            sub(/^[├╰]─ /, "");
            # Split on the box-drawing vertical bar separator. awk in this
            # locale handles utf-8 char │ as multi-byte but split with that
            # literal works under LC_ALL=C.UTF-8.
            n = split($0, parts, /│/);
            # parts[1] = "<name>  <fastest_val> <unit>"
            # parts[3] = " <median_val> <unit>" with leading/trailing space
            split(parts[1], head, /[ \t]+/);
            name = head[1];
            if (name != target) next;
            split(parts[3], med, /[ \t]+/);
            # med array: [empty?] median_val median_unit ... handle leading space
            # Walk to find first nonempty token = val, next = unit.
            val=""; unit="";
            for (i = 1; i <= length(med); i++) {
                if (med[i] != "") {
                    if (val == "") val = med[i];
                    else { unit = med[i]; break; }
                }
            }
            print val, unit;
            exit;
        }
    '
}

run_one() {
    local bin="$1"
    local pattern="$2"
    "$bin" "$pattern" --min-time 0.5 --bench 2>&1
}

# Measure one cell: returns best-of-3 median ns on stdout.
measure_cell() {
    local bin="$1"
    local t="$2"
    local w="$3"
    local variant="$4"
    local fname="${variant}__${t}__w${w}"
    local pattern="${variant}__${t}__w${w}\$"
    local best=""
    for i in 1 2 3; do
        local out
        out=$(run_one "$bin" "$pattern")
        local parsed
        parsed=$(printf '%s\n' "$out" | parse_median "$fname")
        if [ -z "$parsed" ]; then
            # Try alternate pattern: maybe filter doesn't anchor properly
            echo "WARN: empty parse for $fname pass $i" >&2
            continue
        fi
        local val unit
        val=$(echo "$parsed" | awk '{print $1}')
        unit=$(echo "$parsed" | awk '{print $2}')
        local ns
        ns=$(to_ns "$val" "$unit")
        if [ -z "$best" ]; then
            best="$ns"
        else
            best=$(awk -v a="$best" -v b="$ns" 'BEGIN{ if (b<a) print b; else print a }')
        fi
    done
    echo "$best"
}

# Type -> max W table.
declare -A MAXW
MAXW[u8]=8
MAXW[u16]=16
MAXW[u32]=32
MAXW[u64]=64

TOTAL=720
COUNTER=0
START=$(date +%s)

for T in u8 u16 u32 u64; do
    MAX="${MAXW[$T]}"
    for W in $(seq 1 "$MAX"); do
        for SIMD_PAIR in "sse2:$BIN_SSE2" "ymm:$BIN_AVX2" "zmm:$BIN_AVX512"; do
            SIMD="${SIMD_PAIR%%:*}"
            BIN="${SIMD_PAIR#*:}"
            for VARIANT in bare_unpack fused_for; do
                COUNTER=$((COUNTER+1))
                KEY="$T,$W,$SIMD,$VARIANT"
                if [ "${DONE[$KEY]:-}" = "1" ]; then
                    continue
                fi
                NS=$(measure_cell "$BIN" "$T" "$W" "$VARIANT")
                if [ -z "$NS" ]; then
                    echo "ERROR: no median for $KEY" >&2
                    NS="NaN"
                fi
                printf '%s,%s\n' "$KEY" "$NS" >> "$OUT"
                NOW=$(date +%s)
                ELAPSED=$((NOW-START))
                if [ "$COUNTER" -gt 0 ]; then
                    ETA=$(awk -v e="$ELAPSED" -v c="$COUNTER" -v t="$TOTAL" 'BEGIN{ printf "%.0f", e/c*(t-c) }')
                else
                    ETA=0
                fi
                printf '[%4d/%4d  elapsed=%ds eta=%ds] %s -> %s ns\n' "$COUNTER" "$TOTAL" "$ELAPSED" "$ETA" "$KEY" "$NS"
            done
        done
    done
done

echo "Done. Wrote $OUT"
wc -l "$OUT"

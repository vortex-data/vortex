#!/usr/bin/env bash
# Measure AOT bitpacking-kernel code size and compile time. Run from the
# crate root. Reports, per binary:
#   - clean release build wall time
#   - .text bytes consumed by fastlanes `unpack` / `unpack_cmp` symbols
#   - total .text bytes
#   - count of monomorphized kernel symbols
set -euo pipefail

cd "$(dirname "$0")"
RELEASE=target/release

measure() {
  local bin="$1"
  cargo clean -q
  local t0 t1
  t0=$(date +%s.%N)
  cargo build --release --bin "$bin" -q 2>&1 | tail -3
  t1=$(date +%s.%N)
  local build_secs
  build_secs=$(awk "BEGIN { printf \"%.2f\", $t1 - $t0 }")

  local elf="$RELEASE/$bin"
  local total_text bp_summary bp_bytes bp_count
  total_text=$(readelf -SW "$elf" | awk '$2==".text" { print $6 }' | head -1)
  total_text=$(printf "%d" "0x$total_text")

  bp_summary=$(
    nm -S --radix=d --demangle "$elf" 2>/dev/null \
      | awk '
          /fastlanes::bitpacking/ && /(::unpack|::unpack_cmp|::pack)/ {
            n++; s += $2 + 0;
          }
          END { printf "%d %d", s+0, n+0 }'
  )
  bp_bytes=${bp_summary% *}
  bp_count=${bp_summary#* }

  printf "%-26s  build=%6ss  .text=%9s B  fastlanes kernels: %5s syms / %8s B\n" \
    "$bin" "$build_secs" "$total_text" "$bp_count" "$bp_bytes"
}

echo "Building each binary cleanly (codegen-units=1, no LTO):"
for bin in unpack_only cmp_one_op cmp_all_ops_u32 cmp_all_ops_all_types; do
  measure "$bin"
done

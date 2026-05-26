#!/usr/bin/env bash
# Emits assembly for the codegen subjects and asserts each kernel vectorizes:
#   - contains wide vector registers (ymm/zmm),
#   - performs no `memcpy` (by-value [T; N] blocks stay register-resident),
#   - is not left as a scalar fallback.
#
# Modeled on zerocopy's cargo-show-asm codegen tests, but uses the zero-
# dependency `cargo rustc --emit asm` so no extra tooling is required.
set -euo pipefail

cd "$(dirname "$0")"

TARGET_FEATURE="${TARGET_FEATURE:-+avx2,+fma}"

echo "Emitting assembly (target-feature=${TARGET_FEATURE})..."
RUSTFLAGS="-C target-feature=${TARGET_FEATURE}" \
    cargo rustc --release --example simd_subjects -- --emit asm >/dev/null

ASM=$(find ../target/release/examples -name 'simd_subjects-*.s' -print -quit)
if [[ -z "${ASM}" ]]; then
    echo "ERROR: could not locate emitted assembly" >&2
    exit 1
fi
echo "Assembly: ${ASM}"
echo

SUBJECTS=(subject_sum_u32 subject_mul3_u32 subject_dot_f32 subject_add_i32)
status=0

for fn in "${SUBJECTS[@]}"; do
    body=$(awk "/^${fn}:/{p=1} p{print} /\.size[[:space:]]+${fn}/{exit}" "${ASM}")
    ymm=$(grep -Eoc '%(y|z)mm' <<<"${body}" || true)
    memcpy=$(grep -c 'memcpy' <<<"${body}" || true)

    if [[ "${ymm}" -gt 0 && "${memcpy}" -eq 0 ]]; then
        echo "PASS ${fn}: ${ymm} wide-vector refs, no memcpy"
    else
        echo "FAIL ${fn}: wide-vector refs=${ymm}, memcpy=${memcpy}"
        status=1
    fi
done

echo
if [[ "${status}" -eq 0 ]]; then
    echo "All subjects vectorized with zero copy."
else
    echo "Some subjects did not vectorize as expected." >&2
fi
exit "${status}"

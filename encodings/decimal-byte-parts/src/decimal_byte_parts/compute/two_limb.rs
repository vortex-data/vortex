// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fused single-pass kernels over the two-limb (signed-high `i64`, unsigned-low `u64`) i128
//! representation, shared by `between` and `compare`.
//!
//! arrow's i128 comparison has no SIMD form on any x86 (there is no 128-bit-integer vector
//! compare), so it is inherently scalar. The limbs, by contrast, are native widths: on AVX-512 we
//! compare 8 lanes of `i64`/`u64` with `vpcmpq`/`vpcmpuq`, each producing a `__mmask8` that is
//! exactly one byte of the output bitmap — no serial bit-packing. A scalar reconstruct path (which
//! is itself competitive with arrow) is used when AVX-512 is unavailable.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::between::BetweenOptions;
use vortex_array::scalar_fn::fns::between::StrictComparison;
use vortex_array::scalar_fn::fns::operators::CompareOperator;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;

/// Materialize the two limbs, run a fused bit-kernel over them, and wrap the result as a boolean
/// array carrying the combined validity. Shared entry point for the two-limb `compare`/`between`
/// paths: the caller supplies a kernel mapping the high (`i64`) and low (`u64`) limb slices to a
/// packed [`BitBuffer`].
pub(crate) fn eval(
    arr: &ArrayView<'_, DecimalByteParts>,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
    kernel: impl FnOnce(&[i64], &[u64]) -> BitBuffer,
) -> VortexResult<ArrayRef> {
    let high = arr.msp().clone().execute::<PrimitiveArray>(ctx)?;
    let low = arr
        .lower()
        .vortex_expect("two-limb path requires a lower limb")
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let validity = high.validity()?.union_nullability(nullability);
    let bits = kernel(high.as_slice::<i64>(), low.as_slice::<u64>());
    Ok(BoolArray::new(bits, validity).into_array())
}

/// Reconstruct the i128 value from its limbs: sign-extend the high limb, zero-extend the low limb.
#[inline(always)]
fn reconstruct(high: i64, low: u64) -> i128 {
    ((high as i128) << 64) | (low as i128)
}

// Operator codes, used as const generics to monomorphize the comparison out of the hot loop.
const OP_LT: u8 = 0;
const OP_LE: u8 = 1;
const OP_GT: u8 = 2;
const OP_GE: u8 = 3;
const OP_EQ: u8 = 4;
const OP_NE: u8 = 5;

#[inline(always)]
fn op_code(op: CompareOperator) -> u8 {
    match op {
        CompareOperator::Lt => OP_LT,
        CompareOperator::Lte => OP_LE,
        CompareOperator::Gt => OP_GT,
        CompareOperator::Gte => OP_GE,
        CompareOperator::Eq => OP_EQ,
        CompareOperator::NotEq => OP_NE,
    }
}

#[inline(always)]
fn cmp_i128<const OP: u8>(value: i128, bound: i128) -> bool {
    match OP {
        OP_LT => value < bound,
        OP_LE => value <= bound,
        OP_GT => value > bound,
        OP_GE => value >= bound,
        OP_EQ => value == bound,
        OP_NE => value != bound,
        _ => unreachable!(),
    }
}

/// `value <op> bound` over the two-limb representation, returned as a packed [`BitBuffer`].
pub(crate) fn compare_bits(
    high: &[i64],
    low: &[u64],
    bound: i128,
    op: CompareOperator,
) -> BitBuffer {
    debug_assert_eq!(high.len(), low.len());
    let code = op_code(op);

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") {
        let mut bytes = BufferMut::<u8>::zeroed(high.len().div_ceil(8));
        // SAFETY: avx512f is detected, and `bytes` has `len.div_ceil(8)` bytes.
        unsafe { avx512::compare(high, low, bound, code, bytes.as_mut_slice()) };
        return BitBuffer::new(bytes.freeze(), high.len());
    }

    compare_scalar(high, low, bound, code)
}

/// `lower <= value <= upper` (respecting strictness) over the two-limb representation, returned as
/// a packed [`BitBuffer`].
pub(crate) fn between_bits(
    high: &[i64],
    low: &[u64],
    lower: i128,
    upper: i128,
    options: &BetweenOptions,
) -> BitBuffer {
    debug_assert_eq!(high.len(), low.len());
    let lower_op = match options.lower_strict {
        StrictComparison::Strict => OP_GT,
        StrictComparison::NonStrict => OP_GE,
    };
    let upper_op = match options.upper_strict {
        StrictComparison::Strict => OP_LT,
        StrictComparison::NonStrict => OP_LE,
    };

    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512f") {
        let mut bytes = BufferMut::<u8>::zeroed(high.len().div_ceil(8));
        // SAFETY: avx512f is detected, and `bytes` has `len.div_ceil(8)` bytes.
        unsafe {
            avx512::between(
                high,
                low,
                lower,
                upper,
                lower_op,
                upper_op,
                bytes.as_mut_slice(),
            )
        };
        return BitBuffer::new(bytes.freeze(), high.len());
    }

    between_scalar(high, low, lower, upper, lower_op, upper_op)
}

// ---- Scalar reconstruct fallback ----

fn compare_scalar(high: &[i64], low: &[u64], bound: i128, code: u8) -> BitBuffer {
    match code {
        OP_LT => compare_scalar_impl::<OP_LT>(high, low, bound),
        OP_LE => compare_scalar_impl::<OP_LE>(high, low, bound),
        OP_GT => compare_scalar_impl::<OP_GT>(high, low, bound),
        OP_GE => compare_scalar_impl::<OP_GE>(high, low, bound),
        OP_EQ => compare_scalar_impl::<OP_EQ>(high, low, bound),
        _ => compare_scalar_impl::<OP_NE>(high, low, bound),
    }
}

fn compare_scalar_impl<const OP: u8>(high: &[i64], low: &[u64], bound: i128) -> BitBuffer {
    BitBuffer::collect_bool(high.len(), |idx| {
        // SAFETY: collect_bool yields idx in 0..high.len(), and low.len() == high.len().
        let value = reconstruct(unsafe { *high.get_unchecked(idx) }, unsafe {
            *low.get_unchecked(idx)
        });
        cmp_i128::<OP>(value, bound)
    })
}

fn between_scalar(
    high: &[i64],
    low: &[u64],
    lower: i128,
    upper: i128,
    lower_op: u8,
    upper_op: u8,
) -> BitBuffer {
    match (lower_op, upper_op) {
        (OP_GT, OP_LT) => between_scalar_impl::<OP_GT, OP_LT>(high, low, lower, upper),
        (OP_GT, _) => between_scalar_impl::<OP_GT, OP_LE>(high, low, lower, upper),
        (_, OP_LT) => between_scalar_impl::<OP_GE, OP_LT>(high, low, lower, upper),
        _ => between_scalar_impl::<OP_GE, OP_LE>(high, low, lower, upper),
    }
}

fn between_scalar_impl<const LOWER: u8, const UPPER: u8>(
    high: &[i64],
    low: &[u64],
    lower: i128,
    upper: i128,
) -> BitBuffer {
    BitBuffer::collect_bool(high.len(), |idx| {
        // SAFETY: collect_bool yields idx in 0..high.len(), and low.len() == high.len().
        let value = reconstruct(unsafe { *high.get_unchecked(idx) }, unsafe {
            *low.get_unchecked(idx)
        });
        cmp_i128::<LOWER>(value, lower) & cmp_i128::<UPPER>(value, upper)
    })
}

// ---- AVX-512 fast path ----

#[cfg(target_arch = "x86_64")]
mod avx512 {
    // Extracting the low 64 bits of an i128 bound via `as u64` is intentional limb truncation.
    #![allow(clippy::cast_possible_truncation)]

    use std::arch::x86_64::__m512i;
    use std::arch::x86_64::_mm512_cmpeq_epi64_mask;
    use std::arch::x86_64::_mm512_cmpge_epu64_mask;
    use std::arch::x86_64::_mm512_cmpgt_epi64_mask;
    use std::arch::x86_64::_mm512_cmpgt_epu64_mask;
    use std::arch::x86_64::_mm512_cmple_epu64_mask;
    use std::arch::x86_64::_mm512_cmplt_epi64_mask;
    use std::arch::x86_64::_mm512_cmplt_epu64_mask;
    use std::arch::x86_64::_mm512_loadu_epi64;
    use std::arch::x86_64::_mm512_set1_epi64;

    use super::OP_EQ;
    use super::OP_GE;
    use super::OP_GT;
    use super::OP_LE;
    use super::OP_LT;
    use super::OP_NE;
    use super::cmp_i128;
    use super::reconstruct;

    /// Per-chunk mask for `value <OP> bound`, combining the signed-high and unsigned-low limb
    /// comparisons lexicographically. Each operand is a vector of 8 lanes; the result is a
    /// `__mmask8` (one bit per lane).
    ///
    /// # Safety
    /// The caller must ensure the `avx512f` feature is enabled.
    #[target_feature(enable = "avx512f")]
    fn chunk_mask<const OP: u8>(h: __m512i, l: __m512i, bh: __m512i, bl: __m512i) -> u8 {
        match OP {
            OP_LT => {
                _mm512_cmplt_epi64_mask(h, bh)
                    | (_mm512_cmpeq_epi64_mask(h, bh) & _mm512_cmplt_epu64_mask(l, bl))
            }
            OP_LE => {
                _mm512_cmplt_epi64_mask(h, bh)
                    | (_mm512_cmpeq_epi64_mask(h, bh) & _mm512_cmple_epu64_mask(l, bl))
            }
            OP_GT => {
                _mm512_cmpgt_epi64_mask(h, bh)
                    | (_mm512_cmpeq_epi64_mask(h, bh) & _mm512_cmpgt_epu64_mask(l, bl))
            }
            OP_GE => {
                _mm512_cmpgt_epi64_mask(h, bh)
                    | (_mm512_cmpeq_epi64_mask(h, bh) & _mm512_cmpge_epu64_mask(l, bl))
            }
            OP_EQ => _mm512_cmpeq_epi64_mask(h, bh) & _mm512_cmpeq_epi64_mask(l, bl),
            OP_NE => !(_mm512_cmpeq_epi64_mask(h, bh) & _mm512_cmpeq_epi64_mask(l, bl)),
            _ => unreachable!(),
        }
    }

    /// # Safety
    /// The caller must ensure `avx512f` is enabled and `out.len() == high.len().div_ceil(8)`.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn compare(high: &[i64], low: &[u64], bound: i128, code: u8, out: &mut [u8]) {
        // SAFETY: avx512f is enabled by the target_feature on this fn and its callees.
        unsafe {
            match code {
                OP_LT => compare_impl::<OP_LT>(high, low, bound, out),
                OP_LE => compare_impl::<OP_LE>(high, low, bound, out),
                OP_GT => compare_impl::<OP_GT>(high, low, bound, out),
                OP_GE => compare_impl::<OP_GE>(high, low, bound, out),
                OP_EQ => compare_impl::<OP_EQ>(high, low, bound, out),
                _ => compare_impl::<OP_NE>(high, low, bound, out),
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn compare_impl<const OP: u8>(high: &[i64], low: &[u64], bound: i128, out: &mut [u8]) {
        let bh = (bound >> 64) as i64;
        let bl = bound as u64;
        // SAFETY: avx512f enabled; loads are unaligned; indices stay within the 8-lane chunks.
        unsafe {
            let bhv = _mm512_set1_epi64(bh);
            let blv = _mm512_set1_epi64(bl as i64);

            let chunks = high.len() / 8;
            for c in 0..chunks {
                let h = _mm512_loadu_epi64(high.as_ptr().add(c * 8));
                let l = _mm512_loadu_epi64(low.as_ptr().add(c * 8).cast());
                out[c] = chunk_mask::<OP>(h, l, bhv, blv);
            }

            for i in (chunks * 8)..high.len() {
                if cmp_i128::<OP>(reconstruct(high[i], low[i]), bound) {
                    out[i / 8] |= 1 << (i % 8);
                }
            }
        }
    }

    /// # Safety
    /// The caller must ensure `avx512f` is enabled and `out.len() == high.len().div_ceil(8)`.
    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn between(
        high: &[i64],
        low: &[u64],
        lower: i128,
        upper: i128,
        lower_op: u8,
        upper_op: u8,
        out: &mut [u8],
    ) {
        // SAFETY: avx512f is enabled by the target_feature on this fn and its callees.
        unsafe {
            match (lower_op, upper_op) {
                (OP_GT, OP_LT) => between_impl::<OP_GT, OP_LT>(high, low, lower, upper, out),
                (OP_GT, _) => between_impl::<OP_GT, OP_LE>(high, low, lower, upper, out),
                (_, OP_LT) => between_impl::<OP_GE, OP_LT>(high, low, lower, upper, out),
                _ => between_impl::<OP_GE, OP_LE>(high, low, lower, upper, out),
            }
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn between_impl<const LOWER: u8, const UPPER: u8>(
        high: &[i64],
        low: &[u64],
        lower: i128,
        upper: i128,
        out: &mut [u8],
    ) {
        // SAFETY: avx512f enabled; loads are unaligned; indices stay within the 8-lane chunks.
        unsafe {
            let lhv = _mm512_set1_epi64((lower >> 64) as i64);
            let llv = _mm512_set1_epi64(lower as u64 as i64);
            let uhv = _mm512_set1_epi64((upper >> 64) as i64);
            let ulv = _mm512_set1_epi64(upper as u64 as i64);

            let chunks = high.len() / 8;
            for c in 0..chunks {
                let h = _mm512_loadu_epi64(high.as_ptr().add(c * 8));
                let l = _mm512_loadu_epi64(low.as_ptr().add(c * 8).cast());
                out[c] = chunk_mask::<LOWER>(h, l, lhv, llv) & chunk_mask::<UPPER>(h, l, uhv, ulv);
            }

            for i in (chunks * 8)..high.len() {
                let value = reconstruct(high[i], low[i]);
                if cmp_i128::<LOWER>(value, lower) & cmp_i128::<UPPER>(value, upper) {
                    out[i / 8] |= 1 << (i % 8);
                }
            }
        }
    }
}

/// Test helper: build a two-limb [`DecimalByteParts`] array from i128 values, splitting each into a
/// signed high limb and an unsigned low limb.
#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn two_limb_array(
    values: &[i128],
    validity: vortex_array::validity::Validity,
    dt: vortex_array::dtype::DecimalDType,
) -> crate::DecimalBytePartsArray {
    let highs: vortex_buffer::Buffer<i64> = values.iter().map(|v| (v >> 64) as i64).collect();
    let lows: vortex_buffer::Buffer<u64> = values.iter().map(|v| *v as u64).collect();
    DecimalByteParts::try_new_with_lower(
        PrimitiveArray::new(highs, validity).into_array(),
        PrimitiveArray::new(lows, vortex_array::validity::Validity::NonNullable).into_array(),
        dt,
    )
    .unwrap()
}

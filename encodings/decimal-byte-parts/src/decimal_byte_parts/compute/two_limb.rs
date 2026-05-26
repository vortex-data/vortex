// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared helpers for fused single-pass kernels over the two-limb (signed-high i64, unsigned-low
//! u64) i128 representation, used by both `between` and `compare`.

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::DecimalBytePartsArrayExt;

/// Materialize the high (signed `i64`) and low (unsigned `u64`) limbs of a two-limb array.
///
/// The caller is responsible for checking [`DecimalBytePartsArrayExt::lower`] is present before
/// calling; this panics otherwise.
pub(crate) fn materialize_limbs(
    arr: &ArrayView<'_, DecimalByteParts>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(PrimitiveArray, PrimitiveArray)> {
    let high = arr.msp().clone().execute::<PrimitiveArray>(ctx)?;
    let low = arr
        .lower()
        .vortex_expect("two-limb path requires a lower limb")
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    Ok((high, low))
}

/// Reconstruct the i128 value from its limbs: sign-extend the high limb, zero-extend the low limb,
/// so `value = (high << 64) | low`.
#[inline(always)]
pub(crate) fn reconstruct(high: i64, low: u64) -> i128 {
    ((high as i128) << 64) | (low as i128)
}

pub(crate) fn i128_eq(a: i128, b: i128) -> bool {
    a == b
}
pub(crate) fn i128_ne(a: i128, b: i128) -> bool {
    a != b
}
pub(crate) fn i128_ge(a: i128, b: i128) -> bool {
    a >= b
}
pub(crate) fn i128_gt(a: i128, b: i128) -> bool {
    a > b
}
pub(crate) fn i128_le(a: i128, b: i128) -> bool {
    a <= b
}
pub(crate) fn i128_lt(a: i128, b: i128) -> bool {
    a < b
}

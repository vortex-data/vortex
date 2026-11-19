// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod blocked;
mod global;

pub use blocked::bitpack_blocked_to_best_bit_widths;
pub use blocked::bitpack_encode_blocked;
pub use blocked::find_best_bit_widths;
pub use global::bit_width_histogram;
pub use global::bitpack_encode;
pub use global::bitpack_encode_unchecked;
pub use global::bitpack_primitive;
pub use global::bitpack_to_best_bit_width;
pub use global::bitpack_unchecked;
pub use global::find_best_bit_width;
pub use global::gather_patches;
#[cfg(feature = "_test-harness")]
pub use global::test_harness;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::NativePType;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

/// Return an error unless `array` holds integers that are all non-negative, which bit-packing
/// requires.
fn ensure_non_negative_integers(
    array: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    vortex_ensure!(
        array.ptype().is_int(),
        "cannot bitpack {} array",
        array.ptype()
    );
    if array.ptype().is_signed_int() {
        let has_negative_values =
            match_each_integer_ptype!(array.ptype(), |P| { has_negative_min::<P>(array, ctx) });
        if has_negative_values {
            vortex_bail!("cannot bitpack array containing negative integers")
        }
    }
    Ok(())
}

/// Whether the minimum of `array` as `P` is negative, treating an unknown minimum as zero.
fn has_negative_min<P>(array: &PrimitiveArray, ctx: &mut ExecutionCtx) -> bool
where
    P: NativePType + PartialOrd + for<'a> TryFrom<&'a Scalar, Error = VortexError>,
{
    array.statistics().compute_min::<P>(ctx).unwrap_or_default() < P::zero()
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Constant encoding schemes for binary, bool, float, integer, and string arrays.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

mod binary;
mod bool;
mod float;
mod integer;
mod string;

pub use binary::BinaryConstantScheme;
pub use bool::BoolConstantScheme;
pub use float::FloatConstantScheme;
pub use integer::IntConstantScheme;
pub use string::StringConstantScheme;

/// Shared helper for compressing a constant array (binary, bool, int, float, string) into a
/// [`ConstantArray`].
///
/// Assumes that the source array has constant valid scalars.
///
/// If the array has any nulls, returns a [`MaskedArray`] with a [`ConstantArray`] child.`
fn compress_constant_array_with_validity(
    source: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if source.all_invalid(ctx)? {
        return Ok(
            ConstantArray::new(Scalar::null(source.dtype().clone()), source.len()).into_array(),
        );
    }

    let scalar_idx = (0..source.len())
        .position(|idx| source.is_valid(idx, ctx).unwrap_or(false))
        .vortex_expect("We checked that there exists a scalar that is not invalid");

    let scalar = source.execute_scalar(scalar_idx, ctx)?;
    let const_arr = ConstantArray::new(scalar, source.len()).into_array();

    if !source.all_valid(ctx)? {
        Ok(MaskedArray::try_new(const_arr, source.validity()?)?.into_array())
    } else {
        Ok(const_arr)
    }
}

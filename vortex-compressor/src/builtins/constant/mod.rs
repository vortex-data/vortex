// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Constant encoding schemes for bool, float, integer, and string arrays.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::is_float_primitive;
use super::is_integer_primitive;
use super::is_utf8_string;

/// Constant encoding for bool arrays where all valid values are the same.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct BoolConstantScheme;

/// Constant encoding for integer arrays with a single distinct value.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct IntConstantScheme;

/// Constant encoding for float arrays with a single distinct value.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FloatConstantScheme;

/// Constant encoding for string arrays with a single distinct value.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct StringConstantScheme;

mod bool;
mod float;
mod integer;
mod string;

/// Shared helper for compressing a constant array (bool, int, float, string) into a
/// [`ConstantArray`].
///
/// Assumes that the source array has constant valid scalars.
///
/// If the array has any nulls, returns a [`MaskedArray`] with a [`ConstantArray`] child.`
fn compress_constant_array_with_validity(source: &ArrayRef) -> VortexResult<ArrayRef> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    if source.all_invalid(&mut ctx)? {
        return Ok(
            ConstantArray::new(Scalar::null(source.dtype().clone()), source.len()).into_array(),
        );
    }

    let scalar_idx = (0..source.len())
        .position(|idx| source.is_valid(idx, &mut ctx).unwrap_or(false))
        .vortex_expect("We checked that there exists a scalar that is not invalid");

    let scalar = source.execute_scalar(scalar_idx, &mut ctx)?;
    let const_arr = ConstantArray::new(scalar, source.len()).into_array();

    if !source.all_valid(&mut ctx)? {
        Ok(MaskedArray::try_new(const_arr, source.validity()?)?.into_array())
    } else {
        Ok(const_arr)
    }
}

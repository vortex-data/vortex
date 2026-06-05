// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Boolean-value execution: the optimized [`Interleave`](super::super::Interleave) path for
//! boolean values.

use num_traits::AsPrimitive;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use crate::ArrayRef;
use crate::array::Array;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Primitive;
use crate::arrays::bool::BoolArrayExt;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_unsigned_integer_ptype;
use crate::require_child;
use crate::validity::Validity;

/// Gathers `N` boolean values under unsigned `array_indices` / `row_indices` selectors, scattering
/// each selected bit (and its validity) into the output position it routes to.
pub(super) fn execute(
    array: Array<Interleave>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let num_values = array.num_values();

    // Drive every value and both selectors to canonical encodings so we can operate on raw bits.
    let mut array = array;
    for i in 0..num_values {
        array = require_child!(array, array.value(i), i => Bool);
    }
    array = require_child!(array, array.array_indices(), num_values => Primitive);
    array = require_child!(array, array.row_indices(), num_values + 1 => Primitive);

    let dtype = array.as_ref().dtype().clone();
    let len = array.as_ref().len();
    let nullable = dtype.is_nullable();

    // Materialize each value's bits, and its validity mask only when the output can be null.
    let mut value_bits = Vec::with_capacity(num_values);
    let mut value_validity = Vec::with_capacity(num_values);
    for i in 0..num_values {
        let value = array.value(i).as_::<Bool>();
        let bits = value.to_bit_buffer();
        let validity = nullable
            .then(|| value.validity()?.execute_mask(bits.len(), ctx))
            .transpose()?;
        value_bits.push(bits);
        value_validity.push(validity);
    }

    // Decode the selectors once so the scatter loop indexes plain `usize`s.
    let branch_of = selector_to_usize(array.array_indices());
    let row_of = selector_to_usize(array.row_indices());

    let mut values = BitBufferMut::new_unset(len);
    let mut validity = nullable.then(|| BitBufferMut::new_set(len));

    for i in 0..len {
        // Random access: gather one bit (and its validity) from the selected value at the position
        // `row_indices` names — no cursor is advanced.
        let branch = branch_of[i];
        let row = row_of[i];
        vortex_ensure!(branch < num_values, "interleave array index out of bounds");
        let bits = &value_bits[branch];
        vortex_ensure!(row < bits.len(), "interleave row index out of bounds");

        if bits.value(row) {
            values.set(i);
        }
        // A missing per-value mask means every row of that value is valid; `validity` is `Some`
        // exactly when `nullable`.
        let valid = value_validity[branch].as_ref().is_none_or(|m| m.value(row));
        if !valid {
            validity
                .as_mut()
                .vortex_expect("validity buffer present when nullable")
                .unset(i);
        }
    }

    let validity = match validity {
        Some(bits) => Validity::from(bits.freeze()),
        None => Validity::NonNullable,
    };
    Ok(ExecutionResult::done(BoolArray::try_new(
        values.freeze(),
        validity,
    )?))
}

/// Decodes a canonical, non-nullable unsigned-integer selector into a `usize` vector.
fn selector_to_usize(selector: &ArrayRef) -> Vec<usize> {
    let selector = selector.as_::<Primitive>();
    match_each_unsigned_integer_ptype!(selector.ptype(), |T| {
        selector.as_slice::<T>().iter().map(|&v| v.as_()).collect()
    })
}

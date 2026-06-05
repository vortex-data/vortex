// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Boolean-value execution: the optimized two-value [`Interleave`](super::super::Interleave) path.

use num_traits::AsPrimitive;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
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

/// Gathers two boolean values under a non-nullable boolean `array_indices` selector and an unsigned
/// `row_indices` selector, scattering each selected bit (and its validity) into the output.
pub(super) fn execute(
    array: Array<Interleave>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    // This kernel currently implements only the boolean (two-value) `array_indices` form; routing
    // boolean values with an unsigned-integer selector is left for the generic-`T` work.
    if !array.array_indices().dtype().is_boolean() {
        vortex_panic!(
            "interleave over boolean values currently requires a boolean array_indices, got {} \
             (todo: support integer selectors)",
            array.array_indices().dtype()
        );
    }
    debug_assert_eq!(
        array.num_values(),
        2,
        "a boolean array_indices implies exactly two values"
    );

    // Drive the values and selectors to canonical encodings so we can operate on raw bits.
    let array = require_child!(array, array.value(0), 0 => Bool);
    let array = require_child!(array, array.value(1), 1 => Bool);
    let array = require_child!(array, array.array_indices(), 2 => Bool);
    let array = require_child!(array, array.row_indices(), 3 => Primitive);

    let dtype = array.as_ref().dtype().clone();
    let len = array.as_ref().len();
    let nullable = dtype.is_nullable();

    let v0 = array.value(0).as_::<Bool>();
    let v1 = array.value(1).as_::<Bool>();
    // The selector is non-nullable (enforced at construction), so its bits route directly: `false`
    // selects value 0, `true` selects value 1.
    let array_indices = Mask::from_buffer(array.array_indices().as_::<Bool>().to_bit_buffer());
    let row_indices = array.row_indices().as_::<Primitive>();

    let v0_bits = v0.to_bit_buffer();
    let v1_bits = v1.to_bit_buffer();

    // Value validity is only materialized when the output can be null.
    let (valid0, valid1) = if nullable {
        (
            Some(v0.validity()?.execute_mask(v0_bits.len(), ctx)?),
            Some(v1.validity()?.execute_mask(v1_bits.len(), ctx)?),
        )
    } else {
        (None, None)
    };

    let mut values = BitBufferMut::new_unset(len);
    let mut validity = nullable.then(|| BitBufferMut::new_set(len));

    match_each_unsigned_integer_ptype!(row_indices.ptype(), |R| {
        let rows = row_indices.as_slice::<R>();
        for i in 0..len {
            // Random access: gather one bit (and its validity) from the selected value at the
            // position `row_indices` names — no cursor is advanced.
            let row: usize = rows[i].as_();
            let (bit, valid) = if array_indices.value(i) {
                vortex_ensure!(
                    row < v1_bits.len(),
                    "interleave row index out of bounds for value 1"
                );
                (
                    v1_bits.value(row),
                    valid1.as_ref().is_none_or(|m| m.value(row)),
                )
            } else {
                vortex_ensure!(
                    row < v0_bits.len(),
                    "interleave row index out of bounds for value 0"
                );
                (
                    v0_bits.value(row),
                    valid0.as_ref().is_none_or(|m| m.value(row)),
                )
            };
            if bit {
                values.set(i);
            }
            // `validity` is `Some` exactly when `nullable`, and `valid` is always true otherwise.
            if !valid {
                validity
                    .as_mut()
                    .vortex_expect("validity buffer present when nullable")
                    .unset(i);
            }
        }
    });

    let validity = match validity {
        Some(bits) => Validity::from(bits.freeze()),
        None => Validity::NonNullable,
    };
    Ok(ExecutionResult::done(BoolArray::try_new(
        values.freeze(),
        validity,
    )?))
}

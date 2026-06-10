// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Boolean-value execution: the optimized [`Interleave`](super::super::Interleave) path for
//! boolean values.

use num_traits::AsPrimitive;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use super::check_selector_bounds;
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

    // Scatter directly from the typed selector buffers — no intermediate `usize` materialization.
    let array_indices = array.array_indices().as_::<Primitive>();
    let row_indices = array.row_indices().as_::<Primitive>();
    let (values, validity) = match_each_unsigned_integer_ptype!(array_indices.ptype(), |A| {
        match_each_unsigned_integer_ptype!(row_indices.ptype(), |R| {
            gather(
                len,
                &value_bits,
                &value_validity,
                array_indices.as_slice::<A>(),
                row_indices.as_slice::<R>(),
                nullable,
            )?
        })
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

/// The scatter, monomorphized on the selector integer widths so each `(array_index, row_index)`
/// pair is read straight from its packed buffer.
///
/// Output bits (and validity) are produced with [`BitBufferMut::collect_bool`], which packs 64
/// results per word: every output bit is written branchlessly, avoiding a per-row `set`/`unset`
/// (each of which would bounds-check and branch on the random bit value).
#[allow(clippy::too_many_arguments)]
fn gather<A: AsPrimitive<usize>, R: AsPrimitive<usize>>(
    len: usize,
    value_bits: &[BitBuffer],
    value_validity: &[Option<Mask>],
    branches: &[A],
    rows: &[R],
    nullable: bool,
) -> VortexResult<(BitBufferMut, Option<BitBufferMut>)> {
    // Validate the per-row bounds once up front (returning an error rather than panicking), so the
    // word-packing passes below are tight branchless loops.
    let value_lens: Vec<usize> = value_bits.iter().map(|b| b.len()).collect();
    check_selector_bounds(branches, rows, &value_lens)?;

    let values =
        BitBufferMut::collect_bool(len, |i| value_bits[branches[i].as_()].value(rows[i].as_()));

    // A missing per-value mask means every row of that value is valid; only materialized when the
    // output can be null.
    let validity = nullable.then(|| {
        BitBufferMut::collect_bool(len, |i| {
            value_validity[branches[i].as_()]
                .as_ref()
                .is_none_or(|mask| mask.value(rows[i].as_()))
        })
    });

    Ok((values, validity))
}

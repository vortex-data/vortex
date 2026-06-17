// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optimized [`Interleave`] implementation for boolean values.

use num_traits::AsPrimitive;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

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

/// Gathers `N` boolean values under unsigned `array_indices` / `row_indices` selectors, scattering
/// each selected bit into the output position it routes to.
pub(super) fn execute(
    array: Array<Interleave>,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let num_values = array.num_values();

    // Drive both selectors and every value to canonical encodings so we can operate on raw bits.
    let mut array = array;
    array = require_child!(array, array.array_indices(), 0 => Primitive);
    array = require_child!(array, array.row_indices(), 1 => Primitive);
    for i in 0..num_values {
        array = require_child!(array, array.value(i), i + 2 => Bool);
    }

    // Materialize each value's bits; the selectors gather one bit per output below.
    let mut value_bits = Vec::with_capacity(num_values);
    for i in 0..num_values {
        value_bits.push(array.value(i).as_::<Bool>().to_bit_buffer());
    }

    // Hold the validity as a pushed-down interleave rather than applying it: the routing pair for
    // each output selects the value *and* its validity bit, so the output validity is itself an
    // interleave (by these selectors) of the values' validities. This bottoms out lazily.
    let validity = array.as_ref().validity()?;

    // Scatter directly from the typed selector buffers — no intermediate `usize` materialization.
    let array_indices = array.array_indices().as_::<Primitive>();
    let row_indices = array.row_indices().as_::<Primitive>();
    let values = match_each_unsigned_integer_ptype!(array_indices.ptype(), |A| {
        match_each_unsigned_integer_ptype!(row_indices.ptype(), |R| {
            gather(
                &value_bits,
                array_indices.as_slice::<A>(),
                row_indices.as_slice::<R>(),
            )?
        })
    });

    Ok(ExecutionResult::done(BoolArray::try_new(
        values.freeze(),
        validity,
    )?))
}

/// The scatter, monomorphized on the selector integer widths so each `(array_index, row_index)`
/// pair is read straight from its packed buffer.
fn gather<A: AsPrimitive<usize>, R: AsPrimitive<usize>>(
    value_bits: &[BitBuffer],
    branches: &[A],
    rows: &[R],
) -> VortexResult<BitBufferMut> {
    let len = validate_selectors(value_bits, branches, rows)?;

    // SAFETY: `validate_selectors` proved `branches.len() == rows.len() == len`, and for every
    // `i < len` that `branches[i] < value_bits.len()` and `rows[i] < value_bits[branches[i]].len()`.
    Ok(unsafe { gather_bits(len, value_bits, branches, rows) })
}

/// Validates the per-row selector bounds, returning the output length (`branches.len()`).
///
/// On success, `rows.len() == branches.len() == len` and, for every `i < len`,
/// `branches[i] < value_bits.len()` and `rows[i] < value_bits[branches[i]].len()` — exactly the
/// preconditions of [`gather_bits`]. Errors (rather than panics) on any out-of-bounds selector.
fn validate_selectors<A: AsPrimitive<usize>, R: AsPrimitive<usize>>(
    value_bits: &[BitBuffer],
    branches: &[A],
    rows: &[R],
) -> VortexResult<usize> {
    // The two selectors are validated to equal length at construction, which is the output length.
    let len = branches.len();
    vortex_ensure!(
        rows.len() == len,
        "interleave selectors differ in length: array_indices {len}, row_indices {}",
        rows.len()
    );

    for i in 0..len {
        let branch = branches[i].as_();
        vortex_ensure!(
            branch < value_bits.len(),
            "interleave array index out of bounds"
        );
        vortex_ensure!(
            rows[i].as_() < value_bits[branch].len(),
            "interleave row index out of bounds"
        );
    }

    Ok(len)
}

/// Gathers one bit per output from `bits[branches[i]]` at position `rows[i]`, packing 64 results per
/// word with [`BitBufferMut::collect_bool`].
///
/// For a random-access gather there is no word-level shortcut on the read side — consecutive outputs
/// read unrelated source words — so the work is one bit read per output. Each read indexes the
/// `&[BitBuffer]` slice and uses [`BitBuffer::value_unchecked`]. Benchmarking (`gather_values` in
/// `benches/interleave.rs`) showed this beats pre-hoisting a `(ptr, bit_offset)` table per buffer:
/// the table's extra indirection and per-call allocation cost more than reloading the selected
/// buffer's pointer/offset, which stay hot in its `BitBuffer` struct. The bounds-checked
/// `BitBuffer::value` is slower still.
///
/// # Safety
///
/// `branches` and `rows` must both contain at least `len` elements. For every `i < len`,
/// `branches[i] < bits.len()` and `rows[i] < bits[branches[i]].len()`.
unsafe fn gather_bits<A: AsPrimitive<usize>, R: AsPrimitive<usize>>(
    len: usize,
    bits: &[BitBuffer],
    branches: &[A],
    rows: &[R],
) -> BitBufferMut {
    // SAFETY: `collect_bool` calls this for `i < len`, and the caller guarantees `branches[i]` and
    // `rows[i]` are in bounds for `bits` / the selected buffer.
    BitBufferMut::collect_bool(len, |i| unsafe {
        bits.get_unchecked(branches.get_unchecked(i).as_())
            .value_unchecked(rows.get_unchecked(i).as_())
    })
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Optimized [`Interleave`] implementation for boolean values.

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::get_bit_unchecked;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
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
    // Both selectors are dispatched over their concrete unsigned width, and each value is converted
    // to an index with a plain `as usize` cast on that concrete type (no `AsPrimitive`).
    let array_indices = array.array_indices().as_::<Primitive>();
    let row_indices = array.row_indices().as_::<Primitive>();
    let (values, validity) = match_each_unsigned_integer_ptype!(array_indices.ptype(), |A| {
        match_each_unsigned_integer_ptype!(row_indices.ptype(), |R| {
            let branches = array_indices.as_slice::<A>();
            let rows = row_indices.as_slice::<R>();
            // SAFETY: both accessors are only ever called with `i < len`, and `branches`/`rows`
            // each have length `len` (the selectors are equal-length and equal to the output len).
            //
            // The `as usize` widens a concrete unsigned selector (`u8`..`u64`) to an index. Selector
            // values index in-memory arrays, so they always fit in `usize`; the cast is lossless.
            #[allow(clippy::cast_possible_truncation)]
            let result = gather(
                len,
                num_values,
                &value_bits,
                &value_validity,
                |i| unsafe { *branches.get_unchecked(i) as usize },
                |i| unsafe { *rows.get_unchecked(i) as usize },
                nullable,
            )?;
            result
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

/// The scatter, monomorphized on the selector integer widths via the `branch_at` / `row_at`
/// accessors so each `(array_index, row_index)` pair is read straight from its packed buffer at its
/// native width and cast to an index with a concrete `as usize`.
///
/// Output bits (and validity) are produced with [`BitBufferMut::collect_bool`], which packs 64
/// results per word. For a random-access gather there is no word-level shortcut on the read side —
/// consecutive outputs read unrelated source words — so the work is one bit read per output. The
/// per-read overhead is what we trim: a raw `(ptr, bit_offset)` is hoisted per value buffer and the
/// bit is read with [`get_bit_unchecked`], avoiding the wide `&[BitBuffer]` struct index and the
/// redundant bounds assert in `BitBuffer::value`. Validity is materialized into one full-length
/// [`BitBuffer`] per branch so its gather is the same uniform unchecked read rather than a per-row
/// `Option`/`Mask`-variant dispatch.
#[allow(clippy::too_many_arguments)]
fn gather(
    len: usize,
    num_values: usize,
    value_bits: &[BitBuffer],
    value_validity: &[Option<Mask>],
    branch_at: impl Fn(usize) -> usize,
    row_at: impl Fn(usize) -> usize,
    nullable: bool,
) -> VortexResult<(BitBufferMut, Option<BitBufferMut>)> {
    // Validate the per-row bounds once up front (returning an error rather than panicking), so the
    // word-packing passes below are tight unchecked loops.
    for i in 0..len {
        let branch = branch_at(i);
        vortex_ensure!(branch < num_values, "interleave array index out of bounds");
        vortex_ensure!(
            row_at(i) < value_bits[branch].len(),
            "interleave row index out of bounds"
        );
    }

    // Raw (byte pointer, bit offset) per value buffer; the offset folds the buffer's own bit offset
    // into the index so the read is a single `get_bit_unchecked`.
    let val_ptrs: Vec<(*const u8, usize)> = value_bits
        .iter()
        .map(|b| (b.inner().as_ptr(), b.offset()))
        .collect();

    // SAFETY (both passes): `i < len`, and the loop above proved `branch_at(i) < num_values` and
    // `row_at(i) < value_bits[branch_at(i)].len()`, which equals the validity length for that
    // branch. So every `get_unchecked` / `get_bit_unchecked` is in bounds.
    let values = BitBufferMut::collect_bool(len, |i| unsafe {
        let (ptr, off) = *val_ptrs.get_unchecked(branch_at(i));
        get_bit_unchecked(ptr, row_at(i) + off)
    });

    // A missing per-value mask means every row of that value is valid; validity is only materialized
    // when the output can be null.
    let validity = nullable.then(|| {
        let validity_bits: Vec<BitBuffer> = value_validity
            .iter()
            .enumerate()
            .map(|(j, mask)| match mask {
                Some(mask) => mask.to_bit_buffer(),
                None => BitBuffer::new_set(value_bits[j].len()),
            })
            .collect();
        let vld_ptrs: Vec<(*const u8, usize)> = validity_bits
            .iter()
            .map(|b| (b.inner().as_ptr(), b.offset()))
            .collect();
        BitBufferMut::collect_bool(len, |i| unsafe {
            let (ptr, off) = *vld_ptrs.get_unchecked(branch_at(i));
            get_bit_unchecked(ptr, row_at(i) + off)
        })
    });

    Ok((values, validity))
}

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Decimal;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::match_each_power_of_two_width;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_compute::lane_kernels::for_each_mask_word;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::zip::zip_validity;

/// Fixed-width canonical arrays that support branchless record-level zip.
pub(super) trait FixedWidthZipArray: VTable {
    fn byte_width(array: ArrayView<'_, Self>) -> usize;

    fn values(array: ArrayView<'_, Self>) -> ByteBuffer;

    fn with_values(
        array: ArrayView<'_, Self>,
        values: ByteBuffer,
        len: usize,
        validity: Validity,
    ) -> VortexResult<Array<Self>>;
}

impl FixedWidthZipArray for Primitive {
    fn byte_width(array: ArrayView<'_, Self>) -> usize {
        array.ptype().byte_width()
    }

    fn values(array: ArrayView<'_, Self>) -> ByteBuffer {
        array.buffer_handle().to_host_sync()
    }

    fn with_values(
        array: ArrayView<'_, Self>,
        values: ByteBuffer,
        _len: usize,
        validity: Validity,
    ) -> VortexResult<PrimitiveArray> {
        Ok(PrimitiveArray::from_byte_buffer(
            values,
            array.ptype(),
            validity,
        ))
    }
}

impl FixedWidthZipArray for Decimal {
    fn byte_width(array: ArrayView<'_, Self>) -> usize {
        array.values_type().byte_width()
    }

    fn values(array: ArrayView<'_, Self>) -> ByteBuffer {
        array.buffer_handle().to_host_sync()
    }

    fn with_values(
        array: ArrayView<'_, Self>,
        values: ByteBuffer,
        _len: usize,
        validity: Validity,
    ) -> VortexResult<DecimalArray> {
        // Fixed-width kernels copy values and validity together. Check the new byte layout.
        DecimalArray::try_new_handle_from_trusted_values(
            BufferHandle::new_host(values),
            array.values_type(),
            array.decimal_dtype(),
            validity,
        )
    }
}

/// Selects each record from `if_true` where `mask` is set and from `if_false` where it is not.
///
/// The generic zip path copies runs of `if_true`/`if_false` between mask boundaries, which is fast
/// for clustered masks but degrades to per-element work on fragmented masks. This kernel instead
/// walks the mask as 64-bit chunks and blends both sides per record without a data-dependent
/// branch, so the inner loop stays branch-free and auto-vectorizable regardless of mask shape.
///
/// The caller must check that both arrays share a logical type. Trivial masks return `None`, so the
/// generic zip casts the surviving side instead.
pub(super) fn zip_fixed_width<V: FixedWidthZipArray>(
    if_true: ArrayView<'_, V>,
    if_false: ArrayView<'_, V>,
    mask: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let byte_width = V::byte_width(if_true);
    vortex_ensure_eq!(byte_width, V::byte_width(if_false));

    // Null mask entries select `if_false`, matching `Zip`'s SQL ELSE semantics.
    let mask = mask.clone().fill_null(false)?.execute::<Mask>(ctx)?;
    let Mask::Values(mask_values) = &mask else {
        return Ok(None);
    };

    let validity = zip_validity(if_true.validity()?, if_false.validity()?, &mask)?;

    // TODO(connor): inspect the mask's true_count (and validity) to special-case heavily-skewed
    // masks. When one side dominates (true_count near 0 or near len), it is cheaper to bulk
    // copy (or mutate in place, if the dominant side is uniquely owned) that side's values
    // and validity, then conditionally pull in only the minority rows from the other side,
    // rather than blending every row.
    let values = select_records(
        V::values(if_true),
        V::values(if_false),
        byte_width,
        mask_values,
    )?;

    let len = if_true.len();
    let expected_len = len
        .checked_mul(byte_width)
        .ok_or_else(|| vortex_err!("Fixed-width values buffer length overflows usize"))?;
    vortex_ensure_eq!(
        values.len(),
        expected_len,
        "Fixed-width values buffer length does not match expected length",
    );

    let array = V::with_values(if_true, values, len, validity)?;
    Ok(Some(array.into_array()))
}

/// Blends `byte_width`-sized records from both buffers into a fresh buffer.
fn select_records(
    true_values: ByteBuffer,
    false_values: ByteBuffer,
    byte_width: usize,
    mask: &MaskValues,
) -> VortexResult<ByteBuffer> {
    match_each_power_of_two_width!(
        byte_width,
        |W| {
            Ok(select_values(
                true_values.as_slice().as_chunks::<W>().0,
                false_values.as_slice().as_chunks::<W>().0,
                mask,
            )
            .freeze()
            .into_byte_buffer()
            .aligned(Alignment::new(W)))
        },
        _ => {
            vortex_bail!("zip does not support fixed-width records of {byte_width} bytes")
        },
    )
}

/// Branchlessly blend `if_true` and `if_false` per row into a fresh value buffer.
fn select_values<T: Copy>(
    true_values: &[T],
    false_values: &[T],
    mask: &MaskValues,
) -> BufferMut<T> {
    let len = true_values.len();
    let mut out = BufferMut::<T>::with_capacity(len);
    {
        let out_slice = out.spare_capacity_mut();

        for_each_mask_word(mask.bit_buffer(), |word, base, n| {
            let end = base + n;
            select_block(
                word,
                &true_values[base..end],
                &false_values[base..end],
                &mut out_slice[base..end],
            );
        });
    }

    // SAFETY: `select_block` initialized every slot covered by the mask words, i.e. `len`.
    unsafe { out.set_len(len) };
    out
}

/// Blend one 64-bit mask chunk's worth of rows: bit `j` (LSB-first) keeps `true_values[j]`, an unset
/// bit keeps `false_values[j]`. Slices are trimmed to the output length up front so the compiler can
/// elide bounds checks and lower the body to a vector blend / conditional move.
#[inline]
fn select_block<T: Copy>(
    word: u64,
    true_values: &[T],
    false_values: &[T],
    out: &mut [MaybeUninit<T>],
) {
    let n = out.len();
    let true_values = &true_values[..n];
    let false_values = &false_values[..n];
    for j in 0..n {
        let pick = (word >> j) & 1 == 1;
        out[j].write(if pick {
            true_values[j]
        } else {
            false_values[j]
        });
    }
}

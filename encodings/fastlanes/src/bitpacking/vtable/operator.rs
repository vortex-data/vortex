// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::BitPacking;
use vortex_array::execution::{BatchKernelRef, BindCtx, MaskExecution, kernel};
use vortex_array::vtable::{OperatorVTable, ValidityHelper};
use vortex_array::{ArrayRef, IntoArray, ToCanonical, compute};
use vortex_buffer::{Buffer, byte_buffer_to_buffer};
use vortex_compute::filter::Filter;
use vortex_dtype::{NativePType, match_each_unsigned_integer_ptype};
use vortex_error::VortexResult;
use vortex_vector::PVector;

use crate::{BitPackedArray, BitPackedVTable};

impl OperatorVTable<BitPackedVTable> for BitPackedVTable {
    fn bind(
        array: &BitPackedArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let selection_mask = ctx.bind_selection(array.len(), selection)?;
        let validity = ctx.bind_validity(array.validity(), array.len(), selection)?;

        // Since the fastlanes crate only supports unsigned integers, and since we know that all
        // numbers are going to be non-negative, we can safely "cast" to unsigned.
        let ptype = array.ptype().to_unsigned();

        match_each_unsigned_integer_ptype!(ptype, |U| {
            Ok(bitpack_filter_kernel::<U>(array, selection_mask, validity))
        })
    }
}

/// Creates the [`BitPackedArray`] filter kernel.
///
/// Note that the generic type parameter `U` may be the unsigned version of the signed [`PType`] of
/// the input array. This is fine because we know that all values in bitpacked arrays are
/// non-negative.
fn bitpack_filter_kernel<U: NativePType + BitPacking>(
    array: &BitPackedArray,
    selection_mask: MaskExecution,
    validity: MaskExecution,
) -> BatchKernelRef {
    let array = array.clone(); // Cheap clone due to many internal `Arc`s.
    kernel(move || {
        let selection_mask = selection_mask.execute()?;
        let filtered_validity = validity.execute()?;

        // TODO(connor): This function is implemented in a very roundabout way where we use the
        // existing `BitPackedArray` `filter` implementation that gives us an array, and then we
        // extract out the underlying buffer of the `PrimitiveArray` to create a `PrimitiveVector`.
        //
        // Ideally, we should take the underlying `ByteBuffer` of the `BitPackedArray` and unpack
        // that directly into a `Buffer<T>` via a `BufferMut<T>`. This is a much more general
        // solution that does not force everyone to use `PrimitiveBuilder`.
        //
        // However, the current decompression implementation for `BitPackedArray` is heavily tied
        // to the `PrimitiveBuilder` and `UninitRange` API. What we really need to do is _replace_
        // the `PrimitiveBuilder` with `PrimitiveVectorMut`, where instead of `UninitRange` we can
        // write directly to a `PVectorMut`.
        //
        // For the sake of time to get this working, we have implemented this like so.
        // When we eventually replace our builders with vectors, we can revisit this.

        // Use the existing `filter` implementation over `PrimitiveArray` and extract the underlying
        // `ByteBuffer`.
        let filtered_array = compute::filter(&array.into_array(), &selection_mask)?.to_primitive();
        debug_assert_eq!(filtered_array.ptype().byte_width(), size_of::<U>());

        let byte_buffer = filtered_array.into_byte_buffer();

        // SAFETY: The `filter` compute function maintains the type of the bitpacked array, which
        // must have the same byte representation and alignment as `U`, so it is safe to reinterpret
        // this buffer.
        let buffer: Buffer<U> = unsafe { byte_buffer_to_buffer(byte_buffer) };
        let filtered_buffer = buffer.filter(&selection_mask);

        debug_assert_eq!(filtered_buffer.len(), filtered_validity.len());

        // SAFETY: The buffer and validity (which should have started with the same length) were
        // filtered by the same mask, which means their new lengths should also be the same.
        Ok(unsafe { PVector::new_unchecked(filtered_buffer, filtered_validity) }.into())
    })
}

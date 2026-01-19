// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_vector::binaryview::BinaryView;

use crate::Canonical;
use crate::ToCanonical;
use crate::arrays::VarBinVTable;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbin::VarBinArray;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<VarBinVTable> for VarBinVTable {
    fn canonicalize(array: &VarBinArray) -> VortexResult<Canonical> {
        // Zero the offsets first to ensure the bytes buffer starts at 0
        let array = array.clone().zero_offsets();
        let (dtype, bytes, offsets, validity) = array.into_parts();

        let offsets = offsets.to_primitive();

        // Build views directly from offsets
        #[expect(clippy::cast_possible_truncation, reason = "BinaryView offset is u32")]
        let views: Buffer<BinaryView> = match_each_integer_ptype!(offsets.ptype(), |O| {
            let offsets_slice = offsets.as_slice::<O>();
            let bytes_slice = bytes.as_ref();

            let mut views = BufferMut::<BinaryView>::with_capacity(offsets_slice.len() - 1);
            for window in offsets_slice.windows(2) {
                let start: usize = window[0].as_();
                let end: usize = window[1].as_();
                let value = &bytes_slice[start..end];
                views.push(BinaryView::make_view(value, 0, start as u32));
            }
            views.freeze()
        });

        // Create VarBinViewArray with the original bytes buffer and computed views
        // SAFETY: views are correctly computed from valid offsets
        let varbinview =
            unsafe { VarBinViewArray::new_unchecked(views, Arc::from([bytes]), dtype, validity) };
        Ok(Canonical::VarBinView(varbinview))
    }
}

// Convert a VarBinArray to VarBinViewArray using Arrow's conversion.
//
// This method leverages Arrow's `From<&GenericByteArray<FROM>> for GenericByteViewArray<V>`
// implementation to perform the conversion, then converts back to Vortex.
// pub fn canonicalize_via_arrow(array: &VarBinArray) -> VortexResult<VarBinViewArray> {
//     match array.dtype() {
//         DType::Utf8(_) => canonicalize_via_arrow_typed::<Utf8Type, StringViewType>(array),
//         DType::Binary(_) => canonicalize_via_arrow_typed::<BinaryType, BinaryViewType>(array),
//         _ => unreachable!("VarBinArray must have Utf8 or Binary dtype"),
//     }
// }
//
// fn canonicalize_via_arrow_typed<FROM, V>(array: &VarBinArray) -> VortexResult<VarBinViewArray>
// where
//     FROM: ByteArrayType,
//     FROM::Offset: NativePType,
//     V: ByteViewType<Native = FROM::Native>,
// {
//     let nullable = array.dtype().is_nullable();
//
//     // Build Arrow GenericByteArray from VarBinArray
//     // Cast offsets to the required offset type (i32 for Utf8/Binary, i64 for Large variants)
//     let offsets = cast(
//         array.offsets().as_ref(),
//         &DType::Primitive(FROM::Offset::PTYPE, Nullability::NonNullable),
//     )?
//     .to_primitive()
//     .to_buffer::<FROM::Offset>()
//     .into_arrow_offset_buffer();
//
//     let data = array.bytes().clone().into_arrow_buffer();
//
//     // Convert validity mask to Arrow NullBuffer
//     let null_buffer = match array.validity_mask() {
//         Mask::AllTrue(_) => None,
//         Mask::AllFalse(len) => Some(NullBuffer::new_null(len)),
//         Mask::Values(values) => Some(NullBuffer::from(BooleanBuffer::from(
//             values.bit_buffer().clone(),
//         ))),
//     };
//     let null_buffer = crate::arrow::null_buffer::to_null_buffer()
//
//     // SAFETY: VarBinArray invariants guarantee valid offsets and UTF-8 (if Utf8 dtype)
//     let arrow_byte_array =
//         unsafe { GenericByteArray::<FROM>::new_unchecked(offsets, data, null_buffer) };
//
//     // Use Arrow's From impl to convert to view array
//     let arrow_view_array: GenericByteViewArray<V> = GenericByteViewArray::from(&arrow_byte_array);
//
//     // Convert back to Vortex
//     let vortex_array = ArrayRef::from_arrow(&arrow_view_array, nullable);
//     Ok(vortex_array.as_::<VarBinViewVTable>().clone())
// }

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;

    use crate::arrays::varbin::builder::VarBinBuilder;
    use crate::canonical::ToCanonical;

    #[rstest]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[case(DType::Binary(Nullability::Nullable))]
    fn test_canonical_varbin(#[case] dtype: DType) {
        let mut varbin = VarBinBuilder::<i32>::with_capacity(10);
        varbin.append_null();
        varbin.append_null();
        // inlined value
        varbin.append_value("123456789012".as_bytes());
        // non-inlinable value
        varbin.append_value("1234567890123".as_bytes());
        let varbin = varbin.finish(dtype.clone());

        let varbin = varbin.slice(1..4);

        let canonical = varbin.to_varbinview();
        assert_eq!(canonical.dtype(), &dtype);

        assert!(!canonical.is_valid(0));

        // First value is inlined (12 bytes)
        assert!(canonical.views()[1].is_inlined());
        assert_eq!(canonical.bytes_at(1).as_slice(), "123456789012".as_bytes());

        // Second value is not inlined (13 bytes)
        assert!(!canonical.views()[2].is_inlined());
        assert_eq!(canonical.bytes_at(2).as_slice(), "1234567890123".as_bytes());
    }
}

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
use crate::arrays::VarBinViewArray;
use crate::arrays::varbin::VarBinArray;

/// Converts a VarBinArray to its canonical form (VarBinViewArray).
///
/// This is a shared helper used by both `canonicalize` and `execute`.
pub(crate) fn varbin_to_canonical(array: &VarBinArray) -> VortexResult<Canonical> {
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

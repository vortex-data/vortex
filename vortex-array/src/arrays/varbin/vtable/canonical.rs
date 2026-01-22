// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;

use crate::Canonical;
use crate::ExecutionCtx;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::build_views::MAX_BUFFER_LEN;
use crate::arrays::build_views::build_views;
use crate::arrays::build_views::offsets_to_lengths;
use crate::arrays::varbin::VarBinArray;

/// Converts a VarBinArray to its canonical form (VarBinViewArray).
///
/// This is a shared helper used by both `canonicalize` and `execute`.
pub(crate) fn varbin_to_canonical(
    array: &VarBinArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Canonical> {
    // Zero the offsets first to ensure the bytes buffer starts at 0
    let array = array.clone().zero_offsets();
    let (dtype, bytes, offsets, validity) = array.into_parts();

    // offsets_to_lengths
    let offsets = offsets.execute::<PrimitiveArray>(ctx)?;
    let bytes = bytes.into_mut();

    match_each_integer_ptype!(offsets.ptype(), |P| {
        let lens = offsets_to_lengths(offsets.as_slice::<P>());
        let (buffers, views) = build_views(0, MAX_BUFFER_LEN, bytes, lens.as_slice());

        let varbinview =
            unsafe { VarBinViewArray::new_unchecked(views, Arc::from(buffers), dtype, validity) };

        // Create VarBinViewArray with the original bytes buffer and computed views
        // SAFETY: views are correctly computed from valid offsets
        Ok(Canonical::VarBinView(varbinview))
    })
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

        let varbin = varbin.slice(1..4).unwrap();

        let canonical = varbin.to_varbinview();
        assert_eq!(canonical.dtype(), &dtype);

        assert!(!canonical.is_valid(0).unwrap());

        // First value is inlined (12 bytes)
        assert!(canonical.views()[1].is_inlined());
        assert_eq!(canonical.bytes_at(1).as_slice(), "123456789012".as_bytes());

        // Second value is not inlined (13 bytes)
        assert!(!canonical.views()[2].is_inlined());
        assert_eq!(canonical.bytes_at(2).as_slice(), "1234567890123".as_bytes());
    }
}

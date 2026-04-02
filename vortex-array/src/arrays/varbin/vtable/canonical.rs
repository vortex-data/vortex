// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use num_traits::AsPrimitive;
use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use crate::arrays::varbinview::build_views::build_views;
use crate::arrays::varbinview::build_views::offsets_to_lengths;
use crate::match_each_integer_ptype;

/// Converts a VarBinArray to its canonical form (VarBinViewArray).
///
/// This is a shared helper used by both `canonicalize` and `execute`.
pub(crate) fn varbin_to_canonical(
    array: ArrayView<'_, VarBin>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<VarBinViewArray> {
    let (dtype, bytes, offsets, validity) = array.into_owned().into_data().into_parts();

    let offsets = offsets.execute::<PrimitiveArray>(ctx)?;

    match_each_integer_ptype!(offsets.ptype(), |P| {
        let offsets_slice = offsets.as_slice::<P>();
        let first: usize = offsets_slice[0].as_();
        let last: usize = offsets_slice[offsets_slice.len() - 1].as_();
        let bytes = bytes.unwrap_host().slice(first..last).into_mut();

        let lens = offsets_to_lengths(offsets_slice);
        let (buffers, views) = build_views(0, MAX_BUFFER_LEN, bytes, lens.as_slice());

        // SAFETY: views are correctly computed from valid offsets
        Ok(unsafe { VarBinViewArray::new_unchecked(views, Arc::from(buffers), dtype, validity) })
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::varbin::builder::VarBinBuilder;
    use crate::assert_arrays_eq;
    use crate::canonical::ToCanonical;
    use crate::dtype::DType;
    use crate::dtype::Nullability;

    #[rstest]
    #[case(DType::Utf8(Nullability::Nullable))]
    #[case(DType::Binary(Nullability::Nullable))]
    fn test_canonical_varbin_sliced(#[case] dtype: DType) {
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

    #[rstest]
    #[case(DType::Utf8(Nullability::NonNullable))]
    #[case(DType::Binary(Nullability::NonNullable))]
    fn test_canonical_varbin_unsliced(#[case] dtype: DType) {
        let varbin = VarBinArray::from_iter_nonnull(["foo", "bar", "baz"], dtype.clone());
        let canonical = varbin.as_array().to_varbinview();
        let expected = match dtype {
            DType::Utf8(_) => VarBinViewArray::from_iter_str(["foo", "bar", "baz"]),
            _ => VarBinViewArray::from_iter_bin(["foo", "bar", "baz"]),
        };
        assert_arrays_eq!(canonical, expected);
    }

    // Empty array: offsets has exactly one element; no elements to canonicalize.
    #[test]
    fn test_canonical_varbin_empty() {
        let varbin =
            VarBinArray::from_iter_nonnull([] as [&str; 0], DType::Utf8(Nullability::NonNullable));
        let canonical = varbin.as_array().to_varbinview();
        assert_eq!(canonical.len(), 0);
    }
}
